"""CarAgent unified dashboard for launch/test/calibration workflows."""

from __future__ import annotations

import glob as glob_mod
import html
import json
import math
import mimetypes
import os
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

import rclpy
from rclpy.node import Node


WORKSPACE = Path(os.environ.get("CARAGENT_WORKSPACE", "~/caragent_ws")).expanduser()
MAPS_DIR = WORKSPACE / "maps"
KEYFRAMES_DIR = WORKSPACE / "keyframes"
OBJECT_DEPTH_DATASETS_DIR = WORKSPACE / "perception_datasets" / "object_depth"
MODELS_DIR = WORKSPACE / "models"
CALIB_DIR = WORKSPACE / "calibration"
RESULT_ROOTS = [
    WORKSPACE / "perception_outputs",
    KEYFRAMES_DIR,
    MAPS_DIR,
    CALIB_DIR,
    WORKSPACE / "logs",
]
AGENT_WEB_PORT = int(os.environ.get("CARAGENT_AGENT_WEB_PORT", "8123"))
AGENT_SIM_WEB_PORT = int(os.environ.get("CARAGENT_AGENT_SIM_WEB_PORT", "8124"))
CLIP_MODEL = MODELS_DIR / "clip-vit-base-patch32" / "image_encoder.xml"
DINO_MODEL = MODELS_DIR / "dinov2"
DEFAULT_STEREO_CALIB = CALIB_DIR / "stereo_current" / "stereo_calibration.npz"
DEFAULT_EXTRINSICS = CALIB_DIR / "lidar_camera" / "lidar_camera_extrinsics_calibrated.json"
DEFAULT_CAMERA_RESOLUTION = "3840x1200"
ROTATION_TUNE_CONFIG = WORKSPACE / "config" / "dashboard_rotation_tune.json"
DEMO_LAYOUT_SCRIPT_NAME = "start_demo_layout.ps1"
CAMERA_RESOLUTIONS = {
    "3840x1200": (3840, 1200, 1920, 1920, 30.0),
    "3840x1080": (3840, 1080, 1920, 1920, 30.0),
    "2560x720": (2560, 720, 1280, 1280, 30.0),
    "1280x480": (1280, 480, 640, 640, 30.0),
}
UTF8_ENV_DEFAULTS = {
    "PYTHONIOENCODING": "utf-8",
    "PYTHONUTF8": "1",
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
}


def _now_hms() -> str:
    return time.strftime("%H:%M:%S")


def _now_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _expand_path(value: str | Path | None, default: Path) -> Path:
    text = str(value or "").strip()
    if not text:
        return default
    return Path(text).expanduser()


def _safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return value.strip("_") or _now_stamp()


def _yaw_from_quaternion(q: Any) -> float:
    siny = 2.0 * (float(q.w) * float(q.z) + float(q.x) * float(q.y))
    cosy = 1.0 - 2.0 * (float(q.y) * float(q.y) + float(q.z) * float(q.z))
    return math.atan2(siny, cosy)


def _normalize_angle_rad(value: float) -> float:
    return math.atan2(math.sin(value), math.cos(value))


def _ccw_delta_rad(current: float, target: float) -> float:
    return (float(target) - float(current)) % (2.0 * math.pi)


def _angle_deg(value: float | None) -> float | None:
    if value is None:
        return None
    return math.degrees(float(value))


def _default_rotation_tune_params() -> dict[str, float]:
    return {
        "fast_omega": 3.4,
        "mid_omega": 2.5,
        "slow_omega": 1.5,
        "fast_threshold_deg": 20.0,
        "mid_threshold_deg": 10.0,
        "yaw_tolerance_deg": 4.0,
        "right_turn_shortcut_deg": 90.0,
        "settle_time_sec": 0.20,
        "timeout_sec": 20.0,
        "omega_cap": 3.5,
    }


def _count_images(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(
        len(list(path.glob(pattern)))
        for pattern in ("*.png", "*.jpg", "*.jpeg")
    )


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _read_jsonl_objects(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except Exception:
                continue
            if isinstance(item, dict):
                rows.append(item)
    return rows


def _write_jsonl_objects(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for item in rows:
            handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")


def _parse_keyframe_id_list(value: Any) -> list[int]:
    if isinstance(value, (list, tuple, set)):
        raw_parts: list[str] = []
        for item in value:
            raw_parts.extend(re.split(r"[,，\s]+", str(item)))
    else:
        raw_parts = re.split(r"[,，\s]+", str(value or ""))
    ids: set[int] = set()
    for part in raw_parts:
        text = part.strip()
        if not text:
            continue
        match = re.fullmatch(r"(?:kf_?)?0*(\d+)(?:\s*[-~～]\s*(?:kf_?)?0*(\d+))?", text, re.IGNORECASE)
        if not match:
            raise ValueError(f"Invalid keyframe id/range: {text}")
        start = int(match.group(1))
        end = int(match.group(2) or start)
        if end < start:
            start, end = end, start
        if end - start > 2000:
            raise ValueError(f"Keyframe id range is too large: {text}")
        ids.update(range(start, end + 1))
    return sorted(ids)


def _read_simple_yaml_map(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    result: dict[str, Any] = {}
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if key in {"resolution", "occupied_thresh", "free_thresh"}:
            try:
                result[key] = float(value)
            except ValueError:
                pass
        elif key == "origin":
            try:
                result[key] = [float(item.strip()) for item in value.strip("[]").split(",")]
            except ValueError:
                pass
        else:
            result[key] = value
    return result


def _map_png_preview(image_path: Path) -> Path:
    if image_path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
        return image_path
    preview = image_path.with_name(f"{image_path.stem}_dashboard.png")
    if preview.exists() and preview.stat().st_mtime >= image_path.stat().st_mtime:
        return preview
    try:
        import cv2  # type: ignore

        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is not None:
            cv2.imwrite(str(preview), image)
            return preview
    except Exception:
        pass
    return image_path


def _map_metadata_for_dataset(dataset: Path) -> dict[str, Any]:
    session = dataset.parent if dataset.name == "selected" else dataset
    session_config = _read_json_object(session / "session.json")
    params = session_config.get("parameters") if isinstance(session_config.get("parameters"), dict) else {}
    map_value = str(params.get("map_file_name") or session_config.get("map_file_name") or "").strip()
    fallback = False
    if not map_value:
        maps = [item for item in _map_entries() if item.get("yaml_path")]
        map_value = str((maps[0] if maps else {}).get("base_path") or "")
        fallback = bool(map_value)
    if not map_value:
        return {}

    map_base = Path(map_value).expanduser()
    yaml_path = map_base if map_base.suffix == ".yaml" else map_base.with_suffix(".yaml")
    if not yaml_path.exists():
        return {}
    yaml_data = _read_simple_yaml_map(yaml_path)
    image_value = str(yaml_data.get("image") or "").strip()
    if not image_value:
        return {}
    image_path = Path(image_value)
    if not image_path.is_absolute():
        image_path = yaml_path.parent / image_path
    if not image_path.exists():
        return {}
    preview = _map_png_preview(image_path)
    origin = yaml_data.get("origin") if isinstance(yaml_data.get("origin"), list) else [0.0, 0.0, 0.0]
    while len(origin) < 3:
        origin.append(0.0)
    return {
        "yaml": str(yaml_path.resolve()),
        "image": str(preview.resolve()),
        "source_image": str(image_path.resolve()),
        "resolution": float(yaml_data.get("resolution") or 0.05),
        "origin": [float(origin[0]), float(origin[1]), float(origin[2])],
        "fallback_latest": fallback,
    }


def _camera_resolution_values(body: dict[str, Any]) -> tuple[int, int, int, int, float]:
    selected = str(body.get("camera_resolution") or DEFAULT_CAMERA_RESOLUTION).strip()
    if selected in CAMERA_RESOLUTIONS:
        return CAMERA_RESOLUTIONS[selected]
    match = re.fullmatch(r"(\d+)x(\d+)", selected)
    if match:
        width = int(match.group(1))
        height = int(match.group(2))
        left_width = width // 2
        return width, height, left_width, width - left_width, 30.0
    return CAMERA_RESOLUTIONS[DEFAULT_CAMERA_RESOLUTION]


def _camera_launch_args(body: dict[str, Any]) -> list[str]:
    width, height, left_width, right_width, fps = _camera_resolution_values(body)
    return [
        f"camera_width:={width}",
        f"camera_height:={height}",
        f"camera_left_width:={left_width}",
        f"camera_right_width:={right_width}",
        f"camera_fps:={fps:g}",
    ]


def _stereo_camera_launch_args(body: dict[str, Any]) -> list[str]:
    width, height, left_width, right_width, fps = _camera_resolution_values(body)
    return [
        f"width:={width}",
        f"height:={height}",
        f"left_width:={left_width}",
        f"right_width:={right_width}",
        f"fps:={fps:g}",
    ]


def _scan_ports() -> dict[str, list[str]]:
    serial: list[str] = []
    cameras: list[str] = []
    if sys.platform.startswith("linux"):
        for pat in ["/dev/ttyUSB*", "/dev/ttyACM*"]:
            serial.extend(sorted(glob_mod.glob(pat)))
        cameras.extend(sorted(glob_mod.glob("/dev/video*")))
    elif sys.platform.startswith("win"):
        serial = [f"COM{i}" for i in range(1, 10)]
        cameras = ["0"]
    return {"serial": sorted(set(serial)), "cameras": sorted(set(cameras))}


def _path_with_suffixes(base: Path) -> list[Path]:
    candidates = [base]
    for suffix in [".posegraph", ".data", ".yaml", ".pgm", ".png"]:
        candidates.append(base.with_suffix(suffix))
    return [path for path in candidates if path.exists()]


def _map_entries() -> list[dict[str, Any]]:
    MAPS_DIR.mkdir(parents=True, exist_ok=True)
    bases: dict[str, set[Path]] = {}
    for path in MAPS_DIR.glob("*"):
        if path.is_dir():
            continue
        if path.name.endswith("_dashboard.png"):
            continue
        if path.suffix in {".posegraph", ".data", ".yaml", ".pgm", ".png"}:
            base = path.with_suffix("")
        else:
            base = path
        bases.setdefault(str(base), set()).add(path)
    entries = []
    for base_text, files in bases.items():
        base = Path(base_text)
        all_files = sorted(files | set(_path_with_suffixes(base)), key=lambda p: p.name)
        if not all_files:
            continue
        entries.append(
            {
                "name": base.name,
                "base_path": str(base),
                "yaml_path": str(base.with_suffix(".yaml")) if base.with_suffix(".yaml").exists() else "",
                "mtime": max(path.stat().st_mtime for path in all_files),
                "files": [str(path) for path in all_files],
            }
        )
    return sorted(entries, key=lambda item: item["mtime"], reverse=True)


def _keyframe_sessions() -> list[dict[str, Any]]:
    KEYFRAMES_DIR.mkdir(parents=True, exist_ok=True)
    sessions = []
    for session in sorted(KEYFRAMES_DIR.glob("*"), key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True):
        if not session.is_dir():
            continue
        selected = session / "selected"
        node_dir = selected / "constructed_memory" / "keyframe_nodes"
        summary_path = selected / "scene_memory_summary.json"
        manifest_path = selected / "constructed_memory" / "scene_memory_manifest.json"
        chunk_records_path = selected / "constructed_memory" / "semantic_chunk_index_records.json"
        summary = _read_json_object(summary_path)
        manifest = _read_json_object(manifest_path)
        counts = summary.get("counts") if isinstance(summary.get("counts"), dict) else {}
        chunk_payload = _read_json_object(chunk_records_path)
        chunk_records = chunk_payload.get("records") if isinstance(chunk_payload.get("records"), list) else []
        semantics = 0
        nodes = list(node_dir.glob("kf_*.json")) if node_dir.exists() else []
        for node_file in nodes:
            try:
                data = json.loads(node_file.read_text(encoding="utf-8"))
            except Exception:
                continue
            if str(data.get("semantic") or "").strip():
                semantics += 1
        sessions.append(
            {
                "name": session.name,
                "path": str(session),
                "selected_path": str(selected) if selected.exists() else "",
                "candidate_frames": _count_images(session / "left"),
                "selected_frames": _count_images(selected / "left") if selected.exists() else 0,
                "keyframe_nodes": len(nodes),
                "semantic_nodes": semantics,
                "semantic_chunks": int(counts.get("semantic_chunks") or len(chunk_records) or 0),
                "build_status": str(summary.get("status") or ("ready" if nodes else "recorded")),
                "summary_json": str(summary_path) if summary_path.exists() else "",
                "manifest_json": str(manifest_path) if manifest_path.exists() else "",
                "review_html": str(selected / "review.html") if (selected / "review.html").exists() else "",
                "scene_html": str(selected / "scene_map.html") if (selected / "scene_map.html").exists() else "",
                "mtime": max(
                    session.stat().st_mtime,
                    selected.stat().st_mtime if selected.exists() else 0,
                    summary_path.stat().st_mtime if summary_path.exists() else 0,
                    manifest_path.stat().st_mtime if manifest_path.exists() else 0,
                ),
                "memory_format": str(manifest.get("format") or ""),
            }
        )
    return sessions


def _object_depth_datasets() -> list[dict[str, Any]]:
    OBJECT_DEPTH_DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    datasets = []
    for dataset in sorted(OBJECT_DEPTH_DATASETS_DIR.glob("*"), key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True):
        if not dataset.is_dir():
            continue
        manifest = dataset / "manifest.jsonl"
        sample_count = 0
        targets: set[str] = set()
        if manifest.exists():
            for line in manifest.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                sample_count += 1
                try:
                    data = json.loads(line)
                    target = str(data.get("target") or "").strip()
                    if target:
                        targets.add(target)
                except Exception:
                    continue
        eval_dir = dataset / "evaluations"
        summaries = []
        if eval_dir.exists():
            for path in eval_dir.rglob("summary.csv"):
                summaries.append(str(path))
        datasets.append(
            {
                "name": dataset.name,
                "path": str(dataset),
                "manifest": str(manifest) if manifest.exists() else "",
                "sample_count": sample_count,
                "targets": sorted(targets),
                "summaries": summaries,
                "mtime": dataset.stat().st_mtime,
            }
        )
    return datasets


def _keyframe_images(dataset: str | Path | None) -> list[dict[str, Any]]:
    dataset_path = _expand_path(dataset, KEYFRAMES_DIR / "")
    if dataset_path.name != "selected" and (dataset_path / "selected").exists():
        dataset_path = dataset_path / "selected"
    left_dir = dataset_path / "left"
    if not left_dir.exists():
        return []
    images = []
    for path in sorted(
        list(left_dir.glob("*.png"))
        + list(left_dir.glob("*.jpg"))
        + list(left_dir.glob("*.jpeg"))
    ):
        images.append(
            {
                "name": path.name,
                "path": str(path),
                "mtime": path.stat().st_mtime,
            }
        )
    return images


def _latest_keyframe_session_path() -> Path | None:
    sessions = _keyframe_sessions()
    if not sessions:
        return None
    return Path(str(sessions[0].get("path") or ""))


def _manifest_count_and_last_id(session: Path) -> tuple[int, str]:
    manifest = session / "manifest.jsonl"
    if not manifest.exists():
        return 0, ""
    count = 0
    last_id = ""
    with manifest.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.strip():
                continue
            count += 1
            try:
                item = json.loads(line)
                last_id = str(item.get("frame_id") or last_id)
            except Exception:
                pass
    return count, last_id


def _recent_results(limit: int = 80) -> list[dict[str, Any]]:
    files: list[Path] = []
    for root in RESULT_ROOTS:
        if root.exists():
            for pattern in ["*.html", "*.png", "*.jpg", "*.jpeg", "*.json", "*.jsonl", "*.csv", "*.npy", "*.npz"]:
                files.extend(root.rglob(pattern))
    files = sorted(set(files), key=lambda p: p.stat().st_mtime, reverse=True)[:limit]
    return [
        {"path": str(path), "name": path.name, "mtime": path.stat().st_mtime, "kind": path.suffix.lower().lstrip(".")}
        for path in files
        if path.is_file()
    ]


def _process_stage_from_logs(name: str, logs: list[str], running: bool, returncode: int | None) -> dict[str, str]:
    text = "\n".join(logs[-80:])
    lower = text.lower()
    status = "running" if running else ("failed" if returncode not in {None, 0} else "idle")
    stage = "运行中" if running else ("已结束" if returncode == 0 else "未运行")
    if name == "keyframe_record":
        if "saved keyframe candidate" in lower:
            stage = "正在采集关键帧"
    elif name in {"keyframe_build", "keyframe_annotate"}:
        if "select_keyframes" in lower or "selected" in lower:
            stage = "正在筛选关键帧"
        if "annotating" in lower or "kf_" in lower:
            stage = "正在进行 VLM 语义标注"
        if "semantic chunk" in lower or "chunk_index" in lower:
            stage = "正在构建语义索引"
        if "build_scene_memory complete" in lower or "done." in lower:
            stage = "场景记忆构建完成"
    elif name == "slam":
        stage = "正在建图" if running else stage
    elif name == "agent":
        stage = "Agent 正在运行" if running else stage
    if "error" in lower or "failed" in lower or "traceback" in lower:
        status = "warning" if running else "failed"
    return {"status": status, "stage": stage}


def _latest_saved_keyframe(logs: list[str]) -> dict[str, Any] | None:
    for line in reversed(logs):
        match = re.search(r"saved keyframe candidate\s+(\d+)", line, re.IGNORECASE)
        if match:
            return {"frame_id": match.group(1), "message": f"保存关键帧 {int(match.group(1))}"}
    return None


def _json_response(handler: BaseHTTPRequestHandler, data: object, status: int = 200) -> None:
    body = json.dumps(data, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Cache-Control", "no-store")
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
    handler.send_header("Access-Control-Allow-Headers", "Content-Type")
    handler.end_headers()
    handler.wfile.write(body)


def _text_response(
    handler: BaseHTTPRequestHandler,
    text: str,
    *,
    content_type: str = "text/plain; charset=utf-8",
    filename: str | None = None,
) -> None:
    body = text.encode("utf-8")
    handler.send_response(200)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Cache-Control", "no-store")
    if filename:
        handler.send_header("Content-Disposition", f'attachment; filename="{filename}"')
    handler.end_headers()
    handler.wfile.write(body)


def _shell_join(args: list[str]) -> str:
    return " ".join(shlex.quote(str(arg)) for arg in args)


def _scene_map_generation_command(dataset: Path) -> str:
    code = "\n".join(
        [
            "from pathlib import Path",
            "from caragent_ui.dashboard_node import DashboardNode",
            f"dataset = Path({str(dataset)!r})",
            "if dataset.name == 'selected':",
            "    dataset = dataset.parent",
            "selected = dataset / 'selected'",
            "selected.mkdir(parents=True, exist_ok=True)",
            "node = DashboardNode.__new__(DashboardNode)",
            "DashboardNode._write_scene_map_html(node, dataset, selected / 'scene_map.html')",
            "print('scene_map_html=' + str(selected / 'scene_map.html'))",
        ]
    )
    return _shell_join(["python3", "-c", code])


def _utf8_subprocess_env(overrides: dict[str, str] | None = None) -> dict[str, str]:
    env = os.environ.copy()
    env.update(UTF8_ENV_DEFAULTS)
    if overrides:
        env.update(overrides)
    return env


def _run_shell_utf8(command: str, *, timeout: float) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        shell=True,
        executable="/bin/bash",
        cwd=str(WORKSPACE),
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        timeout=timeout,
        env=_utf8_subprocess_env(),
    )


@dataclass
class ManagedProcess:
    name: str
    command: str
    cwd: Path = WORKSPACE
    env_overrides: dict[str, str] = field(default_factory=dict)
    proc: subprocess.Popen | None = None
    started_at: float | None = None
    stopped_at: float | None = None
    returncode: int | None = None
    logs: deque[str] = field(default_factory=lambda: deque(maxlen=1200))
    lock: threading.RLock = field(default_factory=threading.RLock)

    def is_running(self) -> bool:
        with self.lock:
            if self.proc is None:
                return False
            rc = self.proc.poll()
            if rc is not None:
                self.returncode = rc
                self.stopped_at = self.stopped_at or time.time()
                return False
            return True

    def start(self) -> dict[str, Any]:
        with self.lock:
            if self.is_running():
                return {"ok": False, "error": f"{self.name} is already running."}
            self.logs.clear()
            self.logs.append(f"[{_now_hms()}] COMMAND: {self.command}")
            env = _utf8_subprocess_env()
            env.setdefault("PYTHONUNBUFFERED", "1")
            env.update(self.env_overrides)
            self.proc = subprocess.Popen(
                self.command,
                cwd=str(self.cwd),
                shell=True,
                executable="/bin/bash" if sys.platform.startswith("linux") else None,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                env=env,
                preexec_fn=os.setsid if sys.platform.startswith("linux") else None,
            )
            self.started_at = time.time()
            self.stopped_at = None
            self.returncode = None
            threading.Thread(target=self._pump_output, daemon=True).start()
            return {"ok": True, "message": f"{self.name} started.", "command": self.command}

    def stop(self, timeout: float = 8.0) -> dict[str, Any]:
        with self.lock:
            proc = self.proc
        if proc is None or proc.poll() is not None:
            return {"ok": True, "message": f"{self.name} is not running."}
        self.append_log("Stopping process...")
        try:
            if sys.platform.startswith("linux"):
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            else:
                proc.terminate()
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.append_log("Terminate timed out; killing process group.")
            if sys.platform.startswith("linux"):
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            else:
                proc.kill()
            proc.wait()
        with self.lock:
            self.returncode = proc.returncode
            self.stopped_at = time.time()
        self.append_log(f"Process stopped with return code {proc.returncode}.")
        return {"ok": True, "message": f"{self.name} stopped.", "returncode": proc.returncode}

    def append_log(self, line: str) -> None:
        with self.lock:
            self.logs.append(f"[{_now_hms()}] {line}")

    def _pump_output(self) -> None:
        proc = self.proc
        if proc is None or proc.stdout is None:
            return
        try:
            for line in proc.stdout:
                stripped = line.rstrip("\n")
                if stripped:
                    self.append_log(stripped)
        finally:
            rc = proc.poll()
            with self.lock:
                self.returncode = rc
                self.stopped_at = time.time()
            self.append_log(f"Process exited with return code {rc}.")

    def status(self) -> dict[str, Any]:
        running = self.is_running()
        with self.lock:
            return {
                "name": self.name,
                "running": running,
                "command": self.command,
                "started_at": self.started_at,
                "stopped_at": self.stopped_at,
                "returncode": self.returncode,
                "log_tail": list(self.logs)[-80:],
            }


class _DashboardHandler(BaseHTTPRequestHandler):
    @property
    def node(self) -> "DashboardNode":
        return self.server.dashboard_node

    def log_message(self, *_args: object) -> None:
        return

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        if not raw:
            return {}
        return json.loads(raw.decode("utf-8"))

    def _serve_file(self, path: Path) -> None:
        if not path.exists() or not path.is_file():
            self.send_error(404)
            return
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        if path.suffix.lower() in {".html", ".htm"}:
            content_type = "text/html; charset=utf-8"
        elif path.suffix.lower() in {".json", ".jsonl"}:
            content_type = "application/json; charset=utf-8"
        elif path.suffix.lower() == ".csv":
            content_type = "text/csv; charset=utf-8"
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Max-Age", "600")
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)
        if path in {"/", "/index.html"}:
            static_path = Path(__file__).resolve().parent / "static" / "dashboard.html"
            if not static_path.exists():
                static_path = WORKSPACE / "install" / "caragent_ui" / "share" / "caragent_ui" / "static" / "dashboard.html"
            return self._serve_file(static_path)
        if path in {"/demo", "/demo.html"}:
            static_path = Path(__file__).resolve().parent / "static" / "demo_dashboard.html"
            if not static_path.exists():
                static_path = WORKSPACE / "install" / "caragent_ui" / "share" / "caragent_ui" / "static" / "demo_dashboard.html"
            return self._serve_file(static_path)
        if path == f"/{DEMO_LAYOUT_SCRIPT_NAME}":
            return _text_response(
                self,
                self.node.demo_layout_script(),
                content_type="text/plain; charset=utf-8",
                filename=DEMO_LAYOUT_SCRIPT_NAME,
            )
        if path == "/api/status":
            return _json_response(self, self.node.get_status())
        if path == "/api/demo/status":
            return _json_response(self, self.node.demo_status())
        if path == "/api/rotation_tune/status":
            return _json_response(self, self.node.rotation_tune_status())
        if path == "/api/ports":
            return _json_response(self, _scan_ports())
        if path == "/api/maps":
            return _json_response(self, {"maps": _map_entries()})
        if path == "/api/keyframes":
            return _json_response(self, {"sessions": _keyframe_sessions()})
        if path == "/api/keyframe_images":
            dataset = query.get("dataset", [""])[0]
            return _json_response(self, {"images": _keyframe_images(unquote(dataset))})
        if path == "/api/object_depth_datasets":
            return _json_response(self, {"datasets": _object_depth_datasets()})
        if path == "/api/results":
            return _json_response(self, {"results": _recent_results()})
        if path == "/api/logs":
            process = query.get("process", [""])[0]
            return _json_response(self, self.node.logs(process or None))
        if path == "/api/file":
            raw = query.get("path", [""])[0]
            return self._serve_file(Path(unquote(raw)).expanduser())
        self.send_error(404)

    def do_POST(self) -> None:
        body = self._read_json()
        path = urlparse(self.path).path
        routes = {
            "/api/process/start": self.node.api_process_start,
            "/api/process/stop": self.node.api_process_stop,
            "/api/process/stop_all": self.node.api_process_stop_all,
            "/api/logs/clear": self.node.api_clear_logs,
            "/api/map/save": self.node.api_save_map,
            "/api/keyframes/select": self.node.api_select_keyframes,
            "/api/keyframes/visualize": self.node.api_visualize_keyframes,
            "/api/keyframes/annotate": self.node.api_annotate_keyframes,
            "/api/keyframes/nodes": self.node.api_keyframe_nodes,
            "/api/keyframes/remove": self.node.api_keyframe_remove,
            "/api/keyframes/capture_once": self.node.api_keyframe_capture_once,
            "/api/keyframes/manual_record/start": self.node.api_keyframe_manual_record_start,
            "/api/live_config": self.node.api_live_config,
            "/api/object_depth_dataset/config": self.node.api_object_depth_dataset_config,
            "/api/slam/initial_pose": self.node.api_slam_initial_pose,
            "/api/slam/clear_changes": self.node.api_slam_clear_changes,
            "/api/slam/auto_clear": self.node.api_slam_auto_clear,
            "/api/teleop": self.node.api_teleop,
            "/api/rotation_tune/start": self.node.api_rotation_tune_start,
            "/api/rotation_tune/stop": self.node.api_rotation_tune_stop,
            "/api/rotation_tune/save": self.node.api_rotation_tune_save,
        }
        func = routes.get(path)
        if func is None:
            self.send_error(404)
            return
        try:
            result = func(body)
            _json_response(self, result, 200 if result.get("ok", True) else 400)
        except Exception as exc:
            self.node.add_global_log(f"API error on {path}: {exc}")
            _json_response(self, {"ok": False, "error": str(exc)}, 500)


class DashboardNode(Node):
    def __init__(self, port: int = 8234) -> None:
        super().__init__("caragent_dashboard")
        self._port = int(os.environ.get("CARAGENT_DASHBOARD_PORT", str(port)))
        self._processes: dict[str, ManagedProcess] = {}
        self._global_logs: deque[str] = deque(maxlen=2000)
        self._lock = threading.RLock()
        self._slam_map_base: str = ""
        self._last_keyframe_session_name: str = ""
        self._auto_clear_enabled: bool = False
        self._auto_clear_interval: float = 8.0
        self._teleop_deadline_monotonic: float = 0.0
        self._teleop_last_command: tuple[float, float] = (0.0, 0.0)
        self._teleop_timeout_sec: float = 0.30
        from geometry_msgs.msg import Twist
        self._teleop_twist_type = Twist
        self._teleop_cmd_vel_pub = self.create_publisher(Twist, "/cmd_vel_nav", 10)
        self._teleop_stop_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self._teleop_timer = self.create_timer(0.05, self._teleop_watchdog_tick)
        self._rotation_imu_yaw_rad: float | None = None
        self._rotation_imu_time: float = 0.0
        self._rotation_odom_yaw_rad: float | None = None
        self._rotation_odom_time: float = 0.0
        self._rotation_tune: dict[str, Any] = {
            "active": False,
            "status": "idle",
            "message": "",
            "target_yaw_rad": None,
            "start_time": 0.0,
            "settle_since": 0.0,
            "current_omega": 0.0,
            "direction": "stop",
            "params": self._load_rotation_tune_params(),
        }
        from nav_msgs.msg import Odometry
        from sensor_msgs.msg import Imu
        self._rotation_imu_sub = self.create_subscription(Imu, "/imu", self._on_rotation_imu, 10)
        self._rotation_odom_sub = self.create_subscription(Odometry, "/odom", self._on_rotation_odom, 10)
        self._rotation_timer = self.create_timer(0.05, self._rotation_tune_tick)
        from geometry_msgs.msg import PoseWithCovarianceStamped
        self._initialpose_sub = self.create_subscription(
            PoseWithCovarianceStamped, "/initialpose", self._on_initialpose, 1)
        threading.Thread(target=self._run_http, daemon=True).start()
        self.add_global_log(f"CarAgent Dashboard ready: http://0.0.0.0:{self._port}")

    def _run_http(self) -> None:
        server = ThreadingHTTPServer(("0.0.0.0", self._port), _DashboardHandler)
        server.dashboard_node = self
        server.serve_forever()

    @staticmethod
    def _clamp(value: float, limit: float) -> float:
        limit = abs(float(limit))
        return max(-limit, min(limit, float(value)))

    def _load_rotation_tune_params(self) -> dict[str, float]:
        params = _default_rotation_tune_params()
        saved = _read_json_object(ROTATION_TUNE_CONFIG)
        if isinstance(saved.get("params"), dict):
            saved = saved["params"]
        if isinstance(saved, dict):
            for key in params:
                if key not in saved:
                    continue
                try:
                    value = float(saved[key])
                except Exception:
                    continue
                if math.isfinite(value):
                    params[key] = value
        return params

    def _publish_teleop(self, linear: float, angular: float) -> None:
        msg = self._teleop_twist_type()
        msg.linear.x = float(linear)
        msg.angular.z = float(angular)
        self._teleop_cmd_vel_pub.publish(msg)
        self._teleop_stop_pub.publish(msg)

    def _publish_teleop_stop(self) -> None:
        msg = self._teleop_twist_type()
        self._teleop_cmd_vel_pub.publish(msg)
        self._teleop_stop_pub.publish(msg)

    def _on_rotation_imu(self, msg) -> None:
        q = msg.orientation
        if abs(float(q.x)) + abs(float(q.y)) + abs(float(q.z)) + abs(float(q.w)) < 1e-6:
            return
        yaw = _yaw_from_quaternion(q)
        if not math.isfinite(yaw):
            return
        with self._lock:
            self._rotation_imu_yaw_rad = yaw
            self._rotation_imu_time = time.monotonic()

    def _on_rotation_odom(self, msg) -> None:
        yaw = _yaw_from_quaternion(msg.pose.pose.orientation)
        if not math.isfinite(yaw):
            return
        with self._lock:
            self._rotation_odom_yaw_rad = yaw
            self._rotation_odom_time = time.monotonic()

    def _rotation_current_yaw_locked(self) -> tuple[float | None, str, float | None]:
        now = time.monotonic()
        if self._rotation_imu_yaw_rad is not None and now - self._rotation_imu_time <= 1.0:
            return self._rotation_imu_yaw_rad, "imu", now - self._rotation_imu_time
        if self._rotation_odom_yaw_rad is not None and now - self._rotation_odom_time <= 2.0:
            return self._rotation_odom_yaw_rad, "odom", now - self._rotation_odom_time
        if self._rotation_imu_yaw_rad is not None:
            return self._rotation_imu_yaw_rad, "imu_stale", now - self._rotation_imu_time
        if self._rotation_odom_yaw_rad is not None:
            return self._rotation_odom_yaw_rad, "odom_stale", now - self._rotation_odom_time
        return None, "unavailable", None

    def _rotation_status_locked(self) -> dict[str, Any]:
        yaw, source, age = self._rotation_current_yaw_locked()
        tune = dict(self._rotation_tune)
        params = dict(tune.get("params") or {})
        target = tune.get("target_yaw_rad")
        shortest_error = None
        ccw_remaining = None
        if yaw is not None and target is not None:
            shortest_error = abs(_normalize_angle_rad(float(target) - float(yaw)))
            ccw_remaining = _ccw_delta_rad(float(yaw), float(target))
        return {
            "ok": True,
            "active": bool(tune.get("active")),
            "status": str(tune.get("status") or "idle"),
            "message": str(tune.get("message") or ""),
            "yaw_source": source,
            "yaw_age_sec": age,
            "current_yaw_rad": yaw,
            "current_yaw_deg": _angle_deg(yaw),
            "target_yaw_rad": target,
            "target_yaw_deg": _angle_deg(target),
            "shortest_error_deg": _angle_deg(shortest_error),
            "ccw_remaining_deg": _angle_deg(ccw_remaining),
            "current_omega": float(tune.get("current_omega") or 0.0),
            "direction": str(tune.get("direction") or "stop"),
            "elapsed_sec": max(0.0, time.monotonic() - float(tune.get("start_time") or time.monotonic())),
            "params": params,
            "config_path": str(ROTATION_TUNE_CONFIG),
            "config_saved": ROTATION_TUNE_CONFIG.exists(),
            "cmd_vel_topic": "/cmd_vel_nav",
            "stop_topics": ["/cmd_vel_nav", "/cmd_vel"],
            "firmware_angular_limit_radps": 3.5,
        }

    def _rotation_omega_for_delta(self, ccw_delta: float, params: dict[str, Any]) -> float:
        cap = max(0.0, min(3.5, float(params.get("omega_cap", 3.5))))
        fast = max(0.0, min(cap, abs(float(params.get("fast_omega", 3.4)))))
        mid = max(0.0, min(cap, abs(float(params.get("mid_omega", 2.5)))))
        slow = max(0.0, min(cap, abs(float(params.get("slow_omega", 1.5)))))
        fast_threshold = math.radians(max(0.0, float(params.get("fast_threshold_deg", 20.0))))
        mid_threshold = math.radians(max(0.0, float(params.get("mid_threshold_deg", 10.0))))
        if ccw_delta > fast_threshold:
            return fast
        if ccw_delta > mid_threshold:
            return mid
        return slow

    def _rotation_omega_for_target(self, current_yaw: float, target_yaw: float, params: dict[str, Any]) -> tuple[float, str]:
        shortest_delta = _normalize_angle_rad(float(target_yaw) - float(current_yaw))
        right_turn_shortcut = math.radians(max(0.0, float(params.get("right_turn_shortcut_deg", 90.0))))
        if shortest_delta < 0.0 and abs(shortest_delta) < right_turn_shortcut:
            return -self._rotation_omega_for_delta(abs(shortest_delta), params), "right_shortcut"
        return self._rotation_omega_for_delta(_ccw_delta_rad(current_yaw, target_yaw), params), "left_only"

    def _rotation_tune_tick(self) -> None:
        with self._lock:
            tune = self._rotation_tune
            if not bool(tune.get("active")):
                return
            target = tune.get("target_yaw_rad")
            params = dict(tune.get("params") or {})
            current_yaw, source, _age = self._rotation_current_yaw_locked()
            if current_yaw is None or target is None:
                tune["status"] = "error"
                tune["message"] = "当前没有可用 IMU/里程计角度。"
                tune["active"] = False
                tune["current_omega"] = 0.0
                tune["direction"] = "stop"
                publish_omega = None
                stop = True
            else:
                timeout_sec = max(0.5, float(params.get("timeout_sec", 20.0)))
                elapsed = time.monotonic() - float(tune.get("start_time") or time.monotonic())
                yaw_tolerance = math.radians(max(0.1, float(params.get("yaw_tolerance_deg", 4.0))))
                settle_time = max(0.0, float(params.get("settle_time_sec", 0.20)))
                shortest_error = abs(_normalize_angle_rad(float(target) - float(current_yaw)))
                if elapsed > timeout_sec:
                    tune["status"] = "timeout"
                    tune["message"] = f"旋转超时，剩余误差 {math.degrees(shortest_error):.1f}°。"
                    tune["active"] = False
                    tune["current_omega"] = 0.0
                    tune["direction"] = "stop"
                    publish_omega = None
                    stop = True
                elif shortest_error <= yaw_tolerance:
                    if not float(tune.get("settle_since") or 0.0):
                        tune["settle_since"] = time.monotonic()
                    if time.monotonic() - float(tune.get("settle_since") or time.monotonic()) >= settle_time:
                        tune["status"] = "done"
                        tune["message"] = f"对齐完成，误差 {math.degrees(shortest_error):.1f}°。"
                        tune["active"] = False
                        tune["current_omega"] = 0.0
                        tune["direction"] = "stop"
                        publish_omega = None
                        stop = True
                    else:
                        tune["status"] = "settling"
                        tune["message"] = f"进入容差，等待稳定。source={source}"
                        tune["current_omega"] = 0.0
                        tune["direction"] = "stop"
                        publish_omega = 0.0
                        stop = False
                else:
                    tune["settle_since"] = 0.0
                    omega, direction = self._rotation_omega_for_target(float(current_yaw), float(target), params)
                    tune["status"] = "running"
                    tune["message"] = f"旋转中，剩余误差 {math.degrees(shortest_error):.1f}°。source={source}"
                    tune["current_omega"] = omega
                    tune["direction"] = direction
                    publish_omega = omega
                    stop = False
        if stop:
            self._publish_teleop_stop()
        elif publish_omega is not None:
            self._publish_teleop(0.0, float(publish_omega))

    def _teleop_watchdog_tick(self) -> None:
        with self._lock:
            deadline = float(self._teleop_deadline_monotonic or 0.0)
            last_command = self._teleop_last_command
            if deadline <= 0.0:
                return
            if time.monotonic() <= deadline:
                return
            self._teleop_deadline_monotonic = 0.0
            self._teleop_last_command = (0.0, 0.0)
        if last_command != (0.0, 0.0):
            self._publish_teleop_stop()

    def add_global_log(self, message: str) -> None:
        line = f"[{_now_hms()}] {message}"
        with self._lock:
            self._global_logs.append(line)
        self.get_logger().info(message)

    def _on_initialpose(self, msg) -> None:
        map_base = self._slam_map_base
        if not map_base:
            self.add_global_log("ignored /initialpose: no incremental map loaded")
            return
        if "slam" not in self._processes or not self._processes["slam"].is_running():
            self.add_global_log("ignored /initialpose: SLAM not running")
            return
        x = float(msg.pose.pose.position.x)
        y = float(msg.pose.pose.position.y)
        q = msg.pose.pose.orientation
        import math as _math
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = _math.atan2(siny, cosy)
        cmd = self._ros_command(
            _shell_join([
                "ros2", "service", "call", "/slam_toolbox/deserialize_map",
                "slam_toolbox/srv/DeserializePoseGraph",
                f"{{filename: '{map_base}', match_type: 2, initial_pose: {{x: {x}, y: {y}, theta: {yaw}}}}}",
            ])
        )
        self.add_global_log(f"Relaying /initialpose → deserialize_map: x={x:.2f} y={y:.2f} yaw={yaw:.2f}")
        _run_shell_utf8(cmd, timeout=15)

    def _ros_command(self, command: str) -> str:
        return (
            "source /opt/ros/humble/setup.bash && "
            f"cd {shlex.quote(str(WORKSPACE))} && "
            "source install/setup.bash && "
            f"{command}"
        )

    def _start_process(
        self,
        name: str,
        command: str,
        env_overrides: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            proc = self._processes.get(name)
            if proc is None:
                proc = ManagedProcess(
                    name=name,
                    command=command,
                    env_overrides=dict(env_overrides or {}),
                )
                self._processes[name] = proc
            elif proc.is_running():
                return {"ok": False, "error": f"{name} is already running.", "status": proc.status()}
            else:
                proc.command = command
                proc.env_overrides = dict(env_overrides or {})
        result = proc.start()
        self.add_global_log(f"{name}: {result.get('message', result)}")
        return result

    def _stop_process(self, name: str) -> dict[str, Any]:
        with self._lock:
            proc = self._processes.get(name)
        if proc is None:
            return {"ok": True, "message": f"{name} is not running."}
        result = proc.stop()
        self.add_global_log(f"{name}: {result.get('message', result)}")
        return result

    def _start_keyframe_scene_memory_build(self, body: dict[str, Any]) -> dict[str, Any]:
        dataset_value = body.get("dataset")
        if dataset_value:
            dataset = _expand_path(dataset_value, KEYFRAMES_DIR / "")
        else:
            session = str(body.get("session_name") or self._last_keyframe_session_name or "").strip()
            if not session:
                sessions = _keyframe_sessions()
                latest = sessions[0]["path"] if sessions else ""
                if not latest:
                    return {"ok": False, "error": "No keyframe session is available to build."}
                dataset = Path(latest)
            else:
                dataset = KEYFRAMES_DIR / session
        if dataset.name == "selected":
            dataset = dataset.parent
        if dataset.resolve() == KEYFRAMES_DIR.resolve():
            return {"ok": False, "error": "Refusing to build the keyframes root; choose one session."}
        if not dataset.exists():
            return {"ok": False, "error": f"Dataset not found: {dataset}"}

        build_body = dict(body)
        build_body["kind"] = "keyframe_build"
        build_body["dataset"] = str(dataset)
        command = self._ros_command(self._command_for_kind("keyframe_build", build_body))
        env_overrides: dict[str, str] = {}
        self.add_global_log(f"Scene-memory build queued for {dataset}")
        return self._start_process("keyframe_build", command, env_overrides=env_overrides)

    def get_status(self) -> dict[str, Any]:
        maps = _map_entries()
        sessions = _keyframe_sessions()
        return {
            "workspace": str(WORKSPACE),
            "dashboard_port": self._port,
            "agent_web_port": AGENT_WEB_PORT,
            "agent_sim_web_port": AGENT_SIM_WEB_PORT,
            "ssh_tunnel": (
                f"ssh -L {self._port}:localhost:{self._port} "
                f"-L {AGENT_WEB_PORT}:localhost:{AGENT_WEB_PORT} "
                f"-L {AGENT_SIM_WEB_PORT}:localhost:{AGENT_SIM_WEB_PORT} car@10.181.156.54"
            ),
            "processes": {name: proc.status() for name, proc in sorted(self._processes.items())},
            "latest_map": maps[0] if maps else None,
            "latest_keyframe_session": sessions[0] if sessions else None,
            "paths": {
                "maps": str(MAPS_DIR),
                "keyframes": str(KEYFRAMES_DIR),
                "models": str(MODELS_DIR),
                "calibration": str(CALIB_DIR),
                "stereo_calib": str(DEFAULT_STEREO_CALIB),
                "extrinsics": str(DEFAULT_EXTRINSICS),
            },
        }

    def demo_status(self) -> dict[str, Any]:
        maps = _map_entries()
        sessions = _keyframe_sessions()
        with self._lock:
            process_status = {name: proc.status() for name, proc in sorted(self._processes.items())}
            global_logs = list(self._global_logs)[-40:]
        stages = []
        latest_keyframe_notice = None
        for name, status in process_status.items():
            logs = list(status.get("log_tail") or [])
            stage = _process_stage_from_logs(
                name,
                logs,
                bool(status.get("running")),
                status.get("returncode"),
            )
            stages.append(
                {
                    "name": name,
                    "running": bool(status.get("running")),
                    "stage": stage["stage"],
                    "status": stage["status"],
                    "started_at": status.get("started_at"),
                    "returncode": status.get("returncode"),
                    "latest": logs[-1] if logs else "",
                }
            )
            if name == "keyframe_record":
                latest_keyframe_notice = _latest_saved_keyframe(logs) or latest_keyframe_notice

        light_logs = []
        for line in global_logs:
            if any(
                keyword in line.lower()
                for keyword in (
                    "started",
                    "stopped",
                    "queued",
                    "saving",
                    "saved",
                    "error",
                    "failed",
                    "ready",
                    "complete",
                    "keyframe",
                    "scene-memory",
                    "agent",
                )
            ):
                light_logs.append(line)
        for item in stages:
            if item["running"] or item["status"] in {"warning", "failed"}:
                light_logs.append(f"[{_now_hms()}] {item['name']}: {item['stage']}")
        return {
            "ok": True,
            "workspace": str(WORKSPACE),
            "dashboard_port": self._port,
            "agent_web_port": AGENT_WEB_PORT,
            "agent_sim_web_port": AGENT_SIM_WEB_PORT,
            "demo_url": f"http://{self._host_hint()}:{self._port}/demo",
            "agent_lite_url": f"http://{self._host_hint()}:{AGENT_WEB_PORT}/lite",
            "agent_sim_url": f"http://{self._host_hint()}:{AGENT_SIM_WEB_PORT}/sim",
            "maps": maps,
            "sessions": sessions,
            "latest_map": maps[0] if maps else None,
            "latest_keyframe_session": sessions[0] if sessions else None,
            "processes": stages,
            "light_logs": light_logs[-80:],
            "latest_keyframe_notice": latest_keyframe_notice,
            "layout_script_url": f"/{DEMO_LAYOUT_SCRIPT_NAME}",
        }

    def _host_hint(self) -> str:
        return os.environ.get("CARAGENT_HOST_HINT", "10.181.156.54")

    def demo_layout_script(self) -> str:
        host = self._host_hint()
        dashboard_url = f"http://{host}:{self._port}/demo"
        agent_url = f"http://{host}:{AGENT_WEB_PORT}/lite"
        return f"""# CarAgent local demo window layout helper.
# Run on the Windows demo laptop after SSH port forwarding or direct LAN access is ready.
param(
  [string]$DashboardUrl = "{dashboard_url}",
  [string]$AgentUrl = "{agent_url}",
  [int]$LeftWidth = 720
)

Add-Type @"
using System;
using System.Runtime.InteropServices;
public class WinApi {{
  [DllImport("user32.dll")] public static extern bool MoveWindow(IntPtr hWnd, int X, int Y, int nWidth, int nHeight, bool bRepaint);
  [DllImport("user32.dll")] public static extern bool ShowWindow(IntPtr hWnd, int nCmdShow);
}}
"@

function Get-BrowserPath {{
  $candidates = @(
    "$env:ProgramFiles\\Google\\Chrome\\Application\\chrome.exe",
    "${{env:ProgramFiles(x86)}}\\Google\\Chrome\\Application\\chrome.exe",
    "$env:ProgramFiles\\Microsoft\\Edge\\Application\\msedge.exe",
    "${{env:ProgramFiles(x86)}}\\Microsoft\\Edge\\Application\\msedge.exe"
  )
  foreach ($p in $candidates) {{ if (Test-Path $p) {{ return $p }} }}
  return "msedge.exe"
}}

function Move-ByTitle([string]$Pattern, [int]$X, [int]$Y, [int]$W, [int]$H) {{
  $deadline = (Get-Date).AddSeconds(10)
  do {{
    $proc = Get-Process | Where-Object {{ $_.MainWindowHandle -ne 0 -and $_.MainWindowTitle -match $Pattern }} | Select-Object -First 1
    if ($proc) {{
      [WinApi]::ShowWindow($proc.MainWindowHandle, 9) | Out-Null
      [WinApi]::MoveWindow($proc.MainWindowHandle, $X, $Y, $W, $H, $true) | Out-Null
      Write-Host "Moved $Pattern -> $X,$Y $W x $H"
      return $true
    }}
    Start-Sleep -Milliseconds 400
  }} while ((Get-Date) -lt $deadline)
  Write-Warning "Window not found: $Pattern"
  return $false
}}

Add-Type -AssemblyName System.Windows.Forms
$screen = [System.Windows.Forms.Screen]::PrimaryScreen.WorkingArea
$browser = Get-BrowserPath
$left = [Math]::Min($LeftWidth, [int]($screen.Width * 0.42))
$rightX = $screen.X + $left
$rightW = $screen.Width - $left
$rvizH = [int]($screen.Height * 0.66)
$camY = $screen.Y + $rvizH
$camH = $screen.Height - $rvizH

Start-Process $browser -ArgumentList @("--new-window", "--window-position=$($screen.X),$($screen.Y)", "--window-size=$left,$($screen.Height)", $DashboardUrl)
Start-Sleep -Milliseconds 900
Start-Process $browser -ArgumentList @("--new-window", "--window-position=$($screen.X),$($screen.Y)", "--window-size=$left,$($screen.Height)", $AgentUrl)

Start-Sleep -Seconds 2
Move-ByTitle "CarAgent|Dashboard|导引|Agent" $screen.X $screen.Y $left $screen.Height | Out-Null
Move-ByTitle "RViz|rviz" $rightX $screen.Y $rightW $rvizH | Out-Null
Move-ByTitle "Huibo|Stereo|Camera|stereo|image" $rightX $camY $rightW $camH | Out-Null
Write-Host "CarAgent demo layout attempted once. It will not keep moving windows."
"""

    def logs(self, process: str | None = None) -> dict[str, Any]:
        if process:
            with self._lock:
                proc = self._processes.get(process)
            return {"process": process, "logs": proc.status()["log_tail"] if proc else []}
        with self._lock:
            logs = list(self._global_logs)
            procs = {name: proc.status()["log_tail"] for name, proc in sorted(self._processes.items())}
        return {"process": "", "logs": logs, "process_logs": procs}

    def api_clear_logs(self, body: dict[str, Any]) -> dict[str, Any]:
        name = str(body.get("process") or "")
        if name:
            proc = self._processes.get(name)
            if proc:
                with proc.lock:
                    proc.logs.clear()
            return {"ok": True}
        with self._lock:
            self._global_logs.clear()
            for proc in self._processes.values():
                with proc.lock:
                    proc.logs.clear()
        return {"ok": True}

    def api_live_config(self, body: dict[str, Any]) -> dict[str, Any]:
        output_dir = Path(str(body.get("output_dir") or ""))
        if not output_dir.is_absolute():
            output_dir = self.workspace / "perception_outputs" / "scan_monodepth_validation"
        output_dir.mkdir(parents=True, exist_ok=True)
        config_path = output_dir / "live_config.json"
        cfg = {
            "target": str(body.get("target") or "").strip(),
            "label_query": str(body.get("label_query") or "").strip(),
            "truth_distance_m": body.get("truth_distance_m"),
        }
        config_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
        self.add_global_log(f"Live config updated: target={cfg['target']}")
        return {"ok": True, "config": cfg, "path": str(config_path)}

    def api_object_depth_dataset_config(self, body: dict[str, Any]) -> dict[str, Any]:
        dataset_name = _safe_name(str(body.get("dataset_name") or f"object_depth_{_now_stamp()}"))
        dataset_dir = OBJECT_DEPTH_DATASETS_DIR / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        cfg = {
            "target": str(body.get("target") or "chair").strip(),
            "label_query": str(body.get("label_query") or body.get("target") or "chair").strip(),
            "truth_distance_m": body.get("truth_distance_m"),
            "note": str(body.get("note") or "").strip(),
        }
        config_path = dataset_dir / "live_config.json"
        config_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
        self.add_global_log(f"Object-depth dataset config updated: {dataset_name} target={cfg['target']}")
        return {"ok": True, "dataset_dir": str(dataset_dir), "path": str(config_path), "config": cfg}

    def api_process_start(self, body: dict[str, Any]) -> dict[str, Any]:
        kind = str(body.get("kind") or "").strip()
        if kind == "slam":
            self._slam_map_base = str(body.get("map_base") or "")
        if kind == "keyframe_record":
            session = _safe_name(str(body.get("session_name") or f"session_{_now_stamp()}"))
            body["session_name"] = session
            self._last_keyframe_session_name = session
        if kind == "keyframe_build":
            return self._start_keyframe_scene_memory_build(body)
        if kind == "agent_sim":
            return self._start_agent_sim(body)
        if kind == "object_approach_live" and kind in self._processes and self._processes[kind].is_running():
            self._stop_process(kind)
            self.add_global_log("Restarted object_approach_live with current UI parameters.")
        # Auto-start sensor dependencies for tools that need them
        deps_map = {
            "object_pipeline": ["lidar_test", "camera_test"],
            "object_approach_live": ["camera_test", "navigation"],
            "object_depth_collect": ["lidar_test", "camera_test"],
            "lidar_camera_collect": ["lidar_test", "camera_test"],
            "lidar_camera_calib_run": ["lidar_test", "camera_test"],
            "agent": ["camera_test"],
        }
        # Agent also needs navigation stack when map_base is set
        if kind == "agent" and str(body.get("map_base") or ""):
            deps_map["agent"].insert(0, "navigation")
        for dep in deps_map.get(kind, []):
            if dep not in self._processes or not self._processes[dep].is_running():
                dep_cmd = self._command_for_kind(dep, body)
                self._start_process(dep, self._ros_command(dep_cmd))
                self.add_global_log(f"Auto-started dependency: {dep}")
        command = self._command_for_kind(kind, body)
        return self._start_process(kind, self._ros_command(command))

    def api_process_stop(self, body: dict[str, Any]) -> dict[str, Any]:
        kind = str(body.get("kind") or "")
        if kind == "slam":
            self._auto_clear_enabled = False
            self.add_global_log("Auto-clear disabled (SLAM stopped)")
        # Stop auto-started dependencies as well
        stop_deps = {
            "object_pipeline": ["lidar_test", "camera_test"],
            "object_depth_collect": ["lidar_test", "camera_test"],
            "lidar_camera_collect": ["lidar_test", "camera_test"],
        }
        result = self._stop_process(kind)
        if kind == "keyframe_record" and bool(body.get("build_scene_memory", True)):
            build_result = self._start_keyframe_scene_memory_build(body)
            result = {**result, "scene_memory_build": build_result}
        for dep in stop_deps.get(kind, []):
            if dep in self._processes and self._processes[dep].is_running():
                # Only stop if no other active process depends on it
                self._stop_process(dep)
                self.add_global_log(f"Auto-stopped dependency: {dep}")
        return result

    def api_process_stop_all(self, _body: dict[str, Any]) -> dict[str, Any]:
        results = {}
        for name in list(self._processes):
            results[name] = self._stop_process(name)
        return {"ok": True, "results": results}

    def api_teleop(self, body: dict[str, Any]) -> dict[str, Any]:
        command = str(body.get("command") or "").strip().lower()
        if command in {"stop", "zero", "emergency_stop"}:
            with self._lock:
                self._teleop_deadline_monotonic = 0.0
                self._teleop_last_command = (0.0, 0.0)
            self._publish_teleop_stop()
            return {"ok": True, "status": "ok", "summary": "stopped", "linear": 0.0, "angular": 0.0}

        mode = str(body.get("mode") or "normal").strip().lower()
        slow = mode in {"slow", "extra_slow", "safer"}
        linear_limit = 0.06 if slow else 0.18
        angular_limit = 0.20 if slow else 0.50
        try:
            linear = self._clamp(float(body.get("linear") or 0.0), linear_limit)
            angular = self._clamp(float(body.get("angular") or 0.0), angular_limit)
        except Exception:
            return {"ok": False, "error": "invalid teleop command"}
        if not (math.isfinite(linear) and math.isfinite(angular)):
            return {"ok": False, "error": "invalid teleop command"}
        with self._lock:
            self._teleop_deadline_monotonic = time.monotonic() + self._teleop_timeout_sec
            self._teleop_last_command = (linear, angular)
        self._publish_teleop(linear, angular)
        return {
            "ok": True,
            "status": "ok",
            "linear": linear,
            "angular": angular,
            "mode": "slow" if slow else "normal",
            "timeout_sec": self._teleop_timeout_sec,
            "source": "dashboard",
        }

    def rotation_tune_status(self) -> dict[str, Any]:
        with self._lock:
            return self._rotation_status_locked()

    def _rotation_params_from_body(self, body: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            params = dict(self._rotation_tune.get("params") or {})
        numeric_fields = {
            "fast_omega": (0.0, 3.5),
            "mid_omega": (0.0, 3.5),
            "slow_omega": (0.0, 3.5),
            "omega_cap": (0.0, 3.5),
            "fast_threshold_deg": (0.0, 180.0),
            "mid_threshold_deg": (0.0, 180.0),
            "yaw_tolerance_deg": (0.1, 30.0),
            "right_turn_shortcut_deg": (0.0, 180.0),
            "settle_time_sec": (0.0, 5.0),
            "timeout_sec": (0.5, 120.0),
        }
        for key, (lo, hi) in numeric_fields.items():
            if key not in body:
                continue
            value = float(body.get(key))
            if not math.isfinite(value):
                raise ValueError(f"invalid {key}")
            params[key] = max(lo, min(hi, value))
        cap = float(params.get("omega_cap", 3.5))
        for key in ("fast_omega", "mid_omega", "slow_omega"):
            params[key] = max(0.0, min(cap, float(params.get(key, 0.0))))
        if float(params.get("mid_threshold_deg", 10.0)) > float(params.get("fast_threshold_deg", 20.0)):
            params["mid_threshold_deg"] = params["fast_threshold_deg"]
        return params

    def api_rotation_tune_start(self, body: dict[str, Any]) -> dict[str, Any]:
        try:
            params = self._rotation_params_from_body(body)
        except Exception as exc:
            return {"ok": False, "error": f"invalid rotation params: {exc}"}
        with self._lock:
            current_yaw, source, _age = self._rotation_current_yaw_locked()
            mode = str(body.get("mode") or "absolute").strip().lower()
            if mode in {"relative", "delta"} or "relative_delta_deg" in body:
                if current_yaw is None:
                    return {"ok": False, "error": "当前没有可用 IMU/里程计角度，无法执行相对旋转。"}
                delta_deg = float(body.get("relative_delta_deg") or 0.0)
                if not math.isfinite(delta_deg):
                    return {"ok": False, "error": "invalid relative_delta_deg"}
                target_yaw = _normalize_angle_rad(float(current_yaw) + math.radians(delta_deg))
            else:
                target_deg = float(body.get("target_yaw_deg") or 0.0)
                if not math.isfinite(target_deg):
                    return {"ok": False, "error": "invalid target_yaw_deg"}
                target_yaw = _normalize_angle_rad(math.radians(target_deg))
            self._teleop_deadline_monotonic = 0.0
            self._teleop_last_command = (0.0, 0.0)
            self._rotation_tune.update(
                {
                    "active": True,
                    "status": "starting",
                    "message": f"开始旋转对齐，角度来源 {source}。",
                    "target_yaw_rad": target_yaw,
                    "start_time": time.monotonic(),
                    "settle_since": 0.0,
                    "current_omega": 0.0,
                    "direction": "stop",
                    "params": params,
                }
            )
            status = self._rotation_status_locked()
        self._publish_teleop_stop()
        self.add_global_log(
            "Rotation tune start: "
            f"target={status.get('target_yaw_deg'):.1f}deg "
            f"fast/mid/slow={params['fast_omega']:.2f}/{params['mid_omega']:.2f}/{params['slow_omega']:.2f} "
            f"thresholds={params['fast_threshold_deg']:.1f}/{params['mid_threshold_deg']:.1f} "
            f"right_shortcut={params['right_turn_shortcut_deg']:.1f}"
        )
        return status

    def api_rotation_tune_stop(self, _body: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            self._rotation_tune["active"] = False
            self._rotation_tune["status"] = "stopped"
            self._rotation_tune["message"] = "已停止旋转调参。"
            self._rotation_tune["current_omega"] = 0.0
            self._rotation_tune["direction"] = "stop"
            self._teleop_deadline_monotonic = 0.0
            self._teleop_last_command = (0.0, 0.0)
            status = self._rotation_status_locked()
        self._publish_teleop_stop()
        self.add_global_log("Rotation tune stopped.")
        return status

    def api_rotation_tune_save(self, body: dict[str, Any]) -> dict[str, Any]:
        try:
            params = self._rotation_params_from_body(body)
        except Exception as exc:
            return {"ok": False, "error": f"invalid rotation params: {exc}"}
        ROTATION_TUNE_CONFIG.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "params": params,
        }
        ROTATION_TUNE_CONFIG.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        with self._lock:
            self._rotation_tune["params"] = params
            self._rotation_tune["message"] = f"旋转参数已保存到 {ROTATION_TUNE_CONFIG}。"
            status = self._rotation_status_locked()
        self.add_global_log(
            "Rotation tune params saved: "
            f"fast/mid/slow={params['fast_omega']:.2f}/{params['mid_omega']:.2f}/{params['slow_omega']:.2f}, "
            f"thresholds={params['fast_threshold_deg']:.1f}/{params['mid_threshold_deg']:.1f}"
        )
        return status

    def _start_agent_sim(self, body: dict[str, Any]) -> dict[str, Any]:
        dataset_dir = _expand_path(body.get("dataset_dir"), KEYFRAMES_DIR / "")
        if dataset_dir.name != "selected" and (dataset_dir / "selected").exists():
            dataset_dir = dataset_dir / "selected"
        if not dataset_dir.exists():
            return {"ok": False, "error": f"Scene memory dataset not found: {dataset_dir}"}
        port = int(body.get("web_port") or AGENT_SIM_WEB_PORT)
        delay_sec = float(body.get("delay_sec") or 8.0)
        initial_position = body.get("initial_position")
        if not isinstance(initial_position, list) or len(initial_position) < 2:
            initial_position = [-0.07250774651765823, 0.21917006373405457, 0.0]
        config_file = str(body.get("config_file") or WORKSPACE / "src" / "caragent_agent" / "config" / "config.yaml")
        extra_config = Path(f"/tmp/caragent_agent_sim_{port}.yaml")
        extra_payload = {
            "scene_memory": {"dataset_dir": str(dataset_dir)},
            "paths": {"default_dataset_dir": str(dataset_dir)},
            "navigation": {
                "simulation_mode": True,
                "dry_run_navigation": False,
                "simulation_navigation_delay_sec": delay_sec,
                "simulation_navigation_delay_per_meter_sec": 0.0,
                "simulation_initial_position": initial_position,
                "simulation_initial_yaw_deg": 0.0,
            },
            "agent": {"target_resolution_dry_run": False},
            "web_ui": {
                "enabled": True,
                "host": "0.0.0.0",
                "port": port,
                "thread_id": f"caragent_sim_{port}_{_now_stamp()}",
                "session_checkpoint_enabled": False,
            },
            "interaction_profile": {
                "voice_enabled_default": True,
                "speak_agent_replies": True,
                "speak_guidance_events": True,
                "response_role_enabled": True,
            },
        }
        try:
            import yaml  # type: ignore

            extra_config.write_text(yaml.safe_dump(extra_payload, allow_unicode=True, sort_keys=False), encoding="utf-8")
        except Exception:
            extra_config.write_text(json.dumps(extra_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        args = [
            "ros2", "launch", "caragent_agent", "caragent_agent.launch.py",
            f"config_file:={config_file}",
            f"dataset_dir:={dataset_dir}",
        ]
        command = self._ros_command(_shell_join(args))
        env_overrides = {"CARAGENT_EXTRA_CONFIG_FILE": str(extra_config)}
        result = self._start_process("agent_sim", command, env_overrides=env_overrides)
        if result.get("ok"):
            host = self._host_hint()
            result.update(
                {
                    "sim_url": f"http://{host}:{port}/sim",
                    "lite_url": f"http://{host}:{port}/lite",
                    "web_port": port,
                    "extra_config": str(extra_config),
                }
            )
            self.add_global_log(f"agent_sim ready target: http://{host}:{port}/sim")
        return result

    def _command_for_kind(self, kind: str, body: dict[str, Any]) -> str:
        ports = _scan_ports()
        laser = str(body.get("laser_port") or (ports["serial"][0] if ports["serial"] else "/dev/ttyUSB0"))
        stm32 = str(body.get("stm32_port") or (ports["serial"][1] if len(ports["serial"]) > 1 else "/dev/ttyUSB1"))
        camera = str(body.get("camera_device") or (ports["cameras"][0] if ports["cameras"] else "/dev/video0"))
        if kind == "slam":
            map_base = str(body.get("map_base") or "")
            args = [
                "ros2", "launch", "caragent_bringup", "caragent_full.launch.py",
                "mode:=slam", f"laser_port:={laser}", f"stm32_port:={stm32}",
                f"camera_device:={camera}", "enable_camera:=false", "use_rviz:=true",
            ]
            args.extend(_camera_launch_args(body))
            if map_base:
                args.append(f"map_file_name:={map_base}")
            return _shell_join(args)
        if kind == "navigation":
            map_base = str(body.get("map_base") or (self.get_status().get("latest_map") or {}).get("base_path") or "")
            enable_cmd = "true" if bool(body.get("enable_cmd_vel")) else "false"
            enable_left_only = "true" if bool(body.get("enable_left_only_goal_proxy")) else "false"
            max_linear = str(body.get("max_linear_mps") or "0.40").strip() or "0.40"
            max_angular = "3.50"
            pre_align_strategy = str(body.get("pre_align_strategy") or "direct_bearing").strip()
            if pre_align_strategy not in {"direct_bearing", "path_heading"}:
                pre_align_strategy = "direct_bearing"
            path_heading_lookahead = str(body.get("path_heading_lookahead_m") or "0.70").strip() or "0.70"
            args = [
                "ros2", "launch", "caragent_bringup", "caragent_full.launch.py",
                "mode:=navigation", f"laser_port:={laser}", f"stm32_port:={stm32}",
                f"camera_device:={camera}", f"map_file_name:={map_base}", "use_rviz:=true", f"enable_cmd_vel:={enable_cmd}",
                f"enable_left_only_goal_proxy:={enable_left_only}", f"max_linear_mps:={max_linear}", f"max_angular_radps:={max_angular}",
            ]
            if bool(body.get("enable_left_only_goal_proxy")):
                args.append(f"pre_align_strategy:={pre_align_strategy}")
                args.append(f"path_heading_lookahead_m:={path_heading_lookahead}")
            args.extend(_camera_launch_args(body))
            yaml_path = str(body.get("map_yaml") or "")
            if yaml_path:
                args.append(f"map_yaml_file:={yaml_path}")
            return _shell_join(args)
        if kind == "keyframe_record":
            session = _safe_name(str(body.get("session_name") or f"session_{_now_stamp()}"))
            map_base = str(body.get("map_base") or (self.get_status().get("latest_map") or {}).get("base_path") or "")
            manual_only = "true" if bool(body.get("manual_only")) else "false"
            args = [
                "ros2", "launch", "caragent_memory", "caragent_keyframe_collect.launch.py",
                f"session_name:={session}", f"laser_port:={laser}", f"stm32_port:={stm32}",
                f"camera_device:={camera}", f"map_file_name:={map_base}", "use_rviz:=true",
                "camera_show_image:=true",
                f"manual_only:={manual_only}",
            ]
            args.extend(_camera_launch_args(body))
            return _shell_join(args)
        if kind == "keyframe_build":
            dataset = _expand_path(body.get("dataset"), KEYFRAMES_DIR / str(body.get("session_name") or ""))
            if dataset.name == "selected":
                dataset = dataset.parent
            annotation_mode = str(body.get("annotate") or "auto").strip().lower()
            if annotation_mode not in {"auto", "always", "never"}:
                annotation_mode = "auto"
            chunk_index_mode = str(body.get("chunk_index") or "auto").strip().lower()
            if chunk_index_mode not in {"auto", "always", "never"}:
                chunk_index_mode = "auto"
            batch_size = str(body.get("batch_size") or body.get("annotation_batch_size") or "36")
            args = [
                "ros2", "run", "caragent_memory", "build_scene_memory",
                "--dataset", str(dataset),
                "--clip-model", str(CLIP_MODEL),
                "--dinov2-model", str(DINO_MODEL),
                "--device", "GPU",
                "--dinov2-device", "NPU",
                "--annotate", annotation_mode,
                "--annotation-batch-size", batch_size,
                "--chunk-index", chunk_index_mode,
            ]
            if bool(body.get("adopt_existing")):
                args.append("--adopt-existing")
            if bool(body.get("force_annotation") or body.get("annotation_force")):
                args.append("--annotation-force")
            if bool(body.get("skip_annotation_clip") or body.get("annotation_skip_clip")):
                args.append("--annotation-skip-clip")
            if bool(body.get("force_chunk_index") or body.get("chunk_index_force")):
                args.append("--chunk-index-force")
            build_command = _shell_join(args)
            scene_map_command = _scene_map_generation_command(dataset)
            return (
                f"{build_command}; "
                "rc=$?; "
                f"{scene_map_command} || true; "
                "exit $rc"
            )
        if kind == "agent":
            config_file = str(body.get("config_file") or WORKSPACE / "src" / "caragent_agent" / "config" / "config.yaml")
            dataset_dir = str(body.get("dataset_dir") or "")
            args = ["ros2", "launch", "caragent_agent", "caragent_agent.launch.py", f"config_file:={config_file}"]
            if dataset_dir:
                args.append(f"dataset_dir:={dataset_dir}")
            return _shell_join(args)
        if kind == "lidar_test":
            return _shell_join([
                "ros2", "launch", "caragent_bringup", "rplidar_c1_slam.launch.py",
                f"laser_port:={laser}", "use_stm32_driver_node:=false", "use_slam:=false", "use_static_odom:=true", "use_rviz:=true",
                f"rviz_config:={WORKSPACE / 'src' / 'caragent_bringup' / 'rviz' / 'caragent_scan_tf.rviz'}",
            ])
        if kind == "stm32_test":
            return _shell_join([
                "ros2", "launch", "caragent_bringup", "rplidar_c1_slam.launch.py",
                f"stm32_port:={stm32}", "use_stm32_driver_node:=true", "use_slam:=false",
                "use_static_odom:=false", "use_rviz:=true", "enable_cmd_vel:=false",
                f"rviz_config:={WORKSPACE / 'src' / 'caragent_bringup' / 'rviz' / 'caragent_scan_tf.rviz'}",
            ])
        if kind == "camera_test":
            return _shell_join([
                "ros2", "launch", "caragent_vision", "huibo_stereo_camera.launch.py",
                f"device:={camera}", "show_image:=true", "publish_raw:=true", "publish_left:=true", "publish_right:=true",
                f"calib_file:={body.get('calib_file') or DEFAULT_STEREO_CALIB}",
                "publish_rect:=true",
                *_stereo_camera_launch_args(body),
            ])
        if kind == "object_pipeline":
            target = str(body.get("target") or "chair")
            label = str(body.get("label_query") or target)
            truth = str(body.get("truth_distance_m") or "")
            localization_mode = str(body.get("localization_mode") or "stereo").strip()
            if localization_mode not in {"stereo", "stereo_primary_mono_guard", "mono_relative_lidar", "mono_absolute"}:
                localization_mode = "stereo"
            output_dir = str(body.get("output_dir") or WORKSPACE / "perception_outputs" / "scan_monodepth_validation")
            args = [
                "python3", "-m", "caragent_agent.perception.fusion.live_scan_monodepth_validation",
                "--target", target, "--label-query", label, "--output-dir", output_dir,
                "--localization-mode", localization_mode,
                "--depth-device", "GPU", "--sam-device", "GPU",
            ]
            if truth:
                args.extend(["--truth-distance-m", truth])
            return _shell_join(args)
        if kind == "object_approach_live":
            target = str(body.get("target") or "chair").strip() or "chair"
            depth_backend = str(body.get("depth_backend") or "stereo_primary_mono_guard").strip()
            if depth_backend not in {"auto", "stereo", "stereo_primary_mono_guard", "mono_relative_lidar"}:
                depth_backend = "stereo_primary_mono_guard"
            stop_distance = str(body.get("stop_distance_m") or "0.80").strip() or "0.80"
            output_root = str(body.get("output_root") or WORKSPACE / "perception_outputs" / "object_approach_live")
            left_topic = str(body.get("left_topic") or "/stereo/left/image_raw")
            right_topic = str(body.get("right_topic") or "/stereo/right/image_raw")
            scan_topic = str(body.get("scan_topic") or "/scan")
            map_topic = str(body.get("map_topic") or "/global_costmap/costmap")
            goal_topic = str(body.get("goal_topic") or "/caragent/object_approach_goal")
            map_frame = str(body.get("map_frame") or "map")
            base_frame = str(body.get("base_frame") or "base_link")
            timeout_sec = str(body.get("timeout_sec") or "20.0").strip() or "20.0"
            args = [
                "ros2", "run", "caragent_agent", "object_approach_live_test",
                "--target", target,
                "--depth-backend", depth_backend,
                "--stop-distance", stop_distance,
                "--output-root", output_root,
                "--left-topic", left_topic,
                "--right-topic", right_topic,
                "--scan-topic", scan_topic,
                "--map-topic", map_topic,
                "--goal-topic", goal_topic,
                "--map-frame", map_frame,
                "--base-frame", base_frame,
                "--timeout-sec", timeout_sec,
            ]
            for field, flag in [
                ("grounding_query", "--grounding-query"),
                ("vlm_query", "--vlm-query"),
                ("sam_query", "--sam-query"),
            ]:
                value = str(body.get(field) or "").strip()
                if value:
                    args.extend([flag, value])
            return _shell_join(args)
        if kind == "object_depth_collect":
            dataset_name = _safe_name(str(body.get("dataset_name") or f"object_depth_{_now_stamp()}"))
            target = str(body.get("target") or "chair")
            label = str(body.get("label_query") or target)
            truth = str(body.get("truth_distance_m") or "")
            note = str(body.get("note") or "")
            args = [
                "python3", "-m", "caragent_agent.perception.fusion.collect_object_depth_dataset",
                "--dataset-root", str(OBJECT_DEPTH_DATASETS_DIR),
                "--dataset-name", dataset_name,
                "--target", target,
                "--label-query", label,
                "--note", note,
                "--grounding-device", "GPU",
                "--sam-device", "GPU",
                "--sam-decoder-device", "CPU",
            ]
            if truth:
                args.extend(["--truth-distance-m", truth])
            return _shell_join(args)
        if kind == "object_depth_eval":
            dataset_name = _safe_name(str(body.get("dataset_name") or ""))
            dataset_dir = str(body.get("dataset_dir") or (OBJECT_DEPTH_DATASETS_DIR / dataset_name))
            run_name = _safe_name(str(body.get("run_name") or "baseline"))
            modes = str(body.get("modes") or "stereo,mono_relative_lidar")
            args = [
                "python3", "-m", "caragent_agent.perception.fusion.evaluate_object_depth_dataset",
                "--dataset-dir", dataset_dir,
                "--run-name", run_name,
                "--modes", modes,
                "--grounding-device", "GPU",
                "--depth-device", "GPU",
                "--absolute-depth-device", "GPU",
                "--learned-stereo-device", "GPU",
                "--sam-device", "GPU",
                "--sam-decoder-device", "CPU",
            ]
            if bool(body.get("force")):
                args.append("--force")
            return _shell_join(args)
        if kind == "vlm_box_select":
            image = str(body.get("image") or "")
            query = str(body.get("query") or "").strip()
            grounding_query = str(body.get("grounding_query") or query).strip()
            if not image:
                raise ValueError("image is required for vlm_box_select")
            if not query:
                raise ValueError("query is required for vlm_box_select")
            output_dir = str(body.get("output_dir") or WORKSPACE / "perception_outputs" / "vlm_box_select")
            box_threshold = str(body.get("box_threshold") or "0.25")
            text_threshold = str(body.get("text_threshold") or "0.20")
            max_candidates = str(body.get("max_candidates") or "8")
            vlm_model = str(body.get("vlm_model") or "qwen3-vl-plus")
            grounding_device = str(body.get("grounding_device") or "GPU")
            return _shell_join([
                "python3", "-m", "caragent_agent.perception.grounding.vlm_select_box",
                "--image", image,
                "--query", query,
                "--grounding-query", grounding_query,
                "--output-dir", output_dir,
                "--grounding-device", grounding_device,
                "--box-threshold", box_threshold,
                "--text-threshold", text_threshold,
                "--max-candidates", max_candidates,
                "--vlm-model", vlm_model,
            ])
        if kind == "stereo_calib_capture":
            out = str(body.get("output_dir") or CALIB_DIR / "stereo_new" / f"capture_{_now_stamp()}")
            width, height, left_width, right_width, fps = _camera_resolution_values(body)
            args = [
                "ros2", "run", "caragent_vision", "capture_stereo_calibration",
                "--device", camera,
                "--output-dir", out,
                "--width", str(width),
                "--height", str(height),
                "--left-width", str(left_width),
                "--right-width", str(right_width),
                "--fps", f"{fps:g}",
            ]
            cols = str(body.get("cols") or "").strip()
            rows = str(body.get("rows") or "").strip()
            if cols and rows:
                args.extend(["--cols", cols, "--rows", rows, "--require-corners"])
            return _shell_join(args)
        if kind == "stereo_calib_run":
            image_dir = str(body.get("image_dir") or "")
            if not image_dir:
                raise ValueError("image_dir is required for stereo_calib_run")
            output = str(body.get("output") or Path(image_dir) / "stereo_calibration.npz")
            cols = str(body.get("cols") or "").strip()
            rows = str(body.get("rows") or "").strip()
            square_size = str(body.get("square_size_m") or "").strip()
            if not cols or not rows or not square_size:
                raise ValueError("cols, rows, and square_size_m are required for stereo_calib_run")
            return _shell_join([
                "ros2", "run", "caragent_vision", "calibrate_stereo_camera",
                "--image-dir", image_dir,
                "--output", output,
                "--cols", cols,
                "--rows", rows,
                "--square-size", square_size,
                "--save-overlays",
            ])
        if kind == "lidar_camera_collect":
            out = str(body.get("output_jsonl") or CALIB_DIR / "lidar_camera" / f"correspondences_{_now_stamp()}.jsonl")
            return _shell_join(["ros2", "run", "caragent_vision", "live_lidar_camera_correspondences", "--ros-args", "-p", f"output_jsonl:={out}"])
        if kind == "lidar_camera_calib_run":
            samples_jsonl = str(body.get("samples_jsonl") or "").strip()
            if not samples_jsonl:
                raise ValueError("samples_jsonl is required for lidar_camera_calib_run")
            output_json = str(body.get("output_json") or DEFAULT_EXTRINSICS)
            calib_file = str(body.get("calib_file") or DEFAULT_STEREO_CALIB)
            return _shell_join([
                "ros2", "run", "caragent_vision", "calibrate_lidar_camera_extrinsics",
                "--samples-jsonl", samples_jsonl,
                "--calib-file", calib_file,
                "--output-json", output_json,
            ])
        raise ValueError(f"Unsupported process kind: {kind}")

    def api_slam_initial_pose(self, body: dict[str, Any]) -> dict[str, Any]:
        map_base = str(body.get("map_base") or "")
        if not map_base:
            return {"ok": False, "error": "map_base is required"}
        x = float(body.get("x", 0.0))
        y = float(body.get("y", 0.0))
        yaw = float(body.get("yaw", 0.0))
        cmd = self._ros_command(
            _shell_join([
                "ros2", "service", "call", "/slam_toolbox/deserialize_map",
                "slam_toolbox/srv/DeserializePoseGraph",
                f"{{filename: '{map_base}', match_type: 2, initial_pose: {{x: {x}, y: {y}, theta: {yaw}}}}}",
            ])
        )
        self.add_global_log(f"Setting SLAM initial pose: x={x} y={y} yaw={yaw}")
        proc = _run_shell_utf8(cmd, timeout=15)
        output = (proc.stdout + proc.stderr).strip()
        self.add_global_log(output or f"deserialize_map returncode={proc.returncode}")
        return {"ok": proc.returncode == 0, "returncode": proc.returncode, "output": output}

    def _schedule_auto_clear(self) -> None:
        """Schedule next auto clear_changes call; only fires if still enabled and slam is running."""
        if not self._auto_clear_enabled:
            return
        if "slam" not in self._processes or not self._processes["slam"].is_running():
            self._auto_clear_enabled = False
            self.add_global_log("Auto-clear disabled: SLAM not running")
            return
        if not self._slam_map_base:
            self.add_global_log("Auto-clear disabled: not incremental mapping")
            self._auto_clear_enabled = False
            return
        t = threading.Timer(self._auto_clear_interval, self._do_clear_changes)
        t.daemon = True
        t.start()

    def _do_clear_changes(self) -> None:
        """Call /slam_toolbox/clear_changes and reschedule."""
        cmd = self._ros_command(
            "ros2 service call /slam_toolbox/clear_changes std_srvs/srv/Empty '{}'"
        )
        try:
            proc = _run_shell_utf8(cmd, timeout=10)
            output = (proc.stdout + proc.stderr).strip()
            self.add_global_log(f"auto clear_changes: {output or f'returncode={proc.returncode}'}")
        except Exception as exc:
            self.add_global_log(f"auto clear_changes failed: {exc}")
        self._schedule_auto_clear()

    def api_slam_clear_changes(self, _body: dict[str, Any]) -> dict[str, Any]:
        cmd = self._ros_command(
            "ros2 service call /slam_toolbox/clear_changes std_srvs/srv/Empty '{}'"
        )
        self.add_global_log("Manual clear_changes triggered")
        proc = _run_shell_utf8(cmd, timeout=10)
        output = (proc.stdout + proc.stderr).strip()
        self.add_global_log(output or f"clear_changes returncode={proc.returncode}")
        return {"ok": proc.returncode == 0, "returncode": proc.returncode, "output": output}

    def api_slam_auto_clear(self, body: dict[str, Any]) -> dict[str, Any]:
        enable = bool(body.get("enable"))
        if enable:
            if not self._slam_map_base:
                return {"ok": False, "error": "auto-clear requires incremental mapping (map_base)."}
            if "slam" not in self._processes or not self._processes["slam"].is_running():
                return {"ok": False, "error": "SLAM is not running."}
            self._auto_clear_enabled = True
            self.add_global_log(f"Auto clear_changes enabled (every {self._auto_clear_interval}s)")
            self._schedule_auto_clear()
        else:
            self._auto_clear_enabled = False
            self.add_global_log("Auto clear_changes disabled")
        return {"ok": True, "auto_clear": self._auto_clear_enabled}

    def api_save_map(self, body: dict[str, Any]) -> dict[str, Any]:
        MAPS_DIR.mkdir(parents=True, exist_ok=True)
        base = MAPS_DIR / _safe_name(str(body.get("name") or f"map_{_now_stamp()}"))
        serialize_cmd = self._ros_command(
            _shell_join([
                "ros2", "service", "call", "/slam_toolbox/serialize_map",
                "slam_toolbox/srv/SerializePoseGraph", f"{{filename: '{base}'}}",
            ])
        )
        self.add_global_log(f"Saving serialized SLAM map: {base}")
        first = _run_shell_utf8(serialize_cmd, timeout=30)
        self.add_global_log((first.stdout + first.stderr).strip() or f"serialize_map returncode={first.returncode}")
        map_saver_cmd = self._ros_command(_shell_join(["ros2", "run", "nav2_map_server", "map_saver_cli", "-f", str(base)]))
        second = _run_shell_utf8(map_saver_cmd, timeout=30)
        self.add_global_log((second.stdout + second.stderr).strip() or f"map_saver_cli returncode={second.returncode}")
        files = _path_with_suffixes(base)
        return {
            "ok": bool(files),
            "base_path": str(base),
            "files": [str(path) for path in files],
            "serialize_returncode": first.returncode,
            "map_saver_returncode": second.returncode,
            "error": "" if files else "No map files were found after save commands.",
        }

    def api_select_keyframes(self, body: dict[str, Any]) -> dict[str, Any]:
        dataset = _expand_path(body.get("dataset"), KEYFRAMES_DIR / "")
        if not dataset.exists():
            return {"ok": False, "error": f"Dataset not found: {dataset}"}
        build_body = dict(body)
        build_body["dataset"] = str(dataset)
        build_body.setdefault("annotate", "auto")
        build_body.setdefault("chunk_index", "auto")
        return self._start_keyframe_scene_memory_build(build_body)

    def api_visualize_keyframes(self, body: dict[str, Any]) -> dict[str, Any]:
        dataset = _expand_path(body.get("dataset"), KEYFRAMES_DIR / "")
        session = dataset.parent if dataset.name == "selected" else dataset
        selected = session / "selected"
        if not session.exists():
            return {"ok": False, "error": f"Dataset not found: {session}"}
        selected.mkdir(parents=True, exist_ok=True)
        output = selected / "scene_map.html"
        self._write_scene_map_html(session, output)
        return {"ok": True, "path": str(output)}

    def api_annotate_keyframes(self, body: dict[str, Any]) -> dict[str, Any]:
        dataset = _expand_path(body.get("dataset"), KEYFRAMES_DIR / "")
        if dataset.name != "selected":
            dataset = dataset / "selected"
        batch_size = int(body.get("batch_size") or 36)
        ids_str = str(body.get("ids") or "").strip()
        args = ["python3", "-m", "caragent_agent.scripts.annotate_keyframes",
                "--dataset-dir", str(dataset), "--batch-size", str(batch_size)]
        if ids_str:
            args.extend(["--ids", ids_str])
        elif bool(body.get("force")):
            args.append("--force")
        if bool(body.get("skip_clip")):
            args.append("--skip-clip")
        env_overrides: dict[str, str] = {}
        command = self._ros_command(_shell_join(args))
        return self._start_process("keyframe_annotate", command, env_overrides=env_overrides)

    def api_keyframe_capture_once(self, body: dict[str, Any]) -> dict[str, Any]:
        session: Path | None = None
        before_count = 0
        dataset_value = body.get("dataset")
        if dataset_value:
            dataset = _expand_path(dataset_value, KEYFRAMES_DIR / "")
            session = dataset.parent if dataset.name == "selected" else dataset
            before_count, _ = _manifest_count_and_last_id(session)
        command = self._ros_command(
            "ros2 service call /keyframe_recorder/capture_once std_srvs/srv/Trigger '{}'"
        )
        try:
            proc = _run_shell_utf8(command, timeout=8)
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": "等待关键帧采集服务超时，请确认正在录制并已完成定位。"}
        output = (proc.stdout + proc.stderr).strip()
        compact = output.replace(" ", "").lower()
        ok = proc.returncode == 0 and ("success=true" in compact or "success:true" in compact)
        self.add_global_log(f"manual keyframe capture requested: {output or proc.returncode}")
        if not ok:
            message = "人工补录请求失败，请确认补录采集已启动。"
            return {
                "ok": False,
                "returncode": proc.returncode,
                "output": output,
                "error": message,
                "message": message,
            }
        if session is not None:
            deadline = time.time() + 10.0
            while time.time() < deadline:
                count, last_id = _manifest_count_and_last_id(session)
                if count > before_count:
                    return {
                        "ok": True,
                        "returncode": proc.returncode,
                        "output": output,
                        "frame_id": last_id,
                        "message": f"已保存关键帧 {last_id}。",
                    }
                time.sleep(0.25)
            message = "已发送保存请求，但还没有等到新关键帧落盘；请确认定位稳定、画面正常后再试。"
            return {
                "ok": False,
                "returncode": proc.returncode,
                "output": output,
                "error": message,
                "message": message,
            }
        return {
            "ok": True,
            "returncode": proc.returncode,
            "output": output,
            "message": "已请求保存下一帧关键帧。",
        }

    def api_keyframe_manual_record_start(self, body: dict[str, Any]) -> dict[str, Any]:
        dataset = _expand_path(body.get("dataset"), KEYFRAMES_DIR / "")
        session = dataset.parent if dataset.name == "selected" else dataset
        if not session.exists():
            return {"ok": False, "error": f"Session not found: {session}"}
        if not (session / "manifest.jsonl").exists():
            return {"ok": False, "error": f"Session has no manifest.jsonl: {session}"}
        with self._lock:
            proc = self._processes.get("keyframe_record")
            if proc is not None and proc.is_running():
                return {"ok": False, "error": "关键帧采集正在运行，请先结束当前采集或补录。"}

        session_config = _read_json_object(session / "session.json")
        params = session_config.get("parameters") if isinstance(session_config.get("parameters"), dict) else {}
        map_base = str(body.get("map_base") or params.get("map_file_name") or "").strip()
        build_body = {
            "kind": "keyframe_record",
            "session_name": session.name,
            "map_base": map_base,
            "manual_only": True,
            "build_scene_memory": False,
            "laser_port": body.get("laser_port"),
            "stm32_port": body.get("stm32_port"),
            "camera_device": body.get("camera_device"),
            "camera_resolution": body.get("camera_resolution"),
        }
        self._last_keyframe_session_name = session.name
        self.add_global_log(f"Manual keyframe append mode queued for {session}")
        return self.api_process_start(build_body)

    def _refresh_selected_after_manual_remove(self, selected: Path, removed_ids: set[int]) -> dict[str, Any]:
        import shutil

        constructed = selected / "constructed_memory"
        node_dir = constructed / "keyframe_nodes"
        selected_manifest_path = selected / "selected_manifest.jsonl"
        rejected_manifest_path = selected / "rejected_manifest.jsonl"

        selected_rows = _read_jsonl_objects(selected_manifest_path)
        kept_rows: list[dict[str, Any]] = []
        removed_rows: list[dict[str, Any]] = []
        for item in selected_rows:
            try:
                frame_id = int(str(item.get("frame_id") or "").lstrip("0") or "0")
            except ValueError:
                kept_rows.append(item)
                continue
            if frame_id in removed_ids:
                removed_rows.append(item)
            else:
                kept_rows.append(item)

        if not removed_rows:
            return {"removed": 0, "message": "所选编号不在当前 selected manifest 中。"}

        for item in removed_rows:
            frame = str(item.get("frame_id") or "").zfill(6)
            reject = {
                "frame_id": frame,
                "reject_reason": "manual_removed_after_selection",
                "quality_ok": item.get("quality_ok"),
                "manual": item.get("manual"),
                "max_similarity": item.get("max_similarity"),
                "nearest_distance_m": item.get("nearest_distance_m"),
                "timestamp": item.get("timestamp"),
                "x": item.get("x"),
                "y": item.get("y"),
                "yaw": item.get("yaw"),
            }
            with rejected_manifest_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(reject, ensure_ascii=False, sort_keys=True) + "\n")

        _write_jsonl_objects(selected_manifest_path, kept_rows)

        for frame_id in removed_ids:
            frame = f"{frame_id:06d}"
            for relative in [
                Path("raw") / f"{frame}.png",
                Path("left") / f"{frame}.png",
                Path("right") / f"{frame}.png",
                Path("pose") / f"{frame}_pose.json",
                Path("meta") / f"{frame}_meta.json",
                Path("scan") / f"{frame}_scan.npz",
                Path("embeddings") / "clip" / f"{frame}.npy",
                Path("embeddings") / "dinov2" / f"{frame}.npy",
                Path("constructed_memory") / "keyframe_nodes" / f"kf_{frame}.json",
                Path("constructed_memory") / "keyframe_nodes" / f"kf_{frame_id}.json",
            ]:
                try:
                    (selected / relative).unlink()
                except FileNotFoundError:
                    pass

        # Refresh graph with remaining keyframe node ids in numeric order.
        remaining_ids: list[int] = []
        for path in sorted(node_dir.glob("kf_*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                remaining_ids.append(int(data.get("kf_id")))
            except Exception:
                continue
        remaining_ids = sorted(set(remaining_ids))
        positions: dict[int, tuple[float, float]] = {}
        for path in sorted(node_dir.glob("kf_*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                kf_id = int(data.get("kf_id"))
                pos = data.get("position") or [0.0, 0.0]
                positions[kf_id] = (float(pos[0]), float(pos[1]))
            except Exception:
                continue
        edges = []
        for a, b in zip(remaining_ids, remaining_ids[1:]):
            ax, ay = positions.get(a, (0.0, 0.0))
            bx, by = positions.get(b, (0.0, 0.0))
            edges.append([a, b, {"weight": math.hypot(ax - bx, ay - by), "type": "sequential"}])
        (constructed).mkdir(parents=True, exist_ok=True)
        (constructed / "keyframe_graph.json").write_text(
            json.dumps({"nodes": remaining_ids, "edges": edges}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        # Filter persistent semantic chunk index when row alignment is valid; otherwise remove it.
        chunk_result = "none"
        records_path = constructed / "semantic_chunk_index_records.json"
        matrix_path = constructed / "semantic_chunk_index_matrix.npy"
        if records_path.exists():
            try:
                records_payload = json.loads(records_path.read_text(encoding="utf-8"))
                records = records_payload.get("records") if isinstance(records_payload, dict) else []
                if not isinstance(records, list):
                    records = []
                keep_indices = [
                    index for index, record in enumerate(records)
                    if int(record.get("keyframe_id", -1)) not in removed_ids
                ]
                filtered_records = [records[index] for index in keep_indices]
                matrix_ok = False
                if matrix_path.exists():
                    try:
                        import numpy as np  # type: ignore

                        matrix = np.load(matrix_path)
                        if len(records) == int(matrix.shape[0]):
                            np.save(matrix_path, matrix[keep_indices])
                            matrix_ok = True
                    except Exception:
                        matrix_ok = False
                if matrix_path.exists() and not matrix_ok:
                    matrix_path.unlink(missing_ok=True)
                    chunk_result = "records_filtered_matrix_removed"
                else:
                    chunk_result = "filtered"
                if isinstance(records_payload, dict):
                    metadata = records_payload.get("metadata")
                    if isinstance(metadata, dict):
                        metadata["node_ids"] = remaining_ids
                        metadata["record_count"] = len(filtered_records)
                        if matrix_ok:
                            metadata["matrix_shape"] = [len(filtered_records)] + list(matrix.shape[1:])  # type: ignore[name-defined]
                    records_payload["records"] = filtered_records
                    records_path.write_text(json.dumps(records_payload, ensure_ascii=False, indent=2), encoding="utf-8")
            except Exception:
                try:
                    records_path.unlink()
                except FileNotFoundError:
                    pass
                try:
                    matrix_path.unlink()
                except FileNotFoundError:
                    pass
                chunk_result = "removed_due_to_error"

        # Refresh lightweight review HTML using the same scene-map source.
        try:
            self._write_scene_map_html(selected, selected / "scene_map.html")
        except Exception as exc:
            self.add_global_log(f"Scene map refresh after keyframe removal failed: {exc}")
        try:
            review_items = []
            for item in kept_rows:
                frame = str(item.get("frame_id") or "")
                left_path = selected / str(item.get("left_path") or "")
                img_src = html.escape(f"/api/file?path={left_path.resolve()}")
                reason = html.escape(str(item.get("selected_reason") or "selected"))
                review_items.append(
                    "<article>"
                    f"<img src='{img_src}' alt='kf {html.escape(frame)}'>"
                    f"<h2>#{html.escape(frame)} {reason}</h2>"
                    f"<p>x={float(item.get('x') or 0.0):.2f} y={float(item.get('y') or 0.0):.2f}</p>"
                    "</article>"
                )
            review_html = """<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>CarAgent Keyframe Review</title><style>body{font-family:Arial,sans-serif;margin:24px;background:#f5f5f5;color:#1f2933}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:12px}article{background:white;border:1px solid #d7dde4;border-radius:6px;padding:10px}img{width:100%;aspect-ratio:4/3;object-fit:cover;background:#111}h2{font-size:14px;margin:8px 0 4px}p{font-size:12px;margin:3px 0}</style></head><body><h1>CarAgent Keyframe Review</h1><p>Selected: SELECTED_COUNT</p><section class="grid">CARDS</section></body></html>"""
            review_html = review_html.replace("SELECTED_COUNT", str(len(kept_rows))).replace("CARDS", "\n".join(review_items))
            (selected / "review.html").write_text(review_html, encoding="utf-8")
        except Exception as exc:
            self.add_global_log(f"Review refresh after keyframe removal failed: {exc}")

        # Refresh summary and scene manifest counts.
        summary_path = selected / "scene_memory_summary.json"
        summary = _read_json_object(summary_path)
        rejected_count = len(_read_jsonl_objects(rejected_manifest_path))
        semantic_nodes = 0
        for path in node_dir.glob("kf_*.json"):
            data = _read_json_object(path)
            if str(data.get("semantic") or "").strip():
                semantic_nodes += 1
        if summary:
            selection = summary.setdefault("selection", {})
            if isinstance(selection, dict):
                selection["selected_count"] = len(kept_rows)
                selection["rejected_count"] = rejected_count
            annotation = summary.setdefault("annotation", {})
            if isinstance(annotation, dict):
                annotation["keyframe_nodes"] = len(remaining_ids)
                annotation["semantic_nodes"] = semantic_nodes
            counts = summary.setdefault("counts", {})
            if isinstance(counts, dict):
                counts["selected_frames"] = len(kept_rows)
                counts["rejected_frames"] = rejected_count
                counts["keyframe_nodes"] = len(remaining_ids)
                counts["semantic_nodes"] = semantic_nodes
            summary["manual_removed_keyframes"] = sorted(set(summary.get("manual_removed_keyframes", [])) | set(removed_ids)) if isinstance(summary.get("manual_removed_keyframes", []), list) else sorted(removed_ids)
            summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

        manifest_path = constructed / "scene_memory_manifest.json"
        manifest = _read_json_object(manifest_path)
        if manifest:
            counts = manifest.setdefault("counts", {})
            if isinstance(counts, dict):
                counts["selected_frames"] = len(kept_rows)
                counts["rejected_frames"] = rejected_count
                counts["keyframe_nodes"] = len(remaining_ids)
                counts["semantic_nodes"] = semantic_nodes
                if records_path.exists():
                    payload = _read_json_object(records_path)
                    records = payload.get("records")
                    counts["semantic_chunks"] = len(records) if isinstance(records, list) else counts.get("semantic_chunks")
                elif "semantic_chunks" in counts:
                    counts["semantic_chunks"] = 0
            manifest["manual_removed_keyframes"] = sorted(removed_ids)
            manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

        return {
            "removed": len(removed_rows),
            "selected_frames": len(kept_rows),
            "keyframe_nodes": len(remaining_ids),
            "semantic_nodes": semantic_nodes,
            "chunk_index": chunk_result,
        }

    def api_keyframe_remove(self, body: dict[str, Any]) -> dict[str, Any]:
        dataset = _expand_path(body.get("dataset"), KEYFRAMES_DIR / "")
        selected = dataset if dataset.name == "selected" else dataset / "selected"
        if not selected.exists():
            return {"ok": False, "error": f"Selected dataset not found: {selected}"}
        try:
            ids = _parse_keyframe_id_list(body.get("ids") or body.get("keyframe_ids"))
        except ValueError as exc:
            return {"ok": False, "error": str(exc)}
        if not ids:
            return {"ok": False, "error": "No keyframe ids were provided."}
        backup_root = selected.parent / ".manual_remove_backups"
        backup_root.mkdir(parents=True, exist_ok=True)
        backup = backup_root / f"{selected.parent.name}_{_now_stamp()}"
        try:
            import shutil

            for rel in [
                "selected_manifest.jsonl",
                "rejected_manifest.jsonl",
                "scene_memory_summary.json",
                "review.html",
                "scene_map.html",
                "constructed_memory/keyframe_graph.json",
                "constructed_memory/semantic_chunk_index_records.json",
                "constructed_memory/semantic_chunk_index_matrix.npy",
                "constructed_memory/scene_memory_manifest.json",
            ]:
                src = selected / rel
                if src.exists():
                    dst = backup / rel
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dst)
        except Exception as exc:
            return {"ok": False, "error": f"Failed to create backup before removal: {exc}"}
        result = self._refresh_selected_after_manual_remove(selected, set(ids))
        self.add_global_log(
            f"Manual keyframe removal on {selected}: ids={ids}, removed={result.get('removed')}, "
            f"selected={result.get('selected_frames')}"
        )
        return {"ok": True, "ids": ids, "backup": str(backup), **result}

    def api_keyframe_nodes(self, body: dict[str, Any]) -> dict[str, Any]:
        """Return list of keyframe nodes in a selected dataset, with id, semantic, x, y."""
        dataset = _expand_path(body.get("dataset"), KEYFRAMES_DIR / "")
        if dataset.name != "selected":
            dataset = dataset / "selected"
        node_dir = dataset / "constructed_memory" / "keyframe_nodes"
        nodes = []
        for path in sorted(node_dir.glob("kf_*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            pos = data.get("position") or [0, 0, 0]
            nodes.append({
                "kf_id": data.get("kf_id"),
                "name": data.get("name") or path.stem,
                "x": float(pos[0]) if len(pos) > 0 else 0.0,
                "y": float(pos[1]) if len(pos) > 1 else 0.0,
                "semantic": str(data.get("semantic") or "")[:80],
                "has_semantic": bool(str(data.get("semantic") or "").strip()),
            })
        return {"ok": True, "nodes": nodes}

    def _write_scene_map_html(self, dataset: Path, output: Path) -> None:
        import math as _m
        selected = dataset if dataset.name == "selected" else dataset / "selected"
        source_dataset = selected.parent if selected.name == "selected" else dataset

        def yaw_from_orientation(orient: Any) -> float:
            try:
                qx, qy, qz, qw = float(orient[0]), float(orient[1]), float(orient[2]), float(orient[3])
                return _m.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
            except Exception:
                return 0.0

        def load_raw_nodes(root: Path) -> list[dict[str, Any]]:
            manifest_path = root / "manifest.jsonl"
            raw_nodes: list[dict[str, Any]] = []
            if not manifest_path.exists():
                return raw_nodes
            with manifest_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    try:
                        item = json.loads(line)
                    except Exception:
                        continue
                    frame_id = str(item.get("frame_id") or "").strip()
                    pose_path = root / str(item.get("pose_path") or "")
                    pose = _read_json_object(pose_path)
                    image_path = root / str(item.get("left_path") or item.get("raw_path") or "")
                    position = [
                        float(pose.get("x", item.get("x", 0.0)) or 0.0),
                        float(pose.get("y", item.get("y", 0.0)) or 0.0),
                        float(pose.get("z", item.get("z", 0.0)) or 0.0),
                    ]
                    yaw = float(pose.get("yaw", item.get("yaw", 0.0)) or 0.0)
                    raw_nodes.append(
                        {
                            "id": frame_id,
                            "name": frame_id,
                            "x": position[0],
                            "y": position[1],
                            "yaw": yaw,
                            "image": str(image_path.resolve()) if image_path.exists() else "",
                            "semantic": str(item.get("trigger_reason") or "原始采集关键帧"),
                            "data_path": str(pose_path.resolve()) if pose_path.exists() else str(manifest_path.resolve()),
                        }
                    )
            return raw_nodes

        raw_nodes = load_raw_nodes(source_dataset)
        node_dir = selected / "constructed_memory" / "keyframe_nodes"
        selected_nodes = []
        for path in sorted(node_dir.glob("kf_*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            pos = data.get("position") or [0, 0, 0]
            image_path = str(data.get("rgb_path") or "")
            if image_path and not Path(image_path).is_absolute():
                image_path = str((selected / image_path).resolve())
            orient = data.get("orientation") or [0, 0, 0, 1]
            yaw = yaw_from_orientation(orient)
            selected_nodes.append(
                {
                    "id": data.get("kf_id"),
                    "name": data.get("name") or path.stem,
                    "x": float(pos[0]) if len(pos) > 0 else 0.0,
                    "y": float(pos[1]) if len(pos) > 1 else 0.0,
                    "yaw": yaw,
                    "image": image_path,
                    "semantic": data.get("semantic") or "",
                    "data_path": str(path.resolve()),
                }
            )
        graph_path = selected / "constructed_memory" / "keyframe_graph.json"
        edges = []
        if graph_path.exists():
            try:
                graph = json.loads(graph_path.read_text(encoding="utf-8"))
                edges = graph.get("edges", [])
            except Exception:
                edges = []
        manifest = selected / "constructed_memory" / "scene_memory_manifest.json"
        summary = selected / "scene_memory_summary.json"
        html_text = SCENE_MAP_TEMPLATE.replace(
            "__DATA__",
            json.dumps(
                {
                    "raw_nodes": raw_nodes,
                    "selected_nodes": selected_nodes,
                    "edges": edges,
                    "source_dataset": str(source_dataset.resolve()),
                    "selected_dataset": str(selected.resolve()),
                    "map": _map_metadata_for_dataset(source_dataset),
                    "manifest": str(manifest.resolve()) if manifest.exists() else "",
                    "summary": str(summary.resolve()) if summary.exists() else "",
                },
                ensure_ascii=False,
            ),
        )
        output.write_text(html_text, encoding="utf-8")


SCENE_MAP_TEMPLATE = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>CarAgent Scene Map</title>
<style>
body{font-family:Arial,sans-serif;margin:0;background:#101827;color:#e5e7eb}main{display:grid;grid-template-columns:1fr 410px;height:100vh}.map-wrap{min-width:0;min-height:0;position:relative;background:#0b1020}canvas{width:100%;height:100%;display:block;background:#0b1020}.side{border-left:1px solid #334155;padding:16px;overflow:auto}.muted{color:#94a3b8}.warn{color:#fbbf24}img{max-width:100%;border:1px solid #334155}.card{background:#172033;border:1px solid #334155;border-radius:8px;padding:12px;margin-top:10px}a{color:#7dd3fc}.toolbar{display:flex;gap:8px;margin:10px 0;flex-wrap:wrap}.toolbar button{background:#26364f;color:#e5e7eb;border:1px solid #334155;border-radius:6px;padding:7px 10px}.toolbar button.active{border-color:#38bdf8;background:#075985}.list{display:none;grid-template-columns:1fr;gap:10px}.list.show{display:grid}.map-wrap.hide{display:none}.stats{font-size:13px;line-height:1.45}
</style>
</head><body><main><div class="map-wrap" id="map-wrap"><canvas id="map"></canvas></div><aside class="side"><h2>Scene Map</h2><div class="toolbar"><button id="raw-btn">原始采集</button><button id="selected-btn">筛选结果</button><button id="map-btn">地图视图</button><button id="list-btn">列表视图</button></div><div id="stats" class="stats muted"></div><div id="links" class="muted"></div><div id="info" class="card muted">点击关键帧点查看图像和语义。</div><div id="list" class="list"></div></aside></main>
<script>
(function(){
var data=__DATA__,c=document.getElementById('map'),ctx=c.getContext('2d'),info=document.getElementById('info'),ARR=18,pts=[],list=document.getElementById('list'),wrap=document.getElementById('map-wrap'),mode=(data.selected_nodes&&data.selected_nodes.length)?'selected':'raw',mapImg=null,mapTransform=null;
function fileLink(path,label){return path?'<a target="_blank" href="/api/file?path='+encodeURIComponent(path)+'">'+label+'</a>':''}
function nodes(){return mode==='selected'?(data.selected_nodes||[]):(data.raw_nodes||[])}
function worldBounds(){var all=(data.raw_nodes||[]).concat(data.selected_nodes||[]);if(!all.length)return null;var xs=all.map(function(n){return n.x}),ys=all.map(function(n){return n.y});return {minx:Math.min.apply(null,xs),maxx:Math.max.apply(null,xs),miny:Math.min.apply(null,ys),maxy:Math.max.apply(null,ys)}}
function buildMapTransform(){var m=data.map||{};if(!mapImg||!m.image)return null;var res=Number(m.resolution)||0.05,origin=m.origin||[0,0,0],pad=24*devicePixelRatio,scale=Math.min((c.width-2*pad)/mapImg.naturalWidth,(c.height-2*pad)/mapImg.naturalHeight);if(!isFinite(scale)||scale<=0)scale=1;var drawW=mapImg.naturalWidth*scale,drawH=mapImg.naturalHeight*scale,offX=(c.width-drawW)/2,offY=(c.height-drawH)/2;return {res:res,origin:origin,scale:scale,offX:offX,offY:offY,width:mapImg.naturalWidth,height:mapImg.naturalHeight,drawW:drawW,drawH:drawH}}
function project(n){if(mapTransform){var ox=Number(mapTransform.origin[0])||0,oy=Number(mapTransform.origin[1])||0;var px=(n.x-ox)/mapTransform.res,py=mapTransform.height-(n.y-oy)/mapTransform.res;return {n:n,x:mapTransform.offX+px*mapTransform.scale,y:mapTransform.offY+py*mapTransform.scale}}return null}
function fallbackProjector(ns){var bounds=worldBounds();if(!bounds)return function(n){return {n:n,x:c.width/2,y:c.height/2}};var pad=60*devicePixelRatio,sx=(c.width-2*pad)/Math.max(1,bounds.maxx-bounds.minx),sy=(c.height-2*pad)/Math.max(1,bounds.maxy-bounds.miny),s=Math.min(sx,sy);return function(n){return {n:n,x:pad+(n.x-bounds.minx)*s,y:c.height-pad-(n.y-bounds.miny)*s}}}
function setMode(next){mode=next;document.getElementById('raw-btn').classList.toggle('active',mode==='raw');document.getElementById('selected-btn').classList.toggle('active',mode==='selected');renderList();resize();var ns=nodes();info.className='card muted';info.textContent=ns.length?'点击关键帧点查看图像和语义。':'当前模式没有关键帧数据。'}
document.getElementById('raw-btn').onclick=function(){setMode('raw')};
document.getElementById('selected-btn').onclick=function(){setMode('selected')};
document.getElementById('map-btn').onclick=function(){wrap.classList.remove('hide');list.classList.remove('show');this.classList.add('active');document.getElementById('list-btn').classList.remove('active');resize()};
document.getElementById('list-btn').onclick=function(){wrap.classList.add('hide');list.classList.add('show');this.classList.add('active');document.getElementById('map-btn').classList.remove('active')};
var mapLinks=[];if(data.map&&data.map.yaml)mapLinks.push(fileLink(data.map.yaml,'查看地图 YAML'));if(data.map&&data.map.image)mapLinks.push(fileLink(data.map.image,'查看雷达地图图像'));mapLinks.push(fileLink(data.manifest,'查看场景 manifest'));mapLinks.push(fileLink(data.summary,'查看 summary'));document.getElementById('links').innerHTML=mapLinks.filter(Boolean).join(' · ');
document.getElementById('stats').innerHTML='原始采集关键帧：'+(data.raw_nodes||[]).length+' 个；筛选结果：'+(data.selected_nodes||[]).length+' 个。'+((data.map&&data.map.image)?'<br>底图：真实雷达地图'+(data.map.fallback_latest?'（未在 session 中找到地图名，暂用最新地图）':''):'<br><span class="warn">未找到地图 YAML/PGM，退回关键帧散点视图。</span>')+((!(data.selected_nodes||[]).length && (data.raw_nodes||[]).length)?'<br><span class="warn">筛选结果为空或构建失败，当前显示原始采集结果。</span>':'');
function resize(){c.width=Math.max(1,c.clientWidth*devicePixelRatio);c.height=Math.max(1,c.clientHeight*devicePixelRatio);mapTransform=buildMapTransform();draw()}
window.addEventListener('resize',resize);
function arrow(cx,cy,yaw,len){var ex=cx+Math.cos(yaw)*len,ey=cy-Math.sin(yaw)*len;ctx.beginPath();ctx.moveTo(cx,cy);ctx.lineTo(ex,ey);ctx.stroke();var h=len*0.4;ctx.beginPath();ctx.moveTo(ex,ey);ctx.lineTo(ex-Math.cos(yaw+0.6)*h,ey+Math.sin(yaw+0.6)*h);ctx.lineTo(ex-Math.cos(yaw-0.6)*h,ey+Math.sin(yaw-0.6)*h);ctx.closePath();ctx.fill()}
function draw(){ctx.clearRect(0,0,c.width,c.height);ctx.fillStyle='#0b1020';ctx.fillRect(0,0,c.width,c.height);if(mapImg&&mapTransform){ctx.imageSmoothingEnabled=false;ctx.drawImage(mapImg,mapTransform.offX,mapTransform.offY,mapTransform.drawW,mapTransform.drawH);ctx.strokeStyle='#64748b';ctx.lineWidth=1;ctx.strokeRect(mapTransform.offX,mapTransform.offY,mapTransform.drawW,mapTransform.drawH)}var ns=nodes();if(!ns.length)return;var fp=fallbackProjector(ns);pts=ns.map(function(n){return project(n)||fp(n)});ctx.strokeStyle='#334155';ctx.lineWidth=1;var edges=mode==='selected'?(data.edges||[]):[];for(var i=0;i<edges.length;i++){var e=edges[i],aid=e[0]||e.source||e.u,bid=e[1]||e.target||e.v,a=null,b=null;for(var j=0;j<pts.length;j++){var p=pts[j];if(String(p.n.id)===String(aid))a=p;if(String(p.n.id)===String(bid))b=p}if(a&&b){ctx.beginPath();ctx.moveTo(a.x,a.y);ctx.lineTo(b.x,b.y);ctx.stroke()}}for(var k=0;k<pts.length;k++){var q=pts[k];ctx.fillStyle=mode==='selected'?'#0284c7':'#16a34a';ctx.beginPath();ctx.arc(q.x,q.y,6*devicePixelRatio,0,Math.PI*2);ctx.fill();ctx.strokeStyle='#f59e0b';ctx.lineWidth=2;ctx.fillStyle='#f59e0b';arrow(q.x,q.y,q.n.yaw||0,ARR*devicePixelRatio);ctx.fillStyle='#0f172a';ctx.strokeStyle='#ffffff';ctx.lineWidth=3;ctx.strokeText(String(q.n.id),q.x+8*devicePixelRatio,q.y-8*devicePixelRatio);ctx.fillText(String(q.n.id),q.x+8*devicePixelRatio,q.y-8*devicePixelRatio)}}
c.addEventListener('click',function(ev){var r=c.getBoundingClientRect(),x=(ev.clientX-r.left)*devicePixelRatio,y=(ev.clientY-r.top)*devicePixelRatio,best=null,bd=1e9;for(var i=0;i<pts.length;i++){var p=pts[i],d=Math.hypot(p.x-x,p.y-y);if(d<bd){best=p;bd=d}}if(best&&bd<20*devicePixelRatio){showInfo(best.n)}});
function showInfo(n){var yd=n.yaw!=null?(n.yaw*180/Math.PI).toFixed(0):'?';var h=document.createElement('h3');h.textContent='kf_'+n.id;var pm=document.createElement('p');pm.className='muted';pm.textContent='x='+n.x.toFixed(3)+' y='+n.y.toFixed(3)+' yaw='+yd+'°';info.innerHTML='';info.appendChild(h);info.appendChild(pm);if(n.image){var img=document.createElement('img');img.src='/api/file?path='+encodeURIComponent(n.image);info.appendChild(img)}var ps=document.createElement('p');ps.textContent=n.semantic||'无语义描述';info.appendChild(ps);var link=document.createElement('p');link.innerHTML=fileLink(n.data_path,'查看关键帧 JSON');info.appendChild(link)}
function renderList(){list.innerHTML='';nodes().forEach(function(n){var card=document.createElement('div');card.className='card';var html='<h3>kf_'+n.id+'</h3>';if(n.image)html+='<img src="/api/file?path='+encodeURIComponent(n.image)+'">';html+='<p>'+(n.semantic||'无语义描述')+'</p><p>'+fileLink(n.data_path,'查看关键帧 JSON')+'</p>';card.innerHTML=html;list.appendChild(card)})}
document.getElementById('map-btn').classList.add('active');
if(data.map&&data.map.image){mapImg=new Image();mapImg.onload=function(){resize()};mapImg.onerror=function(){mapImg=null;resize()};mapImg.src='/api/file?path='+encodeURIComponent(data.map.image)}
setMode(mode);
})();
</script></body></html>"""


def main(args: object = None) -> None:
    rclpy.init(args=args)
    node = DashboardNode()
    try:
        rclpy.spin(node)
    finally:
        for proc in list(node._processes.values()):
            if proc.is_running():
                proc.stop(timeout=3)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
