"""CarAgent unified dashboard for launch/test/calibration workflows."""

from __future__ import annotations

import glob as glob_mod
import json
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
CLIP_MODEL = MODELS_DIR / "clip-vit-base-patch32" / "image_encoder.xml"
DINO_MODEL = MODELS_DIR / "dinov2"
DEFAULT_STEREO_CALIB = CALIB_DIR / "stereo_current" / "stereo_calibration.npz"
DEFAULT_EXTRINSICS = CALIB_DIR / "lidar_camera" / "lidar_camera_extrinsics_calibrated.json"
DEFAULT_CAMERA_RESOLUTION = "3840x1200"
CAMERA_RESOLUTIONS = {
    "3840x1200": (3840, 1200, 1920, 1920, 30.0),
    "3840x1080": (3840, 1080, 1920, 1920, 30.0),
    "2560x720": (2560, 720, 1280, 1280, 30.0),
    "1280x480": (1280, 480, 640, 640, 30.0),
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
                "candidate_frames": len(list((session / "left").glob("*.png"))) + len(list((session / "left").glob("*.jpg"))),
                "selected_frames": len(list((selected / "left").glob("*.png"))) + len(list((selected / "left").glob("*.jpg"))) if selected.exists() else 0,
                "keyframe_nodes": len(nodes),
                "semantic_nodes": semantics,
                "review_html": str(selected / "review.html") if (selected / "review.html").exists() else "",
                "scene_html": str(selected / "scene_map.html") if (selected / "scene_map.html").exists() else "",
                "mtime": session.stat().st_mtime,
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


def _json_response(handler: BaseHTTPRequestHandler, data: object, status: int = 200) -> None:
    body = json.dumps(data, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(body)


def _shell_join(args: list[str]) -> str:
    return " ".join(shlex.quote(str(arg)) for arg in args)


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
            env = os.environ.copy()
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
        content_type = "application/octet-stream"
        if path.suffix.lower() in {".html", ".htm"}:
            content_type = "text/html; charset=utf-8"
        elif path.suffix.lower() in {".json", ".jsonl"}:
            content_type = "application/json; charset=utf-8"
        elif path.suffix.lower() == ".csv":
            content_type = "text/csv; charset=utf-8"
        elif path.suffix.lower() in {".png"}:
            content_type = "image/png"
        elif path.suffix.lower() in {".jpg", ".jpeg"}:
            content_type = "image/jpeg"
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)
        if path in {"/", "/index.html"}:
            static_path = Path(__file__).resolve().parent / "static" / "dashboard.html"
            if not static_path.exists():
                static_path = WORKSPACE / "install" / "caragent_ui" / "share" / "caragent_ui" / "static" / "dashboard.html"
            return self._serve_file(static_path)
        if path == "/api/status":
            return _json_response(self, self.node.get_status())
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
            "/api/live_config": self.node.api_live_config,
            "/api/object_depth_dataset/config": self.node.api_object_depth_dataset_config,
            "/api/slam/initial_pose": self.node.api_slam_initial_pose,
            "/api/slam/clear_changes": self.node.api_slam_clear_changes,
            "/api/slam/auto_clear": self.node.api_slam_auto_clear,
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
        self._auto_clear_enabled: bool = False
        self._auto_clear_interval: float = 8.0
        from geometry_msgs.msg import PoseWithCovarianceStamped
        self._initialpose_sub = self.create_subscription(
            PoseWithCovarianceStamped, "/initialpose", self._on_initialpose, 1)
        threading.Thread(target=self._run_http, daemon=True).start()
        self.add_global_log(f"CarAgent Dashboard ready: http://0.0.0.0:{self._port}")

    def _run_http(self) -> None:
        server = ThreadingHTTPServer(("0.0.0.0", self._port), _DashboardHandler)
        server.dashboard_node = self
        server.serve_forever()

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
        subprocess.run(cmd, shell=True, executable="/bin/bash", cwd=str(WORKSPACE), text=True, capture_output=True, timeout=15)

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

    def get_status(self) -> dict[str, Any]:
        maps = _map_entries()
        sessions = _keyframe_sessions()
        return {
            "workspace": str(WORKSPACE),
            "dashboard_port": self._port,
            "agent_web_port": AGENT_WEB_PORT,
            "ssh_tunnel": f"ssh -L {self._port}:localhost:{self._port} -L {AGENT_WEB_PORT}:localhost:{AGENT_WEB_PORT} car@10.181.156.54",
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
            max_angular = "1.25" if bool(body.get("enable_left_only_goal_proxy")) else "1.00"
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
            args = [
                "ros2", "launch", "caragent_memory", "caragent_keyframe_collect.launch.py",
                f"session_name:={session}", f"laser_port:={laser}", f"stm32_port:={stm32}",
                f"camera_device:={camera}", f"map_file_name:={map_base}", "use_rviz:=true",
                "camera_show_image:=true",
            ]
            args.extend(_camera_launch_args(body))
            return _shell_join(args)
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
        proc = subprocess.run(cmd, shell=True, executable="/bin/bash", cwd=str(WORKSPACE), text=True, capture_output=True, timeout=15)
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
            proc = subprocess.run(cmd, shell=True, executable="/bin/bash", cwd=str(WORKSPACE),
                                  text=True, capture_output=True, timeout=10)
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
        proc = subprocess.run(cmd, shell=True, executable="/bin/bash", cwd=str(WORKSPACE),
                              text=True, capture_output=True, timeout=10)
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
        first = subprocess.run(serialize_cmd, shell=True, executable="/bin/bash", cwd=str(WORKSPACE), text=True, capture_output=True, timeout=30)
        self.add_global_log((first.stdout + first.stderr).strip() or f"serialize_map returncode={first.returncode}")
        map_saver_cmd = self._ros_command(_shell_join(["ros2", "run", "nav2_map_server", "map_saver_cli", "-f", str(base)]))
        second = subprocess.run(map_saver_cmd, shell=True, executable="/bin/bash", cwd=str(WORKSPACE), text=True, capture_output=True, timeout=30)
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
        command = self._ros_command(
            _shell_join([
                "ros2", "run", "caragent_memory", "select_keyframes",
                "--dataset", str(dataset), "--clip-model", str(CLIP_MODEL),
                "--dinov2-model", str(DINO_MODEL), "--device", "GPU", "--dinov2-device", "auto",
            ])
        )
        return self._start_process("keyframe_select", command)

    def api_visualize_keyframes(self, body: dict[str, Any]) -> dict[str, Any]:
        dataset = _expand_path(body.get("dataset"), KEYFRAMES_DIR / "")
        selected = dataset if dataset.name == "selected" else dataset / "selected"
        if not selected.exists():
            return {"ok": False, "error": f"Selected dataset not found: {selected}"}
        output = selected / "scene_map.html"
        self._write_scene_map_html(selected, output)
        return {"ok": True, "path": str(output)}

    def api_annotate_keyframes(self, body: dict[str, Any]) -> dict[str, Any]:
        dataset = _expand_path(body.get("dataset"), KEYFRAMES_DIR / "")
        if dataset.name != "selected":
            dataset = dataset / "selected"
        batch_size = int(body.get("batch_size") or 5)
        api_key = str(body.get("api_key") or "").strip()
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
        if api_key:
            command = self._ros_command(_shell_join(args))
            env_overrides["DASHSCOPE_API_KEY"] = api_key
            self.add_global_log("annotate_keyframes: using provided API key")
        else:
            command = self._ros_command(_shell_join(args))
        return self._start_process("keyframe_annotate", command, env_overrides=env_overrides)

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

    def _write_scene_map_html(self, selected: Path, output: Path) -> None:
        import math as _m
        node_dir = selected / "constructed_memory" / "keyframe_nodes"
        nodes = []
        for path in sorted(node_dir.glob("kf_*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            pos = data.get("position") or [0, 0, 0]
            image_path = str(data.get("rgb_path") or "")
            if image_path and not Path(image_path).is_absolute():
                image_path = str((selected / image_path).resolve())
            # Extract yaw from orientation quaternion [x, y, z, w]
            orient = data.get("orientation") or [0, 0, 0, 1]
            qx, qy, qz, qw = float(orient[0]), float(orient[1]), float(orient[2]), float(orient[3])
            yaw = _m.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
            nodes.append(
                {
                    "id": data.get("kf_id"),
                    "name": data.get("name") or path.stem,
                    "x": float(pos[0]) if len(pos) > 0 else 0.0,
                    "y": float(pos[1]) if len(pos) > 1 else 0.0,
                    "yaw": yaw,
                    "image": image_path,
                    "semantic": data.get("semantic") or "",
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
        html_text = SCENE_MAP_TEMPLATE.replace("__DATA__", json.dumps({"nodes": nodes, "edges": edges}, ensure_ascii=False))
        output.write_text(html_text, encoding="utf-8")


SCENE_MAP_TEMPLATE = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>CarAgent Scene Map</title>
<style>body{font-family:Arial,sans-serif;margin:0;background:#101827;color:#e5e7eb}main{display:grid;grid-template-columns:1fr 360px;height:100vh}canvas{width:100%;height:100%;background:#0b1020}.side{border-left:1px solid #334155;padding:16px;overflow:auto}.muted{color:#94a3b8}img{max-width:100%;border:1px solid #334155}.card{background:#172033;border:1px solid #334155;border-radius:8px;padding:12px}</style>
</head><body><main><canvas id="map"></canvas><aside class="side"><h2>Scene Map</h2><div id="info" class="card muted">点击关键帧点查看图像和语义。</div></aside></main>
<script>
(function(){var data=__DATA__,c=document.getElementById('map'),ctx=c.getContext('2d'),info=document.getElementById('info'),ARR=18,pts=[];
function resize(){c.width=c.clientWidth*devicePixelRatio;c.height=c.clientHeight*devicePixelRatio;draw()}
window.addEventListener('resize',resize);
function arrow(cx,cy,yaw,len){var ex=cx+Math.cos(yaw)*len,ey=cy-Math.sin(yaw)*len;ctx.beginPath();ctx.moveTo(cx,cy);ctx.lineTo(ex,ey);ctx.stroke();var h=len*0.4;ctx.beginPath();ctx.moveTo(ex,ey);ctx.lineTo(ex-Math.cos(yaw+0.6)*h,ey+Math.sin(yaw+0.6)*h);ctx.lineTo(ex-Math.cos(yaw-0.6)*h,ey+Math.sin(yaw-0.6)*h);ctx.closePath();ctx.fill()}
function draw(){ctx.clearRect(0,0,c.width,c.height);var ns=data.nodes;if(!ns.length)return;var xs=ns.map(function(n){return n.x}),ys=ns.map(function(n){return n.y}),minx=Math.min.apply(null,xs),maxx=Math.max.apply(null,xs),miny=Math.min.apply(null,ys),maxy=Math.max.apply(null,ys),pad=60*devicePixelRatio,sx=(c.width-2*pad)/Math.max(1,maxx-minx),sy=(c.height-2*pad)/Math.max(1,maxy-miny),s=Math.min(sx,sy);pts=ns.map(function(n){return {n:n,x:pad+(n.x-minx)*s,y:c.height-pad-(n.y-miny)*s}});ctx.strokeStyle='#334155';ctx.lineWidth=1;var edges=data.edges||[];for(var i=0;i<edges.length;i++){var e=edges[i],aid=e[0]||e.source||e.u,bid=e[1]||e.target||e.v,a=null,b=null;for(var j=0;j<pts.length;j++){var p=pts[j];if(String(p.n.id)===String(aid))a=p;if(String(p.n.id)===String(bid))b=p}if(a&&b){ctx.beginPath();ctx.moveTo(a.x,a.y);ctx.lineTo(b.x,b.y);ctx.stroke()}}for(var k=0;k<pts.length;k++){var q=pts[k];ctx.fillStyle='#38bdf8';ctx.beginPath();ctx.arc(q.x,q.y,6*devicePixelRatio,0,Math.PI*2);ctx.fill();ctx.strokeStyle='#f59e0b';ctx.lineWidth=2;ctx.fillStyle='#f59e0b';arrow(q.x,q.y,q.n.yaw||0,ARR*devicePixelRatio);ctx.fillStyle='#e5e7eb';ctx.fillText(String(q.n.id),q.x+8*devicePixelRatio,q.y-8*devicePixelRatio)}}
c.addEventListener('click',function(ev){var r=c.getBoundingClientRect(),x=(ev.clientX-r.left)*devicePixelRatio,y=(ev.clientY-r.top)*devicePixelRatio,best=null,bd=1e9;for(var i=0;i<pts.length;i++){var p=pts[i],d=Math.hypot(p.x-x,p.y-y);if(d<bd){best=p;bd=d}}if(best&&bd<20*devicePixelRatio){showInfo(best.n)}});
function showInfo(n){var yd=n.yaw!=null?(n.yaw*180/Math.PI).toFixed(0):'?';var h=document.createElement('h3');h.textContent='kf_'+n.id;var pm=document.createElement('p');pm.className='muted';pm.textContent='x='+n.x.toFixed(3)+' y='+n.y.toFixed(3)+' yaw='+yd+'°';info.innerHTML='';info.appendChild(h);info.appendChild(pm);if(n.image){var img=document.createElement('img');img.src='/api/file?path='+encodeURIComponent(n.image);info.appendChild(img)}var ps=document.createElement('p');ps.textContent=n.semantic||'无语义描述';info.appendChild(ps)}
resize();
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
