"""Live validation UI for monocular depth plus projected LaserScan fitting.

The script subscribes to the left camera image and LaserScan, shows a live
OpenCV UI, captures a synchronized-ish snapshot on demand, runs the existing
GroundingDINO, EfficientSAM, Depth Anything, and scan-depth fit scripts, then
logs one row per experiment.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image as PILImage, ImageDraw
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan

from caragent_agent.perception.grounding.grounding_dino_openvino import GroundingDINOOpenVINO
from caragent_agent.perception.grounding.run_grounding_dino_openvino import draw_detections
from caragent_agent.perception.sam.efficient_sam_openvino import EfficientSAMOpenVINO
from caragent_agent.perception.sam.run_efficientsam_openvino import choose_detection, save_overlay
from caragent_agent.perception.depth.run_depth_anything_openvino import DepthAnythingOpenVINO, save_depth_outputs
from caragent_agent.perception.fusion.stereo_mono_anchor_fusion import (
    compute_stereo_mono_guard,
    stereo_base_depth_summary,
    write_guard_payload,
)


DEFAULT_WORKSPACE = Path(os.environ.get("CARAGENT_WS", Path.home() / "caragent_ws"))
DEFAULT_OUTPUT_DIR = DEFAULT_WORKSPACE / "perception_outputs" / "scan_monodepth_validation"
DEFAULT_CALIB = DEFAULT_WORKSPACE / "calibration" / "stereo_current" / "stereo_calibration.npz"
DEFAULT_EXTR = DEFAULT_WORKSPACE / "calibration" / "lidar_camera" / "lidar_camera_extrinsics_calibrated.json"
DEFAULT_GROUNDING_MODEL_DIR = DEFAULT_WORKSPACE / "models" / "grounding_dino_openvino"
DEFAULT_GROUNDING_MODEL_ID = DEFAULT_WORKSPACE / "models" / "grounding-dino-tiny"
DEFAULT_DEPTH_MODEL_DIR = DEFAULT_WORKSPACE / "models" / "depth_anything_v2_openvino"
DEFAULT_ABSOLUTE_DEPTH_MODEL_DIR = DEFAULT_WORKSPACE / "models" / "depth_anything_v2_metric_indoor_small_openvino"
DEFAULT_SAM_ENCODER_XML = DEFAULT_WORKSPACE / "models" / "efficient_sam_openvino" / "efficient_sam_vitt_encoder.xml"
DEFAULT_SAM_DECODER_XML = DEFAULT_WORKSPACE / "models" / "efficient_sam_openvino" / "efficient_sam_vitt_decoder.xml"
LOCALIZATION_MODE_CHOICES = ("stereo", "stereo_primary_mono_guard", "mono_relative_lidar", "mono_absolute")
LOCALIZATION_MODE_LABELS = {
    "stereo": "stereo",
    "stereo_primary_mono_guard": "stereo primary + mono guard",
    "mono_relative_lidar": "mono relative + lidar",
    "mono_absolute": "mono absolute",
}


@dataclass
class LaserPose:
    x_m: float = 0.12
    y_m: float = 0.0
    yaw_rad: float = math.pi


@dataclass
class LatestMessages:
    image_bgr: np.ndarray | None = None
    image_recv_time: float = 0.0
    image_stamp_sec: float = 0.0
    image_encoding: str = ""
    right_image_bgr: np.ndarray | None = None
    right_image_recv_time: float = 0.0
    right_image_stamp_sec: float = 0.0
    right_image_encoding: str = ""
    scan_msg: LaserScan | None = None
    scan_recv_time: float = 0.0
    scan_stamp_sec: float = 0.0

    def copy_snapshot(self) -> tuple[np.ndarray | None, np.ndarray | None, LaserScan | None, dict[str, float | str]]:
        image = None if self.image_bgr is None else self.image_bgr.copy()
        right_image = None if self.right_image_bgr is None else self.right_image_bgr.copy()
        meta = {
            "image_recv_time": float(self.image_recv_time),
            "image_stamp_sec": float(self.image_stamp_sec),
            "right_image_recv_time": float(self.right_image_recv_time),
            "right_image_stamp_sec": float(self.right_image_stamp_sec),
            "scan_recv_time": float(self.scan_recv_time),
            "scan_stamp_sec": float(self.scan_stamp_sec),
            "image_encoding": self.image_encoding,
            "right_image_encoding": self.right_image_encoding,
        }
        return image, right_image, self.scan_msg, meta


class LiveCaptureNode(Node):
    def __init__(
        self,
        image_topic: str,
        right_image_topic: str,
        scan_topic: str,
        latest: LatestMessages,
        lock: threading.Lock,
    ) -> None:
        super().__init__("live_scan_monodepth_validation")
        self.latest = latest
        self.lock = lock
        self.create_subscription(Image, image_topic, self._on_image, 1)
        self.create_subscription(Image, right_image_topic, self._on_right_image, 1)
        self.create_subscription(LaserScan, scan_topic, self._on_scan, 1)
        self.get_logger().info(f"subscribed left={image_topic} right={right_image_topic} scan={scan_topic}")

    def _on_image(self, msg: Image) -> None:
        try:
            image_bgr = image_msg_to_bgr(msg)
        except Exception as exc:
            self.get_logger().warning(f"failed to decode image: {exc}")
            return
        with self.lock:
            self.latest.image_bgr = image_bgr
            self.latest.image_recv_time = time.monotonic()
            self.latest.image_stamp_sec = stamp_to_sec(msg)
            self.latest.image_encoding = msg.encoding

    def _on_right_image(self, msg: Image) -> None:
        try:
            image_bgr = image_msg_to_bgr(msg)
        except Exception as exc:
            self.get_logger().warning(f"failed to decode right image: {exc}")
            return
        with self.lock:
            self.latest.right_image_bgr = image_bgr
            self.latest.right_image_recv_time = time.monotonic()
            self.latest.right_image_stamp_sec = stamp_to_sec(msg)
            self.latest.right_image_encoding = msg.encoding

    def _on_scan(self, msg: LaserScan) -> None:
        with self.lock:
            self.latest.scan_msg = msg
            self.latest.scan_recv_time = time.monotonic()
            self.latest.scan_stamp_sec = stamp_to_sec(msg)


def stamp_to_sec(msg: Any) -> float:
    stamp = getattr(getattr(msg, "header", None), "stamp", None)
    if stamp is None:
        return 0.0
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def image_msg_to_bgr(msg: Image) -> np.ndarray:
    encoding = (msg.encoding or "").lower()
    height = int(msg.height)
    width = int(msg.width)
    step = int(msg.step)

    if encoding in {"bgr8", "rgb8", "rgba8", "bgra8"}:
        channels = 4 if encoding in {"rgba8", "bgra8"} else 3
        raw = np.frombuffer(msg.data, dtype=np.uint8).reshape(height, step)
        arr = raw[:, : width * channels].reshape(height, width, channels).copy()
        if encoding == "bgr8":
            return arr
        if encoding == "rgb8":
            return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        if encoding == "rgba8":
            return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
        return cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)

    if encoding in {"mono8", "8uc1"}:
        raw = np.frombuffer(msg.data, dtype=np.uint8).reshape(height, step)
        gray = raw[:, :width].copy()
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    if encoding in {"mono16", "16uc1"}:
        row_words = step // 2
        raw = np.frombuffer(msg.data, dtype=np.uint16).reshape(height, row_words)
        gray16 = raw[:, :width].copy()
        gray8 = cv2.convertScaleAbs(gray16, alpha=255.0 / max(1.0, float(gray16.max())))
        return cv2.cvtColor(gray8, cv2.COLOR_GRAY2BGR)

    raise ValueError(f"unsupported image encoding: {msg.encoding}")


def load_laser_pose(extrinsics_json: Path) -> LaserPose:
    if not extrinsics_json.exists():
        return LaserPose()
    data = json.loads(extrinsics_json.read_text(encoding="utf-8"))
    params = data.get("optimized_params", data)
    return LaserPose(
        x_m=float(params.get("laser_x_m", LaserPose.x_m)),
        y_m=float(params.get("laser_y_m", LaserPose.y_m)),
        yaw_rad=float(params.get("laser_yaw_rad", LaserPose.yaw_rad)),
    )


def scan_to_base_xy(scan_msg: LaserScan, pose: LaserPose) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ranges = np.asarray(scan_msg.ranges, dtype=np.float32)
    angles = float(scan_msg.angle_min) + np.arange(len(ranges), dtype=np.float32) * float(scan_msg.angle_increment)
    valid = (
        np.isfinite(ranges)
        & (ranges >= float(scan_msg.range_min))
        & (ranges <= float(scan_msg.range_max))
    )
    x_l = ranges * np.cos(angles)
    y_l = ranges * np.sin(angles)
    c = math.cos(pose.yaw_rad)
    s = math.sin(pose.yaw_rad)
    x_b = pose.x_m + c * x_l - s * y_l
    y_b = pose.y_m + s * x_l + c * y_l
    return x_b.astype(np.float32), y_b.astype(np.float32), valid


def save_scan_npz(scan_msg: LaserScan, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        ranges=np.asarray(scan_msg.ranges, dtype=np.float32),
        angle_min=np.float32(scan_msg.angle_min),
        angle_max=np.float32(scan_msg.angle_max),
        angle_increment=np.float32(scan_msg.angle_increment),
        range_min=np.float32(scan_msg.range_min),
        range_max=np.float32(scan_msg.range_max),
        time_increment=np.float32(scan_msg.time_increment),
        scan_time=np.float32(scan_msg.scan_time),
    )


def format_grounding_prompt(text: str) -> str:
    normalized = text.strip()
    if not normalized:
        return ""
    terms = [term.strip() for term in normalized.replace(";", ".").split(".") if term.strip()]
    if not terms:
        terms = [normalized]
    return " . ".join(terms) + " ."


def resize_panel(image: np.ndarray, width: int, height: int) -> np.ndarray:
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def draw_text_panel(lines: list[str], width: int, height: int, bg: tuple[int, int, int] = (245, 245, 240)) -> np.ndarray:
    panel = np.full((height, width, 3), bg, dtype=np.uint8)
    y = 28
    for idx, line in enumerate(lines):
        scale = 0.58 if idx else 0.70
        color = (20, 20, 20) if idx else (0, 80, 160)
        cv2.putText(panel, line[:95], (14, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
        y += 26
        if y > height - 16:
            break
    return panel


def draw_camera_panel(image_bgr: np.ndarray | None, width: int, height: int, lines: list[str]) -> np.ndarray:
    if image_bgr is None:
        panel = draw_text_panel(["Waiting for camera image..."] + lines, width, height)
    else:
        panel = resize_panel(image_bgr, width, height)
        bar_h = 22 + 22 * len(lines[:1])
        cv2.rectangle(panel, (0, 0), (width, bar_h), (0, 0, 0), -1)
        y = 18
        for line in lines[:1]:
            cv2.putText(panel, line[:80], (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 200), 1, cv2.LINE_AA)
            y += 22
    return panel


_scan_bg_cache: tuple[int, int, np.ndarray] | None = None


def _build_scan_bg(width: int, height: int, origin: np.ndarray, scale: float) -> np.ndarray:
    panel = np.full((height, width, 3), 248, dtype=np.uint8)
    for x in np.arange(-0.5, 6.6, 0.5):
        yy = int(origin[1] - x * scale)
        cv2.line(panel, (0, yy), (width, yy), (222, 222, 218), 1)
        if 0 <= yy < height:
            cv2.putText(panel, f"x={x:.1f}", (6, yy - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (105, 105, 100), 1)
    for y in np.arange(-3.0, 3.1, 0.5):
        xx = int(origin[0] - y * scale)
        cv2.line(panel, (xx, 0), (xx, height), (222, 222, 218), 1)
    base = (int(origin[0]), int(origin[1]))
    cv2.circle(panel, base, 6, (20, 20, 20), -1)
    cv2.arrowedLine(panel, base, (base[0], max(0, base[1] - int(0.65 * scale))), (20, 20, 20), 2)
    cv2.putText(panel, "base_link top-down (+X forward)", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1)
    return panel


def draw_scan_panel(scan_msg: LaserScan | None, pose: LaserPose, width: int, height: int) -> np.ndarray:
    global _scan_bg_cache
    origin = np.array([width * 0.50, height * 0.84], dtype=np.float32)
    scale = min((height - 70) / 6.5, (width - 40) / 6.8)

    if _scan_bg_cache is None or _scan_bg_cache[:2] != (width, height):
        _scan_bg_cache = (width, height, _build_scan_bg(width, height, origin, scale))
    panel = _scan_bg_cache[2].copy()

    def to_px(x_m: np.ndarray, y_m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return origin[0] - y_m * scale, origin[1] - x_m * scale

    if scan_msg is None:
        cv2.putText(panel, "Waiting for /scan...", (12, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 180), 1)
        return panel

    x_b, y_b, valid = scan_to_base_xy(scan_msg, pose)
    ranges = np.asarray(scan_msg.ranges, dtype=np.float32)
    px, py = to_px(x_b[valid], y_b[valid])
    valid_ranges = ranges[valid]
    if len(valid_ranges):
        lo, hi = np.percentile(valid_ranges, [5, 95])
    else:
        lo, hi = 0.0, 1.0
    span = max(1e-6, float(hi - lo))
    for x, y, r in zip(px.astype(int), py.astype(int), valid_ranges):
        if 0 <= x < width and 0 <= y < height:
            t = float(np.clip((r - lo) / span, 0.0, 1.0))
            color = (int(255 * t), int(180 * (1.0 - t)), int(255 * (1.0 - t)))
            cv2.circle(panel, (x, y), 2, color, -1)
    cv2.putText(
        panel,
        f"valid beams: {int(np.count_nonzero(valid))}  min range: {float(np.nanmin(valid_ranges)):.2f}m" if len(valid_ranges) else "no valid beams",
        (12, 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (20, 20, 20),
        1,
    )
    return panel


def safe_read_image(path: str | Path | None, width: int, height: int, fallback_lines: list[str]) -> np.ndarray:
    if path:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is not None:
            return resize_panel(image, width, height)
    return draw_text_panel(fallback_lines, width, height)


def append_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    fieldnames = [
        "sample_id",
        "timestamp",
        "target_text",
        "label_query",
        "truth_distance_m",
        "localization_mode",
        "recommended_depth_m",
        "recommended_error_m",
        "mono_recommended_depth_m",
        "mono_error_m",
        "mono_p05_m",
        "mono_p10_m",
        "mono_median_m",
        "mono_p90_m",
        "stereo_recommended_depth_m",
        "stereo_error_m",
        "stereo_p05_m",
        "stereo_p10_m",
        "stereo_median_m",
        "stereo_p90_m",
        "stereo_base_median_m",
        "stereo_base_range_xy_m",
        "stereo_recommended_source",
        "stereo_valid_ratio",
        "stereo_valid_pixels",
        "stereo_status",
        "mono_guard_recommended_depth_m",
        "mono_guard_error_m",
        "mono_guard_selected_source",
        "mono_guard_reason",
        "mono_guard_fused_median_m",
        "mono_guard_delta_m",
        "mono_guard_anchor_count",
        "mono_guard_fit_rmse_m",
        "mono_guard_status",
        "absolute_recommended_depth_m",
        "absolute_error_m",
        "absolute_p05_m",
        "absolute_p10_m",
        "absolute_median_m",
        "absolute_p90_m",
        "absolute_valid_ratio",
        "absolute_status",
        "selected_fit",
        "fit_mae_m",
        "fit_p90_m",
        "projected_samples",
        "used_for_fit_samples",
        "edge_rejected_samples",
        "mask_lidar_support_enabled",
        "mask_lidar_has_support",
        "mask_lidar_points",
        "mask_lidar_required_points",
        "mask_lidar_points_per_width",
        "mask_area_px",
        "detection_label",
        "detection_score",
        "status",
        "sample_dir",
    ]
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def load_mask_from_segmentation(segmentation_json: Path, image_size: tuple[int, int]) -> tuple[np.ndarray, dict[str, Any]]:
    seg = json.loads(segmentation_json.read_text(encoding="utf-8"))
    mask_path = Path(seg["mask_path"]).resolve()
    mask = np.asarray(PILImage.open(mask_path).convert("L"))
    width, height = image_size
    if mask.shape != (height, width):
        resampling = getattr(PILImage, "Resampling", PILImage)
        mask = np.asarray(PILImage.fromarray(mask).resize((width, height), resampling.NEAREST))
    return mask > 0, seg


def robust_depth_stats(depth_m: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    values = depth_m[mask]
    values = values[np.isfinite(values) & (values > 0)]
    if len(values) == 0:
        return {}
    return {
        "count": int(len(values)),
        "min": float(np.min(values)),
        "p05": float(np.percentile(values, 5)),
        "p10": float(np.percentile(values, 10)),
        "p25": float(np.percentile(values, 25)),
        "median": float(np.median(values)),
        "p75": float(np.percentile(values, 75)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
    }


def metric_object_depth_payload(
    depth_m: np.ndarray,
    segmentation_json: Path,
    image_size: tuple[int, int],
    model_dir: Path,
    device: str,
    elapsed_ms: float,
    output_dir: Path,
    sample_id: str,
) -> dict[str, Any]:
    mask, seg = load_mask_from_segmentation(segmentation_json, image_size)
    stats = robust_depth_stats(depth_m.astype(np.float32), mask)
    depth_values = depth_m[mask]
    valid_values = depth_values[np.isfinite(depth_values) & (depth_values > 0)]
    payload = {
        "backend": "depth_anything_v2_metric_openvino",
        "mode": "mono_absolute",
        "model_dir": str(model_dir),
        "device": device,
        "elapsed_ms": float(elapsed_ms),
        "segmentation_json": str(segmentation_json),
        "source_detection": seg.get("source_detection"),
        "image_size": [int(image_size[0]), int(image_size[1])],
        "mask": {
            "area_px": int(np.count_nonzero(mask)),
            "valid_depth_pixels": int(len(valid_values)),
            "valid_ratio": float(len(valid_values) / max(1, np.count_nonzero(mask))),
        },
        "object_mask_metric_depth_m": stats,
    }
    json_path = output_dir / f"{sample_id}_mono_absolute_object_depth.json"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    payload["json_path"] = str(json_path)
    return payload


class PipelineRunner:
    def __init__(self, args: argparse.Namespace, status_cb) -> None:
        self.args = args
        self.status_cb = status_cb
        self.output_dir = args.output_dir.resolve()
        self.samples_dir = self.output_dir / "samples"
        self.command_logs_dir = self.output_dir / "command_logs"
        self.fit_output_dir = self.output_dir / "scan_monodepth_fit"
        self.stereo_output_dir = self.output_dir / "stereo_object_depth"
        self.absolute_output_dir = self.output_dir / "mono_absolute_depth"
        self.summary_csv = self.output_dir / "validation_results.csv"
        self.summary_jsonl = self.output_dir / "validation_results.jsonl"
        for path in [self.samples_dir, self.command_logs_dir, self.fit_output_dir, self.stereo_output_dir, self.absolute_output_dir]:
            path.mkdir(parents=True, exist_ok=True)

        self.grounding_model = None
        if not bool(getattr(args, "skip_grounding_model", False)):
            self.set_status("Loading GroundingDINO OpenVINO model...")
            self.grounding_model = GroundingDINOOpenVINO(
                model_dir=args.grounding_model_dir,
                model_id=str(args.grounding_model_id),
                device=args.grounding_device,
            )
        self.set_status("Loading EfficientSAM OpenVINO model...")
        self.sam_model = EfficientSAMOpenVINO(
            encoder_xml=args.sam_encoder_xml,
            decoder_xml=args.sam_decoder_xml,
            device=args.sam_device,
            encoder_device=args.sam_encoder_device or args.sam_device,
            decoder_device=args.sam_decoder_device or "CPU",
        )
        self.depth_model: DepthAnythingOpenVINO | None = None
        self.absolute_depth_model = None
        self.set_status("All models loaded.")

    def set_status(self, text: str) -> None:
        self.status_cb(text)
        print(text, flush=True)

    def get_relative_depth_model(self) -> DepthAnythingOpenVINO:
        if self.depth_model is None:
            self.set_status("Loading relative Depth Anything OpenVINO model...")
            self.depth_model = DepthAnythingOpenVINO(
                model_dir=self.args.depth_model_dir,
                device=self.args.depth_device,
            )
        return self.depth_model

    def get_absolute_depth_model(self) -> DepthAnythingOpenVINO:
        if not self.args.absolute_depth_model_dir.exists():
            raise RuntimeError(f"Metric depth model not available: {self.args.absolute_depth_model_dir}")
        if self.absolute_depth_model is None:
            self.set_status("Loading metric Depth Anything OpenVINO model...")
            self.absolute_depth_model = DepthAnythingOpenVINO(
                model_dir=self.args.absolute_depth_model_dir,
                device=self.args.absolute_depth_device,
            )
        return self.absolute_depth_model

    def run_sample(
        self,
        image_bgr: np.ndarray,
        right_image_bgr: np.ndarray | None,
        scan_msg: LaserScan | None,
        snapshot_meta: dict[str, float | str],
        target_text: str,
        label_query: str,
        truth_distance_m: float | None,
        localization_mode: str,
    ) -> dict[str, Any]:
        if localization_mode not in LOCALIZATION_MODE_CHOICES:
            raise ValueError(f"Unsupported localization mode: {localization_mode}")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sample_id = f"live_{timestamp}"
        image_path = self.samples_dir / f"{sample_id}.png"
        right_image_path = self.samples_dir / f"{sample_id}_right.png"
        scan_path = self.samples_dir / f"{sample_id}_scan.npz"
        cv2.imwrite(str(image_path), image_bgr)
        if right_image_bgr is not None:
            cv2.imwrite(str(right_image_path), right_image_bgr)
        if scan_msg is not None:
            save_scan_npz(scan_msg, scan_path)

        record: dict[str, Any] = {
            "sample_id": sample_id,
            "timestamp": timestamp,
            "target_text": target_text,
            "label_query": label_query,
            "truth_distance_m": truth_distance_m,
            "localization_mode": localization_mode,
            "status": "running",
            "sample_dir": str(self.samples_dir),
            "image": str(image_path),
            "right_image": str(right_image_path) if right_image_bgr is not None else None,
            "scan": str(scan_path) if scan_msg is not None else None,
            "snapshot": snapshot_meta,
        }

        try:
            prompt = format_grounding_prompt(target_text)
            if not prompt:
                raise ValueError("target text is empty")

            self._run_grounding(sample_id, image_path, prompt)
            grounding_json = self.output_dir / f"{sample_id}_grounding_openvino.json"
            grounding_payload = json.loads(grounding_json.read_text(encoding="utf-8"))
            self._run_sam(sample_id, grounding_json, label_query or target_text)
            segmentation_json = self._segmentation_json_path(sample_id)
            seg_payload = json.loads(segmentation_json.read_text(encoding="utf-8"))
            fit_payload: dict[str, Any] = {}
            absolute_payload: dict[str, Any] = {}
            stereo_payload = None
            stereo_status = "skipped"
            absolute_status = "skipped"
            stereo_error_text = None
            mono_guard_payload: dict[str, Any] = {}
            mono_guard_status = "skipped"
            mono_guard_json: Path | None = None

            if localization_mode == "mono_relative_lidar":
                if scan_msg is None:
                    raise RuntimeError("mono_relative_lidar mode requires LaserScan.")
                self._run_depth(
                    sample_id,
                    image_path,
                    self.get_relative_depth_model(),
                    self.args.depth_model_dir,
                    self.args.depth_device,
                )
                depth_npy = self.output_dir / f"{sample_id}_depth.npy"
                self._run_fit(sample_id, image_path, scan_path, depth_npy, segmentation_json)
                fit_json = self.fit_output_dir / f"{sample_id}_scan_monodepth_fit.json"
                fit_payload = json.loads(fit_json.read_text(encoding="utf-8"))
            elif localization_mode == "mono_absolute":
                depth, elapsed_ms, metadata = self._run_depth(
                    sample_id,
                    image_path,
                    self.get_absolute_depth_model(),
                    self.args.absolute_depth_model_dir,
                    self.args.absolute_depth_device,
                    output_dir=self.absolute_output_dir,
                    output_suffix="absolute_depth",
                )
                absolute_payload = metric_object_depth_payload(
                    depth,
                    segmentation_json,
                    (int(metadata["image_size"][0]), int(metadata["image_size"][1])),
                    self.args.absolute_depth_model_dir,
                    self.args.absolute_depth_device,
                    elapsed_ms,
                    self.absolute_output_dir,
                    sample_id,
                )
                absolute_status = "ok"
            elif localization_mode == "stereo":
                if right_image_bgr is None:
                    stereo_status = "missing_right_image"
                else:
                    try:
                        self._run_stereo(sample_id, image_path, right_image_path, segmentation_json)
                        stereo_json = self._stereo_json_path(sample_id)
                        stereo_payload = json.loads(stereo_json.read_text(encoding="utf-8"))
                        stereo_status = "ok"
                    except Exception as exc:
                        stereo_status = "failed"
                        stereo_error_text = str(exc)
                        print(f"[{sample_id}] stereo failed: {exc}", flush=True)
            elif localization_mode == "stereo_primary_mono_guard":
                if right_image_bgr is None:
                    stereo_status = "missing_right_image"
                    mono_guard_status = "missing_right_image"
                else:
                    try:
                        self._run_stereo(sample_id, image_path, right_image_path, segmentation_json)
                        stereo_json = self._stereo_json_path(sample_id)
                        stereo_payload = json.loads(stereo_json.read_text(encoding="utf-8"))
                        stereo_status = "ok"
                        depth, _, _ = self._run_depth(
                            sample_id,
                            image_path,
                            self.get_relative_depth_model(),
                            self.args.depth_model_dir,
                            self.args.depth_device,
                        )
                        depth_npy = self.output_dir / f"{sample_id}_depth.npy"
                        mono_guard_payload = compute_stereo_mono_guard(
                            stereo_payload=stereo_payload,
                            mono_depth=depth,
                            mono_depth_source=depth_npy,
                        )
                        mono_guard_json = self.output_dir / f"{sample_id}_stereo_mono_guard.json"
                        write_guard_payload(mono_guard_json, mono_guard_payload)
                        mono_guard_status = str(mono_guard_payload.get("status") or "unknown")
                    except Exception as exc:
                        mono_guard_status = "failed"
                        stereo_error_text = str(exc)
                        print(f"[{sample_id}] stereo+mono guard failed: {exc}", flush=True)

            if (
                localization_mode not in {"stereo", "stereo_primary_mono_guard"}
                and right_image_bgr is not None
                and self.args.enable_stereo_preview
            ):
                try:
                    self._run_stereo(sample_id, image_path, right_image_path, segmentation_json)
                    stereo_json = self._stereo_json_path(sample_id)
                    stereo_payload = json.loads(stereo_json.read_text(encoding="utf-8"))
                    stereo_status = "preview_ok"
                except Exception as exc:
                    stereo_status = "preview_failed"
                    stereo_error_text = str(exc)
                    print(f"[{sample_id}] stereo failed: {exc}", flush=True)

            stats = fit_payload.get("object_mask_metric_depth_m") or {}
            selected_fit = fit_payload.get("selected_fit") or {}
            samples = fit_payload.get("samples") or {}
            mask_support = fit_payload.get("object_mask_lidar_support") or {}
            source_det = seg_payload.get("source_detection") or {}
            mono_recommended = first_float(stats, ["p10", "p05", "median"])
            mono_error = None
            if mono_recommended is not None and truth_distance_m is not None:
                mono_error = float(mono_recommended) - float(truth_distance_m)

            absolute_stats = absolute_payload.get("object_mask_metric_depth_m") or {}
            absolute_mask = absolute_payload.get("mask") or {}
            absolute_recommended = first_float(absolute_stats, ["p10", "p05", "median"])
            absolute_error = None
            if absolute_recommended is not None and truth_distance_m is not None:
                absolute_error = float(absolute_recommended) - float(truth_distance_m)

            stereo_stats = {}
            stereo_base_stats = {}
            stereo_mask = {}
            stereo_summary: dict[str, Any] = {}
            if stereo_payload is not None:
                stereo_summary = stereo_base_depth_summary(stereo_payload)
                stereo_stats = stereo_summary.get("camera_x_stats") or {}
                stereo_base_stats = stereo_summary.get("base_x_stats") or {}
                stereo_mask = stereo_payload.get("mask") or {}
            stereo_recommended = stereo_summary.get("recommended_depth_m")
            if stereo_recommended is None:
                stereo_recommended = first_float(stereo_base_stats, ["median"]) or first_float(stereo_stats, ["median"])
            stereo_error = None
            if stereo_recommended is not None and truth_distance_m is not None:
                stereo_error = float(stereo_recommended) - float(truth_distance_m)

            mono_guard_recommended = None
            mono_guard_error = None
            mono_guard_selected_source = None
            mono_guard_reason = None
            mono_guard_fused_median = None
            mono_guard_delta = None
            mono_guard_anchor_count = None
            mono_guard_fit_rmse = None
            if mono_guard_payload:
                mono_guard_recommended = mono_guard_payload.get("selected_depth_m")
                mono_guard_selected_source = mono_guard_payload.get("selected_source")
                mono_guard_reason = mono_guard_payload.get("reason")
                mono_guard_delta = mono_guard_payload.get("correction_delta_m")
                mono_guard_anchor_count = mono_guard_payload.get("anchor_count")
                mono_guard_fit_rmse = (mono_guard_payload.get("fit") or {}).get("rmse_keep")
                mono_guard_fused_median = first_float(mono_guard_payload.get("fused_base_x_m") or {}, ["median"])
                if mono_guard_recommended is None:
                    mono_guard_recommended = stereo_recommended
                    mono_guard_selected_source = "stereo_fallback"
                    mono_guard_reason = mono_guard_payload.get("reason") or mono_guard_status
            if mono_guard_recommended is not None and truth_distance_m is not None:
                mono_guard_error = float(mono_guard_recommended) - float(truth_distance_m)

            recommended = None
            recommended_error = None
            if localization_mode == "stereo":
                recommended = stereo_recommended
                recommended_error = stereo_error
            elif localization_mode == "stereo_primary_mono_guard":
                recommended = mono_guard_recommended if mono_guard_recommended is not None else stereo_recommended
                recommended_error = mono_guard_error if mono_guard_recommended is not None else stereo_error
            elif localization_mode == "mono_relative_lidar":
                recommended = mono_recommended
                recommended_error = mono_error
            elif localization_mode == "mono_absolute":
                recommended = absolute_recommended
                recommended_error = absolute_error

            record.update(
                {
                    "status": "ok",
                    "recommended_depth_m": recommended,
                    "recommended_error_m": recommended_error,
                    "mono_recommended_depth_m": mono_recommended,
                    "mono_error_m": mono_error,
                    "mono_p05_m": first_float(stats, ["p05"]),
                    "mono_p10_m": first_float(stats, ["p10"]),
                    "mono_median_m": first_float(stats, ["median"]),
                    "mono_p90_m": first_float(stats, ["p90", "p95"]),
                    "stereo_recommended_depth_m": stereo_recommended,
                    "stereo_error_m": stereo_error,
                    "stereo_p05_m": first_float(stereo_stats, ["p05"]),
                    "stereo_p10_m": first_float(stereo_stats, ["p10"]),
                    "stereo_median_m": first_float(stereo_stats, ["median"]),
                    "stereo_p90_m": first_float(stereo_stats, ["p90", "p95"]),
                    "stereo_base_median_m": first_float(stereo_base_stats, ["median"]),
                    "stereo_base_range_xy_m": stereo_summary.get("base_range_xy_m"),
                    "stereo_recommended_source": stereo_summary.get("recommended_source"),
                    "stereo_valid_ratio": stereo_mask.get("valid_ratio"),
                    "stereo_valid_pixels": stereo_mask.get("valid_stereo_pixels"),
                    "stereo_status": stereo_status,
                    "mono_guard_recommended_depth_m": mono_guard_recommended,
                    "mono_guard_error_m": mono_guard_error,
                    "mono_guard_selected_source": mono_guard_selected_source,
                    "mono_guard_reason": mono_guard_reason,
                    "mono_guard_fused_median_m": mono_guard_fused_median,
                    "mono_guard_delta_m": mono_guard_delta,
                    "mono_guard_anchor_count": mono_guard_anchor_count,
                    "mono_guard_fit_rmse_m": mono_guard_fit_rmse,
                    "mono_guard_status": mono_guard_status,
                    "absolute_recommended_depth_m": absolute_recommended,
                    "absolute_error_m": absolute_error,
                    "absolute_p05_m": first_float(absolute_stats, ["p05"]),
                    "absolute_p10_m": first_float(absolute_stats, ["p10"]),
                    "absolute_median_m": first_float(absolute_stats, ["median"]),
                    "absolute_p90_m": first_float(absolute_stats, ["p90", "p95"]),
                    "absolute_valid_ratio": absolute_mask.get("valid_ratio"),
                    "absolute_status": absolute_status,
                    "selected_fit": selected_fit.get("mode"),
                    "fit_mae_m": selected_fit.get("mae_m"),
                    "fit_p90_m": selected_fit.get("p90_abs_error_m"),
                    "projected_samples": samples.get("projected_inside_image"),
                    "used_for_fit_samples": samples.get("used_for_fit"),
                    "edge_rejected_samples": samples.get("edge_rejected"),
                    "mask_lidar_support_enabled": mask_support.get("enabled"),
                    "mask_lidar_has_support": mask_support.get("has_support"),
                    "mask_lidar_points": mask_support.get("usable_projected_points_in_mask"),
                    "mask_lidar_required_points": mask_support.get("required_usable_points"),
                    "mask_lidar_points_per_width": mask_support.get("points_per_mask_width"),
                    "mask_area_px": seg_payload.get("mask_area_px"),
                    "detection_label": source_det.get("label"),
                    "detection_score": source_det.get("score"),
                    "grounding_json": str(grounding_json),
                    "grounding_overlay": grounding_payload.get("overlay_path"),
                    "segmentation_json": str(segmentation_json),
                    "fit_json": str(self.fit_output_dir / f"{sample_id}_scan_monodepth_fit.json") if fit_payload else None,
                    "segmentation_overlay": seg_payload.get("overlay_path"),
                    "metric_depth_color": (fit_payload.get("outputs") or {}).get("metric_depth_color")
                    or (
                        str(self.output_dir / f"{sample_id}_depth_color.png")
                        if mono_guard_payload
                        else None
                    ),
                    "projection_overlay": (fit_payload.get("outputs") or {}).get("projection"),
                    "fit_plot": (fit_payload.get("outputs") or {}).get("fit_plot"),
                    "absolute_depth_json": absolute_payload.get("json_path"),
                    "absolute_depth_color": str(self.absolute_output_dir / f"{sample_id}_absolute_depth_depth_color.png") if absolute_payload else None,
                }
            )
            if stereo_payload is not None:
                record.update(
                    {
                        "stereo_json": str(self._stereo_json_path(sample_id)),
                        "stereo_overlay": stereo_payload.get("overlay_path"),
                        "stereo_depth_color": stereo_payload.get("depth_color_path"),
                        "stereo_disparity": stereo_payload.get("disparity_path"),
                    }
                )
            if mono_guard_json is not None:
                record["mono_guard_json"] = str(mono_guard_json)
            if stereo_error_text is not None:
                record["stereo_error"] = stereo_error_text
            self.set_status(result_summary(record))
        except Exception as exc:
            record["status"] = "failed"
            record["error"] = str(exc)
            record["traceback"] = traceback.format_exc()
            self.set_status(f"[{sample_id}] failed: {exc}")

        append_csv(self.summary_csv, record)
        append_jsonl(self.summary_jsonl, record)
        return record

    def _run_grounding(self, sample_id: str, image_path: Path, prompt: str) -> None:
        self.set_status(f"[{sample_id}] grounding...")
        if self.grounding_model is None:
            raise RuntimeError("GroundingDINO model was not loaded for this PipelineRunner.")
        t0 = time.perf_counter()
        payload = self.grounding_model.detect(image_path=image_path, text_prompt=prompt)
        elapsed = time.perf_counter() - t0
        grounding_json = self.output_dir / f"{sample_id}_grounding_openvino.json"
        overlay_path = self.output_dir / f"{sample_id}_grounding_openvino.png"
        draw_detections(image_path, payload["detections"], overlay_path)
        payload["overlay_path"] = str(overlay_path)
        grounding_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        log_path = self.command_logs_dir / f"{sample_id}_grounding.log"
        log_path.write_text(
            f"COMMAND: in-process grounding_model.detect()\n"
            f"ELAPSED_SEC: {elapsed:.3f}\n"
            f"detections: {len(payload['detections'])}\n"
            f"inference_ms: {payload['metadata']['elapsed_ms']:.2f}\n",
            encoding="utf-8",
        )
        if not payload.get("detections"):
            raise RuntimeError(f"No GroundingDINO detections for prompt: {prompt}")

    def _run_sam(self, sample_id: str, grounding_json: Path, label_query: str) -> None:
        self.set_status(f"[{sample_id}] sam...")
        data = json.loads(grounding_json.read_text(encoding="utf-8"))
        image_path = Path(data["metadata"]["image"])
        detections = data.get("detections", [])
        if not detections:
            raise RuntimeError("No detections in grounding JSON")
        detection = choose_detection(detections, label_query)
        image_np = np.array(PILImage.open(image_path).convert("RGB"))
        orig_h, orig_w = image_np.shape[:2]
        t0 = time.perf_counter()
        embeddings = self.sam_model.get_embedding(image_np)
        box = detection["box"]
        mask, _, iou = self.sam_model.predict_mask(embeddings, tuple(box), orig_h, orig_w)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        save_overlay(image_path, mask, detection, iou, elapsed_ms, self.output_dir, sample_id)
        log_path = self.command_logs_dir / f"{sample_id}_sam_openvino.log"
        log_path.write_text(
            f"COMMAND: in-process sam_model.get_embedding()+predict_mask()\n"
            f"ELAPSED_MS: {elapsed_ms:.1f}\n"
            f"detection: {detection.get('label')} score={detection.get('score'):.3f}\n"
            f"sam_iou: {iou:.3f}\n",
            encoding="utf-8",
        )

    def _segmentation_json_path(self, sample_id: str) -> Path:
        ov_path = self.output_dir / f"{sample_id}_segmentation_ov.json"
        if ov_path.exists():
            return ov_path
        raise FileNotFoundError(f"segmentation json not found for {sample_id}")

    def _run_depth(
        self,
        sample_id: str,
        image_path: Path,
        model: DepthAnythingOpenVINO,
        model_dir: Path,
        device: str,
        output_dir: Path | None = None,
        output_suffix: str = "depth",
    ) -> tuple[np.ndarray, float, dict[str, Any]]:
        self.set_status(f"[{sample_id}] {output_suffix}...")
        image = PILImage.open(image_path).convert("RGB")
        depth, elapsed_ms = model.predict(image)
        metadata = {
            "backend": "openvino",
            "device": device,
            "model_dir": str(model_dir),
            "image": str(image_path),
            "image_size": [image.width, image.height],
            "depth_shape": list(depth.shape),
            "depth_min": float(np.nanmin(depth)),
            "depth_max": float(np.nanmax(depth)),
            "depth_mean": float(np.nanmean(depth)),
            "elapsed_ms": elapsed_ms,
        }
        stem = sample_id if output_suffix == "depth" else f"{sample_id}_{output_suffix}"
        save_depth_outputs(depth, metadata, output_dir or self.output_dir, stem)
        log_path = self.command_logs_dir / f"{sample_id}_{output_suffix}.log"
        log_path.write_text(
            f"COMMAND: in-process depth_model.predict()\n"
            f"ELAPSED_MS: {elapsed_ms:.1f}\n"
            f"depth_shape: {depth.shape}\n",
            encoding="utf-8",
        )
        return depth, elapsed_ms, metadata

    def _run_fit(
        self,
        sample_id: str,
        image_path: Path,
        scan_path: Path,
        depth_npy: Path,
        segmentation_json: Path,
    ) -> None:
        cmd = [
            sys.executable,
            "-m", "caragent_agent.perception.fusion.project_scan_fit_monodepth",
            "--image",
            str(image_path),
            "--scan",
            str(scan_path),
            "--mono-depth-npy",
            str(depth_npy),
            "--calib-file",
            str(self.args.calib_file),
            "--extrinsics-json",
            str(self.args.extrinsics_json),
            "--segmentation-json",
            str(segmentation_json),
            "--output-dir",
            str(self.fit_output_dir),
            "--fit-modes",
            "log,quadratic",
            "--selection-p90-tolerance",
            str(self.args.selection_p90_tolerance),
        ]
        if self.args.mask_lidar_support_check:
            cmd.extend(
                [
                    "--mask-lidar-support-check",
                    "--min-mask-lidar-points",
                    str(self.args.min_mask_lidar_points),
                    "--min-mask-lidar-density",
                    str(self.args.min_mask_lidar_density),
                ]
            )
        self._run_cmd(sample_id, "scan_monodepth_fit", cmd)

    def _run_stereo(
        self,
        sample_id: str,
        left_image_path: Path,
        right_image_path: Path,
        segmentation_json: Path,
    ) -> None:
        cmd = [
            sys.executable,
            "-m", "caragent_agent.perception.fusion.run_stereo_object_depth",
            "--left-image",
            str(left_image_path),
            "--right-image",
            str(right_image_path),
            "--segmentation-json",
            str(segmentation_json),
            "--calib-file",
            str(self.args.calib_file),
            "--output-dir",
            str(self.stereo_output_dir),
            "--num-disparities",
            str(self.args.stereo_num_disparities),
            "--block-size",
            str(self.args.stereo_block_size),
            "--min-depth",
            str(self.args.stereo_min_depth),
            "--max-depth",
            str(self.args.stereo_max_depth),
        ]
        self._run_cmd(sample_id, "stereo_object_depth", cmd)

    def _stereo_json_path(self, sample_id: str) -> Path:
        return self.stereo_output_dir / f"{sample_id}_stereo_object_3d.json"

    def _run_cmd(self, sample_id: str, step: str, cmd: list[str]) -> None:
        self.set_status(f"[{sample_id}] {step}...")
        env = os.environ.copy()
        env.setdefault("HF_HOME", str(self.args.workspace / "hf_cache"))
        env.setdefault("HUGGINGFACE_HUB_CACHE", str(self.args.workspace / "hf_cache" / "hub"))
        env.setdefault("TRANSFORMERS_CACHE", str(self.args.workspace / "hf_cache" / "transformers"))
        started = time.perf_counter()
        proc = subprocess.run(
            cmd,
            cwd=str(self.args.workspace),
            env=env,
            text=True,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
        )
        elapsed = time.perf_counter() - started
        log_path = self.command_logs_dir / f"{sample_id}_{step}.log"
        log_path.write_text(
            "COMMAND:\n"
            + " ".join(cmd)
            + f"\n\nRETURN_CODE: {proc.returncode}\nELAPSED_SEC: {elapsed:.3f}\n\nSTDOUT:\n"
            + proc.stdout
            + "\n\nSTDERR:\n"
            + proc.stderr,
            encoding="utf-8",
        )
        if proc.returncode != 0:
            raise RuntimeError(f"{step} failed, see {log_path}")


def first_float(data: dict[str, Any], keys: list[str]) -> float | None:
    for key in keys:
        value = data.get(key)
        if value is None:
            continue
        try:
            value_f = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value_f):
            return value_f
    return None


def result_summary(record: dict[str, Any]) -> str:
    mode = str(record.get("localization_mode") or "")
    mode_label = LOCALIZATION_MODE_LABELS.get(mode, mode or "mode")
    rec = record.get("recommended_depth_m")
    truth = record.get("truth_distance_m")
    err = record.get("recommended_error_m")
    fit = record.get("selected_fit")
    stereo = record.get("stereo_recommended_depth_m")
    stereo_status = record.get("stereo_status")
    if rec is None:
        return f"[{record.get('sample_id')}] {mode_label} ok, but no valid mask depth stats"
    extras = ""
    if mode == "mono_relative_lidar":
        extras = f" fit={fit}"
    elif mode == "stereo":
        extras = f" stereo_status={stereo_status} source={record.get('stereo_recommended_source')}"
    elif mode == "stereo_primary_mono_guard":
        extras = (
            f" stereo_status={stereo_status}"
            f" guard={record.get('mono_guard_selected_source') or record.get('mono_guard_status')}"
            f" reason={record.get('mono_guard_reason')}"
        )
    elif mode == "mono_absolute":
        extras = f" valid={record.get('absolute_valid_ratio')}"
    if truth is None or err is None:
        return f"[{record.get('sample_id')}] {mode_label} recommended={fmt_m(rec)}{extras}"
    return (
        f"[{record.get('sample_id')}] {mode_label} recommended={fmt_m(rec)} "
        f"err={err:+.3f}m truth={truth:.3f}m{extras}"
    )


def fmt_m(value: Any) -> str:
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(value_f):
        return "n/a"
    return f"{value_f:.3f}m"


def parse_optional_float(text: str) -> float | None:
    text = text.strip()
    if not text:
        return None
    value = float(text)
    if value <= 0:
        raise ValueError("distance must be positive")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Live validation UI for scan-monodepth fusion.")
    parser.add_argument("--workspace", default=DEFAULT_WORKSPACE, type=Path)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, type=Path)
    parser.add_argument("--image-topic", default="/stereo/left/image_raw")
    parser.add_argument("--right-image-topic", default="/stereo/right/image_raw")
    parser.add_argument("--scan-topic", default="/scan")
    parser.add_argument("--target", default="", help="Initial GroundingDINO target text, e.g. 'door'.")
    parser.add_argument("--label-query", default="", help="Optional SAM detection label filter. Defaults to target.")
    parser.add_argument("--truth-distance-m", default=None, type=float)
    parser.add_argument("--calib-file", default=DEFAULT_CALIB, type=Path)
    parser.add_argument("--extrinsics-json", default=DEFAULT_EXTR, type=Path)
    parser.add_argument("--grounding-model-dir", default=DEFAULT_GROUNDING_MODEL_DIR, type=Path)
    parser.add_argument("--grounding-model-id", default=DEFAULT_GROUNDING_MODEL_ID)
    parser.add_argument("--grounding-device", default="GPU")
    parser.add_argument("--depth-model-dir", default=DEFAULT_DEPTH_MODEL_DIR, type=Path)
    parser.add_argument("--depth-device", default="GPU")
    parser.add_argument("--absolute-depth-model-dir", default=DEFAULT_ABSOLUTE_DEPTH_MODEL_DIR, type=Path)
    parser.add_argument("--absolute-depth-device", default="GPU")
    parser.add_argument(
        "--localization-mode",
        default="mono_relative_lidar",
        choices=LOCALIZATION_MODE_CHOICES,
        help="Object depth backend. Press m in the UI to cycle modes.",
    )
    parser.add_argument("--sam-device", default="GPU", help="OpenVINO SAM device (used when per-stage override is empty).")
    parser.add_argument("--sam-encoder-device", default="", help="Override SAM encoder device (default: --sam-device).")
    parser.add_argument("--sam-decoder-device", default="CPU", help="Override SAM decoder device (default: CPU to avoid NaN on GPU).")
    parser.add_argument("--sam-encoder-xml", default=DEFAULT_SAM_ENCODER_XML, type=Path)
    parser.add_argument("--sam-decoder-xml", default=DEFAULT_SAM_DECODER_XML, type=Path)
    parser.add_argument(
        "--enable-stereo-preview",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also run stereo as a preview when the selected mode is not stereo.",
    )
    parser.add_argument("--stereo-num-disparities", default=96, type=int)
    parser.add_argument("--stereo-block-size", default=5, type=int)
    parser.add_argument("--stereo-min-depth", default=0.15, type=float)
    parser.add_argument("--stereo-max-depth", default=8.0, type=float)
    parser.add_argument("--selection-p90-tolerance", default=0.10, type=float)
    parser.add_argument(
        "--mask-lidar-support-check",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Optional diagnostic only; does not change mono/stereo recommendation or fallback behavior.",
    )
    parser.add_argument("--min-mask-lidar-points", default=2, type=int)
    parser.add_argument("--min-mask-lidar-density", default=0.035, type=float)
    parser.add_argument("--max-age-sec", default=3.0, type=float)
    parser.add_argument("--panel-width", default=640, type=int)
    parser.add_argument("--panel-height", default=420, type=int)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.workspace = args.workspace.resolve()
    args.output_dir = args.output_dir.resolve()
    args.calib_file = args.calib_file.resolve()
    args.extrinsics_json = args.extrinsics_json.resolve()
    args.grounding_model_dir = args.grounding_model_dir.resolve()
    if isinstance(args.grounding_model_id, Path):
        args.grounding_model_id = args.grounding_model_id.resolve()
    args.depth_model_dir = args.depth_model_dir.resolve()
    args.absolute_depth_model_dir = args.absolute_depth_model_dir.resolve()
    args.sam_encoder_xml = args.sam_encoder_xml.resolve()
    args.sam_decoder_xml = args.sam_decoder_xml.resolve()

    target_text = args.target.strip()
    label_query = args.label_query.strip()
    truth_distance_m = args.truth_distance_m
    localization_mode = args.localization_mode
    if not target_text and sys.stdin.isatty():
        target_text = input("Target description, e.g. door/elevator door/pillar: ").strip()
    if not label_query:
        label_query = target_text
    if truth_distance_m is None:
        if sys.stdin.isatty():
            raw = input("Reference distance from left camera in meters (empty to skip): ")
            truth_distance_m = parse_optional_float(raw)

    config_path = args.output_dir / "live_config.json"
    config_mtime: float = 0.0
    config_last_check: float = 0.0

    def load_live_config() -> tuple[str, str, float | None]:
        if not config_path.exists():
            return target_text, label_query, truth_distance_m
        try:
            cfg = json.loads(config_path.read_text(encoding="utf-8"))
            t = str(cfg.get("target", target_text)).strip() or target_text
            l = str(cfg.get("label_query", label_query)).strip() or t
            d = cfg.get("truth_distance_m")
            return t, l, (float(d) if d is not None else truth_distance_m)
        except Exception:
            return target_text, label_query, truth_distance_m

    latest = LatestMessages()
    lock = threading.Lock()
    status_lock = threading.Lock()
    status_text = "Ready. Press r to run, m mode, t target, d distance, l label, q quit."
    last_record: dict[str, Any] | None = None
    worker: threading.Thread | None = None

    def set_status(text: str) -> None:
        nonlocal status_text
        with status_lock:
            status_text = text

    win = "Live Preview (r=capture q=quit)"
    result_win = "Inference Results"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.namedWindow(result_win, cv2.WINDOW_NORMAL)
    loading_panel = draw_text_panel(["Loading models...", "", "GroundingDINO, EfficientSAM, DepthAnything"], args.panel_width * 2, args.panel_height)
    cv2.imshow(win, loading_panel)
    cv2.waitKey(1)

    runner = PipelineRunner(args, set_status)
    laser_pose = load_laser_pose(args.extrinsics_json)

    rclpy.init()
    node = LiveCaptureNode(args.image_topic, args.right_image_topic, args.scan_topic, latest, lock)

    def launch_worker() -> None:
        nonlocal worker, last_record
        with lock:
            image_bgr, right_image_bgr, scan_msg, meta = latest.copy_snapshot()
            image_age = time.monotonic() - latest.image_recv_time if latest.image_recv_time else float("inf")
            right_age = time.monotonic() - latest.right_image_recv_time if latest.right_image_recv_time else float("inf")
            scan_age = time.monotonic() - latest.scan_recv_time if latest.scan_recv_time else float("inf")
        if image_bgr is None:
            set_status("No camera image yet.")
            return
        if localization_mode == "mono_relative_lidar" and scan_msg is None:
            set_status("No LaserScan yet.")
            return
        ages = [image_age]
        if localization_mode == "mono_relative_lidar":
            ages.append(scan_age)
        if localization_mode in {"stereo", "stereo_primary_mono_guard"} or args.enable_stereo_preview:
            ages.append(right_age)
        if max(ages) > args.max_age_sec:
            set_status(f"Stale input: left={image_age:.1f}s right={right_age:.1f}s scan={scan_age:.1f}s")
            return

        def run() -> None:
            nonlocal last_record
            result = runner.run_sample(
                image_bgr=image_bgr,
                right_image_bgr=right_image_bgr,
                scan_msg=scan_msg,
                snapshot_meta=meta,
                target_text=target_text,
                label_query=label_query,
                truth_distance_m=truth_distance_m,
                localization_mode=localization_mode,
            )
            last_record = result

        worker = threading.Thread(target=run, daemon=True)
        worker.start()

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.01)
            # Watch config file for live target updates (every ~1s)
            now = time.monotonic()
            if now - config_last_check > 1.0:
                config_last_check = now
                try:
                    new_mtime = config_path.stat().st_mtime if config_path.exists() else 0.0
                    if new_mtime != config_mtime:
                        config_mtime = new_mtime
                        new_target, new_label, new_truth = load_live_config()
                        if new_target != target_text or new_label != label_query or new_truth != truth_distance_m:
                            target_text, label_query, truth_distance_m = new_target, new_label, new_truth
                            set_status(f"Config reloaded: target={target_text} label={label_query} truth={fmt_m(truth_distance_m)}")
                except Exception:
                    pass
            with lock:
                image = None if latest.image_bgr is None else latest.image_bgr.copy()
                right_image = None if latest.right_image_bgr is None else latest.right_image_bgr.copy()
                scan = latest.scan_msg
                image_age = time.monotonic() - latest.image_recv_time if latest.image_recv_time else float("inf")
                right_age = time.monotonic() - latest.right_image_recv_time if latest.right_image_recv_time else float("inf")
                scan_age = time.monotonic() - latest.scan_recv_time if latest.scan_recv_time else float("inf")
            with status_lock:
                current_status = status_text

            busy = worker is not None and worker.is_alive()
            if worker is not None and not worker.is_alive():
                worker = None

            # Window 1: clean camera + scan (minimal status bar)
            mode_label = LOCALIZATION_MODE_LABELS[localization_mode]
            status_bar = f"mode={mode_label} | target={target_text or 'n/a'} | {'RUNNING' if busy else 'idle'} | m=mode r=run q=quit"
            camera_panel_img = draw_camera_panel(image, args.panel_width, args.panel_height, [status_bar])
            scan_panel_img = draw_scan_panel(scan, laser_pose, args.panel_width, args.panel_height)
            live_canvas = np.hstack([camera_panel_img, scan_panel_img])
            cv2.imshow(win, live_canvas)

            if last_record:
                rw, rh = 400, 300
                grid_images: list[np.ndarray] = []
                mono_depth_panel = last_record.get("metric_depth_color")
                projection_panel = last_record.get("projection_overlay")
                fit_panel = last_record.get("fit_plot")
                absolute_panel = last_record.get("absolute_depth_color")
                stereo_panel = last_record.get("stereo_overlay") or last_record.get("stereo_depth_color")
                if last_record.get("localization_mode") == "mono_absolute":
                    mono_depth_panel = absolute_panel
                    projection_panel = None
                    fit_panel = None
                grid_labels = [
                    ("GroundingDINO", last_record.get("grounding_overlay")),
                    ("SAM segmentation", last_record.get("segmentation_overlay")),
                    ("Selected depth", mono_depth_panel),
                    ("LiDAR projection", projection_panel),
                    ("Fit curve", fit_panel),
                    ("Stereo depth", stereo_panel),
                ]
                for label, path in grid_labels:
                    panel = safe_read_image(path, rw, rh, [f"No {label.lower()}"])
                    cv2.rectangle(panel, (0, 0), (rw, 22), (0, 0, 0), -1)
                    cv2.putText(panel, label, (6, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                    grid_images.append(panel)
                row1 = np.hstack(grid_images[:3])
                row2 = np.hstack(grid_images[3:6])
                body = np.vstack([row1, row2])
                # Title bar at top
                title_h = 36
                title_bar = np.full((title_h, rw * 3, 3), (30, 30, 30), dtype=np.uint8)
                cv2.putText(title_bar, result_summary(last_record), (12, 26),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 255), 1, cv2.LINE_AA)
                result_canvas = np.vstack([title_bar, body])
                cv2.imshow(result_win, result_canvas)
            else:
                if cv2.getWindowProperty(result_win, cv2.WND_PROP_VISIBLE) >= 0:
                    cv2.destroyWindow(result_win)
            key = cv2.waitKey(30) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("r"):
                if busy:
                    set_status("Inference is already running.")
                else:
                    launch_worker()
            elif key == ord("m"):
                if busy:
                    set_status("Cannot switch mode while inference is running.")
                else:
                    idx = LOCALIZATION_MODE_CHOICES.index(localization_mode)
                    localization_mode = LOCALIZATION_MODE_CHOICES[(idx + 1) % len(LOCALIZATION_MODE_CHOICES)]
                    set_status(f"localization mode: {LOCALIZATION_MODE_LABELS[localization_mode]}")
            elif key == ord("t") and sys.stdin.isatty():
                new_target = input("Target description: ").strip()
                if new_target:
                    target_text = new_target
                    if not args.label_query:
                        label_query = new_target
                    set_status(f"target set to: {target_text}")
            elif key == ord("l") and sys.stdin.isatty():
                new_label = input("SAM label query: ").strip()
                if new_label:
                    label_query = new_label
                    set_status(f"label query set to: {label_query}")
            elif key == ord("d") and sys.stdin.isatty():
                raw = input("Reference distance from left camera in meters (empty to skip): ")
                truth_distance_m = parse_optional_float(raw)
                set_status(f"truth distance set to: {fmt_m(truth_distance_m)}")
    finally:
        if worker is not None and worker.is_alive():
            print("Waiting for running inference to finish...", flush=True)
            worker.join()
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

    print(f"CSV: {runner.summary_csv}")
    print(f"JSONL: {runner.summary_jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
