"""Collect object-depth benchmark samples from live stereo camera and LaserScan."""

from __future__ import annotations

import argparse
import json
import re
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan


DEFAULT_WORKSPACE = Path.home() / "caragent_ws"
DEFAULT_DATASET_ROOT = DEFAULT_WORKSPACE / "perception_datasets" / "object_depth"
DEFAULT_GROUNDING_MODEL_DIR = DEFAULT_WORKSPACE / "models" / "grounding_dino_openvino"
DEFAULT_GROUNDING_MODEL_ID = DEFAULT_WORKSPACE / "models" / "grounding-dino-tiny"
DEFAULT_SAM_ENCODER_XML = DEFAULT_WORKSPACE / "models" / "efficient_sam_openvino" / "efficient_sam_vitt_encoder.xml"
DEFAULT_SAM_DECODER_XML = DEFAULT_WORKSPACE / "models" / "efficient_sam_openvino" / "efficient_sam_vitt_decoder.xml"


def safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return value.strip("_") or datetime.now().strftime("dataset_%Y%m%d_%H%M%S")


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

    raise ValueError(f"unsupported image encoding: {msg.encoding}")


def save_scan_npz(scan_msg: LaserScan, output_path: Path) -> None:
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


class DatasetCollector(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("object_depth_dataset_collector")
        self.args = args
        self.dataset_dir = args.dataset_root / safe_name(args.dataset_name)
        self.samples_dir = self.dataset_dir / "samples"
        self.preview_dir = self.dataset_dir / "previews"
        self.samples_dir.mkdir(parents=True, exist_ok=True)
        self.preview_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.dataset_dir / "manifest.jsonl"
        self.live_config_path = self.dataset_dir / "live_config.json"
        self.lock = threading.RLock()
        self.left_bgr: np.ndarray | None = None
        self.right_bgr: np.ndarray | None = None
        self.scan_msg: LaserScan | None = None
        self.left_stamp = 0.0
        self.right_stamp = 0.0
        self.scan_stamp = 0.0
        self.left_recv = 0.0
        self.right_recv = 0.0
        self.scan_recv = 0.0
        self.target = args.target
        self.label_query = args.label_query or args.target
        self.truth_distance_m = args.truth_distance_m
        self.note = args.note
        self.grounding_model = None
        self.sam_model = None
        self.preview_overlay: np.ndarray | None = None
        self.preview_status = "press v to validate detection/mask"
        self._write_dataset_json()
        self._write_live_config()
        self.create_subscription(Image, args.left_topic, self._on_left, 1)
        self.create_subscription(Image, args.right_topic, self._on_right, 1)
        self.create_subscription(LaserScan, args.scan_topic, self._on_scan, 1)
        self.get_logger().info(f"dataset: {self.dataset_dir}")
        self.get_logger().info("keys: v=validate detection/mask, c=capture, q=quit")

    def _on_left(self, msg: Image) -> None:
        try:
            image = image_msg_to_bgr(msg)
        except Exception as exc:
            self.get_logger().warning(f"decode left failed: {exc}")
            return
        with self.lock:
            self.left_bgr = image
            self.left_stamp = stamp_to_sec(msg)
            self.left_recv = time.monotonic()

    def _on_right(self, msg: Image) -> None:
        try:
            image = image_msg_to_bgr(msg)
        except Exception as exc:
            self.get_logger().warning(f"decode right failed: {exc}")
            return
        with self.lock:
            self.right_bgr = image
            self.right_stamp = stamp_to_sec(msg)
            self.right_recv = time.monotonic()

    def _on_scan(self, msg: LaserScan) -> None:
        with self.lock:
            self.scan_msg = msg
            self.scan_stamp = stamp_to_sec(msg)
            self.scan_recv = time.monotonic()

    def _write_dataset_json(self) -> None:
        payload = {
            "name": self.dataset_dir.name,
            "dataset_dir": str(self.dataset_dir),
            "samples_dir": str(self.samples_dir),
            "created_or_updated_at": datetime.now().isoformat(timespec="seconds"),
            "schema": "caragent_object_depth_dataset_v1",
        }
        (self.dataset_dir / "dataset.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _write_live_config(self) -> None:
        payload = {
            "target": self.target,
            "label_query": self.label_query,
            "truth_distance_m": self.truth_distance_m,
            "note": self.note,
        }
        self.live_config_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _reload_live_config(self) -> None:
        if not self.live_config_path.exists():
            return
        try:
            data = json.loads(self.live_config_path.read_text(encoding="utf-8"))
        except Exception:
            return
        self.target = str(data.get("target") or self.target).strip() or self.target
        self.label_query = str(data.get("label_query") or self.label_query).strip() or self.target
        raw_truth = data.get("truth_distance_m", self.truth_distance_m)
        self.truth_distance_m = None if raw_truth in {"", None} else float(raw_truth)
        self.note = str(data.get("note") or "")

    def _sample_count(self) -> int:
        if not self.manifest_path.exists():
            return 0
        return sum(1 for line in self.manifest_path.read_text(encoding="utf-8").splitlines() if line.strip())

    def capture(self) -> None:
        self._reload_live_config()
        with self.lock:
            left = None if self.left_bgr is None else self.left_bgr.copy()
            right = None if self.right_bgr is None else self.right_bgr.copy()
            scan = self.scan_msg
            meta = {
                "left_stamp_sec": self.left_stamp,
                "right_stamp_sec": self.right_stamp,
                "scan_stamp_sec": self.scan_stamp,
                "left_age_sec": time.monotonic() - self.left_recv if self.left_recv else None,
                "right_age_sec": time.monotonic() - self.right_recv if self.right_recv else None,
                "scan_age_sec": time.monotonic() - self.scan_recv if self.scan_recv else None,
            }
        if left is None:
            self.get_logger().warning("no left image yet")
            return
        sample_id = datetime.now().strftime("sample_%Y%m%d_%H%M%S_%f")[:-3]
        left_path = self.samples_dir / f"{sample_id}_left.png"
        right_path = self.samples_dir / f"{sample_id}_right.png"
        scan_path = self.samples_dir / f"{sample_id}_scan.npz"
        meta_path = self.samples_dir / f"{sample_id}.json"
        cv2.imwrite(str(left_path), left)
        if right is not None:
            cv2.imwrite(str(right_path), right)
        if scan is not None:
            save_scan_npz(scan, scan_path)
        record = {
            "sample_id": sample_id,
            "target": self.target,
            "label_query": self.label_query,
            "truth_distance_m": self.truth_distance_m,
            "note": self.note,
            "left_image": str(left_path),
            "right_image": str(right_path) if right is not None else "",
            "scan": str(scan_path) if scan is not None else "",
            "captured_at": datetime.now().isoformat(timespec="milliseconds"),
            "image_size": {
                "left": [int(left.shape[1]), int(left.shape[0])],
                "right": [int(right.shape[1]), int(right.shape[0])] if right is not None else None,
            },
            "meta": meta,
        }
        meta_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
        with self.manifest_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        right_size = f"{right.shape[1]}x{right.shape[0]}" if right is not None else "none"
        self.get_logger().info(
            f"captured {sample_id}: target={self.target} label={self.label_query} truth={self.truth_distance_m} "
            f"left={left.shape[1]}x{left.shape[0]} right={right_size}"
        )

    def _format_grounding_prompt(self, text: str) -> str:
        normalized = text.strip()
        if not normalized:
            return ""
        terms = [term.strip() for term in normalized.replace(";", ".").split(".") if term.strip()]
        if not terms:
            terms = [normalized]
        return " . ".join(terms) + " ."

    def _ensure_preview_models(self) -> None:
        if self.grounding_model is None:
            from caragent_agent.perception.grounding.grounding_dino_openvino import GroundingDINOOpenVINO

            self.preview_status = "loading GroundingDINO..."
            self.grounding_model = GroundingDINOOpenVINO(
                model_dir=self.args.grounding_model_dir,
                model_id=str(self.args.grounding_model_id),
                device=self.args.grounding_device,
            )
        if self.sam_model is None:
            from caragent_agent.perception.sam.efficient_sam_openvino import EfficientSAMOpenVINO

            self.preview_status = "loading EfficientSAM..."
            self.sam_model = EfficientSAMOpenVINO(
                encoder_xml=self.args.sam_encoder_xml,
                decoder_xml=self.args.sam_decoder_xml,
                device=self.args.sam_device,
                encoder_device=self.args.sam_encoder_device or self.args.sam_device,
                decoder_device=self.args.sam_decoder_device,
            )

    def validate_current_frame(self) -> None:
        self._reload_live_config()
        with self.lock:
            left = None if self.left_bgr is None else self.left_bgr.copy()
        if left is None:
            self.preview_status = "no left image"
            self.get_logger().warning(self.preview_status)
            return
        try:
            self._ensure_preview_models()
            assert self.grounding_model is not None
            assert self.sam_model is not None
            from PIL import Image as PILImage
            from caragent_agent.perception.grounding.run_grounding_dino_openvino import draw_detections
            from caragent_agent.perception.sam.run_efficientsam_openvino import choose_detection, save_overlay

            stamp = datetime.now().strftime("preview_%Y%m%d_%H%M%S")
            image_path = self.preview_dir / f"{stamp}.png"
            grounding_json = self.preview_dir / f"{stamp}_grounding.json"
            grounding_overlay = self.preview_dir / f"{stamp}_grounding.png"
            cv2.imwrite(str(image_path), left)

            prompt = self._format_grounding_prompt(self.target)
            self.preview_status = f"detecting: {prompt}"
            payload = self.grounding_model.detect(image_path=image_path, text_prompt=prompt)
            payload["metadata"]["target"] = self.target
            payload["metadata"]["label_query"] = self.label_query
            draw_detections(image_path, payload.get("detections", []), grounding_overlay)
            payload["overlay_path"] = str(grounding_overlay)
            grounding_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
            detections = payload.get("detections", [])
            if not detections:
                self.preview_overlay = cv2.imread(str(grounding_overlay), cv2.IMREAD_COLOR)
                self.preview_status = f"no detection for {self.target}"
                return

            detection = choose_detection(detections, self.label_query or self.target)
            image_np = np.asarray(PILImage.open(image_path).convert("RGB"))
            orig_h, orig_w = image_np.shape[:2]
            embeddings = self.sam_model.get_embedding(image_np)
            mask, _, iou = self.sam_model.predict_mask(embeddings, tuple(detection["box"]), orig_h, orig_w)
            save_overlay(image_path, mask, detection, iou, 0.0, self.preview_dir, stamp)
            seg_json = self.preview_dir / f"{stamp}_segmentation_ov.json"
            seg_data = json.loads(seg_json.read_text(encoding="utf-8"))
            overlay_path = seg_data.get("overlay_path") or grounding_overlay
            self.preview_overlay = cv2.imread(str(overlay_path), cv2.IMREAD_COLOR)
            self.preview_status = (
                f"ok label={detection.get('label')} score={float(detection.get('score', 0.0)):.2f} "
                f"iou={iou:.2f}"
            )
            self.get_logger().info(self.preview_status)
        except Exception as exc:
            self.preview_status = f"validate failed: {exc}"
            self.get_logger().error(self.preview_status)

    def draw_preview(self) -> np.ndarray:
        self._reload_live_config()
        with self.lock:
            left = None if self.left_bgr is None else self.left_bgr.copy()
            left_age = time.monotonic() - self.left_recv if self.left_recv else float("inf")
            right_age = time.monotonic() - self.right_recv if self.right_recv else float("inf")
            scan_age = time.monotonic() - self.scan_recv if self.scan_recv else float("inf")
            right_shape = None if self.right_bgr is None else self.right_bgr.shape[:2]
        if self.preview_overlay is not None:
            source = self.preview_overlay
        elif left is None:
            source = None
        else:
            source = left

        canvas_w, canvas_h = 960, 600
        if source is None:
            panel = np.full((canvas_h, canvas_w, 3), 30, dtype=np.uint8)
            cv2.putText(panel, "Waiting for left image...", (34, canvas_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (230, 230, 230), 2)
            left_size = "none"
        else:
            h, w = source.shape[:2]
            left_size = f"{w}x{h}"
            scale = min(canvas_w / max(1, w), canvas_h / max(1, h))
            view_w = max(1, int(round(w * scale)))
            view_h = max(1, int(round(h * scale)))
            view = cv2.resize(source, (view_w, view_h), interpolation=cv2.INTER_AREA)
            panel = np.full((canvas_h, canvas_w, 3), 24, dtype=np.uint8)
            ox = (canvas_w - view_w) // 2
            oy = (canvas_h - view_h) // 2
            panel[oy : oy + view_h, ox : ox + view_w] = view

        right_size = "none" if right_shape is None else f"{right_shape[1]}x{right_shape[0]}"
        lines = [
            f"dataset={self.dataset_dir.name} samples={self._sample_count()}",
            f"target={self.target} label={self.label_query} truth={self.truth_distance_m}",
            f"size left={left_size} right={right_size} | ages left={left_age:.1f}s right={right_age:.1f}s scan={scan_age:.1f}s",
            f"preview={self.preview_status}",
            "v=validate mask  c=capture  q=quit",
        ]
        overlay_h = 112
        y0 = canvas_h - overlay_h
        overlay = panel.copy()
        cv2.rectangle(overlay, (0, y0), (canvas_w, canvas_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.58, panel, 0.42, 0, panel)
        y = y0 + 24
        for line in lines:
            cv2.putText(panel, line[:130], (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (235, 235, 235), 1, cv2.LINE_AA)
            y += 22
        return panel


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect object depth benchmark samples.")
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT, type=Path)
    parser.add_argument("--dataset-name", default="", help="Existing or new dataset name.")
    parser.add_argument("--target", default="chair")
    parser.add_argument("--label-query", default="")
    parser.add_argument("--truth-distance-m", type=float)
    parser.add_argument("--note", default="")
    parser.add_argument("--left-topic", default="/stereo/left/image_raw")
    parser.add_argument("--right-topic", default="/stereo/right/image_raw")
    parser.add_argument("--scan-topic", default="/scan")
    parser.add_argument("--grounding-model-dir", default=DEFAULT_GROUNDING_MODEL_DIR, type=Path)
    parser.add_argument("--grounding-model-id", default=DEFAULT_GROUNDING_MODEL_ID)
    parser.add_argument("--grounding-device", default="GPU")
    parser.add_argument("--sam-device", default="GPU")
    parser.add_argument("--sam-encoder-device", default="")
    parser.add_argument("--sam-decoder-device", default="CPU")
    parser.add_argument("--sam-encoder-xml", default=DEFAULT_SAM_ENCODER_XML, type=Path)
    parser.add_argument("--sam-decoder-xml", default=DEFAULT_SAM_DECODER_XML, type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.dataset_root = args.dataset_root.expanduser().resolve()
    if not args.dataset_name:
        args.dataset_name = "object_depth_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    rclpy.init()
    node = DatasetCollector(args)
    win = "Object Depth Dataset Collector"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 960, 600)
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.01)
            cv2.imshow(win, node.draw_preview())
            key = cv2.waitKey(30) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("v"):
                node.validate_current_frame()
            if key == ord("c"):
                node.capture()
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
