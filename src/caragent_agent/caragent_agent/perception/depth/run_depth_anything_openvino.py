"""Run Depth Anything V2 OpenVINO IR on one image."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


DEFAULT_MODEL_DIR = Path("/home/car/caragent_ws/models/depth_anything_v2_openvino")
DEFAULT_OUTPUT_DIR = Path("/home/car/caragent_ws/perception_outputs/depth_anything")
RESAMPLING = getattr(Image, "Resampling", Image)


def normalize_depth(depth: np.ndarray) -> np.ndarray:
    depth = depth.astype(np.float32)
    valid = np.isfinite(depth)
    if not valid.any():
        return np.zeros_like(depth, dtype=np.uint8)
    lo = float(np.percentile(depth[valid], 2.0))
    hi = float(np.percentile(depth[valid], 98.0))
    if hi <= lo:
        hi = float(depth[valid].max())
        lo = float(depth[valid].min())
    if hi <= lo:
        return np.zeros_like(depth, dtype=np.uint8)
    norm = np.clip((depth - lo) / (hi - lo), 0.0, 1.0)
    return (norm * 255.0).astype(np.uint8)


def colorize_depth(depth_u8: np.ndarray) -> np.ndarray:
    try:
        import cv2

        return cv2.applyColorMap(depth_u8, cv2.COLORMAP_INFERNO)[:, :, ::-1]
    except Exception:
        x = depth_u8.astype(np.float32) / 255.0
        r = np.clip(255.0 * (1.5 * x), 0, 255)
        g = np.clip(255.0 * (1.5 - np.abs(2.0 * x - 1.0) * 1.5), 0, 255)
        b = np.clip(255.0 * (1.5 * (1.0 - x)), 0, 255)
        return np.stack([r, g, b], axis=-1).astype(np.uint8)


class DepthAnythingOpenVINO:
    def __init__(self, model_dir: str | Path = DEFAULT_MODEL_DIR, device: str = "CPU") -> None:
        import openvino as ov
        from transformers import AutoImageProcessor

        self.model_dir = Path(model_dir)
        self.device = device
        self.model_xml = self.model_dir / "openvino_model.xml"
        if not self.model_xml.exists():
            raise FileNotFoundError(self.model_xml)
        self.processor = AutoImageProcessor.from_pretrained(self.model_dir)
        core = ov.Core()
        self.compiled_model = core.compile_model(str(self.model_xml), device)
        self.input_name = self.compiled_model.input(0).get_any_name()
        self.output_port = self.compiled_model.output(0)

    def predict(self, image: Image.Image) -> tuple[np.ndarray, float]:
        inputs = self.processor(images=image, return_tensors="np")
        pixel_values = np.asarray(inputs["pixel_values"], dtype=np.float32)
        start = time.perf_counter()
        outputs = self.compiled_model({self.input_name: pixel_values})
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        depth = np.squeeze(outputs[self.output_port]).astype(np.float32)
        if depth.shape != (image.height, image.width):
            depth_img = Image.fromarray(depth)
            depth_img = depth_img.resize((image.width, image.height), resample=RESAMPLING.BICUBIC)
            depth = np.asarray(depth_img, dtype=np.float32)
        return depth, elapsed_ms


def save_depth_outputs(depth: np.ndarray, metadata: dict[str, Any], output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    depth_u8 = normalize_depth(depth)
    depth_color = colorize_depth(depth_u8)
    gray_path = output_dir / f"{stem}_depth_gray.png"
    color_path = output_dir / f"{stem}_depth_color.png"
    npy_path = output_dir / f"{stem}_depth.npy"
    json_path = output_dir / f"{stem}_depth.json"
    Image.fromarray(depth_u8).save(gray_path)
    Image.fromarray(depth_color).save(color_path)
    np.save(npy_path, depth)
    payload = {
        "metadata": metadata,
        "gray_path": str(gray_path),
        "color_path": str(color_path),
        "npy_path": str(npy_path),
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"gray: {gray_path}")
    print(f"color: {color_path}")
    print(f"npy: {npy_path}")
    print(f"json: {json_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Depth Anything V2 with OpenVINO.")
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR, type=Path)
    parser.add_argument("--device", default="CPU")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    image_path = args.image.resolve()
    image = Image.open(image_path).convert("RGB")
    detector = DepthAnythingOpenVINO(args.model_dir, device=args.device)
    depth, elapsed_ms = detector.predict(image)
    metadata = {
        "backend": "openvino",
        "device": args.device,
        "model_dir": str(args.model_dir),
        "image": str(image_path),
        "image_size": [image.width, image.height],
        "depth_shape": list(depth.shape),
        "depth_min": float(np.nanmin(depth)),
        "depth_max": float(np.nanmax(depth)),
        "depth_mean": float(np.nanmean(depth)),
        "elapsed_ms": elapsed_ms,
        "note": "Depth Anything V2 output is relative unless using a metric fine-tuned model.",
    }
    print(f"image: {image_path}")
    print(f"elapsed_ms: {elapsed_ms:.2f}")
    print(f"depth_shape: {metadata['depth_shape']}")
    print(f"depth_min/max: {metadata['depth_min']:.4f}/{metadata['depth_max']:.4f}")
    save_depth_outputs(depth, metadata, args.output_dir.resolve(), image_path.stem)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
