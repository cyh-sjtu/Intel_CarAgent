"""Run monocular depth estimation with Depth Anything V2.

The output depth is relative by default. For robot navigation, use it as a
shape/ordering signal and align scale later with LiDAR, stereo, or ground-plane
constraints.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


DEFAULT_MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"
RESAMPLING = getattr(Image, "Resampling", Image)


def _normalize_depth(depth: np.ndarray) -> np.ndarray:
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


def _colorize_depth(depth_u8: np.ndarray) -> np.ndarray:
    try:
        import cv2

        return cv2.applyColorMap(depth_u8, cv2.COLORMAP_INFERNO)[:, :, ::-1]
    except Exception:
        # Simple fallback: blue-to-yellow ramp without requiring OpenCV.
        x = depth_u8.astype(np.float32) / 255.0
        r = np.clip(255.0 * (1.5 * x), 0, 255)
        g = np.clip(255.0 * (1.5 - np.abs(2.0 * x - 1.0) * 1.5), 0, 255)
        b = np.clip(255.0 * (1.5 * (1.0 - x)), 0, 255)
        return np.stack([r, g, b], axis=-1).astype(np.uint8)


def _depth_to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, Image.Image):
        return np.array(value)
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    elif hasattr(value, "cpu"):
        value = value.cpu().numpy()
    elif hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def run_depth(image_path: Path, model_id: str, device: str) -> tuple[np.ndarray, dict[str, Any]]:
    import torch
    from transformers import pipeline

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe_device = 0 if device == "cuda" else -1
    print(f"loading depth model: {model_id}", flush=True)
    pipe = pipeline(task="depth-estimation", model=model_id, device=pipe_device)

    image = Image.open(image_path).convert("RGB")
    print(f"running depth inference on {device}", flush=True)
    result = pipe(image)

    if "predicted_depth" in result:
        depth = _depth_to_numpy(result["predicted_depth"])
    elif "depth" in result:
        depth = _depth_to_numpy(result["depth"])
    else:
        raise RuntimeError(f"Unexpected depth result keys: {sorted(result.keys())}")

    depth = np.squeeze(depth).astype(np.float32)
    if depth.shape != (image.height, image.width):
        depth_img = Image.fromarray(depth)
        depth_img = depth_img.resize((image.width, image.height), resample=RESAMPLING.BICUBIC)
        depth = np.array(depth_img, dtype=np.float32)

    metadata = {
        "image": str(image_path),
        "image_size": [image.width, image.height],
        "model_id": model_id,
        "device": device,
        "depth_shape": list(depth.shape),
        "depth_min": float(np.nanmin(depth)),
        "depth_max": float(np.nanmax(depth)),
        "depth_mean": float(np.nanmean(depth)),
        "note": "Depth Anything V2 output is relative unless using a metric fine-tuned model.",
    }
    return depth, metadata


def save_depth_outputs(depth: np.ndarray, metadata: dict[str, Any], output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    depth_u8 = _normalize_depth(depth)
    depth_color = _colorize_depth(depth_u8)

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
    parser = argparse.ArgumentParser(description="Run Depth Anything V2 on one image.")
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--output-dir", default=Path("outputs"), type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    image_path = args.image.resolve()
    if not image_path.exists():
        raise FileNotFoundError(image_path)

    depth, metadata = run_depth(image_path=image_path, model_id=args.model_id, device=args.device)
    save_depth_outputs(depth, metadata, args.output_dir.resolve(), image_path.stem)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

