"""Estimate object depth from a learned stereo disparity model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from .run_stereo_object_depth import (
    RESAMPLING,
    colorize_depth,
    load_calibration,
    load_json,
    normalize_depth,
    optical_to_project,
    project_camera_to_base,
    project_points_from_disparity,
    robust_stats,
    save_outputs,
)


DEFAULT_MODEL_DIR = Path.home() / "caragent_ws" / "models" / "hitnet_openvino"


def find_model_file(model_dir: Path) -> Path:
    candidates = [
        model_dir / "openvino_model.xml",
        model_dir / "model.xml",
        model_dir / "model_float32.xml",
        model_dir / "model_float16.xml",
        model_dir / "model_float32.onnx",
        model_dir / "model.onnx",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    for pattern in ("*.xml", "*.onnx"):
        found = sorted(model_dir.rglob(pattern))
        if found:
            return found[0]
    raise FileNotFoundError(f"No .xml or .onnx model found under {model_dir}")


def input_hw_from_shape(shape: list[Any]) -> tuple[int | None, int | None]:
    if len(shape) < 4:
        return None, None
    h_raw, w_raw = shape[-2], shape[-1]
    try:
        h = int(h_raw)
    except Exception:
        h = None
    try:
        w = int(w_raw)
    except Exception:
        w = None
    return h, w


def prepare_hitnet_input(left: np.ndarray, right: np.ndarray, input_shape: list[Any], model_type: str) -> np.ndarray:
    in_h, in_w = input_hw_from_shape(input_shape)
    if not in_h or not in_w:
        in_h, in_w = left.shape[:2]
    left_r = cv2.resize(left, (in_w, in_h), interpolation=cv2.INTER_LINEAR)
    right_r = cv2.resize(right, (in_w, in_h), interpolation=cv2.INTER_LINEAR)
    if model_type == "eth3d":
        left_r = cv2.cvtColor(left_r, cv2.COLOR_BGR2GRAY)[..., None]
        right_r = cv2.cvtColor(right_r, cv2.COLOR_BGR2GRAY)[..., None]
    else:
        left_r = cv2.cvtColor(left_r, cv2.COLOR_BGR2RGB)
        right_r = cv2.cvtColor(right_r, cv2.COLOR_BGR2RGB)
    tensor = np.concatenate([left_r, right_r], axis=-1).astype(np.float32) / 255.0
    return tensor.transpose(2, 0, 1)[None, ...]


def run_openvino_disparity(
    model_path: Path,
    left: np.ndarray,
    right: np.ndarray,
    device: str,
    model_type: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    import openvino as ov

    core = ov.Core()
    model = core.read_model(str(model_path))
    compiled = core.compile_model(model, device)
    input_port = compiled.inputs[0]
    input_shape = list(input_port.partial_shape.get_min_shape())
    tensor = prepare_hitnet_input(left, right, input_shape, model_type)
    outputs = compiled({input_port.get_any_name(): tensor})
    raw = next(iter(outputs.values()))
    disparity = np.squeeze(np.asarray(raw)).astype(np.float32)
    if disparity.ndim == 3:
        disparity = disparity[0]
    meta = {
        "backend": "openvino",
        "device": device,
        "model_path": str(model_path),
        "model_type": model_type,
        "input_shape": input_shape,
        "output_shape": list(np.asarray(raw).shape),
    }
    return disparity, meta


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Estimate object depth with a learned stereo disparity model.")
    parser.add_argument("--left-image", required=True, type=Path)
    parser.add_argument("--right-image", required=True, type=Path)
    parser.add_argument("--segmentation-json", required=True, type=Path)
    parser.add_argument("--calib-file", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR, type=Path)
    parser.add_argument("--model-file", default="", type=Path)
    parser.add_argument("--device", default="GPU")
    parser.add_argument("--model-type", default="eth3d", choices=["eth3d", "middlebury", "flyingthings"])
    parser.add_argument("--min-depth", default=0.15, type=float)
    parser.add_argument("--max-depth", default=8.0, type=float)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    left = cv2.imread(str(args.left_image), cv2.IMREAD_COLOR)
    right = cv2.imread(str(args.right_image), cv2.IMREAD_COLOR)
    if left is None:
        raise FileNotFoundError(args.left_image)
    if right is None:
        raise FileNotFoundError(args.right_image)
    if left.shape[:2] != right.shape[:2]:
        raise ValueError(f"left/right shape mismatch: {left.shape} vs {right.shape}")

    h, w = left.shape[:2]
    calib = load_calibration(args.calib_file.resolve(), (w, h))
    rect_l = cv2.remap(left, calib["map_lx"], calib["map_ly"], cv2.INTER_LINEAR)
    rect_r = cv2.remap(right, calib["map_rx"], calib["map_ry"], cv2.INTER_LINEAR)

    model_path = args.model_file.expanduser() if str(args.model_file) else find_model_file(args.model_dir.expanduser())
    disparity_small, model_meta = run_openvino_disparity(model_path, rect_l, rect_r, args.device, args.model_type)
    disparity = cv2.resize(disparity_small, (w, h), interpolation=cv2.INTER_LINEAR)
    scale_x = float(disparity_small.shape[1]) / float(w)
    disparity = disparity / max(scale_x, 1e-6)
    disparity[~np.isfinite(disparity) | (disparity <= 0.5)] = np.nan

    seg = load_json(args.segmentation_json.resolve())
    mask_path = Path(seg["mask_path"]).resolve()
    mask_src = np.asarray(Image.open(mask_path).convert("L")) > 0
    if mask_src.shape != (h, w):
        mask_src = np.asarray(Image.fromarray(mask_src.astype(np.uint8) * 255).resize((w, h), RESAMPLING.NEAREST)) > 0
    mask = cv2.remap((mask_src.astype(np.uint8) * 255), calib["map_lx"], calib["map_ly"], cv2.INTER_NEAREST) > 0

    points_opt, valid_disp = project_points_from_disparity(disparity, calib["P1"], calib["baseline_m"])
    points_cam = optical_to_project(points_opt)
    points_base = project_camera_to_base(points_cam)
    depth_m = points_cam[..., 0]
    valid_depth = valid_disp & (depth_m >= args.min_depth) & (depth_m <= args.max_depth)
    object_valid = mask & valid_depth
    if not object_valid.any():
        raise RuntimeError("No valid learned-stereo depth pixels inside object mask.")

    obj_cam = points_cam[object_valid]
    obj_base = points_base[object_valid]
    valid_ratio = float(object_valid.sum() / max(1, mask.sum()))
    cam_stats = {
        "x_forward_m": robust_stats(obj_cam[:, 0]),
        "y_left_m": robust_stats(obj_cam[:, 1]),
        "z_up_m": robust_stats(obj_cam[:, 2]),
    }
    base_stats = {
        "x_forward_m": robust_stats(obj_base[:, 0]),
        "y_left_m": robust_stats(obj_base[:, 1]),
        "z_up_m": robust_stats(obj_base[:, 2]),
    }
    result = {
        "left_image": str(args.left_image.resolve()),
        "right_image": str(args.right_image.resolve()),
        "segmentation_json": str(args.segmentation_json.resolve()),
        "calib_file": str(args.calib_file.resolve()),
        "image_size": [w, h],
        "source_detection": seg.get("source_detection"),
        "learned_stereo": {
            **model_meta,
            "baseline_m": calib["baseline_m"],
            "P1": calib["P1"].astype(float).tolist(),
            "disparity_resize_scale_x": scale_x,
        },
        "mask": {
            "source_area_px": int(mask_src.sum()),
            "area_px": int(mask.sum()),
            "valid_stereo_pixels": int(object_valid.sum()),
            "valid_ratio": valid_ratio,
            "rectified": True,
        },
        "object_camera_project": {
            "median_xyz_m": np.median(obj_cam, axis=0).astype(float).tolist(),
            "mean_xyz_m": np.mean(obj_cam, axis=0).astype(float).tolist(),
            "stats": cam_stats,
            "height_m_p05_p95": float(cam_stats["z_up_m"]["p95"] - cam_stats["z_up_m"]["p05"]),
        },
        "object_base": {
            "median_xyz_m": np.median(obj_base, axis=0).astype(float).tolist(),
            "mean_xyz_m": np.mean(obj_base, axis=0).astype(float).tolist(),
            "stats": base_stats,
            "height_m_p05_p95": float(base_stats["z_up_m"]["p95"] - base_stats["z_up_m"]["p05"]),
        },
    }
    save_outputs(
        args.output_dir.resolve(),
        args.left_image.stem + "_learned",
        rect_l,
        disparity,
        depth_m,
        mask,
        valid_depth,
        seg.get("source_detection") or {},
        result,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
