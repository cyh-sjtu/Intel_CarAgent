"""Estimate object-level 3D geometry from stereo disparity and a SAM mask."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw


RESAMPLING = getattr(Image, "Resampling", Image)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_calibration(calib_file: Path, image_size: tuple[int, int]) -> dict[str, Any]:
    data = np.load(calib_file)
    mtx_l = data["mtx_l"]
    dist_l = data["dist_l"]
    mtx_r = data["mtx_r"]
    dist_r = data["dist_r"]
    R = data["R"]
    T = data["T"]
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        mtx_l,
        dist_l,
        mtx_r,
        dist_r,
        image_size,
        R,
        T,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0,
    )
    map_lx, map_ly = cv2.initUndistortRectifyMap(
        mtx_l, dist_l, R1, P1, image_size, cv2.CV_32FC1
    )
    map_rx, map_ry = cv2.initUndistortRectifyMap(
        mtx_r, dist_r, R2, P2, image_size, cv2.CV_32FC1
    )
    return {
        "mtx_l": mtx_l,
        "dist_l": dist_l,
        "mtx_r": mtx_r,
        "dist_r": dist_r,
        "R": R,
        "T": T,
        "R1": R1,
        "R2": R2,
        "P1": P1,
        "P2": P2,
        "Q": Q,
        "map_lx": map_lx,
        "map_ly": map_ly,
        "map_rx": map_rx,
        "map_ry": map_ry,
        "baseline_m": float(np.linalg.norm(T)),
    }


def make_matcher(num_disparities: int, block_size: int) -> cv2.StereoMatcher:
    num_disparities = int(np.ceil(num_disparities / 16.0) * 16)
    block_size = int(block_size)
    if block_size % 2 == 0:
        block_size += 1
    return cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=num_disparities,
        blockSize=block_size,
        P1=8 * 3 * block_size**2,
        P2=32 * 3 * block_size**2,
        disp12MaxDiff=1,
        uniquenessRatio=8,
        speckleWindowSize=80,
        speckleRange=2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )


def normalize_depth(depth: np.ndarray) -> np.ndarray:
    valid = np.isfinite(depth) & (depth > 0)
    if not valid.any():
        return np.zeros(depth.shape, dtype=np.uint8)
    lo, hi = np.percentile(depth[valid], [2, 98])
    if hi <= lo:
        return np.zeros(depth.shape, dtype=np.uint8)
    norm = np.clip((depth - lo) / (hi - lo), 0.0, 1.0)
    norm[~valid] = 0.0
    return (norm * 255).astype(np.uint8)


def colorize_depth(depth: np.ndarray) -> np.ndarray:
    depth_u8 = normalize_depth(depth)
    try:
        return cv2.applyColorMap(depth_u8, cv2.COLORMAP_INFERNO)[:, :, ::-1]
    except Exception:
        return np.stack([depth_u8, depth_u8, depth_u8], axis=-1)


def project_points_from_disparity(disparity: np.ndarray, P1: np.ndarray, baseline_m: float) -> tuple[np.ndarray, np.ndarray]:
    fx = float(P1[0, 0])
    cx = float(P1[0, 2])
    cy = float(P1[1, 2])
    valid = np.isfinite(disparity) & (disparity > 0.5)
    z = np.zeros_like(disparity, dtype=np.float32)
    z[valid] = fx * baseline_m / disparity[valid]
    h, w = disparity.shape
    uu, vv = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    x_opt = np.zeros_like(z, dtype=np.float32)
    y_opt = np.zeros_like(z, dtype=np.float32)
    x_opt[valid] = (uu[valid] - cx) * z[valid] / fx
    y_opt[valid] = (vv[valid] - cy) * z[valid] / fx
    points_opt = np.stack([x_opt, y_opt, z], axis=-1)
    return points_opt, valid


def optical_to_project(points_opt: np.ndarray) -> np.ndarray:
    # optical: +X right, +Y down, +Z forward
    # project/base-style camera: +X forward, +Y left, +Z up
    x = points_opt[..., 2]
    y = -points_opt[..., 0]
    z = -points_opt[..., 1]
    return np.stack([x, y, z], axis=-1)


def project_camera_to_base(points_cam: np.ndarray) -> np.ndarray:
    # URDF currently models camera_left axes aligned with base_link.
    out = points_cam.copy()
    out[..., 0] += 0.30
    out[..., 1] += 0.03
    out[..., 2] += 0.185
    return out


def robust_stats(values: np.ndarray) -> dict[str, float]:
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {}
    return {
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


def save_outputs(
    output_dir: Path,
    stem: str,
    rect_l: np.ndarray,
    disparity: np.ndarray,
    depth_m: np.ndarray,
    mask: np.ndarray,
    valid_mask: np.ndarray,
    detection: dict[str, Any],
    result: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rect_path = output_dir / f"{stem}_stereo_rect_left.png"
    disp_path = output_dir / f"{stem}_stereo_disparity.png"
    depth_color_path = output_dir / f"{stem}_stereo_depth_color.png"
    overlay_path = output_dir / f"{stem}_stereo_object_overlay.png"
    depth_npy_path = output_dir / f"{stem}_stereo_depth_m.npy"
    mask_rect_path = output_dir / f"{stem}_stereo_rect_mask.png"
    json_path = output_dir / f"{stem}_stereo_object_3d.json"

    Image.fromarray(cv2.cvtColor(rect_l, cv2.COLOR_BGR2RGB)).save(rect_path)
    disp_u8 = normalize_depth(disparity)
    Image.fromarray(disp_u8).save(disp_path)
    Image.fromarray(colorize_depth(depth_m)).save(depth_color_path)
    Image.fromarray((mask.astype(np.uint8) * 255)).save(mask_rect_path)
    np.save(depth_npy_path, depth_m)

    overlay = cv2.cvtColor(rect_l, cv2.COLOR_BGR2RGB).copy()
    green = np.array([0, 255, 120], dtype=np.uint8)
    overlay[mask] = (overlay[mask] * 0.58 + green * 0.42).astype(np.uint8)
    red = np.array([255, 70, 70], dtype=np.uint8)
    overlay[mask & ~valid_mask] = (overlay[mask & ~valid_mask] * 0.55 + red * 0.45).astype(np.uint8)
    overlay_img = Image.fromarray(overlay)
    draw = ImageDraw.Draw(overlay_img)
    # Use rectified mask bbox (detection box is in original unrectified coords)
    ys, xs = np.where(mask)
    if len(xs) > 0:
        draw.rectangle([int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())], outline=(255, 70, 70), width=3)
    center = result.get("object_camera_project", {}).get("median_xyz_m")
    text = "stereo object"
    if center:
        text = f"x={center[0]:.2f} y={center[1]:.2f} z={center[2]:.2f}m"
    draw.rectangle([4, 4, min(overlay_img.width - 1, 4 + len(text) * 8), 25], fill=(0, 0, 0))
    draw.text((8, 8), text, fill=(255, 255, 255))
    overlay_img.save(overlay_path)

    result.update(
        {
            "rect_left_path": str(rect_path),
            "disparity_path": str(disp_path),
            "depth_color_path": str(depth_color_path),
            "depth_npy_path": str(depth_npy_path),
            "rectified_mask_path": str(mask_rect_path),
            "overlay_path": str(overlay_path),
        }
    )
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"rect: {rect_path}")
    print(f"disparity: {disp_path}")
    print(f"depth: {depth_color_path}")
    print(f"overlay: {overlay_path}")
    print(f"npy: {depth_npy_path}")
    print(f"json: {json_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Estimate object 3D from stereo disparity inside SAM mask.")
    parser.add_argument("--left-image", required=True, type=Path)
    parser.add_argument("--right-image", required=True, type=Path)
    parser.add_argument("--segmentation-json", required=True, type=Path)
    parser.add_argument("--calib-file", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--num-disparities", default=96, type=int)
    parser.add_argument("--block-size", default=5, type=int)
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

    gray_l = cv2.cvtColor(rect_l, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(rect_r, cv2.COLOR_BGR2GRAY)
    matcher = make_matcher(args.num_disparities, args.block_size)
    disparity = matcher.compute(gray_l, gray_r).astype(np.float32) / 16.0
    disparity[disparity <= 0] = np.nan

    seg = load_json(args.segmentation_json.resolve())
    mask_path = Path(seg["mask_path"]).resolve()
    mask_src = np.asarray(Image.open(mask_path).convert("L")) > 0
    if mask_src.shape != (h, w):
        mask_src = np.asarray(Image.fromarray(mask_src.astype(np.uint8) * 255).resize((w, h), RESAMPLING.NEAREST)) > 0
    mask = cv2.remap(
        (mask_src.astype(np.uint8) * 255),
        calib["map_lx"],
        calib["map_ly"],
        cv2.INTER_NEAREST,
    ) > 0

    points_opt, valid_disp = project_points_from_disparity(disparity, calib["P1"], calib["baseline_m"])
    points_cam = optical_to_project(points_opt)
    points_base = project_camera_to_base(points_cam)
    depth_m = points_cam[..., 0]
    valid_depth = valid_disp & (depth_m >= args.min_depth) & (depth_m <= args.max_depth)
    object_valid = mask & valid_depth
    if not object_valid.any():
        raise RuntimeError("No valid stereo depth pixels inside object mask.")

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
        "stereo": {
            "baseline_m": calib["baseline_m"],
            "P1": calib["P1"].astype(float).tolist(),
            "num_disparities": int(args.num_disparities),
            "block_size": int(args.block_size),
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
    print(f"valid stereo pixels in mask: {object_valid.sum()} / {mask.sum()} ({valid_ratio:.3f})")
    print(f"object camera median xyz: {result['object_camera_project']['median_xyz_m']}")
    print(f"object base median xyz: {result['object_base']['median_xyz_m']}")
    print(f"object height p05-p95: {result['object_base']['height_m_p05_p95']:.3f} m")
    save_outputs(
        args.output_dir.resolve(),
        args.left_image.stem,
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
