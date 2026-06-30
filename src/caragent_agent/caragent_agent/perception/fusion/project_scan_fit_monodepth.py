"""Project 2D LaserScan into the left image and fit metric scale for mono depth."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw


RESAMPLING = getattr(Image, "Resampling", Image)


LASER_X_M = 0.12
LASER_Y_M = 0.0
LASER_Z_M = 0.30
LASER_YAW_RAD = math.pi

CAMERA_LEFT_X_M = 0.30
CAMERA_LEFT_Y_M = 0.03
CAMERA_LEFT_Z_M = 0.185

DEFAULT_EXTRINSICS_JSON = (
    Path(__file__).resolve().parents[5]
    / "calibration"
    / "lidar_camera"
    / "lidar_camera_extrinsics_calibrated.json"
)


@dataclass(frozen=True)
class Extrinsics:
    laser_x_m: float = LASER_X_M
    laser_y_m: float = LASER_Y_M
    laser_z_m: float = LASER_Z_M
    laser_roll_rad: float = 0.0
    laser_pitch_rad: float = 0.0
    laser_yaw_rad: float = LASER_YAW_RAD
    camera_x_m: float = CAMERA_LEFT_X_M
    camera_y_m: float = CAMERA_LEFT_Y_M
    camera_z_m: float = CAMERA_LEFT_Z_M
    camera_roll_rad: float = 0.0
    camera_pitch_rad: float = 0.0
    camera_yaw_rad: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "laser_x_m": self.laser_x_m,
            "laser_y_m": self.laser_y_m,
            "laser_z_m": self.laser_z_m,
            "laser_roll_rad": self.laser_roll_rad,
            "laser_pitch_rad": self.laser_pitch_rad,
            "laser_yaw_rad": self.laser_yaw_rad,
            "camera_x_m": self.camera_x_m,
            "camera_y_m": self.camera_y_m,
            "camera_z_m": self.camera_z_m,
            "camera_roll_rad": self.camera_roll_rad,
            "camera_pitch_rad": self.camera_pitch_rad,
            "camera_yaw_rad": self.camera_yaw_rad,
        }


def rodrigues_xyz(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    return rz @ ry @ rx


def load_extrinsics(extrinsics_json: Path | None) -> tuple[Extrinsics, str | None]:
    path = extrinsics_json
    if path is None and DEFAULT_EXTRINSICS_JSON.exists():
        path = DEFAULT_EXTRINSICS_JSON
    if path is None:
        return Extrinsics(), None

    data = json.loads(path.read_text(encoding="utf-8"))
    params = data.get("optimized_params", data)
    defaults = Extrinsics().as_dict()
    values = {key: float(params.get(key, value)) for key, value in defaults.items()}
    return Extrinsics(**values), str(path.resolve())


def load_calib(calib_file: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(calib_file)
    return data["mtx_l"].astype(np.float64), data["dist_l"].astype(np.float64)


def scan_points_in_base(scan_file: Path, extrinsics: Extrinsics) -> tuple[np.ndarray, np.ndarray]:
    scan = np.load(scan_file)
    ranges = scan["ranges"].astype(np.float32)
    angle_min = float(scan["angle_min"])
    angle_increment = float(scan["angle_increment"])
    range_min = float(scan["range_min"])
    range_max = float(scan["range_max"])
    angles_laser = angle_min + np.arange(len(ranges), dtype=np.float32) * angle_increment
    valid = np.isfinite(ranges) & (ranges >= range_min) & (ranges <= range_max)

    x_l = ranges[valid] * np.cos(angles_laser[valid])
    y_l = ranges[valid] * np.sin(angles_laser[valid])
    z_l = np.zeros_like(x_l)

    points_laser = np.stack([x_l, y_l, z_l], axis=1).astype(np.float64)
    r_lidar_to_base = rodrigues_xyz(
        extrinsics.laser_roll_rad,
        extrinsics.laser_pitch_rad,
        extrinsics.laser_yaw_rad,
    )
    t_lidar_to_base = np.array(
        [extrinsics.laser_x_m, extrinsics.laser_y_m, extrinsics.laser_z_m],
        dtype=np.float64,
    )
    points_base = (t_lidar_to_base + points_laser @ r_lidar_to_base.T).astype(np.float32)
    return points_base, ranges[valid].astype(np.float32)


def base_to_left_optical(points_base: np.ndarray, extrinsics: Extrinsics) -> np.ndarray:
    camera_t = np.array(
        [extrinsics.camera_x_m, extrinsics.camera_y_m, extrinsics.camera_z_m],
        dtype=np.float64,
    )
    points_camera_project = points_base.astype(np.float64) - camera_t
    r_camera_to_base = rodrigues_xyz(
        extrinsics.camera_roll_rad,
        extrinsics.camera_pitch_rad,
        extrinsics.camera_yaw_rad,
    )
    points_camera_project = points_camera_project @ r_camera_to_base

    # base/project camera: +X forward, +Y left, +Z up
    # OpenCV optical camera: +X right, +Y down, +Z forward
    x_opt = -points_camera_project[:, 1]
    y_opt = -points_camera_project[:, 2]
    z_opt = points_camera_project[:, 0]
    return np.stack([x_opt, y_opt, z_opt], axis=1).astype(np.float32)


def project_points(points_opt: np.ndarray, mtx: np.ndarray, dist: np.ndarray) -> np.ndarray:
    rvec = np.zeros((3, 1), dtype=np.float64)
    tvec = np.zeros((3, 1), dtype=np.float64)
    projected, _ = cv2.projectPoints(points_opt.astype(np.float64), rvec, tvec, mtx, dist)
    return projected.reshape(-1, 2).astype(np.float32)


def sample_depth(depth: np.ndarray, uv: np.ndarray) -> np.ndarray:
    h, w = depth.shape
    u = np.clip(np.round(uv[:, 0]).astype(np.int32), 0, w - 1)
    v = np.clip(np.round(uv[:, 1]).astype(np.int32), 0, h - 1)
    return depth[v, u].astype(np.float32)


def sample_bool_mask(mask: np.ndarray, uv: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    u = np.clip(np.round(uv[:, 0]).astype(np.int32), 0, w - 1)
    v = np.clip(np.round(uv[:, 1]).astype(np.int32), 0, h - 1)
    return mask[v, u].astype(bool)


FIT_MODE_CHOICES = ("linear", "inverse", "log", "sqrt", "quadratic")
DEFAULT_FIT_MODES = "log,quadratic"


def _feature_stack(x: np.ndarray, mode: str) -> np.ndarray:
    values = np.asarray(x, dtype=np.float64)
    safe = np.maximum(values, 1e-6)
    ones = np.ones_like(values, dtype=np.float64)
    if mode == "linear":
        features = [values, ones]
    elif mode == "inverse":
        features = [1.0 / safe, ones]
    elif mode == "log":
        features = [np.log1p(safe), ones]
    elif mode == "sqrt":
        features = [np.sqrt(safe), ones]
    elif mode == "quadratic":
        features = [values * values, values, ones]
    else:
        raise ValueError(f"Unsupported fit mode: {mode}")
    return np.stack(features, axis=-1)


def _num_fit_params(mode: str) -> int:
    return int(_feature_stack(np.array([1.0], dtype=np.float64), mode).shape[-1])


def fit_model(x: np.ndarray, y: np.ndarray, mode: str) -> np.ndarray:
    features = _feature_stack(x, mode)
    a_mat = features.reshape(-1, features.shape[-1])
    coef, *_ = np.linalg.lstsq(a_mat, np.asarray(y, dtype=np.float64).reshape(-1), rcond=None)
    return coef.astype(np.float64)


def predict_model(x: np.ndarray, params: np.ndarray, mode: str) -> np.ndarray:
    features = _feature_stack(x, mode)
    return np.tensordot(features, np.asarray(params, dtype=np.float64), axes=([-1], [0]))


def fit_formula(mode: str) -> str:
    formulas = {
        "linear": "z = p0 * d + p1",
        "inverse": "z = p0 / d + p1",
        "log": "z = p0 * log1p(d) + p1",
        "sqrt": "z = p0 * sqrt(d) + p1",
        "quadratic": "z = p0 * d^2 + p1 * d + p2",
    }
    return formulas[mode]


def robust_fit(x: np.ndarray, y: np.ndarray, mode: str) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    indices = np.arange(len(x))
    rng = np.random.default_rng(7)
    min_params = _num_fit_params(mode)

    for _ in range(250):
        if len(indices) < max(8, min_params):
            subset = indices
        else:
            subset = rng.choice(indices, size=min(24, len(indices)), replace=False)
        params = fit_model(x[subset], y[subset], mode)
        pred = predict_model(x, params, mode)
        residual = np.abs(pred - y)
        med = float(np.median(residual))
        inliers = residual < max(0.20, 2.5 * med)
        score = int(inliers.sum())
        mae = float(np.mean(residual[inliers])) if score else float("inf")
        if best is None or (score, -mae) > (best["inlier_count"], -best["mae_m"]):
            best = {
                "mode": mode,
                "params": params,
                "inliers": inliers,
                "inlier_count": score,
                "mae_m": mae,
            }

    assert best is not None
    inliers = best["inliers"]
    if int(inliers.sum()) >= min_params:
        params = fit_model(x[inliers], y[inliers], mode)
    else:
        params = np.asarray(best["params"], dtype=np.float64)
    pred = predict_model(x, params, mode)
    residual = np.abs(pred - y)
    inlier_residual = residual[inliers]
    best.update(
        {
            "params": params,
            "formula": fit_formula(mode),
            "inliers": inliers,
            "inlier_count": int(inliers.sum()),
            "mae_m": float(np.mean(inlier_residual)),
            "rmse_m": float(np.sqrt(np.mean(inlier_residual * inlier_residual))),
            "median_abs_error_m": float(np.median(inlier_residual)),
            "p90_abs_error_m": float(np.percentile(inlier_residual, 90)),
            "max_abs_error_m": float(np.max(inlier_residual)),
        }
    )
    if len(params) >= 2:
        best["a"] = float(params[0])
        best["b"] = float(params[1])
    return best


def select_fit(fits: list[dict[str, Any]], p90_tolerance: float = 0.10) -> dict[str, Any]:
    finite_fits = [
        fit
        for fit in fits
        if np.isfinite(fit.get("p90_abs_error_m", np.inf))
        and np.isfinite(fit.get("mae_m", np.inf))
    ]
    if not finite_fits:
        raise RuntimeError("No finite fit results.")
    best_p90 = min(float(fit["p90_abs_error_m"]) for fit in finite_fits)
    p90_limit = best_p90 * (1.0 + max(0.0, float(p90_tolerance)))
    candidates = [
        fit
        for fit in finite_fits
        if float(fit["p90_abs_error_m"]) <= p90_limit + 1e-9
    ]
    return min(
        candidates,
        key=lambda fit: (
            float(fit["mae_m"]),
            float(fit["p90_abs_error_m"]),
            -int(fit["inlier_count"]),
        ),
    )


def build_depth_edge_mask(
    depth: np.ndarray,
    percentile: float,
    dilate_px: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    depth = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(depth)
    if not np.any(valid):
        return np.zeros(depth.shape, dtype=bool), {"available": False, "reason": "no_valid_depth"}

    lo, hi = np.percentile(depth[valid], [2.0, 98.0])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros(depth.shape, dtype=bool), {"available": False, "reason": "flat_depth_range"}

    norm = np.zeros(depth.shape, dtype=np.float32)
    norm[valid] = np.clip((depth[valid] - float(lo)) / max(1e-6, float(hi - lo)), 0.0, 1.0)
    smooth = cv2.GaussianBlur(norm, (5, 5), 0)
    gx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx * gx + gy * gy)
    threshold = float(np.percentile(grad[valid], float(percentile)))
    edge = grad > threshold
    if dilate_px > 1:
        kernel_size = int(dilate_px)
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        edge = cv2.dilate(edge.astype(np.uint8), kernel) > 0
    edge &= valid
    return edge, {
        "available": True,
        "percentile": float(percentile),
        "threshold": threshold,
        "dilate_px": int(dilate_px),
        "edge_pixel_count": int(np.count_nonzero(edge)),
    }


def colorize_metric_depth(depth_m: np.ndarray, max_depth_m: float) -> np.ndarray:
    valid = np.isfinite(depth_m) & (depth_m > 0)
    norm = np.zeros(depth_m.shape, dtype=np.float32)
    norm[valid] = np.clip(depth_m[valid] / max_depth_m, 0.0, 1.0)
    inv = (255 * (1.0 - norm)).astype(np.uint8)
    inv[~valid] = 0
    return cv2.applyColorMap(inv, cv2.COLORMAP_TURBO)[:, :, ::-1]


def load_optional_mask(segmentation_json: Path | None, image_size: tuple[int, int]) -> np.ndarray | None:
    if segmentation_json is None:
        return None
    data = json.loads(segmentation_json.read_text(encoding="utf-8"))
    mask_path = Path(data["mask_path"]).resolve()
    mask = np.asarray(Image.open(mask_path).convert("L"))
    w, h = image_size
    if mask.shape != (h, w):
        mask = np.asarray(Image.fromarray(mask).resize((w, h), RESAMPLING.NEAREST))
    return mask > 0


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


def mask_bbox(mask: np.ndarray) -> list[int]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return []
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def summarize_mask_lidar_support(
    mask: np.ndarray | None,
    raw_uv: np.ndarray,
    usable_uv: np.ndarray,
    inlier_uv: np.ndarray,
    min_points: int,
    min_density: float,
) -> dict[str, Any] | None:
    if mask is None:
        return None
    bbox = mask_bbox(mask)
    area = int(np.count_nonzero(mask))
    if not bbox or area == 0:
        return {
            "available": False,
            "reason": "empty_mask",
            "mask_area_px": area,
            "mask_bbox": bbox,
        }
    width_px = int(bbox[2] - bbox[0] + 1)
    raw_count = int(np.count_nonzero(sample_bool_mask(mask, raw_uv))) if len(raw_uv) else 0
    usable_count = int(np.count_nonzero(sample_bool_mask(mask, usable_uv))) if len(usable_uv) else 0
    inlier_count = int(np.count_nonzero(sample_bool_mask(mask, inlier_uv))) if len(inlier_uv) else 0
    required = max(int(min_points), int(math.ceil(width_px * float(min_density))))
    density = float(usable_count / max(1, width_px))
    return {
        "available": True,
        "mask_area_px": area,
        "mask_bbox": bbox,
        "mask_width_px": width_px,
        "raw_projected_points_in_mask": raw_count,
        "usable_projected_points_in_mask": usable_count,
        "fit_inlier_projected_points_in_mask": inlier_count,
        "points_per_mask_width": density,
        "min_points": int(min_points),
        "min_points_per_mask_width": float(min_density),
        "required_usable_points": int(required),
        "has_support": bool(usable_count >= required),
        "reason": "ok" if usable_count >= required else "too_few_usable_projected_points_in_mask",
    }


def draw_projection(
    image: np.ndarray,
    uv: np.ndarray,
    metric_z: np.ndarray,
    mono_values: np.ndarray,
    inliers: np.ndarray,
    output_path: Path,
    edge_rejected_uv: np.ndarray | None = None,
) -> None:
    img = Image.fromarray(image.copy())
    draw = ImageDraw.Draw(img)
    if edge_rejected_uv is not None:
        for u, v in edge_rejected_uv:
            r = 3
            draw.ellipse([u - r, v - r, u + r, v + r], fill=(175, 80, 220), outline=(0, 0, 0))
    z_min, z_max = np.percentile(metric_z, [5, 95])
    for (u, v), z, mono, ok in zip(uv, metric_z, mono_values, inliers):
        if ok:
            t = float(np.clip((z - z_min) / max(1e-6, z_max - z_min), 0, 1))
            color = (int(255 * t), int(220 * (1 - t)), int(255 * (1 - t)))
            r = 4
        else:
            color = (255, 60, 60)
            r = 2
        draw.ellipse([u - r, v - r, u + r, v + r], fill=color, outline=(0, 0, 0))
    draw.rectangle([4, 4, 520, 30], fill=(0, 0, 0))
    draw.text((10, 10), "LaserScan projected into left image: colored=in fit, red=fit outlier, purple=depth-edge rejected", fill=(255, 255, 255))
    img.save(output_path)


def draw_fit_plot(
    mono_values: np.ndarray,
    metric_z: np.ndarray,
    fit: dict[str, Any],
    all_fits: list[dict[str, Any]],
    output_path: Path,
) -> None:
    w, h = 720, 460
    pad_l, pad_r, pad_t, pad_b = 70, 25, 30, 55
    img = Image.new("RGB", (w, h), (250, 250, 246))
    draw = ImageDraw.Draw(img)
    x_min, x_max = np.percentile(mono_values, [1, 99])
    y_min, y_max = np.percentile(metric_z, [1, 99])
    y_min = min(0.0, float(y_min))
    x_span = max(1e-6, float(x_max - x_min))
    y_span = max(1e-6, float(y_max - y_min))

    def to_px(x: float, y: float) -> tuple[float, float]:
        px = pad_l + (x - x_min) / x_span * (w - pad_l - pad_r)
        py = h - pad_b - (y - y_min) / y_span * (h - pad_t - pad_b)
        return px, py

    draw.rectangle([pad_l, pad_t, w - pad_r, h - pad_b], outline=(80, 80, 75), width=1)
    draw.text((20, 12), "Projected LiDAR metric depth vs monocular relative depth", fill=(20, 20, 20))
    draw.text((w // 2 - 90, h - 35), "mono relative depth value", fill=(30, 30, 30))
    draw.text((8, h // 2 - 12), "Z m", fill=(30, 30, 30))

    xs = np.linspace(float(x_min), float(x_max), 160)
    palette = {
        "linear": (20, 20, 20),
        "inverse": (222, 80, 50),
        "log": (30, 130, 80),
        "sqrt": (95, 70, 190),
        "quadratic": (210, 135, 20),
    }
    for item in all_fits:
        ys = predict_model(xs, np.asarray(item["params"], dtype=np.float64), item["mode"])
        pts = [to_px(float(x), float(y)) for x, y in zip(xs, ys) if np.isfinite(y)]
        if len(pts) < 2:
            continue
        selected = item["mode"] == fit["mode"] and np.allclose(item["params"], fit["params"])
        color = (20, 20, 20) if selected else palette.get(item["mode"], (90, 90, 90))
        draw.line(pts, fill=color, width=4 if selected else 2)

    inliers = fit["inliers"]
    for x, y, ok in zip(mono_values, metric_z, inliers):
        px, py = to_px(float(x), float(y))
        color = (35, 125, 215) if ok else (230, 55, 55)
        r = 3 if ok else 2
        draw.ellipse([px - r, py - r, px + r, py + r], fill=color)

    text = (
        f"best={fit['mode']} MAE={fit['mae_m']:.3f}m "
        f"median={fit['median_abs_error_m']:.3f}m p90={fit['p90_abs_error_m']:.3f}m"
    )
    draw.rectangle([pad_l + 8, pad_t + 8, pad_l + 560, pad_t + 32], fill=(255, 255, 255))
    draw.text((pad_l + 14, pad_t + 12), text, fill=(20, 20, 20))
    legend_y = pad_t + 38
    for item in all_fits[:5]:
        color = (20, 20, 20) if item["mode"] == fit["mode"] else palette.get(item["mode"], (90, 90, 90))
        line = f"{item['mode']}: MAE={item['mae_m']:.3f}, P90={item['p90_abs_error_m']:.3f}"
        draw.text((pad_l + 14, legend_y), line, fill=color)
        legend_y += 15
    img.save(output_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fit metric mono depth scale from projected LaserScan points.")
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--scan", required=True, type=Path)
    parser.add_argument("--mono-depth-npy", required=True, type=Path)
    parser.add_argument("--calib-file", required=True, type=Path)
    parser.add_argument(
        "--extrinsics-json",
        type=Path,
        help=(
            "LiDAR-camera extrinsics JSON. Defaults to "
            f"{DEFAULT_EXTRINSICS_JSON} when that file exists; otherwise uses legacy constants."
        ),
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--segmentation-json", type=Path)
    parser.add_argument("--min-camera-z", default=0.20, type=float)
    parser.add_argument("--max-camera-z", default=6.0, type=float)
    parser.add_argument("--max-output-depth", default=6.0, type=float)
    parser.add_argument(
        "--fit-modes",
        default=DEFAULT_FIT_MODES,
        help=f"Comma-separated fit modes to compare. Choices: {', '.join(FIT_MODE_CHOICES)}",
    )
    parser.add_argument(
        "--selection-p90-tolerance",
        default=0.10,
        type=float,
        help="Select from fits within this fraction of the best P90, then choose lowest MAE.",
    )
    parser.add_argument("--edge-filter", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--edge-percentile", default=90.0, type=float)
    parser.add_argument("--edge-dilate-px", default=5, type=int)
    parser.add_argument("--min-edge-filtered-samples", default=40, type=int)
    parser.add_argument(
        "--mask-lidar-support-check",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Optional diagnostic for projected LiDAR support inside the object mask. Disabled by default.",
    )
    parser.add_argument("--min-mask-lidar-points", default=2, type=int)
    parser.add_argument("--min-mask-lidar-density", default=0.035, type=float)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    image_bgr = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(args.image)
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]

    mono_depth = np.load(args.mono_depth_npy).astype(np.float32)
    if mono_depth.shape != (h, w):
        mono_depth = cv2.resize(mono_depth, (w, h), interpolation=cv2.INTER_LINEAR)

    mtx, dist = load_calib(args.calib_file.resolve())
    extrinsics, extrinsics_source = load_extrinsics(args.extrinsics_json.resolve() if args.extrinsics_json else None)
    points_base, lidar_ranges = scan_points_in_base(args.scan.resolve(), extrinsics)
    points_opt = base_to_left_optical(points_base, extrinsics)
    uv = project_points(points_opt, mtx, dist)
    z_metric = points_opt[:, 2]

    inside = (
        (z_metric >= args.min_camera_z)
        & (z_metric <= args.max_camera_z)
        & (uv[:, 0] >= 0)
        & (uv[:, 0] < w)
        & (uv[:, 1] >= 0)
        & (uv[:, 1] < h)
    )
    uv_in_raw = uv[inside]
    z_in_raw = z_metric[inside]
    mono_in_raw = sample_depth(mono_depth, uv_in_raw)
    finite = np.isfinite(mono_in_raw) & np.isfinite(z_in_raw) & (mono_in_raw > 0)
    uv_finite = uv_in_raw[finite]
    z_finite = z_in_raw[finite]
    mono_finite = mono_in_raw[finite]

    edge_mask, edge_info = build_depth_edge_mask(
        mono_depth,
        args.edge_percentile,
        args.edge_dilate_px,
    )
    if args.edge_filter and edge_info.get("available"):
        on_edge = sample_bool_mask(edge_mask, uv_finite)
    else:
        on_edge = np.zeros(len(uv_finite), dtype=bool)
    keep_by_edge = ~on_edge
    edge_candidate_count = int(np.count_nonzero(on_edge))
    if int(np.count_nonzero(keep_by_edge)) >= int(args.min_edge_filtered_samples):
        uv_in = uv_finite[keep_by_edge]
        z_in = z_finite[keep_by_edge]
        mono_in = mono_finite[keep_by_edge]
        edge_filter_applied = bool(args.edge_filter and edge_info.get("available"))
        edge_rejected_uv = uv_finite[on_edge]
    else:
        uv_in = uv_finite
        z_in = z_finite
        mono_in = mono_finite
        edge_filter_applied = False
        edge_rejected_uv = np.empty((0, 2), dtype=np.float32)
    edge_rejected_count = edge_candidate_count if edge_filter_applied else 0

    if len(mono_in) < 8:
        raise RuntimeError(f"Not enough projected scan/depth samples: {len(mono_in)}")

    fit_modes = [item.strip() for item in args.fit_modes.split(",") if item.strip()]
    unknown_modes = sorted(set(fit_modes) - set(FIT_MODE_CHOICES))
    if unknown_modes:
        raise ValueError(f"Unsupported fit modes: {unknown_modes}. Choices: {FIT_MODE_CHOICES}")
    fits = [robust_fit(mono_in, z_in, mode) for mode in fit_modes]
    best = select_fit(fits, args.selection_p90_tolerance)
    fits = sorted(
        fits,
        key=lambda item: (
            0 if item is best else 1,
            item["p90_abs_error_m"],
            item["mae_m"],
            -item["inlier_count"],
        ),
    )

    metric_depth = predict_model(
        mono_depth,
        np.asarray(best["params"], dtype=np.float64),
        best["mode"],
    )
    metric_depth = metric_depth.astype(np.float32)
    metric_depth[(metric_depth <= 0) | ~np.isfinite(metric_depth)] = np.nan
    metric_valid_mask = np.isfinite(metric_depth) & (metric_depth > 0)

    stem = args.image.stem
    projection_path = output_dir / f"{stem}_scan_projected_to_image.png"
    metric_color_path = output_dir / f"{stem}_mono_metric_depth_color.png"
    metric_npy_path = output_dir / f"{stem}_mono_metric_depth.npy"
    metric_valid_mask_path = output_dir / f"{stem}_mono_metric_depth_valid.png"
    fit_plot_path = output_dir / f"{stem}_scan_monodepth_fit_plot.png"
    json_path = output_dir / f"{stem}_scan_monodepth_fit.json"

    draw_projection(image, uv_in, z_in, mono_in, best["inliers"], projection_path, edge_rejected_uv)
    draw_fit_plot(mono_in, z_in, best, fits, fit_plot_path)
    Image.fromarray(colorize_metric_depth(metric_depth, args.max_output_depth)).save(metric_color_path)
    Image.fromarray((metric_valid_mask.astype(np.uint8) * 255)).save(metric_valid_mask_path)
    np.save(metric_npy_path, metric_depth)

    mask = load_optional_mask(args.segmentation_json.resolve() if args.segmentation_json else None, (w, h))
    object_depth_stats = robust_depth_stats(metric_depth, mask) if mask is not None else None
    best_inlier_uv = uv_in[np.asarray(best["inliers"], dtype=bool)]
    mask_lidar_support = None
    if mask is not None:
        if args.mask_lidar_support_check:
            mask_lidar_support = summarize_mask_lidar_support(
                mask,
                uv_finite,
                uv_in,
                best_inlier_uv,
                args.min_mask_lidar_points,
                args.min_mask_lidar_density,
            )
            if mask_lidar_support is not None:
                mask_lidar_support["enabled"] = True
        else:
            mask_lidar_support = {
                "enabled": False,
                "reason": "disabled",
            }

    result = {
        "image": str(args.image.resolve()),
        "scan": str(args.scan.resolve()),
        "mono_depth_npy": str(args.mono_depth_npy.resolve()),
        "calib_file": str(args.calib_file.resolve()),
        "extrinsics": {
            "source": extrinsics_source or "legacy_constants",
            "params": extrinsics.as_dict(),
        },
        "samples": {
            "projected_inside_image": int(len(uv_finite)),
            "used_for_fit": int(len(mono_in)),
            "edge_candidates": int(edge_candidate_count),
            "edge_rejected": int(edge_rejected_count),
            "edge_filter_applied": bool(edge_filter_applied),
            "mono_depth_min_max": [float(np.nanmin(mono_in)), float(np.nanmax(mono_in))],
            "metric_z_min_max_m": [float(np.nanmin(z_in)), float(np.nanmax(z_in))],
        },
        "depth_edge_filter": {
            **edge_info,
            "enabled": bool(args.edge_filter),
            "applied": bool(edge_filter_applied),
            "candidate_projected_samples": int(edge_candidate_count),
            "rejected_projected_samples": int(edge_rejected_count),
            "min_edge_filtered_samples": int(args.min_edge_filtered_samples),
        },
        "selection_rule": {
            "name": "p90_guarded_mae",
            "p90_tolerance": float(args.selection_p90_tolerance),
            "description": "Choose fits within tolerance of the best P90 absolute error, then select lowest MAE.",
        },
        "fits": [
            {
                "mode": fit["mode"],
                "formula": fit["formula"],
                "params": [float(value) for value in fit["params"]],
                "a": float(fit.get("a", float("nan"))),
                "b": float(fit.get("b", float("nan"))),
                "inlier_count": int(fit["inlier_count"]),
                "mae_m": float(fit["mae_m"]),
                "rmse_m": float(fit["rmse_m"]),
                "median_abs_error_m": float(fit["median_abs_error_m"]),
                "p90_abs_error_m": float(fit["p90_abs_error_m"]),
                "max_abs_error_m": float(fit["max_abs_error_m"]),
            }
            for fit in fits
        ],
        "selected_fit": {
            "mode": best["mode"],
            "formula": best["formula"],
            "params": [float(value) for value in best["params"]],
            "a": float(best.get("a", float("nan"))),
            "b": float(best.get("b", float("nan"))),
            "inlier_count": int(best["inlier_count"]),
            "mae_m": float(best["mae_m"]),
            "rmse_m": float(best["rmse_m"]),
            "median_abs_error_m": float(best["median_abs_error_m"]),
            "p90_abs_error_m": float(best["p90_abs_error_m"]),
            "max_abs_error_m": float(best["max_abs_error_m"]),
        },
        "outputs": {
            "projection": str(projection_path),
            "fit_plot": str(fit_plot_path),
            "metric_depth_color": str(metric_color_path),
            "metric_depth_valid_mask": str(metric_valid_mask_path),
            "metric_depth_npy": str(metric_npy_path),
        },
    }
    if object_depth_stats is not None:
        result["object_mask_metric_depth_m"] = object_depth_stats
    if mask_lidar_support is not None:
        result["object_mask_lidar_support"] = mask_lidar_support
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print(
        f"projected samples: {len(uv_finite)} "
        f"used_for_fit: {len(mono_in)} edge_rejected: {edge_rejected_count}"
    )
    print(f"selected fit: {best['mode']} {best['formula']} params={[round(float(v), 6) for v in best['params']]}")
    print(
        f"inliers: {best['inlier_count']} mae={best['mae_m']:.3f}m "
        f"median={best['median_abs_error_m']:.3f}m p90={best['p90_abs_error_m']:.3f}m"
    )
    print("fit comparison:")
    for fit in fits:
        print(
            f"  {fit['mode']}: mae={fit['mae_m']:.3f}m "
            f"median={fit['median_abs_error_m']:.3f}m p90={fit['p90_abs_error_m']:.3f}m"
        )
    if object_depth_stats is not None:
        print(f"object metric depth median: {object_depth_stats.get('median', float('nan')):.3f}m")
    print(f"projection: {projection_path}")
    print(f"fit plot: {fit_plot_path}")
    print(f"metric depth: {metric_color_path}")
    print(f"json: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
