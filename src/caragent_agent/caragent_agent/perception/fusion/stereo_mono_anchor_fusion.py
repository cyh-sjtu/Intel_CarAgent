"""Stereo-primary object depth with monocular far-anomaly correction."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np


CAMERA_TO_BASE_X_M = 0.30


def _stats(values: np.ndarray) -> dict[str, float]:
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {}
    return {
        "p01": float(np.percentile(values, 1)),
        "p05": float(np.percentile(values, 5)),
        "p10": float(np.percentile(values, 10)),
        "p25": float(np.percentile(values, 25)),
        "median": float(np.median(values)),
        "p75": float(np.percentile(values, 75)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "mean": float(np.mean(values)),
    }


def _first_float(data: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        value = data.get(key)
        try:
            value_f = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value_f):
            return value_f
    return None


def stereo_base_depth_summary(stereo_payload: dict[str, Any]) -> dict[str, Any]:
    """Return Agent-compatible stereo depth summary in base frame."""
    camera_stats = (
        (stereo_payload.get("object_camera_project") or {})
        .get("stats", {})
        .get("x_forward_m", {})
    )
    base_stats = (
        (stereo_payload.get("object_base") or {})
        .get("stats", {})
        .get("x_forward_m", {})
    )
    base_xyz = (stereo_payload.get("object_base") or {}).get("median_xyz_m") or []
    base_range_xy = None
    if isinstance(base_xyz, (list, tuple)) and len(base_xyz) >= 2:
        try:
            x = float(base_xyz[0])
            y = float(base_xyz[1])
            if math.isfinite(x) and math.isfinite(y):
                base_range_xy = float(math.hypot(x, y))
        except (TypeError, ValueError):
            base_range_xy = None
    base_median = _first_float(base_stats, ("median",))
    camera_median = _first_float(camera_stats, ("median",))
    recommended = base_range_xy
    source = "base_range_xy"
    if recommended is None:
        recommended = base_median
        source = "base_x_median"
    if recommended is None:
        recommended = camera_median
        source = "camera_x_median"
    return {
        "recommended_depth_m": recommended,
        "recommended_source": source if recommended is not None else None,
        "camera_x_stats": camera_stats,
        "base_x_stats": base_stats,
        "base_median_xyz_m": base_xyz,
        "base_range_xy_m": base_range_xy,
    }


def _collect_anchors(
    stereo_depth_m: np.ndarray,
    mono_depth: np.ndarray,
    mask: np.ndarray,
    *,
    z_min_m: float = 0.6,
    z_max_m: float = 6.0,
    step_px: int = 16,
    window_px: int = 17,
    max_mad_m: float = 0.12,
    min_valid_fraction: float = 0.72,
    max_per_cell: int = 18,
) -> np.ndarray:
    h, w = stereo_depth_m.shape[:2]
    half = max(1, int(window_px) // 2)
    cell_px = 96
    rows: list[tuple[float, float, float, float, float, float]] = []
    per_cell: dict[tuple[int, int], int] = {}
    mask_dilated = cv2.dilate(mask.astype(np.uint8), np.ones((31, 31), np.uint8), iterations=1).astype(bool)
    for y in range(half, h - half, step_px):
        for x in range(half, w - half, step_px):
            if mask_dilated[y, x]:
                continue
            z = float(stereo_depth_m[y, x])
            mono = float(mono_depth[y, x])
            if not (math.isfinite(z) and math.isfinite(mono) and z_min_m <= z <= z_max_m):
                continue
            patch = stereo_depth_m[y - half : y + half + 1, x - half : x + half + 1]
            valid = np.isfinite(patch) & (patch >= z_min_m) & (patch <= z_max_m)
            valid_fraction = float(valid.mean())
            if valid_fraction < min_valid_fraction:
                continue
            values = patch[valid]
            median_z = float(np.median(values))
            mad_z = float(np.median(np.abs(values - median_z)))
            if mad_z > max_mad_m:
                continue
            if abs(z - median_z) > max(0.18, 2.5 * mad_z):
                continue
            cell = (x // cell_px, y // cell_px)
            if per_cell.get(cell, 0) >= max_per_cell:
                continue
            per_cell[cell] = per_cell.get(cell, 0) + 1
            rows.append((float(x), float(y), median_z, mono, mad_z, valid_fraction))
    if not rows:
        return np.zeros((0, 6), dtype=np.float64)
    return np.asarray(rows, dtype=np.float64)


def _robust_linear_fit(x: np.ndarray, y: np.ndarray, min_keep: int = 80) -> dict[str, Any] | None:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if len(x) < min_keep:
        return None
    keep = np.ones(len(x), dtype=bool)
    for _ in range(5):
        xx = x[keep]
        yy = y[keep]
        if len(xx) < min_keep:
            return None
        a, b = np.linalg.lstsq(np.vstack([xx, np.ones_like(xx)]).T, yy, rcond=None)[0]
        residual = y - (a * x + b)
        med = float(np.median(residual[keep]))
        mad = float(np.median(np.abs(residual[keep] - med))) + 1e-6
        new_keep = np.abs(residual - med) < max(0.25, 3.0 * 1.4826 * mad)
        if np.array_equal(new_keep, keep):
            break
        keep = new_keep
    xx = x[keep]
    yy = y[keep]
    if len(xx) < min_keep:
        return None
    a, b = np.linalg.lstsq(np.vstack([xx, np.ones_like(xx)]).T, yy, rcond=None)[0]
    rmse = float(np.sqrt(np.mean((a * xx + b - yy) ** 2)))
    return {"a": float(a), "b": float(b), "n": int(len(x)), "n_keep": int(len(xx)), "rmse_keep": rmse}


def _load_mask(path: str | Path, shape_hw: tuple[int, int]) -> np.ndarray:
    mask_img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask_img is None:
        raise FileNotFoundError(path)
    if mask_img.shape[:2] != shape_hw:
        mask_img = cv2.resize(mask_img, (shape_hw[1], shape_hw[0]), interpolation=cv2.INTER_NEAREST)
    return mask_img > 0


def compute_stereo_mono_guard(
    *,
    stereo_payload: dict[str, Any],
    mono_depth: np.ndarray,
    mono_depth_source: str | Path | None = None,
) -> dict[str, Any]:
    """Fuse only when mono-relative depth indicates stereo is likely far-biased low.

    The selected depth is reported in base-frame meters to match object approach.
    """
    stereo_summary = stereo_base_depth_summary(stereo_payload)
    stereo_depth_path = stereo_payload.get("depth_npy_path")
    mask_path = stereo_payload.get("rectified_mask_path")
    if not stereo_depth_path or not mask_path:
        return {"status": "failed", "reason": "missing_stereo_depth_or_mask", "stereo": stereo_summary}
    stereo_depth = np.load(str(stereo_depth_path)).astype(np.float32)
    mono = np.asarray(mono_depth, dtype=np.float32)
    if mono.shape[:2] != stereo_depth.shape[:2]:
        mono = cv2.resize(mono, (stereo_depth.shape[1], stereo_depth.shape[0]), interpolation=cv2.INTER_LINEAR)
    mask = _load_mask(mask_path, stereo_depth.shape[:2])
    target_mono = mono[mask & np.isfinite(mono)]
    if len(target_mono) < 50:
        return {"status": "failed", "reason": "too_few_target_mono_pixels", "stereo": stereo_summary}
    anchors = _collect_anchors(stereo_depth, mono, mask)
    if len(anchors) < 80:
        return {
            "status": "failed",
            "reason": "too_few_stable_stereo_anchors",
            "anchor_count": int(len(anchors)),
            "stereo": stereo_summary,
        }
    anchor_z = anchors[:, 2]
    anchor_mono = anchors[:, 3]
    c = min(float(np.percentile(anchor_mono, 5)), float(np.percentile(target_mono, 5))) - 0.02
    eps = 0.15
    anchor_feature = 1.0 / np.maximum(anchor_mono - c + eps, 1e-6)
    fit = _robust_linear_fit(anchor_feature, anchor_z)
    if fit is None:
        return {"status": "failed", "reason": "fit_failed", "anchor_count": int(len(anchors)), "stereo": stereo_summary}
    target_feature = 1.0 / np.maximum(target_mono.astype(np.float64) - c + eps, 1e-6)
    fused_camera_x = fit["a"] * target_feature + fit["b"]
    fused_camera_x = fused_camera_x[np.isfinite(fused_camera_x) & (fused_camera_x > 0.05) & (fused_camera_x < 20.0)]
    if len(fused_camera_x) < 50:
        return {"status": "failed", "reason": "invalid_fused_target_depth", "anchor_count": int(len(anchors)), "stereo": stereo_summary}
    fused_base_x = fused_camera_x + CAMERA_TO_BASE_X_M
    fused_stats = _stats(fused_base_x.astype(np.float64))
    stereo_depth_m = stereo_summary.get("recommended_depth_m")
    try:
        stereo_depth_m = float(stereo_depth_m)
    except (TypeError, ValueError):
        stereo_depth_m = None
    fused_median = _first_float(fused_stats, ("median",))
    fused_iqr = None
    if "p75" in fused_stats and "p25" in fused_stats:
        fused_iqr = float(fused_stats["p75"] - fused_stats["p25"])
    target_mono_stats = _stats(target_mono.astype(np.float64))
    target_mono_median = _first_float(target_mono_stats, ("median",))
    selected = stereo_depth_m
    selected_source = "stereo"
    reason = "stereo_primary"
    correction_delta = None
    if stereo_depth_m is not None and fused_median is not None:
        correction_delta = float(fused_median - stereo_depth_m)
    return {
        "status": "ok",
        "selected_depth_m": selected,
        "selected_source": selected_source,
        "reason": reason,
        "stereo": stereo_summary,
        "fused_base_x_m": fused_stats,
        "fused_camera_x_m": _stats(fused_camera_x.astype(np.float64)),
        "correction_delta_m": correction_delta,
        "fused_iqr_m": fused_iqr,
        "anchor_count": int(len(anchors)),
        "anchor_z_m": _stats(anchor_z.astype(np.float64)),
        "anchor_mono": _stats(anchor_mono.astype(np.float64)),
        "target_mono": target_mono_stats,
        "fit": fit,
        "fit_transform": {"mode": "inverse_adaptive", "c": float(c), "eps": float(eps)},
        "mono_depth_source": str(mono_depth_source) if mono_depth_source is not None else None,
    }


def write_guard_payload(path: str | Path, payload: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
