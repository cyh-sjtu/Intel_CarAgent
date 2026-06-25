"""Offline experiments for LiDAR anchors in mono-depth metric fitting."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .project_scan_fit_monodepth import (
    build_depth_edge_mask,
    base_to_left_optical,
    load_calib,
    load_extrinsics,
    load_optional_mask,
    predict_model,
    project_points,
    robust_fit,
    sample_bool_mask,
    sample_depth,
    scan_points_in_base,
    select_fit,
)


DEFAULT_WORKSPACE = Path.home() / "caragent_ws"
DEFAULT_CALIB = DEFAULT_WORKSPACE / "calibration" / "stereo_current" / "stereo_calibration.npz"
DEFAULT_EXTR = DEFAULT_WORKSPACE / "calibration" / "lidar_camera" / "lidar_camera_extrinsics_calibrated.json"
QUANTILES = ("p05", "p10", "p20", "p30", "median")


@dataclass
class AnchorSet:
    mono: np.ndarray
    z: np.ndarray
    uv: np.ndarray
    segment_ids: np.ndarray
    note: str = ""


def read_manifest(dataset_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for line in (dataset_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def safe_float(value: Any) -> float | None:
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return None
    return value_f if math.isfinite(value_f) else None


def load_scan_valid_indices(scan_file: Path) -> tuple[np.ndarray, np.ndarray]:
    scan = np.load(scan_file)
    ranges = scan["ranges"].astype(np.float32)
    range_min = float(scan["range_min"])
    range_max = float(scan["range_max"])
    valid = np.isfinite(ranges) & (ranges >= range_min) & (ranges <= range_max)
    return np.flatnonzero(valid), ranges[valid]


def split_scan_segments(
    valid_indices: np.ndarray,
    ranges: np.ndarray,
    max_jump_m: float,
    max_jump_ratio: float,
) -> np.ndarray:
    segment_ids = np.zeros(len(ranges), dtype=np.int32)
    current = 0
    for i in range(1, len(ranges)):
        gap = int(valid_indices[i] - valid_indices[i - 1])
        jump = abs(float(ranges[i] - ranges[i - 1]))
        rel = jump / max(1e-6, min(float(ranges[i]), float(ranges[i - 1])))
        if gap > 1 or (jump > max_jump_m and rel > max_jump_ratio):
            current += 1
        segment_ids[i] = current
    return segment_ids


def stable_segment_ids(
    segment_ids: np.ndarray,
    ranges: np.ndarray,
    min_points: int,
    max_mad_m: float,
) -> set[int]:
    stable: set[int] = set()
    for sid in np.unique(segment_ids):
        values = ranges[segment_ids == sid]
        if len(values) < min_points:
            continue
        med = float(np.median(values))
        mad = float(np.median(np.abs(values - med)))
        if mad <= max_mad_m:
            stable.add(int(sid))
    return stable


def image_edge_mask(image_bgr: np.ndarray, percentile: float = 90.0, dilate_px: int = 3) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    smooth = cv2.GaussianBlur(gray, (5, 5), 0)
    gx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx * gx + gy * gy)
    threshold = float(np.percentile(grad, percentile))
    edge = grad > threshold
    if dilate_px > 1:
        kernel = np.ones((int(dilate_px), int(dilate_px)), dtype=np.uint8)
        edge = cv2.dilate(edge.astype(np.uint8), kernel) > 0
    return edge


def erode_mask(mask: np.ndarray | None, px: int) -> np.ndarray | None:
    if mask is None:
        return None
    if px <= 0:
        return mask
    size = max(1, int(px))
    if size % 2 == 0:
        size += 1
    kernel = np.ones((size, size), dtype=np.uint8)
    return cv2.erode(mask.astype(np.uint8), kernel) > 0


def border_filter(uv: np.ndarray, width: int, height: int, border_px: int) -> np.ndarray:
    return (
        (uv[:, 0] >= border_px)
        & (uv[:, 0] < width - border_px)
        & (uv[:, 1] >= border_px)
        & (uv[:, 1] < height - border_px)
    )


def aggregate_by_segment(anchors: AnchorSet, min_projected_points: int) -> AnchorSet:
    mono_values = []
    z_values = []
    uv_values = []
    segment_values = []
    for sid in np.unique(anchors.segment_ids):
        idx = anchors.segment_ids == sid
        if int(np.count_nonzero(idx)) < min_projected_points:
            continue
        mono_seg = anchors.mono[idx]
        z_seg = anchors.z[idx]
        uv_seg = anchors.uv[idx]
        mono_med = float(np.median(mono_seg))
        z_med = float(np.median(z_seg))
        mono_mad = float(np.median(np.abs(mono_seg - mono_med)))
        if not np.isfinite(mono_med) or not np.isfinite(z_med):
            continue
        mono_values.append(mono_med)
        z_values.append(z_med)
        uv_values.append(np.median(uv_seg, axis=0))
        segment_values.append(int(sid))
    if not mono_values:
        return AnchorSet(
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.float32),
            np.empty((0, 2), dtype=np.float32),
            np.empty(0, dtype=np.int32),
            note="no_segment_anchors",
        )
    return AnchorSet(
        np.asarray(mono_values, dtype=np.float32),
        np.asarray(z_values, dtype=np.float32),
        np.asarray(uv_values, dtype=np.float32),
        np.asarray(segment_values, dtype=np.int32),
    )


def trim_segment_edges(segment_ids: np.ndarray, trim_points: int) -> np.ndarray:
    keep = np.ones(len(segment_ids), dtype=bool)
    if trim_points <= 0:
        return keep
    for sid in np.unique(segment_ids):
        idx = np.flatnonzero(segment_ids == sid)
        if len(idx) <= 2 * trim_points:
            keep[idx] = False
            continue
        keep[idx[:trim_points]] = False
        keep[idx[-trim_points:]] = False
    return keep


def quantile_stats(values: np.ndarray) -> dict[str, float]:
    values = values[np.isfinite(values) & (values > 0)]
    if len(values) == 0:
        return {}
    return {
        "p05": float(np.percentile(values, 5)),
        "p10": float(np.percentile(values, 10)),
        "p20": float(np.percentile(values, 20)),
        "p30": float(np.percentile(values, 30)),
        "median": float(np.median(values)),
    }


def fit_and_score(
    sample: dict[str, Any],
    strategy: str,
    anchors: AnchorSet,
    mono_depth: np.ndarray,
    object_mask: np.ndarray | None,
    min_anchors: int,
    max_depth_m: float,
) -> dict[str, Any]:
    truth = safe_float(sample.get("truth_distance_m"))
    base = {
        "sample_id": sample.get("sample_id"),
        "target": sample.get("target"),
        "truth_distance_m": truth,
        "strategy": strategy,
        "anchor_count": int(len(anchors.mono)),
        "segment_count": int(len(set(int(v) for v in anchors.segment_ids))) if len(anchors.segment_ids) else 0,
        "status": "ok",
        "failure_reason": "",
    }
    if object_mask is None or not np.any(object_mask):
        base.update({"status": "failed", "failure_reason": "missing_object_mask"})
        return base
    if len(anchors.mono) < min_anchors:
        base.update({"status": "failed", "failure_reason": f"too_few_anchors_{len(anchors.mono)}"})
        return base

    fit_modes = ["linear", "inverse", "log", "quadratic"]
    fits = []
    for mode in fit_modes:
        try:
            if len(anchors.mono) < 3 and mode == "quadratic":
                continue
            fits.append(robust_fit(anchors.mono, anchors.z, mode))
        except Exception:
            continue
    if not fits:
        base.update({"status": "failed", "failure_reason": "fit_failed"})
        return base
    best = select_fit(fits, p90_tolerance=0.10)
    metric_depth = predict_model(mono_depth, np.asarray(best["params"], dtype=np.float64), best["mode"]).astype(np.float32)
    metric_depth[(metric_depth <= 0.0) | (metric_depth > max_depth_m) | ~np.isfinite(metric_depth)] = np.nan
    stats = quantile_stats(metric_depth[object_mask])
    if not stats:
        base.update({"status": "failed", "failure_reason": "empty_metric_depth_in_mask"})
        return base

    base.update(
        {
            "fit_mode": best["mode"],
            "fit_mae_m": float(best["mae_m"]),
            "fit_p90_m": float(best["p90_abs_error_m"]),
            "fit_inliers": int(best["inlier_count"]),
        }
    )
    for key in QUANTILES:
        pred = stats.get(key)
        err = None if pred is None or truth is None else pred - truth
        base[f"{key}_m"] = pred
        base[f"{key}_error_m"] = err
        base[f"{key}_abs_error_m"] = abs(err) if err is not None else None
    return base


def build_anchor_sets(
    image_bgr: np.ndarray,
    mono_depth: np.ndarray,
    scan: Path,
    calib: Path,
    extrinsics_json: Path,
    object_mask: np.ndarray | None,
) -> dict[str, AnchorSet]:
    h, w = mono_depth.shape
    mtx, dist = load_calib(calib)
    extrinsics, _ = load_extrinsics(extrinsics_json)
    valid_indices, valid_ranges = load_scan_valid_indices(scan)
    segment_ids_all = split_scan_segments(valid_indices, valid_ranges, max_jump_m=0.18, max_jump_ratio=0.08)
    stable_ids = stable_segment_ids(segment_ids_all, valid_ranges, min_points=5, max_mad_m=0.08)

    points_base, _ = scan_points_in_base(scan, extrinsics)
    points_opt = base_to_left_optical(points_base, extrinsics)
    uv = project_points(points_opt, mtx, dist)
    z = points_opt[:, 2]
    mono = sample_depth(mono_depth, uv)
    inside = (
        (z >= 0.20)
        & (z <= 6.0)
        & (uv[:, 0] >= 0)
        & (uv[:, 0] < w)
        & (uv[:, 1] >= 0)
        & (uv[:, 1] < h)
        & np.isfinite(mono)
        & (mono > 0)
    )
    uv = uv[inside]
    z = z[inside]
    mono = mono[inside]
    segment_ids = segment_ids_all[inside]

    base = AnchorSet(mono=mono, z=z, uv=uv, segment_ids=segment_ids)
    keep_border = border_filter(uv, w, h, 10)
    depth_edge, depth_edge_info = build_depth_edge_mask(mono_depth, percentile=90.0, dilate_px=5)
    keep_depth_edge = ~sample_bool_mask(depth_edge, uv) if depth_edge_info.get("available") else np.ones(len(uv), dtype=bool)
    strict_depth_edge, strict_depth_edge_info = build_depth_edge_mask(mono_depth, percentile=80.0, dilate_px=7)
    keep_strict_depth_edge = (
        ~sample_bool_mask(strict_depth_edge, uv)
        if strict_depth_edge_info.get("available")
        else np.ones(len(uv), dtype=bool)
    )
    rgb_edge = image_edge_mask(image_bgr, percentile=92.0, dilate_px=3)
    keep_rgb_edge = ~sample_bool_mask(rgb_edge, uv)
    strict_rgb_edge = image_edge_mask(image_bgr, percentile=85.0, dilate_px=5)
    keep_strict_rgb_edge = ~sample_bool_mask(strict_rgb_edge, uv)
    keep_stable = np.array([int(sid) in stable_ids for sid in segment_ids], dtype=bool)
    keep_trim1 = trim_segment_edges(segment_ids, trim_points=1)
    keep_trim2 = trim_segment_edges(segment_ids, trim_points=2)

    masks: dict[str, np.ndarray] = {
        "global_no_edge": np.ones(len(uv), dtype=bool),
        "global_depth_edge": keep_depth_edge,
        "global_depth_rgb_edge": keep_depth_edge & keep_rgb_edge & keep_border,
        "stable_points_global": keep_stable & keep_depth_edge & keep_border,
        "sparse_smooth_segments": keep_stable & keep_trim1 & keep_strict_depth_edge & keep_strict_rgb_edge & keep_border,
        "sparse_smooth_segments_trim2": keep_stable & keep_trim2 & keep_strict_depth_edge & keep_strict_rgb_edge & keep_border,
    }
    if object_mask is not None:
        for px in (5, 11):
            eroded = erode_mask(object_mask, px)
            if eroded is not None:
                in_mask = sample_bool_mask(eroded, uv)
                masks[f"target_erode{px}_depth_edge"] = in_mask & keep_depth_edge & keep_border
                masks[f"stable_target_erode{px}"] = in_mask & keep_stable & keep_depth_edge & keep_border

    out: dict[str, AnchorSet] = {}
    for name, keep in masks.items():
        out[name] = AnchorSet(mono=mono[keep], z=z[keep], uv=uv[keep], segment_ids=segment_ids[keep])
    for name in list(out):
        if name.startswith("stable_"):
            out[name + "_segment_median"] = aggregate_by_segment(out[name], min_projected_points=2)
        if name.startswith("sparse_smooth"):
            out[name + "_segment_median"] = aggregate_by_segment(out[name], min_projected_points=2)
    out["global_depth_edge_segment_median"] = aggregate_by_segment(out["global_depth_edge"], min_projected_points=3)
    return out


def find_depth_and_seg(eval_dir: Path, sample_id: str) -> tuple[Path | None, Path | None]:
    sample_dir = eval_dir / sample_id / "mono_relative_lidar"
    depth = next((sample_dir / "depth").glob("*_depth.npy"), None) if (sample_dir / "depth").exists() else None
    seg = sample_dir / f"{sample_id}_segmentation_ov.json"
    if not seg.exists():
        segs = sorted((eval_dir / sample_id).glob("*/" + f"{sample_id}_segmentation_ov.json"))
        seg = segs[0] if segs else None
    return depth, seg


def hybrid_row(
    sparse_row: dict[str, Any] | None,
    fallback_row: dict[str, Any] | None,
    strategy: str,
    max_fit_p90_m: float,
) -> dict[str, Any] | None:
    if fallback_row is None:
        return None
    chosen = fallback_row
    chosen_strategy = fallback_row.get("strategy", "")
    if sparse_row is not None and sparse_row.get("status") == "ok":
        fit_p90 = safe_float(sparse_row.get("fit_p90_m"))
        if fit_p90 is not None and fit_p90 <= max_fit_p90_m:
            chosen = sparse_row
            chosen_strategy = sparse_row.get("strategy", "")
    row = dict(chosen)
    row["strategy"] = strategy
    row["chosen_strategy"] = chosen_strategy
    row["hybrid_fit_p90_limit_m"] = max_fit_p90_m
    return row


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary = []
    for strategy in sorted({str(row["strategy"]) for row in rows}):
        strategy_rows = [row for row in rows if row["strategy"] == strategy]
        ok_rows = [row for row in strategy_rows if row.get("status") == "ok"]
        item: dict[str, Any] = {
            "strategy": strategy,
            "total": len(strategy_rows),
            "ok": len(ok_rows),
            "failed": len(strategy_rows) - len(ok_rows),
        }
        for q in QUANTILES:
            errors = [safe_float(row.get(f"{q}_abs_error_m")) for row in ok_rows]
            errors = [value for value in errors if value is not None]
            signed = [safe_float(row.get(f"{q}_error_m")) for row in ok_rows]
            signed = [value for value in signed if value is not None]
            if errors:
                item[f"{q}_mae_m"] = float(np.mean(errors))
                item[f"{q}_median_abs_m"] = float(np.median(errors))
                item[f"{q}_max_abs_m"] = float(np.max(errors))
                item[f"{q}_bias_m"] = float(np.mean(signed)) if signed else None
        summary.append(item)
    return summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare mono+LiDAR anchor selection strategies on a dataset.")
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--eval-run", default="baseline_20260614")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--calib-file", default=DEFAULT_CALIB, type=Path)
    parser.add_argument("--extrinsics-json", default=DEFAULT_EXTR, type=Path)
    parser.add_argument("--min-anchors", default=4, type=int)
    parser.add_argument("--max-depth", default=6.0, type=float)
    parser.add_argument("--hybrid-max-fit-p90", default=0.30, type=float)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    dataset_dir = args.dataset_dir.expanduser().resolve()
    eval_dir = dataset_dir / "evaluations" / args.eval_run
    output_dir = Path(args.output_dir).expanduser().resolve() if str(args.output_dir) else dataset_dir / "evaluations" / "anchor_strategy_experiments"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for sample in read_manifest(dataset_dir):
        sample_id = str(sample["sample_id"])
        depth_path, seg_path = find_depth_and_seg(eval_dir, sample_id)
        if depth_path is None or seg_path is None or not depth_path.exists() or not seg_path.exists():
            rows.append(
                {
                    "sample_id": sample_id,
                    "target": sample.get("target"),
                    "truth_distance_m": sample.get("truth_distance_m"),
                    "strategy": "input",
                    "status": "failed",
                    "failure_reason": "missing_depth_or_segmentation",
                }
            )
            continue
        image_path = Path(sample["left_image"])
        scan_path = Path(sample["scan"])
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            continue
        mono_depth = np.load(depth_path).astype(np.float32)
        if mono_depth.shape != image_bgr.shape[:2]:
            mono_depth = cv2.resize(mono_depth, (image_bgr.shape[1], image_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
        object_mask = load_optional_mask(seg_path, (image_bgr.shape[1], image_bgr.shape[0]))
        try:
            anchor_sets = build_anchor_sets(
                image_bgr,
                mono_depth,
                scan_path,
                args.calib_file.expanduser().resolve(),
                args.extrinsics_json.expanduser().resolve(),
                object_mask,
            )
        except Exception as exc:
            rows.append(
                {
                    "sample_id": sample_id,
                    "target": sample.get("target"),
                    "truth_distance_m": sample.get("truth_distance_m"),
                    "strategy": "projection",
                    "status": "failed",
                    "failure_reason": str(exc),
                }
            )
            continue
        sample_rows: dict[str, dict[str, Any]] = {}
        for strategy, anchors in anchor_sets.items():
            row = fit_and_score(
                sample,
                strategy,
                anchors,
                mono_depth,
                object_mask,
                min_anchors=args.min_anchors,
                max_depth_m=args.max_depth,
            )
            sample_rows[strategy] = row
            rows.append(row)
        for strategy, sparse_name in [
            ("hybrid_sparse_smooth_or_global", "sparse_smooth_segments_segment_median"),
            ("hybrid_sparse_trim2_or_global", "sparse_smooth_segments_trim2_segment_median"),
            ("hybrid_stable_or_global", "stable_points_global"),
        ]:
            row = hybrid_row(
                sample_rows.get(sparse_name),
                sample_rows.get("global_depth_edge"),
                strategy,
                max_fit_p90_m=args.hybrid_max_fit_p90,
            )
            if row is not None:
                rows.append(row)

    detail_csv = output_dir / "anchor_strategy_detail.csv"
    summary_csv = output_dir / "anchor_strategy_summary.csv"
    write_csv(detail_csv, rows)
    summary_rows = summarize(rows)
    write_csv(summary_csv, summary_rows)
    print(f"detail_csv: {detail_csv}")
    print(f"summary_csv: {summary_csv}")
    for row in sorted(summary_rows, key=lambda item: item.get("p10_mae_m", float("inf")))[:10]:
        print(
            f"{row['strategy']}: ok={row['ok']}/{row['total']} "
            f"p10_mae={row.get('p10_mae_m')} p10_max={row.get('p10_max_abs_m')} "
            f"median_mae={row.get('median_mae_m')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
