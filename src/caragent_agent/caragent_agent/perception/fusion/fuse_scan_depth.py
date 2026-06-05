"""Fuse a segmentation mask, relative monocular depth, and a 2D LiDAR scan.

This is a first-pass geometric experiment. It estimates the target bearing from
the image, samples LaserScan ranges around that bearing, and uses those metric
ranges to assign a real scale to the target mask/depth.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


# From caragent_description/urdf/caragent.urdf
LASER_X_M = 0.12
LASER_Y_M = 0.0
LASER_YAW_RAD = math.pi
CAMERA_LEFT_X_M = 0.30
CAMERA_LEFT_Y_M = 0.03
CAMERA_RIGHT_X_M = 0.30
CAMERA_RIGHT_Y_M = -0.03
CAMERA_HEIGHT_M = 0.185
LASER_HEIGHT_M = 0.30

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
    laser_z_m: float = LASER_HEIGHT_M
    laser_roll_rad: float = 0.0
    laser_pitch_rad: float = 0.0
    laser_yaw_rad: float = LASER_YAW_RAD
    camera_x_m: float = CAMERA_LEFT_X_M
    camera_y_m: float = CAMERA_LEFT_Y_M
    camera_z_m: float = CAMERA_HEIGHT_M
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

    data = load_json(path)
    params = data.get("optimized_params", data)
    defaults = Extrinsics().as_dict()
    values = {key: float(params.get(key, value)) for key, value in defaults.items()}
    return Extrinsics(**values), str(path.resolve())


def load_calib(calib_file: Path | None) -> tuple[np.ndarray, np.ndarray] | None:
    if calib_file is None:
        return None
    data = np.load(calib_file)
    return data["mtx_l"].astype(np.float64), data["dist_l"].astype(np.float64)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def bbox_center(box: list[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = [float(v) for v in box]
    return (0.5 * (x1 + x2), 0.5 * (y1 + y2))


def mask_stats(mask_path: Path) -> dict[str, Any]:
    mask = np.array(Image.open(mask_path).convert("L")) > 0
    ys, xs = np.where(mask)
    if len(xs) == 0:
        raise ValueError(f"Mask is empty: {mask_path}")
    return {
        "mask": mask,
        "centroid_px": [float(xs.mean()), float(ys.mean())],
        "bottom_point_px": [float(xs[ys.argmax()]), float(ys.max())],
        "bbox_px": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
        "area_px": int(mask.sum()),
    }


def pixel_to_bearing_rad(x_px: float, image_width: int, horizontal_fov_deg: float) -> float:
    cx = 0.5 * image_width
    fov = math.radians(horizontal_fov_deg)
    fx = cx / math.tan(0.5 * fov)
    # Image u grows to the right, while base_link +Y is robot-left.
    # Therefore pixels on the right side of the image have negative bearing.
    return math.atan2(cx - x_px, fx)


def pixel_to_base_bearing_rad(
    x_px: float,
    y_px: float,
    image_width: int,
    horizontal_fov_deg: float,
    extrinsics: Extrinsics,
    calib: tuple[np.ndarray, np.ndarray] | None,
) -> float:
    if calib is not None:
        import cv2

        mtx, dist = calib
        point = np.array([[[x_px, y_px]]], dtype=np.float64)
        undistorted = cv2.undistortPoints(point, mtx, dist).reshape(2)
        x_opt, y_opt = float(undistorted[0]), float(undistorted[1])
        dir_camera = np.array([1.0, -x_opt, -y_opt], dtype=np.float64)
    else:
        camera_bearing_rad = pixel_to_bearing_rad(x_px, image_width, horizontal_fov_deg)
        dir_camera = np.array(
            [math.cos(camera_bearing_rad), math.sin(camera_bearing_rad), 0.0],
            dtype=np.float64,
        )

    r_camera_to_base = rodrigues_xyz(
        extrinsics.camera_roll_rad,
        extrinsics.camera_pitch_rad,
        extrinsics.camera_yaw_rad,
    )
    dir_base = dir_camera @ r_camera_to_base.T
    return math.atan2(float(dir_base[1]), float(dir_base[0]))


def base_bearing_to_laser_angle_rad(base_bearing_rad: float, extrinsics: Extrinsics) -> float:
    # Camera frame and base_link both use +X forward, +Y left in this project.
    # Laser link has yaw=pi relative to base_link, so a base-frame bearing theta
    # appears as theta - pi in laser frame. Wrap to [-pi, pi].
    angle = base_bearing_rad - extrinsics.laser_yaw_rad
    return math.atan2(math.sin(angle), math.cos(angle))


def sample_scan_metric_range(
    scan_path: Path,
    laser_angle_rad: float,
    window_deg: float,
) -> dict[str, Any]:
    scan = np.load(scan_path, allow_pickle=True)
    ranges = scan["ranges"].astype(np.float32)
    angle_min = float(scan["angle_min"])
    angle_increment = float(scan["angle_increment"])
    range_min = float(scan["range_min"])
    range_max = float(scan["range_max"])
    angles = angle_min + np.arange(len(ranges), dtype=np.float32) * angle_increment
    angle_delta = np.arctan2(np.sin(angles - laser_angle_rad), np.cos(angles - laser_angle_rad))
    valid = np.isfinite(ranges) & (ranges >= range_min) & (ranges <= range_max)
    in_window = np.abs(angle_delta) <= math.radians(window_deg)
    selected = valid & in_window

    if not selected.any():
        # Fall back to nearest valid beam.
        valid_indices = np.where(valid)[0]
        if len(valid_indices) == 0:
            raise ValueError(f"No valid scan ranges in {scan_path}")
        idx = int(valid_indices[np.argmin(np.abs(angle_delta[valid_indices]))])
        selected_indices = np.array([idx], dtype=np.int32)
    else:
        selected_indices = np.where(selected)[0]

    selected_ranges = ranges[selected_indices]
    selected_angles = angles[selected_indices]
    nearest_idx = int(selected_indices[np.argmin(np.abs(selected_angles - laser_angle_rad))])

    # Prefer the nearest contiguous scan cluster in the angular window. This is
    # better than a fixed percentile when a thin object/leg produces only 2-3
    # beams and a wall behind it produces many more beams.
    clusters: list[np.ndarray] = []
    start = 0
    for i in range(1, len(selected_indices)):
        index_gap = int(selected_indices[i] - selected_indices[i - 1])
        range_gap = abs(float(selected_ranges[i] - selected_ranges[i - 1]))
        if index_gap > 1 or range_gap > 0.35:
            clusters.append(np.arange(start, i, dtype=np.int32))
            start = i
    clusters.append(np.arange(start, len(selected_indices), dtype=np.int32))

    ranked_clusters = []
    for cluster in clusters:
        cluster_ranges = selected_ranges[cluster]
        cluster_indices = selected_indices[cluster]
        ranked_clusters.append(
            {
                "selected_offsets": cluster.astype(int).tolist(),
                "beam_indices": cluster_indices.astype(int).tolist(),
                "range_median_m": float(np.median(cluster_ranges)),
                "range_min_m": float(np.min(cluster_ranges)),
                "range_max_m": float(np.max(cluster_ranges)),
                "beam_count": int(len(cluster)),
            }
        )
    usable_clusters = [c for c in ranked_clusters if c["beam_count"] >= 2]
    if not usable_clusters:
        usable_clusters = ranked_clusters
    chosen_cluster = min(usable_clusters, key=lambda c: (c["range_median_m"], -c["beam_count"]))
    metric_range = float(chosen_cluster["range_median_m"])
    return {
        "metric_range_m": metric_range,
        "range_selection_method": "nearest_contiguous_cluster",
        "chosen_cluster": chosen_cluster,
        "range_clusters": ranked_clusters,
        "nearest_beam_index": nearest_idx,
        "nearest_beam_range_m": float(ranges[nearest_idx]),
        "selected_beam_indices": selected_indices.astype(int).tolist(),
        "selected_ranges_m": selected_ranges.astype(float).tolist(),
        "selected_angles_rad": selected_angles.astype(float).tolist(),
        "range_min_m": range_min,
        "range_max_m": range_max,
    }


def scan_to_base_points(scan_path: Path, extrinsics: Extrinsics) -> np.ndarray:
    scan = np.load(scan_path, allow_pickle=True)
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
    points_base = t_lidar_to_base + points_laser @ r_lidar_to_base.T
    return points_base[:, :2]


def estimate_target_base_position(
    bearing_rad: float,
    metric_range_m: float,
    extrinsics: Extrinsics,
) -> dict[str, Any]:
    # Treat the LiDAR metric range at the same horizontal bearing as a base-frame
    # obstacle range, then express the target point from base_link.
    x = extrinsics.laser_x_m + metric_range_m * math.cos(bearing_rad)
    y = extrinsics.laser_y_m + metric_range_m * math.sin(bearing_rad)
    camera_forward = x - extrinsics.camera_x_m
    camera_left = y - extrinsics.camera_y_m
    camera_planar_distance = math.hypot(camera_forward, camera_left)
    return {
        "base_xy_m": [x, y],
        "range_from_base_m": math.hypot(x, y),
        "bearing_base_rad": bearing_rad,
        "bearing_base_deg": math.degrees(bearing_rad),
        "camera_planar_distance_m": camera_planar_distance,
        "camera_forward_m": camera_forward,
        "camera_left_m": camera_left,
    }


def depth_stats(depth_path: Path, mask: np.ndarray) -> dict[str, Any]:
    depth = np.load(depth_path)
    if depth.shape != mask.shape:
        depth_img = Image.fromarray(depth.astype(np.float32))
        depth_img = depth_img.resize((mask.shape[1], mask.shape[0]), resample=Image.Resampling.BICUBIC)
        depth = np.array(depth_img, dtype=np.float32)
    values = depth[mask]
    return {
        "relative_depth_median": float(np.median(values)),
        "relative_depth_mean": float(np.mean(values)),
        "relative_depth_p25": float(np.percentile(values, 25.0)),
        "relative_depth_p75": float(np.percentile(values, 75.0)),
    }


def draw_image_overlay(
    image_path: Path,
    mask: np.ndarray,
    box: list[float],
    bearing_px: float,
    result: dict[str, Any],
    output_path: Path,
) -> None:
    image = Image.open(image_path).convert("RGB")
    arr = np.array(image)
    overlay = arr.copy()
    green = np.array([0, 255, 120], dtype=np.uint8)
    overlay[mask] = (overlay[mask] * 0.58 + green * 0.42).astype(np.uint8)
    out = Image.fromarray(overlay)
    draw = ImageDraw.Draw(out)

    x1, y1, x2, y2 = [round(float(v)) for v in box]
    draw.rectangle([x1, y1, x2, y2], outline=(255, 70, 70), width=3)
    draw.line([bearing_px, 0, bearing_px, image.height], fill=(255, 255, 0), width=2)
    target = result["target_base"]["base_xy_m"]
    text = (
        f"bearing={result['target_base']['bearing_base_deg']:.1f} deg, "
        f"lidar={result['scan_fusion']['metric_range_m']:.2f} m, "
        f"base_xy=({target[0]:.2f},{target[1]:.2f}) m"
    )
    draw.rectangle([4, 4, min(image.width - 1, 4 + len(text) * 7), 24], fill=(0, 0, 0))
    draw.text((8, 7), text, fill=(255, 255, 255))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.save(output_path)


def draw_topdown(
    scan_path: Path,
    result: dict[str, Any],
    output_path: Path,
    extrinsics: Extrinsics,
    meters_per_side: float = 5.0,
    pixels: int = 900,
) -> None:
    points = scan_to_base_points(scan_path, extrinsics)
    image = Image.new("RGB", (pixels, pixels), (18, 20, 24))
    draw = ImageDraw.Draw(image)
    center = (pixels // 2, pixels // 2)
    scale = pixels / meters_per_side

    def to_px(x: float, y: float) -> tuple[int, int]:
        # Intuitive robot view: screen up is +X forward, screen left is +Y left.
        return (int(center[0] - y * scale), int(center[1] - x * scale))

    # Grid.
    for m in np.arange(-meters_per_side / 2, meters_per_side / 2 + 0.001, 0.5):
        x0, y0 = to_px(-meters_per_side / 2, float(m))
        x1, y1 = to_px(meters_per_side / 2, float(m))
        draw.line([x0, y0, x1, y1], fill=(42, 45, 52))
        x0, y0 = to_px(float(m), -meters_per_side / 2)
        x1, y1 = to_px(float(m), meters_per_side / 2)
        draw.line([x0, y0, x1, y1], fill=(42, 45, 52))

    for x, y in points:
        px, py = to_px(float(x), float(y))
        if 0 <= px < pixels and 0 <= py < pixels:
            draw.ellipse([px - 1, py - 1, px + 1, py + 1], fill=(120, 170, 255))

    # Robot base, lidar, cameras, selected beams, and target.
    footprint = [
        to_px(0.225, 0.21),
        to_px(0.225, -0.21),
        to_px(-0.225, -0.21),
        to_px(-0.225, 0.21),
    ]
    draw.polygon(footprint, outline=(235, 235, 235), fill=(45, 48, 56))
    bx, by = to_px(0.0, 0.0)
    draw.ellipse([bx - 6, by - 6, bx + 6, by + 6], fill=(255, 255, 255))
    draw.text((bx + 8, by + 8), "base", fill=(255, 255, 255))

    fx, fy = to_px(0.8, 0.0)
    ly_axis = to_px(0.0, 0.8)
    draw.line([bx, by, fx, fy], fill=(255, 255, 255), width=3)
    draw.text((fx + 8, fy - 10), "+X forward", fill=(255, 255, 255))
    draw.line([bx, by, ly_axis[0], ly_axis[1]], fill=(120, 220, 255), width=3)
    draw.text((ly_axis[0] - 85, ly_axis[1] - 10), "+Y left", fill=(120, 220, 255))

    lx, ly = to_px(extrinsics.laser_x_m, extrinsics.laser_y_m)
    draw.ellipse([lx - 5, ly - 5, lx + 5, ly + 5], fill=(255, 80, 80))
    draw.text((lx + 8, ly - 12), "lidar", fill=(255, 80, 80))
    cx, cy = to_px(extrinsics.camera_x_m, extrinsics.camera_y_m)
    draw.ellipse([cx - 5, cy - 5, cx + 5, cy + 5], fill=(80, 255, 120))
    draw.text((cx - 95, cy - 12), "camera_left", fill=(80, 255, 120))
    crx, cry = to_px(CAMERA_RIGHT_X_M, CAMERA_RIGHT_Y_M)
    draw.ellipse([crx - 5, cry - 5, crx + 5, cry + 5], fill=(80, 210, 255))
    draw.text((crx + 8, cry - 12), "camera_right", fill=(80, 210, 255))

    selected_ranges = result.get("scan_fusion", {}).get("selected_ranges_m", [])
    selected_angles = result.get("scan_fusion", {}).get("selected_angles_rad", [])
    for r, a_laser in zip(selected_ranges, selected_angles):
        x_l = float(r) * math.cos(float(a_laser))
        y_l = float(r) * math.sin(float(a_laser))
        r_lidar_to_base = rodrigues_xyz(
            extrinsics.laser_roll_rad,
            extrinsics.laser_pitch_rad,
            extrinsics.laser_yaw_rad,
        )
        p_b = (
            np.array([extrinsics.laser_x_m, extrinsics.laser_y_m, extrinsics.laser_z_m])
            + r_lidar_to_base @ np.array([x_l, y_l, 0.0])
        )
        x_b, y_b = float(p_b[0]), float(p_b[1])
        spx, spy = to_px(x_b, y_b)
        draw.line([lx, ly, spx, spy], fill=(255, 230, 0), width=1)
        draw.ellipse([spx - 3, spy - 3, spx + 3, spy + 3], fill=(255, 230, 0))

    tx, ty = result["target_base"]["base_xy_m"]
    tpx, tpy = to_px(float(tx), float(ty))
    draw.line([lx, ly, tpx, tpy], fill=(255, 255, 0), width=3)
    draw.ellipse([tpx - 8, tpy - 8, tpx + 8, tpy + 8], fill=(255, 255, 0))
    draw.text((tpx + 10, tpy - 14), f"target ({tx:.2f},{ty:.2f})", fill=(255, 255, 0))

    lines = [
        "Topdown: up=+X forward, left=+Y robot-left",
        "Blue dots=LaserScan in base_link; yellow=selected beams/target",
    ]
    y_text = 8
    for line in lines:
        draw.rectangle([6, y_text - 2, min(pixels - 1, 14 + len(line) * 7), y_text + 14], fill=(0, 0, 0))
        draw.text((10, y_text), line, fill=(255, 255, 255))
        y_text += 17

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fuse mask/depth outputs with a keyframe LaserScan.")
    parser.add_argument("--segmentation-json", required=True, type=Path)
    parser.add_argument("--depth-npy", required=True, type=Path)
    parser.add_argument("--scan", required=True, type=Path)
    parser.add_argument("--calib-file", type=Path, help="Optional left-camera calibration .npz for calibrated bearing.")
    parser.add_argument(
        "--extrinsics-json",
        type=Path,
        help=(
            "LiDAR-camera extrinsics JSON. Defaults to "
            f"{DEFAULT_EXTRINSICS_JSON} when that file exists; otherwise uses legacy constants."
        ),
    )
    parser.add_argument("--horizontal-fov-deg", default=87.0, type=float)
    parser.add_argument("--scan-window-deg", default=4.0, type=float)
    parser.add_argument("--output-dir", default=Path("outputs"), type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    seg = load_json(args.segmentation_json.resolve())
    image_path = Path(seg["image"]).resolve()
    source_detection = seg["source_detection"]
    mask_path = Path(seg["mask_path"]).resolve()
    stats = mask_stats(mask_path)
    extrinsics, extrinsics_source = load_extrinsics(args.extrinsics_json.resolve() if args.extrinsics_json else None)
    calib = load_calib(args.calib_file.resolve() if args.calib_file else None)

    image = Image.open(image_path)
    # Use the bottom point by default because it is often closest to the
    # contact/surface point needed for navigation.
    x_px, y_px = stats["bottom_point_px"]
    if not (0 <= x_px < image.width):
        x_px, y_px = bbox_center(source_detection["box"])

    bearing_rad = pixel_to_base_bearing_rad(
        x_px,
        y_px,
        image.width,
        args.horizontal_fov_deg,
        extrinsics,
        calib,
    )
    laser_angle_rad = base_bearing_to_laser_angle_rad(bearing_rad, extrinsics)
    scan_fusion = sample_scan_metric_range(args.scan.resolve(), laser_angle_rad, args.scan_window_deg)
    target_base = estimate_target_base_position(bearing_rad, scan_fusion["metric_range_m"], extrinsics)
    rel_depth = depth_stats(args.depth_npy.resolve(), stats["mask"])

    result = {
        "image": str(image_path),
        "mask_path": str(mask_path),
        "scan": str(args.scan.resolve()),
        "calib_file": str(args.calib_file.resolve()) if args.calib_file else None,
        "horizontal_fov_deg": args.horizontal_fov_deg,
        "scan_window_deg": args.scan_window_deg,
        "extrinsics": {
            "source": extrinsics_source or "legacy_constants",
            "params": extrinsics.as_dict(),
        },
        "bearing_method": "calibrated_intrinsics" if calib is not None else "horizontal_fov_approx",
        "camera_height_m": extrinsics.camera_z_m,
        "laser_height_m": extrinsics.laser_z_m,
        "query_pixel": [x_px, y_px],
        "mask": {
            "area_px": stats["area_px"],
            "bbox_px": stats["bbox_px"],
            "centroid_px": stats["centroid_px"],
            "bottom_point_px": stats["bottom_point_px"],
        },
        "source_detection": source_detection,
        "relative_depth": rel_depth,
        "scan_fusion": scan_fusion,
        "target_base": target_base,
        "notes": [
            "When --calib-file is set, target bearing is computed from left-camera intrinsics and calibrated camera rpy.",
            "Without --calib-file, target bearing falls back to the older horizontal-FOV approximation.",
            "LaserScan is 2D at about 0.30m height, so it measures surfaces intersecting that plane; low table legs are suitable, tabletops may not be.",
            "Depth Anything V2 values are relative; scan_fusion.metric_range_m provides the metric range scale along the target bearing.",
        ],
    }

    output_dir = args.output_dir.resolve()
    stem = image_path.stem
    image_overlay = output_dir / f"{stem}_fused_image_overlay.png"
    topdown = output_dir / f"{stem}_fused_topdown.png"
    json_path = output_dir / f"{stem}_fused_depth_scan.json"
    draw_image_overlay(image_path, stats["mask"], source_detection["box"], x_px, result, image_overlay)
    draw_topdown(args.scan.resolve(), result, topdown, extrinsics)
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"bearing: {target_base['bearing_base_deg']:.2f} deg")
    print(f"metric lidar range: {scan_fusion['metric_range_m']:.3f} m")
    print(f"target base xy: [{target_base['base_xy_m'][0]:.3f}, {target_base['base_xy_m'][1]:.3f}] m")
    print(f"camera planar distance: {target_base['camera_planar_distance_m']:.3f} m")
    print(f"image overlay: {image_overlay}")
    print(f"topdown: {topdown}")
    print(f"json: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
