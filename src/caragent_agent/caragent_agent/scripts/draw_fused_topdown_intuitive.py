"""Draw an intuitive top-down localization view.

Image convention:
  screen up    = base_link +X = robot forward
  screen left  = base_link +Y = robot left
  screen right = base_link -Y = robot right

This is easier to read than a mathematical x-right/y-up plot when reasoning
about the car from above.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


LASER_X_M = 0.12
LASER_Y_M = 0.0
LASER_YAW_RAD = math.pi
CAMERA_LEFT_X_M = 0.30
CAMERA_LEFT_Y_M = 0.03
CAMERA_RIGHT_X_M = 0.30
CAMERA_RIGHT_Y_M = -0.03


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def scan_to_base_points(scan_path: Path) -> np.ndarray:
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
    c = math.cos(LASER_YAW_RAD)
    s = math.sin(LASER_YAW_RAD)
    x_b = LASER_X_M + c * x_l - s * y_l
    y_b = LASER_Y_M + s * x_l + c * y_l
    return np.stack([x_b, y_b], axis=1)


def draw_intuitive_topdown(
    fused: dict[str, Any],
    output_path: Path,
    meters_per_side: float = 5.0,
    pixels: int = 900,
) -> None:
    scan_path = Path(fused["scan"]).resolve()
    points = scan_to_base_points(scan_path)
    image = Image.new("RGB", (pixels, pixels), (18, 20, 24))
    draw = ImageDraw.Draw(image)
    center = (pixels // 2, pixels // 2)
    scale = pixels / meters_per_side

    def to_px(x: float, y: float) -> tuple[int, int]:
        # x forward -> up, y left -> screen left.
        return (int(center[0] - y * scale), int(center[1] - x * scale))

    half = meters_per_side / 2.0

    # Grid and labels.
    for m in np.arange(-half, half + 0.001, 0.5):
        px0, py0 = to_px(-half, float(m))
        px1, py1 = to_px(half, float(m))
        draw.line([px0, py0, px1, py1], fill=(42, 45, 52))
        px0, py0 = to_px(float(m), -half)
        px1, py1 = to_px(float(m), half)
        draw.line([px0, py0, px1, py1], fill=(42, 45, 52))

    # Axes.
    bx, by = to_px(0.0, 0.0)
    fx, fy = to_px(0.8, 0.0)
    lx, ly = to_px(0.0, 0.8)
    rx, ry = to_px(0.0, -0.8)
    draw.line([bx, by, fx, fy], fill=(255, 255, 255), width=3)
    draw.polygon([(fx, fy), (fx - 7, fy + 16), (fx + 7, fy + 16)], fill=(255, 255, 255))
    draw.text((fx + 8, fy - 8), "+X forward", fill=(255, 255, 255))
    draw.line([bx, by, lx, ly], fill=(120, 220, 255), width=3)
    draw.text((lx - 95, ly - 8), "+Y left", fill=(120, 220, 255))
    draw.text((rx + 8, ry - 8), "right side", fill=(160, 160, 160))

    # Scan points.
    for x, y in points:
        px, py = to_px(float(x), float(y))
        if 0 <= px < pixels and 0 <= py < pixels:
            draw.ellipse([px - 1, py - 1, px + 1, py + 1], fill=(110, 170, 255))

    # Robot footprint.
    base_px = to_px(0.0, 0.0)
    front_px = to_px(0.225, 0.0)
    rear_px = to_px(-0.225, 0.0)
    left_px = to_px(0.0, 0.21)
    right_px = to_px(0.0, -0.21)
    footprint = [
        to_px(0.225, 0.21),
        to_px(0.225, -0.21),
        to_px(-0.225, -0.21),
        to_px(-0.225, 0.21),
    ]
    draw.polygon(footprint, outline=(235, 235, 235), fill=(45, 48, 56))
    draw.line([rear_px[0], rear_px[1], front_px[0], front_px[1]], fill=(255, 255, 255), width=2)
    draw.line([right_px[0], right_px[1], left_px[0], left_px[1]], fill=(120, 220, 255), width=2)
    draw.ellipse([base_px[0] - 6, base_px[1] - 6, base_px[0] + 6, base_px[1] + 6], fill=(255, 255, 255))
    draw.text((base_px[0] + 8, base_px[1] + 8), "base_link", fill=(255, 255, 255))

    # Sensors.
    laser_px = to_px(LASER_X_M, LASER_Y_M)
    cam_l_px = to_px(CAMERA_LEFT_X_M, CAMERA_LEFT_Y_M)
    cam_r_px = to_px(CAMERA_RIGHT_X_M, CAMERA_RIGHT_Y_M)
    draw.ellipse([laser_px[0] - 6, laser_px[1] - 6, laser_px[0] + 6, laser_px[1] + 6], fill=(255, 80, 80))
    draw.text((laser_px[0] + 8, laser_px[1] - 12), "laser", fill=(255, 80, 80))
    draw.ellipse([cam_l_px[0] - 6, cam_l_px[1] - 6, cam_l_px[0] + 6, cam_l_px[1] + 6], fill=(80, 255, 120))
    draw.text((cam_l_px[0] - 105, cam_l_px[1] - 14), "camera_left", fill=(80, 255, 120))
    draw.ellipse([cam_r_px[0] - 5, cam_r_px[1] - 5, cam_r_px[0] + 5, cam_r_px[1] + 5], fill=(80, 210, 255))
    draw.text((cam_r_px[0] + 10, cam_r_px[1] - 12), "camera_right", fill=(80, 210, 255))

    # Selected scan beams, if present.
    selected_ranges = fused.get("scan_fusion", {}).get("selected_ranges_m", [])
    selected_angles = fused.get("scan_fusion", {}).get("selected_angles_rad", [])
    for r, a_laser in zip(selected_ranges, selected_angles):
        x_l = float(r) * math.cos(float(a_laser))
        y_l = float(r) * math.sin(float(a_laser))
        c = math.cos(LASER_YAW_RAD)
        s = math.sin(LASER_YAW_RAD)
        x_b = LASER_X_M + c * x_l - s * y_l
        y_b = LASER_Y_M + s * x_l + c * y_l
        p = to_px(x_b, y_b)
        draw.line([laser_px[0], laser_px[1], p[0], p[1]], fill=(255, 230, 0), width=1)
        draw.ellipse([p[0] - 3, p[1] - 3, p[0] + 3, p[1] + 3], fill=(255, 230, 0))

    # Target.
    tx, ty = fused["target_base"]["base_xy_m"]
    target_px = to_px(float(tx), float(ty))
    draw.line([laser_px[0], laser_px[1], target_px[0], target_px[1]], fill=(255, 255, 0), width=3)
    draw.ellipse([target_px[0] - 9, target_px[1] - 9, target_px[0] + 9, target_px[1] + 9], fill=(255, 255, 0))
    draw.text(
        (target_px[0] + 10, target_px[1] - 20),
        f"target ({float(tx):.2f}, {float(ty):.2f}) m",
        fill=(255, 255, 0),
    )

    # Legend.
    lines = [
        "Intuitive topdown: up=+X forward, left=+Y robot-left",
        "Blue dots=LaserScan points in base_link",
        "Yellow rays/points=scan beams selected by image bearing",
        "Green=camera_left; cyan=camera_right; red=laser",
    ]
    y_text = 8
    for line in lines:
        draw.rectangle([6, y_text - 2, min(pixels - 1, 14 + len(line) * 7), y_text + 14], fill=(0, 0, 0))
        draw.text((10, y_text), line, fill=(255, 255, 255))
        y_text += 17

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Draw intuitive topdown object localization view.")
    parser.add_argument("--fused-json", required=True, type=Path)
    parser.add_argument("--output", default="", type=Path)
    parser.add_argument("--meters-per-side", default=5.0, type=float)
    parser.add_argument("--pixels", default=900, type=int)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    fused_path = args.fused_json.resolve()
    fused = load_json(fused_path)
    output = args.output.resolve() if args.output else fused_path.with_name(f"{fused_path.stem}_topdown_intuitive.png")
    draw_intuitive_topdown(fused, output, meters_per_side=args.meters_per_side, pixels=args.pixels)
    print(f"intuitive topdown: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
