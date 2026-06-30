"""Draw LaserScan diagnostics in intuitive base_link coordinates."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


LASER_X_M = 0.12
LASER_Y_M = 0.0
LASER_YAW_RAD = math.pi


def scan_to_base(scan_path: Path) -> tuple[np.ndarray, np.ndarray]:
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
    return np.stack([x_b, y_b], axis=1), ranges[valid]


def draw_scan(scan_path: Path, output_path: Path, meters_per_side: float, pixels: int) -> None:
    points, ranges = scan_to_base(scan_path)
    image = Image.new("RGB", (pixels, pixels), (18, 20, 24))
    draw = ImageDraw.Draw(image)
    center = (pixels // 2, int(pixels * 0.72))
    scale = pixels / meters_per_side
    half_y = meters_per_side / 2.0
    max_x = meters_per_side * 0.72
    min_x = -meters_per_side * 0.28

    def to_px(x: float, y: float) -> tuple[int, int]:
        return int(center[0] - y * scale), int(center[1] - x * scale)

    # Grid.
    for x in np.arange(math.floor(min_x * 2) / 2, max_x + 0.001, 0.5):
        p0 = to_px(float(x), -half_y)
        p1 = to_px(float(x), half_y)
        draw.line([p0[0], p0[1], p1[0], p1[1]], fill=(42, 45, 52))
        if 0 <= p0[1] < pixels:
            draw.text((6, p0[1] - 7), f"x={x:.1f}", fill=(95, 100, 110))
    for y in np.arange(-half_y, half_y + 0.001, 0.5):
        p0 = to_px(min_x, float(y))
        p1 = to_px(max_x, float(y))
        draw.line([p0[0], p0[1], p1[0], p1[1]], fill=(42, 45, 52))

    # Axes.
    base = to_px(0.0, 0.0)
    front = to_px(0.8, 0.0)
    left = to_px(0.0, 0.8)
    draw.line([base[0], base[1], front[0], front[1]], fill=(255, 255, 255), width=3)
    draw.text((front[0] + 8, front[1] - 10), "+X forward", fill=(255, 255, 255))
    draw.line([base[0], base[1], left[0], left[1]], fill=(120, 220, 255), width=3)
    draw.text((left[0] - 85, left[1] - 10), "+Y left", fill=(120, 220, 255))

    # Categorize points.
    x = points[:, 0]
    y = points[:, 1]
    front_wall = (x > 2.3) & (x < 3.1) & (np.abs(y) < 1.5)
    near = ranges < 1.0
    side = np.abs(y) > 1.5

    for idx, (px_m, py_m) in enumerate(points):
        p = to_px(float(px_m), float(py_m))
        if not (0 <= p[0] < pixels and 0 <= p[1] < pixels):
            continue
        color = (110, 170, 255)
        radius = 2
        if front_wall[idx]:
            color = (255, 230, 0)
            radius = 3
        elif near[idx]:
            color = (255, 90, 90)
            radius = 3
        elif side[idx]:
            color = (150, 150, 180)
        draw.ellipse([p[0] - radius, p[1] - radius, p[0] + radius, p[1] + radius], fill=color)

    laser = to_px(LASER_X_M, LASER_Y_M)
    draw.ellipse([laser[0] - 6, laser[1] - 6, laser[0] + 6, laser[1] + 6], fill=(255, 80, 80))
    draw.text((laser[0] + 8, laser[1] - 12), "laser", fill=(255, 80, 80))
    draw.ellipse([base[0] - 6, base[1] - 6, base[0] + 6, base[1] + 6], fill=(255, 255, 255))
    draw.text((base[0] + 8, base[1] + 8), "base_link", fill=(255, 255, 255))

    if front_wall.any():
        wall_x = float(np.median(x[front_wall]))
        p0 = to_px(wall_x, -1.5)
        p1 = to_px(wall_x, 1.5)
        draw.line([p0[0], p0[1], p1[0], p1[1]], fill=(255, 255, 0), width=2)
        draw.text((p1[0] + 8, p1[1] - 18), f"front cluster median x={wall_x:.2f}m", fill=(255, 255, 0))

    lines = [
        f"scan: {scan_path.name}",
        "up=+X forward, left=+Y robot-left",
        "yellow=front wall-like cluster (x 2.3..3.1m, |y|<1.5m)",
        "red=near returns (<1m), blue=other valid returns",
        f"valid points={len(points)}, front cluster={int(front_wall.sum())}",
    ]
    y_text = 8
    for line in lines:
        draw.rectangle([6, y_text - 2, min(pixels - 1, 14 + len(line) * 7), y_text + 14], fill=(0, 0, 0))
        draw.text((10, y_text), line, fill=(255, 255, 255))
        y_text += 17

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Draw LaserScan diagnostic topdown.")
    parser.add_argument("--scan", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--meters-per-side", type=float, default=5.0)
    parser.add_argument("--pixels", type=int, default=1000)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    draw_scan(args.scan.resolve(), args.output.resolve(), args.meters_per_side, args.pixels)
    print(f"scan diagnostic: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
