"""Interactively collect image pixel to LaserScan beam correspondences.

Usage idea:
1. Put a thin vertical object at LiDAR scan height.
2. Capture/load a left image and the matching scan npz.
3. Click the object contact point in the image.
4. Use left/right keys to select the corresponding scan beam/cluster.
5. Press s to append one JSONL correspondence.

This script is intentionally lightweight and offline-friendly; it does not need
ROS to run if image and scan files already exist.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np


LASER_X_M = 0.12
LASER_Y_M = 0.0
LASER_YAW_RAD = math.pi


def scan_to_base(scan_file: Path) -> dict[str, np.ndarray | float]:
    scan = np.load(scan_file)
    ranges = scan["ranges"].astype(np.float32)
    angle_min = float(scan["angle_min"])
    angle_increment = float(scan["angle_increment"])
    range_min = float(scan["range_min"])
    range_max = float(scan["range_max"])
    angles = angle_min + np.arange(len(ranges), dtype=np.float32) * angle_increment
    valid = np.isfinite(ranges) & (ranges >= range_min) & (ranges <= range_max)
    x_l = ranges * np.cos(angles)
    y_l = ranges * np.sin(angles)
    c = math.cos(LASER_YAW_RAD)
    s = math.sin(LASER_YAW_RAD)
    x_b = LASER_X_M + c * x_l - s * y_l
    y_b = LASER_Y_M + s * x_l + c * y_l
    return {
        "ranges": ranges,
        "angles": angles,
        "valid": valid,
        "x_base": x_b,
        "y_base": y_b,
        "angle_min": angle_min,
        "angle_increment": angle_increment,
    }


def draw_scan_panel(scan_data: dict[str, np.ndarray | float], selected_idx: int, width: int, height: int) -> np.ndarray:
    panel = np.full((height, width, 3), 246, dtype=np.uint8)
    valid = scan_data["valid"].astype(bool)
    x_b = scan_data["x_base"].astype(np.float32)
    y_b = scan_data["y_base"].astype(np.float32)
    scale = 90.0
    origin = np.array([width * 0.50, height * 0.82], dtype=np.float32)

    def to_px(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return origin[0] - y * scale, origin[1] - x * scale

    for gx in np.arange(0.0, 4.1, 0.5):
        yy = int(origin[1] - gx * scale)
        cv2.line(panel, (0, yy), (width, yy), (220, 220, 214), 1)
        cv2.putText(panel, f"x={gx:.1f}", (6, yy - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (110, 110, 100), 1)
    for gy in np.arange(-2.5, 2.6, 0.5):
        xx = int(origin[0] - gy * scale)
        cv2.line(panel, (xx, 0), (xx, height), (220, 220, 214), 1)

    px, py = to_px(x_b[valid], y_b[valid])
    for x, y in zip(px.astype(int), py.astype(int)):
        if 0 <= x < width and 0 <= y < height:
            cv2.circle(panel, (x, y), 2, (70, 120, 180), -1)

    base = (int(origin[0]), int(origin[1]))
    cv2.circle(panel, base, 5, (20, 20, 20), -1)
    cv2.putText(panel, "base", (base[0] + 8, base[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1)

    if 0 <= selected_idx < len(valid) and valid[selected_idx]:
        sx, sy = to_px(np.array([x_b[selected_idx]]), np.array([y_b[selected_idx]]))
        p = (int(sx[0]), int(sy[0]))
        cv2.circle(panel, p, 7, (20, 30, 230), -1)
        cv2.line(panel, base, p, (20, 30, 230), 2)
        text = f"beam {selected_idx} r={scan_data['ranges'][selected_idx]:.3f} angle={scan_data['angles'][selected_idx]:.3f}"
    else:
        text = f"beam {selected_idx} invalid"
    cv2.putText(panel, text, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1)
    return panel


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect LiDAR-camera manual correspondences.")
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--scan", required=True, type=Path)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--start-beam", type=int)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(args.image)
    scan_data = scan_to_base(args.scan.resolve())
    valid_indices = np.flatnonzero(scan_data["valid"])
    selected_idx = int(args.start_beam) if args.start_beam is not None else int(valid_indices[len(valid_indices) // 2])
    clicked: tuple[int, int] | None = None

    def on_mouse(event, x, y, flags, userdata):
        nonlocal clicked
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked = (int(x), int(y))

    win = "lidar-camera correspondences"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    print("Controls: click image point, left/right/a/d choose beam, s save, q quit")
    while True:
        canvas_img = image.copy()
        if clicked is not None:
            cv2.drawMarker(canvas_img, clicked, (0, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=18, thickness=2)
        cv2.putText(canvas_img, "click image point; choose beam; s=save q=quit", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3)
        cv2.putText(canvas_img, "click image point; choose beam; s=save q=quit", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)
        panel = draw_scan_panel(scan_data, selected_idx, image.shape[1], image.shape[0])
        canvas = np.vstack([canvas_img, panel])
        cv2.imshow(win, canvas)
        key = cv2.waitKey(30) & 0xFF
        if key in (ord("q"), 27):
            break
        if key in (81, ord("a")):
            selected_idx = max(0, selected_idx - 1)
        elif key in (83, ord("d")):
            selected_idx = min(len(scan_data["ranges"]) - 1, selected_idx + 1)
        elif key in (ord("A"),):
            selected_idx = max(0, selected_idx - 10)
        elif key in (ord("D"),):
            selected_idx = min(len(scan_data["ranges"]) - 1, selected_idx + 10)
        elif key == ord("s"):
            if clicked is None:
                print("No image point clicked yet.")
                continue
            if not bool(scan_data["valid"][selected_idx]):
                print(f"Selected beam {selected_idx} is invalid.")
                continue
            record = {
                "image": str(args.image.resolve()),
                "scan": str(args.scan.resolve()),
                "image_point": [clicked[0], clicked[1]],
                "scan_point": {
                    "beam_index": int(selected_idx),
                    "range_m": float(scan_data["ranges"][selected_idx]),
                    "angle_rad": float(scan_data["angles"][selected_idx]),
                    "base_xy_m": [
                        float(scan_data["x_base"][selected_idx]),
                        float(scan_data["y_base"][selected_idx]),
                    ],
                },
            }
            with args.output_jsonl.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"saved: pixel={record['image_point']} beam={selected_idx} range={record['scan_point']['range_m']:.3f}")

    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
