"""Capture stereo checkerboard image pairs from the Huibo side-by-side camera."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Capture stereo calibration image pairs.")
    parser.add_argument("--device", default="/dev/video0")
    parser.add_argument("--width", type=int, default=3840)
    parser.add_argument("--height", type=int, default=1200)
    parser.add_argument("--left-width", type=int, default=1920)
    parser.add_argument("--right-width", type=int, default=1920)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--prefix", default="stereo")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--cols", type=int, default=0, help="Checkerboard inner corners per row; enables live detection.")
    parser.add_argument("--rows", type=int, default=0, help="Checkerboard inner corners per column; enables live detection.")
    parser.add_argument("--require-corners", action="store_true", help="Only save when both left and right corners are detected.")
    return parser


def open_camera(device: str, width: int, height: int, fps: float) -> cv2.VideoCapture:
    candidate = int(device) if str(device).isdigit() else device
    cap = cv2.VideoCapture(candidate, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap = cv2.VideoCapture(candidate)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera: {device}")
    return cap


def _sharpness(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _find_corners(image: np.ndarray, cols: int, rows: int) -> tuple[bool, np.ndarray | None]:
    if cols <= 0 or rows <= 0:
        return False, None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    pattern = (cols, rows)
    if hasattr(cv2, "findChessboardCornersSB"):
        ok, corners = cv2.findChessboardCornersSB(
            gray,
            pattern,
            flags=cv2.CALIB_CB_NORMALIZE_IMAGE,
        )
        if ok:
            return True, corners
    ok, corners = cv2.findChessboardCorners(
        gray,
        pattern,
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
    )
    if ok:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return bool(ok), corners if ok else None


def _draw_detection(image: np.ndarray, cols: int, rows: int, ok: bool, corners: np.ndarray | None) -> np.ndarray:
    view = image.copy()
    if cols > 0 and rows > 0 and ok and corners is not None:
        cv2.drawChessboardCorners(view, (cols, rows), corners, ok)
    return view


def _next_index(left_dir: Path, prefix: str) -> int:
    max_index = -1
    marker = f"{prefix}_"
    for path in left_dir.glob(f"{prefix}_*.png"):
        stem = path.stem
        if not stem.startswith(marker):
            continue
        try:
            value = int(stem[len(marker) :])
        except ValueError:
            continue
        max_index = max(max_index, value)
    return max_index + 1


def main() -> int:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    left_dir = args.output_dir / "left"
    right_dir = args.output_dir / "right"
    raw_dir = args.output_dir / "raw"
    left_dir.mkdir(exist_ok=True)
    right_dir.mkdir(exist_ok=True)
    raw_dir.mkdir(exist_ok=True)
    metadata_path = args.output_dir / "capture_metadata.jsonl"
    config_path = args.output_dir / "capture_config.json"
    config_path.write_text(
        json.dumps(
            {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "device": args.device,
                "width": args.width,
                "height": args.height,
                "left_width": args.left_width,
                "right_width": args.right_width,
                "fps": args.fps,
                "cols": args.cols,
                "rows": args.rows,
                "require_corners": bool(args.require_corners),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    cap = open_camera(args.device, args.width, args.height, args.fps)
    print("Press SPACE to save a pair, q/ESC to quit.")
    if args.cols > 0 and args.rows > 0:
        print(f"Live checkerboard detection enabled: inner corners {args.cols}x{args.rows}")
    print(f"Output: {args.output_dir}")
    count = _next_index(left_dir, args.prefix)
    if count > 0:
        print(f"Resuming from index {count}; existing pairs will not be overwritten.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.02)
                continue
            left = frame[:, : args.left_width]
            right = frame[:, args.left_width : args.left_width + args.right_width]
            left_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
            right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
            sharp_l = _sharpness(left_gray)
            sharp_r = _sharpness(right_gray)
            ok_l, corners_l = _find_corners(left, args.cols, args.rows)
            ok_r, corners_r = _find_corners(right, args.cols, args.rows)
            preview = cv2.hconcat(
                [
                    _draw_detection(left, args.cols, args.rows, ok_l, corners_l),
                    _draw_detection(right, args.cols, args.rows, ok_r, corners_r),
                ]
            )
            status_color = (0, 220, 0) if ok_l and ok_r else (0, 180, 255)
            cv2.putText(
                preview,
                f"saved={count} SPACE=save q=quit corners L/R={int(ok_l)}/{int(ok_r)} sharp={sharp_l:.0f}/{sharp_r:.0f}",
                (16, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                status_color,
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("stereo calibration capture", preview)
            key = cv2.waitKey(1) & 0xFF
            if key in {ord("q"), 27}:
                break
            if key == 32:
                if args.require_corners and args.cols > 0 and args.rows > 0 and not (ok_l and ok_r):
                    print(f"not saved: corners left={ok_l} right={ok_r}")
                    continue
                stem = f"{args.prefix}_{count:03d}.png"
                cv2.imwrite(str(raw_dir / stem), frame)
                cv2.imwrite(str(left_dir / stem), left)
                cv2.imwrite(str(right_dir / stem), right)
                record = {
                    "saved_at": datetime.now().isoformat(timespec="milliseconds"),
                    "index": count,
                    "stem": stem,
                    "raw_path": str(raw_dir / stem),
                    "left_path": str(left_dir / stem),
                    "right_path": str(right_dir / stem),
                    "raw_size": [int(frame.shape[1]), int(frame.shape[0])],
                    "left_size": [int(left.shape[1]), int(left.shape[0])],
                    "right_size": [int(right.shape[1]), int(right.shape[0])],
                    "checkerboard": {"cols": args.cols, "rows": args.rows},
                    "corners_left": bool(ok_l),
                    "corners_right": bool(ok_r),
                    "sharpness_left": sharp_l,
                    "sharpness_right": sharp_r,
                }
                with metadata_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                print(f"saved pair {count}: {stem} corners L/R={ok_l}/{ok_r} sharp={sharp_l:.0f}/{sharp_r:.0f}")
                count += 1
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
