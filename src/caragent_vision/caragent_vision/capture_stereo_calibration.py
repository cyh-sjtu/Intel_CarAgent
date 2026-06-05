"""Capture stereo checkerboard image pairs from the Huibo side-by-side camera."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Capture stereo calibration image pairs.")
    parser.add_argument("--device", default="/dev/video0")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--left-width", type=int, default=640)
    parser.add_argument("--right-width", type=int, default=640)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--prefix", default="stereo")
    parser.add_argument("--fps", type=float, default=30.0)
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


def main() -> int:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    left_dir = args.output_dir / "left"
    right_dir = args.output_dir / "right"
    raw_dir = args.output_dir / "raw"
    left_dir.mkdir(exist_ok=True)
    right_dir.mkdir(exist_ok=True)
    raw_dir.mkdir(exist_ok=True)

    cap = open_camera(args.device, args.width, args.height, args.fps)
    print("Press SPACE to save a pair, q/ESC to quit.")
    print(f"Output: {args.output_dir}")
    count = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.02)
                continue
            left = frame[:, : args.left_width]
            right = frame[:, args.left_width : args.left_width + args.right_width]
            preview = cv2.hconcat([left, right])
            cv2.putText(
                preview,
                f"saved={count} SPACE=save q=quit",
                (16, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("stereo calibration capture", preview)
            key = cv2.waitKey(1) & 0xFF
            if key in {ord("q"), 27}:
                break
            if key == 32:
                stem = f"{args.prefix}_{count:03d}.png"
                cv2.imwrite(str(raw_dir / stem), frame)
                cv2.imwrite(str(left_dir / stem), left)
                cv2.imwrite(str(right_dir / stem), right)
                print(f"saved pair {count}: {stem}")
                count += 1
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
