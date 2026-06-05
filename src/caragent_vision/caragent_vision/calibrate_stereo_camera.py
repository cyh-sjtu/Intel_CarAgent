"""Calibrate Huibo stereo camera from checkerboard image pairs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Calibrate stereo camera from checkerboard pairs.")
    parser.add_argument("--image-dir", required=True, type=Path, help="Directory containing left/ and right/ images.")
    parser.add_argument("--output", required=True, type=Path, help="Output stereo_calibration.npz.")
    parser.add_argument("--cols", type=int, required=True, help="Checkerboard inner corners per row.")
    parser.add_argument("--rows", type=int, required=True, help="Checkerboard inner corners per column.")
    parser.add_argument("--square-size", type=float, required=True, help="Checker square size in meters.")
    parser.add_argument("--show", action="store_true")
    return parser


def make_object_points(cols: int, rows: int, square_size: float) -> np.ndarray:
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= float(square_size)
    return objp


def find_corners(image_path: Path, cols: int, rows: int, show: bool) -> tuple[bool, np.ndarray | None, tuple[int, int]]:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    pattern = (cols, rows)
    ok, corners = cv2.findChessboardCorners(
        gray,
        pattern,
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
    )
    if ok:
        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            30,
            0.001,
        )
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    if show:
        vis = image.copy()
        cv2.drawChessboardCorners(vis, pattern, corners if corners is not None else [], ok)
        cv2.imshow(str(image_path.name), vis)
        cv2.waitKey(80)
    return ok, corners, (gray.shape[1], gray.shape[0])


def main() -> int:
    args = build_parser().parse_args()
    left_dir = args.image_dir / "left"
    right_dir = args.image_dir / "right"
    left_paths = sorted(left_dir.glob("*.png"))
    right_paths = sorted(right_dir.glob("*.png"))
    right_by_name = {p.name: p for p in right_paths}
    pairs = [(lp, right_by_name[lp.name]) for lp in left_paths if lp.name in right_by_name]
    if not pairs:
        raise RuntimeError(f"No matching left/right pairs found under {args.image_dir}")

    objp = make_object_points(args.cols, args.rows, args.square_size)
    objpoints: list[np.ndarray] = []
    imgpoints_l: list[np.ndarray] = []
    imgpoints_r: list[np.ndarray] = []
    used: list[str] = []
    image_size: tuple[int, int] | None = None

    for left_path, right_path in pairs:
        ok_l, corners_l, size_l = find_corners(left_path, args.cols, args.rows, args.show)
        ok_r, corners_r, size_r = find_corners(right_path, args.cols, args.rows, args.show)
        if size_l != size_r:
            print(f"skip {left_path.name}: size mismatch {size_l} vs {size_r}")
            continue
        image_size = size_l
        if not ok_l or not ok_r or corners_l is None or corners_r is None:
            print(f"skip {left_path.name}: corners left={ok_l} right={ok_r}")
            continue
        objpoints.append(objp)
        imgpoints_l.append(corners_l)
        imgpoints_r.append(corners_r)
        used.append(left_path.name)
        print(f"use {left_path.name}")

    if args.show:
        cv2.destroyAllWindows()
    if image_size is None:
        raise RuntimeError("Could not read calibration images.")
    if len(objpoints) < 10:
        raise RuntimeError(f"Need at least 10 good stereo pairs, got {len(objpoints)}.")

    flags = 0
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        100,
        1e-5,
    )
    rms_l, mtx_l, dist_l, rvecs_l, tvecs_l = cv2.calibrateCamera(
        objpoints, imgpoints_l, image_size, None, None
    )
    rms_r, mtx_r, dist_r, rvecs_r, tvecs_r = cv2.calibrateCamera(
        objpoints, imgpoints_r, image_size, None, None
    )
    stereo_rms, mtx_l, dist_l, mtx_r, dist_r, R, T, E, F = cv2.stereoCalibrate(
        objpoints,
        imgpoints_l,
        imgpoints_r,
        mtx_l,
        dist_l,
        mtx_r,
        dist_r,
        image_size,
        criteria=criteria,
        flags=cv2.CALIB_FIX_INTRINSIC | flags,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        mtx_l=mtx_l,
        dist_l=dist_l,
        mtx_r=mtx_r,
        dist_r=dist_r,
        R=R,
        T=T,
        E=E,
        F=F,
        image_size=np.array(image_size, dtype=np.int32),
        square_size_m=np.array([args.square_size], dtype=np.float32),
    )
    report = {
        "output": str(args.output),
        "image_size": list(image_size),
        "used_pairs": used,
        "num_used_pairs": len(used),
        "rms_left": float(rms_l),
        "rms_right": float(rms_r),
        "rms_stereo": float(stereo_rms),
        "baseline_m": float(np.linalg.norm(T)),
        "T": T.reshape(-1).astype(float).tolist(),
    }
    report_path = args.output.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
