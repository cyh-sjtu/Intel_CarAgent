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
    parser.add_argument("--save-overlays", action="store_true", help="Save checkerboard corner preview images.")
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
    corners = None
    if hasattr(cv2, "findChessboardCornersSB"):
        ok, corners = cv2.findChessboardCornersSB(
            gray,
            pattern,
            flags=cv2.CALIB_CB_NORMALIZE_IMAGE,
        )
    else:
        ok = False
    if not ok:
        ok, corners = cv2.findChessboardCorners(
            gray,
            pattern,
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
        )
    if ok and corners is not None:
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


def save_corner_overlay(
    image_path: Path,
    output_path: Path,
    cols: int,
    rows: int,
    ok: bool,
    corners: np.ndarray | None,
) -> None:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        return
    cv2.drawChessboardCorners(image, (cols, rows), corners if corners is not None else [], ok)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)


def per_view_errors(
    objpoints: list[np.ndarray],
    imgpoints: list[np.ndarray],
    rvecs: tuple[np.ndarray, ...] | list[np.ndarray],
    tvecs: tuple[np.ndarray, ...] | list[np.ndarray],
    mtx: np.ndarray,
    dist: np.ndarray,
) -> list[float]:
    errors: list[float] = []
    for objp, imgp, rvec, tvec in zip(objpoints, imgpoints, rvecs, tvecs):
        projected, _ = cv2.projectPoints(objp, rvec, tvec, mtx, dist)
        err = cv2.norm(imgp, projected, cv2.NORM_L2) / max(1, len(projected))
        errors.append(float(err))
    return errors


def align_corner_order(
    corners_l: np.ndarray,
    corners_r: np.ndarray,
) -> tuple[np.ndarray, bool, float, float]:
    """Keep left/right checkerboard corner indices consistent.

    OpenCV can return the same symmetric checkerboard in opposite traversal
    directions in one camera. Stereo calibration needs point i on the left to
    be the same physical corner as point i on the right, so choose the right
    ordering with the smaller mean left/right pixel distance.
    """

    direct = float(np.mean(np.linalg.norm(corners_l.reshape(-1, 2) - corners_r.reshape(-1, 2), axis=1)))
    reversed_error = float(
        np.mean(np.linalg.norm(corners_l.reshape(-1, 2) - corners_r[::-1].reshape(-1, 2), axis=1))
    )
    if reversed_error < direct:
        return corners_r[::-1].copy(), True, direct, reversed_error
    return corners_r, False, direct, reversed_error


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
    rejected: list[dict[str, object]] = []
    corner_order: list[dict[str, object]] = []
    image_size: tuple[int, int] | None = None
    overlay_dir = args.output.parent / "corner_overlays"

    for left_path, right_path in pairs:
        ok_l, corners_l, size_l = find_corners(left_path, args.cols, args.rows, args.show)
        ok_r, corners_r, size_r = find_corners(right_path, args.cols, args.rows, args.show)
        if args.save_overlays:
            save_corner_overlay(left_path, overlay_dir / "left" / left_path.name, args.cols, args.rows, ok_l, corners_l)
            save_corner_overlay(right_path, overlay_dir / "right" / right_path.name, args.cols, args.rows, ok_r, corners_r)
        if size_l != size_r:
            print(f"skip {left_path.name}: size mismatch {size_l} vs {size_r}")
            rejected.append({"name": left_path.name, "reason": "size_mismatch", "left_size": size_l, "right_size": size_r})
            continue
        image_size = size_l
        if not ok_l or not ok_r or corners_l is None or corners_r is None:
            print(f"skip {left_path.name}: corners left={ok_l} right={ok_r}")
            rejected.append({"name": left_path.name, "reason": "corners_not_found", "left_ok": bool(ok_l), "right_ok": bool(ok_r)})
            continue
        corners_r, reversed_right, pair_error_direct, pair_error_reversed = align_corner_order(corners_l, corners_r)
        corner_order.append(
            {
                "name": left_path.name,
                "right_reversed": bool(reversed_right),
                "direct_mean_px": pair_error_direct,
                "reversed_mean_px": pair_error_reversed,
            }
        )
        objpoints.append(objp)
        imgpoints_l.append(corners_l)
        imgpoints_r.append(corners_r)
        used.append(left_path.name)
        print(
            f"use {left_path.name}"
            + (" right_order=reversed" if reversed_right else "")
        )

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
    errors_l = per_view_errors(objpoints, imgpoints_l, rvecs_l, tvecs_l, mtx_l, dist_l)
    errors_r = per_view_errors(objpoints, imgpoints_r, rvecs_r, tvecs_r, mtx_r, dist_r)
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
        "rejected_pairs": rejected,
        "corner_order": corner_order,
        "num_used_pairs": len(used),
        "cols": int(args.cols),
        "rows": int(args.rows),
        "square_size_m": float(args.square_size),
        "rms_left": float(rms_l),
        "rms_right": float(rms_r),
        "rms_stereo": float(stereo_rms),
        "baseline_m": float(np.linalg.norm(T)),
        "T": T.reshape(-1).astype(float).tolist(),
        "per_view_errors": [
            {"name": name, "left_px": errors_l[index], "right_px": errors_r[index]}
            for index, name in enumerate(used)
        ],
        "worst_pairs": sorted(
            [
                {"name": name, "left_px": errors_l[index], "right_px": errors_r[index]}
                for index, name in enumerate(used)
            ],
            key=lambda item: max(float(item["left_px"]), float(item["right_px"])),
            reverse=True,
        )[:8],
    }
    report_path = args.output.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
