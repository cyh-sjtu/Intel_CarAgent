"""Calibrate left-camera pose and small LiDAR pitch from matched pixels.

Input is a JSONL file. Each line is one correspondence:

{
  "image_point": [u, v],
  "scan_point": {"range_m": 1.23, "angle_rad": -0.4}
}

The scan point is interpreted in the LiDAR frame at the LiDAR scan plane.
LiDAR translation and yaw are treated as measured installation values. The
optimizer adjusts LiDAR pitch plus the left-camera xyz/rpy so the projected
LiDAR point lands on the selected image pixel.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np


DEFAULT_LASER_X_M = 0.115
DEFAULT_LASER_Y_M = 0.0
DEFAULT_LASER_Z_M = 0.30
DEFAULT_LASER_YAW_RAD = math.pi
DEFAULT_CAMERA_X_M = 0.30
DEFAULT_CAMERA_Y_M = 0.03
DEFAULT_CAMERA_Z_M = 0.22

PARAM_LOWER = np.array(
    [
        -math.radians(1.5),  # laser_pitch_rad
        0.22,  # camera_x_m
        -0.06,  # camera_y_m
        0.14,  # camera_z_m
        -math.radians(20.0),  # camera_roll_rad
        -math.radians(20.0),  # camera_pitch_rad
        -math.radians(20.0),  # camera_yaw_rad
    ],
    dtype=np.float64,
)
PARAM_UPPER = np.array(
    [
        math.radians(1.5),
        0.38,
        0.12,
        0.32,
        math.radians(20.0),
        math.radians(20.0),
        math.radians(20.0),
    ],
    dtype=np.float64,
)


def load_calib(calib_file: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(calib_file)
    return data["mtx_l"].astype(np.float64), data["dist_l"].astype(np.float64)


def load_samples(path: Path) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        item = json.loads(line)
        if "image_point" not in item:
            raise ValueError(f"{path}:{line_no}: missing image_point")
        if "scan_point" not in item:
            raise ValueError(f"{path}:{line_no}: missing scan_point")
        samples.append(item)
    if len(samples) < 4:
        raise ValueError("Need at least 4 correspondences; 8+ is better.")
    return samples


def rodrigues_xyz(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    return rz @ ry @ rx


def params_to_dict(x: np.ndarray) -> dict[str, float]:
    laser_pitch, camera_x, camera_y, camera_z, camera_roll, camera_pitch, camera_yaw = x
    return {
        "laser_x_m": DEFAULT_LASER_X_M,
        "laser_y_m": DEFAULT_LASER_Y_M,
        "laser_z_m": DEFAULT_LASER_Z_M,
        "laser_roll_rad": 0.0,
        "laser_pitch_rad": float(laser_pitch),
        "laser_yaw_rad": DEFAULT_LASER_YAW_RAD,
        "camera_x_m": float(camera_x),
        "camera_y_m": float(camera_y),
        "camera_z_m": float(camera_z),
        "camera_roll_rad": float(camera_roll),
        "camera_pitch_rad": float(camera_pitch),
        "camera_yaw_rad": float(camera_yaw),
    }


def initial_params() -> np.ndarray:
    return np.array(
        [
            0.0,
            DEFAULT_CAMERA_X_M,
            DEFAULT_CAMERA_Y_M,
            DEFAULT_CAMERA_Z_M,
            0.0,
            0.0,
            0.0,
        ],
        dtype=np.float64,
    )


def project_sample(
    sample: dict[str, Any],
    x: np.ndarray,
    mtx: np.ndarray,
    dist: np.ndarray,
) -> tuple[np.ndarray, float]:
    laser_pitch, camera_x, camera_y, camera_z, camera_roll, camera_pitch, camera_yaw = x

    scan = sample["scan_point"]
    r = float(scan["range_m"])
    a = float(scan["angle_rad"])
    p_l = np.array([r * math.cos(a), r * math.sin(a), 0.0], dtype=np.float64)

    r_bl = rodrigues_xyz(0.0, laser_pitch, DEFAULT_LASER_YAW_RAD)
    p_b = np.array([DEFAULT_LASER_X_M, DEFAULT_LASER_Y_M, DEFAULT_LASER_Z_M], dtype=np.float64) + r_bl @ p_l

    p_cam_project = p_b - np.array([camera_x, camera_y, camera_z], dtype=np.float64)
    r_cam = rodrigues_xyz(camera_roll, camera_pitch, camera_yaw)
    p_cam_project = r_cam.T @ p_cam_project

    # project/base camera: +X forward, +Y left, +Z up
    # OpenCV optical: +X right, +Y down, +Z forward
    p_opt = np.array([-p_cam_project[1], -p_cam_project[2], p_cam_project[0]], dtype=np.float64)
    if p_opt[2] <= 1e-6:
        return np.array([np.nan, np.nan], dtype=np.float64), p_opt[2]
    projected, _ = cv2.projectPoints(
        p_opt.reshape(1, 3),
        np.zeros((3, 1), dtype=np.float64),
        np.zeros((3, 1), dtype=np.float64),
        mtx,
        dist,
    )
    return projected.reshape(2), float(p_opt[2])


def residuals(
    samples: list[dict[str, Any]],
    x: np.ndarray,
    mtx: np.ndarray,
    dist: np.ndarray,
    prior_weight: float,
) -> np.ndarray:
    res: list[float] = []
    for sample in samples:
        uv_obs = np.asarray(sample["image_point"], dtype=np.float64)
        uv_pred, z = project_sample(sample, x, mtx, dist)
        if not np.isfinite(uv_pred).all() or z <= 0:
            res.extend([2000.0, 2000.0])
            continue
        diff = uv_pred - uv_obs
        res.extend(np.clip(diff, -2000.0, 2000.0).tolist())
    if prior_weight > 0:
        x0 = initial_params()
        # Prior residuals are expressed in "pixel-equivalent" units. The
        # denominators encode how far we are comfortable moving from tape-measure
        # estimates before the optimizer should pay a meaningful penalty.
        scales = [
            math.radians(0.75),
            0.04,
            0.03,
            0.04,
            math.radians(5.0),
            math.radians(5.0),
            math.radians(5.0),
        ]
        prior = prior_weight * (x - x0) / np.asarray(scales, dtype=np.float64)
        res.extend(prior.tolist())
    return np.asarray(res, dtype=np.float64)


def huber_cost(res: np.ndarray, delta: float) -> float:
    abs_r = np.abs(res)
    quad = np.minimum(abs_r, delta)
    lin = abs_r - quad
    return float(np.sum(0.5 * quad**2 + delta * lin))


def numerical_jacobian(fun, x: np.ndarray, steps: np.ndarray) -> np.ndarray:
    y0 = fun(x)
    jac = np.zeros((len(y0), len(x)), dtype=np.float64)
    for i, step in enumerate(steps):
        xp = x.copy()
        xm = x.copy()
        xp[i] += step
        xm[i] -= step
        jac[:, i] = (fun(xp) - fun(xm)) / (2.0 * step)
    return jac


def optimize(
    samples: list[dict[str, Any]],
    mtx: np.ndarray,
    dist: np.ndarray,
    prior_weight: float,
    iterations: int,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    x = initial_params()
    steps = np.array([1e-5, 1e-4, 1e-4, 1e-4, 1e-5, 1e-5, 1e-5], dtype=np.float64)
    damping = 1e-2
    history: list[dict[str, float]] = []

    def fun(values: np.ndarray) -> np.ndarray:
        return residuals(samples, values, mtx, dist, prior_weight)

    for it in range(iterations):
        r = fun(x)
        cost = huber_cost(r, 12.0)
        jac = numerical_jacobian(fun, x, steps)
        weights = 1.0 / np.maximum(1.0, np.abs(r) / 12.0)
        jw = jac * weights[:, None]
        rw = r * weights
        h = jw.T @ jw + damping * np.eye(len(x))
        g = jw.T @ rw
        try:
            dx = -np.linalg.solve(h, g)
        except np.linalg.LinAlgError:
            dx = -np.linalg.pinv(h) @ g

        # Keep updates physically modest per iteration.
        dx[0] = float(np.clip(dx[0], -math.radians(0.25), math.radians(0.25)))
        dx[1:4] = np.clip(dx[1:4], -0.02, 0.02)
        dx[4:7] = np.clip(dx[4:7], -math.radians(2.0), math.radians(2.0))
        candidate = np.clip(x + dx, PARAM_LOWER, PARAM_UPPER)
        cand_cost = huber_cost(fun(candidate), 12.0)
        if cand_cost < cost:
            x = candidate
            damping = max(1e-6, damping * 0.5)
        else:
            damping = min(1e6, damping * 5.0)
        image_res = residuals(samples, x, mtx, dist, 0.0)
        err = image_res.reshape(-1, 2)
        px = np.linalg.norm(err, axis=1)
        history.append(
            {
                "iteration": float(it),
                "cost": float(huber_cost(fun(x), 12.0)),
                "median_px": float(np.median(px)),
                "mean_px": float(np.mean(px)),
                "p90_px": float(np.percentile(px, 90)),
                "damping": float(damping),
            }
        )
        if np.linalg.norm(dx) < 1e-7:
            break
    return x, history


def build_report(
    samples: list[dict[str, Any]],
    x: np.ndarray,
    mtx: np.ndarray,
    dist: np.ndarray,
) -> dict[str, Any]:
    rows = []
    for sample in samples:
        uv_pred, z = project_sample(sample, x, mtx, dist)
        uv_obs = np.asarray(sample["image_point"], dtype=np.float64)
        err = uv_pred - uv_obs
        rows.append(
            {
                "image_point": uv_obs.astype(float).tolist(),
                "projected_point": uv_pred.astype(float).tolist(),
                "error_px": err.astype(float).tolist(),
                "error_norm_px": float(np.linalg.norm(err)),
                "camera_z_m": float(z),
                "scan_point": sample["scan_point"],
                "note": sample.get("note"),
            }
        )
    errors = np.array([row["error_norm_px"] for row in rows], dtype=np.float64)
    return {
        "num_samples": len(samples),
        "median_error_px": float(np.median(errors)),
        "mean_error_px": float(np.mean(errors)),
        "p90_error_px": float(np.percentile(errors, 90)),
        "max_error_px": float(np.max(errors)),
        "samples": rows,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Calibrate LiDAR-camera extrinsics from manual correspondences.")
    parser.add_argument("--samples-jsonl", required=True, type=Path)
    parser.add_argument("--calib-file", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--iterations", default=60, type=int)
    parser.add_argument("--optimize-camera", action="store_true")
    parser.add_argument(
        "--optimize-camera-rpy",
        action="store_true",
        help="Deprecated alias for --optimize-camera.",
    )
    parser.add_argument("--prior-weight", default=8.0, type=float)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    samples = load_samples(args.samples_jsonl.resolve())
    mtx, dist = load_calib(args.calib_file.resolve())

    if args.optimize_camera or args.optimize_camera_rpy:
        print("note: --optimize-camera is now the default and is kept only for compatibility.")
    x0 = initial_params()
    initial_report = build_report(samples, x0, mtx, dist)
    x_opt, history = optimize(samples, mtx, dist, args.prior_weight, args.iterations)
    final_report = build_report(samples, x_opt, mtx, dist)

    result = {
        "input": {
            "samples_jsonl": str(args.samples_jsonl.resolve()),
            "calib_file": str(args.calib_file.resolve()),
            "model": "fixed_lidar_xyz_yaw__optimize_lidar_pitch_and_camera_xyz_rpy",
            "prior_weight": args.prior_weight,
        },
        "initial_params": params_to_dict(x0),
        "optimized_params": params_to_dict(x_opt),
        "initial_report": initial_report,
        "final_report": final_report,
        "history": history,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"initial median error: {initial_report['median_error_px']:.2f}px")
    print(f"final median error: {final_report['median_error_px']:.2f}px")
    print(f"final p90 error: {final_report['p90_error_px']:.2f}px")
    print(json.dumps(result["optimized_params"], indent=2))
    print(f"output: {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
