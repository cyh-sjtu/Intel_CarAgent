"""LaserScan summary utilities for keyframe metadata."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def summarize_scan_arrays(
    *,
    ranges: Iterable[float],
    angle_min: float,
    angle_increment: float,
    range_min: float,
    range_max: float,
    angle_max: float | None = None,
) -> dict:
    """Summarize front/left/right/rear free distances from scan arrays."""

    del angle_max

    ranges_arr = np.asarray(list(ranges), dtype=np.float32)
    if ranges_arr.size == 0:
        return {
            "available": False,
            "front_min_m": None,
            "left_min_m": None,
            "right_min_m": None,
            "rear_min_m": None,
            "valid_count": 0,
        }

    indices = np.arange(ranges_arr.size, dtype=np.float32)
    angles = float(angle_min) + indices * float(angle_increment)
    valid = np.isfinite(ranges_arr) & (ranges_arr >= float(range_min)) & (ranges_arr <= float(range_max))

    def sector_min(center_rad: float, half_width_rad: float) -> float | None:
        delta = np.arctan2(np.sin(angles - center_rad), np.cos(angles - center_rad))
        mask = valid & (np.abs(delta) <= half_width_rad)
        if not np.any(mask):
            return None
        return float(np.min(ranges_arr[mask]))

    return {
        "available": True,
        "front_min_m": sector_min(0.0, math.radians(25.0)),
        "left_min_m": sector_min(math.pi / 2.0, math.radians(25.0)),
        "right_min_m": sector_min(-math.pi / 2.0, math.radians(25.0)),
        "rear_min_m": sector_min(math.pi, math.radians(25.0)),
        "valid_count": int(np.count_nonzero(valid)),
        "range_min_m": float(range_min),
        "range_max_m": float(range_max),
    }


def scan_msg_to_arrays(scan_msg) -> dict:
    """Convert a ROS LaserScan message into serializable numpy arrays."""

    return {
        "ranges": np.asarray(scan_msg.ranges, dtype=np.float32),
        "angle_min": float(scan_msg.angle_min),
        "angle_max": float(scan_msg.angle_max),
        "angle_increment": float(scan_msg.angle_increment),
        "range_min": float(scan_msg.range_min),
        "range_max": float(scan_msg.range_max),
    }
