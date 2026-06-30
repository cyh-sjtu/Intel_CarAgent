"""Geometry helpers for keyframe recording and selection."""

from __future__ import annotations

import math
from typing import Iterable


def normalize_angle_rad(angle: float) -> float:
    """Normalize an angle to [-pi, pi)."""

    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def yaw_difference_rad(a: float, b: float) -> float:
    """Return the absolute shortest yaw difference in radians."""

    return abs(normalize_angle_rad(float(a) - float(b)))


def yaw_difference_deg(a: float, b: float) -> float:
    """Return the absolute shortest yaw difference in degrees."""

    return math.degrees(yaw_difference_rad(a, b))


def quaternion_xyzw_to_yaw(q: Iterable[float]) -> float:
    """Return yaw in radians from a quaternion in [x, y, z, w] order."""

    x, y, z, w = [float(value) for value in q]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def planar_distance(a: dict, b: dict) -> float:
    """Return 2D distance between pose-like dicts with x/y fields."""

    return math.hypot(float(a["x"]) - float(b["x"]), float(a["y"]) - float(b["y"]))


def pose_xy_yaw(x: float, y: float, yaw: float) -> dict:
    """Build a compact pose dictionary."""

    return {"x": float(x), "y": float(y), "yaw": float(yaw)}
