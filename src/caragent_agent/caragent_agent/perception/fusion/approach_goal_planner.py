"""Cost-aware approach-goal planning for object-level navigation tests."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw

RESAMPLING_NEAREST = getattr(getattr(Image, "Resampling", Image), "NEAREST")


def normalize_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


@dataclass(frozen=True)
class GridInfo:
    width: int
    height: int
    resolution: float
    origin_x: float
    origin_y: float
    data: np.ndarray
    source: str = ""


@dataclass(frozen=True)
class RobotPose2D:
    x: float
    y: float
    yaw: float
    source: str = ""


@dataclass(frozen=True)
class ApproachPlannerParams:
    stop_distance_m: float = 0.8
    min_stop_distance_m: float = 0.65
    max_stop_distance_m: float = 1.20
    robot_radius_m: float = 0.28
    safety_margin_m: float = 0.12
    unknown_is_obstacle: bool = True
    max_candidates: int = 480
    lethal_cost_threshold: int = 50


def grid_from_mapping(value: Any) -> GridInfo | None:
    if not isinstance(value, dict):
        return None
    try:
        width = int(value["width"])
        height = int(value["height"])
        resolution = float(value["resolution"])
        origin = value.get("origin") or value.get("origin_xy") or [value.get("origin_x"), value.get("origin_y")]
        origin_x = float(origin[0])
        origin_y = float(origin[1])
        data = np.asarray(value["data"], dtype=np.int16).reshape(height, width)
    except Exception:
        return None
    return GridInfo(
        width=width,
        height=height,
        resolution=resolution,
        origin_x=origin_x,
        origin_y=origin_y,
        data=data,
        source=str(value.get("source") or value.get("topic") or ""),
    )


def pose_from_state(current_state: dict[str, Any]) -> RobotPose2D | None:
    position = current_state.get("position") if isinstance(current_state, dict) else None
    orientation = current_state.get("orientation") if isinstance(current_state, dict) else None
    if not isinstance(position, (list, tuple)) or len(position) < 2:
        return None
    yaw = None
    if isinstance(current_state.get("yaw_rad"), (int, float)):
        yaw = float(current_state["yaw_rad"])
    elif isinstance(orientation, (list, tuple)) and len(orientation) >= 4:
        qx, qy, qz, qw = [float(v) for v in orientation[:4]]
        yaw = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
    if yaw is None or not math.isfinite(yaw):
        return None
    return RobotPose2D(float(position[0]), float(position[1]), yaw, str(current_state.get("source") or ""))


def base_xy_to_map(base_xy: np.ndarray, pose: RobotPose2D) -> np.ndarray:
    c = math.cos(pose.yaw)
    s = math.sin(pose.yaw)
    x = pose.x + c * float(base_xy[0]) - s * float(base_xy[1])
    y = pose.y + s * float(base_xy[0]) + c * float(base_xy[1])
    return np.asarray([x, y], dtype=np.float64)


def map_xy_to_cell(grid: GridInfo, xy: np.ndarray) -> tuple[int, int]:
    mx = int(math.floor((float(xy[0]) - grid.origin_x) / grid.resolution))
    my = int(math.floor((float(xy[1]) - grid.origin_y) / grid.resolution))
    return mx, my


def cell_to_map_xy(grid: GridInfo, mx: int, my: int) -> np.ndarray:
    return np.asarray(
        [
            grid.origin_x + (float(mx) + 0.5) * grid.resolution,
            grid.origin_y + (float(my) + 0.5) * grid.resolution,
        ],
        dtype=np.float64,
    )


def _clearance_map_m(grid: GridInfo, unknown_is_obstacle: bool) -> np.ndarray:
    occupied = grid.data >= 50
    if unknown_is_obstacle:
        occupied |= grid.data < 0
    free = (~occupied).astype(np.uint8)
    clearance_px = cv2.distanceTransform(free, cv2.DIST_L2, 5)
    return clearance_px.astype(np.float32) * float(grid.resolution)


def _line_cost(grid: GridInfo, a_xy: np.ndarray, b_xy: np.ndarray) -> tuple[bool, float]:
    ax, ay = map_xy_to_cell(grid, a_xy)
    bx, by = map_xy_to_cell(grid, b_xy)
    steps = max(abs(bx - ax), abs(by - ay), 1)
    worst = 0.0
    for i in range(steps + 1):
        t = i / steps
        mx = int(round(ax + (bx - ax) * t))
        my = int(round(ay + (by - ay) * t))
        if mx < 0 or my < 0 or mx >= grid.width or my >= grid.height:
            return False, 1000.0
        value = int(grid.data[my, mx])
        if value < 0:
            worst = max(worst, 70.0)
        elif value >= 50:
            return False, float(value)
        else:
            worst = max(worst, float(value))
    return True, worst


def _path_cost(grid: GridInfo, a_xy: np.ndarray, b_xy: np.ndarray) -> tuple[bool, float]:
    ax, ay = map_xy_to_cell(grid, a_xy)
    bx, by = map_xy_to_cell(grid, b_xy)
    steps = max(abs(bx - ax), abs(by - ay), 1)
    worst = 0.0
    total = 0.0
    count = 0
    for i in range(steps + 1):
        t = i / steps
        mx = int(round(ax + (bx - ax) * t))
        my = int(round(ay + (by - ay) * t))
        if mx < 0 or my < 0 or mx >= grid.width or my >= grid.height:
            return False, 1000.0
        value = int(grid.data[my, mx])
        if value < 0:
            value = 70
        if value >= 50:
            return False, float(value)
        worst = max(worst, float(value))
        total += max(0.0, float(value))
        count += 1
    mean = total / max(1, count)
    return True, max(worst, mean)


def _direct_goal(robot_xy: np.ndarray, object_xy: np.ndarray, stop_distance_m: float) -> dict[str, Any]:
    delta = object_xy - robot_xy
    dist = float(np.linalg.norm(delta))
    if dist <= 1e-6:
        unit = np.asarray([1.0, 0.0], dtype=np.float64)
    else:
        unit = delta / dist
    goal_xy = object_xy - stop_distance_m * unit
    yaw = math.atan2(float(object_xy[1] - goal_xy[1]), float(object_xy[0] - goal_xy[0]))
    return {
        "status": "degraded",
        "mode": "direct_fallback",
        "position": [float(goal_xy[0]), float(goal_xy[1]), 0.0],
        "yaw_rad": yaw,
        "yaw_deg": math.degrees(yaw),
        "score": None,
        "reason": "costmap_unavailable",
    }


def plan_approach_goal(
    *,
    robot_pose: RobotPose2D,
    object_map_xy: np.ndarray,
    grid: GridInfo | None,
    params: ApproachPlannerParams,
) -> dict[str, Any]:
    robot_xy = np.asarray([robot_pose.x, robot_pose.y], dtype=np.float64)
    if grid is None:
        fallback = _direct_goal(robot_xy, object_map_xy, params.stop_distance_m)
        fallback["candidate_count"] = 0
        return fallback

    clearance = _clearance_map_m(grid, params.unknown_is_obstacle)
    min_clearance = params.robot_radius_m + params.safety_margin_m
    radii = np.linspace(params.min_stop_distance_m, params.max_stop_distance_m, 6)
    angles = np.linspace(-math.pi, math.pi, 80, endpoint=False)
    candidates: list[dict[str, Any]] = []

    for radius in radii:
        for theta in angles:
            face_unit = np.asarray([math.cos(theta), math.sin(theta)], dtype=np.float64)
            candidate_xy = object_map_xy - float(radius) * face_unit
            mx, my = map_xy_to_cell(grid, candidate_xy)
            if mx < 0 or my < 0 or mx >= grid.width or my >= grid.height:
                continue
            occ = int(grid.data[my, mx])
            if occ < 0 and params.unknown_is_obstacle:
                continue
            if occ >= params.lethal_cost_threshold:
                continue
            clear_m = float(clearance[my, mx])
            if clear_m < min_clearance:
                continue
            line_ok, line_occ = _line_cost(grid, candidate_xy, object_map_xy)
            path_ok, path_occ = _path_cost(grid, robot_xy, candidate_xy)
            travel = float(np.linalg.norm(candidate_xy - robot_xy))
            yaw = math.atan2(float(object_map_xy[1] - candidate_xy[1]), float(object_map_xy[0] - candidate_xy[0]))
            turn = abs(normalize_angle(yaw - robot_pose.yaw))
            radius_penalty = abs(float(radius) - params.stop_distance_m)
            clearance_reward = min(clear_m, 1.2)
            approach_from_robot = math.atan2(
                float(candidate_xy[1] - robot_xy[1]),
                float(candidate_xy[0] - robot_xy[0]),
            )
            visibility_penalty = abs(normalize_angle(approach_from_robot - robot_pose.yaw))
            score = (
                8.0 * max(0.0, min_clearance - clear_m)
                + 0.65 * travel
                + 0.30 * turn
                + 1.80 * radius_penalty
                + 0.20 * visibility_penalty
                + 0.10 * max(0.0, float(occ))
                + 0.18 * max(0.0, line_occ)
                + 0.16 * max(0.0, path_occ)
                - 0.20 * clearance_reward
            )
            if not path_ok:
                score += 12.0
            candidates.append(
                {
                    "position": [float(candidate_xy[0]), float(candidate_xy[1]), 0.0],
                    "yaw_rad": yaw,
                    "yaw_deg": math.degrees(yaw),
                    "score": float(score),
                    "distance_to_object_m": float(radius),
                    "travel_distance_m": travel,
                    "clearance_m": clear_m,
                    "occupancy": occ,
                    "line_of_sight_ok": bool(line_ok),
                    "line_occupancy_max": float(line_occ),
                    "path_to_candidate_ok": bool(path_ok),
                    "path_occupancy_max": float(path_occ),
                    "visibility_penalty_rad": float(visibility_penalty),
                }
            )

    candidates.sort(key=lambda item: float(item["score"]))
    if not candidates:
        return {
            "status": "unreliable",
            "mode": "costmap_ring_search",
            "reason": "no_safe_costmap_candidate",
            "candidate_count": 0,
            "grid_source": grid.source,
            "safety": {
                "robot_radius_m": params.robot_radius_m,
                "safety_margin_m": params.safety_margin_m,
                "min_clearance_m": min_clearance,
            },
        }
    best = dict(candidates[0])
    best.update(
        {
            "status": "ok",
            "mode": "costmap_ring_search",
            "candidate_count": len(candidates),
            "top_candidates": candidates[: min(20, len(candidates))],
            "grid_source": grid.source,
            "safety": {
                "robot_radius_m": params.robot_radius_m,
                "safety_margin_m": params.safety_margin_m,
                "min_clearance_m": min_clearance,
            },
        }
    )
    return best


def draw_approach_debug(
    *,
    output_path: Path,
    robot_pose: RobotPose2D,
    object_map_xy: np.ndarray,
    goal: dict[str, Any],
    grid: GridInfo | None,
    params: ApproachPlannerParams,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if grid is not None:
        data = grid.data.astype(np.int16)
        img = np.full((grid.height, grid.width), 180, dtype=np.uint8)
        img[data < 0] = 110
        img[(data >= 0) & (data < 50)] = 245
        img[data >= 50] = 30
        rgb = np.stack([img, img, img], axis=-1)
        canvas = Image.fromarray(np.flipud(rgb), "RGB").resize((900, 900), RESAMPLING_NEAREST)

        def to_px(xy: np.ndarray) -> tuple[float, float]:
            mx, my = map_xy_to_cell(grid, xy)
            sx = 900.0 / max(1, grid.width)
            sy = 900.0 / max(1, grid.height)
            return (mx + 0.5) * sx, 900.0 - (my + 0.5) * sy

        scale_px_per_m = 900.0 / max(grid.width * grid.resolution, grid.height * grid.resolution)
    else:
        canvas = Image.new("RGB", (900, 900), (245, 245, 240))
        center = np.asarray([450.0, 450.0], dtype=np.float64)
        scale_px_per_m = 120.0

        def to_px(xy: np.ndarray) -> tuple[float, float]:
            rel = xy - np.asarray([robot_pose.x, robot_pose.y], dtype=np.float64)
            return float(center[0] + rel[0] * scale_px_per_m), float(center[1] - rel[1] * scale_px_per_m)

    draw = ImageDraw.Draw(canvas)
    robot_xy = np.asarray([robot_pose.x, robot_pose.y], dtype=np.float64)
    goal_xy = np.asarray(goal.get("position", [robot_pose.x, robot_pose.y])[:2], dtype=np.float64)
    rx, ry = to_px(robot_xy)
    ox, oy = to_px(object_map_xy)
    gx, gy = to_px(goal_xy)
    ring_r = params.stop_distance_m * scale_px_per_m
    draw.ellipse([ox - ring_r, oy - ring_r, ox + ring_r, oy + ring_r], outline=(80, 140, 240), width=2)
    for cand in goal.get("top_candidates", [])[:15]:
        cxy = np.asarray(cand.get("position", [0, 0])[:2], dtype=np.float64)
        cx, cy = to_px(cxy)
        draw.ellipse([cx - 3, cy - 3, cx + 3, cy + 3], fill=(255, 190, 30))
    draw.line([rx, ry, gx, gy], fill=(80, 180, 80), width=3)
    draw.line([gx, gy, ox, oy], fill=(40, 120, 240), width=3)
    draw.ellipse([rx - 7, ry - 7, rx + 7, ry + 7], fill=(40, 180, 80), outline=(0, 0, 0))
    draw.text((rx + 9, ry - 8), "robot", fill=(0, 0, 0))
    draw.ellipse([ox - 8, oy - 8, ox + 8, oy + 8], fill=(230, 60, 60), outline=(0, 0, 0))
    draw.text((ox + 10, oy - 8), "object", fill=(0, 0, 0))
    draw.rectangle([gx - 7, gy - 7, gx + 7, gy + 7], fill=(40, 100, 240), outline=(0, 0, 0))
    draw.text((gx + 9, gy - 8), "goal", fill=(0, 0, 0))
    text = f"{goal.get('mode')} score={goal.get('score')} yaw={goal.get('yaw_deg', 0):.1f}"
    draw.rectangle([8, 8, 8 + min(860, len(text) * 8), 32], fill=(255, 255, 255))
    draw.text((14, 13), text, fill=(0, 0, 0))
    canvas.save(output_path)
