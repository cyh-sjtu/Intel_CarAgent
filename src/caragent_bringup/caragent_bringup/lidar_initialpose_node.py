#!/usr/bin/env python3
"""Startup lidar-assisted initial pose estimation for slam_toolbox.

This node does not replace slam_toolbox localization. It waits for a usable
laser scan and an occupancy grid, samples pose hypotheses, scores each
hypothesis against the map using both laser endpoints and free-space ray
consistency, then publishes the best pose to ``/initialpose`` so slam_toolbox
can continue scan matching from that estimate.

Compared with a simple endpoint-only matcher, this version also:
- uses ``math.floor`` for world->grid conversion, so negative map coordinates
  near the map boundary are handled correctly;
- applies a ray-tracing penalty when a laser beam would pass through an occupied
  cell before its measured endpoint;
- supports laser-to-base extrinsics;
- waits briefly for the map to become stable before localizing;
- runs a coarse search followed by local refinement;
- checks the gap between the best and second-best candidates and inflates
  covariance, or rejects the pose if configured, when the result is ambiguous;
- supports multiprocessing-based parallel scoring for faster particle search.
"""

from __future__ import annotations

import math
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Optional

import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped, Quaternion
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import LaserScan


OCCUPIED_THRESHOLD = 50
FREE_THRESHOLD = 25


@dataclass(frozen=True)
class GridInfo:
    data: list[int]
    width: int
    height: int
    resolution: float
    origin_x: float
    origin_y: float


@dataclass(frozen=True)
class ScanBeam:
    radius: float
    angle: float
    is_max_range: bool


@dataclass(frozen=True)
class ScoreConfig:
    hit_radius_cells: int
    min_beams: int
    laser_x: float
    laser_y: float
    laser_yaw: float
    raytrace_step_m: float
    hit_reward: float
    miss_penalty: float
    wall_penalty: float
    unknown_penalty: float
    max_range_clear_reward: float


@dataclass(frozen=True)
class PoseScore:
    x: float
    y: float
    yaw: float
    score: float
    valid_beams: int
    hits: int
    blocked: int
    misses: int
    unknown_ratio_sum: float


def _normalize_angle(angle_rad: float) -> float:
    """Normalize an angle to [-pi, pi)."""
    return (angle_rad + math.pi) % (2.0 * math.pi) - math.pi


def _yaw_to_quat(yaw_rad: float) -> Quaternion:
    q = Quaternion()
    q.x = 0.0
    q.y = 0.0
    q.z = math.sin(yaw_rad / 2.0)
    q.w = math.cos(yaw_rad / 2.0)
    return q


def _grid_index(gx: int, gy: int, grid: GridInfo) -> int:
    return gy * grid.width + gx


def _world_to_grid(x: float, y: float, grid: GridInfo) -> tuple[int, int]:
    # Important: int() truncates toward zero. floor() is correct for maps whose
    # origin is negative or when x/y are just outside the lower map boundary.
    return (
        math.floor((x - grid.origin_x) / grid.resolution),
        math.floor((y - grid.origin_y) / grid.resolution),
    )


def _grid_to_world_random(gx: int, gy: int, grid: GridInfo) -> tuple[float, float]:
    return (
        grid.origin_x + (gx + random.random()) * grid.resolution,
        grid.origin_y + (gy + random.random()) * grid.resolution,
    )


def _grid_to_world_center(gx: int, gy: int, grid: GridInfo) -> tuple[float, float]:
    return (
        grid.origin_x + (gx + 0.5) * grid.resolution,
        grid.origin_y + (gy + 0.5) * grid.resolution,
    )


def _is_in_grid(gx: int, gy: int, grid: GridInfo) -> bool:
    return 0 <= gx < grid.width and 0 <= gy < grid.height


def _cell_value(gx: int, gy: int, grid: GridInfo) -> Optional[int]:
    if not _is_in_grid(gx, gy, grid):
        return None
    return grid.data[_grid_index(gx, gy, grid)]


def _is_free_cell(gx: int, gy: int, grid: GridInfo) -> bool:
    value = _cell_value(gx, gy, grid)
    return value is not None and 0 <= value <= FREE_THRESHOLD


def _is_traversable_pose(
    x: float,
    y: float,
    grid: GridInfo,
    clearance_cells: int,
) -> bool:
    gx, gy = _world_to_grid(x, y, grid)
    if not _is_free_cell(gx, gy, grid):
        return False

    radius_sq = clearance_cells * clearance_cells
    for dy in range(-clearance_cells, clearance_cells + 1):
        for dx in range(-clearance_cells, clearance_cells + 1):
            if dx * dx + dy * dy > radius_sq:
                continue
            cx = gx + dx
            cy = gy + dy
            value = _cell_value(cx, cy, grid)
            if value is None:
                return False
            if value >= OCCUPIED_THRESHOLD:
                return False

    return True


def _has_occupied_near(
    gx: int,
    gy: int,
    grid: GridInfo,
    radius_cells: int,
) -> bool:
    radius_sq = radius_cells * radius_cells
    for dy in range(-radius_cells, radius_cells + 1):
        for dx in range(-radius_cells, radius_cells + 1):
            if dx * dx + dy * dy > radius_sq:
                continue
            value = _cell_value(gx + dx, gy + dy, grid)
            if value is not None and value >= OCCUPIED_THRESHOLD:
                return True
    return False


def _valid_scan_ranges(
    scan: LaserScan,
    max_beams: int,
    score_max_range_m: float,
    max_range_margin_m: float,
) -> list[ScanBeam]:
    """Downsample valid laser beams.

    ``score_max_range_m`` clips very long beams. Clipped beams are treated like
    max-range beams: they can penalize early wall collisions, but they are not
    expected to end on an occupied cell.
    """
    beams: list[ScanBeam] = []
    if not scan.ranges:
        return beams

    beam_count = len(scan.ranges)
    stride = max(1, math.ceil(beam_count / max_beams)) if max_beams > 0 else 1

    effective_max_range = scan.range_max
    if score_max_range_m > 0.0:
        effective_max_range = min(effective_max_range, score_max_range_m)

    for index in range(0, beam_count, stride):
        raw_radius = scan.ranges[index]
        if not math.isfinite(raw_radius):
            continue
        if raw_radius < scan.range_min or raw_radius > scan.range_max:
            continue

        clipped = score_max_range_m > 0.0 and raw_radius > score_max_range_m
        is_max_range = raw_radius >= (scan.range_max - max_range_margin_m) or clipped
        radius = min(raw_radius, effective_max_range)

        if radius <= 0.0:
            continue

        angle = scan.angle_min + index * scan.angle_increment
        beams.append(
            ScanBeam(
                radius=radius,
                angle=angle,
                is_max_range=is_max_range,
            )
        )

    return beams


def _trace_ray_for_early_obstacle(
    start_x: float,
    start_y: float,
    end_x: float,
    end_y: float,
    grid: GridInfo,
    hit_radius_cells: int,
    raytrace_step_m: float,
) -> tuple[bool, int, int]:
    """Return (blocked_before_endpoint, unknown_cells, sampled_cells).

    The last cells around the measured endpoint are ignored because those cells
    are allowed to contain the obstacle that generated the laser return.
    """
    dx = end_x - start_x
    dy = end_y - start_y
    distance = math.hypot(dx, dy)
    if distance <= 1e-6:
        return False, 0, 0

    step_m = max(raytrace_step_m, grid.resolution)
    steps = max(1, math.ceil(distance / step_m))
    endpoint_ignore_dist = max(
        grid.resolution,
        (hit_radius_cells + 1) * grid.resolution,
    )

    unknown_cells = 0
    sampled_cells = 0
    last_cell: Optional[tuple[int, int]] = None

    for i in range(1, steps):
        t = i / steps
        if distance * t >= distance - endpoint_ignore_dist:
            break

        x = start_x + dx * t
        y = start_y + dy * t
        gx, gy = _world_to_grid(x, y, grid)

        if last_cell == (gx, gy):
            continue

        last_cell = (gx, gy)
        value = _cell_value(gx, gy, grid)

        if value is None:
            continue

        sampled_cells += 1

        if value >= OCCUPIED_THRESHOLD:
            return True, unknown_cells, sampled_cells

        if value < 0:
            unknown_cells += 1

    return False, unknown_cells, sampled_cells


def _score_pose(
    px: float,
    py: float,
    pyaw: float,
    scan_beams: list[ScanBeam],
    grid: GridInfo,
    cfg: ScoreConfig,
) -> PoseScore:
    """Score a base_link pose hypothesis against the occupancy grid."""
    base_cos = math.cos(pyaw)
    base_sin = math.sin(pyaw)

    laser_x_world = px + base_cos * cfg.laser_x - base_sin * cfg.laser_y
    laser_y_world = py + base_sin * cfg.laser_x + base_cos * cfg.laser_y
    laser_yaw_world = _normalize_angle(pyaw + cfg.laser_yaw)

    laser_gx, laser_gy = _world_to_grid(laser_x_world, laser_y_world, grid)
    if not _is_in_grid(laser_gx, laser_gy, grid):
        return PoseScore(px, py, pyaw, -999.0, 0, 0, 0, 0, 0.0)

    score_sum = 0.0
    valid = 0
    hits = 0
    blocked = 0
    misses = 0
    unknown_ratio_sum = 0.0

    for beam in scan_beams:
        angle = laser_yaw_world + beam.angle
        end_x = laser_x_world + beam.radius * math.cos(angle)
        end_y = laser_y_world + beam.radius * math.sin(angle)
        end_gx, end_gy = _world_to_grid(end_x, end_y, grid)

        if not _is_in_grid(end_gx, end_gy, grid):
            continue

        valid += 1

        is_blocked, unknown_cells, sampled_cells = _trace_ray_for_early_obstacle(
            laser_x_world,
            laser_y_world,
            end_x,
            end_y,
            grid,
            cfg.hit_radius_cells,
            cfg.raytrace_step_m,
        )

        if sampled_cells > 0 and unknown_cells > 0:
            unknown_ratio = unknown_cells / sampled_cells
            unknown_ratio_sum += unknown_ratio
            score_sum -= cfg.unknown_penalty * unknown_ratio

        if is_blocked:
            blocked += 1
            score_sum -= cfg.wall_penalty
            continue

        endpoint_hit = _has_occupied_near(
            end_gx,
            end_gy,
            grid,
            cfg.hit_radius_cells,
        )

        if not beam.is_max_range and endpoint_hit:
            hits += 1
            score_sum += cfg.hit_reward
        elif beam.is_max_range:
            # Long/max-range rays are useful because they say the ray should
            # not hit a wall early, but their endpoint should not be forced to
            # land on an occupied cell.
            score_sum += cfg.max_range_clear_reward
        else:
            misses += 1
            score_sum -= cfg.miss_penalty

    if valid < cfg.min_beams:
        return PoseScore(
            px,
            py,
            pyaw,
            -999.0,
            valid,
            hits,
            blocked,
            misses,
            unknown_ratio_sum,
        )

    return PoseScore(
        x=px,
        y=py,
        yaw=_normalize_angle(pyaw),
        score=score_sum / valid,
        valid_beams=valid,
        hits=hits,
        blocked=blocked,
        misses=misses,
        unknown_ratio_sum=unknown_ratio_sum,
    )


# ---------------------------------------------------------------------------
# Multiprocessing scoring worker state.
#
# These globals live inside each worker process. The map, beams and scoring
# config are initialized once per worker so every candidate chunk does not need
# to resend large immutable data repeatedly.
# ---------------------------------------------------------------------------

_SCORE_WORKER_GRID: Optional[GridInfo] = None
_SCORE_WORKER_SCAN_BEAMS: Optional[list[ScanBeam]] = None
_SCORE_WORKER_CFG: Optional[ScoreConfig] = None


def _init_score_worker(
    grid: GridInfo,
    scan_beams: list[ScanBeam],
    cfg: ScoreConfig,
) -> None:
    """Initialize immutable scoring data once per worker process."""
    global _SCORE_WORKER_GRID
    global _SCORE_WORKER_SCAN_BEAMS
    global _SCORE_WORKER_CFG

    _SCORE_WORKER_GRID = grid
    _SCORE_WORKER_SCAN_BEAMS = scan_beams
    _SCORE_WORKER_CFG = cfg


def _score_candidate_chunk_worker(
    args: tuple[list[tuple[float, float, float]], int],
) -> list[PoseScore]:
    """Score one candidate chunk inside a worker process."""
    chunk, top_k = args

    grid = _SCORE_WORKER_GRID
    scan_beams = _SCORE_WORKER_SCAN_BEAMS
    cfg = _SCORE_WORKER_CFG

    if grid is None or scan_beams is None or cfg is None:
        raise RuntimeError("Score worker was not initialized.")

    scores: list[PoseScore] = []

    for px, py, pyaw in chunk:
        score = _score_pose(px, py, pyaw, scan_beams, grid, cfg)
        if score.valid_beams >= cfg.min_beams:
            scores.append(score)

    scores.sort(key=lambda item: item.score, reverse=True)
    return scores[:top_k]


def _candidate_chunks(
    candidates: list[tuple[float, float, float]],
    chunk_size: int,
) -> list[list[tuple[float, float, float]]]:
    chunk_size = max(1, chunk_size)
    return [
        candidates[index:index + chunk_size]
        for index in range(0, len(candidates), chunk_size)
    ]


class LidarInitialposeNode(Node):
    """Estimate and publish a startup initial pose from lidar and map data."""

    def __init__(self) -> None:
        super().__init__("lidar_initialpose_node")

        # Topics and frames.
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("map_topic", "/map")
        self.declare_parameter("initialpose_topic", "/initialpose")
        self.declare_parameter("initial_pose_frame", "map")

        # Startup timing.
        self.declare_parameter("wait_scan_timeout", 30.0)
        self.declare_parameter("wait_map_timeout", 60.0)
        self.declare_parameter("map_stable_delay", 1.5)
        self.declare_parameter("map_stable_required_updates", 1)

        # Hint/fallback pose in the map frame.
        self.declare_parameter("initial_x", 0.0)
        self.declare_parameter("initial_y", 0.0)
        self.declare_parameter("initial_yaw_deg", 0.0)

        # Laser pose relative to base_link. Set these if /scan is not exactly
        # located at base_link with the same yaw.
        self.declare_parameter("laser_x", 0.0)
        self.declare_parameter("laser_y", 0.0)
        self.declare_parameter("laser_yaw_deg", 0.0)

        # Particle localization.
        self.declare_parameter("use_particle_filter", True)
        self.declare_parameter("publish_fallback_on_failure", True)
        self.declare_parameter("use_global_localization", True)
        self.declare_parameter("random_seed", 0)
        self.declare_parameter("num_particles", 2500)
        self.declare_parameter("yaw_bins", 36)
        self.declare_parameter("search_range_m", 3.0)
        self.declare_parameter("local_yaw_search_deg", 45.0)
        self.declare_parameter("max_sampling_attempts", 40000)

        # Parallel scoring.
        #
        # 0 or 1 disables multiprocessing. For this CPU-bound scoring workload,
        # processes are usually faster than threads because of Python's GIL.
        #
        # Keep at least 1 CPU core free for ROS, slam_toolbox and sensor nodes.
        self.declare_parameter(
            "scoring_workers",
            max(1, (os.cpu_count() or 2) - 1),
        )
        self.declare_parameter("scoring_chunk_size", 64)
        self.declare_parameter("parallel_min_candidates", 300)

        # Refinement after coarse scoring.
        self.declare_parameter("top_k", 12)
        self.declare_parameter("refinement_rounds", 2)
        self.declare_parameter("refinement_particles", 600)
        self.declare_parameter("refinement_top_k", 5)
        self.declare_parameter("refinement_xy_std_m", 0.35)
        self.declare_parameter("refinement_yaw_std_deg", 12.0)

        # Laser/map scoring.
        self.declare_parameter("hit_threshold_m", 0.08)
        self.declare_parameter("robot_clearance_m", 0.20)
        self.declare_parameter("min_beams", 80)
        self.declare_parameter("max_beams", 180)
        self.declare_parameter("score_max_range_m", 8.0)
        self.declare_parameter("max_range_margin_m", 0.10)
        self.declare_parameter("raytrace_step_m", 0.10)
        self.declare_parameter("hit_reward", 1.0)
        self.declare_parameter("miss_penalty", 0.35)
        self.declare_parameter("wall_penalty", 1.0)
        self.declare_parameter("unknown_penalty", 0.15)
        self.declare_parameter("max_range_clear_reward", 0.05)
        self.declare_parameter("min_score", 0.25)

        # Ambiguity / confidence checks.
        self.declare_parameter("check_score_margin", True)
        self.declare_parameter("reject_ambiguous_pose", False)
        self.declare_parameter("min_score_margin", 0.06)
        self.declare_parameter("min_score_ratio", 1.12)
        self.declare_parameter("ambiguous_cov_xy", 0.80)
        self.declare_parameter("ambiguous_cov_yaw", 0.60)

        # Covariance for a confident pose. Values are standard deviations.
        self.declare_parameter("pose_cov_xy", 0.20)
        self.declare_parameter("pose_cov_yaw", 0.15)

        # Repeat publishing can help if slam_toolbox activates slowly.
        self.declare_parameter("retry_interval", 2.0)
        self.declare_parameter("max_retries", 2)

        self._scan_msg: Optional[LaserScan] = None
        self._map_msg: Optional[OccupancyGrid] = None
        self._pose_published = False
        self._retry_count = 0
        self._start_time = time.monotonic()
        self._map_ready_logged = False
        self._map_signature: Optional[tuple[int, int, float, float, float]] = None
        self._map_stable_since: Optional[float] = None
        self._map_stable_updates = 0
        self._last_wait_log_second = -1
        self._free_cell_cache_signature: Optional[
            tuple[int, int, float, float, float, int]
        ] = None
        self._free_cell_cache: list[tuple[int, int]] = []
        self._last_pose_ambiguous = False

        random_seed = self.get_parameter("random_seed").get_parameter_value().integer_value
        if random_seed != 0:
            random.seed(random_seed)

        scan_topic = self.get_parameter("scan_topic").get_parameter_value().string_value
        map_topic = self.get_parameter("map_topic").get_parameter_value().string_value
        initialpose_topic = (
            self.get_parameter("initialpose_topic")
            .get_parameter_value()
            .string_value
        )

        self._scan_sub = self.create_subscription(
            LaserScan,
            scan_topic,
            self._scan_callback,
            10,
        )

        map_qos = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._map_sub = self.create_subscription(
            OccupancyGrid,
            map_topic,
            self._map_callback,
            map_qos,
        )

        self._initialpose_pub = self.create_publisher(
            PoseWithCovarianceStamped,
            initialpose_topic,
            10,
        )

        self._check_timer = self.create_timer(1.0, self._check_and_publish)

        self.get_logger().info(
            f"[InitialPose] Waiting for lidar({scan_topic}) and map({map_topic})..."
        )

    def _scan_callback(self, msg: LaserScan) -> None:
        valid = [
            r for r in msg.ranges
            if math.isfinite(r) and msg.range_min <= r <= msg.range_max
        ]

        if len(valid) < 10:
            if self._scan_msg is None:
                self.get_logger().warn(
                    f"[InitialPose] Scan has only {len(valid)} valid beams; waiting."
                )
            return

        first_scan = self._scan_msg is None
        self._scan_msg = msg

        if first_scan:
            elapsed = time.monotonic() - self._start_time
            self.get_logger().info(
                f"[InitialPose] Lidar ready: {len(valid)} valid beams in {elapsed:.1f}s"
            )

    def _map_callback(self, msg: OccupancyGrid) -> None:
        if msg.info.width == 0 or msg.info.height == 0 or msg.info.resolution <= 0.0:
            return

        signature = (
            msg.info.width,
            msg.info.height,
            round(msg.info.resolution, 6),
            round(msg.info.origin.position.x, 3),
            round(msg.info.origin.position.y, 3),
        )

        now = time.monotonic()

        if signature != self._map_signature:
            if self._map_signature is not None:
                self.get_logger().info(
                    "[InitialPose] Map geometry changed; resetting map stability timer."
                )

            self._map_signature = signature
            self._map_stable_since = now
            self._map_stable_updates = 1
            self._free_cell_cache_signature = None
            self._free_cell_cache = []
        else:
            self._map_stable_updates += 1

        self._map_msg = msg

        if not self._map_ready_logged:
            self._map_ready_logged = True
            elapsed = time.monotonic() - self._start_time
            self.get_logger().info(
                f"[InitialPose] Map ready: {msg.info.width}x{msg.info.height} "
                f"res={msg.info.resolution:.3f}m in {elapsed:.1f}s"
            )

    def _check_and_publish(self) -> None:
        if self._pose_published:
            self._check_timer.cancel()
            return

        elapsed = time.monotonic() - self._start_time

        wait_scan = (
            self.get_parameter("wait_scan_timeout")
            .get_parameter_value()
            .double_value
        )
        wait_map = (
            self.get_parameter("wait_map_timeout")
            .get_parameter_value()
            .double_value
        )

        if self._scan_msg is None and elapsed > wait_scan:
            self.get_logger().error(
                f"[InitialPose] Lidar timeout ({wait_scan:.0f}s). "
                "Check lidar connection."
            )
            self._check_timer.cancel()
            return

        if self._map_msg is None and elapsed > wait_map:
            self.get_logger().error(
                f"[InitialPose] Map timeout ({wait_map:.0f}s). Check slam_toolbox."
            )
            self._check_timer.cancel()
            return

        if self._scan_msg is None or self._map_msg is None:
            self._log_waiting(elapsed)
            return

        if not self._is_map_stable():
            self._log_waiting(elapsed, extra="map_stable=NO")
            return

        self._do_localize_and_publish()

    def _log_waiting(self, elapsed: float, extra: str = "") -> None:
        second = int(elapsed)

        if second == self._last_wait_log_second or second % 5 != 0:
            return

        self._last_wait_log_second = second
        suffix = f" {extra}" if extra else ""

        self.get_logger().info(
            f"[InitialPose] Waiting... scan={'OK' if self._scan_msg else 'NO'} "
            f"map={'OK' if self._map_msg else 'NO'} "
            f"elapsed={elapsed:.0f}s{suffix}"
        )

    def _is_map_stable(self) -> bool:
        if self._map_msg is None or self._map_stable_since is None:
            return False

        delay = (
            self.get_parameter("map_stable_delay")
            .get_parameter_value()
            .double_value
        )
        required_updates = max(
            1,
            self.get_parameter("map_stable_required_updates")
            .get_parameter_value()
            .integer_value,
        )

        stable_age = time.monotonic() - self._map_stable_since

        return stable_age >= delay and self._map_stable_updates >= required_updates

    def _do_localize_and_publish(self) -> None:
        use_pf = (
            self.get_parameter("use_particle_filter")
            .get_parameter_value()
            .bool_value
        )
        publish_fallback = (
            self.get_parameter("publish_fallback_on_failure")
            .get_parameter_value()
            .bool_value
        )
        max_retries = max(
            1,
            self.get_parameter("max_retries").get_parameter_value().integer_value,
        )
        retry_interval = (
            self.get_parameter("retry_interval")
            .get_parameter_value()
            .double_value
        )

        published_this_round = False

        if use_pf:
            result = self._run_particle_filter()

            if result is None:
                self.get_logger().warn("[InitialPose] Particle localization failed.")

                if publish_fallback:
                    self.get_logger().warn(
                        "[InitialPose] Publishing hint pose as fallback."
                    )
                    self._publish_hint_pose()
                    published_this_round = True
            else:
                self.get_logger().info(
                    f"[InitialPose] Particle best pose: "
                    f"x={result.x:.3f} y={result.y:.3f} "
                    f"yaw={math.degrees(result.yaw):.1f}deg "
                    f"score={result.score:.3f} "
                    f"hits={result.hits}/{result.valid_beams} "
                    f"blocked={result.blocked} misses={result.misses}"
                )

                cov_xy = (
                    self.get_parameter("pose_cov_xy")
                    .get_parameter_value()
                    .double_value
                )
                cov_yaw = (
                    self.get_parameter("pose_cov_yaw")
                    .get_parameter_value()
                    .double_value
                )

                if self._last_pose_ambiguous:
                    cov_xy = max(
                        cov_xy,
                        self.get_parameter("ambiguous_cov_xy")
                        .get_parameter_value()
                        .double_value,
                    )
                    cov_yaw = max(
                        cov_yaw,
                        self.get_parameter("ambiguous_cov_yaw")
                        .get_parameter_value()
                        .double_value,
                    )
                    self.get_logger().warn(
                        "[InitialPose] Best and second-best poses are close; "
                        "publishing with enlarged covariance."
                    )

                self._publish_pose(result.x, result.y, result.yaw, cov_xy, cov_yaw)
                published_this_round = True
        else:
            self.get_logger().info(
                "[InitialPose] Particle filter disabled; publishing hint pose."
            )
            self._publish_hint_pose()
            published_this_round = True

        if published_this_round:
            self._retry_count += 1

        if self._retry_count >= max_retries:
            self.get_logger().info(
                f"[InitialPose] Published {self._retry_count} initial pose(s). Done."
            )
            self._pose_published = True
            self._check_timer.cancel()
            return

        self._check_timer.cancel()
        self._check_timer = self.create_timer(
            retry_interval,
            self._do_localize_and_publish,
        )

    def _run_particle_filter(self) -> Optional[PoseScore]:
        """Sample startup pose particles and return the best valid hypothesis."""
        self._last_pose_ambiguous = False

        scan = self._scan_msg
        occ = self._map_msg

        if scan is None or occ is None:
            return None

        random_seed = self.get_parameter("random_seed").get_parameter_value().integer_value
        if random_seed != 0:
            random.seed(random_seed + self._retry_count)

        grid = GridInfo(
            data=list(occ.data),
            width=occ.info.width,
            height=occ.info.height,
            resolution=occ.info.resolution,
            origin_x=occ.info.origin.position.x,
            origin_y=occ.info.origin.position.y,
        )

        if not grid.data or grid.resolution <= 0.0:
            self.get_logger().warn("[ParticleFilter] Empty or invalid occupancy grid.")
            return None

        max_beams = (
            self.get_parameter("max_beams")
            .get_parameter_value()
            .integer_value
        )
        min_beams = (
            self.get_parameter("min_beams")
            .get_parameter_value()
            .integer_value
        )
        score_max_range_m = (
            self.get_parameter("score_max_range_m")
            .get_parameter_value()
            .double_value
        )
        max_range_margin_m = (
            self.get_parameter("max_range_margin_m")
            .get_parameter_value()
            .double_value
        )

        scan_beams = _valid_scan_ranges(
            scan,
            max_beams=max_beams,
            score_max_range_m=score_max_range_m,
            max_range_margin_m=max_range_margin_m,
        )

        if len(scan_beams) < min_beams:
            self.get_logger().warn(
                f"[ParticleFilter] Only {len(scan_beams)} valid beams; "
                f"need at least {min_beams}."
            )
            return None

        hint_x = (
            self.get_parameter("initial_x")
            .get_parameter_value()
            .double_value
        )
        hint_y = (
            self.get_parameter("initial_y")
            .get_parameter_value()
            .double_value
        )
        hint_yaw = math.radians(
            self.get_parameter("initial_yaw_deg")
            .get_parameter_value()
            .double_value
        )

        num_particles = (
            self.get_parameter("num_particles")
            .get_parameter_value()
            .integer_value
        )

        if num_particles < 1:
            self.get_logger().warn("[ParticleFilter] num_particles must be >= 1.")
            return None

        hit_threshold_m = (
            self.get_parameter("hit_threshold_m")
            .get_parameter_value()
            .double_value
        )
        robot_clearance_m = (
            self.get_parameter("robot_clearance_m")
            .get_parameter_value()
            .double_value
        )

        hit_radius_cells = max(0, math.ceil(hit_threshold_m / grid.resolution))
        clearance_cells = max(0, math.ceil(robot_clearance_m / grid.resolution))

        cfg = ScoreConfig(
            hit_radius_cells=hit_radius_cells,
            min_beams=min_beams,
            laser_x=(
                self.get_parameter("laser_x")
                .get_parameter_value()
                .double_value
            ),
            laser_y=(
                self.get_parameter("laser_y")
                .get_parameter_value()
                .double_value
            ),
            laser_yaw=math.radians(
                self.get_parameter("laser_yaw_deg")
                .get_parameter_value()
                .double_value
            ),
            raytrace_step_m=(
                self.get_parameter("raytrace_step_m")
                .get_parameter_value()
                .double_value
            ),
            hit_reward=(
                self.get_parameter("hit_reward")
                .get_parameter_value()
                .double_value
            ),
            miss_penalty=(
                self.get_parameter("miss_penalty")
                .get_parameter_value()
                .double_value
            ),
            wall_penalty=(
                self.get_parameter("wall_penalty")
                .get_parameter_value()
                .double_value
            ),
            unknown_penalty=(
                self.get_parameter("unknown_penalty")
                .get_parameter_value()
                .double_value
            ),
            max_range_clear_reward=(
                self.get_parameter("max_range_clear_reward")
                .get_parameter_value()
                .double_value
            ),
        )

        use_global = (
            self.get_parameter("use_global_localization")
            .get_parameter_value()
            .bool_value
        )

        candidates = self._sample_candidates(
            grid=grid,
            hint_x=hint_x,
            hint_y=hint_y,
            hint_yaw=hint_yaw,
            clearance_cells=clearance_cells,
            use_global=use_global,
            count=num_particles,
        )

        if not candidates:
            self.get_logger().warn(
                "[ParticleFilter] No traversable candidate poses sampled."
            )
            return None

        top_k = max(
            2,
            self.get_parameter("top_k").get_parameter_value().integer_value,
        )

        scoring_workers = max(
            1,
            self.get_parameter("scoring_workers")
            .get_parameter_value()
            .integer_value,
        )
        scoring_chunk_size = max(
            1,
            self.get_parameter("scoring_chunk_size")
            .get_parameter_value()
            .integer_value,
        )
        parallel_min_candidates = max(
            1,
            self.get_parameter("parallel_min_candidates")
            .get_parameter_value()
            .integer_value,
        )

        t0 = time.monotonic()

        self.get_logger().info(
            f"[ParticleFilter] Coarse scoring {len(candidates)} particles "
            f"with {len(scan_beams)} beams, ray_step={cfg.raytrace_step_m:.2f}m, "
            f"workers={scoring_workers}, chunk={scoring_chunk_size}..."
        )

        executor: Optional[ProcessPoolExecutor] = None

        try:
            if scoring_workers > 1 and len(candidates) >= parallel_min_candidates:
                executor = ProcessPoolExecutor(
                    max_workers=scoring_workers,
                    initializer=_init_score_worker,
                    initargs=(grid, scan_beams, cfg),
                )

            top_scores = self._score_candidate_set(
                candidates=candidates,
                scan_beams=scan_beams,
                grid=grid,
                cfg=cfg,
                top_k=top_k,
                executor=executor,
                chunk_size=scoring_chunk_size,
                parallel_min_candidates=parallel_min_candidates,
            )

            if not top_scores:
                self.get_logger().warn(
                    "[ParticleFilter] No candidate had enough valid beams."
                )
                return None

            refinement_rounds = max(
                0,
                self.get_parameter("refinement_rounds")
                .get_parameter_value()
                .integer_value,
            )
            refinement_particles = max(
                0,
                self.get_parameter("refinement_particles")
                .get_parameter_value()
                .integer_value,
            )
            refinement_top_k = max(
                1,
                self.get_parameter("refinement_top_k")
                .get_parameter_value()
                .integer_value,
            )
            xy_std = (
                self.get_parameter("refinement_xy_std_m")
                .get_parameter_value()
                .double_value
            )
            yaw_std = math.radians(
                self.get_parameter("refinement_yaw_std_deg")
                .get_parameter_value()
                .double_value
            )

            for round_index in range(refinement_rounds):
                anchors = top_scores[:refinement_top_k]

                refine_candidates = self._sample_refinement_candidates(
                    anchors=anchors,
                    grid=grid,
                    clearance_cells=clearance_cells,
                    count=refinement_particles,
                    xy_std=max(xy_std, grid.resolution),
                    yaw_std=max(yaw_std, math.radians(1.0)),
                )

                if not refine_candidates:
                    self.get_logger().warn(
                        f"[ParticleFilter] Refinement round {round_index + 1}: "
                        "no candidates."
                    )
                    break

                self.get_logger().info(
                    f"[ParticleFilter] Refinement round {round_index + 1}: "
                    f"scoring {len(refine_candidates)} particles."
                )

                refined_scores = self._score_candidate_set(
                    candidates=refine_candidates,
                    scan_beams=scan_beams,
                    grid=grid,
                    cfg=cfg,
                    top_k=top_k,
                    executor=executor,
                    chunk_size=scoring_chunk_size,
                    parallel_min_candidates=parallel_min_candidates,
                )

                top_scores = self._merge_top_scores(
                    top_scores,
                    refined_scores,
                    top_k,
                )

                xy_std *= 0.5
                yaw_std *= 0.5

        finally:
            if executor is not None:
                executor.shutdown(wait=True)

        elapsed = time.monotonic() - t0
        best = top_scores[0]
        second = top_scores[1] if len(top_scores) > 1 else None

        self.get_logger().info(
            f"[ParticleFilter] Done in {elapsed:.2f}s. "
            f"Best score={best.score:.3f} "
            f"at ({best.x:.3f},{best.y:.3f},{math.degrees(best.yaw):.1f}deg), "
            f"hits={best.hits}/{best.valid_beams}, "
            f"blocked={best.blocked}, misses={best.misses}"
        )

        if second is not None:
            margin = best.score - second.score
            ratio = (
                best.score / max(second.score, 1e-6)
                if second.score > 0.0
                else float("inf")
            )

            self.get_logger().info(
                f"[ParticleFilter] Second score={second.score:.3f} "
                f"at ({second.x:.3f},{second.y:.3f},"
                f"{math.degrees(second.yaw):.1f}deg), "
                f"margin={margin:.3f}, ratio={ratio:.2f}"
            )

        min_score = (
            self.get_parameter("min_score")
            .get_parameter_value()
            .double_value
        )

        if best.score < min_score:
            self.get_logger().warn(
                f"[ParticleFilter] Best score {best.score:.3f} "
                f"below min_score {min_score:.3f}."
            )
            return None

        self._check_ambiguity(best, second)

        if self._last_pose_ambiguous:
            reject_ambiguous = (
                self.get_parameter("reject_ambiguous_pose")
                .get_parameter_value()
                .bool_value
            )
            if reject_ambiguous:
                self.get_logger().warn(
                    "[ParticleFilter] Ambiguous pose rejected by configuration."
                )
                return None

        return best

    def _check_ambiguity(
        self,
        best: PoseScore,
        second: Optional[PoseScore],
    ) -> None:
        self._last_pose_ambiguous = False

        check_margin = (
            self.get_parameter("check_score_margin")
            .get_parameter_value()
            .bool_value
        )

        if not check_margin or second is None:
            return

        min_margin = (
            self.get_parameter("min_score_margin")
            .get_parameter_value()
            .double_value
        )
        min_ratio = (
            self.get_parameter("min_score_ratio")
            .get_parameter_value()
            .double_value
        )

        margin = best.score - second.score
        ratio = (
            best.score / max(second.score, 1e-6)
            if second.score > 0.0
            else float("inf")
        )

        if margin < min_margin and ratio < min_ratio:
            self._last_pose_ambiguous = True
            self.get_logger().warn(
                f"[ParticleFilter] Ambiguous result: "
                f"margin={margin:.3f} < {min_margin:.3f} "
                f"and ratio={ratio:.2f} < {min_ratio:.2f}."
            )

    def _score_candidate_set_serial(
        self,
        candidates: list[tuple[float, float, float]],
        scan_beams: list[ScanBeam],
        grid: GridInfo,
        cfg: ScoreConfig,
        top_k: int,
    ) -> list[PoseScore]:
        scores: list[PoseScore] = []

        for px, py, pyaw in candidates:
            score = _score_pose(px, py, pyaw, scan_beams, grid, cfg)
            if score.valid_beams >= cfg.min_beams:
                scores.append(score)

        scores.sort(key=lambda item: item.score, reverse=True)
        return scores[:top_k]

    def _score_candidate_set(
        self,
        candidates: list[tuple[float, float, float]],
        scan_beams: list[ScanBeam],
        grid: GridInfo,
        cfg: ScoreConfig,
        top_k: int,
        executor: Optional[ProcessPoolExecutor] = None,
        chunk_size: int = 64,
        parallel_min_candidates: int = 300,
    ) -> list[PoseScore]:
        if executor is None or len(candidates) < parallel_min_candidates:
            return self._score_candidate_set_serial(
                candidates,
                scan_beams,
                grid,
                cfg,
                top_k,
            )

        chunks = _candidate_chunks(candidates, chunk_size)
        scores: list[PoseScore] = []

        try:
            for chunk_scores in executor.map(
                _score_candidate_chunk_worker,
                ((chunk, top_k) for chunk in chunks),
                chunksize=1,
            ):
                scores.extend(chunk_scores)

        except Exception as exc:
            self.get_logger().warn(
                f"[ParticleFilter] Parallel scoring failed: {exc}. "
                "Falling back to serial scoring."
            )
            return self._score_candidate_set_serial(
                candidates,
                scan_beams,
                grid,
                cfg,
                top_k,
            )

        scores.sort(key=lambda item: item.score, reverse=True)
        return scores[:top_k]

    def _merge_top_scores(
        self,
        primary: list[PoseScore],
        secondary: list[PoseScore],
        top_k: int,
    ) -> list[PoseScore]:
        merged = primary + secondary
        merged.sort(key=lambda item: item.score, reverse=True)
        return merged[:top_k]

    def _sample_candidates(
        self,
        grid: GridInfo,
        hint_x: float,
        hint_y: float,
        hint_yaw: float,
        clearance_cells: int,
        use_global: bool,
        count: int,
    ) -> list[tuple[float, float, float]]:
        candidates: list[tuple[float, float, float]] = []

        if _is_traversable_pose(hint_x, hint_y, grid, clearance_cells):
            candidates.append((hint_x, hint_y, _normalize_angle(hint_yaw)))
        else:
            self.get_logger().warn(
                "[ParticleFilter] Hint pose is not on a free map cell; "
                "it will only be used for fallback."
            )

        max_attempts = (
            self.get_parameter("max_sampling_attempts")
            .get_parameter_value()
            .integer_value
        )
        max_attempts = max(max_attempts, count * 4)

        if use_global:
            free_cells = self._free_cells(grid, clearance_cells)

            if not free_cells:
                self.get_logger().warn(
                    "[ParticleFilter] Map has no free cells suitable for global sampling."
                )
                return candidates

            yaw_bins = (
                self.get_parameter("yaw_bins")
                .get_parameter_value()
                .integer_value
            )

            self.get_logger().info(
                f"[ParticleFilter] Global search over {len(free_cells)} free cells, "
                f"target_particles={count}, yaw_bins={yaw_bins}."
            )

            attempts = 0

            while len(candidates) < count and attempts < max_attempts:
                attempts += 1
                gx, gy = random.choice(free_cells)
                px, py = _grid_to_world_random(gx, gy, grid)
                candidates.append((px, py, self._sample_global_yaw(yaw_bins)))

            if len(candidates) < count:
                self.get_logger().warn(
                    f"[ParticleFilter] Sampled {len(candidates)}/{count} "
                    "global particles."
                )

            return candidates[:count]

        search_range = (
            self.get_parameter("search_range_m")
            .get_parameter_value()
            .double_value
        )
        local_yaw_deg = (
            self.get_parameter("local_yaw_search_deg")
            .get_parameter_value()
            .double_value
        )

        yaw_range = math.radians(abs(local_yaw_deg))
        x_min = hint_x - search_range
        x_max = hint_x + search_range
        y_min = hint_y - search_range
        y_max = hint_y + search_range

        self.get_logger().info(
            f"[ParticleFilter] Local search around ({hint_x:.2f},{hint_y:.2f}) "
            f"range={search_range:.1f}m yaw=+/-{local_yaw_deg:.1f}deg."
        )

        attempts = 0

        while len(candidates) < count and attempts < max_attempts:
            attempts += 1
            px = random.uniform(x_min, x_max)
            py = random.uniform(y_min, y_max)

            if not _is_traversable_pose(px, py, grid, clearance_cells):
                continue

            pyaw = _normalize_angle(
                hint_yaw + random.uniform(-yaw_range, yaw_range)
            )
            candidates.append((px, py, pyaw))

        if len(candidates) < count:
            self.get_logger().warn(
                f"[ParticleFilter] Sampled {len(candidates)}/{count} local particles "
                f"after {attempts} attempts."
            )

        return candidates

    def _sample_global_yaw(self, yaw_bins: int) -> float:
        if yaw_bins >= 2:
            step = 2.0 * math.pi / yaw_bins
            return _normalize_angle(
                random.randrange(yaw_bins) * step
                + random.uniform(-0.5 * step, 0.5 * step)
            )

        return random.uniform(-math.pi, math.pi)

    def _sample_refinement_candidates(
        self,
        anchors: list[PoseScore],
        grid: GridInfo,
        clearance_cells: int,
        count: int,
        xy_std: float,
        yaw_std: float,
    ) -> list[tuple[float, float, float]]:
        if not anchors or count <= 0:
            return []

        candidates: list[tuple[float, float, float]] = [
            (anchor.x, anchor.y, anchor.yaw)
            for anchor in anchors
        ]

        max_attempts = max(count * 10, 1000)
        attempts = 0

        while len(candidates) < count and attempts < max_attempts:
            attempts += 1
            anchor = random.choice(anchors)

            px = random.gauss(anchor.x, xy_std)
            py = random.gauss(anchor.y, xy_std)

            if not _is_traversable_pose(px, py, grid, clearance_cells):
                continue

            pyaw = _normalize_angle(random.gauss(anchor.yaw, yaw_std))
            candidates.append((px, py, pyaw))

        if len(candidates) < count:
            self.get_logger().warn(
                f"[ParticleFilter] Refinement sampled {len(candidates)}/{count} "
                "particles."
            )

        return candidates

    def _free_cells(
        self,
        grid: GridInfo,
        clearance_cells: int,
    ) -> list[tuple[int, int]]:
        signature = (
            grid.width,
            grid.height,
            round(grid.resolution, 6),
            round(grid.origin_x, 3),
            round(grid.origin_y, 3),
            clearance_cells,
        )

        if self._free_cell_cache_signature == signature and self._free_cell_cache:
            return self._free_cell_cache

        cells: list[tuple[int, int]] = []

        for gy in range(grid.height):
            for gx in range(grid.width):
                x, y = _grid_to_world_center(gx, gy, grid)

                if _is_traversable_pose(x, y, grid, clearance_cells):
                    cells.append((gx, gy))

        self._free_cell_cache_signature = signature
        self._free_cell_cache = cells

        return cells

    def _publish_pose(
        self,
        x: float,
        y: float,
        yaw_rad: float,
        cov_xy: float,
        cov_yaw: float,
    ) -> None:
        msg = PoseWithCovarianceStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = (
            self.get_parameter("initial_pose_frame")
            .get_parameter_value()
            .string_value
        )

        msg.pose.pose.position.x = x
        msg.pose.pose.position.y = y
        msg.pose.pose.position.z = 0.0
        msg.pose.pose.orientation = _yaw_to_quat(yaw_rad)

        # PoseWithCovariance stores variances, not standard deviations.
        cov = [0.0] * 36
        cov[0] = cov_xy * cov_xy
        cov[7] = cov_xy * cov_xy
        cov[35] = cov_yaw * cov_yaw
        msg.pose.covariance = cov

        self._initialpose_pub.publish(msg)

        self.get_logger().info(
            f"[InitialPose] Published pose #{self._retry_count + 1}: "
            f"x={x:.3f} y={y:.3f} yaw={math.degrees(yaw_rad):.1f}deg "
            f"cov_xy={cov_xy:.3f} cov_yaw={cov_yaw:.3f}"
        )

    def _publish_hint_pose(self) -> None:
        x = (
            self.get_parameter("initial_x")
            .get_parameter_value()
            .double_value
        )
        y = (
            self.get_parameter("initial_y")
            .get_parameter_value()
            .double_value
        )
        yaw_deg = (
            self.get_parameter("initial_yaw_deg")
            .get_parameter_value()
            .double_value
        )

        cov_xy = (
            self.get_parameter("pose_cov_xy")
            .get_parameter_value()
            .double_value
        )
        cov_yaw = (
            self.get_parameter("pose_cov_yaw")
            .get_parameter_value()
            .double_value
        )

        use_global = (
            self.get_parameter("use_global_localization")
            .get_parameter_value()
            .bool_value
        )

        if use_global:
            cov_xy = max(cov_xy, 2.0)
            cov_yaw = max(cov_yaw, 1.0)

        self._publish_pose(
            x,
            y,
            math.radians(yaw_deg),
            cov_xy,
            cov_yaw,
        )


def main(args=None) -> int:
    rclpy.init(args=args)
    node = LidarInitialposeNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
