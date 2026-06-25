"""Nav2 controller adapter for the async-agent controller interface."""

from __future__ import annotations

import json
import math
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import OccupancyGrid, Odometry
from PIL import Image as PILImage
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import Image as ROSImage, LaserScan
from tf2_ros import Buffer, TransformException, TransformListener

from caragent_agent.config.config import config
from caragent_agent.controller.controller_base import Base_Controller


def _yaw_deg_to_quaternion(yaw_deg: float) -> list[float]:
    yaw = math.radians(float(yaw_deg))
    half = yaw * 0.5
    return [0.0, 0.0, math.sin(half), math.cos(half)]


def _quaternion_to_yaw_rad(quat: Any) -> float:
    x, y, z, w = (float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def _normalize_angle_rad(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def _stamp_sec(msg: Any) -> float:
    stamp = getattr(getattr(msg, "header", None), "stamp", None)
    if stamp is None:
        return 0.0
    return float(getattr(stamp, "sec", 0)) + float(getattr(stamp, "nanosec", 0)) * 1e-9


def _occupancy_grid_to_dict(msg: OccupancyGrid, topic: str) -> dict[str, Any]:
    origin = msg.info.origin.position
    return {
        "source": msg.header.frame_id or "map",
        "topic": str(topic or ""),
        "width": int(msg.info.width),
        "height": int(msg.info.height),
        "resolution": float(msg.info.resolution),
        "origin": [float(origin.x), float(origin.y)],
        "data": list(msg.data),
        "stamp_sec": _stamp_sec(msg),
    }


def _ccw_delta_rad(current_yaw: float, target_yaw: float) -> float:
    return (float(target_yaw) - float(current_yaw)) % (2.0 * math.pi)


def _shortest_delta_rad(current_yaw: float, target_yaw: float) -> float:
    return _normalize_angle_rad(float(target_yaw) - float(current_yaw))


class Nav2Controller(Base_Controller):
    """Wrap Nav2 NavigateToPose while preserving the existing controller API."""

    def __init__(
        self,
        node: Node,
        *,
        action_name: str = "navigate_to_pose",
        global_frame: str = "map",
        base_frame: str = "base_link",
        dry_run: bool = False,
        camera_topic: str = "/stereo/left/image_raw",
        right_image_topic: str = "/stereo/right/image_raw",
        odom_topic: str = "/odom",
        scan_topic: str = "/scan",
        map_topic: str = "/global_costmap/costmap",
        arrival_tolerance_m: float = 0.25,
        enable_rotation_takeover: bool = True,
        rotation_policy: str = "left_only",
        pre_align_enabled: bool = True,
        final_align_enabled: bool = True,
        yaw_tolerance_deg: float = 4.0,
        settle_time_sec: float = 0.7,
        fast_omega: float = 1.45,
        mid_omega: float = 0.95,
        slow_omega: float = 0.40,
        rotation_timeout_sec: float = 15.0,
        rotation_loop_rate_hz: float = 20.0,
        right_turn_shortcut_deg: float = 90.0,
        localization_handoff_gate_enabled: bool = True,
        localization_handoff_settle_sec: float = 1.0,
        localization_handoff_timeout_sec: float = 2.0,
        localization_handoff_max_translation_m: float = 0.12,
        localization_handoff_max_yaw_deg: float = 8.0,
        localization_handoff_sensor_max_age_sec: float = 1.5,
        simulation_mode: bool = False,
        simulation_navigation_delay_sec: float = 30.0,
        simulation_navigation_delay_per_meter_sec: float = 0.0,
        simulation_initial_position: list[float] | None = None,
        simulation_initial_yaw_deg: float = 0.0,
    ) -> None:
        self._node = node
        self._action_name = action_name
        self._global_frame = global_frame
        self._base_frame = base_frame
        self._dry_run = bool(dry_run)
        self._arrival_tolerance_m = max(0.0, float(arrival_tolerance_m))
        self._enable_rotation_takeover = bool(enable_rotation_takeover)
        self._rotation_policy = str(rotation_policy or "left_only").strip().lower()
        self._pre_align_enabled = bool(pre_align_enabled)
        self._final_align_enabled = bool(final_align_enabled)
        self._yaw_tolerance_rad = math.radians(max(0.1, float(yaw_tolerance_deg)))
        self._settle_time_sec = max(0.0, float(settle_time_sec))
        self._fast_omega = max(0.0, float(fast_omega))
        self._mid_omega = max(0.0, float(mid_omega))
        self._slow_omega = max(0.0, float(slow_omega))
        self._rotation_timeout_sec = max(0.5, float(rotation_timeout_sec))
        self._rotation_loop_period_sec = 1.0 / max(1.0, float(rotation_loop_rate_hz))
        self._right_turn_shortcut_rad = math.radians(
            max(0.0, float(right_turn_shortcut_deg))
        )
        self._localization_handoff_gate_enabled = bool(localization_handoff_gate_enabled)
        self._localization_handoff_settle_sec = max(
            0.0,
            float(localization_handoff_settle_sec),
        )
        self._localization_handoff_timeout_sec = max(
            self._localization_handoff_settle_sec,
            float(localization_handoff_timeout_sec),
        )
        self._localization_handoff_max_translation_m = max(
            0.0,
            float(localization_handoff_max_translation_m),
        )
        self._localization_handoff_max_yaw_rad = math.radians(
            max(0.0, float(localization_handoff_max_yaw_deg))
        )
        self._localization_handoff_sensor_max_age_sec = max(
            0.1,
            float(localization_handoff_sensor_max_age_sec),
        )
        self._simulation_mode = bool(simulation_mode)
        self._simulation_navigation_delay_sec = max(
            0.0,
            float(simulation_navigation_delay_sec),
        )
        self._simulation_navigation_delay_per_meter_sec = max(
            0.0,
            float(simulation_navigation_delay_per_meter_sec),
        )
        initial_position = simulation_initial_position
        if not isinstance(initial_position, (list, tuple)) or len(initial_position) < 2:
            initial_position = [0.0, 0.0, 0.0]
        self._sim_position = [
            float(initial_position[0]),
            float(initial_position[1]),
            float(initial_position[2]) if len(initial_position) >= 3 else 0.0,
        ]
        self._sim_yaw_deg = float(simulation_initial_yaw_deg)
        self._sim_keyframes_cache: list[dict[str, Any]] | None = None
        self._sim_image_cache: tuple[int, PILImage.Image] | None = None
        self._sim_right_image_cache: tuple[int, PILImage.Image] | None = None
        self._sim_image_metadata: dict[str, Any] | None = None
        self._cmd_vel_pub = node.create_publisher(Twist, "/cmd_vel", 10)
        self._action_client = ActionClient(node, NavigateToPose, action_name)
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, node)
        self._status = "idle"
        self._latest_msg = ""
        self._goal_handle = None
        self._current_goal_position: list[float] | None = None
        self._current_goal_final_yaw_deg = 0.0
        self._goal_sequence = 0
        self._active_goal_sequence = 0
        self._cancel_requested = False
        self._last_localization_handoff_reason = "not checked"
        self._lock = threading.RLock()
        self._latest_image: PILImage.Image | None = None
        self._latest_right_image: PILImage.Image | None = None
        self._latest_odom: Odometry | None = None
        self._latest_scan: LaserScan | None = None
        self._latest_odom_monotonic = 0.0
        self._latest_scan_monotonic = 0.0
        self._latest_map: dict[str, Any] | None = None
        self._image_sub = node.create_subscription(
            ROSImage, camera_topic, self._on_camera_image, 10
        )
        self._right_image_sub = node.create_subscription(
            ROSImage, right_image_topic, self._on_right_camera_image, 10
        )
        self._odom_sub = node.create_subscription(
            Odometry, odom_topic, self._on_odom, 10
        )
        self._scan_sub = node.create_subscription(
            LaserScan, scan_topic, self._on_scan, 10
        )
        self._map_topic = str(map_topic or "/global_costmap/costmap")
        self._map_sub = node.create_subscription(
            OccupancyGrid, self._map_topic, self._on_map, 1
        )

    def update_path(self, new_path: list[list[float]]) -> None:
        if not new_path:
            raise ValueError("Nav2Controller.update_path received an empty path.")

        waypoint = list(new_path[-1])
        if len(waypoint) < 2:
            raise ValueError(f"Waypoint must contain at least x and y: {waypoint}")

        final_yaw_deg = float(waypoint[3]) if len(waypoint) >= 4 else 0.0
        nav_yaw_deg = final_yaw_deg
        bearing_yaw = None
        if self._should_takeover_rotation():
            bearing_yaw = self._bearing_yaw_to_waypoint(waypoint)
            if bearing_yaw is not None:
                nav_yaw_deg = math.degrees(bearing_yaw)

        yaw_quat = _yaw_deg_to_quaternion(nav_yaw_deg)
        pose = PoseStamped()
        pose.header.frame_id = self._global_frame
        pose.header.stamp = self._node.get_clock().now().to_msg()
        pose.pose.position.x = float(waypoint[0])
        pose.pose.position.y = float(waypoint[1])
        pose.pose.position.z = float(waypoint[2]) if len(waypoint) >= 3 else 0.0
        pose.pose.orientation.x = yaw_quat[0]
        pose.pose.orientation.y = yaw_quat[1]
        pose.pose.orientation.z = yaw_quat[2]
        pose.pose.orientation.w = yaw_quat[3]
        goal_position = [pose.pose.position.x, pose.pose.position.y, pose.pose.position.z]
        with self._lock:
            self._goal_sequence += 1
            goal_sequence = self._goal_sequence
            self._active_goal_sequence = goal_sequence
            self._current_goal_position = goal_position
            self._current_goal_final_yaw_deg = final_yaw_deg
            self._cancel_requested = False
            self._latest_msg = ""

        self._node.get_logger().info(
            "Nav2 handoff debug: seq={seq} goal=({x:.3f},{y:.3f},{z:.3f}) "
            "final_yaw={final_yaw:.1f}deg nav_yaw={nav_yaw:.1f}deg "
            "bearing_yaw={bearing} takeover={takeover} pre_align={pre_align} "
            "handoff_gate={gate} timeout={timeout:.2f}s sensor_max_age={sensor_age:.2f}s".format(
                seq=goal_sequence,
                x=goal_position[0],
                y=goal_position[1],
                z=goal_position[2],
                final_yaw=final_yaw_deg,
                nav_yaw=nav_yaw_deg,
                bearing=(
                    f"{math.degrees(bearing_yaw):.1f}deg"
                    if bearing_yaw is not None
                    else "none"
                ),
                takeover=self._should_takeover_rotation(),
                pre_align=self._pre_align_enabled,
                gate=self._localization_handoff_gate_enabled,
                timeout=self._localization_handoff_timeout_sec,
                sensor_age=self._localization_handoff_sensor_max_age_sec,
            )
        )

        if self._simulation_mode:
            self._dispatch_simulated_navigation(
                goal_position,
                final_yaw_deg=final_yaw_deg,
                goal_sequence=goal_sequence,
            )
            return

        if self._dry_run:
            self.update_status("dry_run_dispatched")
            self.update_latest_msg(self._arrival_message(goal_position))
            return

        if not self._action_client.wait_for_server(timeout_sec=5.0):
            self.update_status("failed")
            raise RuntimeError(f"Nav2 action server '{self._action_name}' is unavailable.")

        if not self._wait_for_localization_handoff(
            goal_sequence=goal_sequence,
            stage="before-pre-align",
        ):
            self.update_status("failed")
            raise RuntimeError(
                "Localization handoff gate failed before pre-align/Nav2 dispatch: "
                f"{self._get_localization_handoff_reason()}"
            )
        self._node.get_logger().info(
            "Nav2 handoff debug: before-pre-align accepted seq={seq}: {reason}".format(
                seq=goal_sequence,
                reason=self._get_localization_handoff_reason(),
            )
        )

        if (
            self._should_takeover_rotation()
            and self._pre_align_enabled
            and bearing_yaw is not None
        ):
            self.update_status("pre_aligning")
            pre_align_started = time.monotonic()
            current_yaw = self._current_yaw_rad()
            self._node.get_logger().info(
                "Nav2 handoff debug: pre-align starting seq={seq} "
                "current_yaw={current} target_yaw={target:.1f}deg".format(
                    seq=goal_sequence,
                    current=(
                        f"{math.degrees(current_yaw):.1f}deg"
                        if current_yaw is not None
                        else "unavailable"
                    ),
                    target=math.degrees(bearing_yaw),
                )
            )
            aligned = self._rotate_left_only_to_yaw(
                bearing_yaw,
                goal_sequence=goal_sequence,
                stage="pre-align",
            )
            if not self._is_active_goal_sequence(goal_sequence):
                self._publish_stop()
                return
            current_yaw = self._current_yaw_rad()
            self._node.get_logger().info(
                "Nav2 handoff debug: pre-align finished seq={seq} aligned={aligned} "
                "elapsed={elapsed:.2f}s current_yaw={current}".format(
                    seq=goal_sequence,
                    aligned=aligned,
                    elapsed=time.monotonic() - pre_align_started,
                    current=(
                        f"{math.degrees(current_yaw):.1f}deg"
                        if current_yaw is not None
                        else "unavailable"
                    ),
                )
            )
            if aligned:
                time.sleep(self._settle_time_sec)
            else:
                self._node.get_logger().warning(
                    "Pre-align rotation did not converge before timeout; dispatching Nav2 anyway."
                )
            if not self._wait_for_localization_handoff(
                goal_sequence=goal_sequence,
                stage="after-pre-align",
            ):
                self.update_status("failed")
                raise RuntimeError(
                    "Localization handoff gate failed after pre-align; Nav2 goal was not dispatched: "
                    f"{self._get_localization_handoff_reason()}"
                )
            self._node.get_logger().info(
                "Nav2 handoff debug: after-pre-align advisory complete seq={seq}: {reason}".format(
                    seq=goal_sequence,
                    reason=self._get_localization_handoff_reason(),
                )
            )

        goal = NavigateToPose.Goal()
        goal.pose = pose
        self.update_status("navigating")
        self.update_latest_msg(
            f"Dispatched Nav2 goal x={pose.pose.position.x:.2f}, y={pose.pose.position.y:.2f}"
        )
        self._node.get_logger().info(
            "Nav2 goal dispatched: seq={seq} x={x:.3f}, y={y:.3f}, z={z:.3f}, "
            "nav_yaw={nav_yaw:.1f}deg final_yaw={final_yaw:.1f}deg "
            "arrival_tolerance={tol:.3f}m".format(
                seq=goal_sequence,
                x=goal_position[0],
                y=goal_position[1],
                z=goal_position[2],
                nav_yaw=nav_yaw_deg,
                final_yaw=final_yaw_deg,
                tol=self._arrival_tolerance_m,
            )
        )
        send_future = self._action_client.send_goal_async(goal)
        send_future.add_done_callback(lambda future, seq=goal_sequence: self._on_goal_response(future, seq))

    def update_status(self, status: str) -> None:
        with self._lock:
            self._status = str(status or "idle")

    def update_latest_msg(self, msg: str) -> None:
        with self._lock:
            self._latest_msg = str(msg or "")

    def get_current_state(self) -> dict[str, Any]:
        if self._simulation_mode:
            with self._lock:
                position = list(self._sim_position)
                yaw_deg = float(self._sim_yaw_deg)
                occupancy_grid = self._latest_map
            state = {
                "position": position,
                "orientation": _yaw_deg_to_quaternion(yaw_deg),
                "yaw_deg": yaw_deg,
                "status": self.get_status(),
                "source": "simulation",
                "simulation": {
                    "enabled": True,
                    "navigation_delay_sec": self._simulation_navigation_delay_sec,
                    "navigation_delay_per_meter_sec": self._simulation_navigation_delay_per_meter_sec,
                },
            }
            nearest_keyframe = self._nearest_simulation_keyframe()
            if nearest_keyframe is not None:
                state["simulation"]["nearest_keyframe_id"] = nearest_keyframe.get("kf_id")
            if occupancy_grid is not None:
                state["occupancy_grid"] = occupancy_grid
            return state

        try:
            transform = self._tf_buffer.lookup_transform(
                self._global_frame,
                self._base_frame,
                Time(),
                timeout=Duration(seconds=0.2),
            )
        except TransformException as exc:
            fallback = self._get_odom_state()
            if fallback is not None:
                fallback["status"] = self.get_status()
                fallback["source"] = "odom"
                fallback["tf_error"] = str(exc)
                return fallback
            return {
                "position": None,
                "orientation": None,
                "status": self.get_status(),
                "source": "unavailable",
                "error": str(exc),
            }

        translation = transform.transform.translation
        rotation = transform.transform.rotation
        with self._lock:
            occupancy_grid = self._latest_map
        state = {
            "position": [translation.x, translation.y, translation.z],
            "orientation": [rotation.x, rotation.y, rotation.z, rotation.w],
            "status": self.get_status(),
            "source": "tf",
        }
        if occupancy_grid is not None:
            state["occupancy_grid"] = occupancy_grid
        return state

    def get_current_image(self) -> Any:
        with self._lock:
            image = self._latest_image
        if image is not None or not self._simulation_mode:
            return image
        return self._get_simulated_current_image()

    def get_current_image_metadata(self) -> dict[str, Any]:
        """Return compact metadata for the image returned by get_current_image()."""

        if not self._simulation_mode:
            return {"source_type": "live_view"}
        with self._lock:
            if self._latest_image is not None:
                return {"source_type": "live_view", "simulation_mode": True}
            if self._sim_image_metadata:
                return dict(self._sim_image_metadata)
        return {"source_type": "simulated_view_unavailable", "simulation_mode": True}

    def get_current_right_image(self) -> Any:
        with self._lock:
            image = self._latest_right_image
        if image is not None or not self._simulation_mode:
            return image
        return self._get_simulated_current_right_image()

    def get_current_scan(self) -> Any:
        with self._lock:
            return self._latest_scan

    def _simulation_dataset_dir(self) -> Path | None:
        scene_cfg = config.get("scene_memory") if isinstance(config.get("scene_memory"), dict) else {}
        paths_cfg = config.get("paths") if isinstance(config.get("paths"), dict) else {}
        raw_path = (
            scene_cfg.get("dataset_dir")
            or paths_cfg.get("default_dataset_dir")
            or paths_cfg.get("scene_memory_dataset_dir")
        )
        if not raw_path:
            return None
        return Path(str(raw_path)).expanduser()

    def _load_simulation_keyframes(self) -> list[dict[str, Any]]:
        if self._sim_keyframes_cache is not None:
            return self._sim_keyframes_cache

        dataset_dir = self._simulation_dataset_dir()
        keyframes: list[dict[str, Any]] = []
        if dataset_dir is None:
            self._sim_keyframes_cache = keyframes
            return keyframes

        node_dir = dataset_dir / "constructed_memory" / "keyframe_nodes"
        if not node_dir.exists():
            self._node.get_logger().warning(
                f"Simulation current-image fallback skipped; keyframe dir not found: {node_dir}"
            )
            self._sim_keyframes_cache = keyframes
            return keyframes

        for json_path in sorted(node_dir.glob("kf_*.json")):
            try:
                data = json.loads(json_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            try:
                position = [float(v) for v in (data.get("position") or [0.0, 0.0, 0.0])]
                kf_id = int(data.get("kf_id"))
            except Exception:
                continue
            image_path = data.get("rgb_path") or data.get("left_path")
            if not image_path:
                continue
            resolved_image_path = Path(str(image_path))
            if not resolved_image_path.is_absolute():
                resolved_image_path = dataset_dir / resolved_image_path
            right_image_path = data.get("right_path")
            resolved_right_image_path = None
            if right_image_path:
                resolved_right_image_path = Path(str(right_image_path))
                if not resolved_right_image_path.is_absolute():
                    resolved_right_image_path = dataset_dir / resolved_right_image_path
            keyframes.append(
                {
                    "kf_id": kf_id,
                    "position": position,
                    "image_path": resolved_image_path,
                    "right_image_path": resolved_right_image_path,
                }
            )

        self._sim_keyframes_cache = keyframes
        return keyframes

    def _nearest_simulation_keyframe(self) -> dict[str, Any] | None:
        keyframes = self._load_simulation_keyframes()
        if not keyframes:
            return None
        with self._lock:
            position = list(self._sim_position)
        best: dict[str, Any] | None = None
        best_distance = float("inf")
        for keyframe in keyframes:
            kf_position = keyframe.get("position") or []
            if len(kf_position) < 2:
                continue
            distance = math.hypot(float(position[0]) - float(kf_position[0]), float(position[1]) - float(kf_position[1]))
            if distance < best_distance:
                best = keyframe
                best_distance = distance
        return best

    def _get_simulated_current_image(self) -> PILImage.Image | None:
        keyframe = self._nearest_simulation_keyframe()
        if not keyframe:
            return None
        try:
            kf_id = int(keyframe["kf_id"])
        except Exception:
            return None
        with self._lock:
            if self._sim_image_cache is not None and self._sim_image_cache[0] == kf_id:
                self._sim_image_metadata = {
                    "source_type": "simulated_keyframe_view",
                    "simulation_mode": True,
                    "keyframe_id": kf_id,
                    "image_path": str(keyframe.get("image_path") or ""),
                    "semantic_warning": (
                        "This image is from historical keyframe memory, not a live camera frame; "
                        "temporary/new objects may be absent."
                    ),
                }
                return self._sim_image_cache[1].copy()

        image_path = keyframe.get("image_path")
        if not isinstance(image_path, Path) or not image_path.exists():
            self._node.get_logger().warning(
                f"Simulation current-image fallback missing image for keyframe {kf_id}: {image_path}"
            )
            return None
        try:
            image = PILImage.open(image_path).convert("RGB")
        except Exception as exc:
            self._node.get_logger().warning(
                f"Simulation current-image fallback failed for keyframe {kf_id}: {exc}"
            )
            return None
        with self._lock:
            self._sim_image_cache = (kf_id, image.copy())
            self._sim_image_metadata = {
                "source_type": "simulated_keyframe_view",
                "simulation_mode": True,
                "keyframe_id": kf_id,
                "image_path": str(image_path),
                "semantic_warning": (
                    "This image is from historical keyframe memory, not a live camera frame; "
                    "temporary/new objects may be absent."
                ),
            }
        self._node.get_logger().info(
            f"Simulation current-image fallback using keyframe {kf_id}: {image_path}"
        )
        return image

    def _get_simulated_current_right_image(self) -> PILImage.Image | None:
        keyframe = self._nearest_simulation_keyframe()
        if not keyframe:
            return None
        try:
            kf_id = int(keyframe["kf_id"])
        except Exception:
            return None
        with self._lock:
            if self._sim_right_image_cache is not None and self._sim_right_image_cache[0] == kf_id:
                return self._sim_right_image_cache[1].copy()

        image_path = keyframe.get("right_image_path")
        if not isinstance(image_path, Path) or not image_path.exists():
            self._node.get_logger().warning(
                f"Simulation right-image fallback missing image for keyframe {kf_id}: {image_path}"
            )
            return None
        try:
            image = PILImage.open(image_path).convert("RGB")
        except Exception as exc:
            self._node.get_logger().warning(
                f"Simulation right-image fallback failed for keyframe {kf_id}: {exc}"
            )
            return None
        with self._lock:
            self._sim_right_image_cache = (kf_id, image.copy())
        self._node.get_logger().info(
            f"Simulation right-image fallback using keyframe {kf_id}: {image_path}"
        )
        return image

    def _on_camera_image(self, msg: ROSImage) -> None:
        try:
            image = self._ros_image_to_pil(msg)
            with self._lock:
                self._latest_image = image
        except Exception:
            pass  # silently skip corrupt frames

    def _on_right_camera_image(self, msg: ROSImage) -> None:
        try:
            image = self._ros_image_to_pil(msg)
            with self._lock:
                self._latest_right_image = image
        except Exception:
            pass

    def _on_odom(self, msg: Odometry) -> None:
        with self._lock:
            self._latest_odom = msg
            self._latest_odom_monotonic = time.monotonic()

    def _on_scan(self, msg: LaserScan) -> None:
        with self._lock:
            self._latest_scan = msg
            self._latest_scan_monotonic = time.monotonic()

    def _on_map(self, msg: OccupancyGrid) -> None:
        grid = _occupancy_grid_to_dict(msg, self._map_topic)
        with self._lock:
            self._latest_map = grid

    def _ros_image_to_pil(self, msg: ROSImage) -> PILImage.Image:
        encoding = (msg.encoding or "").lower()
        height = int(msg.height)
        width = int(msg.width)
        step = int(msg.step)
        if encoding in {"bgr8", "rgb8", "rgba8", "bgra8"}:
            channels = 4 if encoding in {"rgba8", "bgra8"} else 3
            raw = np.frombuffer(msg.data, dtype=np.uint8).reshape(height, step)
            data = raw[:, : width * channels].reshape(height, width, channels).copy()
            if encoding == "bgr8":
                data = data[:, :, ::-1]
            elif encoding == "bgra8":
                data = data[:, :, [2, 1, 0, 3]]
            if channels == 4:
                data = data[:, :, :3]
            return PILImage.fromarray(data).convert("RGB")
        if encoding in {"mono8", "8uc1"}:
            raw = np.frombuffer(msg.data, dtype=np.uint8).reshape(height, step)
            return PILImage.fromarray(raw[:, :width].copy(), mode="L").convert("RGB")
        if encoding in {"mono16", "16uc1"}:
            row_words = step // 2
            raw = np.frombuffer(msg.data, dtype=np.uint16).reshape(height, row_words)
            gray16 = raw[:, :width].copy()
            scale = 255.0 / max(1.0, float(gray16.max()))
            gray8 = np.clip(gray16.astype(np.float32) * scale, 0, 255).astype(np.uint8)
            return PILImage.fromarray(gray8, mode="L").convert("RGB")
        raise ValueError(f"unsupported image encoding: {msg.encoding}")

    def _get_odom_state(self) -> dict[str, Any] | None:
        with self._lock:
            odom = self._latest_odom
        if odom is None:
            return None
        position = odom.pose.pose.position
        orientation = odom.pose.pose.orientation
        return {
            "position": [position.x, position.y, position.z],
            "orientation": [orientation.x, orientation.y, orientation.z, orientation.w],
        }

    def _lookup_map_base_state(self, *, timeout_sec: float = 0.1) -> dict[str, Any] | None:
        try:
            transform = self._tf_buffer.lookup_transform(
                self._global_frame,
                self._base_frame,
                Time(),
                timeout=Duration(seconds=max(0.0, float(timeout_sec))),
            )
        except TransformException:
            return None
        translation = transform.transform.translation
        rotation = transform.transform.rotation
        return {
            "position": [translation.x, translation.y, translation.z],
            "yaw": _quaternion_to_yaw_rad([rotation.x, rotation.y, rotation.z, rotation.w]),
        }

    def _sensor_freshness(self) -> tuple[float | None, float | None]:
        now = time.monotonic()
        with self._lock:
            odom_time = float(self._latest_odom_monotonic or 0.0)
            scan_time = float(self._latest_scan_monotonic or 0.0)
        odom_age = now - odom_time if odom_time > 0.0 else None
        scan_age = now - scan_time if scan_time > 0.0 else None
        return odom_age, scan_age

    def _handoff_sensors_are_fresh(self) -> tuple[bool, str]:
        odom_age, scan_age = self._sensor_freshness()
        max_age = self._localization_handoff_sensor_max_age_sec
        if odom_age is None:
            return False, "no /odom sample received"
        if scan_age is None:
            return False, "no /scan sample received"
        if odom_age > max_age:
            return False, f"/odom age {odom_age:.2f}s > {max_age:.2f}s"
        if scan_age > max_age:
            return False, f"/scan age {scan_age:.2f}s > {max_age:.2f}s"
        return True, f"odom_age={odom_age:.2f}s scan_age={scan_age:.2f}s"

    def _set_localization_handoff_reason(self, reason: str) -> None:
        with self._lock:
            self._last_localization_handoff_reason = str(reason or "unknown")

    def _get_localization_handoff_reason(self) -> str:
        with self._lock:
            return str(self._last_localization_handoff_reason or "unknown")

    def _wait_for_localization_handoff(self, *, goal_sequence: int, stage: str) -> bool:
        if (
            not self._localization_handoff_gate_enabled
            or self._simulation_mode
            or self._dry_run
        ):
            self._set_localization_handoff_reason(
                f"{stage}: skipped (enabled={self._localization_handoff_gate_enabled}, "
                f"simulation={self._simulation_mode}, dry_run={self._dry_run})"
            )
            return True
        if not self._is_active_goal_sequence(goal_sequence):
            self._set_localization_handoff_reason(f"{stage}: inactive goal sequence")
            return False

        self._publish_stop()
        timeout_sec = self._localization_handoff_timeout_sec
        stage_key = str(stage or "").strip().lower()
        readiness_only = stage_key == "before-pre-align"
        soft_stability = stage_key == "after-pre-align"
        if soft_stability:
            fresh, reason = self._handoff_sensors_are_fresh()
            state = self._lookup_map_base_state(timeout_sec=0.05) if fresh else None
            if state is not None:
                position = state.get("position")
                yaw = state.get("yaw")
                detail = (
                    "advisory ready; dispatching immediately: "
                    "x={x:.3f}, y={y:.3f}, yaw={yaw:.2f}deg {fresh}".format(
                        x=float(position[0])
                        if isinstance(position, (list, tuple)) and position
                        else float("nan"),
                        y=float(position[1])
                        if isinstance(position, (list, tuple)) and len(position) > 1
                        else float("nan"),
                        yaw=math.degrees(float(yaw)) if yaw is not None else float("nan"),
                        fresh=reason,
                    )
                )
                self._set_localization_handoff_reason(f"{stage}: {detail}")
                self._node.get_logger().info(
                    "Localization handoff advisory passed ({stage}); {detail}".format(
                        stage=stage,
                        detail=detail,
                    )
                )
            else:
                detail = reason if fresh else "sensors not fresh"
                self._set_localization_handoff_reason(
                    f"{stage}: advisory unavailable; dispatching immediately: {detail}"
                )
                self._node.get_logger().warning(
                    "Localization handoff advisory unavailable ({stage}); dispatching Nav2 immediately: "
                    "{reason}".format(
                        stage=stage,
                        reason=detail,
                    )
                )
            return True
        settle_sec = self._localization_handoff_settle_sec
        max_translation = self._localization_handoff_max_translation_m
        max_yaw = self._localization_handoff_max_yaw_rad
        started = time.monotonic()
        samples: list[tuple[float, float, float, float]] = []
        last_reason = "waiting for localization samples"
        self._node.get_logger().info(
            "Localization handoff gate started ({stage}): "
            "settle={settle:.2f}s timeout={timeout:.2f}s max_translation={trans:.3f}m "
            "max_yaw={yaw:.1f}deg".format(
                stage=stage,
                settle=settle_sec,
                timeout=timeout_sec,
                trans=max_translation,
                yaw=math.degrees(max_yaw),
            )
        )

        while self._is_active_goal_sequence(goal_sequence):
            now = time.monotonic()
            if now - started > timeout_sec:
                self._publish_stop()
                self._set_localization_handoff_reason(f"{stage}: {last_reason}")
                self._node.get_logger().warning(
                    f"Localization handoff gate failed ({stage}): {last_reason}"
                )
                return False

            fresh, reason = self._handoff_sensors_are_fresh()
            if not fresh:
                last_reason = reason
                samples.clear()
                time.sleep(0.1)
                continue

            state = self._lookup_map_base_state(timeout_sec=0.1)
            if state is None:
                last_reason = "map->base_link transform unavailable"
                samples.clear()
                time.sleep(0.1)
                continue

            position = state.get("position")
            yaw = state.get("yaw")
            if not isinstance(position, (list, tuple)) or len(position) < 2 or yaw is None:
                last_reason = "map->base_link transform is incomplete"
                samples.clear()
                time.sleep(0.1)
                continue

            if readiness_only:
                detail = (
                    "ready: x={x:.3f}, y={y:.3f}, yaw={yaw:.2f}deg {fresh}".format(
                        x=float(position[0]),
                        y=float(position[1]),
                        yaw=math.degrees(float(yaw)),
                        fresh=reason,
                    )
                )
                self._set_localization_handoff_reason(f"{stage}: {detail}")
                self._node.get_logger().info(
                    "Localization handoff readiness passed ({stage}): "
                    "map->base_link available at x={x:.3f}, y={y:.3f}, yaw={yaw:.2f}deg {fresh}".format(
                        stage=stage,
                        x=float(position[0]),
                        y=float(position[1]),
                        yaw=math.degrees(float(yaw)),
                        fresh=reason,
                    )
                )
                return True

            samples.append((now, float(position[0]), float(position[1]), float(yaw)))
            min_time = now - settle_sec
            samples = [sample for sample in samples if sample[0] >= min_time]
            if samples and samples[-1][0] - samples[0][0] >= settle_sec:
                x0, y0, yaw0 = samples[0][1], samples[0][2], samples[0][3]
                max_translation_seen = max(
                    math.hypot(sample[1] - x0, sample[2] - y0)
                    for sample in samples
                )
                max_yaw_seen = max(
                    abs(_shortest_delta_rad(yaw0, sample[3]))
                    for sample in samples
                )
                if max_translation_seen <= max_translation and max_yaw_seen <= max_yaw:
                    detail = (
                        "stable: samples={samples} translation_delta={trans:.3f}m "
                        "yaw_delta={yaw:.2f}deg {fresh}".format(
                            samples=len(samples),
                            trans=max_translation_seen,
                            yaw=math.degrees(max_yaw_seen),
                            fresh=reason,
                        )
                    )
                    self._set_localization_handoff_reason(f"{stage}: {detail}")
                    self._node.get_logger().info(
                        "Localization handoff gate passed ({stage}): samples={samples} "
                        "translation_delta={trans:.3f}m yaw_delta={yaw:.2f}deg {fresh}".format(
                            stage=stage,
                            samples=len(samples),
                            trans=max_translation_seen,
                            yaw=math.degrees(max_yaw_seen),
                            fresh=reason,
                        )
                    )
                    return True
                last_reason = (
                    "pose still changing: translation_delta={trans:.3f}m yaw_delta={yaw:.2f}deg".format(
                        trans=max_translation_seen,
                        yaw=math.degrees(max_yaw_seen),
                    )
                )
            time.sleep(0.1)

        self._publish_stop()
        return False

    def check_for_new_messages(self) -> str:
        with self._lock:
            message = self._latest_msg
            self._latest_msg = ""
            return message or ""

    def get_status(self) -> str:
        with self._lock:
            return self._status

    def cancel_navigation(self) -> None:
        with self._lock:
            goal_handle = self._goal_handle
            self._cancel_requested = True
        if goal_handle is not None:
            goal_handle.cancel_goal_async()
        if not self._simulation_mode:
            self._publish_stop()
        self.update_status("cancelled")
        self.update_latest_msg("Navigation cancelled.")

    def _dispatch_simulated_navigation(
        self,
        goal_position: list[float],
        *,
        final_yaw_deg: float,
        goal_sequence: int,
    ) -> None:
        distance = self._distance_to_current_goal(self._sim_position)
        delay_sec = self._simulation_navigation_delay_sec
        if distance is not None:
            delay_sec += distance * self._simulation_navigation_delay_per_meter_sec
        self.update_status("sim_navigating")
        self.update_latest_msg(
            "Simulated navigation dispatched: x={x:.2f}, y={y:.2f}, delay={delay:.1f}s".format(
                x=float(goal_position[0]),
                y=float(goal_position[1]),
                delay=delay_sec,
            )
        )
        self._node.get_logger().info(
            "Simulation navigation goal dispatched: x={x:.3f}, y={y:.3f}, z={z:.3f}, "
            "yaw={yaw:.1f}deg, delay={delay:.1f}s, goal_sequence={seq}".format(
                x=float(goal_position[0]),
                y=float(goal_position[1]),
                z=float(goal_position[2]) if len(goal_position) >= 3 else 0.0,
                yaw=float(final_yaw_deg),
                delay=delay_sec,
                seq=goal_sequence,
            )
        )
        threading.Thread(
            target=self._complete_simulated_navigation_after_delay,
            args=(list(goal_position), float(final_yaw_deg), float(delay_sec), int(goal_sequence)),
            daemon=True,
            name=f"sim-nav-goal-{goal_sequence}",
        ).start()

    def _complete_simulated_navigation_after_delay(
        self,
        goal_position: list[float],
        final_yaw_deg: float,
        delay_sec: float,
        goal_sequence: int,
    ) -> None:
        deadline = time.monotonic() + max(0.0, delay_sec)
        while time.monotonic() < deadline:
            if not self._is_active_goal_sequence(goal_sequence):
                return
            time.sleep(min(0.2, max(0.0, deadline - time.monotonic())))
        if not self._is_active_goal_sequence(goal_sequence):
            return
        z = float(goal_position[2]) if len(goal_position) >= 3 else 0.0
        with self._lock:
            if int(goal_sequence) != int(self._active_goal_sequence) or self._cancel_requested:
                return
            self._sim_position = [float(goal_position[0]), float(goal_position[1]), z]
            self._sim_yaw_deg = float(final_yaw_deg)
        self.update_status("arrived")
        self.update_latest_msg(self._arrival_message(goal_position))
        self._node.get_logger().info(
            f"Simulation navigation arrived; arrival message queued. goal_sequence={goal_sequence}"
        )

    def _on_goal_response(self, future, goal_sequence: int) -> None:
        if not self._is_active_goal_sequence(goal_sequence):
            self._node.get_logger().info(
                f"Ignoring stale Nav2 goal response for goal_sequence={goal_sequence}."
            )
            return
        try:
            goal_handle = future.result()
        except Exception as exc:
            if self._is_active_goal_sequence(goal_sequence):
                self.update_status("failed")
                self.update_latest_msg(f"Nav2 goal request failed: {exc}")
            return

        if not goal_handle.accepted:
            if self._is_active_goal_sequence(goal_sequence):
                self.update_status("failed")
                self.update_latest_msg("Nav2 goal rejected.")
                self._node.get_logger().warning("Nav2 goal rejected by action server.")
            return

        with self._lock:
            if goal_sequence != self._active_goal_sequence:
                self._node.get_logger().info(
                    f"Ignoring stale accepted Nav2 goal for goal_sequence={goal_sequence}."
                )
                return
            self._goal_handle = goal_handle
        self._node.get_logger().info(f"Nav2 goal accepted; waiting for result. goal_sequence={goal_sequence}")
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(lambda future, seq=goal_sequence: self._on_result(future, seq))

    def _on_result(self, future, goal_sequence: int) -> None:
        if not self._is_active_goal_sequence(goal_sequence):
            self._node.get_logger().info(
                f"Ignoring stale Nav2 result for goal_sequence={goal_sequence}."
            )
            return
        try:
            result = future.result()
            status = int(result.status)
        except Exception as exc:
            if self._is_active_goal_sequence(goal_sequence):
                self.update_status("failed")
                self.update_latest_msg(f"Nav2 result failed: {exc}")
                self._node.get_logger().error(f"Nav2 result future failed: {exc}")
            return

        state = self.get_current_state()
        position = state.get("position") if isinstance(state, dict) else None
        distance_to_goal = self._distance_to_current_goal(position)
        debug_suffix = self._result_debug_suffix(position, distance_to_goal)
        self._node.get_logger().info(f"Nav2 result status={status}.{debug_suffix}")

        if status == 4 and self._is_outside_arrival_tolerance(distance_to_goal):
            self.update_status("failed")
            self.update_latest_msg(
                f"Nav2 reported success, but current pose is outside arrival tolerance.{debug_suffix}"
            )
            self._node.get_logger().warning(
                f"Nav2 reported SUCCEEDED outside arrival tolerance; not treating as arrived.{debug_suffix}"
            )
        elif status == 4:
            self._handle_arrived(position, goal_sequence, "Nav2 reported SUCCEEDED")
        elif status == 5:
            self.update_status("cancelled")
            self.update_latest_msg("Navigation goal was cancelled.")
            self._node.get_logger().warning(f"Nav2 goal cancelled.{debug_suffix}")
        elif status == 6 and self._is_within_arrival_tolerance(distance_to_goal):
            self._node.get_logger().warning(
                "Nav2 reported ABORTED, but robot is within arrival tolerance; treating as arrived. "
                f"{debug_suffix}"
            )
            self._handle_arrived(position, goal_sequence, "Nav2 aborted within arrival tolerance")
        else:
            self.update_status("failed")
            self.update_latest_msg(f"Navigation goal finished with status {status}.{debug_suffix}")
            self._node.get_logger().warning(f"Nav2 goal finished without arrival status.{debug_suffix}")

    def _is_active_goal_sequence(self, goal_sequence: int) -> bool:
        with self._lock:
            return (
                int(goal_sequence) == int(self._active_goal_sequence)
                and not self._cancel_requested
            )

    def _should_takeover_rotation(self) -> bool:
        return self._enable_rotation_takeover and self._rotation_policy == "left_only"

    def _bearing_yaw_to_waypoint(self, waypoint: list[float]) -> float | None:
        state = self.get_current_state()
        position = state.get("position") if isinstance(state, dict) else None
        if not isinstance(position, (list, tuple)) or len(position) < 2:
            self._node.get_logger().warning("Cannot pre-align: current position is unavailable.")
            return None
        try:
            dx = float(waypoint[0]) - float(position[0])
            dy = float(waypoint[1]) - float(position[1])
        except (TypeError, ValueError):
            return None
        if math.hypot(dx, dy) < 0.05:
            return None
        return math.atan2(dy, dx)

    def _current_yaw_rad(self) -> float | None:
        state = self.get_current_state()
        orientation = state.get("orientation") if isinstance(state, dict) else None
        if not isinstance(orientation, (list, tuple)) or len(orientation) < 4:
            return None
        try:
            return _quaternion_to_yaw_rad(orientation)
        except (TypeError, ValueError):
            return None

    def _handle_arrived(self, position: Any, goal_sequence: int, reason: str) -> None:
        if not self._is_active_goal_sequence(goal_sequence):
            return
        if self._should_takeover_rotation() and self._final_align_enabled:
            self._publish_stop()
            time.sleep(min(self._settle_time_sec, 0.3))
            with self._lock:
                final_yaw_deg = self._current_goal_final_yaw_deg
            self.update_status("final_aligning")
            aligned = self._rotate_left_only_to_yaw(
                math.radians(final_yaw_deg),
                goal_sequence=goal_sequence,
                stage="final-align",
            )
            if not self._is_active_goal_sequence(goal_sequence):
                self._publish_stop()
                return
            if aligned:
                time.sleep(self._settle_time_sec)
            else:
                self.update_status("arrived_alignment_failed")
                self.update_latest_msg("Arrived at destination, but final yaw alignment failed.")
                self._node.get_logger().warning(
                    f"{reason}; final left-only alignment failed before timeout."
                )
                return

        self.update_status("arrived")
        self.update_latest_msg(self._arrival_message(position))
        self._node.get_logger().info(f"{reason}; arrival message queued.")

    def _rotate_left_only_to_yaw(self, target_yaw_rad: float, *, goal_sequence: int, stage: str) -> bool:
        start_time = time.monotonic()
        last_error = None
        self._node.get_logger().info(
            f"Starting {stage} left-only rotation to yaw={math.degrees(target_yaw_rad):.1f}deg."
        )
        try:
            while self._is_active_goal_sequence(goal_sequence):
                current_yaw = self._current_yaw_rad()
                if current_yaw is None:
                    self._node.get_logger().warning(f"{stage}: current yaw is unavailable.")
                    return False

                shortest_delta = _shortest_delta_rad(current_yaw, target_yaw_rad)
                shortest_error = abs(shortest_delta)
                if shortest_error <= self._yaw_tolerance_rad:
                    self._publish_stop()
                    self._node.get_logger().info(
                        f"{stage} aligned with yaw_error={math.degrees(shortest_error):.2f}deg."
                    )
                    return True

                # left-only: if we are within 10 deg but the CCW path wraps >180 deg,
                # accept the small error instead of rotating nearly a full circle.
                ccw_remaining = _ccw_delta_rad(current_yaw, target_yaw_rad)
                if shortest_error <= math.radians(10.0) and ccw_remaining > math.pi:
                    self._publish_stop()
                    self._node.get_logger().info(
                        f"{stage} accepted near-target yaw (error={math.degrees(shortest_error):.1f}deg, "
                        f"ccw={math.degrees(ccw_remaining):.1f}deg > 180deg)."
                    )
                    return True

                if time.monotonic() - start_time > self._rotation_timeout_sec:
                    self._publish_stop()
                    if last_error is None:
                        last_error = shortest_error
                    self._node.get_logger().warning(
                        f"{stage} timed out with yaw_error={math.degrees(last_error):.2f}deg."
                    )
                    return False

                omega = self._omega_for_rotation_delta(
                    current_yaw,
                    target_yaw_rad,
                )
                self._publish_rotation(omega)
                last_error = shortest_error
                time.sleep(self._rotation_loop_period_sec)
        finally:
            self._publish_stop()
        return False

    def _omega_for_left_only_delta(self, ccw_delta: float) -> float:
        if ccw_delta > math.radians(45.0):
            return self._fast_omega
        if ccw_delta > math.radians(15.0):
            return self._mid_omega
        return self._slow_omega

    def _omega_for_rotation_delta(self, current_yaw: float, target_yaw: float) -> float:
        shortest_delta = _shortest_delta_rad(current_yaw, target_yaw)
        if (
            shortest_delta < 0.0
            and abs(shortest_delta) < self._right_turn_shortcut_rad
        ):
            return -self._omega_for_left_only_delta(abs(shortest_delta))
        ccw_remaining = _ccw_delta_rad(current_yaw, target_yaw)
        return self._omega_for_left_only_delta(ccw_remaining)

    def _publish_rotation(self, angular_z: float) -> None:
        msg = Twist()
        msg.linear.x = 0.0
        msg.angular.z = float(angular_z)
        self._cmd_vel_pub.publish(msg)

    def _publish_stop(self) -> None:
        self._cmd_vel_pub.publish(Twist())

    def _distance_to_current_goal(self, position: Any) -> float | None:
        if not isinstance(position, (list, tuple)) or len(position) < 2:
            return None
        with self._lock:
            goal = list(self._current_goal_position) if self._current_goal_position is not None else None
        if goal is None or len(goal) < 2:
            return None
        try:
            dx = float(position[0]) - float(goal[0])
            dy = float(position[1]) - float(goal[1])
        except (TypeError, ValueError):
            return None
        return math.hypot(dx, dy)

    def _is_within_arrival_tolerance(self, distance_to_goal: float | None) -> bool:
        return distance_to_goal is not None and distance_to_goal <= self._arrival_tolerance_m

    def _is_outside_arrival_tolerance(self, distance_to_goal: float | None) -> bool:
        return distance_to_goal is not None and distance_to_goal > self._arrival_tolerance_m

    def _result_debug_suffix(self, position: Any, distance_to_goal: float | None) -> str:
        with self._lock:
            goal = list(self._current_goal_position) if self._current_goal_position is not None else None
        position_text = "current_position=unavailable"
        if isinstance(position, (list, tuple)) and len(position) >= 2:
            z = float(position[2]) if len(position) >= 3 else 0.0
            position_text = "current_position=[{x:.3f}, {y:.3f}, {z:.3f}]".format(
                x=float(position[0]),
                y=float(position[1]),
                z=z,
            )
        goal_text = "goal=unavailable"
        if isinstance(goal, list) and len(goal) >= 2:
            z = float(goal[2]) if len(goal) >= 3 else 0.0
            goal_text = "goal=[{x:.3f}, {y:.3f}, {z:.3f}]".format(
                x=float(goal[0]),
                y=float(goal[1]),
                z=z,
            )
        distance_text = "distance_to_goal=unavailable"
        if distance_to_goal is not None:
            distance_text = f"distance_to_goal={distance_to_goal:.3f}m"
        return (
            f" {position_text}, {goal_text}, {distance_text}, "
            f"arrival_tolerance={self._arrival_tolerance_m:.3f}m"
        )

    def _arrival_message(self, position: Any = None) -> str:
        coords = None
        if isinstance(position, (list, tuple)) and len(position) >= 2:
            z = float(position[2]) if len(position) >= 3 else 0.0
            coords = [float(position[0]), float(position[1]), z]
        if coords is None:
            return "Arrived at destination."
        return f"Arrived at destination [{coords[0]:.3f}, {coords[1]:.3f}, {coords[2]:.3f}]"
