"""Navigation-side left-only rotation proxy for Nav2 goal testing."""

from __future__ import annotations

import math
import threading
import time
from typing import Any

import rclpy
from builtin_interfaces.msg import Duration as DurationMsg
from geometry_msgs.msg import PoseStamped, Twist
from nav2_msgs.action import ComputePathToPose, NavigateToPose, Spin
from nav_msgs.msg import Odometry
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import LaserScan
from tf2_ros import Buffer, TransformException, TransformListener


def quaternion_to_yaw_rad(quat: Any) -> float:
    x, y, z, w = (float(quat.x), float(quat.y), float(quat.z), float(quat.w))
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def yaw_rad_to_quaternion(yaw: float) -> tuple[float, float, float, float]:
    half = yaw * 0.5
    return 0.0, 0.0, math.sin(half), math.cos(half)


def normalize_angle_rad(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def ccw_delta_rad(current_yaw: float, target_yaw: float) -> float:
    return (float(target_yaw) - float(current_yaw)) % (2.0 * math.pi)


def shortest_delta_rad(current_yaw: float, target_yaw: float) -> float:
    return normalize_angle_rad(float(target_yaw) - float(current_yaw))


class LeftOnlyGoalProxy(Node):
    """Accept PoseStamped test goals, rotate left-only, then dispatch to Nav2."""

    def __init__(self) -> None:
        super().__init__("left_only_goal_proxy")
        self.declare_parameter("input_goal_topic", "/caragent/left_only_goal")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("action_name", "navigate_to_pose")
        self.declare_parameter("spin_action_name", "spin")
        self.declare_parameter("global_frame", "map")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("pre_align_enabled", True)
        self.declare_parameter("pre_align_strategy", "direct_bearing")
        self.declare_parameter("path_heading_action_name", "compute_path_to_pose")
        self.declare_parameter("path_heading_lookahead_m", 0.70)
        self.declare_parameter("path_heading_min_goal_distance_m", 0.35)
        self.declare_parameter("path_heading_timeout_sec", 3.0)
        self.declare_parameter("path_heading_fallback_to_direct", True)
        self.declare_parameter("final_align_enabled", True)
        self.declare_parameter("arrival_tolerance_m", 0.25)
        self.declare_parameter("yaw_tolerance_deg", 4.0)
        self.declare_parameter("settle_time_sec", 0.7)
        self.declare_parameter("fast_omega", 3.40)
        self.declare_parameter("mid_omega", 2.50)
        self.declare_parameter("slow_omega", 1.50)
        self.declare_parameter("fast_threshold_deg", 20.0)
        self.declare_parameter("mid_threshold_deg", 10.0)
        self.declare_parameter("rotation_timeout_sec", 15.0)
        self.declare_parameter("rotation_loop_rate_hz", 20.0)
        self.declare_parameter("right_turn_shortcut_deg", 90.0)
        self.declare_parameter("safety_check_enabled", True)
        self.declare_parameter("safety_radius_m", 0.38)
        self.declare_parameter("safety_front_radius_m", 0.45)
        self.declare_parameter("safety_side_radius_m", 0.34)
        self.declare_parameter("safety_rear_radius_m", 0.30)
        self.declare_parameter("safety_scan_max_age_sec", 0.5)
        self.declare_parameter("nav2_final_align_fallback", True)
        self.declare_parameter("nav2_final_align_timeout_sec", 20.0)

        self._input_goal_topic = self.get_parameter("input_goal_topic").value
        self._cmd_vel_topic = self.get_parameter("cmd_vel_topic").value
        self._action_name = self.get_parameter("action_name").value
        self._spin_action_name = self.get_parameter("spin_action_name").value
        self._global_frame = self.get_parameter("global_frame").value
        self._base_frame = self.get_parameter("base_frame").value
        self._pre_align_enabled = bool(self.get_parameter("pre_align_enabled").value)
        self._pre_align_strategy = (
            str(self.get_parameter("pre_align_strategy").value or "direct_bearing")
            .strip()
            .lower()
        )
        self._path_heading_action_name = self.get_parameter("path_heading_action_name").value
        self._path_heading_lookahead_m = max(
            0.05, float(self.get_parameter("path_heading_lookahead_m").value)
        )
        self._path_heading_min_goal_distance_m = max(
            0.0, float(self.get_parameter("path_heading_min_goal_distance_m").value)
        )
        self._path_heading_timeout_sec = max(
            0.2, float(self.get_parameter("path_heading_timeout_sec").value)
        )
        self._path_heading_fallback_to_direct = bool(
            self.get_parameter("path_heading_fallback_to_direct").value
        )
        self._final_align_enabled = bool(self.get_parameter("final_align_enabled").value)
        self._arrival_tolerance_m = max(
            0.0, float(self.get_parameter("arrival_tolerance_m").value)
        )
        self._yaw_tolerance_rad = math.radians(
            max(0.1, float(self.get_parameter("yaw_tolerance_deg").value))
        )
        self._settle_time_sec = max(0.0, float(self.get_parameter("settle_time_sec").value))
        self._fast_omega = max(0.0, float(self.get_parameter("fast_omega").value))
        self._mid_omega = max(0.0, float(self.get_parameter("mid_omega").value))
        self._slow_omega = max(0.0, float(self.get_parameter("slow_omega").value))
        self._fast_threshold_rad = math.radians(
            max(0.0, float(self.get_parameter("fast_threshold_deg").value))
        )
        self._mid_threshold_rad = math.radians(
            max(0.0, float(self.get_parameter("mid_threshold_deg").value))
        )
        self._rotation_timeout_sec = max(
            0.5, float(self.get_parameter("rotation_timeout_sec").value)
        )
        loop_rate = max(1.0, float(self.get_parameter("rotation_loop_rate_hz").value))
        self._rotation_loop_period_sec = 1.0 / loop_rate
        self._right_turn_shortcut_rad = math.radians(
            max(0.0, float(self.get_parameter("right_turn_shortcut_deg").value))
        )
        self._safety_check_enabled = bool(self.get_parameter("safety_check_enabled").value)
        self._safety_radius_m = max(0.0, float(self.get_parameter("safety_radius_m").value))
        self._safety_front_radius_m = max(
            self._safety_radius_m, float(self.get_parameter("safety_front_radius_m").value)
        )
        self._safety_side_radius_m = max(
            self._safety_radius_m, float(self.get_parameter("safety_side_radius_m").value)
        )
        self._safety_rear_radius_m = max(
            self._safety_radius_m, float(self.get_parameter("safety_rear_radius_m").value)
        )
        self._safety_scan_max_age_sec = max(
            0.1, float(self.get_parameter("safety_scan_max_age_sec").value)
        )
        self._nav2_final_align_fallback = bool(
            self.get_parameter("nav2_final_align_fallback").value
        )
        self._nav2_final_align_timeout_sec = max(
            1.0, float(self.get_parameter("nav2_final_align_timeout_sec").value)
        )

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._action_client = ActionClient(self, NavigateToPose, self._action_name)
        self._spin_client = ActionClient(self, Spin, self._spin_action_name)
        self._path_client = ActionClient(
            self, ComputePathToPose, self._path_heading_action_name
        )
        self._cmd_vel_pub = self.create_publisher(Twist, self._cmd_vel_topic, 10)
        self._goal_sub = self.create_subscription(
            PoseStamped,
            self._input_goal_topic,
            self._on_goal,
            10,
        )
        self._odom_sub = self.create_subscription(
            Odometry,
            self.get_parameter("odom_topic").value,
            self._on_odom,
            10,
        )
        self._scan_sub = self.create_subscription(
            LaserScan,
            self.get_parameter("scan_topic").value,
            self._on_scan,
            10,
        )

        self._latest_odom: Odometry | None = None
        self._latest_scan: LaserScan | None = None
        self._latest_scan_monotonic = 0.0
        self._lock = threading.RLock()
        self._goal_sequence = 0
        self._active_goal_sequence = 0

        self.get_logger().info(
            "Left-only goal proxy ready: send PoseStamped to "
            f"{self._input_goal_topic}; dispatches Nav2 action {self._action_name}."
        )

    def _on_odom(self, msg: Odometry) -> None:
        with self._lock:
            self._latest_odom = msg

    def _on_scan(self, msg: LaserScan) -> None:
        with self._lock:
            self._latest_scan = msg
            self._latest_scan_monotonic = time.monotonic()

    def _on_goal(self, msg: PoseStamped) -> None:
        with self._lock:
            self._goal_sequence += 1
            goal_sequence = self._goal_sequence
            self._active_goal_sequence = goal_sequence
        threading.Thread(target=self._run_goal, args=(msg, goal_sequence), daemon=True).start()

    def _run_goal(self, msg: PoseStamped, goal_sequence: int) -> None:
        goal_pose = PoseStamped()
        goal_pose.header = msg.header
        goal_pose.pose = msg.pose
        if not goal_pose.header.frame_id:
            goal_pose.header.frame_id = self._global_frame
        if goal_pose.header.frame_id != self._global_frame:
            self.get_logger().warning(
                f"Goal frame is {goal_pose.header.frame_id}; expected {self._global_frame}."
            )

        final_yaw = quaternion_to_yaw_rad(goal_pose.pose.orientation)
        pre_align_yaw = self._pre_align_yaw_to_goal(goal_pose)
        if self._pre_align_enabled and pre_align_yaw is not None:
            if not self._rotation_is_safe("pre-align"):
                self.get_logger().warning(
                    "Pre-align skipped because the rotation safety check is blocked."
                )
            else:
                aligned = self._rotate_left_only_to_yaw(
                    pre_align_yaw,
                    goal_sequence=goal_sequence,
                    stage="pre-align",
                )
                if not self._is_active_goal(goal_sequence):
                    return
                if aligned:
                    time.sleep(self._settle_time_sec)
                else:
                    self.get_logger().warning(
                        "Pre-align timed out or was blocked; dispatching Nav2 goal anyway."
                    )

        nav_goal_yaw = self._bearing_yaw_to_goal(goal_pose)
        if nav_goal_yaw is not None:
            x, y, z, w = yaw_rad_to_quaternion(nav_goal_yaw)
            goal_pose.pose.orientation.x = x
            goal_pose.pose.orientation.y = y
            goal_pose.pose.orientation.z = z
            goal_pose.pose.orientation.w = w

        if not self._action_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error(f"Nav2 action server {self._action_name} is unavailable.")
            return

        action_goal = NavigateToPose.Goal()
        action_goal.pose = goal_pose
        self.get_logger().info(
            "Dispatching Nav2 goal after left-only pre-align: "
            f"x={goal_pose.pose.position.x:.3f}, y={goal_pose.pose.position.y:.3f}"
        )
        send_future = self._action_client.send_goal_async(action_goal)
        send_future.add_done_callback(
            lambda future, seq=goal_sequence, yaw=final_yaw, pose=goal_pose: self._on_goal_response(
                future,
                seq,
                yaw,
                pose,
            )
        )

    def _on_goal_response(
        self,
        future,
        goal_sequence: int,
        final_yaw: float,
        goal_pose: PoseStamped,
    ) -> None:
        if not self._is_active_goal(goal_sequence):
            return
        try:
            goal_handle = future.result()
        except Exception as exc:
            self.get_logger().error(f"Nav2 goal request failed: {exc}")
            return
        if not goal_handle.accepted:
            self.get_logger().warning("Nav2 goal rejected.")
            return
        self.get_logger().info("Nav2 goal accepted; waiting for result.")
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(
            lambda future, seq=goal_sequence, yaw=final_yaw, pose=goal_pose: self._on_result(
                future,
                seq,
                yaw,
                pose,
            )
        )

    def _on_result(
        self,
        future,
        goal_sequence: int,
        final_yaw: float,
        goal_pose: PoseStamped,
    ) -> None:
        if not self._is_active_goal(goal_sequence):
            return
        try:
            result = future.result()
            status = int(result.status)
        except Exception as exc:
            self.get_logger().error(f"Nav2 result failed: {exc}")
            return
        distance_to_goal = self._distance_to_goal(goal_pose)
        if distance_to_goal is None:
            distance_text = "distance_to_goal=unavailable"
        else:
            distance_text = f"distance_to_goal={distance_to_goal:.3f}m"
        self.get_logger().info(f"Nav2 result status={status}; {distance_text}.")
        if status != 4 and not self._is_within_arrival_tolerance(distance_to_goal):
            self.get_logger().warning(
                "Nav2 did not report arrival and robot is not within arrival tolerance; "
                "skipping final align."
            )
            return
        if status != 4:
            self.get_logger().warning(
                "Nav2 did not report SUCCEEDED, but robot is within arrival tolerance; "
                "running final align anyway."
            )
        if self._final_align_enabled:
            self._publish_stop()
            time.sleep(min(self._settle_time_sec, 0.3))
            if not self._rotation_is_safe("final-align"):
                self.get_logger().warning(
                    "Final left-only align is blocked; handing final yaw to Nav2 Spin."
                )
                self._dispatch_nav2_final_spin(final_yaw, goal_sequence)
                return
            aligned = self._rotate_left_only_to_yaw(
                final_yaw,
                goal_sequence=goal_sequence,
                stage="final-align",
            )
            if aligned:
                time.sleep(self._settle_time_sec)
                self.get_logger().info("Final left-only alignment complete.")
            else:
                self.get_logger().warning(
                    "Final left-only alignment did not converge; handing final yaw to Nav2 Spin."
                )
                self._dispatch_nav2_final_spin(final_yaw, goal_sequence)

    def _is_active_goal(self, goal_sequence: int) -> bool:
        with self._lock:
            return int(goal_sequence) == int(self._active_goal_sequence)

    def _bearing_yaw_to_goal(self, goal_pose: PoseStamped) -> float | None:
        state = self._current_state()
        position = state.get("position") if isinstance(state, dict) else None
        if not isinstance(position, (list, tuple)) or len(position) < 2:
            self.get_logger().warning("Cannot pre-align: current position is unavailable.")
            return None
        dx = float(goal_pose.pose.position.x) - float(position[0])
        dy = float(goal_pose.pose.position.y) - float(position[1])
        if math.hypot(dx, dy) < 0.05:
            return None
        return math.atan2(dy, dx)

    def _pre_align_yaw_to_goal(self, goal_pose: PoseStamped) -> float | None:
        direct_yaw = self._bearing_yaw_to_goal(goal_pose)
        if self._pre_align_strategy in {"", "direct", "direct_bearing", "bearing"}:
            return direct_yaw
        if self._pre_align_strategy not in {"path", "path_heading"}:
            self.get_logger().warning(
                f"Unknown pre_align_strategy={self._pre_align_strategy!r}; using direct_bearing."
            )
            return direct_yaw

        path_yaw = self._path_heading_yaw_to_goal(goal_pose)
        if path_yaw is not None:
            return path_yaw
        if self._path_heading_fallback_to_direct:
            self.get_logger().warning(
                "Path-heading pre-align unavailable; falling back to direct bearing."
            )
            return direct_yaw
        self.get_logger().warning("Path-heading pre-align unavailable; skipping pre-align.")
        return None

    def _path_heading_yaw_to_goal(self, goal_pose: PoseStamped) -> float | None:
        distance_to_goal = self._distance_to_goal(goal_pose)
        if (
            distance_to_goal is not None
            and distance_to_goal < self._path_heading_min_goal_distance_m
        ):
            self.get_logger().info(
                "Skipping path-heading pre-align because goal is already close: "
                f"distance={distance_to_goal:.2f}m."
            )
            return None
        if not self._path_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().warning(
                f"Path-heading pre-align unavailable: action server "
                f"{self._path_heading_action_name!r} is not ready."
            )
            return None

        goal = ComputePathToPose.Goal()
        goal.goal = goal_pose
        goal.planner_id = ""
        goal.use_start = False
        future = self._path_client.send_goal_async(goal)
        if not self._spin_until_future_complete(future, self._path_heading_timeout_sec):
            self.get_logger().warning("Path-heading pre-align timed out sending plan request.")
            return None
        try:
            goal_handle = future.result()
        except Exception as exc:
            self.get_logger().warning(f"Path-heading plan request failed: {exc}")
            return None
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().warning("Path-heading plan request was rejected.")
            return None

        result_future = goal_handle.get_result_async()
        if not self._spin_until_future_complete(
            result_future, self._path_heading_timeout_sec
        ):
            self.get_logger().warning("Path-heading plan result timed out.")
            return None
        try:
            result = result_future.result()
        except Exception as exc:
            self.get_logger().warning(f"Path-heading plan result failed: {exc}")
            return None
        if result is None or int(result.status) != 4:
            status = "unknown" if result is None else int(result.status)
            self.get_logger().warning(
                f"Path-heading plan did not succeed; status={status}."
            )
            return None

        path = result.result.path
        yaw = self._yaw_from_path_lookahead(path.poses)
        if yaw is None:
            self.get_logger().warning("Path-heading plan has no usable lookahead segment.")
            return None
        self.get_logger().info(
            "Path-heading pre-align selected yaw="
            f"{math.degrees(yaw):.1f}deg from {len(path.poses)} path poses."
        )
        return yaw

    def _yaw_from_path_lookahead(self, poses: list[PoseStamped]) -> float | None:
        state = self._current_state()
        position = state.get("position") if isinstance(state, dict) else None
        if not isinstance(position, (list, tuple)) or len(position) < 2:
            return None
        current_x = float(position[0])
        current_y = float(position[1])

        best_pose = None
        best_distance = float("inf")
        for pose in poses:
            dx = float(pose.pose.position.x) - current_x
            dy = float(pose.pose.position.y) - current_y
            distance = math.hypot(dx, dy)
            if distance < best_distance:
                best_distance = distance
                best_pose = pose
            if distance >= self._path_heading_lookahead_m:
                best_pose = pose
                best_distance = distance
                break

        if best_pose is None or best_distance < 0.05:
            return None
        dx = float(best_pose.pose.position.x) - current_x
        dy = float(best_pose.pose.position.y) - current_y
        if math.hypot(dx, dy) < 0.05:
            return None
        return math.atan2(dy, dx)

    def _spin_until_future_complete(self, future, timeout_sec: float) -> bool:
        deadline = time.monotonic() + max(0.0, float(timeout_sec))
        while rclpy.ok() and not future.done() and time.monotonic() < deadline:
            time.sleep(0.02)
        return future.done()

    def _current_state(self) -> dict[str, Any]:
        try:
            transform = self._tf_buffer.lookup_transform(
                self._global_frame,
                self._base_frame,
                Time(),
                timeout=Duration(seconds=0.2),
            )
            translation = transform.transform.translation
            rotation = transform.transform.rotation
            return {
                "position": [translation.x, translation.y, translation.z],
                "yaw": quaternion_to_yaw_rad(rotation),
                "source": "tf",
            }
        except TransformException:
            with self._lock:
                odom = self._latest_odom
            if odom is None:
                return {"position": None, "yaw": None, "source": "unavailable"}
            position = odom.pose.pose.position
            return {
                "position": [position.x, position.y, position.z],
                "yaw": quaternion_to_yaw_rad(odom.pose.pose.orientation),
                "source": "odom",
            }

    def _current_yaw_rad(self) -> float | None:
        yaw = self._current_state().get("yaw")
        return float(yaw) if yaw is not None else None

    def _distance_to_goal(self, goal_pose: PoseStamped) -> float | None:
        state = self._current_state()
        position = state.get("position") if isinstance(state, dict) else None
        if not isinstance(position, (list, tuple)) or len(position) < 2:
            return None
        dx = float(position[0]) - float(goal_pose.pose.position.x)
        dy = float(position[1]) - float(goal_pose.pose.position.y)
        return math.hypot(dx, dy)

    def _is_within_arrival_tolerance(self, distance_to_goal: float | None) -> bool:
        return distance_to_goal is not None and distance_to_goal <= self._arrival_tolerance_m

    def _rotate_left_only_to_yaw(self, target_yaw: float, *, goal_sequence: int, stage: str) -> bool:
        start_time = time.monotonic()
        self.get_logger().info(
            f"Starting {stage} left-only rotation to yaw={math.degrees(target_yaw):.1f}deg."
        )
        try:
            while self._is_active_goal(goal_sequence):
                if not self._rotation_is_safe(stage):
                    self._publish_stop()
                    self.get_logger().warning(f"{stage} stopped by rotation safety check.")
                    return False
                current_yaw = self._current_yaw_rad()
                if current_yaw is None:
                    self.get_logger().warning(f"{stage}: current yaw is unavailable.")
                    return False
                shortest_delta = shortest_delta_rad(current_yaw, target_yaw)
                shortest_error = abs(shortest_delta)
                if shortest_error <= self._yaw_tolerance_rad:
                    self._publish_stop()
                    self.get_logger().info(
                        f"{stage} aligned with yaw_error={math.degrees(shortest_error):.2f}deg."
                    )
                    return True
                if time.monotonic() - start_time > self._rotation_timeout_sec:
                    self._publish_stop()
                    self.get_logger().warning(
                        f"{stage} timed out with yaw_error={math.degrees(shortest_error):.2f}deg."
                    )
                    return False
                self._publish_rotation(self._omega_for_rotation_delta(current_yaw, target_yaw))
                time.sleep(self._rotation_loop_period_sec)
        finally:
            self._publish_stop()
        return False

    def _omega_for_delta(self, ccw_delta: float) -> float:
        if ccw_delta > self._fast_threshold_rad:
            return self._fast_omega
        if ccw_delta > self._mid_threshold_rad:
            return self._mid_omega
        return self._slow_omega

    def _omega_for_rotation_delta(self, current_yaw: float, target_yaw: float) -> float:
        shortest_delta = shortest_delta_rad(current_yaw, target_yaw)
        if (
            shortest_delta < 0.0
            and abs(shortest_delta) < self._right_turn_shortcut_rad
        ):
            return -self._omega_for_delta(abs(shortest_delta))
        return self._omega_for_delta(ccw_delta_rad(current_yaw, target_yaw))

    def _publish_rotation(self, angular_z: float) -> None:
        msg = Twist()
        msg.linear.x = 0.0
        msg.angular.z = float(angular_z)
        self._cmd_vel_pub.publish(msg)

    def _publish_stop(self) -> None:
        self._cmd_vel_pub.publish(Twist())

    def _dispatch_nav2_final_spin(self, final_yaw: float, goal_sequence: int) -> None:
        if not self._nav2_final_align_fallback:
            self.get_logger().warning("Nav2 final-align fallback is disabled.")
            return
        if not self._is_active_goal(goal_sequence):
            return
        current_yaw = self._current_yaw_rad()
        if current_yaw is None:
            self.get_logger().warning("Cannot hand final yaw to Nav2 Spin: current yaw unavailable.")
            return
        delta = normalize_angle_rad(final_yaw - current_yaw)
        if abs(delta) <= self._yaw_tolerance_rad:
            self.get_logger().info("Final yaw already within tolerance before Nav2 Spin fallback.")
            return
        if not self._spin_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().error(f"Nav2 Spin action server {self._spin_action_name} is unavailable.")
            return

        goal = Spin.Goal()
        goal.target_yaw = float(delta)
        goal.time_allowance = self._duration_msg(self._nav2_final_align_timeout_sec)
        self.get_logger().warning(
            "Dispatching Nav2 Spin final-align fallback: "
            f"delta_yaw={math.degrees(delta):.1f}deg."
        )
        future = self._spin_client.send_goal_async(goal)
        future.add_done_callback(
            lambda done, seq=goal_sequence: self._on_spin_goal_response(done, seq)
        )

    def _on_spin_goal_response(self, future, goal_sequence: int) -> None:
        if not self._is_active_goal(goal_sequence):
            return
        try:
            goal_handle = future.result()
        except Exception as exc:
            self.get_logger().error(f"Nav2 Spin final-align request failed: {exc}")
            return
        if not goal_handle.accepted:
            self.get_logger().warning("Nav2 Spin final-align fallback was rejected.")
            return
        self.get_logger().info("Nav2 Spin final-align fallback accepted.")
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(
            lambda done, seq=goal_sequence: self._on_spin_result(done, seq)
        )

    def _on_spin_result(self, future, goal_sequence: int) -> None:
        if not self._is_active_goal(goal_sequence):
            return
        try:
            result = future.result()
            status = int(result.status)
        except Exception as exc:
            self.get_logger().error(f"Nav2 Spin final-align result failed: {exc}")
            return
        if status == 4:
            self.get_logger().info("Nav2 Spin final-align fallback SUCCEEDED.")
        else:
            self.get_logger().warning(f"Nav2 Spin final-align fallback finished with status {status}.")

    def _duration_msg(self, seconds: float) -> DurationMsg:
        whole = int(seconds)
        msg = DurationMsg()
        msg.sec = whole
        msg.nanosec = int((float(seconds) - whole) * 1_000_000_000)
        return msg

    def _rotation_is_safe(self, stage: str) -> bool:
        if not self._safety_check_enabled:
            return True
        blocked = self._rotation_safety_block()
        if blocked is None:
            return True
        sector, distance, limit = blocked
        self.get_logger().warning(
            f"{stage} safety blocked: nearest {sector} obstacle {distance:.2f}m "
            f"<= limit {limit:.2f}m."
        )
        return False

    def _rotation_safety_block(self) -> tuple[str, float, float] | None:
        with self._lock:
            scan = self._latest_scan
            scan_age = time.monotonic() - self._latest_scan_monotonic
        if scan is None:
            self.get_logger().warning("Rotation safety blocked: no LaserScan received yet.")
            return ("scan", 0.0, self._safety_radius_m)
        if scan_age > self._safety_scan_max_age_sec:
            self.get_logger().warning(
                f"Rotation safety blocked: LaserScan age {scan_age:.2f}s is stale."
            )
            return ("scan", scan_age, self._safety_scan_max_age_sec)

        nearest: tuple[str, float, float] | None = None
        angle = float(scan.angle_min)
        for value in scan.ranges:
            distance = float(value)
            if not math.isfinite(distance):
                angle += float(scan.angle_increment)
                continue
            min_range = float(scan.range_min) if scan.range_min > 0.0 else 0.02
            max_range = float(scan.range_max) if scan.range_max > 0.0 else float("inf")
            if distance < min_range or distance > max_range:
                angle += float(scan.angle_increment)
                continue

            sector, limit = self._safety_limit_for_angle(angle)
            if distance <= limit and (
                nearest is None or distance < nearest[1]
            ):
                nearest = (sector, distance, limit)
            angle += float(scan.angle_increment)
        return nearest

    def _safety_limit_for_angle(self, angle: float) -> tuple[str, float]:
        forward = abs(normalize_angle_rad(angle))
        rear = abs(normalize_angle_rad(angle - math.pi))
        if forward <= math.radians(45.0):
            return "front", self._safety_front_radius_m
        if rear <= math.radians(45.0):
            return "rear", self._safety_rear_radius_m
        return "side", self._safety_side_radius_m


def main(args=None) -> None:
    rclpy.init(args=args)
    node = LeftOnlyGoalProxy()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
