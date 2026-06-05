"""Nav2 controller adapter for the async-agent controller interface."""

from __future__ import annotations

import math
import threading
from typing import Any

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import Odometry
from PIL import Image as PILImage
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import Image as ROSImage
from tf2_ros import Buffer, TransformException, TransformListener

from caragent_agent.controller.controller_base import Base_Controller


def _yaw_deg_to_quaternion(yaw_deg: float) -> list[float]:
    yaw = math.radians(float(yaw_deg))
    half = yaw * 0.5
    return [0.0, 0.0, math.sin(half), math.cos(half)]


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
        odom_topic: str = "/odom",
        arrival_tolerance_m: float = 0.50,
    ) -> None:
        self._node = node
        self._action_name = action_name
        self._global_frame = global_frame
        self._base_frame = base_frame
        self._dry_run = bool(dry_run)
        self._arrival_tolerance_m = max(0.0, float(arrival_tolerance_m))
        self._action_client = ActionClient(node, NavigateToPose, action_name)
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, node)
        self._status = "idle"
        self._latest_msg = ""
        self._goal_handle = None
        self._current_goal_position: list[float] | None = None
        self._lock = threading.RLock()
        self._latest_image: PILImage.Image | None = None
        self._latest_odom: Odometry | None = None
        self._image_sub = node.create_subscription(
            ROSImage, camera_topic, self._on_camera_image, 10
        )
        self._odom_sub = node.create_subscription(
            Odometry, odom_topic, self._on_odom, 10
        )

    def update_path(self, new_path: list[list[float]]) -> None:
        if not new_path:
            raise ValueError("Nav2Controller.update_path received an empty path.")

        waypoint = list(new_path[-1])
        if len(waypoint) < 2:
            raise ValueError(f"Waypoint must contain at least x and y: {waypoint}")

        yaw_quat = _yaw_deg_to_quaternion(waypoint[3] if len(waypoint) >= 4 else 0.0)
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
            self._current_goal_position = goal_position

        if self._dry_run:
            self.update_status("dry_run_dispatched")
            self.update_latest_msg(self._arrival_message(goal_position))
            return

        if not self._action_client.wait_for_server(timeout_sec=5.0):
            self.update_status("failed")
            raise RuntimeError(f"Nav2 action server '{self._action_name}' is unavailable.")

        goal = NavigateToPose.Goal()
        goal.pose = pose
        self.update_status("navigating")
        self.update_latest_msg(
            f"Dispatched Nav2 goal x={pose.pose.position.x:.2f}, y={pose.pose.position.y:.2f}"
        )
        self._node.get_logger().info(
            "Nav2 goal dispatched: x={x:.3f}, y={y:.3f}, z={z:.3f}, arrival_tolerance={tol:.3f}m".format(
                x=goal_position[0],
                y=goal_position[1],
                z=goal_position[2],
                tol=self._arrival_tolerance_m,
            )
        )
        send_future = self._action_client.send_goal_async(goal)
        send_future.add_done_callback(self._on_goal_response)

    def update_status(self, status: str) -> None:
        with self._lock:
            self._status = str(status or "idle")

    def update_latest_msg(self, msg: str) -> None:
        with self._lock:
            self._latest_msg = str(msg or "")

    def get_current_state(self) -> dict[str, Any]:
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
        return {
            "position": [translation.x, translation.y, translation.z],
            "orientation": [rotation.x, rotation.y, rotation.z, rotation.w],
            "status": self.get_status(),
            "source": "tf",
        }

    def get_current_image(self) -> Any:
        with self._lock:
            return self._latest_image

    def _on_camera_image(self, msg: ROSImage) -> None:
        try:
            data = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
            if msg.encoding == "bgr8":
                data = data[:, :, ::-1]  # BGR → RGB
            image = PILImage.fromarray(data)
            with self._lock:
                self._latest_image = image
        except Exception:
            pass  # silently skip corrupt frames

    def _on_odom(self, msg: Odometry) -> None:
        with self._lock:
            self._latest_odom = msg

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

    def check_for_new_messages(self) -> str:
        with self._lock:
            message = self._latest_msg
            self._latest_msg = ""
            return message

    def get_status(self) -> str:
        with self._lock:
            return self._status

    def cancel_navigation(self) -> None:
        with self._lock:
            goal_handle = self._goal_handle
        if goal_handle is not None:
            goal_handle.cancel_goal_async()
        self.update_status("cancelled")
        self.update_latest_msg("Navigation cancelled.")

    def _on_goal_response(self, future) -> None:
        try:
            goal_handle = future.result()
        except Exception as exc:
            self.update_status("failed")
            self.update_latest_msg(f"Nav2 goal request failed: {exc}")
            return

        if not goal_handle.accepted:
            self.update_status("failed")
            self.update_latest_msg("Nav2 goal rejected.")
            self._node.get_logger().warning("Nav2 goal rejected by action server.")
            return

        with self._lock:
            self._goal_handle = goal_handle
        self._node.get_logger().info("Nav2 goal accepted; waiting for result.")
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._on_result)

    def _on_result(self, future) -> None:
        try:
            result = future.result()
            status = int(result.status)
        except Exception as exc:
            self.update_status("failed")
            self.update_latest_msg(f"Nav2 result failed: {exc}")
            self._node.get_logger().error(f"Nav2 result future failed: {exc}")
            return

        state = self.get_current_state()
        position = state.get("position") if isinstance(state, dict) else None
        distance_to_goal = self._distance_to_current_goal(position)
        debug_suffix = self._result_debug_suffix(position, distance_to_goal)
        self._node.get_logger().info(f"Nav2 result status={status}.{debug_suffix}")

        if status == 4:
            self.update_status("arrived")
            self.update_latest_msg(self._arrival_message(position))
            self._node.get_logger().info("Nav2 reported SUCCEEDED; arrival message queued.")
        elif status == 5:
            self.update_status("cancelled")
            self.update_latest_msg("Navigation goal was cancelled.")
            self._node.get_logger().warning(f"Nav2 goal cancelled.{debug_suffix}")
        elif status == 6 and self._is_within_arrival_tolerance(distance_to_goal):
            self.update_status("arrived")
            msg = self._arrival_message(position)
            self.update_latest_msg(msg)
            self._node.get_logger().warning(
                "Nav2 reported ABORTED, but robot is within arrival tolerance; treating as arrived. "
                f"{debug_suffix}"
            )
        else:
            self.update_status("failed")
            self.update_latest_msg(f"Navigation goal finished with status {status}.{debug_suffix}")
            self._node.get_logger().warning(f"Nav2 goal finished without arrival status.{debug_suffix}")

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
