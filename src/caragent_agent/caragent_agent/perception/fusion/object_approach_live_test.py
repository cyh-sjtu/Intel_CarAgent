"""Live object-level localization and approach-goal test without dispatching Nav2."""

from __future__ import annotations

import argparse
import json
import math
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image as PILImage
import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import Image, LaserScan
from tf2_ros import Buffer, TransformException, TransformListener

from caragent_agent.perception.fusion.live_scan_monodepth_validation import DEFAULT_WORKSPACE
from caragent_agent.perception.fusion.object_approach_pipeline import run_object_approach_from_snapshot


def stamp_to_sec(msg: Any) -> float:
    stamp = getattr(getattr(msg, "header", None), "stamp", None)
    if stamp is None:
        return 0.0
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def image_msg_to_bgr(msg: Image) -> np.ndarray:
    encoding = (msg.encoding or "").lower()
    height = int(msg.height)
    width = int(msg.width)
    step = int(msg.step)
    if encoding in {"bgr8", "rgb8", "rgba8", "bgra8"}:
        channels = 4 if encoding in {"rgba8", "bgra8"} else 3
        raw = np.frombuffer(msg.data, dtype=np.uint8).reshape(height, step)
        arr = raw[:, : width * channels].reshape(height, width, channels).copy()
        if encoding == "bgr8":
            return arr
        if encoding == "rgb8":
            return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        if encoding == "rgba8":
            return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
        return cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
    if encoding in {"mono8", "8uc1"}:
        raw = np.frombuffer(msg.data, dtype=np.uint8).reshape(height, step)
        gray = raw[:, :width].copy()
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    raise ValueError(f"unsupported image encoding: {msg.encoding}")


def bgr_to_pil(image_bgr: np.ndarray) -> PILImage.Image:
    return PILImage.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))


def yaw_to_quat(yaw: float) -> tuple[float, float, float, float]:
    half = 0.5 * float(yaw)
    return 0.0, 0.0, math.sin(half), math.cos(half)


def occupancy_grid_to_dict(msg: OccupancyGrid, topic: str) -> dict[str, Any]:
    origin = msg.info.origin.position
    return {
        "source": msg.header.frame_id or "map",
        "topic": str(topic or ""),
        "width": int(msg.info.width),
        "height": int(msg.info.height),
        "resolution": float(msg.info.resolution),
        "origin": [float(origin.x), float(origin.y)],
        "data": list(msg.data),
        "stamp_sec": stamp_to_sec(msg),
    }


class ObjectApproachLiveTestNode(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("object_approach_live_test")
        self.args = args
        self.lock = threading.RLock()
        self.left_bgr: np.ndarray | None = None
        self.right_bgr: np.ndarray | None = None
        self.scan_msg: LaserScan | None = None
        self.map_msg: OccupancyGrid | None = None
        self.left_stamp = 0.0
        self.right_stamp = 0.0
        self.scan_stamp = 0.0
        self.map_stamp = 0.0
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.goal_pub = self.create_publisher(PoseStamped, args.goal_topic, 1)
        self.create_subscription(Image, args.left_topic, self._on_left, 1)
        self.create_subscription(Image, args.right_topic, self._on_right, 1)
        self.create_subscription(LaserScan, args.scan_topic, self._on_scan, 1)
        self.create_subscription(OccupancyGrid, args.map_topic, self._on_map, 1)
        self.get_logger().info(
            f"subscribed left={args.left_topic} right={args.right_topic} scan={args.scan_topic} map={args.map_topic}"
        )

    def _on_left(self, msg: Image) -> None:
        try:
            image = image_msg_to_bgr(msg)
        except Exception as exc:
            self.get_logger().warning(f"decode left failed: {exc}")
            return
        with self.lock:
            self.left_bgr = image
            self.left_stamp = stamp_to_sec(msg)

    def _on_right(self, msg: Image) -> None:
        try:
            image = image_msg_to_bgr(msg)
        except Exception as exc:
            self.get_logger().warning(f"decode right failed: {exc}")
            return
        with self.lock:
            self.right_bgr = image
            self.right_stamp = stamp_to_sec(msg)

    def _on_scan(self, msg: LaserScan) -> None:
        with self.lock:
            self.scan_msg = msg
            self.scan_stamp = stamp_to_sec(msg)

    def _on_map(self, msg: OccupancyGrid) -> None:
        with self.lock:
            self.map_msg = msg
            self.map_stamp = stamp_to_sec(msg)

    def snapshot_ready(self) -> tuple[bool, str]:
        with self.lock:
            if self.left_bgr is None:
                return False, "waiting for left image"
            if self.args.depth_backend in {"auto", "stereo", "stereo_primary_mono_guard"} and self.right_bgr is None:
                return False, "waiting for right image"
            if self.args.depth_backend == "mono_relative_lidar" and self.scan_msg is None:
                return False, "waiting for LaserScan"
            if self.map_msg is None:
                return False, f"waiting for {self.args.map_topic}"
        if self._current_state() is None:
            return False, "waiting for TF map->base_link"
        return True, "ready"

    def _current_state(self) -> dict[str, Any] | None:
        try:
            tf = self.tf_buffer.lookup_transform(
                self.args.map_frame,
                self.args.base_frame,
                Time(),
                timeout=Duration(seconds=0.2),
            )
        except TransformException:
            return None
        tr = tf.transform.translation
        rot = tf.transform.rotation
        with self.lock:
            grid = occupancy_grid_to_dict(self.map_msg, self.args.map_topic) if self.map_msg is not None else None
        return {
            "position": [float(tr.x), float(tr.y), float(tr.z)],
            "orientation": [float(rot.x), float(rot.y), float(rot.z), float(rot.w)],
            "source": "tf",
            "occupancy_grid": grid,
        }

    def run_once(self) -> dict[str, Any]:
        deadline = time.monotonic() + float(self.args.timeout_sec)
        while rclpy.ok() and time.monotonic() < deadline:
            ready, reason = self.snapshot_ready()
            if ready:
                break
            self.get_logger().info(reason)
            rclpy.spin_once(self, timeout_sec=0.25)
        else:
            raise TimeoutError("timed out waiting for live snapshot")

        with self.lock:
            left = None if self.left_bgr is None else self.left_bgr.copy()
            right = None if self.right_bgr is None else self.right_bgr.copy()
            scan = self.scan_msg
            stamps = {
                "left_stamp_sec": self.left_stamp,
                "right_stamp_sec": self.right_stamp,
                "scan_stamp_sec": self.scan_stamp,
                "map_stamp_sec": self.map_stamp,
            }
        state = self._current_state()
        if left is None or state is None:
            raise RuntimeError("snapshot became unavailable")

        output_root = self.args.output_root / datetime.now().strftime("live_object_approach_%Y%m%d")
        result = run_object_approach_from_snapshot(
            image=bgr_to_pil(left),
            right_image=bgr_to_pil(right) if right is not None else None,
            scan_msg=scan,
            target_description=self.args.target,
            current_state=state,
            grounding_query=self.args.grounding_query,
            vlm_query=self.args.vlm_query,
            sam_query=self.args.sam_query,
            stop_distance_m=self.args.stop_distance,
            depth_backend=self.args.depth_backend,
            dispatch=False,
            output_root=output_root,
        )
        result["snapshot_stamps"] = stamps
        self._publish_goal(result)
        summary_path = Path(result.get("summary_json") or "")
        if summary_path.exists():
            summary_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        return result

    def _publish_goal(self, result: dict[str, Any]) -> None:
        approach = result.get("approach") if isinstance(result, dict) else None
        if not isinstance(approach, dict):
            return
        goal = approach.get("map_goal") or {}
        position = goal.get("position")
        yaw = goal.get("yaw_rad")
        if not isinstance(position, list) or len(position) < 2 or yaw is None:
            return
        msg = PoseStamped()
        msg.header.frame_id = self.args.map_frame
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.position.x = float(position[0])
        msg.pose.position.y = float(position[1])
        msg.pose.position.z = float(position[2] if len(position) > 2 else 0.0)
        qx, qy, qz, qw = yaw_to_quat(float(yaw))
        msg.pose.orientation.x = qx
        msg.pose.orientation.y = qy
        msg.pose.orientation.z = qz
        msg.pose.orientation.w = qw
        self.goal_pub.publish(msg)
        self.get_logger().info(f"published approach goal to {self.args.goal_topic}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run live object approach localization without dispatching navigation.")
    parser.add_argument("--target", required=True)
    parser.add_argument("--grounding-query", default="")
    parser.add_argument("--vlm-query", default="")
    parser.add_argument("--sam-query", default="")
    parser.add_argument(
        "--depth-backend",
        default="auto",
        choices=["auto", "stereo", "stereo_primary_mono_guard", "mono_relative_lidar"],
    )
    parser.add_argument("--stop-distance", default=0.8, type=float)
    parser.add_argument("--left-topic", default="/stereo/left/image_raw")
    parser.add_argument("--right-topic", default="/stereo/right/image_raw")
    parser.add_argument("--scan-topic", default="/scan")
    parser.add_argument("--map-topic", default="/global_costmap/costmap")
    parser.add_argument("--goal-topic", default="/caragent/object_approach_goal")
    parser.add_argument("--map-frame", default="map")
    parser.add_argument("--base-frame", default="base_link")
    parser.add_argument("--timeout-sec", default=20.0, type=float)
    parser.add_argument("--output-root", default=DEFAULT_WORKSPACE / "perception_outputs" / "object_approach_live", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    rclpy.init()
    node = ObjectApproachLiveTestNode(args)
    try:
        result = node.run_once()
        print(json.dumps(
            {
                "status": result.get("status"),
                "summary_json": result.get("summary_json"),
                "approach_goal_json": result.get("approach_goal_json"),
                "approach_debug_png": (result.get("approach") or {}).get("debug_png"),
                "goal_topic": args.goal_topic,
                "approach": result.get("approach"),
            },
            indent=2,
            ensure_ascii=False,
        ))
        return 0 if result.get("status") in {"ok", "degraded", "already_close"} else 2
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
