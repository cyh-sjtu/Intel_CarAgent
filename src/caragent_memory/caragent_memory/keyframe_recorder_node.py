#!/usr/bin/env python3
"""Online stereo keyframe candidate recorder."""

from __future__ import annotations

import json
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import Image, LaserScan
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformException, TransformListener

from caragent_memory.dataset import append_jsonl, ensure_candidate_dataset, write_json
from caragent_memory.geometry import planar_distance, quaternion_xyzw_to_yaw, yaw_difference_deg
from caragent_memory.image_quality import compute_image_quality, split_side_by_side
from caragent_memory.scan_summary import scan_msg_to_arrays, summarize_scan_arrays


def _stamp_to_float(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def _default_session_name() -> str:
    return datetime.now().strftime("session_%Y%m%d_%H%M%S")


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


class KeyframeRecorderNode(Node):
    """Record side-by-side stereo candidate frames with pose and scan metadata."""

    def __init__(self) -> None:
        super().__init__("caragent_keyframe_recorder")

        self.declare_parameter("image_topic", "/stereo/image_raw")
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("session_name", "")
        self.declare_parameter("output_root", "~/caragent_ws/keyframes")
        self.declare_parameter("left_width", 640)
        self.declare_parameter("right_width", 640)
        self.declare_parameter("min_time_sec", 1.5)
        self.declare_parameter("min_distance_m", 0.65)
        self.declare_parameter("min_yaw_deg", 30.0)
        self.declare_parameter("init_pose_delay_sec", 3.0)
        self.declare_parameter("max_tf_age_sec", 0.5)
        self.declare_parameter("enforce_tf_age", False)
        self.declare_parameter("use_latest_tf_on_failure", True)
        self.declare_parameter("pose_jump_max_m", 1.5)
        self.declare_parameter("pose_jump_window_sec", 1.0)
        self.declare_parameter("blur_min", 80.0)
        self.declare_parameter("brightness_min", 35.0)
        self.declare_parameter("brightness_max", 235.0)
        self.declare_parameter("contrast_min", 15.0)
        self.declare_parameter("jpeg_quality", 95)
        self.declare_parameter("save_format", "png")

        session_name = str(self.get_parameter("session_name").value).strip() or _default_session_name()
        output_root = Path(str(self.get_parameter("output_root").value)).expanduser()
        self.dataset_root = output_root / session_name
        ensure_candidate_dataset(self.dataset_root)

        self.session = {
            "session_name": session_name,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "node": self.get_name(),
            "parameters": self._parameter_snapshot(),
        }
        write_json(self.dataset_root / "session.json", self.session)

        self.bridge = CvBridge()
        self.tf_buffer = Buffer(cache_time=Duration(seconds=30.0))
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.latest_scan_msg: Optional[LaserScan] = None
        self.latest_odom_msg: Optional[Odometry] = None
        self.pending_manual_capture = False

        self.saved_count = 0
        self.last_saved_wall_time: Optional[float] = None
        self.last_saved_pose: Optional[dict] = None
        self.last_pose_seen: Optional[dict] = None
        self.last_pose_seen_time: Optional[float] = None
        self._init_pose_time: Optional[float] = None

        image_topic = str(self.get_parameter("image_topic").value)
        scan_topic = str(self.get_parameter("scan_topic").value)
        odom_topic = str(self.get_parameter("odom_topic").value)
        self.create_subscription(Image, image_topic, self._handle_image, 10)
        self.create_subscription(LaserScan, scan_topic, self._handle_scan, 10)
        self.create_subscription(Odometry, odom_topic, self._handle_odom, 10)
        self.create_subscription(PoseWithCovarianceStamped, "/initialpose", self._handle_initialpose, 10)
        self.create_service(Trigger, "/keyframe_recorder/capture_once", self._capture_once)

        self.get_logger().info(
            f"keyframe recorder ready: dataset={self.dataset_root} image_topic={image_topic}"
        )

    def _parameter_snapshot(self) -> dict:
        names = [
            "image_topic",
            "scan_topic",
            "odom_topic",
            "map_frame",
            "base_frame",
            "output_root",
            "left_width",
            "right_width",
            "min_time_sec",
            "min_distance_m",
            "min_yaw_deg",
            "init_pose_delay_sec",
            "max_tf_age_sec",
            "enforce_tf_age",
            "use_latest_tf_on_failure",
            "pose_jump_max_m",
            "pose_jump_window_sec",
            "blur_min",
            "brightness_min",
            "brightness_max",
            "contrast_min",
            "jpeg_quality",
            "save_format",
        ]
        return {name: self.get_parameter(name).value for name in names}

    def _handle_scan(self, msg: LaserScan) -> None:
        self.latest_scan_msg = msg

    def _handle_odom(self, msg: Odometry) -> None:
        self.latest_odom_msg = msg

    def _capture_once(self, request, response):
        del request
        self.pending_manual_capture = True
        response.success = True
        response.message = "manual capture will be saved on the next image frame"
        return response

    def _lookup_pose(self, image_stamp) -> tuple[Optional[dict], Optional[dict]]:
        map_frame = str(self.get_parameter("map_frame").value)
        base_frame = str(self.get_parameter("base_frame").value)
        max_tf_age = float(self.get_parameter("max_tf_age_sec").value)
        image_time = _stamp_to_float(image_stamp)

        lookup_mode = "stamp"
        lookup_error = ""
        try:
            transform = self.tf_buffer.lookup_transform(
                map_frame,
                base_frame,
                Time.from_msg(image_stamp),
                timeout=Duration(seconds=0.1),
            )
        except TransformException as exc:
            lookup_error = str(exc)
            if not bool(self.get_parameter("use_latest_tf_on_failure").value):
                return None, {
                    "ok": False,
                    "reason": "tf_lookup_failed",
                    "message": lookup_error,
                }
            try:
                transform = self.tf_buffer.lookup_transform(
                    map_frame,
                    base_frame,
                    Time(),
                    timeout=Duration(seconds=0.1),
                )
                lookup_mode = "latest"
            except TransformException as latest_exc:
                return None, {
                    "ok": False,
                    "reason": "tf_lookup_failed",
                    "message": lookup_error,
                    "latest_message": str(latest_exc),
                }

        tf_time = _stamp_to_float(transform.header.stamp)
        tf_age = image_time - tf_time
        tf_stale = abs(tf_age) > max_tf_age
        enforce_tf_age = bool(self.get_parameter("enforce_tf_age").value)
        if tf_stale and enforce_tf_age:
            return None, {
                "ok": False,
                "reason": "tf_too_old",
                "tf_time": tf_time,
                "image_time": image_time,
                "tf_age_sec": tf_age,
            }

        t = transform.transform.translation
        q = transform.transform.rotation
        orientation = [float(q.x), float(q.y), float(q.z), float(q.w)]
        yaw = quaternion_xyzw_to_yaw(orientation)
        pose = {
            "frame_id": map_frame,
            "child_frame_id": base_frame,
            "x": float(t.x),
            "y": float(t.y),
            "z": float(t.z),
            "yaw": float(yaw),
            "yaw_deg": float(math.degrees(yaw)),
            "orientation_xyzw": orientation,
            "timestamp": image_time,
            "tf_timestamp": tf_time,
            "tf_age_sec": float(tf_age),
            "tf_lookup_mode": lookup_mode,
            "tf_stale": bool(tf_stale),
        }
        return pose, {
            "ok": True,
            "tf_time": tf_time,
            "image_time": image_time,
            "tf_age_sec": float(tf_age),
            "tf_stale": bool(tf_stale),
            "max_tf_age_sec": max_tf_age,
            "enforce_tf_age": enforce_tf_age,
            "lookup_mode": lookup_mode,
            "stamp_lookup_error": lookup_error if lookup_mode == "latest" else "",
        }

    def _pose_jump_check(self, pose: dict, now_sec: float) -> dict:
        if self.last_pose_seen is None or self.last_pose_seen_time is None:
            self.last_pose_seen = pose
            self.last_pose_seen_time = now_sec
            return {"ok": True, "distance_m": 0.0, "dt_sec": None}

        distance = planar_distance(pose, self.last_pose_seen)
        dt = now_sec - self.last_pose_seen_time
        self.last_pose_seen = pose
        self.last_pose_seen_time = now_sec

        max_jump = float(self.get_parameter("pose_jump_max_m").value)
        window = float(self.get_parameter("pose_jump_window_sec").value)
        ok = not (dt >= 0.0 and dt <= window and distance > max_jump)
        return {
            "ok": bool(ok),
            "distance_m": float(distance),
            "dt_sec": float(dt),
            "max_jump_m": max_jump,
            "window_sec": window,
        }

    def _handle_initialpose(self, msg: PoseWithCovarianceStamped) -> None:
        stamp_sec = _stamp_to_float(msg.header.stamp)
        if stamp_sec <= 0.0:
            stamp_sec = self.get_clock().now().nanoseconds * 1e-9
        self._init_pose_time = stamp_sec
        delay = float(self.get_parameter("init_pose_delay_sec").value)
        self.get_logger().info(
            f"initial pose received at t={stamp_sec:.1f}, "
            f"keyframe recording enabled after {delay:.1f}s delay"
        )

    def _should_save(self, pose: dict, now_sec: float, manual: bool, quality_ok: bool) -> tuple[bool, str]:
        if manual:
            return True, "manual"
        if not quality_ok:
            return False, "quality"
        if self._init_pose_time is None:
            return False, "no_initial_pose"
        delay = float(self.get_parameter("init_pose_delay_sec").value)
        if now_sec - self._init_pose_time < delay:
            return False, "init_pose_delay"
        if self.last_saved_pose is None or self.last_saved_wall_time is None:
            return True, "first"

        dt = now_sec - self.last_saved_wall_time
        min_time = float(self.get_parameter("min_time_sec").value)
        if dt < min_time:
            return False, "time"

        distance = planar_distance(pose, self.last_saved_pose)
        yaw_delta = yaw_difference_deg(pose["yaw"], self.last_saved_pose["yaw"])
        if distance >= float(self.get_parameter("min_distance_m").value):
            return True, "distance"
        if yaw_delta >= float(self.get_parameter("min_yaw_deg").value):
            return True, "yaw"
        return False, "pose"

    def _scan_payload(self) -> tuple[dict, Optional[dict]]:
        if self.latest_scan_msg is None:
            return {
                "available": False,
                "front_min_m": None,
                "left_min_m": None,
                "right_min_m": None,
                "rear_min_m": None,
                "valid_count": 0,
            }, None
        arrays = scan_msg_to_arrays(self.latest_scan_msg)
        summary = summarize_scan_arrays(**arrays)
        return summary, arrays

    def _odom_covariance(self) -> Optional[dict]:
        if self.latest_odom_msg is None:
            return None
        pose_cov = list(self.latest_odom_msg.pose.covariance)
        twist_cov = list(self.latest_odom_msg.twist.covariance)
        return {
            "pose_covariance_diag": [
                float(pose_cov[0]),
                float(pose_cov[7]),
                float(pose_cov[14]),
                float(pose_cov[21]),
                float(pose_cov[28]),
                float(pose_cov[35]),
            ],
            "twist_covariance_diag": [
                float(twist_cov[0]),
                float(twist_cov[7]),
                float(twist_cov[14]),
                float(twist_cov[21]),
                float(twist_cov[28]),
                float(twist_cov[35]),
            ],
        }

    def _handle_image(self, msg: Image) -> None:
        now_sec = self.get_clock().now().nanoseconds * 1e-9
        manual = self.pending_manual_capture

        pose, tf_status = self._lookup_pose(msg.header.stamp)
        if pose is None:
            if manual:
                self.pending_manual_capture = False
            self.get_logger().warn(
                f"skip keyframe: {tf_status.get('reason', 'tf_failed')}",
                throttle_duration_sec=2.0,
            )
            return

        pose_jump = self._pose_jump_check(pose, now_sec)
        if not pose_jump["ok"] and not manual:
            self.get_logger().warn(
                "skip keyframe: pose jump distance=%.2fm dt=%.2fs"
                % (pose_jump["distance_m"], pose_jump["dt_sec"]),
                throttle_duration_sec=2.0,
            )
            return

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            self.get_logger().error(f"failed to convert image: {exc}")
            return

        left_width = int(self.get_parameter("left_width").value)
        right_width = int(self.get_parameter("right_width").value)
        try:
            left, right = split_side_by_side(frame, left_width=left_width, right_width=right_width)
        except ValueError as exc:
            self.get_logger().warn(f"skip keyframe: {exc}", throttle_duration_sec=2.0)
            return

        quality = compute_image_quality(
            left,
            blur_min=float(self.get_parameter("blur_min").value),
            brightness_min=float(self.get_parameter("brightness_min").value),
            brightness_max=float(self.get_parameter("brightness_max").value),
            contrast_min=float(self.get_parameter("contrast_min").value),
        )

        should_save, reason = self._should_save(pose, now_sec, manual, quality.quality_ok)
        if not should_save:
            if manual:
                self.pending_manual_capture = False
            return

        self.pending_manual_capture = False
        self._save_frame(
            msg=msg,
            frame=frame,
            left=left,
            right=right,
            pose=pose,
            quality=quality.to_dict(),
            tf_status=tf_status,
            pose_jump=pose_jump,
            trigger_reason=reason,
            manual=manual,
        )
        self.last_saved_wall_time = now_sec
        self.last_saved_pose = pose

    def _image_write_params(self) -> list[int]:
        save_format = str(self.get_parameter("save_format").value).lower()
        if save_format in {"jpg", "jpeg"}:
            return [cv2.IMWRITE_JPEG_QUALITY, int(self.get_parameter("jpeg_quality").value)]
        if save_format == "png":
            return [cv2.IMWRITE_PNG_COMPRESSION, 3]
        return []

    def _save_frame(
        self,
        *,
        msg: Image,
        frame: np.ndarray,
        left: np.ndarray,
        right: Optional[np.ndarray],
        pose: dict,
        quality: dict,
        tf_status: dict,
        pose_jump: dict,
        trigger_reason: str,
        manual: bool,
    ) -> None:
        self.saved_count += 1
        frame_id = f"{self.saved_count:06d}"
        ext = str(self.get_parameter("save_format").value).lower()
        if ext == "jpeg":
            ext = "jpg"

        raw_path = self.dataset_root / "raw" / f"{frame_id}.{ext}"
        left_path = self.dataset_root / "left" / f"{frame_id}.{ext}"
        right_path = self.dataset_root / "right" / f"{frame_id}.{ext}" if right is not None else None
        pose_path = self.dataset_root / "pose" / f"{frame_id}_pose.json"
        meta_path = self.dataset_root / "meta" / f"{frame_id}_meta.json"
        scan_path = self.dataset_root / "scan" / f"{frame_id}_scan.npz"

        write_params = self._image_write_params()
        cv2.imwrite(str(raw_path), frame, write_params)
        cv2.imwrite(str(left_path), left, write_params)
        if right_path is not None:
            cv2.imwrite(str(right_path), right, write_params)

        scan_summary, scan_arrays = self._scan_payload()
        if scan_arrays is not None:
            np.savez_compressed(scan_path, **scan_arrays)
        else:
            scan_path = None

        pose_payload = dict(pose)
        pose_payload["image_timestamp"] = _stamp_to_float(msg.header.stamp)
        write_json(pose_path, pose_payload)

        meta = {
            "frame_id": frame_id,
            "manual": bool(manual),
            "trigger_reason": trigger_reason,
            "quality_ok": bool(quality["quality_ok"]),
            "quality": quality,
            "tf_status": tf_status,
            "pose_jump": pose_jump,
            "scan_summary": scan_summary,
            "odom_covariance": self._odom_covariance(),
            "image": {
                "header_frame_id": msg.header.frame_id,
                "height": int(msg.height),
                "width": int(msg.width),
                "encoding": msg.encoding,
                "raw_path": _relative(raw_path, self.dataset_root),
                "left_path": _relative(left_path, self.dataset_root),
                "right_path": _relative(right_path, self.dataset_root) if right_path is not None else None,
            },
        }
        write_json(meta_path, meta)

        manifest_item = {
            "frame_id": frame_id,
            "raw_path": _relative(raw_path, self.dataset_root),
            "left_path": _relative(left_path, self.dataset_root),
            "right_path": _relative(right_path, self.dataset_root) if right_path is not None else None,
            "pose_path": _relative(pose_path, self.dataset_root),
            "meta_path": _relative(meta_path, self.dataset_root),
            "scan_path": _relative(scan_path, self.dataset_root) if scan_path is not None else None,
            "trigger_reason": trigger_reason,
            "quality_ok": bool(quality["quality_ok"]),
            "timestamp": pose_payload["image_timestamp"],
            "x": float(pose["x"]),
            "y": float(pose["y"]),
            "yaw": float(pose["yaw"]),
        }
        append_jsonl(self.dataset_root / "manifest.jsonl", manifest_item)

        self.get_logger().info(
            "saved keyframe candidate %s reason=%s quality=%s x=%.2f y=%.2f yaw=%.1f"
            % (
                frame_id,
                trigger_reason,
                quality["quality_ok"],
                pose["x"],
                pose["y"],
                pose["yaw_deg"],
            )
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = KeyframeRecorderNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
