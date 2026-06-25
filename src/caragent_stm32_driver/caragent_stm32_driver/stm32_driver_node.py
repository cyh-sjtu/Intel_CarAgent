#!/usr/bin/env python3
"""STM32 serial driver: reads ODOM telemetry, publishes /odom and odom->base_link TF, relays /cmd_vel to STM32."""

import math
import time
from typing import Optional

import rclpy
from rclpy._rclpy_pybind11 import RCLError
from geometry_msgs.msg import Quaternion, TransformStamped, Twist, Vector3
from nav_msgs.msg import Odometry
from rclpy.duration import Duration
from rclpy.node import Node
from std_msgs.msg import Header

try:
    import serial
    import serial.tools.list_ports
except ImportError:
    serial = None


def _deg_to_quat(roll_deg: float, pitch_deg: float, yaw_deg: float) -> Quaternion:
    cy = math.cos(yaw_deg * 0.008726646259971648)   # cos(yaw/2) with precomputed pi/360
    sy = math.sin(yaw_deg * 0.008726646259971648)
    cp = math.cos(pitch_deg * 0.008726646259971648)
    sp = math.sin(pitch_deg * 0.008726646259971648)
    cr = math.cos(roll_deg * 0.008726646259971648)
    sr = math.sin(roll_deg * 0.008726646259971648)

    q = Quaternion()
    q.w = cr * cp * cy + sr * sp * sy
    q.x = sr * cp * cy - cr * sp * sy
    q.y = cr * sp * cy + sr * cp * sy
    q.z = cr * cp * sy - sr * sp * cy
    return q


class Stm32DriverNode(Node):
    def __init__(self) -> None:
        super().__init__("caragent_stm32_driver")

        self.declare_parameter("stm32_port", "/dev/ttyUSB0")
        self.declare_parameter("baud_rate", 115200)
        self.declare_parameter("odom_frame", "odom")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("log_raw_lines", False)
        self.declare_parameter("warn_no_data_sec", 3.0)
        self.declare_parameter("tf_publish_rate_hz", 50.0)
        self.declare_parameter("tf_future_offset_sec", 0.05)
        self.declare_parameter("odom_yaw_offset_deg", 0.0)
        self.declare_parameter("odom_yaw_sign", 1.0)
        self.declare_parameter("base_link_yaw_offset_deg", 180.0)
        self.declare_parameter("linear_velocity_sign", 1.0)
        self.declare_parameter("angular_velocity_sign", 1.0)
        self.declare_parameter("reconnect_period_sec", 1.0)
        self.declare_parameter("zero_odom_on_start", True)
        self.declare_parameter("zero_odom_min_imu_frames", 10)
        self.declare_parameter("enable_cmd_vel", False)
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("cmd_send_rate_hz", 20.0)
        self.declare_parameter("cmd_timeout_sec", 0.3)
        self.declare_parameter("max_linear_mps", 0.12)
        self.declare_parameter("max_angular_radps", 0.8)
        self.declare_parameter("log_cmd_serial", False)
        self.declare_parameter("log_pc_debug", False)

        self._odom_frame = self.get_parameter("odom_frame").get_parameter_value().string_value
        self._base_frame = self.get_parameter("base_frame").get_parameter_value().string_value
        base_link_yaw_offset_deg = (
            self.get_parameter("base_link_yaw_offset_deg").get_parameter_value().double_value
        )

        self._odom_pub = self.create_publisher(Odometry, "/odom", 10)
        self._tf_broadcaster = None
        self._ser: Optional["serial.Serial"] = None

        self._bad_lines = 0
        self._sample_count = 0
        self._raw_line_count = 0
        self._last_rx_time = time.monotonic()
        self._latest_odom = None
        self._last_reconnect_attempt = 0.0
        self._odom_origin = None
        self._cmd_vel_enabled = self.get_parameter("enable_cmd_vel").get_parameter_value().bool_value
        self._cmd_target = (0.0, 0.0)
        self._last_cmd_rx_time = 0.0
        self._last_cmd_sent = None
        self._last_cmd_log_time = 0.0
        self._pc_debug_count = 0
        self._rc_debug_count = 0

        self._connect_serial()

        self._tf_broadcaster = self._create_tf_broadcaster()
        self._read_timer = self.create_timer(0.02, self._read_serial)
        tf_publish_rate_hz = self.get_parameter("tf_publish_rate_hz").get_parameter_value().double_value
        self._tf_timer = self.create_timer(1.0 / tf_publish_rate_hz, self._publish_latest_tf)
        self._watchdog_timer = self.create_timer(1.0, self._watch_serial)

        if self._cmd_vel_enabled:
            cmd_vel_topic = self.get_parameter("cmd_vel_topic").get_parameter_value().string_value
            cmd_send_rate_hz = self.get_parameter("cmd_send_rate_hz").get_parameter_value().double_value
            self._cmd_vel_sub = self.create_subscription(
                Twist,
                cmd_vel_topic,
                self._handle_cmd_vel,
                10,
            )
            self._cmd_timer = self.create_timer(1.0 / cmd_send_rate_hz, self._publish_serial_cmd)
            self.get_logger().warn(
                f"cmd_vel control ENABLED: topic={cmd_vel_topic}, "
                f"rate={cmd_send_rate_hz:.1f}Hz"
            )

        if self._ser is not None:
            self.get_logger().info(
                f"caragent_stm32_driver ready, port={self._ser.port}, "
                f"base_link_yaw_offset_deg={base_link_yaw_offset_deg:.1f}"
            )
        else:
            self.get_logger().error("Serial connection failed, waiting to reconnect")

    def _create_tf_broadcaster(self):
        # Lazy-import to avoid crash when tf2_ros is not available
        from tf2_ros import TransformBroadcaster
        return TransformBroadcaster(self)

    def _connect_serial(self) -> None:
        if serial is None:
            self.get_logger().error("pyserial not installed")
            return

        port = self.get_parameter("stm32_port").get_parameter_value().string_value
        baud = self.get_parameter("baud_rate").get_parameter_value().integer_value

        opened = False
        for attempt in range(3):
            try:
                self._ser = serial.Serial(port, baud, timeout=0.005)
                self._ser.write_timeout = 0.02
                self._ser.reset_input_buffer()
                opened = True
                break
            except serial.SerialException as exc:
                self.get_logger().warn(f"Serial open attempt {attempt + 1}/3 failed: {exc}")
                time.sleep(1.0)

        if not opened:
            self.get_logger().error(f"Could not open {port}")

    @staticmethod
    def _clamp_float(value: float, min_value: float, max_value: float) -> float:
        return min(max(value, min_value), max_value)

    def _handle_cmd_vel(self, msg: Twist) -> None:
        max_linear = self.get_parameter("max_linear_mps").get_parameter_value().double_value
        max_angular = self.get_parameter("max_angular_radps").get_parameter_value().double_value

        if not (math.isfinite(msg.linear.x) and math.isfinite(msg.angular.z)):
            self._cmd_target = (0.0, 0.0)
            self._last_cmd_rx_time = time.monotonic()
            self.get_logger().warn("Invalid cmd_vel received, sending stop")
            return

        linear = self._clamp_float(msg.linear.x, -max_linear, max_linear)
        angular = self._clamp_float(msg.angular.z, -max_angular, max_angular)
        self._cmd_target = (linear, angular)
        self._last_cmd_rx_time = time.monotonic()

    def _publish_serial_cmd(self) -> None:
        if not self._cmd_vel_enabled:
            return

        timeout_sec = self.get_parameter("cmd_timeout_sec").get_parameter_value().double_value
        now = time.monotonic()

        if self._last_cmd_rx_time <= 0.0 or (now - self._last_cmd_rx_time) > timeout_sec:
            linear = 0.0
            angular = 0.0
        else:
            linear, angular = self._cmd_target

        self._write_serial_cmd(linear, angular)

    def _write_serial_cmd(self, linear_mps: float, angular_radps: float) -> None:
        if self._ser is None or not self._ser.is_open:
            self._try_reconnect_serial()
            return

        v_mmps = int(round(linear_mps * 1000.0))
        w_mradps = int(round(angular_radps * 1000.0))

        if self._last_cmd_sent == (v_mmps, w_mradps) and v_mmps == 0 and w_mradps == 0:
            return

        line = f"CMD,{v_mmps},{w_mradps}\n".encode("ascii")
        try:
            self._ser.write(line)
            self._log_serial_cmd(v_mmps, w_mradps)
            self._last_cmd_sent = (v_mmps, w_mradps)
        except serial.SerialException as exc:
            self.get_logger().error(f"Serial write failed, will reconnect: {exc}")
            self._close_serial()

    def _log_serial_cmd(self, v_mmps: int, w_mradps: int) -> None:
        if not self.get_parameter("log_cmd_serial").get_parameter_value().bool_value:
            return

        now = time.monotonic()
        changed = self._last_cmd_sent != (v_mmps, w_mradps)
        if changed or (now - self._last_cmd_log_time) >= 1.0:
            self._last_cmd_log_time = now
            self.get_logger().info(f"serial tx: CMD,{v_mmps},{w_mradps}")

    def stop_robot(self) -> None:
        if self._cmd_vel_enabled:
            self._write_serial_cmd(0.0, 0.0)

    def _close_serial(self) -> None:
        if self._ser is None:
            return

        try:
            if self._ser.is_open:
                self._ser.close()
        except serial.SerialException as exc:
            self.get_logger().warn(f"Serial close failed: {exc}")
        finally:
            self._ser = None

    def _try_reconnect_serial(self) -> None:
        reconnect_period = self.get_parameter("reconnect_period_sec").get_parameter_value().double_value
        now = time.monotonic()

        if (now - self._last_reconnect_attempt) < reconnect_period:
            return

        self._last_reconnect_attempt = now
        self.get_logger().warn("Trying to reconnect STM32 serial port")
        self._connect_serial()
        if self._ser is not None:
            self._latest_odom = None
            self._last_rx_time = time.monotonic()
            self.get_logger().info(f"Reconnected STM32 serial port: {self._ser.port}")

    def _parse_line(self, line: str) -> Optional[dict]:
        line = line.strip()
        if not line.startswith("ODOM,"):
            return None

        parts = line.split(",")
        if len(parts) < 15:
            return None

        try:
            return {
                "t_ms": int(parts[1]),
                "x_m": int(parts[2]) / 1000.0,
                "y_m": int(parts[3]) / 1000.0,
                "odom_yaw_deg": int(parts[4]) / 1000.0,
                "imu_yaw_deg": int(parts[5]) / 1000.0,
                "imu_roll_deg": int(parts[6]) / 1000.0,
                "imu_pitch_deg": int(parts[7]) / 1000.0,
                "v_mps": int(parts[8]) / 1000.0,
                "w_radps": int(parts[9]) / 1000.0,
                "odom_ok": int(parts[10]),
                "imu_frames": int(parts[11]),
                "fused_x_m": int(parts[12]) / 1000.0,
                "fused_y_m": int(parts[13]) / 1000.0,
                "fused_yaw_deg": int(parts[14]) / 1000.0,
            }
        except (ValueError, IndexError):
            return None

    def _handle_pc_debug_line(self, line: str) -> bool:
        line = line.strip()
        if not line.startswith("PCDBG,"):
            return False

        self._pc_debug_count += 1
        if not self.get_parameter("log_pc_debug").get_parameter_value().bool_value:
            return True

        parts = line.split(",")
        if len(parts) < 13:
            self.get_logger().warn(f"malformed PCDBG line: {line}")
            return True

        try:
            data = {
                "t_ms": int(parts[1]),
                "bytes": int(parts[2]),
                "lines": int(parts[3]),
                "frames": int(parts[4]),
                "bad": int(parts[5]),
                "fresh": int(parts[6]),
                "v": int(parts[7]),
                "w": int(parts[8]),
                "source": int(parts[9]),
                "left_rpm": int(parts[10]),
                "right_rpm": int(parts[11]),
                "last_byte": int(parts[12]),
                "remote_armed": int(parts[13]) if len(parts) > 13 else -1,
                "remote_linear": int(parts[14]) if len(parts) > 14 else 0,
                "remote_turn": int(parts[15]) if len(parts) > 15 else 0,
                "remote_linear_cmd": int(parts[16]) if len(parts) > 16 else 0,
                "remote_turn_cmd": int(parts[17]) if len(parts) > 17 else 0,
                "remote_linear_limit": int(parts[18]) if len(parts) > 18 else 0,
                "remote_turn_limit": int(parts[19]) if len(parts) > 19 else 0,
            }
        except ValueError:
            self.get_logger().warn(f"malformed PCDBG line: {line}")
            return True

        if self._pc_debug_count <= 5 or self._pc_debug_count % 10 == 0:
            self.get_logger().info(
                "pcdbg "
                f"bytes={data['bytes']} lines={data['lines']} frames={data['frames']} "
                f"bad={data['bad']} fresh={data['fresh']} "
                f"cmd=({data['v']},{data['w']}) source={data['source']} "
                f"rpm=({data['left_rpm']},{data['right_rpm']}) "
                f"remote=(armed={data['remote_armed']} raw={data['remote_linear']},{data['remote_turn']} "
                f"cmd={data['remote_linear_cmd']},{data['remote_turn_cmd']} "
                f"limit={data['remote_linear_limit']},{data['remote_turn_limit']}) "
                f"last_byte=0x{data['last_byte']:02X}"
            )

        return True

    def _handle_rc_debug_line(self, line: str) -> bool:
        line = line.strip()
        if not line.startswith("RCDBG,"):
            return False

        self._rc_debug_count += 1
        if not self.get_parameter("log_pc_debug").get_parameter_value().bool_value:
            return True

        parts = line.split(",")
        if len(parts) < 19:
            self.get_logger().warn(f"malformed RCDBG line: {line}")
            return True

        try:
            t_ms = int(parts[1])
            frames = int(parts[2])
            channels = [int(value) for value in parts[3:19]]
        except ValueError:
            self.get_logger().warn(f"malformed RCDBG line: {line}")
            return True

        if self._rc_debug_count <= 5 or self._rc_debug_count % 4 == 0:
            channel_text = " ".join(
                f"ch{index + 1}={value}" for index, value in enumerate(channels)
            )
            self.get_logger().info(
                f"rcdbg t={t_ms} frames={frames} {channel_text}"
            )

        return True

    def _read_serial(self) -> None:
        if self._ser is None or not self._ser.is_open:
            self._try_reconnect_serial()
            return

        deadline = time.monotonic() + 0.015
        while time.monotonic() < deadline:
            try:
                if self._ser.in_waiting <= 0:
                    break
                raw = self._ser.readline()
            except serial.SerialException as exc:
                self.get_logger().error(f"Serial read failed, will reconnect: {exc}")
                self._close_serial()
                return

            if not raw:
                break

            line = raw.decode("ascii", errors="ignore")
            self._raw_line_count += 1
            self._last_rx_time = time.monotonic()

            if self.get_parameter("log_raw_lines").get_parameter_value().bool_value:
                self.get_logger().info(f"raw serial line: {line.strip()}")

            if self._handle_pc_debug_line(line):
                continue
            if self._handle_rc_debug_line(line):
                continue

            parsed = self._parse_line(line)
            if parsed is None:
                self._bad_lines += 1
                if self._bad_lines <= 5 or self._bad_lines % 50 == 0:
                    self.get_logger().warn(
                        f"ignored non-ODOM serial line #{self._bad_lines}: {line.strip()}"
                    )
                continue

            self._sample_count += 1
            self._latest_odom = parsed
            self._publish_odom(parsed)

    def _watch_serial(self) -> None:
        if self._ser is None or not self._ser.is_open:
            return

        warn_after = self.get_parameter("warn_no_data_sec").get_parameter_value().double_value
        idle_sec = time.monotonic() - self._last_rx_time

        if self._sample_count == 0 and idle_sec >= warn_after:
            self.get_logger().warn(
                f"no valid ODOM received yet on {self._ser.port}; "
                f"raw_lines={self._raw_line_count} bad_lines={self._bad_lines}"
            )
            self._last_rx_time = time.monotonic()

    def _publish_latest_tf(self) -> None:
        if self._latest_odom is None:
            return
        self._publish_tf(self._latest_odom)

    @staticmethod
    def _wrap_deg(angle: float) -> float:
        while angle > 180.0:
            angle -= 360.0
        while angle < -180.0:
            angle += 360.0
        return angle

    def _correct_odom_raw(self, data: dict) -> dict:
        yaw_offset_deg = self.get_parameter("odom_yaw_offset_deg").get_parameter_value().double_value
        yaw_sign = self.get_parameter("odom_yaw_sign").get_parameter_value().double_value
        linear_sign = self.get_parameter("linear_velocity_sign").get_parameter_value().double_value
        angular_sign = self.get_parameter("angular_velocity_sign").get_parameter_value().double_value
        yaw_offset_rad = math.radians(yaw_offset_deg)
        cos_yaw = math.cos(yaw_offset_rad)
        sin_yaw = math.sin(yaw_offset_rad)
        raw_x = data["fused_x_m"]
        raw_y = data["fused_y_m"]

        return {
            "x_m": cos_yaw * raw_x - sin_yaw * raw_y,
            "y_m": sin_yaw * raw_x + cos_yaw * raw_y,
            "yaw_deg": yaw_sign * data["fused_yaw_deg"] + yaw_offset_deg,
            "v_mps": linear_sign * data["v_mps"],
            "w_radps": angular_sign * data["w_radps"],
        }

    def _zero_odom(self, data: dict) -> Optional[dict]:
        odom_data = self._correct_odom_raw(data)
        zero_on_start = self.get_parameter("zero_odom_on_start").get_parameter_value().bool_value

        if not zero_on_start:
            return odom_data

        if self._odom_origin is None:
            min_imu_frames = self.get_parameter("zero_odom_min_imu_frames").get_parameter_value().integer_value
            if data["odom_ok"] == 0 or data["imu_frames"] < min_imu_frames:
                return None

            self._odom_origin = {
                "x_m": odom_data["x_m"],
                "y_m": odom_data["y_m"],
                "yaw_deg": odom_data["yaw_deg"],
            }
            self.get_logger().info(
                "Odom origin initialized at "
                f"x={odom_data['x_m']:.3f} y={odom_data['y_m']:.3f} yaw={odom_data['yaw_deg']:.1f}"
            )

        dx = odom_data["x_m"] - self._odom_origin["x_m"]
        dy = odom_data["y_m"] - self._odom_origin["y_m"]
        origin_yaw_rad = math.radians(self._odom_origin["yaw_deg"])
        cos_yaw = math.cos(origin_yaw_rad)
        sin_yaw = math.sin(origin_yaw_rad)

        return {
            "x_m": cos_yaw * dx + sin_yaw * dy,
            "y_m": -sin_yaw * dx + cos_yaw * dy,
            "yaw_deg": self._wrap_deg(odom_data["yaw_deg"] - self._odom_origin["yaw_deg"]),
            "v_mps": odom_data["v_mps"],
            "w_radps": odom_data["w_radps"],
        }

    def _correct_odom(self, data: dict) -> Optional[dict]:
        odom_data = self._zero_odom(data)
        if odom_data is None:
            return None

        # This offset defines the fixed rotation from the STM32 odometry axes
        # into the ROS robot axes. Apply it to both position and yaw after
        # zeroing, otherwise RViz can show a corrected arrow but an uncorrected
        # odom track.
        base_link_yaw_offset_deg = (
            self.get_parameter("base_link_yaw_offset_deg").get_parameter_value().double_value
        )
        offset_rad = math.radians(base_link_yaw_offset_deg)
        cos_offset = math.cos(offset_rad)
        sin_offset = math.sin(offset_rad)

        return {
            "x_m": cos_offset * odom_data["x_m"] - sin_offset * odom_data["y_m"],
            "y_m": sin_offset * odom_data["x_m"] + cos_offset * odom_data["y_m"],
            "yaw_deg": self._wrap_deg(odom_data["yaw_deg"] + base_link_yaw_offset_deg),
            "v_x_mps": odom_data["v_mps"],
            "v_y_mps": 0.0,
            "w_radps": odom_data["w_radps"],
        }

    def _publish_odom(self, data: dict) -> None:
        now = self.get_clock().now().to_msg()
        odom_data = self._correct_odom(data)
        if odom_data is None:
            return

        x = odom_data["x_m"]
        y = odom_data["y_m"]
        yaw = odom_data["yaw_deg"]
        v_x = odom_data["v_x_mps"]
        v_y = odom_data["v_y_mps"]
        w = odom_data["w_radps"]

        quat = _deg_to_quat(0.0, 0.0, yaw)

        odom = Odometry()
        odom.header = Header(stamp=now, frame_id=self._odom_frame)
        odom.child_frame_id = self._base_frame
        odom.pose.pose.position.x = x
        odom.pose.pose.position.y = y
        odom.pose.pose.position.z = 0.0
        odom.pose.pose.orientation = quat
        odom.twist.twist.linear = Vector3(x=v_x, y=v_y, z=0.0)
        odom.twist.twist.angular = Vector3(x=0.0, y=0.0, z=w)

        # Covariance: order is (x, y, z, roll, pitch, yaw)
        # Pose covariance: ~1cm in x/y, large uncertainty in others
        odom.pose.covariance[0] = 0.01   # x
        odom.pose.covariance[7] = 0.01   # y
        odom.pose.covariance[14] = 1e9   # z (unobservable)
        odom.pose.covariance[21] = 1e9   # roll
        odom.pose.covariance[28] = 1e9   # pitch
        odom.pose.covariance[35] = 0.001 # yaw (~1.8 deg std)

        # Twist covariance: ~0.1m/s in linear, ~0.05rad/s in angular
        odom.twist.covariance[0] = 0.01
        odom.twist.covariance[7] = 1e9
        odom.twist.covariance[14] = 1e9
        odom.twist.covariance[21] = 1e9
        odom.twist.covariance[28] = 1e9
        odom.twist.covariance[35] = 0.0025

        self._odom_pub.publish(odom)
        self._publish_tf(data)

        if self._sample_count % 50 == 0:
            self.get_logger().info(
                f"x={x:.3f} y={y:.3f} yaw={yaw:.1f} vx={v_x:.3f} vy={v_y:.3f} w={w:.3f} "
                f"samples={self._sample_count} bad={self._bad_lines}"
            )

    def _publish_tf(self, data: dict) -> None:
        if self._tf_broadcaster is not None:
            odom_data = self._correct_odom(data)
            if odom_data is None:
                return

            future_offset = self.get_parameter("tf_future_offset_sec").get_parameter_value().double_value
            stamp = self.get_clock().now() + Duration(seconds=future_offset)
            quat = _deg_to_quat(0.0, 0.0, odom_data["yaw_deg"])

            t = TransformStamped()
            t.header = Header(stamp=stamp.to_msg(), frame_id=self._odom_frame)
            t.child_frame_id = self._base_frame
            t.transform.translation.x = odom_data["x_m"]
            t.transform.translation.y = odom_data["y_m"]
            t.transform.translation.z = 0.0
            t.transform.rotation = quat
            self._tf_broadcaster.sendTransform(t)


def main(args=None) -> int:
    rclpy.init(args=args)
    node = Stm32DriverNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_robot()
        if node._ser is not None and node._ser.is_open:
            node._ser.close()
        node.destroy_node()
        if rclpy.ok():
            try:
                rclpy.shutdown()
            except RCLError:
                pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
