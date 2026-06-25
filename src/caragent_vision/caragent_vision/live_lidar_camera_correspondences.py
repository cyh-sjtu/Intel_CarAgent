"""Live collection of LiDAR-camera correspondences from ROS topics."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan


LASER_X_M = 0.115
LASER_Y_M = 0.0
LASER_YAW_RAD = math.pi


class LiveLidarCameraCorrespondences(Node):
    def __init__(self) -> None:
        super().__init__("live_lidar_camera_correspondences")
        self.declare_parameter("image_topic", "/stereo/left/image_raw")
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("output_jsonl", "/home/car/caragent_ws/calibration/lidar_camera/correspondences.jsonl")
        self.declare_parameter("window_name", "LiDAR-Camera Correspondences")
        self.declare_parameter("display_width", 1500)
        self.declare_parameter("display_height", 820)

        self.image_topic = str(self.get_parameter("image_topic").value)
        self.scan_topic = str(self.get_parameter("scan_topic").value)
        self.output_jsonl = Path(str(self.get_parameter("output_jsonl").value)).expanduser()
        self.window_name = str(self.get_parameter("window_name").value)
        self.display_width = int(self.get_parameter("display_width").value)
        self.display_height = int(self.get_parameter("display_height").value)

        self.bridge = CvBridge()
        self.latest_image: Optional[np.ndarray] = None
        self.latest_scan: Optional[LaserScan] = None
        self.clicked: Optional[tuple[int, int]] = None
        self.hover_pixel: Optional[tuple[int, int]] = None
        self.cursor_pixel: Optional[tuple[int, int]] = None
        self.selected_beam: int = 0
        self.saved_count = 0

        self._scan_bg_cache: Optional[tuple[int, int, np.ndarray, np.ndarray, float]] = None
        self._image_rect: Optional[tuple[int, int, int, int, float]] = None
        self._scan_rect: Optional[tuple[int, int, int, int]] = None
        self._scan_display_points: Optional[tuple[np.ndarray, np.ndarray, np.ndarray]] = None

        self.create_subscription(Image, self.image_topic, self._handle_image, 1)
        self.create_subscription(LaserScan, self.scan_topic, self._handle_scan, 1)
        self.timer = self.create_timer(0.05, self._draw)

        self.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.display_width, self.display_height)
        cv2.setMouseCallback(self.window_name, self._on_mouse)

        self.get_logger().info(
            f"live calibration collector ready: image={self.image_topic} scan={self.scan_topic} output={self.output_jsonl}"
        )
        self.get_logger().info(
            "Controls: click image point, click scan point to select nearest beam, h/j/k/l move image cursor, "
            "a/d or arrows select beam, A/D jump 10 beams, s save, q quit"
        )

    def _handle_image(self, msg: Image) -> None:
        self.latest_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        if self.cursor_pixel is None:
            h, w = self.latest_image.shape[:2]
            self.cursor_pixel = (w // 2, h // 2)

    def _handle_scan(self, msg: LaserScan) -> None:
        self.latest_scan = msg
        if self.selected_beam <= 0 or self.selected_beam >= len(msg.ranges):
            self.selected_beam = len(msg.ranges) // 2

    def _on_mouse(self, event, x, y, flags, userdata) -> None:
        image_pixel = self._display_to_image_pixel(x, y)
        if image_pixel is not None:
            self.hover_pixel = image_pixel
            self.cursor_pixel = image_pixel
        elif event == cv2.EVENT_LBUTTONDOWN and self._select_scan_at_display(x, y):
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            if image_pixel is None:
                self.get_logger().info("Click ignored: click the image panel for pixels, or the scan panel for a LiDAR beam.")
                return
            self.clicked = image_pixel
            self.cursor_pixel = self.clicked
            self.get_logger().info(f"clicked image pixel: {self.clicked}")

    def _display_to_image_pixel(self, x: int, y: int) -> Optional[tuple[int, int]]:
        if self.latest_image is None or self._image_rect is None:
            return None
        rx, ry, rw, rh, scale = self._image_rect
        if not (rx <= x < rx + rw and ry <= y < ry + rh):
            return None
        h, w = self.latest_image.shape[:2]
        u = int(round((x - rx) / scale))
        v = int(round((y - ry) / scale))
        return int(np.clip(u, 0, w - 1)), int(np.clip(v, 0, h - 1))

    def _select_scan_at_display(self, x: int, y: int) -> bool:
        if self._scan_rect is None or self._scan_display_points is None:
            return False
        rx, ry, rw, rh = self._scan_rect
        if not (rx <= x < rx + rw and ry <= y < ry + rh):
            return False
        beams, px, py = self._scan_display_points
        if beams.size == 0:
            return False
        dx = px.astype(np.float32) - float(x - rx)
        dy = py.astype(np.float32) - float(y - ry)
        nearest = int(np.argmin(dx * dx + dy * dy))
        self.selected_beam = int(beams[nearest])
        self.get_logger().info(f"selected scan beam by click: {self.selected_beam}")
        return True

    def _scan_arrays(self) -> Optional[dict[str, np.ndarray]]:
        if self.latest_scan is None:
            return None
        msg = self.latest_scan
        ranges = np.asarray(msg.ranges, dtype=np.float32)
        angles = msg.angle_min + np.arange(len(ranges), dtype=np.float32) * msg.angle_increment
        valid = np.isfinite(ranges) & (ranges >= msg.range_min) & (ranges <= msg.range_max)
        x_l = ranges * np.cos(angles)
        y_l = ranges * np.sin(angles)
        c = math.cos(LASER_YAW_RAD)
        s = math.sin(LASER_YAW_RAD)
        x_b = LASER_X_M + c * x_l - s * y_l
        y_b = LASER_Y_M + s * x_l + c * y_l
        return {"ranges": ranges, "angles": angles, "valid": valid, "x_base": x_b, "y_base": y_b}

    def _draw_scan_panel(self, scan: dict[str, np.ndarray], width: int, height: int) -> np.ndarray:
        scale = min(width / 5.2, height / 4.6)
        origin = np.array([width * 0.50, height * 0.82], dtype=np.float32)
        cache_key = (width, height)
        if self._scan_bg_cache is None or self._scan_bg_cache[:2] != cache_key:
            bg = np.full((height, width, 3), 246, dtype=np.uint8)
            for gx in np.arange(0.0, 4.1, 0.5):
                yy = int(origin[1] - gx * scale)
                cv2.line(bg, (0, yy), (width, yy), (220, 220, 214), 1)
                cv2.putText(bg, f"x={gx:.1f}", (6, yy - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (110, 110, 100), 1)
            for gy in np.arange(-2.5, 2.6, 0.5):
                xx = int(origin[0] - gy * scale)
                cv2.line(bg, (xx, 0), (xx, height), (220, 220, 214), 1)
            base = (int(origin[0]), int(origin[1]))
            cv2.circle(bg, base, 5, (20, 20, 20), -1)
            cv2.putText(bg, "base", (base[0] + 8, base[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1)
            self._scan_bg_cache = (width, height, bg, origin, scale)
        _, _, bg, origin, scale = self._scan_bg_cache
        panel = bg.copy()

        def to_px(x, y):
            return origin[0] - y * scale, origin[1] - x * scale

        valid = scan["valid"].astype(bool)
        px, py = to_px(scan["x_base"][valid], scan["y_base"][valid])
        px = px.astype(int)
        py = py.astype(int)
        in_bounds = (0 <= px) & (px < width) & (0 <= py) & (py < height)
        panel[py[in_bounds], px[in_bounds]] = (70, 120, 180)
        valid_indices = np.flatnonzero(valid)[in_bounds]
        self._scan_display_points = (valid_indices.astype(np.int32), px[in_bounds].astype(np.int32), py[in_bounds].astype(np.int32))

        base = (int(origin[0]), int(origin[1]))
        idx = int(np.clip(self.selected_beam, 0, len(scan["ranges"]) - 1))
        self.selected_beam = idx
        if bool(scan["valid"][idx]):
            sx, sy = to_px(np.array([scan["x_base"][idx]]), np.array([scan["y_base"][idx]]))
            p = (int(sx[0]), int(sy[0]))
            cv2.circle(panel, p, 7, (20, 30, 230), -1)
            cv2.line(panel, base, p, (20, 30, 230), 2)
            text = f"beam {idx} r={scan['ranges'][idx]:.3f}m angle={scan['angles'][idx]:.3f}"
        else:
            text = f"beam {idx} invalid"
        cv2.putText(panel, text, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1)
        cv2.putText(panel, "click nearest LiDAR point | a/d fine | A/D coarse", (12, height - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (20, 20, 20), 1)
        return panel

    def _fit_image_panel(self, image: np.ndarray, width: int, height: int) -> tuple[np.ndarray, float]:
        h, w = image.shape[:2]
        scale = min(width / max(1, w), height / max(1, h))
        view_w = max(1, int(round(w * scale)))
        view_h = max(1, int(round(h * scale)))
        view = cv2.resize(image, (view_w, view_h), interpolation=cv2.INTER_AREA)
        panel = np.full((height, width, 3), 30, dtype=np.uint8)
        ox = (width - view_w) // 2
        oy = (height - view_h) // 2
        panel[oy : oy + view_h, ox : ox + view_w] = view
        self._image_rect = (ox, oy, view_w, view_h, scale)
        return panel, scale

    def _draw(self) -> None:
        if self.latest_image is None:
            return
        image = self.latest_image.copy()
        scan = self._scan_arrays()
        h, w = image.shape[:2]
        if self.cursor_pixel is None:
            self.cursor_pixel = (w // 2, h // 2)
        marker_size = max(18, int(round(min(w, h) * 0.015)))
        if self.cursor_pixel is not None:
            cv2.drawMarker(image, self.cursor_pixel, (255, 0, 255), markerType=cv2.MARKER_TILTED_CROSS, markerSize=marker_size, thickness=2)
        if self.clicked is not None:
            cv2.drawMarker(image, self.clicked, (0, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=marker_size, thickness=2)
        cv2.putText(image, "click image | click scan | h/j/k/l pixel | a/d beam | s save | q quit", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 255, 0), 2)
        cv2.putText(image, f"saved={self.saved_count}", (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 255, 0), 2)
        coord_text = f"cursor={self.cursor_pixel}"
        if self.clicked is not None:
            coord_text += f" selected={self.clicked}"
        cv2.putText(image, coord_text, (10, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 255, 0), 2)

        canvas_w = max(1000, self.display_width)
        canvas_h = max(620, self.display_height)
        header_h = 34
        gap = 10
        body_h = canvas_h - header_h - gap * 2
        image_w = int(canvas_w * 0.62)
        scan_w = canvas_w - image_w - gap * 3
        image_panel, _ = self._fit_image_panel(image, image_w, body_h)
        if scan is None:
            panel = np.full((body_h, scan_w, 3), 230, dtype=np.uint8)
            cv2.putText(panel, "waiting for /scan", (20, body_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2)
            self._scan_display_points = None
        else:
            panel = self._draw_scan_panel(scan, scan_w, body_h)
        canvas = np.full((canvas_h, canvas_w, 3), 18, dtype=np.uint8)
        cv2.putText(canvas, "LiDAR-Camera Correspondence Collector", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (230, 230, 230), 1)
        img_x, body_y = gap, header_h + gap
        scan_x = image_w + gap * 2
        canvas[body_y : body_y + body_h, img_x : img_x + image_w] = image_panel
        canvas[body_y : body_y + body_h, scan_x : scan_x + scan_w] = panel
        if self._image_rect is not None:
            ix, iy, iw, ih, scale = self._image_rect
            self._image_rect = (img_x + ix, body_y + iy, iw, ih, scale)
        self._scan_rect = (scan_x, body_y, scan_w, body_h)
        cv2.rectangle(canvas, (img_x, body_y), (img_x + image_w - 1, body_y + body_h - 1), (90, 90, 90), 1)
        cv2.rectangle(canvas, (scan_x, body_y), (scan_x + scan_w - 1, body_y + body_h - 1), (90, 90, 90), 1)
        cv2.imshow(self.window_name, canvas)
        key = cv2.waitKey(1) & 0xFF
        self._handle_key(key, scan)

    def _handle_key(self, key: int, scan: Optional[dict[str, np.ndarray]]) -> None:
        if key in (255,):
            return
        if key in (ord("q"), 27):
            rclpy.shutdown()
            return
        if key in (81, ord("a")):
            self.selected_beam = max(0, self.selected_beam - 1)
        elif key in (83, ord("d")):
            if scan is not None:
                self.selected_beam = min(len(scan["ranges"]) - 1, self.selected_beam + 1)
        elif key == ord("A"):
            self.selected_beam = max(0, self.selected_beam - 10)
        elif key == ord("D"):
            if scan is not None:
                self.selected_beam = min(len(scan["ranges"]) - 1, self.selected_beam + 10)
        elif key in (ord("h"), ord("j"), ord("k"), ord("l"), ord("H"), ord("J"), ord("K"), ord("L")):
            self._move_cursor(key)
        elif key == ord("p"):
            self._select_cursor_pixel()
        elif key == ord("i"):
            self._prompt_pixel()
        elif key == ord("s"):
            self._save_current(scan)

    def _move_cursor(self, key: int) -> None:
        if self.latest_image is None:
            return
        h, w = self.latest_image.shape[:2]
        if self.cursor_pixel is None:
            self.cursor_pixel = (w // 2, h // 2)
        u, v = self.cursor_pixel
        step = 10 if chr(key).isupper() else 1
        ch = chr(key).lower()
        if ch == "h":
            u -= step
        elif ch == "l":
            u += step
        elif ch == "k":
            v -= step
        elif ch == "j":
            v += step
        self.cursor_pixel = (int(np.clip(u, 0, w - 1)), int(np.clip(v, 0, h - 1)))

    def _select_cursor_pixel(self) -> None:
        if self.cursor_pixel is not None:
            self.clicked = self.cursor_pixel
            self.get_logger().info(f"selected current cursor pixel: {self.clicked}")
            return
        self.get_logger().warn("No cursor pixel available yet.")

    def _prompt_pixel(self) -> None:
        try:
            text = input("Enter image pixel as 'u v': ").strip()
            u_str, v_str = text.replace(",", " ").split()[:2]
            u = int(float(u_str))
            v = int(float(v_str))
        except Exception as exc:
            self.get_logger().warn(f"Invalid pixel input: {exc}")
            return
        if self.latest_image is not None:
            h, w = self.latest_image.shape[:2]
            if not (0 <= u < w and 0 <= v < h):
                self.get_logger().warn(f"Pixel out of image bounds: {(u, v)} for image {w}x{h}")
                return
        self.clicked = (u, v)
        self.cursor_pixel = self.clicked
        self.get_logger().info(f"typed image pixel: {self.clicked}")

    def _save_current(self, scan: Optional[dict[str, np.ndarray]]) -> None:
        if self.clicked is None:
            self.get_logger().warn("No image point clicked yet.")
            return
        if scan is None:
            self.get_logger().warn("No scan received yet.")
            return
        idx = int(self.selected_beam)
        if not bool(scan["valid"][idx]):
            self.get_logger().warn(f"Selected beam {idx} is invalid.")
            return
        record = {
            "image_topic": self.image_topic,
            "scan_topic": self.scan_topic,
            "image_point": [int(self.clicked[0]), int(self.clicked[1])],
            "scan_point": {
                "beam_index": idx,
                "range_m": float(scan["ranges"][idx]),
                "angle_rad": float(scan["angles"][idx]),
                "base_xy_m": [float(scan["x_base"][idx]), float(scan["y_base"][idx])],
            },
        }
        with self.output_jsonl.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.saved_count += 1
        self.get_logger().info(
            f"saved #{self.saved_count}: pixel={record['image_point']} beam={idx} range={record['scan_point']['range_m']:.3f}"
        )

    def destroy_node(self) -> bool:
        cv2.destroyAllWindows()
        return super().destroy_node()


def main() -> None:
    rclpy.init()
    node = LiveLidarCameraCorrespondences()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
