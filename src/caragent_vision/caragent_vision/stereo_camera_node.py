import sys
import threading
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image


def _default_video_format():
    return "dshow" if sys.platform == "win32" else "v4l2"


class StereoCameraNode(Node):
    """Publish Huibo stereo camera images split from a side-by-side UVC frame."""

    def __init__(self):
        super().__init__("caragent_stereo_camera_node")

        # ---- capture parameters ----
        self.declare_parameter("backend", "pyav")
        self.declare_parameter("device", "/dev/video0")
        self.declare_parameter("video_format", "auto")
        self.declare_parameter("width", 3840)
        self.declare_parameter("height", 1200)
        self.declare_parameter("fps", 30.0)
        self.declare_parameter("left_width", 1920)
        self.declare_parameter("right_width", 1920)
        self.declare_parameter("rtbufsize", "64M")
        self.declare_parameter("frame_id_raw", "camera_link")
        self.declare_parameter("frame_id_left", "camera_left")
        self.declare_parameter("frame_id_right", "camera_right")
        self.declare_parameter("publish_raw", True)
        self.declare_parameter("publish_left", True)
        self.declare_parameter("publish_right", True)
        self.declare_parameter("show_image", False)
        self.declare_parameter("display_max_width", 1600)
        self.declare_parameter("display_max_height", 900)
        self.declare_parameter("display_scale", 0.0)
        self.declare_parameter("timer_rate_hz", 30.0)

        # ---- calibration / rectification ----
        self.declare_parameter("calib_file", "")
        self.declare_parameter("publish_rect", False)
        self.declare_parameter("publish_disparity", False)

        self._read_params()

        self.bridge = CvBridge()
        self.frame_lock = threading.Lock()
        self.latest_frame: Optional[np.ndarray] = None
        self.running = True
        self.window_name = "CarAgent Stereo Camera"
        self._preview_window_created = False
        self.frame_count = 0
        self.last_stat_time = time.monotonic()
        self.last_frame_count = 0

        # ---- rectification maps (lazy init from calibration) ----
        self._map_lx: Optional[np.ndarray] = None
        self._map_ly: Optional[np.ndarray] = None
        self._map_rx: Optional[np.ndarray] = None
        self._map_ry: Optional[np.ndarray] = None
        self._Q: Optional[np.ndarray] = None
        self._stereo_matcher: Optional[cv2.StereoMatcher] = None
        self._rect_initialized = False
        if (self.publish_rect or self.publish_disparity) and self.calib_file and Path(self.calib_file).exists():
            self._init_rectification()

        # ---- publishers ----
        self.raw_pub = self._make_publisher("stereo/image_raw", self.publish_raw)
        self.left_pub = self._make_publisher("stereo/left/image_raw", self.publish_left)
        self.right_pub = self._make_publisher("stereo/right/image_raw", self.publish_right)
        self.rect_left_pub = self._make_publisher("stereo/left/image_rect", self.publish_rect)
        self.rect_right_pub = self._make_publisher("stereo/right/image_rect", self.publish_rect)
        self.disparity_pub = self._make_publisher("stereo/disparity", self.publish_disparity)

        # ---- capture worker ----
        backend = self.backend
        if backend == "pyav":
            self.worker = threading.Thread(target=self._pyav_capture_worker, daemon=True)
        elif backend == "opencv":
            self.worker = threading.Thread(target=self._opencv_capture_worker, daemon=True)
        else:
            raise ValueError(f"unsupported camera backend: {backend}")

        self.worker.start()

        period = 1.0 / max(self.timer_rate_hz, 1.0)
        self.timer = self.create_timer(period, self._publish_latest_frame)

        self.get_logger().info(
            "stereo camera node ready: backend=%s device=%s format=%s size=%dx%d fps=%.1f "
            "split=%d+%d rect=%s disparity=%s"
            % (
                self.backend,
                self.device,
                self.video_format,
                self.width,
                self.height,
                self.fps,
                self.left_width,
                self.right_width,
                self.publish_rect,
                self.publish_disparity,
            )
        )

    # ------------------------------------------------------------------
    #  parameter helpers
    # ------------------------------------------------------------------
    def _read_params(self):
        self.backend = str(self.get_parameter("backend").value).lower()
        self.device = str(self.get_parameter("device").value)
        raw_fmt = str(self.get_parameter("video_format").value)
        self.video_format = _default_video_format() if raw_fmt == "auto" else raw_fmt
        self.width = int(self.get_parameter("width").value)
        self.height = int(self.get_parameter("height").value)
        self.fps = float(self.get_parameter("fps").value)
        self.left_width = int(self.get_parameter("left_width").value)
        self.right_width = int(self.get_parameter("right_width").value)
        self.rtbufsize = str(self.get_parameter("rtbufsize").value)
        self.frame_id_raw = str(self.get_parameter("frame_id_raw").value)
        self.frame_id_left = str(self.get_parameter("frame_id_left").value)
        self.frame_id_right = str(self.get_parameter("frame_id_right").value)
        self.publish_raw = bool(self.get_parameter("publish_raw").value)
        self.publish_left = bool(self.get_parameter("publish_left").value)
        self.publish_right = bool(self.get_parameter("publish_right").value)
        self.show_image = bool(self.get_parameter("show_image").value)
        self.display_max_width = int(self.get_parameter("display_max_width").value)
        self.display_max_height = int(self.get_parameter("display_max_height").value)
        self.display_scale = float(self.get_parameter("display_scale").value)
        self.timer_rate_hz = float(self.get_parameter("timer_rate_hz").value)
        self.calib_file = str(self.get_parameter("calib_file").value)
        self.publish_rect = bool(self.get_parameter("publish_rect").value)
        self.publish_disparity = bool(self.get_parameter("publish_disparity").value)

    def _make_publisher(self, topic: str, enabled: bool):
        return self.create_publisher(Image, topic, 10) if enabled else None

    # ------------------------------------------------------------------
    #  calibration / rectification
    # ------------------------------------------------------------------
    def _init_rectification(self):
        try:
            data = np.load(self.calib_file)
            mtx_l = data["mtx_l"]
            dist_l = data["dist_l"]
            mtx_r = data["mtx_r"]
            dist_r = data["dist_r"]
            R = data["R"]
            T = data["T"]

            image_size = (self.left_width, self.height)

            R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
                mtx_l, dist_l, mtx_r, dist_r,
                image_size, R, T,
                flags=cv2.CALIB_ZERO_DISPARITY,
                alpha=0,
            )

            self._map_lx, self._map_ly = cv2.initUndistortRectifyMap(
                mtx_l, dist_l, R1, P1, image_size, cv2.CV_32FC1,
            )
            self._map_rx, self._map_ry = cv2.initUndistortRectifyMap(
                mtx_r, dist_r, R2, P2, image_size, cv2.CV_32FC1,
            )
            self._Q = Q

            if self.publish_disparity:
                self._stereo_matcher = cv2.StereoSGBM_create(
                    minDisparity=0,
                    numDisparities=64,
                    blockSize=5,
                    P1=8 * 3 * 5 ** 2,
                    P2=32 * 3 * 5 ** 2,
                    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
                )

            self._rect_initialized = True
            self.get_logger().info(f"stereo calibration loaded from {self.calib_file}")
        except Exception as exc:
            self.get_logger().error(f"failed to load calibration: {exc}")

    # ------------------------------------------------------------------
    #  PyAV capture (primary)
    # ------------------------------------------------------------------
    def _pyav_device_name(self) -> str:
        """Resolve device to a PyAV-compatible name."""
        raw = self.device
        # already in PyAV format  "video=..." or contains "Camera"
        if raw.startswith("video=") or "Camera" in raw:
            return raw
        # Linux V4L2 path like /dev/video0
        if raw.startswith("/dev/video"):
            return raw
        # numeric index
        if raw.isdigit():
            idx = int(raw)
            if sys.platform == "win32":
                return f"video=video{idx}"
            return f"/dev/video{idx}"
        return raw

    def _pyav_capture_worker(self):
        try:
            import av
        except ImportError:
            self.get_logger().error(
                "backend:=pyav needs 'pip install av'.  Falling back to OpenCV."
            )
            self._opencv_capture_worker()
            return

        device_name = self._pyav_device_name()
        # Force v4l2 pixel format to MJPG before opening (YUYV caps at 5-10 fps)
        if self.video_format == "v4l2" and device_name.startswith("/dev/video"):
            import subprocess
            try:
                subprocess.run(
                    ["v4l2-ctl", "-d", device_name,
                     "--set-fmt-video", f"width={self.width},height={self.height},pixelformat=MJPG",
                     "--set-parm", str(int(self.fps))],
                    capture_output=True, timeout=2,
                )
            except Exception:
                pass
        options = {
            "video_size": f"{self.width}x{self.height}",
            "framerate": str(int(self.fps)),
            "pixel_format": "mjpeg",
            "vcodec": "mjpeg",
            "rtbufsize": self.rtbufsize,
        }

        self.get_logger().info(
            f"PyAV opening: device={device_name} format={self.video_format} "
            f"size={self.width}x{self.height} fps={int(self.fps)} rtbufsize={self.rtbufsize}"
        )

        try:
            container = av.open(device_name, format=self.video_format, options=options)
        except Exception as exc:
            self.get_logger().error(
                f"PyAV failed to open {device_name} (format={self.video_format}): {exc}"
            )
            self.get_logger().info("falling back to OpenCV...")
            self._opencv_capture_worker()
            return

        stream = next((s for s in container.streams if s.type == "video"), None)
        if stream is None:
            self.get_logger().error("PyAV found no video stream")
            container.close()
            return

        self.get_logger().info("PyAV camera opened successfully")
        try:
            for packet in container.demux(stream):
                if not self.running or not rclpy.ok():
                    break
                for frame in packet.decode():
                    if not self.running or not rclpy.ok():
                        break
                    self._set_latest_frame(frame.to_ndarray(format="bgr24"))
        finally:
            container.close()

    # ------------------------------------------------------------------
    #  OpenCV capture (fallback)
    # ------------------------------------------------------------------
    def _opencv_capture_worker(self):
        cap = self._open_opencv_camera()
        if not cap.isOpened():
            self.get_logger().error(f"failed to open camera with OpenCV: {self.device}")
            return

        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)

        actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        self.get_logger().info(
            "OpenCV camera opened: actual size=%.0fx%.0f fps=%.1f" % (actual_w, actual_h, actual_fps)
        )

        try:
            while self.running and rclpy.ok():
                ok, frame = cap.read()
                if not ok or frame is None:
                    time.sleep(0.02)
                    continue
                self._set_latest_frame(frame)
        finally:
            cap.release()

    def _open_opencv_camera(self):
        candidates = []
        if isinstance(self.device, str):
            candidates.append(self.device)
            index = self._opencv_device_index(self.device)
            if index is not None:
                candidates.append(index)
        else:
            candidates.append(self.device)

        for candidate in candidates:
            cap = cv2.VideoCapture(candidate, cv2.CAP_V4L2)
            if cap.isOpened():
                self.get_logger().info(f"OpenCV opened camera candidate={candidate!r} with CAP_V4L2")
                return cap
            cap.release()

        for candidate in candidates:
            cap = cv2.VideoCapture(candidate)
            if cap.isOpened():
                self.get_logger().info(f"OpenCV opened camera candidate={candidate!r} with default backend")
                return cap
            cap.release()

        return cv2.VideoCapture()

    @staticmethod
    def _opencv_device_index(device):
        if isinstance(device, str) and device.startswith("/dev/video"):
            suffix = device.removeprefix("/dev/video")
            if suffix.isdigit():
                return int(suffix)
        return None

    # ------------------------------------------------------------------
    #  frame store
    # ------------------------------------------------------------------
    def _set_latest_frame(self, frame: np.ndarray):
        with self.frame_lock:
            self.latest_frame = frame
            self.frame_count += 1

    # ------------------------------------------------------------------
    #  publish  (called by ROS timer)
    # ------------------------------------------------------------------
    def _publish_latest_frame(self):
        with self.frame_lock:
            frame = None if self.latest_frame is None else self.latest_frame.copy()
            frame_count = self.frame_count

        if frame is None:
            return

        left, right = self._split_stereo_frame(frame)
        stamp = self.get_clock().now().to_msg()

        if self.raw_pub is not None:
            self.raw_pub.publish(self._to_image_msg(frame, stamp, self.frame_id_raw))
        if left is not None and self.left_pub is not None:
            self.left_pub.publish(self._to_image_msg(left, stamp, self.frame_id_left))
        if right is not None and self.right_pub is not None:
            self.right_pub.publish(self._to_image_msg(right, stamp, self.frame_id_right))

        # rectified + disparity
        rect_l, rect_r = None, None
        if self._rect_initialized and left is not None and right is not None:
            rect_l = cv2.remap(left, self._map_lx, self._map_ly, cv2.INTER_LINEAR)
            rect_r = cv2.remap(right, self._map_rx, self._map_ry, cv2.INTER_LINEAR)

            if self.rect_left_pub is not None:
                self.rect_left_pub.publish(self._to_image_msg(rect_l, stamp, self.frame_id_left))
            if self.rect_right_pub is not None:
                self.rect_right_pub.publish(self._to_image_msg(rect_r, stamp, self.frame_id_right))

            if self.disparity_pub is not None and self._stereo_matcher is not None:
                gray_l = cv2.cvtColor(rect_l, cv2.COLOR_BGR2GRAY)
                gray_r = cv2.cvtColor(rect_r, cv2.COLOR_BGR2GRAY)
                disp = self._stereo_matcher.compute(gray_l, gray_r).astype(np.float32) / 16.0
                disp[disp <= 0] = 0
                disp_msg = self.bridge.cv2_to_imgmsg(disp.astype(np.uint16), encoding="mono16")
                disp_msg.header.stamp = stamp
                disp_msg.header.frame_id = self.frame_id_left
                self.disparity_pub.publish(disp_msg)

        if self.show_image:
            self._show_frame(frame, left, right, rect_l, rect_r)

        # stats
        now = time.monotonic()
        if now - self.last_stat_time >= 5.0:
            fps = (frame_count - self.last_frame_count) / (now - self.last_stat_time)
            self.get_logger().info(f"camera capture fps={fps:.1f}, frame shape={frame.shape}")
            self.last_frame_count = frame_count
            self.last_stat_time = now

    def _split_stereo_frame(self, frame) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        h, w = frame.shape[:2]
        expected_width = self.left_width + self.right_width
        if w < expected_width:
            self.get_logger().warn(
                f"frame width {w} is smaller than configured stereo split width {expected_width}",
                throttle_duration_sec=5.0,
            )
            return frame, None
        left = frame[:, : self.left_width]
        right = frame[:, self.left_width : self.left_width + self.right_width]
        return left, right

    def _to_image_msg(self, frame, stamp, frame_id):
        msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        msg.header.stamp = stamp
        msg.header.frame_id = frame_id
        return msg

    def _show_frame(self, frame, left, right, rect_l, rect_r):
        if (self.publish_rect or self.publish_disparity) and rect_l is not None and rect_r is not None:
            # draw horizontal lines for rectification verification
            for y in range(0, self.height, 40):
                cv2.line(rect_l, (0, y), (self.left_width, y), (0, 255, 0), 1)
                cv2.line(rect_r, (0, y), (self.right_width, y), (0, 255, 0), 1)
            show = cv2.hconcat([rect_l, rect_r])
        elif left is not None and right is not None:
            show = cv2.hconcat([left, right])
        else:
            show = frame
        show = self._resize_for_preview(show)
        if not self._preview_window_created:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, show.shape[1], show.shape[0])
            self._preview_window_created = True
        cv2.imshow(self.window_name, show)
        cv2.waitKey(1)

    def _resize_for_preview(self, image):
        h, w = image.shape[:2]
        scale = self.display_scale if self.display_scale > 0.0 else 1.0
        if self.display_max_width > 0:
            scale = min(scale, float(self.display_max_width) / float(w))
        if self.display_max_height > 0:
            scale = min(scale, float(self.display_max_height) / float(h))
        if scale >= 0.999:
            return image
        new_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
        return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)

    def destroy_node(self):
        self.running = False
        if self.show_image:
            cv2.destroyAllWindows()
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = StereoCameraNode()
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
