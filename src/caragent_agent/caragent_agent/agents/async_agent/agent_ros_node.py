"""ROS2 entry point for the CarAgent async agent with embedded web UI."""

from __future__ import annotations

import os
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Image as ROSImage
from std_msgs.msg import String

from caragent_agent.io_adapters import (
    describe_image_for_navigation,
    normalize_language,
    prepare_user_message_for_agent,
    translate_text_for_user,
)


def _start_web_server(agent, host: str, port: int, thread_id: str) -> None:
    from caragent_agent.scripts.demo_ui.async_agent_web_demo import (
        AsyncAgentWebApp,
        AsyncAgentWebHandler,
    )

    app = AsyncAgentWebApp(agent, thread_id)
    handler = type("BoundHandler", (AsyncAgentWebHandler,), {"app": app})
    server = ThreadingHTTPServer((host, port), handler)
    print(f"CarAgent web UI is ready at http://{host}:{port}")
    server.serve_forever()


class CarAgentROSNode(Node):
    def __init__(self) -> None:
        super().__init__("caragent_agent")
        self.declare_parameter("config_file", "")
        config_file = self.get_parameter("config_file").get_parameter_value().string_value
        if not config_file:
            workspace = Path(os.environ.get("CARAGENT_WORKSPACE", "/home/car/caragent_ws"))
            config_file = str(workspace / "src" / "caragent_agent" / "config" / "config.yaml")
        os.environ["CARAGENT_BASE_CONFIG_FILE"] = config_file

        from caragent_agent.agents.async_agent import AsyncAgent
        from caragent_agent.config.config import config
        from caragent_agent.controller.nav2.nav2_controller import Nav2Controller
        from caragent_agent.impression_graph.scene_memory import SceneMemory

        scene_cfg = config.get("scene_memory", {})
        nav_cfg = config.get("navigation", {})
        agent_cfg = config.get("agent", {})
        self._io_cfg = config.get("io", {})

        dataset_dir = scene_cfg.get("dataset_dir") or config.get("paths", {}).get("default_dataset_dir")
        if not dataset_dir:
            raise ValueError("scene_memory.dataset_dir is required.")

        self.get_logger().info(f"Loading scene memory: {dataset_dir}")
        self.scene_memory = SceneMemory(dataset_dir=dataset_dir, device=scene_cfg.get("device"))
        self.controller = Nav2Controller(
            self,
            action_name=nav_cfg.get("action_name", "navigate_to_pose"),
            global_frame=nav_cfg.get("global_frame", "map"),
            base_frame=nav_cfg.get("base_frame", "base_link"),
            dry_run=bool(nav_cfg.get("dry_run_navigation", False)),
            camera_topic=nav_cfg.get("camera_topic", "/stereo/left/image_raw"),
            right_image_topic=nav_cfg.get("right_image_topic", "/stereo/right/image_raw"),
            odom_topic=nav_cfg.get("odom_topic", "/odom"),
            scan_topic=nav_cfg.get("scan_topic", "/scan"),
            map_topic=nav_cfg.get("map_topic", "/global_costmap/costmap"),
            arrival_tolerance_m=float(nav_cfg.get("arrival_tolerance_m", 0.25)),
            enable_rotation_takeover=bool(nav_cfg.get("enable_rotation_takeover", True)),
            rotation_policy=nav_cfg.get("rotation_policy", "left_only"),
            pre_align_enabled=bool(nav_cfg.get("pre_align_enabled", True)),
            final_align_enabled=bool(nav_cfg.get("final_align_enabled", True)),
            yaw_tolerance_deg=float(nav_cfg.get("yaw_tolerance_deg", 4.0)),
            settle_time_sec=float(nav_cfg.get("settle_time_sec", 0.7)),
            fast_omega=float(nav_cfg.get("fast_omega", 3.40)),
            mid_omega=float(nav_cfg.get("mid_omega", 2.50)),
            slow_omega=float(nav_cfg.get("slow_omega", 1.50)),
            fast_threshold_deg=float(nav_cfg.get("fast_threshold_deg", 20.0)),
            mid_threshold_deg=float(nav_cfg.get("mid_threshold_deg", 10.0)),
            rotation_timeout_sec=float(nav_cfg.get("rotation_timeout_sec", 15.0)),
            rotation_loop_rate_hz=float(nav_cfg.get("rotation_loop_rate_hz", 20.0)),
            right_turn_shortcut_deg=float(nav_cfg.get("right_turn_shortcut_deg", 90.0)),
            localization_handoff_gate_enabled=bool(
                nav_cfg.get("localization_handoff_gate_enabled", True)
            ),
            localization_handoff_settle_sec=float(
                nav_cfg.get("localization_handoff_settle_sec", 1.0)
            ),
            localization_handoff_timeout_sec=float(
                nav_cfg.get("localization_handoff_timeout_sec", 2.0)
            ),
            localization_handoff_max_translation_m=float(
                nav_cfg.get("localization_handoff_max_translation_m", 0.12)
            ),
            localization_handoff_max_yaw_deg=float(
                nav_cfg.get("localization_handoff_max_yaw_deg", 8.0)
            ),
            localization_handoff_sensor_max_age_sec=float(
                nav_cfg.get("localization_handoff_sensor_max_age_sec", 1.5)
            ),
            simulation_mode=bool(nav_cfg.get("simulation_mode", False)),
            simulation_navigation_delay_sec=float(
                nav_cfg.get("simulation_navigation_delay_sec", 30.0)
            ),
            simulation_navigation_delay_per_meter_sec=float(
                nav_cfg.get("simulation_navigation_delay_per_meter_sec", 0.0)
            ),
            simulation_initial_position=nav_cfg.get("simulation_initial_position"),
            simulation_initial_yaw_deg=float(
                nav_cfg.get("simulation_initial_yaw_deg", 0.0)
            ),
        )
        self.agent = AsyncAgent(
            scene_memory=self.scene_memory,
            controller=self.controller,
            controller_type=nav_cfg.get("controller_type", "nav2"),
            is_navigation_mode=bool(agent_cfg.get("is_navigation_mode", True)),
            use_multi_agents=bool(agent_cfg.get("use_multi_agents", True)),
            num_background_workers=int(agent_cfg.get("num_background_workers", 2)),
        )

        self.create_subscription(String, "/caragent_agent/command", self._on_command, 10)
        self.create_subscription(
            ROSImage, "/caragent_agent/query_image", self._on_query_image, 10
        )
        self._response_pub = self.create_publisher(String, "/caragent_agent/response", 10)
        self._photo_pub = self.create_publisher(ROSImage, "/caragent_agent/photo", 10)
        self._photo_response_pub = self.create_publisher(
            String, "/caragent_agent/photo_response", 10
        )
        self._image_description_pub = self.create_publisher(
            String, "/caragent_agent/image_description", 10
        )

        web_cfg = config.get("web_ui", {})
        self._thread_id = str(web_cfg.get("thread_id", "caragent"))
        if web_cfg.get("enabled", True):
            host = str(web_cfg.get("host", "0.0.0.0"))
            port = int(web_cfg.get("port", 8123))
            threading.Thread(
                target=_start_web_server,
                args=(self.agent, host, port, self._thread_id),
                daemon=True,
            ).start()

        self.get_logger().info(
            f"CarAgent agent ready with {len(self.scene_memory.keyframe_nodes)} keyframes."
        )

    def _on_command(self, msg: String) -> None:
        user_input = str(msg.data or "").strip()
        if not user_input:
            return
        threading.Thread(target=self._run_agent, args=(user_input,), daemon=True).start()

    def _run_agent(self, user_input: str) -> None:
        try:
            agent_input = prepare_user_message_for_agent(
                user_input,
                input_language=str(self._io_cfg.get("input_language", "zh")),
                output_language=str(self._io_cfg.get("output_language", "zh")),
                translate_boundary=bool(self._io_cfg.get("translate_boundary", True)),
            )
            result = self.agent.run_message_turn(
                agent_input,
                self._thread_id,
                "user",
            )
            response = self._response_text_from_turn_result(result)
            response = self._translate_response_for_user(response, user_input)
        except Exception as exc:
            self.get_logger().error(f"Agent command failed: {exc}")
            response = f"Agent command failed: {exc}"
        self._response_pub.publish(String(data=response))

    def _response_text_from_turn_result(self, result: object) -> str:
        if isinstance(result, dict):
            text = str(result.get("turn_response_text") or "").strip()
            if text:
                return text
            candidates = result.get("answer_candidates")
            if isinstance(candidates, list) and candidates:
                return str(candidates[-1])
            items = result.get("response_items")
            if isinstance(items, list) and items:
                last = items[-1]
                if isinstance(last, dict) and last.get("response_text"):
                    return str(last.get("response_text"))
            return "任务已结束，但没有生成可展示的回答。请查看日志中的 turn_response 字段。"
        return str(result)

    def _translate_response_for_user(self, response: str, user_input: str) -> str:
        output_language = str(self._io_cfg.get("output_language", "zh"))
        target = normalize_language(output_language, fallback="zh")
        if target == "en":
            return response
        return translate_text_for_user(response, target_language=target)

    def _on_query_image(self, msg: ROSImage) -> None:
        threading.Thread(target=self._describe_query_image, args=(msg,), daemon=True).start()

    def _describe_query_image(self, msg: ROSImage) -> None:
        try:
            image = self._ros_image_to_pil(msg)
            description = describe_image_for_navigation(image)
            self._image_description_pub.publish(
                String(
                    data=json_like(
                        {
                            "status": "ok",
                            "description": description,
                            "hint": (
                                "Send this description to /caragent_agent/command "
                                "to search or navigate by image content."
                            ),
                        }
                    )
                )
            )
        except Exception as exc:
            self.get_logger().exception(f"Image query failed: {exc}")
            self._image_description_pub.publish(
                String(
                    data=json_like(
                        {
                            "status": "error",
                            "error": str(exc),
                        }
                    )
                )
            )

    def _publish_photo_if_requested(self, user_input: str) -> None:
        self._photo_response_pub.publish(
            String(
                data=json_like(
                    {
                        "status": "deprecated",
                        "summary": (
                            "Automatic keyword-based photo publishing is disabled. "
                            "Ask the Agent to take a photo so it can use capture_current_view."
                        ),
                    }
                )
            )
        )

    def _pil_to_ros_image(self, image) -> ROSImage:
        rgb = image.convert("RGB")
        array = np.asarray(rgb, dtype=np.uint8)
        msg = ROSImage()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "camera"
        msg.height = int(array.shape[0])
        msg.width = int(array.shape[1])
        msg.encoding = "rgb8"
        msg.is_bigendian = 0
        msg.step = int(array.shape[1] * 3)
        msg.data = array.tobytes()
        return msg

    def _ros_image_to_pil(self, msg: ROSImage):
        from PIL import Image

        channels = 3
        if msg.encoding in {"rgba8", "bgra8"}:
            channels = 4
        elif msg.encoding in {"mono8", "8UC1"}:
            channels = 1

        array = np.frombuffer(msg.data, dtype=np.uint8).reshape(
            msg.height, msg.width, channels
        )
        if msg.encoding == "bgr8":
            array = array[:, :, ::-1]
        elif msg.encoding == "bgra8":
            array = array[:, :, [2, 1, 0, 3]]
        if channels == 1:
            return Image.fromarray(array[:, :, 0], mode="L").convert("RGB")
        return Image.fromarray(array[:, :, :3]).convert("RGB")


def json_like(payload: dict) -> str:
    import json

    return json.dumps(payload, ensure_ascii=False)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CarAgentROSNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
