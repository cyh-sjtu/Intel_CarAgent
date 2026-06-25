"""Audit Nav2Controller workflow simulation mode without LLM/API dependencies."""

from __future__ import annotations

import time

import rclpy
from geometry_msgs.msg import Twist

from caragent_agent.agents.tools.navigation.navigator import NavigationToPositionTool
from caragent_agent.controller.nav2.nav2_controller import Nav2Controller


def _assert_close(actual: float, expected: float, *, tol: float = 1e-3) -> None:
    if abs(float(actual) - float(expected)) > tol:
        raise AssertionError(f"expected {expected}, got {actual}")


def _assert_position(actual: list[float] | None, expected: list[float]) -> None:
    if not isinstance(actual, list) or len(actual) < 3:
        raise AssertionError(f"position unavailable: {actual}")
    for index, value in enumerate(expected):
        _assert_close(actual[index], value)


def _wait_until(node, predicate, *, timeout_sec: float = 2.0, poll_sec: float = 0.05) -> None:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.0)
        if predicate():
            return
        time.sleep(poll_sec)
    raise AssertionError("condition did not become true before timeout")


def main() -> None:
    rclpy.init()
    node = rclpy.create_node("nav2_simulation_audit")
    cmd_vel_messages: list[Twist] = []
    node.create_subscription(Twist, "/cmd_vel", cmd_vel_messages.append, 10)
    try:
        controller = Nav2Controller(
            node,
            simulation_mode=True,
            simulation_navigation_delay_sec=0.25,
            simulation_initial_position=[0.0, 0.0, 0.0],
            simulation_initial_yaw_deg=5.0,
            enable_rotation_takeover=True,
        )

        initial = controller.get_current_state()
        assert initial["source"] == "simulation"
        _assert_position(initial["position"], [0.0, 0.0, 0.0])
        _assert_close(initial["yaw_deg"], 5.0)

        controller.update_path([[1.0, 2.0, 0.0, 30.0]])
        assert controller.get_status() == "sim_navigating"
        dispatch_msg = controller.check_for_new_messages()
        assert "Simulated navigation dispatched" in dispatch_msg
        _wait_until(node, lambda: controller.get_status() == "arrived", timeout_sec=1.5)
        arrived = controller.get_current_state()
        _assert_position(arrived["position"], [1.0, 2.0, 0.0])
        _assert_close(arrived["yaw_deg"], 30.0)
        arrival_msg = controller.check_for_new_messages()
        assert "Arrived at destination [1.000, 2.000, 0.000]" in arrival_msg

        controller.update_path([[5.0, 5.0, 0.0, 90.0]])
        time.sleep(0.05)
        controller.update_path([[2.0, -1.0, 0.0, -45.0]])
        _wait_until(node, lambda: controller.get_status() == "arrived", timeout_sec=1.5)
        stale_checked = controller.get_current_state()
        _assert_position(stale_checked["position"], [2.0, -1.0, 0.0])
        _assert_close(stale_checked["yaw_deg"], -45.0)
        stale_arrival_msg = controller.check_for_new_messages()
        assert "Arrived at destination [2.000, -1.000, 0.000]" in stale_arrival_msg
        assert "5.000, 5.000" not in stale_arrival_msg

        before_cancel = controller.get_current_state()["position"]
        controller.update_path([[8.0, 8.0, 0.0, 180.0]])
        controller.cancel_navigation()
        _wait_until(node, lambda: controller.get_status() == "cancelled", timeout_sec=0.5)
        time.sleep(0.3)
        rclpy.spin_once(node, timeout_sec=0.0)
        assert controller.get_status() == "cancelled"
        after_cancel = controller.get_current_state()["position"]
        _assert_position(after_cancel, before_cancel)
        cancel_msg = controller.check_for_new_messages()
        assert "Navigation cancelled" in cancel_msg

        nav_tool = NavigationToPositionTool(controller)
        tool_result = nav_tool.execute(x=-3.0, y=0.5, z=0.0, yaw_deg=15.0)
        assert tool_result["status"] == "ok"
        assert tool_result["data"]["navigation_status"] == "sim_navigating"
        controller.check_for_new_messages()
        _wait_until(node, lambda: controller.get_status() == "arrived", timeout_sec=1.5)
        tool_state = controller.get_current_state()
        _assert_position(tool_state["position"], [-3.0, 0.5, 0.0])
        _assert_close(tool_state["yaw_deg"], 15.0)
        tool_arrival_msg = controller.check_for_new_messages()
        assert "Arrived at destination [-3.000, 0.500, 0.000]" in tool_arrival_msg

        rclpy.spin_once(node, timeout_sec=0.0)
        if cmd_vel_messages:
            raise AssertionError(f"simulation mode published cmd_vel messages: {len(cmd_vel_messages)}")

        print(
            {
                "status": "ok",
                "initial_source": initial["source"],
                "final_position": controller.get_current_state()["position"],
                "final_status": controller.get_status(),
                "cmd_vel_messages": len(cmd_vel_messages),
                "checks": [
                    "initial_state",
                    "delayed_arrival",
                    "stale_goal_ignored",
                    "cancel_without_arrival",
                    "go_to_position_tool",
                    "no_cmd_vel_published",
                ],
            }
        )
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
