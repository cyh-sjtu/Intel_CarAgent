"""Navigation tool that dispatches selected keyframes to Nav2."""

from __future__ import annotations

import math
from typing import Any

import networkx as nx
import numpy as np

from caragent_agent.agents.tools.base.tool_base import ToolBase


class NavigationTool(ToolBase):
    def __init__(self, controller: Any | None = None):
        super().__init__(
            name="go_to_keyframe",
            description="""
                Plans and executes a path to a specified keyframe node.

                Args:
                    keyframe_node_id (int): The ID of the target keyframe node.
            """,
            capability_tags=("navigation", "background_unsafe"),
        )
        self.controller = controller

    def execute(self, keyframe_node_id: int):
        if self.controller is None:
            return self.blocked(
                f"Controller is unavailable, so navigation to keyframe {keyframe_node_id} cannot be dispatched.",
                error={"code": "controller_unavailable", "message": "Controller reference is missing."},
                provenance={"source_type": "controller"},
            )

        try:
            current_state = self.controller.get_current_state()
            current_position = (current_state or {}).get("position")
        except Exception as exc:
            return self.error_result(
                f"Controller raised an exception before navigation to keyframe {keyframe_node_id} could be planned.",
                error={"code": "controller_exception", "message": str(exc)},
                provenance={"source_type": "controller"},
            )

        if current_position is None:
            return self.blocked(
                f"Current position is unavailable, so navigation to keyframe {keyframe_node_id} cannot be planned.",
                data={"current_state": self.to_jsonable(current_state)},
                error={"code": "missing_current_position", "message": "Controller did not provide the current position."},
                provenance={"source_type": "controller"},
            )

        try:
            current_position = self._to_xyz_list(current_position)
            path = self._plan_path_with_keyframe_graph(current_position, int(keyframe_node_id))
        except Exception as exc:
            return self.error_result(
                f"Navigation planning failed for keyframe {keyframe_node_id}.",
                data={"current_position": self.to_jsonable(current_position), "target_keyframe_id": int(keyframe_node_id)},
                error={"code": "navigation_planning_failed", "message": str(exc)},
                provenance={"source_type": "scene_memory"},
            )

        if not path:
            return self.blocked(
                f"No feasible path was found to keyframe {keyframe_node_id}.",
                data={"current_position": current_position, "target_keyframe_id": int(keyframe_node_id), "planned_path": []},
                error={"code": "path_not_found", "message": "Path planner returned no route."},
                provenance={"source_type": "scene_memory"},
            )

        try:
            self.controller.update_path(path)
            nav_status = self.controller.get_status() if hasattr(self.controller, "get_status") else "dispatched"
        except Exception as exc:
            return self.error_result(
                f"Controller failed to accept the planned path to keyframe {keyframe_node_id}.",
                data={"current_position": current_position, "target_keyframe_id": int(keyframe_node_id), "planned_path": self.to_jsonable(path)},
                error={"code": "controller_update_failed", "message": str(exc)},
                provenance={"source_type": "controller"},
            )

        return self.ok(
            f"Dispatched navigation to keyframe {keyframe_node_id}.",
            data={
                "target_keyframe_id": int(keyframe_node_id),
                "target_position": self._get_target_position(int(keyframe_node_id)),
                "current_position": current_position,
                "planned_path": self.to_jsonable(path),
                "path_waypoint_count": len(path),
                "navigation_status": nav_status,
            },
            provenance={"source_type": "controller"},
        )

    def _plan_path_with_keyframe_graph(
        self,
        current_position: list[float],
        target_keyframe_node_id: int,
    ) -> list[list[float]]:
        if target_keyframe_node_id not in self.scene_memory.keyframe_nodes:
            raise KeyError(f"Unknown keyframe id: {target_keyframe_node_id}")

        start_node_id = self.scene_memory.find_nearest_node(current_position)
        try:
            kf_path = nx.shortest_path(
                self.scene_memory.keyframe_graph,
                source=start_node_id,
                target=target_keyframe_node_id,
                weight="weight",
            )
        except nx.NetworkXNoPath:
            return []

        final_path = [
            self._to_xyz_list(self.scene_memory.keyframe_nodes[node_id].position)
            for node_id in kf_path
        ]
        if not final_path:
            return []

        target_node = self.scene_memory.keyframe_nodes[target_keyframe_node_id]
        yaw = _yaw_deg_from_quaternion(target_node.orientation)
        final_path[-1] = self._to_xyz_list(target_node.position)
        final_path[-1].append(float(yaw))
        return final_path

    def _get_target_position(self, keyframe_node_id: int) -> list[float] | None:
        try:
            return self._to_xyz_list(self.scene_memory.keyframe_nodes[keyframe_node_id].position)
        except Exception:
            return None

    def _to_xyz_list(self, pos) -> list[float]:
        if isinstance(pos, dict):
            return [float(pos.get("x", 0.0)), float(pos.get("y", 0.0)), float(pos.get("z", 0.0))]

        arr = np.asarray(pos, dtype=float).reshape(-1)
        if arr.size < 2:
            raise ValueError(f"Position must contain at least x and y: {pos}")
        z = arr[2] if arr.size >= 3 else 0.0
        return [float(arr[0]), float(arr[1]), float(z)]


class NavigationToPositionTool(ToolBase):
    def __init__(self, controller: Any | None = None):
        super().__init__(
            name="go_to_position",
            description="""
                Dispatches navigation to a concrete map-frame position.

                Args:
                    x (float): Target x in map frame.
                    y (float): Target y in map frame.
                    z (float): Target z in map frame, defaults to 0.
                    yaw_deg (float): Target yaw in degrees, defaults to 0.
            """,
            capability_tags=("navigation", "background_unsafe"),
        )
        self.controller = controller

    def execute(self, x: float, y: float, z: float = 0.0, yaw_deg: float = 0.0):
        if self.controller is None:
            return self.blocked(
                "Controller is unavailable, so navigation to position cannot be dispatched.",
                error={"code": "controller_unavailable", "message": "Controller reference is missing."},
                provenance={"source_type": "controller"},
            )
        waypoint = [float(x), float(y), float(z), float(yaw_deg)]
        try:
            self.controller.update_path([waypoint])
            nav_status = self.controller.get_status() if hasattr(self.controller, "get_status") else "dispatched"
        except Exception as exc:
            return self.error_result(
                "Controller failed to accept the map-frame position goal.",
                data={"target_position": waypoint[:3], "target_yaw_deg": waypoint[3]},
                error={"code": "controller_update_failed", "message": str(exc)},
                provenance={"source_type": "controller"},
            )
        return self.ok(
            f"Dispatched navigation to map position x={waypoint[0]:.2f}, y={waypoint[1]:.2f}.",
            data={
                "target_position": waypoint[:3],
                "target_yaw_deg": waypoint[3],
                "planned_path": [waypoint],
                "path_waypoint_count": 1,
                "navigation_status": nav_status,
            },
            provenance={"source_type": "controller"},
        )


def _yaw_deg_from_quaternion(quaternion) -> float:
    q = np.asarray(quaternion, dtype=float).reshape(-1)
    if q.size < 4:
        return 0.0
    x, y, z, w = q[:4]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.degrees(math.atan2(siny_cosp, cosy_cosp))
