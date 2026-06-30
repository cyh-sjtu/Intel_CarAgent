from datetime import datetime
from typing import Any

from caragent_agent.agents.tools.base.tool_base import ToolBase


def _compact_current_state(raw_state: dict[str, Any]) -> dict[str, Any]:
    """Return the LLM-facing state view without maps/costmaps/raw grids."""

    omitted_count = sum(
        1
        for key in ("occupancy_grid", "map", "costmap", "grid", "scan", "raw_scan")
        if key in raw_state
    )
    compact = {
        "position": raw_state.get("position"),
        "orientation": raw_state.get("orientation"),
        "status": raw_state.get("status"),
        "source": raw_state.get("source"),
        "timestamp": raw_state.get("timestamp")
        or datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
    }
    if raw_state.get("error"):
        compact["error"] = str(raw_state.get("error"))
    if raw_state.get("tf_error"):
        compact["tf_error"] = str(raw_state.get("tf_error"))
    if omitted_count:
        compact["large_state_fields_omitted"] = True
        compact["omitted_large_field_count"] = omitted_count
        compact["full_state_note"] = (
            "Large mapping/perception fields are internal-only and are not included "
            "in the LLM-facing tool result."
        )
    return {key: value for key, value in compact.items() if value is not None}

class Get_Current_State_Tool(ToolBase):
    def __init__(self, controller):
        super().__init__(
            name="get_current_state",
            description="""
                Retrieves the current state of the robot.

                Returns:
                    Dict[str, Any]: A dictionary representing the current state of the robot.
                    The dictionary contains keys: 'position', 'orientation', 'status'.
                    'position' should be a list of floats [x, y, z] or None when not available.
                    'orientation' should be a list of floats [x, y, z, w] or None when not available.
                    'status' should be a string indicating the current status.
                """,
            capability_tags=("live_state", "background_unsafe"),
        )
        self.controller = controller

    def execute(self) -> dict:
        if self.controller is None:
            return self.blocked(
                "Controller is unavailable, so current state cannot be queried.",
                error={
                    "code": "controller_unavailable",
                    "message": "Controller reference is missing.",
                },
                provenance={"source_type": "controller"},
            )

        try:
            raw_state = self.controller.get_current_state()
        except Exception as exc:
            return self.error_result(
                "Controller raised an exception while querying current state.",
                error={
                    "code": "controller_exception",
                    "message": str(exc),
                },
                provenance={"source_type": "controller"},
            )

        normalized_state = self.to_jsonable(raw_state)
        if not isinstance(normalized_state, dict):
            return self.error_result(
                "Controller returned an invalid current-state payload.",
                data={"raw_state": normalized_state},
                error={
                    "code": "invalid_state_payload",
                    "message": "Current state payload is not a JSON-safe object.",
                },
                provenance={"source_type": "controller"},
            )

        compact_state = _compact_current_state(normalized_state)
        position = compact_state.get("position")
        orientation = compact_state.get("orientation")
        status = str(compact_state.get("status") or "").strip()

        if position is None and orientation is None and not status:
            return self.partial(
                "Controller returned an empty current-state payload.",
                data=compact_state,
                error={
                    "code": "empty_state_payload",
                    "message": "No position, orientation, or status was available.",
                },
                provenance={"source_type": "controller"},
            )

        return self.ok(
            "Retrieved the current controller state.",
            data=compact_state,
            provenance={"source_type": "controller"},
        )
