from caragent_agent.agents.tools.base.tool_base import ToolBase

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

        position = normalized_state.get("position")
        orientation = normalized_state.get("orientation")
        status = str(normalized_state.get("status") or "").strip()

        if position is None and orientation is None and not status:
            return self.partial(
                "Controller returned an empty current-state payload.",
                data=normalized_state,
                error={
                    "code": "empty_state_payload",
                    "message": "No position, orientation, or status was available.",
                },
                provenance={"source_type": "controller"},
            )

        return self.ok(
            "Retrieved the current controller state.",
            data=normalized_state,
            provenance={"source_type": "controller"},
        )
