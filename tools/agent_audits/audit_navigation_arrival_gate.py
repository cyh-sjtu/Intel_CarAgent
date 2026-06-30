"""Local audit for navigation-arrival dedupe and ownership checks."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys
import types


def _repo_src_root() -> Path:
    here = Path(__file__).resolve()
    for parent in (here.parent, *here.parents):
        candidate = parent / "ros2" / "caragent_ws" / "src" / "caragent_agent"
        if (candidate / "caragent_agent").exists():
            return candidate
        board_candidate = parent / "src" / "caragent_agent"
        if (board_candidate / "caragent_agent").exists():
            return board_candidate
    raise FileNotFoundError("Could not locate caragent_agent source root")


def _install_import_stubs() -> None:
    if "langchain_core.messages" not in sys.modules:
        langchain_core = types.ModuleType("langchain_core")
        messages = types.ModuleType("langchain_core.messages")

        class BaseMessage:  # pragma: no cover
            def __init__(self, content=None, **kwargs):
                self.content = content

        class AIMessage(BaseMessage):  # pragma: no cover
            pass

        class HumanMessage(BaseMessage):  # pragma: no cover
            pass

        class ToolMessage(BaseMessage):  # pragma: no cover
            def __init__(self, content=None, tool_call_id=None, **kwargs):
                super().__init__(content=content, **kwargs)
                self.tool_call_id = tool_call_id

        class SystemMessage(BaseMessage):  # pragma: no cover
            pass

        messages.BaseMessage = BaseMessage
        messages.AIMessage = AIMessage
        messages.HumanMessage = HumanMessage
        messages.ToolMessage = ToolMessage
        messages.SystemMessage = SystemMessage
        sys.modules.setdefault("langchain_core", langchain_core)
        sys.modules["langchain_core.messages"] = messages

    if "langchain_core.tools" not in sys.modules:
        tools = types.ModuleType("langchain_core.tools")

        class BaseTool:  # pragma: no cover
            pass

        class StructuredTool(BaseTool):  # pragma: no cover
            pass

        tools.BaseTool = BaseTool
        tools.StructuredTool = StructuredTool
        sys.modules["langchain_core.tools"] = tools

    if "langchain_core.language_models" not in sys.modules:
        language_models = types.ModuleType("langchain_core.language_models")

        class BaseChatModel:  # pragma: no cover
            pass

        language_models.BaseChatModel = BaseChatModel
        sys.modules["langchain_core.language_models"] = language_models

    if "langgraph.graph.message" not in sys.modules:
        langgraph = types.ModuleType("langgraph")
        graph = types.ModuleType("langgraph.graph")
        graph_message = types.ModuleType("langgraph.graph.message")

        def add_messages(left, right):  # pragma: no cover
            return (left or []) + (right or [])

        graph_message.add_messages = add_messages
        sys.modules.setdefault("langgraph", langgraph)
        sys.modules["langgraph.graph"] = graph
        sys.modules["langgraph.graph.message"] = graph_message


class FakeLogger:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def log_foreground(self, message: str) -> None:
        self.lines.append(str(message))


def _base_task(status: str = "waiting") -> dict:
    return {
        "task_id": 2,
        "task_type": "navigation_action",
        "description": "Navigate to keyframe 5",
        "status": status,
        "next_task_id": 3,
        "plan_id": "plan_a",
        "user_input_id": "input_a",
        "wait_for_event": "navigation_arrived" if status == "waiting" else None,
        "result": [
            {
                "summary": "navigation command issued successfully; waiting for arrival event.",
                "tool_name": "go_to_keyframe",
            }
        ],
    }


def _base_state(status: str = "waiting") -> dict:
    task = _base_task(status=status)
    state = {
        "tasks": {2: task},
        "current_task_id": 2,
        "current_plan_id": "plan_a",
        "processed_event_ids": [],
        "events": [],
        "active_navigation": {
            "navigation_token": "nav_a",
            "task_id": 2,
            "plan_id": "plan_a",
            "user_input_id": "input_a",
            "description": "Navigate to keyframe 5",
            "destination_position": [1.0, 2.0, 0.0],
        },
        "pending_navigation": {
            "navigation_token": "nav_a",
            "task_id": 2,
            "plan_id": "plan_a",
            "user_input_id": "input_a",
            "description": "Navigate to keyframe 5",
            "destination_position": [1.0, 2.0, 0.0],
        },
    }
    return state


def _event(event_id: str = "event_arrival", *, plan_id: str = "plan_a", token: str = "nav_a") -> dict:
    return {
        "event_id": event_id,
        "type": "navigation_arrived",
        "source": "system",
        "created_at": "2026-06-20T00:00:00+08:00",
        "task_id": 2,
        "plan_id": plan_id,
        "user_input_id": "input_a",
        "navigation_token": token,
        "payload": {
            "summary": "Navigation arrival matched active navigation.",
            "navigation_token": token,
            "reported_position": [1.0, 2.0, 0.0],
            "destination_position": [1.0, 2.0, 0.0],
        },
    }


def _context(state: dict, logger: FakeLogger) -> dict:
    return {
        "state": state,
        "tasks": state["tasks"],
        "current_task_id": state.get("current_task_id"),
        "processed_event_ids": list(state.get("processed_event_ids", [])),
        "next_action": state.get("next_action", {"type": "idle"}),
        "logger": logger,
        "run_memory": None,
    }


def _arrival_summary(task: dict | None) -> str:
    return "I have arrived at the navigation target."


def _fallback_response(*args, **kwargs) -> str:
    return ""


def _append_task_result(task: dict, **kwargs) -> None:
    task.setdefault("result", []).append(dict(kwargs))


def _proceed(state: dict, next_id, tasks, context):
    result = dict(state)
    result["tasks"] = tasks
    result["current_task_id"] = next_id
    return result


def _apply_response(state: dict, response_text, **kwargs):
    result = dict(state)
    result.setdefault("turn_response_items", []).append(
        {
            "response_text": response_text,
            **kwargs,
        }
    )
    return result


def _handle(state: dict, event: dict, logger: FakeLogger) -> dict:
    from caragent_agent.agents.async_agent.orchestration.handlers import (
        handle_navigation_arrived_event,
    )

    return handle_navigation_arrived_event(
        event,
        _context(state, logger),
        finish_inserted_task_fn=lambda state, task_id, tasks: state,
        navigation_arrival_summary=_arrival_summary,
        fallback_navigation_user_facing_response_fn=_fallback_response,
        append_task_result=_append_task_result,
        proceed_to_next_task_from_context_fn=_proceed,
        new_structured_id=lambda prefix: f"{prefix}_audit",
        apply_user_facing_response=_apply_response,
    )


def _handle_unmatched(state: dict, event: dict, logger: FakeLogger) -> dict:
    from caragent_agent.agents.async_agent.orchestration.handlers import (
        handle_navigation_arrival_unmatched_event,
    )

    return handle_navigation_arrival_unmatched_event(event, _context(state, logger))


def _assert_duplicate_is_ignored() -> None:
    logger = FakeLogger()
    accepted = _handle(_base_state(), _event("event_1"), logger)
    assert accepted["tasks"][2]["status"] == "completed"
    assert "active_navigation" not in accepted
    assert accepted.get("navigation_arrival_receipts")

    duplicate = _handle(accepted, _event("event_2"), logger)
    assert duplicate["tasks"][2]["status"] == "completed"
    assert len(duplicate["tasks"][2].get("result", [])) == len(accepted["tasks"][2].get("result", []))
    assert any("decision=accepted" in line for line in logger.lines)
    assert any("decision=ignored_duplicate" in line for line in logger.lines)


def _assert_old_plan_is_stale() -> None:
    from caragent_agent.agents.async_agent.orchestration.runtime import build_navigation_arrival_event
    from langchain_core.messages import ToolMessage

    state = _base_state()
    state["current_plan_id"] = "plan_b"
    event = build_navigation_arrival_event(
        ToolMessage(content="Arrived at destination [1.000, 2.000, 0.000]", tool_call_id="tool_a"),
        messages=[],
        tasks=state["tasks"],
        current_task_id=2,
        current_plan_id=state["current_plan_id"],
        active_navigation=state["active_navigation"],
        match_tolerance_meters=0.5,
    )
    assert event["type"] == "navigation_arrival_unmatched"
    assert event["payload"]["unmatched_reason"] == "stale_plan"
    logger = FakeLogger()
    result = _handle_unmatched(state, event, logger)
    assert result["tasks"][2]["status"] == "waiting"
    assert any("decision=ignored_stale" in line for line in logger.lines)


def _assert_cancelled_task_is_stale() -> None:
    state = _base_state(status="cancelled")
    state["active_navigation"] = {
        "navigation_token": "nav_a",
        "task_id": 2,
        "plan_id": "plan_a",
        "destination_position": [1.0, 2.0, 0.0],
    }
    logger = FakeLogger()
    result = _handle(deepcopy(state), _event("event_cancelled"), logger)
    assert result["tasks"][2]["status"] == "cancelled"
    assert any("decision=ignored_stale" in line for line in logger.lines)


def _assert_same_position_new_navigation_is_accepted() -> None:
    state = _base_state()
    state["navigation_arrival_receipts"] = [
        {
            "decision": "accepted",
            "event_id": "old_arrival",
            "navigation_token": "old_nav",
            "task_id": 2,
            "plan_id": "old_plan",
            "reported_position": [1.0, 2.0, 0.0],
            "destination_position": [1.0, 2.0, 0.0],
        }
    ]
    logger = FakeLogger()
    accepted = _handle(state, _event("event_new", plan_id="plan_a", token="nav_a"), logger)
    assert accepted["tasks"][2]["status"] == "completed"
    assert any("decision=accepted" in line for line in logger.lines)


def _assert_old_arrival_does_not_complete_new_navigation() -> None:
    from caragent_agent.agents.async_agent.orchestration.runtime import (
        build_navigation_arrival_event,
    )
    from langchain_core.messages import ToolMessage

    state = _base_state()
    state["active_navigation"] = {
        "navigation_token": "nav_object",
        "task_id": 2,
        "plan_id": "plan_a",
        "user_input_id": "input_a",
        "description": "Approach the black chair",
        "destination_position": [2.0, -1.32, 0.0],
    }
    state["pending_navigation"] = dict(state["active_navigation"])
    state["tasks"][2]["description"] = "Approach the black chair"

    stale_event = build_navigation_arrival_event(
        ToolMessage(
            content="Arrived at destination [0.053, -0.978, 0.000]",
            tool_call_id="tool_stale",
        ),
        messages=[],
        tasks=state["tasks"],
        current_task_id=2,
        current_plan_id="plan_a",
        active_navigation=state["active_navigation"],
        match_tolerance_meters=0.5,
    )
    assert stale_event["type"] == "navigation_arrival_unmatched"
    assert stale_event["payload"]["unmatched_reason"] == "position_mismatch"
    logger = FakeLogger()
    still_waiting = _handle_unmatched(state, stale_event, logger)
    assert still_waiting["tasks"][2]["status"] == "waiting"
    assert any("decision=ignored_stale" in line for line in logger.lines)

    fresh_event = build_navigation_arrival_event(
        ToolMessage(
            content="Arrived at destination [2.000, -1.320, 0.000]",
            tool_call_id="tool_fresh",
        ),
        messages=[],
        tasks=still_waiting["tasks"],
        current_task_id=2,
        current_plan_id="plan_a",
        active_navigation=still_waiting["active_navigation"],
        match_tolerance_meters=0.5,
    )
    assert fresh_event["type"] == "navigation_arrived"
    accepted = _handle(still_waiting, fresh_event, logger)
    assert accepted["tasks"][2]["status"] == "completed"


def main() -> int:
    src_root = _repo_src_root()
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
    _install_import_stubs()
    _assert_duplicate_is_ignored()
    _assert_old_plan_is_stale()
    _assert_cancelled_task_is_stale()
    _assert_same_position_new_navigation_is_accepted()
    _assert_old_arrival_does_not_complete_new_navigation()
    print("navigation_arrival_gate_audit: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
