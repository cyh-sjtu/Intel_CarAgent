"""Audit lite-UI response/voice channel separation.

This script does not drive a browser.  It checks the payload contract that the
lite UI consumes: navigation guidance should appear as guidance events, while
the lite conversation response channel should keep only real final replies.
"""

from __future__ import annotations

from pathlib import Path
import sys
import types
from typing import Any


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

        messages.BaseMessage = BaseMessage
        messages.AIMessage = BaseMessage
        messages.HumanMessage = BaseMessage
        messages.ToolMessage = BaseMessage
        messages.SystemMessage = BaseMessage
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

    if "langchain_openai" not in sys.modules:
        langchain_openai = types.ModuleType("langchain_openai")

        class ChatOpenAI:  # pragma: no cover
            pass

        langchain_openai.ChatOpenAI = ChatOpenAI
        sys.modules["langchain_openai"] = langchain_openai

    if "langgraph.graph.message" not in sys.modules:
        langgraph = types.ModuleType("langgraph")
        graph = types.ModuleType("langgraph.graph")
        graph_message = types.ModuleType("langgraph.graph.message")
        graph_message.add_messages = lambda left, right: (left or []) + (right or [])
        sys.modules.setdefault("langgraph", langgraph)
        sys.modules["langgraph.graph"] = graph
        sys.modules["langgraph.graph.message"] = graph_message


def _visible_lite_response_items(turn: dict[str, Any]) -> list[dict[str, Any]]:
    """Mirror AsyncAgentWebApp._filter_user_turn_response_items(lite=True)."""

    if str(turn.get("status") or "") != "completed":
        return []
    response_items = [
        dict(item)
        for item in list(turn.get("response_items") or [])
        if isinstance(item, dict)
    ]
    guidance_backed_events = {
        "plan_created",
        "plan_edited",
        "plan_updated",
        "task_waiting",
        "navigation_arrived",
        "task_failed",
        "task_cancelled",
    }
    return [
        item
        for item in response_items
        if str(item.get("source_event_type") or "").strip()
        not in guidance_backed_events
    ]


def _voice_candidates(payload: dict[str, Any]) -> list[tuple[str, str]]:
    """Mirror the lite UI's channel choice at the data level."""

    candidates: list[tuple[str, str]] = []
    for event in payload.get("state", {}).get("guidance_events", []) or []:
        event_type = str(event.get("event_type") or "")
        text = str(event.get("text") or "").strip()
        if text and event_type in {
            "request_received",
            "plan_created",
            "plan_updated",
            "navigation_start",
            "arrival",
            "arrival_verification",
            "failed",
            "cancelled",
            "stuck",
        }:
            candidates.append((event_type, text))

    nav_guidance_present = any(
        event_type in {"navigation_start", "arrival", "arrival_verification", "failed", "cancelled", "stuck"}
        for event_type, _ in candidates
    )
    for turn in payload.get("conversation_history", []) or []:
        for item in turn.get("response_items", []) or []:
            if str(item.get("response_type") or "") == "progress":
                continue
            text = str(item.get("response_text") or "").strip()
            if text and not nav_guidance_present:
                candidates.append(("agent_reply", text))
    return candidates


def _assert_guidance_dedupe_is_global() -> None:
    from caragent_agent.agents.async_agent.guidance import append_guidance

    state: dict[str, Any] = {}
    state = append_guidance(
        state,
        event_type="plan_created",
        text="已生成计划，包含 2 个任务。",
        dedupe_key="plan_created:plan_a",
    )
    state = append_guidance(
        state,
        event_type="navigation_start",
        text="正在前往目标关键帧“电梯门左侧的灭火箱”。",
        dedupe_key="navigation_start:1",
    )
    state = append_guidance(
        state,
        event_type="plan_created",
        text="已生成计划，包含 2 个任务。",
        dedupe_key="plan_created:plan_a",
    )
    events = state.get("guidance_events") or []
    assert [event["event_type"] for event in events].count("plan_created") == 1


def _assert_lite_filters_navigation_replies() -> None:
    turn = {
        "status": "completed",
        "response_items": [
            {
                "response_text": "已生成计划，包含 2 个任务。",
                "response_type": "progress",
                "source_event_type": "plan_created",
            },
            {
                "response_text": "正在前往目标关键帧“电梯门左侧的灭火箱”。",
                "response_type": "progress",
                "source_event_type": "task_waiting",
            },
            {
                "response_text": "已到达目标关键帧“电梯门左侧的灭火箱”。",
                "response_type": "result",
                "source_event_type": "navigation_arrived",
            },
            {
                "response_text": "我看到它大约有四个脚。",
                "response_type": "result",
                "source_event_type": "task_completed",
            },
        ],
    }
    visible = _visible_lite_response_items(turn)
    assert [item["response_text"] for item in visible] == ["我看到它大约有四个脚。"]


def _assert_lite_voice_candidates_are_not_duplicated() -> None:
    payload = {
        "state": {
            "guidance_events": [
                {
                    "event_type": "plan_created",
                    "text": "已生成计划，包含 2 个任务。",
                    "dedupe_key": "plan_created:plan_a",
                },
                {
                    "event_type": "navigation_start",
                    "text": "正在前往目标关键帧“电梯门左侧的灭火箱”。",
                    "dedupe_key": "navigation_start:1",
                },
                {
                    "event_type": "arrival",
                    "text": "已到达目标关键帧“电梯门左侧的灭火箱”。",
                    "dedupe_key": "arrival:1",
                },
            ],
        },
        "conversation_history": [
            {
                "status": "completed",
                "response_items": _visible_lite_response_items(
                    {
                        "status": "completed",
                        "response_items": [
                            {
                                "response_text": "已到达目标关键帧“电梯门左侧的灭火箱”。",
                                "response_type": "result",
                                "source_event_type": "navigation_arrived",
                            }
                        ],
                    }
                ),
            }
        ],
    }
    candidates = _voice_candidates(payload)
    texts = [text for _, text in candidates]
    assert texts.count("已生成计划，包含 2 个任务。") == 1
    assert texts.count("正在前往目标关键帧“电梯门左侧的灭火箱”。") == 1
    assert texts.count("已到达目标关键帧“电梯门左侧的灭火箱”。") == 1


def _assert_interaction_profile_defaults_enable_guidance() -> None:
    from caragent_agent.agents.async_agent.guidance import DEFAULT_INTERACTION_PROFILE

    profile = DEFAULT_INTERACTION_PROFILE
    assert profile.get("voice_enabled_default") is True
    assert profile.get("speak_guidance_events") is True
    assert profile.get("speak_agent_replies") is True
    assert profile.get("response_role_enabled") is True


def main() -> int:
    src_root = _repo_src_root()
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
    _install_import_stubs()
    _assert_guidance_dedupe_is_global()
    _assert_lite_filters_navigation_replies()
    _assert_lite_voice_candidates_are_not_duplicated()
    _assert_interaction_profile_defaults_enable_guidance()
    print("lite_voice_contract_audit: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
