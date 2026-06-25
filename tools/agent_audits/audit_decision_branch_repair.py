"""Audit structural repair for decision tasks that omit branches."""

from __future__ import annotations

import json
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
    if "langchain_core.language_models" not in sys.modules:
        langchain_core = types.ModuleType("langchain_core")
        language_models = types.ModuleType("langchain_core.language_models")

        class BaseChatModel:  # pragma: no cover
            pass

        language_models.BaseChatModel = BaseChatModel
        sys.modules.setdefault("langchain_core", langchain_core)
        sys.modules["langchain_core.language_models"] = language_models

    if "langchain_core.messages" not in sys.modules:
        messages = types.ModuleType("langchain_core.messages")

        class BaseMessage:  # pragma: no cover
            pass

        messages.BaseMessage = BaseMessage
        sys.modules["langchain_core.messages"] = messages

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


def main() -> int:
    src_root = _repo_src_root()
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
    _install_import_stubs()

    from caragent_agent.agents.async_agent.planning.task_graph import (
        parse_planned_tasks_from_response,
    )

    plan_text = json.dumps(
        {
            "tasks": [
                {
                    "task_id": 1,
                    "task_type": "llm_action",
                    "description": "Observe whether the doorway is visible",
                    "next_task_id": 2,
                },
                {
                    "task_id": 2,
                    "task_type": "decision",
                    "description": "Choose visible or fallback route",
                    "depends_on": [1],
                    "next_task_id": None,
                },
                {
                    "task_id": 3,
                    "task_type": "llm_action",
                    "description": "Explain the visible doorway evidence",
                    "depends_on": [2],
                    "next_task_id": 5,
                },
                {
                    "task_id": 4,
                    "task_type": "llm_action",
                    "description": "Resolve the fallback wooden doors",
                    "depends_on": [2],
                    "next_task_id": 5,
                },
                {
                    "task_id": 5,
                    "task_type": "llm_action",
                    "description": "Continue shared route",
                    "depends_on": [3, 4],
                    "next_task_id": None,
                },
            ]
        },
        ensure_ascii=False,
    )
    tasks, first_task_id = parse_planned_tasks_from_response(
        plan_text,
        plan_id="plan_a",
        user_input_id="input_a",
        created_at="2026-06-21T00:00:00Z",
    )
    branches = tasks[2].get("branches")
    print(json.dumps({"first_task_id": first_task_id, "branches": branches}, ensure_ascii=False, indent=2))
    assert first_task_id == 1
    assert isinstance(branches, dict)
    assert set(branches.values()) == {3, 4}
    assert tasks[2].get("default_branch") in branches
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
