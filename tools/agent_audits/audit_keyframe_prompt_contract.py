"""Audit keyframe-matching prompt and candidate-summary contracts.

This script is intentionally local and model-free. It checks invariants that
should hold before running more expensive planner/executor API regressions.
"""

from __future__ import annotations

import sys
import types
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace


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


def _agent_package_root() -> Path:
    return _repo_src_root() / "caragent_agent"


def _load_prompt_text() -> str:
    prompt_path = _agent_package_root() / "prompts" / "agent_prompts.yaml"
    return prompt_path.read_text(encoding="utf-8")


def _install_import_stubs() -> None:
    """Install tiny local stubs for optional LLM deps absent on dev laptops."""

    if "langchain_core.language_models" not in sys.modules:
        langchain_core = types.ModuleType("langchain_core")
        language_models = types.ModuleType("langchain_core.language_models")
        messages = types.ModuleType("langchain_core.messages")
        callbacks = types.ModuleType("langchain_core.callbacks")

        class BaseLLM:  # pragma: no cover - only for import-time compatibility
            pass

        class BaseChatModel(BaseLLM):  # pragma: no cover
            pass

        class HumanMessage:  # pragma: no cover
            def __init__(self, content=None, **kwargs):
                self.content = content

        class SystemMessage(HumanMessage):  # pragma: no cover
            pass

        class BaseMessage(HumanMessage):  # pragma: no cover
            pass

        class AsyncCallbackHandler:  # pragma: no cover
            pass

        language_models.BaseLLM = BaseLLM
        language_models.BaseChatModel = BaseChatModel
        messages.BaseMessage = BaseMessage
        messages.HumanMessage = HumanMessage
        messages.SystemMessage = SystemMessage
        callbacks.AsyncCallbackHandler = AsyncCallbackHandler
        sys.modules.setdefault("langchain_core", langchain_core)
        sys.modules["langchain_core.language_models"] = language_models
        sys.modules["langchain_core.messages"] = messages
        sys.modules["langchain_core.callbacks"] = callbacks

    if "langchain_openai" not in sys.modules:
        langchain_openai = types.ModuleType("langchain_openai")

        class ChatOpenAI:  # pragma: no cover
            def __init__(self, *args, **kwargs):
                pass

        langchain_openai.ChatOpenAI = ChatOpenAI
        sys.modules["langchain_openai"] = langchain_openai

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


def _assert_prompt_contract() -> None:
    prompt = _load_prompt_text()
    forbidden = [
        "destination resolver",
        "resolver task",
        "object-level resolver",
        "json destination",
    ]
    prompt_lower = prompt.lower()
    for phrase in forbidden:
        assert phrase not in prompt_lower, f"forbidden prompt phrase remained: {phrase}"
    required = [
        "submit_task_result",
        "semantic_keyframe",
        "semantic_object",
        "Current scene vs attached image",
        "Retrieval scores are initial signals, not proof",
        "target_visibility_hints",
    ]
    missing = [phrase for phrase in required if phrase not in prompt]
    assert not missing, f"prompt is missing expected guidance: {missing}"


@dataclass
class FakeNode:
    kf_id: int
    name: str
    semantic: str
    position: list[float]
    rgb_path: str = "/tmp/fake_keyframe.jpg"
    left_path: str = "/tmp/fake_left.jpg"


def _fake_scene_memory() -> SimpleNamespace:
    return SimpleNamespace(
        keyframe_nodes={
            4: FakeNode(
                kf_id=4,
                name="kf_4",
                semantic=(
                    "A gray four-legged table is clearly visible near the center. "
                    "The table is complete and not occluded. There are chairs nearby."
                ),
                position=[1.0, 2.0, 0.0],
            ),
            5: FakeNode(
                kf_id=5,
                name="kf_5",
                semantic=(
                    "A corridor view with a gray table partly cut off at the left edge. "
                    "The target is partially visible and occluded by chairs."
                ),
                position=[2.0, 3.0, 0.0],
            ),
        }
    )


def _assert_requirement_candidate_contract() -> None:
    from caragent_agent.agents.tools.search.requirement_search import RequirementSearchTool

    tool = object.__new__(RequirementSearchTool)
    tool.scene_memory = _fake_scene_memory()
    summary = tool._candidate_summary(4, "gray four-legged table")

    assert summary["keyframe_id"] == 4
    assert summary["position"] == [1.0, 2.0, 0.0]
    assert "semantic" not in summary
    assert summary.get("semantic_excerpt")
    assert summary.get("short_semantics_excerpt") == summary.get("semantic_excerpt")
    assert summary.get("target_visibility_hints"), summary
    assert "table" in " ".join(summary.get("evidence_terms") or [])


def _assert_attached_candidate_contract() -> None:
    from caragent_agent.agents.tools.analysis.attached_image_tools import _node_records

    records = _node_records(
        _fake_scene_memory(),
        [4, 5],
        "Find a useful staging keyframe for a gray four-legged table.",
    )
    assert len(records) == 2
    for record in records:
        assert "semantic" not in record
        assert record.get("semantic_excerpt")
        assert "target_visibility_hints" in record
        assert record.get("match_reason")


def _assert_planner_parser_preserves_semantic_contract() -> None:
    from caragent_agent.agents.async_agent.planning.task_graph import (
        parse_planned_tasks_from_response,
    )

    plan_text = json.dumps(
        {
            "tasks": [
                {
                    "task_id": 1,
                    "task_type": "navigation_action",
                    "description": "Navigate to a staging keyframe where the gray table is clearly visible",
                    "target": {
                        "type": "semantic_keyframe",
                        "query": "gray four-legged table",
                        "selection_policy": "target_visibility_first",
                    },
                    "primary_target": "gray table",
                    "selection_policy": "target_visibility_first",
                    "image_refs": ["latest"],
                    "outputs": ["destination", "current_place_context"],
                    "next_task_id": 2,
                },
                {
                    "task_id": 2,
                    "task_type": "navigation_action",
                    "description": "Navigate close to the gray table",
                    "target": {
                        "type": "semantic_object",
                        "object_description": "gray table",
                        "inputs_from": {"place_context": "task1.current_place_context"},
                    },
                    "depends_on": [1],
                    "outputs": ["destination", "selected_object"],
                },
            ]
        }
    )
    tasks, first_task_id = parse_planned_tasks_from_response(
        plan_text,
        plan_id="audit_plan",
        user_input_id="audit_input",
        created_at="2026-06-20T00:00:00+08:00",
    )
    assert first_task_id == 1
    assert tasks[1]["target"]["type"] == "semantic_keyframe"
    assert tasks[1]["target"]["query"] == "gray four-legged table"
    assert tasks[1]["selection_policy"] == "target_visibility_first"
    assert tasks[1]["image_refs"] == ["latest"]
    assert tasks[2]["target"]["type"] == "semantic_object"
    assert tasks[2]["target"]["inputs_from"]["place_context"] == "task1.current_place_context"
    assert tasks[2]["outputs"] == ["destination", "selected_object"]


def main() -> None:
    src_root = _repo_src_root()
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))

    _install_import_stubs()
    _assert_prompt_contract()
    _assert_requirement_candidate_contract()
    _assert_attached_candidate_contract()
    _assert_planner_parser_preserves_semantic_contract()
    print("keyframe_prompt_contract_audit: ok")


if __name__ == "__main__":
    main()
