"""Local audit for executor task-failure classification boundaries."""

from __future__ import annotations

from pathlib import Path
import json
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

        class SystemMessage(BaseMessage):  # pragma: no cover
            pass

        class ToolMessage(BaseMessage):  # pragma: no cover
            def __init__(self, content=None, tool_call_id=None, **kwargs):
                super().__init__(content=content, **kwargs)
                self.tool_call_id = tool_call_id

        messages.BaseMessage = BaseMessage
        messages.AIMessage = AIMessage
        messages.HumanMessage = HumanMessage
        messages.SystemMessage = SystemMessage
        messages.ToolMessage = ToolMessage
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

    if "langgraph.prebuilt.tool_node" not in sys.modules:
        langgraph = types.ModuleType("langgraph")
        prebuilt = types.ModuleType("langgraph.prebuilt")
        tool_node = types.ModuleType("langgraph.prebuilt.tool_node")

        class ToolNode:  # pragma: no cover
            pass

        tool_node.ToolNode = ToolNode
        sys.modules.setdefault("langgraph", langgraph)
        sys.modules["langgraph.prebuilt"] = prebuilt
        sys.modules["langgraph.prebuilt.tool_node"] = tool_node

    if "langgraph.graph.message" not in sys.modules:
        langgraph = sys.modules.get("langgraph") or types.ModuleType("langgraph")
        graph = types.ModuleType("langgraph.graph")
        graph_message = types.ModuleType("langgraph.graph.message")

        def add_messages(left, right):  # pragma: no cover
            return (left or []) + (right or [])

        graph_message.add_messages = add_messages
        sys.modules.setdefault("langgraph", langgraph)
        sys.modules["langgraph.graph"] = graph
        sys.modules["langgraph.graph.message"] = graph_message

    react_module_name = "caragent_agent.third_party.from_langgraph.react_agent"
    if react_module_name not in sys.modules:
        react_agent = types.ModuleType(react_module_name)

        def create_react_agent(*args, **kwargs):  # pragma: no cover
            raise RuntimeError("audit stub should not create a react agent")

        react_agent.create_react_agent = create_react_agent
        sys.modules[react_module_name] = react_agent


_install_import_stubs()
sys.path.insert(0, str(_repo_src_root()))

from caragent_agent.agents.async_agent.execution.execute_node import (  # noqa: E402
    _tool_failure_blocks_task,
)


def _structured_result(status: str, *, summary: str, data: dict | None = None) -> str:
    return json.dumps(
        {
            "status": status,
            "summary": summary,
            "data": data or {},
            "provenance": {"tool": "audit"},
        },
        ensure_ascii=False,
    )


def _audit_observation_can_recover_from_one_failed_tool() -> bool:
    task = {
        "task_id": 3,
        "task_type": "llm_action",
        "description": "Observe the current place and decide whether it is suitable.",
        "outputs": ["current_place_context"],
    }
    tool_trace = {
        "tool_results": [
            {
                "name": "analyse_on_current_image",
                "content": _structured_result(
                    "error",
                    summary="Controller returned no current image.",
                ),
            },
            {
                "name": "get_keyframe_nodes_info",
                "content": _structured_result(
                    "ok",
                    summary="Recovered place context from keyframe metadata.",
                    data={"keyframe_id": 12},
                ),
            },
        ],
        "final_ai_content": "The place is suitable based on the available keyframe and state evidence.",
    }
    return not _tool_failure_blocks_task(
        task,
        {},
        tool_trace,
        failure_summary="analyse_on_current_image: Controller returned no current image.",
        final_ai_content=str(tool_trace["final_ai_content"]),
    )


def _audit_semantic_destination_signal_requires_destination() -> bool:
    task = {
        "task_id": 1,
        "task_type": "llm_action",
        "description": "Resolve the target object destination.",
        "outputs": ["destination"],
    }
    tasks = {
        1: task,
        2: {
            "task_id": 2,
            "task_type": "navigation_action",
            "target": {"type": "task_output", "task_id": 1, "field": "destination"},
        },
    }
    tool_trace = {
        "tool_results": [
            {
                "name": "approach_object_in_current_view",
                "content": _structured_result(
                    "error",
                    summary="No object destination could be resolved.",
                ),
            },
            {
                "name": "get_current_state",
                "content": _structured_result(
                    "ok",
                    summary="Current state available.",
                    data={"position": [0, 0, 0]},
                ),
            },
        ],
        "final_ai_content": "I could not resolve a reliable object destination.",
    }
    return _tool_failure_blocks_task(
        task,
        tasks,
        tool_trace,
        failure_summary="approach_object_in_current_view: No object destination could be resolved.",
        final_ai_content=str(tool_trace["final_ai_content"]),
    )


def _audit_semantic_destination_signal_passes_with_destination() -> bool:
    task = {
        "task_id": 1,
        "task_type": "llm_action",
        "description": "Resolve the target object destination.",
        "outputs": ["destination"],
    }
    tasks = {
        1: task,
        2: {
            "task_id": 2,
            "task_type": "navigation_action",
            "target": {"type": "task_output", "task_id": 1, "field": "destination"},
        },
    }
    tool_trace = {
        "tool_results": [
            {
                "name": "analyse_on_current_image",
                "content": _structured_result(
                    "error",
                    summary="Optional image check failed.",
                ),
            },
            {
                "name": "approach_object_in_current_view",
                "content": _structured_result(
                    "ok",
                    summary="Resolved object position.",
                    data={
                        "destination": {
                            "type": "position",
                            "position": [1.0, 2.0, 0.0],
                            "yaw_deg": 30.0,
                        }
                    },
                ),
            },
        ],
        "final_ai_content": json.dumps(
            {
                "destination": {
                    "type": "position",
                    "position": [1.0, 2.0, 0.0],
                    "yaw_deg": 30.0,
                }
            },
            ensure_ascii=False,
        ),
    }
    return not _tool_failure_blocks_task(
        task,
        tasks,
        tool_trace,
        failure_summary="analyse_on_current_image: Optional image check failed.",
        final_ai_content=str(tool_trace["final_ai_content"]),
    )


def main() -> None:
    results = {
        "observation_recovered": _audit_observation_can_recover_from_one_failed_tool(),
        "destination_missing_blocks": _audit_semantic_destination_signal_requires_destination(),
        "destination_signal_passes": _audit_semantic_destination_signal_passes_with_destination(),
    }
    print(json.dumps(results, ensure_ascii=False, indent=2))
    assert all(results.values()), results


if __name__ == "__main__":
    main()
