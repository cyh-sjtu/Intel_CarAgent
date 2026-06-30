"""Execute node for running current async-agent tasks."""

from __future__ import annotations

import json
import traceback
from typing import Any, Optional, Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.prebuilt.tool_node import ToolNode

from caragent_agent.agents.async_agent.execution.context import (
    _extract_named_signal,
    background_result_is_reusable_for_task,
    build_background_reference,
    build_execution_guide,
    build_tool_catalog,
    is_navigation_action,
    prepare_context_bundle,
    task_depends_on_query_result,
    truncate_context_text,
)
from caragent_agent.agents.async_agent.execution.navigation_actions import (
    try_dispatch_structured_navigation_action as _try_dispatch_structured_navigation_action,
)
from caragent_agent.agents.async_agent.execution.support import (
    append_task_result,
    apply_user_facing_response,
    build_precision_support_tools,
    build_task_turn_response_type,
    build_task_user_facing_response,
    count_successful_navigation_commands,
    find_tool_contract_violation_message,
    extract_tool_trace,
    find_tool_failure_message,
    issued_navigation_command,
    navigation_arrival_summary,
    navigation_waiting_summary,
    tool_content_indicates_error,
)
from caragent_agent.agents.async_agent.guidance import (
    append_guidance,
    navigation_waiting_text,
)
from caragent_agent.agents.async_agent.execution.runtime_tool_context import (
    runtime_tool_context,
)
from caragent_agent.agents.async_agent.execution.tool_call_budget import (
    execute_tool_budget_context,
)
from caragent_agent.agents.async_agent.target_resolution.session_anchors import (
    record_anchor_from_object_destination,
    record_anchor_from_resolution_result,
)
from caragent_agent.agents.async_agent.orchestration.node_common import (
    _get_current_task,
    _record_run_memory_event,
    _strip_ignored_state_fields,
)
from caragent_agent.agents.async_agent.orchestration.runtime import (
    build_pending_navigation_snapshot,
    new_structured_id,
    now_iso,
)
from caragent_agent.agents.async_agent.planning.prompting import AGENT_PROMPTS
from caragent_agent.agents.async_agent.planning.task_graph import get_task_progress_context
from caragent_agent.agents.async_agent.runtime.control import (
    record_foreground_task,
    set_background_enabled,
)
from caragent_agent.agents.async_agent.runtime.console import Colors
from caragent_agent.agents.async_agent.runtime.types import (
    AsyncAgentState,
    BackgroundAnalysisItem,
    EventItem,
    TaskItem,
)
from caragent_agent.third_party.from_langgraph.react_agent import create_react_agent


NAVIGATION_TOOL_NAMES = {"go_to_keyframe", "go_to_position"}
KEYFRAME_SEARCH_TOOL_NAMES = {"search_requirement_on_keyframe_nodes"}
ATTACHED_IMAGE_TOOL_NAMES = {
    "analyse_attached_image",
    "match_attached_image_to_keyframes",
}
BACKGROUND_ONLY_TOOL_NAMES = {
    "preanalyze_object_on_keyframe",
}
DIAGNOSTIC_ONLY_TOOL_NAMES = {
    "resolve_object_from_attached_image",
}
SEMANTIC_GROUNDING_TOOL_NAMES = {
    "approach_object_in_current_view",
    "preanalyze_object_on_keyframe",
}


def _navigation_guidance_dedupe_key(task: Optional[TaskItem], task_id: Optional[int]) -> str:
    """Return a plan-scoped guidance key for navigation start events."""

    plan_id = ""
    if isinstance(task, dict):
        plan_id = str(task.get("plan_id") or "").strip()
    if not plan_id:
        plan_id = "no-plan"
    return f"navigation_start:{plan_id}:{task_id}"


def _task_outputs_signal(current_task: Optional[TaskItem], signal_name: str) -> bool:
    outputs = (current_task or {}).get("outputs")
    if outputs is None:
        return True
    if isinstance(outputs, str):
        return outputs.strip() == signal_name
    try:
        return signal_name in {str(item).strip() for item in outputs}
    except Exception:
        return True


def _compact_task_signal(tool_trace: dict[str, Any], signal_name: str) -> Optional[dict[str, Any]]:
    """Extract one reusable task-output signal from a tool trace."""

    value = _extract_named_signal(tool_trace, signal_name)
    return value if isinstance(value, dict) else None


def _submitted_task_result_data(tool_trace: dict[str, Any]) -> dict[str, Any]:
    """Return the latest submit_task_result data payload, if any."""

    for tool_result in reversed(tool_trace.get("tool_results", []) or []):
        if str(tool_result.get("name") or "").strip() != "submit_task_result":
            continue
        content = tool_result.get("content")
        try:
            parsed = json.loads(content) if isinstance(content, str) else content
        except Exception:
            parsed = None
        if isinstance(parsed, dict):
            data = parsed.get("data")
            if isinstance(data, dict):
                return data
    return {}


def _submitted_task_result_summary(tool_trace: dict[str, Any]) -> str:
    """Return the latest submit_task_result summary."""

    for tool_result in reversed(tool_trace.get("tool_results", []) or []):
        if str(tool_result.get("name") or "").strip() != "submit_task_result":
            continue
        content = tool_result.get("content")
        try:
            parsed = json.loads(content) if isinstance(content, str) else content
        except Exception:
            parsed = None
        if isinstance(parsed, dict):
            return str(parsed.get("summary") or "").strip()
    return ""


def _has_submitted_task_result(tool_trace: dict[str, Any]) -> bool:
    """Return True when the executor formally submitted a task result."""

    return any(
        str(tool_result.get("name") or "").strip() == "submit_task_result"
        for tool_result in tool_trace.get("tool_results", []) or []
        if isinstance(tool_result, dict)
    )


def _latest_tool_result_payload(tool_trace: dict[str, Any], tool_name: str) -> Optional[dict[str, Any]]:
    for tool_result in reversed(tool_trace.get("tool_results", []) or []):
        if str(tool_result.get("name") or "").strip() != tool_name:
            continue
        content = tool_result.get("content")
        try:
            parsed = json.loads(content) if isinstance(content, str) else content
        except Exception:
            parsed = None
        if isinstance(parsed, dict):
            return parsed
    return None


def _destination_from_keyframe_search_trace(tool_trace: dict[str, Any]) -> Optional[dict[str, Any]]:
    """Promote deterministic scene-memory search output into a reusable destination."""

    payload = _latest_tool_result_payload(tool_trace, "search_requirement_on_keyframe_nodes")
    if not isinstance(payload, dict):
        return None
    status = str(payload.get("status") or "").strip().lower()
    if status and status not in {"ok", "success", "succeeded"}:
        return None
    data = payload.get("data") if isinstance(payload.get("data"), dict) else payload
    resolution_status = str(data.get("resolution_status") or "").strip().lower()
    if resolution_status and resolution_status != "resolved":
        return None
    keyframe_id = None
    for key in ("recommended_keyframe_id", "keyframe_id", "target_keyframe_id"):
        if data.get(key) is None:
            continue
        try:
            keyframe_id = int(data.get(key))
            break
        except Exception:
            continue
    if keyframe_id is None:
        destination = data.get("recommended_destination") or data.get("destination")
        if isinstance(destination, dict):
            for key in ("keyframe_id", "keyframe_node_id", "target_keyframe_id"):
                if destination.get(key) is None:
                    continue
                try:
                    keyframe_id = int(destination.get(key))
                    break
                except Exception:
                    continue
    if keyframe_id is None:
        return None
    result: dict[str, Any] = {"type": "keyframe", "keyframe_id": keyframe_id}
    for source_key, dest_key in (
        ("position", "position"),
        ("recommended_position", "position"),
        ("target_position", "position"),
        ("display_label", "display_label"),
        ("user_query", "user_query"),
        ("requirement", "query"),
    ):
        value = data.get(source_key)
        if value not in (None, "", [], {}):
            result[dest_key] = value
    return result


def _record_session_anchors_from_tool_trace(
    *,
    shared_runtime_control: Optional[dict[str, Any]],
    current_task: Optional[TaskItem],
    tool_trace: dict[str, Any],
    logger: Optional[Any],
) -> None:
    if shared_runtime_control is None or current_task is None:
        return
    recorded: list[dict[str, Any]] = []
    resolution_result = _latest_tool_result_payload(tool_trace, "target_resolution")
    if isinstance(resolution_result, dict):
        anchor = record_anchor_from_resolution_result(
            shared_runtime_control,
            resolution_result,
            current_task=current_task,
        )
        if anchor:
            recorded.append(anchor)
    resolved_anchor = (
        resolution_result.get("anchor")
        if isinstance(resolution_result, dict) and isinstance(resolution_result.get("anchor"), dict)
        else {}
    )
    resolved_ref = (
        resolution_result.get("target_ref")
        if isinstance(resolution_result, dict) and isinstance(resolution_result.get("target_ref"), dict)
        else {}
    )
    submitted = _submitted_task_result_data(tool_trace)
    destination = submitted.get("destination") if isinstance(submitted.get("destination"), dict) else None
    selected_object = (
        submitted.get("selected_object")
        if isinstance(submitted.get("selected_object"), dict)
        else None
    )
    already_recorded_resolved_object = (
        str(resolved_ref.get("kind") or "") == "object"
        and str(resolved_anchor.get("anchor_type") or "") == "position"
    )
    if destination is not None and selected_object is not None and not already_recorded_resolved_object:
        anchor = record_anchor_from_object_destination(
            shared_runtime_control,
            current_task=current_task,
            destination=destination,
            selected_object=selected_object,
            evidence=[],
        )
        if anchor:
            recorded.append(anchor)
    if logger is not None:
        for anchor in recorded:
            try:
                logger.log_foreground(
                    "session_anchor_recorded: "
                    + json.dumps(anchor, ensure_ascii=False, default=str)
                )
            except Exception:
                pass


def _task_signal_from_trace(tool_trace: dict[str, Any], signal_name: str) -> Optional[dict[str, Any]]:
    """Extract one task output signal, preferring formal submit_task_result."""

    submitted = _submitted_task_result_data(tool_trace)
    value = submitted.get(signal_name)
    if isinstance(value, dict):
        return value
    return _compact_task_signal(tool_trace, signal_name)


def _task_text_signal_from_trace(tool_trace: dict[str, Any], signal_name: str) -> str:
    """Extract one submitted string task-output signal."""

    submitted = _submitted_task_result_data(tool_trace)
    value = submitted.get(signal_name)
    return str(value or "").strip()


def _compact_background_object_preanalysis(value: Any) -> dict[str, Any] | None:
    """Return executor/memory-facing object preanalysis evidence."""

    if not isinstance(value, dict):
        return None
    compact: dict[str, Any] = {}
    for key in ("status", "reason", "mode", "destination"):
        if value.get(key) not in (None, "", [], {}):
            compact[key] = value.get(key)
    approach = value.get("approach") if isinstance(value.get("approach"), dict) else {}
    if approach:
        compact["approach"] = {
            key: approach.get(key)
            for key in ("status", "reason", "mode")
            if approach.get(key) not in (None, "", [], {})
        }
    paths = value.get("paths") if isinstance(value.get("paths"), dict) else {}
    artifact_paths = {
        key: paths.get(key) or value.get(key)
        for key in (
            "output_dir",
            "summary_json",
            "status_json",
            "approach_goal_json",
            "debug_png",
            "mono_guard_json",
            "selected_grounding_json",
        )
        if paths.get(key) or value.get(key)
    }
    if artifact_paths:
        compact["artifact_paths"] = artifact_paths
    return compact or None


def _tool_trace_has_success_evidence(tool_trace: dict[str, Any]) -> bool:
    """Return True when at least one tool result produced usable evidence."""

    for tool_result in tool_trace.get("tool_results", []) or []:
        content = str(tool_result.get("content") or "")
        if tool_content_indicates_error(content):
            continue
        return True
    return False


def _tool_failure_blocks_task(
    current_task: Optional[TaskItem],
    tasks: dict[int, TaskItem],
    tool_trace: dict[str, Any],
    *,
    failure_summary: Optional[str],
    final_ai_content: str,
) -> bool:
    """Decide whether a tool-level failure should fail the whole task.

    Tool failures are fatal when the task's required output contract is missing.
    For ordinary observation/reasoning tasks, one failed tool should not override
    a final answer that is grounded by other successful tool evidence.
    """

    if not failure_summary:
        return False
    if _task_signal_from_trace(tool_trace, "destination") is not None:
        return False
    if not final_ai_content.strip():
        return True

    if (
        _task_outputs_signal(current_task, "destination")
        and _task_produces_reusable_destination_for_navigation(current_task, tasks)
        and _task_signal_from_trace(tool_trace, "destination") is None
    ):
        return True

    return not _tool_trace_has_success_evidence(tool_trace)


def _task_produces_reusable_destination_for_navigation(
    current_task: Optional[TaskItem],
    tasks: dict[int, TaskItem],
) -> bool:
    """Return True when an llm_action feeds a following navigation target."""

    if not current_task or current_task.get("task_type") != "llm_action":
        return False
    try:
        current_task_id = int(current_task.get("task_id"))
    except Exception:
        return False

    for task in tasks.values():
        if not isinstance(task, dict):
            continue
        if str(task.get("task_type") or "").strip() != "navigation_action":
            continue
        target = task.get("target")
        if not isinstance(target, dict):
            continue
        if target.get("type") != "task_output" or target.get("field") != "destination":
            continue
        try:
            if int(target.get("task_id")) == current_task_id:
                return True
        except Exception:
            continue
    return False


def _try_complete_semantic_grounding_from_background(
    current_task: Optional[TaskItem],
    *,
    tasks: dict[int, TaskItem],
    background_result: BackgroundAnalysisItem | str | None,
) -> Optional[dict[str, Any]]:
    """Use completed semantic-grounding background output as the task result."""

    if not _task_produces_reusable_destination_for_navigation(current_task, tasks):
        return None
    if not isinstance(background_result, dict):
        return None
    if str(background_result.get("status") or "").strip().lower() != "completed":
        return None

    raw_destination = background_result.get("recommended_destination")
    if isinstance(raw_destination, dict):
        destination = {"destination": dict(raw_destination)}
        destination_json = json.dumps(destination, ensure_ascii=False)
        reason = truncate_context_text(
            background_result.get("recommendation_reason")
            or background_result.get("summary")
            or background_result.get("failure_reason"),
            limit=320,
        )
        summary = "Resolved destination from background preanalysis."
        if raw_destination.get("type"):
            summary += f" Type: {raw_destination.get('type')}."
        if reason:
            summary += f" Reason: {reason}"
        final_ai_content = f"{summary}\n\n{destination_json}"
        synthetic_tool_payload = {
            "status": "ok",
            "summary": summary,
            "data": {
                "source": "background_preanalysis",
                "destination": destination["destination"],
                "object_preanalysis": _compact_background_object_preanalysis(
                    background_result.get("object_preanalysis")
                ),
            },
        }
        return {
            "event_type": "task_completed",
            "summary": final_ai_content,
            "tool_name": "background_preanalysis",
            "tool_trace": {
                "tool_calls": [],
                "tool_results": [
                    {
                        "name": "background_preanalysis",
                        "content": json.dumps(synthetic_tool_payload, ensure_ascii=False),
                        "tool_call_id": None,
                    }
                ],
                "final_ai_content": final_ai_content,
            },
        }

    raw_keyframe_id = background_result.get("recommended_keyframe_id")
    try:
        keyframe_id = int(raw_keyframe_id)
    except Exception:
        return None

    reason = truncate_context_text(
        background_result.get("recommendation_reason")
        or background_result.get("summary")
        or background_result.get("final_output"),
        limit=320,
    )
    destination = {"destination": {"type": "keyframe", "keyframe_id": keyframe_id}}
    destination_json = json.dumps(destination, ensure_ascii=False)
    summary = (
        f"Resolved destination from background preanalysis: keyframe {keyframe_id}."
    )
    if reason:
        summary += f" Reason: {reason}"
    final_ai_content = f"{summary}\n\n{destination_json}"
    synthetic_tool_payload = {
        "status": "ok",
        "summary": summary,
        "data": {
            "source": "background_preanalysis",
            "destination": destination["destination"],
            "recommended_keyframe_id": keyframe_id,
            "recommendation_confidence": background_result.get(
                "recommendation_confidence"
            ),
            "candidate_keyframe_ids": background_result.get("candidate_keyframe_ids"),
        },
    }

    return {
        "event_type": "task_completed",
        "summary": final_ai_content,
        "tool_name": "background_preanalysis",
        "tool_trace": {
            "tool_calls": [],
            "tool_results": [
                {
                    "name": "background_preanalysis",
                    "content": json.dumps(synthetic_tool_payload, ensure_ascii=False),
                    "tool_call_id": None,
                }
            ],
            "final_ai_content": final_ai_content,
        },
    }


def _build_navigation_memory_context_for_grounding(
    current_task: Optional[TaskItem],
    *,
    tasks: dict[int, TaskItem],
    run_memory: Optional[Any],
) -> str:
    """Inject a compact navigation table for reusable destination grounding."""

    if run_memory is None:
        return ""
    if not _task_produces_reusable_destination_for_navigation(current_task, tasks):
        return ""
    try:
        table = run_memory.query_memory(
            scope="navigation",
            view="summary_table",
            query=str((current_task or {}).get("description") or ""),
            time="all",
            limit=50,
        )
    except Exception:
        return ""

    items = list(table.get("items") or [])
    if not items:
        return (
            "\n--- NAVIGATION MEMORY TABLE ---\n"
            "No prior navigation anchors are available for this session.\n"
            "Proceed to scene-memory search if the destination still needs resolving.\n"
            "--- END NAVIGATION MEMORY TABLE ---\n"
        )

    compact_rows: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        compact_rows.append(
            {
                "row_id": item.get("row_id"),
                "description": item.get("preview"),
                "keyframe_id": item.get("keyframe_id"),
                "position": item.get("position"),
                "order": item.get("order"),
                "time": item.get("time"),
            }
        )

    if not compact_rows:
        return ""

    return (
        "\n--- NAVIGATION MEMORY TABLE ---\n"
        "This reusable destination signal task has already been given the compact navigation summary table.\n"
        "Use these rows only for visited-place reuse. If no row clearly matches, do not query other memory scopes; proceed to scene-memory search.\n"
        f"{json.dumps(compact_rows, ensure_ascii=False, indent=2)}\n"
        "--- END NAVIGATION MEMORY TABLE ---\n"
    )


def _tools_for_current_execute_task(
    current_task: Optional[TaskItem],
    execution_tools: Sequence[BaseTool],
) -> list[BaseTool]:
    """Apply the minimal tool boundary for the new task schema."""

    task_type = str((current_task or {}).get("task_type") or "").strip()
    image_refs = list((current_task or {}).get("image_refs") or [])
    destination_is_in_contract = _task_outputs_signal(current_task, "destination")
    if task_type == "navigation_action":
        target = (current_task or {}).get("target")
        target_type = (
            str((target or {}).get("type") or "").strip()
            if isinstance(target, dict)
            else ""
        )
        allowed_names = set(NAVIGATION_TOOL_NAMES)
        if target_type == "semantic_keyframe":
            allowed_names.update(KEYFRAME_SEARCH_TOOL_NAMES)
            if isinstance(target, dict) and str(target.get("target_source") or "").strip() == "attached_image":
                allowed_names.update({"match_attached_image_to_keyframes"})
        elif target_type == "semantic_object":
            allowed_names.update({"approach_object_in_current_view"})
        return [
            tool
            for tool in execution_tools
            if str(getattr(tool, "name", "") or "").strip() in allowed_names
        ]
    allowed = []
    for tool in execution_tools:
        tool_name = str(getattr(tool, "name", "") or "").strip()
        if tool_name in NAVIGATION_TOOL_NAMES:
            continue
        if tool_name in BACKGROUND_ONLY_TOOL_NAMES:
            continue
        if tool_name in DIAGNOSTIC_ONLY_TOOL_NAMES:
            continue
        if tool_name in ATTACHED_IMAGE_TOOL_NAMES and not image_refs:
            continue
        if tool_name in SEMANTIC_GROUNDING_TOOL_NAMES and not destination_is_in_contract:
            continue
        allowed.append(tool)
    return allowed


def _canonical_tool_call_signature(tool_call: dict[str, Any]) -> str:
    """Return a stable comparable signature for one LLM-requested tool call."""

    try:
        args = tool_call.get("args") or {}
        args_text = json.dumps(args, ensure_ascii=False, sort_keys=True, default=str)
    except Exception:
        args_text = str(tool_call.get("args") or {})
    return f"{tool_call.get('name')}:{args_text}"


def _dedupe_repeated_tool_pairs_for_llm(messages: Sequence[BaseMessage]) -> list[BaseMessage]:
    """Hide repeated identical tool-call pairs from the next provider request.

    Some providers reject histories that contain the same tool call with the
    same arguments across multiple consecutive rounds. The full trace remains
    available in executor state; this only trims the LLM-facing prompt input so
    the model can use the first result and finish the task.
    """

    seen_signatures: set[str] = set()
    skip_tool_call_ids: set[str] = set()
    filtered: list[BaseMessage] = []
    removed_count = 0

    for msg in messages:
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            tool_calls = list(msg.tool_calls or [])
            signatures = [_canonical_tool_call_signature(call) for call in tool_calls]
            if signatures and all(signature in seen_signatures for signature in signatures):
                for call in tool_calls:
                    call_id = str(call.get("id") or "").strip()
                    if call_id:
                        skip_tool_call_ids.add(call_id)
                removed_count += 1
                continue
            seen_signatures.update(signatures)
            filtered.append(msg)
            continue

        if isinstance(msg, ToolMessage):
            tool_call_id = str(getattr(msg, "tool_call_id", "") or "").strip()
            if tool_call_id and tool_call_id in skip_tool_call_ids:
                removed_count += 1
                continue
        filtered.append(msg)

    if removed_count > 0:
        filtered.append(
            SystemMessage(
                content=(
                    "Runtime guard: repeated identical tool-call turns were hidden "
                    "from this model request. Use the first available result for "
                    "that query; do not call the exact same tool with identical "
                    "arguments again unless the arguments change."
                )
            )
        )

    return filtered

def create_execute_node(
    llm: BaseChatModel,
    tools: Sequence[BaseTool],
    tool_node: ToolNode,
    shared_background_results: dict,
    shared_runtime_control: Optional[dict[str, Any]] = None,
    logger: Optional[Any] = None,
    run_memory: Optional[Any] = None,
):
    """Execute the current task using a ReAct-style agent with shared context."""

    del tool_node
    execution_tools = list(tools) + build_precision_support_tools(tools)

    def _build_execute_prompt(
        *,
        current_task: Optional[TaskItem],
        selected_execution_context_packet: dict[str, Any],
        background_context: str,
    ) -> str:
        """Assemble the fixed execute prompt from deterministic sections."""
        allowed_tools = _tools_for_current_execute_task(current_task, execution_tools)
        tool_catalog_text = build_tool_catalog(allowed_tools)
        current_task_payload: dict[str, Any] | str = (
            selected_execution_context_packet.get("current_task")
            if current_task is not None
            else "No active task"
        )
        execution_contract = build_execution_guide(
            current_task,
            selected_execution_context_packet,
        )
        prompt_lines = [
            "You are a sub-task executor.",
            "TASK CONTRACT:",
            "Solve only the current task. Do not implicitly complete future tasks.",
            "MINIMAL EXECUTION CONTEXT:",
            json.dumps(current_task_payload, ensure_ascii=False, indent=2),
            "EXECUTION CONTRACT:",
            execution_contract,
            "CONTINUITY CONTEXT:",
            json.dumps(selected_execution_context_packet, ensure_ascii=False, indent=2),
            "ALLOWED TOOLS:",
            tool_catalog_text,
        ]
        if is_navigation_action(current_task):
            prompt_lines.extend(
                [
                    "NAVIGATION RULES:",
                    "This is a navigation task.",
                    "After a successful navigation command, stop immediately and wait for arrival.",
                    "Do not answer downstream perception or reporting questions in this task.",
                ]
            )
        if selected_execution_context_packet.get("arrival_context") is not None:
            prompt_lines.append(
                "Arrival context is only a reference anchor for this task."
            )
        if selected_execution_context_packet.get("upstream_tasks"):
            prompt_lines.append(
                "Reuse upstream evidence when it already resolves the current task."
            )
        prompt_lines.extend(
            [
                "EXECUTION RULES:",
                "- Treat continuity context as small working memory, not full history.",
                "- If historical facts are needed, call query_memory with the narrowest scope and view.",
                "- query_memory only covers the currently loaded run session. Do not pass the current plan_id unless the user explicitly asks about the current plan; for earlier facts within this run session leave plan_id empty.",
                "- After query_memory summary_table or timeline returns row_id values, use row_id for follow-up detail queries. task_id can repeat across plans and is not a session-global row identifier.",
                "- If a query_memory detail lookup returns no items, do not repeat the same detail lookup. Use the summary/timeline rows you already have or move to scene-memory/tool evidence.",
                "- When the current task has a structured result such as destination, selected_object, visual_observation, or current_place_context, submit it with submit_task_result. Do not invent tools named destination, selected_object, or observation.",
                "- Do not answer historical navigation, task, plan, conversation, or observation questions by guessing from prompt context.",
                "- When a deterministic helper tool is available for numeric computation, use it instead of mental arithmetic.",
            ]
        )
        if background_context:
            prompt_lines.append(background_context.rstrip())
        return "\n".join(prompt_lines) + "\n"

    def prepare_execute_inputs(state: AsyncAgentState) -> dict[str, Any]:
        """Collect current task, context bundle, guards, and prompt inputs."""

        messages = state.get("messages", [])
        tasks = state.get("tasks", {})
        current_task_id = state.get("current_task_id")
        current_task = _get_current_task(tasks, current_task_id)
        existing_events = list(state.get("events", []))

        candidate_background_result: BackgroundAnalysisItem | str | None = None
        if current_task and current_task.get("task_id", -1) >= 0:
            task_id = current_task["task_id"]
            shared_candidate = shared_background_results.get(task_id)
            state_candidate = state.get("background_results", {}).get(task_id)
            candidate_background_result = shared_candidate or state_candidate
            if (
                task_depends_on_query_result(current_task, tasks)
                and candidate_background_result is not None
                and not background_result_is_reusable_for_task(candidate_background_result)
            ):
                candidate_background_result = None
                if logger:
                    logger.log_foreground(
                        "Execute: Ignoring non-actionable background analysis for task {task_id} because its target depends on upstream task evidence.".format(
                            task_id=task_id,
                        )
                    )

        background_context = build_background_reference(candidate_background_result)
        prepared_context = prepare_context_bundle(
            state,
            current_task,
            run_memory=run_memory,
        )
        selected_execution_context_packet = dict(
            prepared_context.get("selected_execution_context_packet") or {}
        )
        plan_context = _build_execute_prompt(
            current_task=current_task,
            selected_execution_context_packet=selected_execution_context_packet,
            background_context=background_context,
        )
        navigation_memory_context = _build_navigation_memory_context_for_grounding(
            current_task,
            tasks=tasks,
            run_memory=run_memory,
        )
        if navigation_memory_context:
            plan_context = plan_context + navigation_memory_context

        return {
            "messages": messages,
            "tasks": tasks,
            "current_task_id": current_task_id,
            "current_task": current_task,
            "existing_events": existing_events,
            "selected_execution_context_packet": selected_execution_context_packet,
            "plan_context": plan_context,
            "background_result": candidate_background_result,
        }

    def run_execute_agent(execute_inputs: dict[str, Any]) -> dict[str, Any]:
        """Run structured navigation dispatch or a ReAct pass."""

        current_task = execute_inputs.get("current_task")
        allowed_tools = _tools_for_current_execute_task(current_task, execution_tools)
        agent_messages: list[BaseMessage] = []
        selected_packet_for_tools = dict(
            execute_inputs.get("selected_execution_context_packet") or {}
        )

        background_fast_path = None
        if not is_navigation_action(current_task):
            background_fast_path = _try_complete_semantic_grounding_from_background(
                current_task,
                tasks=dict(execute_inputs.get("tasks") or {}),
                background_result=execute_inputs.get("background_result"),
            )
        if background_fast_path is not None:
            if logger and current_task:
                logger.log_foreground(
                    "Execute: Reusing completed background preanalysis for task {task_id}.".format(
                        task_id=current_task.get("task_id")
                    )
                )
            return {
                "agent_messages": agent_messages,
                "deterministic_outcome": background_fast_path,
                "execution_error": None,
            }

        tool_context = {
            "current_task": current_task,
            "tasks": dict(execute_inputs.get("tasks") or {}),
            "shared_background_results": shared_background_results,
            "shared_runtime_control": shared_runtime_control,
            "background_result": execute_inputs.get("background_result"),
            "selected_execution_context_packet": selected_packet_for_tools,
            "logger": logger,
            "navigation_start_callback": execute_inputs.get("navigation_start_callback"),
        }
        with execute_tool_budget_context(), runtime_tool_context(tool_context):
            deterministic_outcome = _try_dispatch_structured_navigation_action(
                current_task,
                tasks=dict(execute_inputs.get("tasks") or {}),
                tools=allowed_tools,
            )
        if deterministic_outcome is not None:
            if logger and current_task:
                logger.log_foreground(
                    "Execute: Handled structured navigation_action for task {task_id} without ReAct planning.".format(
                        task_id=current_task.get("task_id")
                    )
                )
            print(
                f"{Colors.REACT}Execute: Handled structured navigation action{Colors.RESET}"
            )
            return {
                "agent_messages": agent_messages,
                "deterministic_outcome": deterministic_outcome,
                "execution_error": None,
            }
        if is_navigation_action(current_task):
            return {
                "agent_messages": agent_messages,
                "deterministic_outcome": {
                    "event_type": "task_failed",
                    "summary": "Structured navigation_action could not be dispatched from its target contract.",
                    "tool_name": None,
                    "tool_trace": {},
                },
                "execution_error": None,
            }

        background_outcome = _try_complete_semantic_grounding_from_background(
            current_task,
            tasks=dict(execute_inputs.get("tasks") or {}),
            background_result=execute_inputs.get("background_result"),
        )
        if background_outcome is not None:
            if logger and current_task:
                logger.log_foreground(
                    "Execute: Used completed background preanalysis for semantic grounding task {task_id}.".format(
                        task_id=current_task.get("task_id")
                    )
                )
            return {
                "agent_messages": agent_messages,
                "deterministic_outcome": background_outcome,
                "execution_error": None,
            }

        execution_error: Optional[Exception] = None

        def _executor_pre_model_hook(state: dict[str, Any]) -> dict[str, Any]:
            messages = list((state or {}).get("messages") or [])
            return {"llm_input_messages": _dedupe_repeated_tool_pairs_for_llm(messages)}

        react_agent = create_react_agent(
            model=llm,
            tools=allowed_tools,
            prompt=AGENT_PROMPTS.get("react_system", ""),
            pre_model_hook=_executor_pre_model_hook,
            logger=logger.log_foreground if logger else None,
        )

        def stream_executor_pass(system_prompt: str) -> None:
            react_input = {
                "messages": [
                    SystemMessage(content=system_prompt),
                ]
            }
            for chunk in react_agent.stream(react_input, stream_mode="values"):
                if "messages" not in chunk:
                    continue
                for msg in chunk["messages"]:
                    if msg in agent_messages:
                        continue
                    agent_messages.append(msg)
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        for tool_call in msg.tool_calls:
                            if logger:
                                logger.log_foreground(
                                    f"Execute: Tool Call to {tool_call['name']} with args: {tool_call['args']}"
                                )
                            print(
                                f"{Colors.TOOL}Tool Call:{Colors.RESET} {tool_call['name']} with args: {tool_call['args']}"
                            )
                    if isinstance(msg, ToolMessage):
                        if logger:
                            logger.log_foreground(
                                f"Execute: Tool Result from {msg.name}: {str(msg.content)[:200]}..."
                            )
                        print(
                            f"{Colors.TOOL}Tool Result:{Colors.RESET} {msg.name} returned: {str(msg.content)[:200]}..."
                        )

        try:
            with execute_tool_budget_context(), runtime_tool_context(tool_context):
                stream_executor_pass(str(execute_inputs.get("plan_context") or ""))
        except Exception as exc:
            print(f"{Colors.REACT}Execute Error:{Colors.RESET} {str(exc)}")
            traceback.print_exc()
            execution_error = exc

        return {
            "agent_messages": agent_messages,
            "deterministic_outcome": None,
            "execution_error": execution_error,
        }

    def classify_execute_result(
        execute_inputs: dict[str, Any],
        execute_run: dict[str, Any],
    ) -> dict[str, Any]:
        """Normalize execute outputs into task_waiting/task_completed/task_failed."""

        current_task = execute_inputs.get("current_task")
        deterministic_outcome = execute_run.get("deterministic_outcome")
        if deterministic_outcome is not None:
            tool_trace = dict(deterministic_outcome.get("tool_trace", {}))
            navigation_command_count = count_successful_navigation_commands(
                tool_trace,
                navigation_tool_names=NAVIGATION_TOOL_NAMES,
            )
            event_type = "task_waiting" if navigation_command_count > 0 else "task_completed"
            if deterministic_outcome.get("event_type"):
                event_type = str(deterministic_outcome.get("event_type"))
            return {
                "agent_messages": list(execute_run.get("agent_messages") or []),
                "tool_trace": tool_trace,
                "event_type": event_type,
                "summary": str(
                    deterministic_outcome.get("summary")
                    or navigation_arrival_summary(current_task)
                ),
                "primary_tool_name": deterministic_outcome.get("tool_name"),
                "navigation_command_count": navigation_command_count,
            }

        agent_messages = list(execute_run.get("agent_messages") or [])
        execution_error = execute_run.get("execution_error")
        tool_trace = extract_tool_trace(agent_messages)
        tool_contract_violation = find_tool_contract_violation_message(tool_trace)
        failure_summary = find_tool_failure_message(tool_trace)
        navigation_command_count = count_successful_navigation_commands(
            tool_trace,
            navigation_tool_names=NAVIGATION_TOOL_NAMES,
        )
        final_ai_content = str(tool_trace.get("final_ai_content") or "").strip()
        submitted_destination = _task_signal_from_trace(tool_trace, "destination")
        has_submitted_result = _has_submitted_task_result(tool_trace)

        if execution_error is not None:
            event_type = "task_failed"
            summary = f"Task execution failed with exception: {str(execution_error)}"
        elif tool_contract_violation and not (
            submitted_destination is not None
            and _task_produces_reusable_destination_for_navigation(
                current_task,
                dict(execute_inputs.get("tasks") or {}),
            )
        ):
            event_type = "task_failed"
            summary = tool_contract_violation
        elif navigation_command_count > 1:
            event_type = "task_failed"
            summary = (
                "Task execution issued multiple navigation commands in a single task. "
                "Each task must resolve to exactly one destination before waiting for arrival."
            )
        elif issued_navigation_command(
            tool_trace,
            navigation_tool_names=NAVIGATION_TOOL_NAMES,
        ):
            event_type = "task_waiting"
            summary = navigation_waiting_summary(current_task)
        elif _tool_failure_blocks_task(
            current_task,
            dict(execute_inputs.get("tasks") or {}),
            tool_trace,
            failure_summary=failure_summary,
            final_ai_content=final_ai_content,
        ):
            event_type = "task_failed"
            summary = failure_summary or "Task failed."
        else:
            event_type = "task_completed"
            summary = (
                _submitted_task_result_summary(tool_trace)
                if has_submitted_result
                else ""
            ) or (
                final_ai_content
                or "Task completed successfully."
            )

        primary_tool_name = None
        if tool_trace.get("tool_calls"):
            primary_tool_name = tool_trace["tool_calls"][-1].get("name")
        elif tool_trace.get("tool_results"):
            primary_tool_name = tool_trace["tool_results"][-1].get("name")

        return {
            "agent_messages": agent_messages,
            "tool_trace": tool_trace,
            "event_type": event_type,
            "summary": summary,
            "primary_tool_name": primary_tool_name,
            "navigation_command_count": navigation_command_count,
        }

    def execute_node(state: AsyncAgentState) -> AsyncAgentState:
        """Run the executor for the current task and emit a structured execution event."""

        state = _strip_ignored_state_fields(state)
        if logger:
            logger.log_foreground("Execute: Starting task execution")
            print(f"{Colors.REACT}Execute: Starting task execution{Colors.RESET}")
        messages = state.get("messages", [])
        tasks = state.get("tasks", {})
        current_task_id = state.get("current_task_id")
        current_task = _get_current_task(tasks, current_task_id)
        existing_events = list(state.get("events", []))
        execution_error: Optional[Exception] = None

        if current_task:
            if (
                shared_runtime_control is not None
                and current_task_id is not None
                and current_task.get("plan_id") == state.get("current_plan_id")
            ):
                record_foreground_task(shared_runtime_control, int(current_task_id))
            progress_context = get_task_progress_context(
                tasks,
                current_task_id=current_task_id,
                current_plan_id=state.get("current_plan_id"),
            )
            task_label = (
                f"task #{current_task_id}"
                if current_task_id is not None and current_task_id >= 0
                else "runtime task"
            )
            progress_label = ""
            if progress_context is not None:
                progress_label = (
                    f" (step {progress_context['position']}/{progress_context['total']})"
                )
            if logger:
                logger.log_foreground(
                    f"Execute: Executing {task_label}{progress_label}: {current_task['description']}"
                )
                print(
                    f"{Colors.REACT}Execute: Executing {task_label}{progress_label}:{Colors.RESET} {current_task['description']}"
                )
        else:
            user_message = next(
                (m for m in reversed(messages) if isinstance(m, HumanMessage)), None
            )
            if not user_message:
                return state
            if logger:
                logger.log_foreground(
                    f"Execute: Executing simple task: {user_message.content[:100]}..."
                )
                print(
                    f"{Colors.REACT}Execute: Executing simple task:{Colors.RESET} {user_message.content[:100]}..."
                )

        try:
            execute_inputs = prepare_execute_inputs(state)
        except Exception:
            execute_inputs = {
                "messages": messages,
                "tasks": tasks,
                "current_task_id": current_task_id,
                "current_task": current_task,
                "existing_events": existing_events,
                "plan_context": "Task plan context unavailable.",
            }

        def navigation_start_callback(event: dict[str, Any]) -> None:
            """Publish a UI guidance event as soon as a navigation tool dispatches."""

            nonlocal state, tasks
            if not isinstance(event, dict):
                return
            text = str(event.get("text") or "").strip()
            if not text:
                return
            raw_task_id = event.get("task_id")
            try:
                callback_task_id = int(raw_task_id)
            except Exception:
                callback_task_id = current_task_id
            state = append_guidance(
                {**state, "tasks": tasks},
                event_type="navigation_start",
                text=text,
                priority="normal",
                interrupt=False,
                dedupe_key=str(event.get("dedupe_key") or f"navigation_start:{callback_task_id}"),
                task_id=callback_task_id,
                payload={
                    "tool_name": event.get("tool_name"),
                    "nav_args": event.get("nav_args"),
                    "dispatch_timing": "tool_dispatch_start",
                },
            )
            tasks = state.get("tasks", tasks)
            if callable(on_update):
                try:
                    on_update(
                        {
                            "node_name": "execute",
                            "node_state": {
                                "guidance_events": list(state.get("guidance_events") or []),
                                "tasks": tasks,
                            },
                            "step_summary": {
                                "node": "execute",
                                "latest_event": {
                                    "type": "navigation_start",
                                    "task_id": callback_task_id,
                                    "summary": text,
                                },
                            },
                            "visited_nodes": ["execute"],
                            "step_trace": [],
                            "state": state,
                        }
                    )
                except Exception:
                    pass

        execute_inputs["navigation_start_callback"] = navigation_start_callback

        task_ref = None
        if current_task_id is not None and current_task_id in tasks:
            task_ref = tasks[current_task_id]
            task_ref["status"] = "running"
            task_ref["updated_at"] = now_iso()
            if is_navigation_action(task_ref):
                state = append_guidance(
                    {**state, "tasks": tasks},
                    event_type="navigation_start",
                    text=navigation_waiting_text(task_ref),
                    priority="normal",
                    interrupt=False,
                    dedupe_key=_navigation_guidance_dedupe_key(task_ref, current_task_id),
                    task_id=int(current_task_id),
                )
                tasks = state.get("tasks", tasks)
                task_ref = tasks[current_task_id]

        execute_run = run_execute_agent(execute_inputs)
        classified_result = classify_execute_result(execute_inputs, execute_run)
        agent_messages = list(classified_result.get("agent_messages") or [])
        tool_trace = dict(classified_result.get("tool_trace") or {})
        navigation_command_count = int(
            classified_result.get("navigation_command_count") or 0
        )
        if navigation_command_count > 0 and shared_runtime_control is not None:
            set_background_enabled(shared_runtime_control, True)

        emitted_event_id = new_structured_id("event")
        event_type = str(classified_result.get("event_type") or "task_completed")
        summary = str(
            classified_result.get("summary") or "Task completed successfully."
        )
        primary_tool_name = classified_result.get("primary_tool_name")

        raw_output = json.dumps(tool_trace, ensure_ascii=False)
        task_destination = _task_signal_from_trace(tool_trace, "destination")
        if task_destination is None and _has_submitted_task_result(tool_trace):
            task_destination = _destination_from_keyframe_search_trace(tool_trace)
        task_destination_description = _task_text_signal_from_trace(
            tool_trace,
            "destination_description",
        )
        task_selected_object = _task_signal_from_trace(tool_trace, "selected_object")
        task_visual_observation = _task_signal_from_trace(tool_trace, "visual_observation")
        task_current_place_context = _task_signal_from_trace(tool_trace, "current_place_context")
        user_facing_response = build_task_user_facing_response(
            current_task,
            event_type=event_type,
            summary=summary,
        )
        turn_response_type = build_task_turn_response_type(
            current_task,
            event_type=event_type,
            summary=summary,
        )

        if task_ref is not None:
            if event_type == "task_waiting":
                task_ref["status"] = "waiting"
                task_ref["wait_for_event"] = "navigation_arrived"
            elif event_type == "task_failed":
                task_ref["status"] = "failed"
                task_ref["terminal_reason"] = summary
            else:
                task_ref["status"] = "completed"
            append_task_result(
                task_ref,
                event_id=emitted_event_id,
                summary=summary,
                raw_output=raw_output,
                tool_name=primary_tool_name,
                destination=task_destination,
                destination_description=task_destination_description,
                selected_object=task_selected_object,
                visual_observation=task_visual_observation,
                current_place_context=task_current_place_context,
            )
            try:
                run_memory.record_task_result(
                    task=task_ref,
                    event_type=event_type,
                    summary=summary,
                    tool_trace=tool_trace,
                )
                run_memory.record_tool_trace(
                    task=task_ref,
                    tool_trace=tool_trace,
                )
            except Exception:
                pass
            try:
                _record_session_anchors_from_tool_trace(
                    shared_runtime_control=shared_runtime_control,
                    current_task=task_ref,
                    tool_trace=tool_trace,
                    logger=logger,
                )
            except Exception as exc:
                if logger:
                    try:
                        logger.log_foreground(
                            f"session_anchor_record_failed: {type(exc).__name__}: {exc}"
                        )
                    except Exception:
                        pass

        if logger:
            logger.log_foreground(
                "Execute Debug: classified event_type={event_type}, task_id={task_id}, tool_calls={tool_calls}, navigation_command_count={navigation_command_count}, summary={summary}".format(
                    event_type=event_type,
                    task_id=current_task_id if task_ref is not None else -1,
                    tool_calls=[call.get("name") for call in tool_trace.get("tool_calls", [])],
                    navigation_command_count=navigation_command_count,
                    summary=summary,
                )
            )
            print(
                f"{Colors.REACT}Execute Debug:{Colors.RESET} "
                f"event_type={event_type} task_id={current_task_id if task_ref is not None else -1}"
            )

        execution_event: EventItem = {
            "event_id": emitted_event_id,
            "type": event_type,
            "source": "executor",
            "created_at": now_iso(),
            "task_id": current_task_id if task_ref is not None else -1,
            "payload": {
                "summary": summary,
                "tool_name": primary_tool_name,
            },
        }
        if user_facing_response and event_type != "task_waiting":
            execution_event["payload"]["user_facing_response"] = user_facing_response
        if turn_response_type and event_type != "task_waiting":
            execution_event["payload"]["turn_response_type"] = turn_response_type
        if task_ref and task_ref.get("user_input_id"):
            execution_event["user_input_id"] = task_ref["user_input_id"]
        _record_run_memory_event(
            run_memory,
            execution_event,
            stage="execute",
        )

        active_navigation_snapshot = None
        if (
            task_ref is not None
            and event_type == "task_waiting"
            and str(primary_tool_name or "") in {"go_to_keyframe", "go_to_position"}
        ):
            active_navigation_snapshot = build_pending_navigation_snapshot(
                task_ref,
                task_id=int(current_task_id),
                created_at=str(execution_event.get("created_at") or ""),
            )

        result_state: AsyncAgentState = {
            **state,
            "tasks": tasks,
            "events": existing_events + [execution_event],
            "next_action": {"type": "idle"},
            "messages": state["messages"] + agent_messages,
        }
        if active_navigation_snapshot is not None:
            result_state["active_navigation"] = active_navigation_snapshot
            result_state["pending_navigation"] = active_navigation_snapshot
        if event_type == "task_waiting":
            return result_state
        return apply_user_facing_response(
            result_state,
            user_facing_response,
            response_type=turn_response_type,
        )

    return execute_node
