"""Execution-result and user-facing-response helpers for the async agent."""

from __future__ import annotations

import ast
import json
import math
from typing import Any, Optional, Sequence

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.tools import BaseTool, StructuredTool

from .context import (
    is_navigation_action,
)
from .tool_results import (
    extract_structured_tool_result,
    extract_tool_result_data,
    extract_tool_result_status,
    is_structured_tool_result,
)
from ..orchestration.runtime import (
    new_structured_id,
    now_iso,
)
from ..runtime.types import (
    AsyncAgentState,
    TaskItem,
    TaskResultItem,
    TurnResponseItem,
    TurnResponseType,
)


def stringify_tool_content(content: Any) -> str:
    """Normalize tool content into a comparable string."""

    if isinstance(content, str):
        return content
    try:
        return json.dumps(content, ensure_ascii=False)
    except Exception:
        return str(content)


def extract_tool_trace(agent_messages: Sequence[BaseMessage]) -> dict[str, Any]:
    """Collect tool calls and tool results from agent messages."""

    tool_calls: list[dict[str, Any]] = []
    tool_results: list[dict[str, Any]] = []
    final_ai_content = ""

    for msg in agent_messages:
        if isinstance(msg, AIMessage):
            if msg.content:
                final_ai_content = str(msg.content)
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    tool_calls.append(
                        {
                            "id": tool_call.get("id"),
                            "name": tool_call.get("name"),
                            "args": tool_call.get("args"),
                        }
                    )
        elif isinstance(msg, ToolMessage):
            tool_results.append(
                {
                    "name": msg.name,
                    "content": stringify_tool_content(msg.content),
                    "tool_call_id": getattr(msg, "tool_call_id", None),
                }
            )

    return {
        "tool_calls": tool_calls,
        "tool_results": tool_results,
        "final_ai_content": final_ai_content,
    }


def tool_content_indicates_error(content: str) -> bool:
    """Detect obvious tool-level error strings."""

    structured_result = extract_structured_tool_result(content)
    if structured_result is not None:
        return str(structured_result.get("status") or "").strip().lower() == "error"

    lowered = content.lower()
    return (
        lowered.startswith("error:")
        or " error:" in lowered
        or lowered.startswith("failed:")
        or " exception" in lowered
        or "traceback" in lowered
    )


def issued_navigation_command(
    tool_trace: dict[str, Any],
    *,
    navigation_tool_names: set[str],
) -> bool:
    """Detect whether a navigation command was successfully sent."""

    for tool_result in tool_trace.get("tool_results", []):
        if tool_result.get("name") not in navigation_tool_names:
            continue
        content = tool_result.get("content", "")
        structured_result = extract_structured_tool_result(content)
        if structured_result is not None:
            status = str(structured_result.get("status") or "").strip().lower()
            data = structured_result.get("data") or {}
            if status != "ok":
                continue
            if isinstance(data, dict) and (
                data.get("planned_path") or data.get("target_keyframe_id") is not None
            ):
                return True
            continue

        lowered = content.lower()
        if tool_content_indicates_error(content):
            continue
        if "sent to controller" in lowered or "path to keyframe" in lowered:
            return True
    return False


def find_tool_failure_message(tool_trace: dict[str, Any]) -> Optional[str]:
    """Return the latest unrecovered tool-level error string, if any."""

    for tool_result in reversed(tool_trace.get("tool_results", [])):
        content = tool_result.get("content", "")
        structured_result = extract_structured_tool_result(content)
        if structured_result is not None:
            status = str(structured_result.get("status") or "").strip().lower()
            if status in {"blocked", "error"}:
                summary = str(structured_result.get("summary") or "").strip()
                error = structured_result.get("error")
                if isinstance(error, dict):
                    error_message = str(error.get("message") or "").strip()
                else:
                    error_message = str(error or "").strip()
                message = error_message or summary or stringify_tool_content(content)
                return f"{tool_result.get('name')}: {message}"
            continue
        if tool_content_indicates_error(content):
            return f"{tool_result.get('name')}: {content}"
    return None


def count_successful_navigation_commands(
    tool_trace: dict[str, Any],
    *,
    navigation_tool_names: set[str],
) -> int:
    """Count successful navigation tool invocations in one execution turn."""

    count = 0
    for tool_result in tool_trace.get("tool_results", []):
        if tool_result.get("name") not in navigation_tool_names:
            continue
        content = tool_result.get("content", "")
        structured_result = extract_structured_tool_result(content)
        if structured_result is not None:
            status = str(structured_result.get("status") or "").strip().lower()
            data = structured_result.get("data") or {}
            if (
                status == "ok"
                and isinstance(data, dict)
                and (
                    data.get("planned_path")
                    or data.get("target_keyframe_id") is not None
                )
            ):
                count += 1
            continue
        if tool_content_indicates_error(content):
            continue
        lowered = content.lower()
        if "sent to controller" in lowered or "path to keyframe" in lowered:
            count += 1
    return count


def append_task_result(
    task: TaskItem,
    *,
    event_id: str,
    summary: str,
    raw_output: Optional[str] = None,
    tool_name: Optional[str] = None,
    decision: Optional[str] = None,
) -> None:
    """Append a structured result entry to a task."""

    result_entries = list(task.get("result", []))
    entry: TaskResultItem = {
        "event_id": event_id,
        "summary": summary,
        "created_at": now_iso(),
    }
    if raw_output is not None:
        entry["raw_output"] = raw_output
    if tool_name is not None:
        entry["tool_name"] = tool_name
    if decision is not None:
        entry["decision"] = decision
    result_entries.append(entry)
    task["result"] = result_entries
    task["updated_at"] = now_iso()


def task_destination_label(task: Optional[TaskItem]) -> str:
    """Return a user-facing destination label derived from the task description."""

    if not task:
        return "the destination"

    description = str(task.get("description") or "").strip()
    if not description:
        return "the destination"

    lowered = description.lower()
    prefixes = (
        "navigate to ",
        "go to ",
        "drive to ",
        "head to ",
        "move to ",
        "proceed to ",
        "return to ",
        "travel to ",
        "guide me to ",
    )
    for prefix in prefixes:
        if lowered.startswith(prefix):
            return description[len(prefix):].strip() or "the destination"

    return description


def ensure_sentence_ending(text: str) -> str:
    """Ensure free-form text ends like a sentence for readable summaries."""

    normalized = text.strip()
    if not normalized:
        return normalized
    if normalized[-1] in ".!?":
        return normalized
    return f"{normalized}."


def navigation_state_for_task(task: Optional[TaskItem]) -> Optional[str]:
    """Return a compact navigation lifecycle label for prompt evidence."""

    if not is_navigation_action(task):
        return None

    if not task:
        return None

    status = task.get("status")
    if status == "waiting" and task.get("wait_for_event") == "navigation_arrived":
        return "in_transit"
    if status == "completed":
        return "arrived"
    if status == "failed":
        return "failed"
    return str(status) if status else "unknown"


def navigation_waiting_summary(task: Optional[TaskItem]) -> str:
    """Build a task-result summary for an issued navigation command."""

    destination_label = task_destination_label(task)
    return f"I am heading to {destination_label}."


def navigation_arrival_summary(task: Optional[TaskItem]) -> str:
    """Build a task-result summary for a confirmed navigation arrival."""

    destination_label = task_destination_label(task)
    return f"I have arrived at {destination_label}."


def task_summary_is_user_visible(
    task: Optional[TaskItem],
    summary: Optional[str],
    *,
    event_type: Optional[str] = None,
) -> bool:
    """Return True when a task summary should be surfaced to the user session."""

    normalized = str(summary or "").strip()
    if not normalized:
        return False

    lowered = normalized.lower()
    if lowered in {
        "task completed successfully.",
        "navigation command issued successfully; waiting for arrival event.",
    }:
        return False

    if event_type == "task_waiting":
        return False

    if task is not None and is_navigation_action(task):
        if lowered in {
            navigation_waiting_summary(task).lower(),
            navigation_arrival_summary(task).lower(),
        }:
            return False

    return True


def build_task_user_facing_response(
    task: Optional[TaskItem],
    *,
    event_type: str,
    summary: Optional[str],
) -> Optional[str]:
    """Build a user-visible reply from one task execution outcome when appropriate."""

    normalized_summary = str(summary or "").strip()
    if not normalized_summary:
        return None

    if event_type == "task_waiting":
        if task is not None and is_navigation_action(task):
            return navigation_waiting_summary(task)
        return None

    if (
        event_type == "task_completed"
        and task is not None
        and is_navigation_action(task)
        and normalized_summary.lower() == navigation_arrival_summary(task).lower()
    ):
        return normalized_summary

    if event_type not in {"task_completed", "task_failed"}:
        return None

    if not task_summary_is_user_visible(task, normalized_summary, event_type=event_type):
        return None

    return normalized_summary


def build_task_turn_response_type(
    task: Optional[TaskItem],
    *,
    event_type: str,
    summary: Optional[str],
) -> Optional[TurnResponseType]:
    """Classify one task outcome into a stable user-visible reply type."""

    normalized_summary = str(summary or "").strip()
    if not normalized_summary:
        return None

    if event_type == "task_waiting":
        if task is not None and is_navigation_action(task):
            return "progress"
        return None

    if event_type == "task_failed":
        return "error"

    if event_type != "task_completed":
        return None

    if task is not None and is_navigation_action(task):
        if normalized_summary.lower() == navigation_arrival_summary(task).lower():
            return "result"

    if task_summary_is_user_visible(task, normalized_summary, event_type=event_type):
        return "result"

    return None


def clear_turn_response(state: AsyncAgentState) -> AsyncAgentState:
    """Clear the current turn reply contract fields."""

    return {
        **state,
        "turn_response_items": [],
        "turn_response_type": "none",
        "turn_response_text": None,
        "turn_response_id": None,
        "user_facing_response": None,
        "user_facing_response_id": None,
    }


def build_turn_response_item(
    *,
    response_text: str,
    response_type: TurnResponseType,
    response_id: Optional[str] = None,
    created_at: Optional[str] = None,
    source_event_type: Optional[str] = None,
    task_id: Optional[int] = None,
) -> TurnResponseItem:
    """Build one normalized user-visible turn-response item."""

    item: TurnResponseItem = {
        "response_id": response_id or new_structured_id("response"),
        "response_type": response_type,
        "response_text": response_text,
        "created_at": created_at or now_iso(),
    }
    if source_event_type:
        item["source_event_type"] = source_event_type
    if task_id is not None:
        item["task_id"] = task_id
    return item


def normalize_turn_response_items(value: Any) -> list[TurnResponseItem]:
    """Coerce arbitrary stored response-item payloads into typed items."""

    if not isinstance(value, list):
        return []

    normalized_items: list[TurnResponseItem] = []
    for raw_item in value:
        if not isinstance(raw_item, dict):
            continue
        response_text = str(raw_item.get("response_text") or "").strip()
        response_type = str(raw_item.get("response_type") or "").strip().lower()
        if not response_text or response_type not in {"result", "progress", "error"}:
            continue
        task_id: Optional[int] = None
        raw_task_id = raw_item.get("task_id")
        if raw_task_id is not None:
            try:
                task_id = int(raw_task_id)
            except Exception:
                task_id = None
        normalized_items.append(
            build_turn_response_item(
                response_text=response_text,
                response_type=response_type,  # type: ignore[arg-type]
                response_id=str(raw_item.get("response_id") or "").strip() or None,
                created_at=str(raw_item.get("created_at") or "").strip() or None,
                source_event_type=str(raw_item.get("source_event_type") or "").strip() or None,
                task_id=task_id,
            )
        )
    return normalized_items


def derive_headline_turn_response(
    items: Sequence[TurnResponseItem],
) -> tuple[TurnResponseType, Optional[str], Optional[str]]:
    """Derive one headline response from a full per-turn response sequence."""

    normalized_items = [item for item in items if str(item.get("response_text") or "").strip()]
    if not normalized_items:
        return "none", None, None

    for preferred_type in ("error", "result"):
        for item in reversed(normalized_items):
            if item.get("response_type") == preferred_type:
                return (
                    preferred_type,  # type: ignore[return-value]
                    str(item.get("response_text") or "").strip(),
                    str(item.get("response_id") or "").strip() or None,
                )

    latest_item = normalized_items[-1]
    return (
        str(latest_item.get("response_type") or "result"),  # type: ignore[return-value]
        str(latest_item.get("response_text") or "").strip(),
        str(latest_item.get("response_id") or "").strip() or None,
    )


def apply_turn_response(
    state: AsyncAgentState,
    *,
    response_text: Optional[str],
    response_type: Optional[TurnResponseType],
    source_event_type: Optional[str] = None,
    task_id: Optional[int] = None,
) -> AsyncAgentState:
    """Append one formal turn reply and update the derived headline fields."""

    normalized_text = str(response_text or "").strip()
    normalized_type = response_type or ("none" if not normalized_text else None)

    if not normalized_text:
        if normalized_type == "none":
            return clear_turn_response(state)
        return state

    if normalized_type is None:
        normalized_type = "result"

    response_items = normalize_turn_response_items(state.get("turn_response_items"))
    duplicate_item = next(
        (
            item
            for item in response_items
            if str(item.get("response_text") or "").strip() == normalized_text
            and item.get("response_type") == normalized_type
        ),
        None,
    )
    if duplicate_item is not None:
        headline_type, headline_text, headline_id = derive_headline_turn_response(
            response_items
        )
        return {
            **state,
            "turn_response_items": response_items,
            "turn_response_type": headline_type,
            "turn_response_text": headline_text,
            "turn_response_id": headline_id,
            "user_facing_response": headline_text,
            "user_facing_response_id": headline_id,
        }

    new_item = build_turn_response_item(
        response_text=normalized_text,
        response_type=normalized_type,
        source_event_type=source_event_type,
        task_id=task_id,
    )
    response_items.append(new_item)
    headline_type, headline_text, headline_id = derive_headline_turn_response(
        response_items
    )
    return {
        **state,
        "turn_response_items": response_items,
        "turn_response_type": headline_type,
        "turn_response_text": headline_text,
        "turn_response_id": headline_id,
        "user_facing_response": headline_text,
        "user_facing_response_id": headline_id,
    }


def apply_user_facing_response(
    state: AsyncAgentState,
    response: Optional[str],
    *,
    response_type: Optional[TurnResponseType] = None,
    source_event_type: Optional[str] = None,
    task_id: Optional[int] = None,
) -> AsyncAgentState:
    """Write a user-facing response through formal turn-response storage."""

    if response is None and response_type is None:
        return state

    return apply_turn_response(
        state,
        response_text=response,
        response_type=response_type,
        source_event_type=source_event_type,
        task_id=task_id,
    )


def extract_latest_tool_payload(
    tool_trace: dict[str, Any],
    tool_name: str,
) -> Optional[Any]:
    """Return the latest tool result payload for one named tool."""

    for tool_result in reversed(tool_trace.get("tool_results", [])):
        if tool_result.get("name") != tool_name:
            continue

        content = tool_result.get("content")
        structured_data = extract_tool_result_data(content)
        if structured_data is not None:
            return structured_data
        if not isinstance(content, str):
            return content
        try:
            return json.loads(content)
        except Exception:
            try:
                return ast.literal_eval(content)
            except Exception:
                return content

    return None


def format_position_vector(position: Sequence[float]) -> str:
    """Format a 3D position with compact fixed precision."""

    return "[{:.2f}, {:.2f}, {:.2f}]".format(
        float(position[0]),
        float(position[1]),
        float(position[2]),
    )


def build_deterministic_task_answer(
    task: Optional[TaskItem],
    tool_trace: dict[str, Any],
) -> Optional[str]:
    """Build a direct answer from deterministic tool outputs when available."""

    del task
    distance_payload = extract_latest_tool_payload(
        tool_trace,
        "calculate_distance_between_positions",
    )
    if isinstance(distance_payload, dict):
        source_position = distance_payload.get("source_position")
        target_position = distance_payload.get("target_position")
        distance_meters = distance_payload.get("distance_meters")
        if (
            isinstance(source_position, (list, tuple))
            and len(source_position) >= 3
            and isinstance(target_position, (list, tuple))
            and len(target_position) >= 3
            and distance_meters is not None
        ):
            try:
                return (
                    f"My current position is {format_position_vector(source_position)}. "
                    f"The distance to {format_position_vector(target_position)} is {float(distance_meters):.2f} meters."
                )
            except Exception:
                return None

    current_state_payload = extract_latest_tool_payload(tool_trace, "get_current_state")
    if isinstance(current_state_payload, dict):
        position = current_state_payload.get("position")
        status = current_state_payload.get("status")
        if isinstance(position, (list, tuple)) and len(position) >= 3:
            try:
                answer = f"I am currently at position {format_position_vector(position)}."
                normalized_status = str(status or "").strip()
                if normalized_status:
                    answer += f" Current status: {normalized_status}."
                return answer
            except Exception:
                return None

    return None


def calculate_distance_between_positions(
    source_position: Sequence[float],
    target_position: Sequence[float],
) -> str:
    """Return the Euclidean distance between two 3D positions."""

    if len(source_position) != len(target_position):
        raise ValueError("source_position and target_position must have the same dimension")
    if len(source_position) < 2:
        raise ValueError("At least 2 dimensions are required to calculate distance")

    distance = math.sqrt(
        sum(
            (float(source_value) - float(target_value)) ** 2
            for source_value, target_value in zip(source_position, target_position)
        )
    )
    return json.dumps(
        {
            "source_position": [float(value) for value in source_position],
            "target_position": [float(value) for value in target_position],
            "distance_meters": distance,
        },
        ensure_ascii=False,
    )


def build_precision_support_tools(existing_tools: Sequence[BaseTool]) -> list[BaseTool]:
    """Add small deterministic helper tools that improve executor accuracy."""

    existing_tool_names = {
        str(getattr(tool, "name", "")).strip()
        for tool in existing_tools
    }
    support_tools: list[BaseTool] = []

    if "calculate_distance_between_positions" not in existing_tool_names:
        support_tools.append(
            StructuredTool.from_function(
                func=calculate_distance_between_positions,
                name="calculate_distance_between_positions",
                description=(
                    "Deterministically compute the Euclidean distance in meters between two numeric positions. "
                    "Use this instead of mental math whenever the task involves coordinates, lengths, or threshold comparisons."
                ),
            )
        )

    return support_tools
