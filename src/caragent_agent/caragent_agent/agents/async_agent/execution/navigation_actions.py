"""Structured navigation_action execution helpers."""

from __future__ import annotations

import re
from typing import Any, Optional, Sequence

from langchain_core.tools import BaseTool

from caragent_agent.agents.async_agent.execution.support import (
    find_tool_failure_message,
    navigation_waiting_summary,
    stringify_tool_content,
)
from caragent_agent.agents.async_agent.execution.tool_results import parse_json_like_payload
from caragent_agent.agents.async_agent.runtime.types import TaskItem


def find_tool_by_name(
    tools: Sequence[BaseTool],
    tool_name: str,
) -> Optional[BaseTool]:
    """Return the first tool whose registered name matches tool_name."""

    for tool in tools:
        if str(getattr(tool, "name", "") or "").strip() == tool_name:
            return tool
    return None


def tool_result_status_ok(raw_value: Any) -> bool:
    """Return True when a tool payload is absent or explicitly reports ok."""

    parsed = parse_json_like_payload(raw_value)
    if isinstance(parsed, dict):
        status = str(parsed.get("status") or "").strip().lower()
        if status and status not in {"ok", "success", "succeeded"}:
            return False
    return True


def _extract_keyframe_id_from_value(value: Any) -> Optional[int]:
    """Extract a keyframe id from common structured destination shapes."""

    if value is None:
        return None
    if isinstance(value, int):
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit():
            return int(stripped)
        parsed = parse_json_like_payload(stripped)
        if parsed is not None and parsed is not value:
            return _extract_keyframe_id_from_value(parsed)
        embedded_json = _extract_json_object_from_text(stripped)
        if embedded_json is not None:
            keyframe_id = _extract_keyframe_id_from_value(embedded_json)
            if keyframe_id is not None:
                return keyframe_id
        match = re.search(
            r"\b(?:keyframe|frame|kf)\s*(?:id\s*)?[^0-9]{0,16}(\d+)\b",
            stripped,
            flags=re.IGNORECASE,
        )
        if match:
            return int(match.group(1))
        return None
    if not isinstance(value, dict):
        return None

    for key in (
        "keyframe_id",
        "keyframe_node_id",
        "target_keyframe_id",
        "destination_keyframe_id",
        "recommended_keyframe_id",
        "kf_id",
    ):
        if value.get(key) is not None:
            try:
                return int(value[key])
            except Exception:
                continue

    for key in (
        "destination",
        "target",
        "data",
        "result",
        "selected_route",
        "record",
        "final_ai_content",
        "summary",
        "raw_output",
        "content",
    ):
        nested = value.get(key)
        if nested is value:
            continue
        keyframe_id = _extract_keyframe_id_from_value(nested)
        if keyframe_id is not None:
            return keyframe_id
    return None


def _extract_json_object_from_text(text: str) -> Optional[dict[str, Any]]:
    """Extract the first JSON object embedded in free-form text."""

    stripped = str(text or "").strip()
    if not stripped:
        return None
    start_index = stripped.find("{")
    while start_index >= 0:
        depth = 0
        in_string = False
        escape = False
        for index in range(start_index, len(stripped)):
            char = stripped[index]
            if escape:
                escape = False
                continue
            if char == "\\":
                escape = True
                continue
            if char == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    parsed = parse_json_like_payload(stripped[start_index : index + 1])
                    if isinstance(parsed, dict):
                        return parsed
                    break
        start_index = stripped.find("{", start_index + 1)
    return None


def _latest_task_result_payload(task: Optional[TaskItem]) -> Any:
    """Return the latest structured payload/result for one task."""

    if not task:
        return None
    results = list(task.get("result") or [])
    if not results:
        return None
    latest = results[-1]
    if not isinstance(latest, dict):
        return latest
    for key in ("destination", "result", "data"):
        if latest.get(key) is not None:
            return latest.get(key)
    raw_output = latest.get("raw_output")
    if raw_output:
        parsed = parse_json_like_payload(raw_output)
        if parsed is not None and parsed is not raw_output:
            return parsed
        embedded_json = _extract_json_object_from_text(str(raw_output))
        if embedded_json is not None:
            return embedded_json
    return latest


def resolve_navigation_action_keyframe_id(
    current_task: Optional[TaskItem],
    tasks: dict[int, TaskItem],
) -> tuple[Optional[int], Optional[str]]:
    """Resolve a navigation_action target to a concrete keyframe id."""

    if not current_task or current_task.get("task_type") != "navigation_action":
        return None, "Task is not a navigation_action."
    target = current_task.get("target")
    if not isinstance(target, dict):
        return None, "navigation_action is missing structured target."

    target_type = str(target.get("type") or "").strip()
    if target_type == "keyframe":
        keyframe_id = _extract_keyframe_id_from_value(target)
        if keyframe_id is None:
            return None, "navigation_action keyframe target is missing keyframe_id."
        return keyframe_id, None
    if target_type == "task_output":
        source_task_id = target.get("task_id")
        try:
            source_task_id = int(source_task_id)
        except Exception:
            return None, "navigation_action task_output target has invalid task_id."
        source_task = tasks.get(source_task_id)
        if source_task is None:
            return None, f"navigation_action target references missing task {source_task_id}."
        field = str(target.get("field") or "destination").strip()
        payload = _latest_task_result_payload(source_task)
        if isinstance(payload, dict) and field and payload.get(field) is not None:
            payload = payload.get(field)
        keyframe_id = _extract_keyframe_id_from_value(payload)
        if keyframe_id is None:
            return None, f"Task {source_task_id} output does not contain a keyframe destination."
        return keyframe_id, None
    if target_type == "position":
        return None, "position navigation targets are not executable until a go_to_position tool exists."
    return None, f"Unsupported navigation_action target type: {target_type or 'missing'}."


def try_dispatch_structured_navigation_action(
    current_task: Optional[TaskItem],
    *,
    tasks: dict[int, TaskItem],
    tools: Sequence[BaseTool],
) -> Optional[dict[str, Any]]:
    """Dispatch navigation_action tasks directly from their structured target."""

    if not current_task or current_task.get("task_type") != "navigation_action":
        return None
    keyframe_id, error = resolve_navigation_action_keyframe_id(current_task, tasks)
    if error:
        return {
            "summary": error,
            "tool_name": None,
            "tool_trace": {"tool_calls": [], "tool_results": [], "final_ai_content": error},
            "event_type": "task_failed",
        }
    navigation_tool = find_tool_by_name(tools, "go_to_keyframe")
    if navigation_tool is None:
        summary = "Navigation tool go_to_keyframe is unavailable."
        return {
            "summary": summary,
            "tool_name": None,
            "tool_trace": {"tool_calls": [], "tool_results": [], "final_ai_content": summary},
            "event_type": "task_failed",
        }

    tool_args = {"keyframe_node_id": keyframe_id}
    try:
        raw_navigation = navigation_tool.invoke(tool_args)
    except Exception as exc:
        raw_navigation = {
            "status": "error",
            "summary": "Navigation dispatch raised an exception.",
            "error": {"message": str(exc)},
        }

    tool_trace = {
        "tool_calls": [{"name": "go_to_keyframe", "args": tool_args}],
        "tool_results": [
            {
                "name": "go_to_keyframe",
                "content": stringify_tool_content(raw_navigation),
                "tool_call_id": None,
            }
        ],
        "final_ai_content": f"Dispatched structured navigation_action to keyframe {keyframe_id}.",
    }
    if tool_result_status_ok(raw_navigation):
        return {
            "summary": navigation_waiting_summary(current_task),
            "tool_name": "go_to_keyframe",
            "tool_trace": tool_trace,
            "event_type": "task_waiting",
        }
    failure = find_tool_failure_message(tool_trace) or "Navigation dispatch failed."
    return {
        "summary": failure,
        "tool_name": "go_to_keyframe",
        "tool_trace": tool_trace,
        "event_type": "task_failed",
    }


__all__ = [
    "find_tool_by_name",
    "resolve_navigation_action_keyframe_id",
    "tool_result_status_ok",
    "try_dispatch_structured_navigation_action",
]
