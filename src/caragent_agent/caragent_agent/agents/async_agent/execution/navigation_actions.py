"""Structured navigation_action execution helpers."""

from __future__ import annotations

import json
import re
import time
from typing import Any, Optional, Sequence

from langchain_core.tools import BaseTool

from caragent_agent.agents.async_agent.execution.runtime_tool_context import (
    get_runtime_tool_context,
)
from caragent_agent.agents.async_agent.execution.support import (
    find_tool_failure_message,
    navigation_waiting_summary,
    submit_task_result,
    stringify_tool_content,
)
from caragent_agent.agents.async_agent.execution.tool_results import parse_json_like_payload
from caragent_agent.agents.async_agent.runtime.types import TaskItem
from caragent_agent.agents.async_agent.target_resolution import TargetResolver
from caragent_agent.config.config import config


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


def _parse_tool_payload(raw_value: Any) -> Any:
    parsed = parse_json_like_payload(raw_value)
    return parsed if parsed is not None else raw_value


def _invoke_tool(tool: BaseTool, args: dict[str, Any]) -> Any:
    invoke = getattr(tool, "invoke", None)
    if not callable(invoke):
        execute = getattr(tool, "execute", None)
        if callable(execute):
            return execute(**args)
        raise TypeError(f"Tool {getattr(tool, 'name', tool)} is not invokable.")
    try:
        return invoke(args)
    except TypeError:
        return invoke(**args)


def _runtime_logger() -> Any:
    logger = get_runtime_tool_context().get("logger")
    if callable(logger):
        return logger
    log_foreground = getattr(logger, "log_foreground", None)
    return log_foreground if callable(log_foreground) else None


def _log_structured_navigation(message: str) -> None:
    logger = _runtime_logger()
    if logger:
        logger(message)


def _log_target_resolution(
    result: dict[str, Any],
    *,
    current_task: Optional[TaskItem] = None,
) -> None:
    payload = {
        "status": result.get("status"),
        "draft": result.get("draft"),
        "target_ref": result.get("target_ref"),
        "anchor": result.get("anchor"),
        "required_next_step": result.get("required_next_step"),
        "failure_reason": result.get("failure_reason"),
        "diagnostics": result.get("diagnostics"),
        "evidence": [
            {
                "tool_name": item.get("tool_name"),
                "status": item.get("status"),
                "summary": item.get("summary"),
                "data": item.get("data"),
            }
            for item in list(result.get("evidence") or [])[:4]
            if isinstance(item, dict)
        ],
    }
    _log_structured_navigation(
        "target_resolution: " + json.dumps(payload, ensure_ascii=False, default=str)
    )
    _log_target_resolution_summary(result, current_task=current_task)


def _log_target_resolution_summary(
    result: dict[str, Any],
    *,
    current_task: Optional[TaskItem] = None,
) -> None:
    draft = result.get("draft") if isinstance(result.get("draft"), dict) else {}
    target_ref = result.get("target_ref") if isinstance(result.get("target_ref"), dict) else {}
    anchor = result.get("anchor") if isinstance(result.get("anchor"), dict) else {}
    diagnostics = result.get("diagnostics") if isinstance(result.get("diagnostics"), dict) else {}
    required_next_step = (
        result.get("required_next_step")
        if isinstance(result.get("required_next_step"), dict)
        else {}
    )
    summary = {
        "task_id": (current_task or {}).get("task_id") if isinstance(current_task, dict) else None,
        "target_type": draft.get("target_type"),
        "source": target_ref.get("source"),
        "kind": target_ref.get("kind"),
        "status": result.get("status"),
        "stage": diagnostics.get("stage"),
        "anchor_type": anchor.get("anchor_type"),
        "keyframe_id": anchor.get("keyframe_id"),
        "position": anchor.get("position"),
        "image_focus": target_ref.get("image_focus"),
        "required_next_step": required_next_step.get("step_type"),
        "tool_names": diagnostics.get("tool_names") or [],
        "elapsed_sec": diagnostics.get("resolver_total_sec"),
        "tool_elapsed_sec": diagnostics.get("tool_elapsed_sec"),
        "decision": diagnostics.get("decision"),
    }
    compact = {
        key: value
        for key, value in summary.items()
        if value not in (None, "", [], {})
    }
    _log_structured_navigation(
        "target_resolution_summary: "
        + json.dumps(compact, ensure_ascii=False, default=str)
    )


def _elapsed_since(start_time: float) -> float:
    return round(max(0.0, time.perf_counter() - start_time), 3)


def _semantic_keyframe_query(current_task: TaskItem, target: dict[str, Any]) -> str:
    query = str(target.get("query") or "").strip()
    if query:
        return query
    return str(current_task.get("description") or "").strip()


def _target_source(target: dict[str, Any], *, default: str) -> str:
    source = str(target.get("target_source") or "").strip().lower()
    return source or default


def _target_image_focus(target: dict[str, Any]) -> str:
    focus = str(target.get("image_focus") or "").strip().lower()
    return focus if focus in {"scene", "object"} else "scene"


def _contains_unresolved_template(value: Any) -> bool:
    if isinstance(value, str):
        return "{{" in value or "}}" in value
    if isinstance(value, dict):
        return any(_contains_unresolved_template(item) for item in value.values())
    if isinstance(value, (list, tuple, set)):
        return any(_contains_unresolved_template(item) for item in value)
    return False


def _task_image_ref(current_task: TaskItem, target: dict[str, Any]) -> Optional[str]:
    runtime_context = get_runtime_tool_context()
    selected_packet = runtime_context.get("selected_execution_context_packet")
    current_user_input = (
        selected_packet.get("current_user_input")
        if isinstance(selected_packet, dict)
        else None
    )
    attached_images = (
        list(current_user_input.get("attached_images") or [])
        if isinstance(current_user_input, dict)
        else []
    )
    by_ref_id = {
        str(item.get("image_ref_id") or "").strip(): item
        for item in attached_images
        if isinstance(item, dict) and str(item.get("image_ref_id") or "").strip()
    }

    for value in list(target.get("image_refs") or []) + list(current_task.get("image_refs") or []):
        text = str(value or "").strip()
        if text.lower() == "latest" and attached_images:
            return json.dumps(attached_images[0], ensure_ascii=False)
        if text in by_ref_id:
            return json.dumps(by_ref_id[text], ensure_ascii=False)
        if text:
            return text
    return None


def _candidate_keyframe_ids(data: dict[str, Any]) -> list[int]:
    ids = data.get("matched_keyframe_ids") or data.get("candidate_keyframe_ids") or []
    normalized: list[int] = []
    for value in list(ids):
        try:
            item = int(value)
        except Exception:
            continue
        if item not in normalized:
            normalized.append(item)
    return normalized


def _recommended_keyframe_id_from_search(raw_search: Any) -> Optional[int]:
    parsed = _parse_tool_payload(raw_search)
    if not isinstance(parsed, dict):
        return None
    data = parsed.get("data") if isinstance(parsed.get("data"), dict) else parsed
    resolution_status = str(data.get("resolution_status") or "").strip().lower()
    if resolution_status and resolution_status != "resolved":
        return None
    for key in ("recommended_keyframe_id", "keyframe_id", "target_keyframe_id"):
        if data.get(key) is not None:
            try:
                return int(data[key])
            except Exception:
                pass
    destination = data.get("recommended_destination") or data.get("destination")
    keyframe_id = _extract_keyframe_id_from_value(destination)
    if keyframe_id is not None:
        return keyframe_id
    ids = _candidate_keyframe_ids(data)
    return ids[0] if ids else None


def _background_result_for_current_task(current_task: TaskItem) -> Optional[dict[str, Any]]:
    """Return completed background result for this task, if available."""

    try:
        task_id = int(current_task.get("task_id"))
    except Exception:
        return None
    runtime_context = get_runtime_tool_context()
    direct = runtime_context.get("background_result")
    if isinstance(direct, dict):
        return direct
    shared = runtime_context.get("shared_background_results")
    if isinstance(shared, dict):
        candidate = shared.get(task_id)
        if isinstance(candidate, dict):
            return candidate
    return None


def _recommended_keyframe_id_from_background(
    background_result: Optional[dict[str, Any]],
) -> Optional[int]:
    """Return a background-recommended keyframe id when it is ready to use."""

    if not isinstance(background_result, dict):
        return None
    if str(background_result.get("status") or "").strip().lower() != "completed":
        return None
    destination = background_result.get("recommended_destination")
    keyframe_id = _extract_keyframe_id_from_value(destination)
    if keyframe_id is not None:
        return keyframe_id
    raw_keyframe_id = background_result.get("recommended_keyframe_id")
    try:
        return int(raw_keyframe_id)
    except Exception:
        return None


def _semantic_keyframe_search_payload_from_background(
    *,
    query: str,
    background_result: dict[str, Any],
    recommended_keyframe_id: int,
) -> dict[str, Any]:
    """Shape background keyframe evidence like a keyframe search tool result."""

    candidate_ids = list(background_result.get("candidate_keyframe_ids") or [])
    if recommended_keyframe_id not in candidate_ids:
        candidate_ids.insert(0, recommended_keyframe_id)
    return {
        "status": "ok",
        "summary": (
            "Reused completed background semantic keyframe preanalysis for "
            f"keyframe {recommended_keyframe_id}."
        ),
        "data": {
            "requirement": query,
            "retrieval_mode": "background_preanalysis",
            "resolution_status": "resolved",
            "recommended_keyframe_id": recommended_keyframe_id,
            "recommended_destination": {
                "type": "keyframe",
                "keyframe_id": recommended_keyframe_id,
            },
            "recommendation_reason": background_result.get("recommendation_reason")
            or background_result.get("summary"),
            "candidate_keyframe_ids": candidate_ids[:8],
            "candidate_keyframes": list(background_result.get("candidate_keyframes") or [])[:5],
        },
    }


def _semantic_keyframe_context(
    *,
    query: str,
    target: dict[str, Any],
    raw_search: Any,
    recommended_keyframe_id: int,
) -> dict[str, Any]:
    parsed = _parse_tool_payload(raw_search)
    data = parsed.get("data") if isinstance(parsed, dict) and isinstance(parsed.get("data"), dict) else {}
    candidates = [
        candidate
        for candidate in list(data.get("candidate_keyframes") or [])[:5]
        if isinstance(candidate, dict)
    ]
    return {
        "target_type": "semantic_keyframe",
        "query": query,
        "selection_policy": target.get("selection_policy"),
        "resolution_status": data.get("resolution_status"),
        "recommended_keyframe_id": recommended_keyframe_id,
        "recommendation_reason": data.get("recommendation_reason"),
        "candidate_keyframe_ids": _candidate_keyframe_ids(data)[:8],
        "candidate_keyframes": candidates,
        "retrieval_mode": data.get("retrieval_mode"),
    }


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
        "tool_evidence",
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


def _extract_position_destination_from_value(value: Any) -> Optional[dict[str, Any]]:
    """Extract a map-frame position destination from structured task output."""

    if value is None:
        return None
    if isinstance(value, str):
        parsed = parse_json_like_payload(value.strip())
        if parsed is not None and parsed is not value:
            return _extract_position_destination_from_value(parsed)
        embedded_json = _extract_json_object_from_text(value)
        if embedded_json is not None:
            return _extract_position_destination_from_value(embedded_json)
        return None
    if isinstance(value, (list, tuple)):
        for item in reversed(list(value)):
            nested = _extract_position_destination_from_value(item)
            if nested is not None:
                return nested
        return None
    if not isinstance(value, dict):
        return None

    structured = value.get("content")
    if isinstance(structured, str):
        parsed_content = parse_json_like_payload(structured)
        if isinstance(parsed_content, dict):
            tool_data = parsed_content.get("data")
            if isinstance(tool_data, dict):
                destination = _extract_position_destination_from_value(tool_data.get("destination"))
                if destination is not None:
                    return destination
            destination = _extract_position_destination_from_value(parsed_content)
            if destination is not None:
                return destination

    tool_results = value.get("tool_results")
    if isinstance(tool_results, list):
        for tool_result in reversed(tool_results):
            destination = _extract_position_destination_from_value(tool_result)
            if destination is not None:
                return destination

    if value.get("type") == "position" and value.get("position") is not None:
        position_payload = value.get("position")
        yaw = value.get("yaw_deg", value.get("yaw", 0.0))
        if isinstance(position_payload, (list, tuple)) and len(position_payload) >= 2:
            try:
                return {
                    "position": [
                        float(position_payload[0]),
                        float(position_payload[1]),
                        float(position_payload[2]) if len(position_payload) >= 3 else 0.0,
                    ],
                    "yaw_deg": float(yaw or 0.0),
                }
            except Exception:
                return None

    for key in (
        "destination",
        "target",
        "data",
        "result",
        "tool_evidence",
        "record",
        "final_ai_content",
        "summary",
        "raw_output",
        "content",
    ):
        nested = value.get(key)
        if nested is value:
            continue
        destination = _extract_position_destination_from_value(nested)
        if destination is not None:
            return destination
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


def _target_resolution_config() -> tuple[bool, bool]:
    agent_cfg = config.get("agent") if isinstance(config.get("agent"), dict) else {}
    return (
        bool(agent_cfg.get("target_resolution_enabled", True)),
        bool(agent_cfg.get("target_resolution_dry_run", False)),
    )


def _log_legacy_semantic_dispatch(
    *,
    target_type: str,
    reason: str,
    current_task: Optional[TaskItem],
) -> None:
    _log_structured_navigation(
        "legacy_semantic_dispatch_used: "
        + json.dumps(
            {
                "target_type": target_type,
                "reason": reason,
                "task_id": (current_task or {}).get("task_id")
                if isinstance(current_task, dict)
                else None,
            },
            ensure_ascii=False,
            default=str,
        )
    )


def _submit_resolution_failure(
    result: dict[str, Any],
    *,
    target_type: str,
) -> dict[str, Any]:
    summary = (
        str(result.get("failure_reason") or "").strip()
        or str((result.get("required_next_step") or {}).get("reason") or "").strip()
        or "Target resolver did not produce a navigable anchor."
    )
    failure = submit_task_result(
        summary=summary,
        current_place_context={
            "target_type": target_type,
            "target_resolution": {
                "status": result.get("status"),
                "target_ref": result.get("target_ref"),
                "required_next_step": result.get("required_next_step"),
                "evidence": result.get("evidence"),
            },
        },
        failure_reason=summary,
    )
    return {
        "summary": summary,
        "tool_name": "target_resolution",
        "tool_trace": {
            "tool_calls": [{"name": "target_resolution", "args": {"target_type": target_type}}],
            "tool_results": [
                {
                    "name": "target_resolution",
                    "content": stringify_tool_content(result),
                    "tool_call_id": None,
                },
                {
                    "name": "submit_task_result",
                    "content": stringify_tool_content(failure),
                    "tool_call_id": None,
                },
            ],
            "final_ai_content": summary,
        },
        "event_type": "task_failed",
    }


def _submit_resolution_needs_observation(
    result: dict[str, Any],
    *,
    target_type: str,
) -> dict[str, Any]:
    required_next_step = (
        result.get("required_next_step")
        if isinstance(result.get("required_next_step"), dict)
        else {}
    )
    summary = (
        str(required_next_step.get("reason") or "").strip()
        or "Target resolver needs more observation before navigation."
    )
    submitted = submit_task_result(
        summary=summary,
        current_place_context={
            "target_type": target_type,
            "target_resolution": {
                "status": result.get("status"),
                "target_ref": result.get("target_ref"),
                "required_next_step": required_next_step,
                "evidence": result.get("evidence"),
            },
            "navigation_deferred": True,
            "required_next_step": required_next_step,
        },
    )
    return {
        "summary": summary,
        "tool_name": "target_resolution",
        "tool_trace": {
            "tool_calls": [{"name": "target_resolution", "args": {"target_type": target_type}}],
            "tool_results": [
                {
                    "name": "target_resolution",
                    "content": stringify_tool_content(result),
                    "tool_call_id": None,
                },
                {
                    "name": "submit_task_result",
                    "content": stringify_tool_content(submitted),
                    "tool_call_id": None,
                },
            ],
            "final_ai_content": summary,
        },
        "event_type": "task_completed",
    }


def _submit_resolution_anchor(result: dict[str, Any]) -> dict[str, Any]:
    anchor = result.get("anchor") if isinstance(result.get("anchor"), dict) else {}
    target_ref = result.get("target_ref") if isinstance(result.get("target_ref"), dict) else {}
    if anchor.get("anchor_type") == "keyframe":
        destination = {"type": "keyframe", "keyframe_id": int(anchor["keyframe_id"])}
        evidence = result.get("evidence")
        first_evidence = {}
        if isinstance(evidence, list):
            for item in evidence:
                if isinstance(item, dict):
                    first_evidence = item
                    break
        evidence_data = (
            first_evidence.get("data")
            if isinstance(first_evidence.get("data"), dict)
            else {}
        )
        target_kind = str(target_ref.get("kind") or "keyframe").strip()
        target_type = "semantic_object" if target_kind == "object" else "semantic_keyframe"
        current_place_context = {
            "target_type": target_type,
            "query": target_ref.get("query") or target_ref.get("description"),
            "target_source": target_ref.get("source"),
            "target_resolution": {
                "status": result.get("status"),
                "target_ref": target_ref,
                "evidence": result.get("evidence"),
                "required_next_step": result.get("required_next_step"),
            },
        }
        if target_type == "semantic_object":
            current_place_context["object_description"] = target_ref.get("description")
            current_place_context["staging_keyframe_id"] = destination["keyframe_id"]
            current_place_context["staging_reason"] = (
                (result.get("required_next_step") or {}).get("reason")
                or "Object target staged through a keyframe before live localization."
            )
        if target_ref.get("image_focus"):
            current_place_context["image_focus"] = target_ref.get("image_focus")
        if target_ref.get("image_refs"):
            current_place_context["image_refs"] = target_ref.get("image_refs")
        for key in (
            "resolution_status",
            "recommended_keyframe_id",
            "candidate_keyframe_ids",
            "candidate_keyframes",
            "recommended_destination",
            "recommendation_reason",
            "retrieval_mode",
            "image_ref",
            "image_focus",
            "elapsed_sec",
        ):
            if evidence_data.get(key) not in (None, "", [], {}):
                current_place_context[key] = evidence_data.get(key)
        return submit_task_result(
            destination=destination,
            current_place_context=current_place_context,
            summary=f"Resolved semantic navigation target to keyframe {destination['keyframe_id']}.",
        )
    if anchor.get("anchor_type") == "position":
        destination = {
            "type": "position",
            "position": list(anchor.get("position") or []),
            "yaw_deg": float(anchor.get("yaw_deg") or 0.0),
        }
        selected_object = {
            "description": target_ref.get("description"),
            "target_type": "semantic_object",
            "target_source": target_ref.get("source"),
            "source": anchor.get("source"),
            "target_resolution": {
                "status": result.get("status"),
                "evidence": result.get("evidence"),
            },
        }
        return submit_task_result(
            destination=destination,
            selected_object=selected_object,
            summary=(
                "Resolved semantic object target "
                f"'{target_ref.get('description')}' to a position destination."
            ),
        )
    return submit_task_result(
        summary="Target resolver anchor type is unsupported.",
        failure_reason="Target resolver anchor type is unsupported.",
    )


def _semantic_object_target_description(task: Optional[TaskItem]) -> str:
    target = task.get("target") if isinstance(task, dict) else None
    if not isinstance(target, dict):
        return ""
    return str(target.get("object_description") or "").strip()


def _target_depends_on_task(target: dict[str, Any], task_id: int) -> bool:
    for item in list(target.get("inputs_from") or []):
        if not isinstance(item, dict):
            continue
        try:
            if int(item.get("task_id")) == task_id:
                return True
        except Exception:
            continue
    return False


def _downstream_object_task_for_keyframe(
    current_task: TaskItem,
    tasks: dict[int, TaskItem],
) -> Optional[TaskItem]:
    try:
        current_task_id = int(current_task.get("task_id"))
    except Exception:
        return None

    candidate_ids: list[int] = []
    raw_next_id = current_task.get("next_task_id")
    if raw_next_id is not None:
        try:
            candidate_ids.append(int(raw_next_id))
        except Exception:
            pass
    for task_id, task in tasks.items():
        if task_id in candidate_ids:
            continue
        dependency_ids = []
        for raw_dependency in list(task.get("depends_on") or []):
            try:
                dependency_ids.append(int(raw_dependency))
            except Exception:
                continue
        target = task.get("target") if isinstance(task, dict) else None
        target_depends = (
            isinstance(target, dict)
            and _target_depends_on_task(target, current_task_id)
        )
        if current_task_id in dependency_ids or target_depends:
            candidate_ids.append(int(task_id))

    for task_id in candidate_ids:
        task = tasks.get(task_id)
        if not isinstance(task, dict):
            continue
        if str(task.get("task_type") or "").strip() != "navigation_action":
            continue
        target = task.get("target")
        if not isinstance(target, dict):
            continue
        if str(target.get("type") or "").strip() != "semantic_object":
            continue
        source = str(target.get("target_source") or "").strip()
        if source not in {"arrived_scene", "upstream_result"}:
            continue
        if _semantic_object_target_description(task):
            return task
    return None


def _refine_attached_image_keyframe_target(
    current_task: TaskItem,
    tasks: dict[int, TaskItem],
    target: dict[str, Any],
) -> dict[str, Any]:
    if str(target.get("type") or "").strip() != "semantic_keyframe":
        return target
    if str(target.get("target_source") or "").strip() != "attached_image":
        return target

    downstream_object_task = _downstream_object_task_for_keyframe(current_task, tasks)
    if downstream_object_task is None:
        return target

    object_description = _semantic_object_target_description(downstream_object_task)
    if not object_description:
        return target

    refined = dict(target)
    changed = False
    if refined.get("image_focus") != "object":
        refined["image_focus"] = "object"
        changed = True
    if refined.get("target_kind") != "object":
        refined["target_kind"] = "object"
        changed = True
    if str(refined.get("query") or "").strip() != object_description:
        refined["query"] = object_description
        changed = True
    if changed:
        _log_structured_navigation(
            "target_resolution_policy_rewrite: "
            + json.dumps(
                {
                    "rule": "attached_image_downstream_object_focus",
                    "reason": "downstream semantic_object depends on this attached-image keyframe target",
                    "current_task_id": current_task.get("task_id"),
                    "downstream_task_id": downstream_object_task.get("task_id"),
                    "object_description": object_description,
                    "image_focus": refined.get("image_focus"),
                    "target_kind": refined.get("target_kind"),
                    "query": refined.get("query"),
                },
                ensure_ascii=False,
                default=str,
            )
        )
    return refined


def _dispatch_keyframe_anchor_navigation(
    current_task: Optional[TaskItem],
    tools: Sequence[BaseTool],
    result: dict[str, Any],
    *,
    target_type: str = "semantic_keyframe",
) -> dict[str, Any]:
    anchor = result.get("anchor") if isinstance(result.get("anchor"), dict) else {}
    try:
        keyframe_id = int(anchor.get("keyframe_id"))
    except Exception:
        return _submit_resolution_failure(result, target_type="semantic_keyframe")
    navigation_tool = find_tool_by_name(tools, "go_to_keyframe")
    if navigation_tool is None:
        summary = "Navigation tool go_to_keyframe is unavailable."
        return {
            "summary": summary,
            "tool_name": None,
            "tool_trace": {"tool_calls": [], "tool_results": [], "final_ai_content": summary},
            "event_type": "task_failed",
        }
    submitted = _submit_resolution_anchor(result)
    tool_args = {"keyframe_node_id": keyframe_id}
    try:
        raw_navigation = _invoke_tool(navigation_tool, tool_args)
    except Exception as exc:
        raw_navigation = {
            "status": "error",
            "summary": "Navigation dispatch raised an exception.",
            "error": {"message": str(exc)},
        }
    tool_trace = {
        "tool_calls": [
            {"name": "target_resolution", "args": {"target_type": target_type}},
            {"name": "submit_task_result", "args": {"destination": {"type": "keyframe", "keyframe_id": keyframe_id}}},
            {"name": "go_to_keyframe", "args": tool_args},
        ],
        "tool_results": [
            {
                "name": "target_resolution",
                "content": stringify_tool_content(result),
                "tool_call_id": None,
            },
            {
                "name": "submit_task_result",
                "content": stringify_tool_content(submitted),
                "tool_call_id": None,
            },
            {
                "name": "go_to_keyframe",
                "content": stringify_tool_content(raw_navigation),
                "tool_call_id": None,
            },
        ],
        "final_ai_content": f"Resolved semantic target to keyframe {keyframe_id} and dispatched navigation.",
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


def _dispatch_position_anchor_navigation(
    current_task: Optional[TaskItem],
    tools: Sequence[BaseTool],
    result: dict[str, Any],
) -> dict[str, Any]:
    anchor = result.get("anchor") if isinstance(result.get("anchor"), dict) else {}
    position = anchor.get("position")
    if not isinstance(position, (list, tuple)) or len(position) < 2:
        return _submit_resolution_failure(result, target_type="semantic_object")
    submitted = _submit_resolution_anchor(result)
    destination = {
        "position": [
            float(position[0]),
            float(position[1]),
            float(position[2]) if len(position) >= 3 else 0.0,
        ],
        "yaw_deg": float(anchor.get("yaw_deg") or 0.0),
    }
    dispatched = _dispatch_position_navigation(current_task, tools, destination, None)
    tool_trace = dict(dispatched.get("tool_trace") or {})
    tool_trace["tool_calls"] = [
        {"name": "target_resolution", "args": {"target_type": "semantic_object"}},
        {"name": "submit_task_result", "args": {"destination": {"type": "position", **destination}}},
    ] + list(tool_trace.get("tool_calls") or [])
    tool_trace["tool_results"] = [
        {
            "name": "target_resolution",
            "content": stringify_tool_content(result),
            "tool_call_id": None,
        },
        {
            "name": "submit_task_result",
            "content": stringify_tool_content(submitted),
            "tool_call_id": None,
        },
    ] + list(tool_trace.get("tool_results") or [])
    dispatched["tool_trace"] = tool_trace
    return dispatched


def _extract_named_field_from_task_payload(value: Any, field: str) -> Any:
    """Extract an explicit named task-output field without scanning tool candidates."""

    if not field:
        return value
    if value is None:
        return None
    if isinstance(value, str):
        parsed = parse_json_like_payload(value.strip())
        if parsed is not None and parsed is not value:
            return _extract_named_field_from_task_payload(parsed, field)
        embedded_json = _extract_json_object_from_text(value)
        if embedded_json is not None:
            return _extract_named_field_from_task_payload(embedded_json, field)
        return None
    if isinstance(value, (list, tuple)):
        for item in reversed(list(value)):
            extracted = _extract_named_field_from_task_payload(item, field)
            if extracted is not None:
                return extracted
        return None
    if not isinstance(value, dict):
        return None
    if value.get(field) is not None:
        return value.get(field)
    for key in (
        "data",
        "result",
        "record",
        "target",
        "tool_evidence",
        "tool_results",
        "final_ai_content",
        "summary",
        "raw_output",
        "content",
    ):
        nested = value.get(key)
        if nested is value:
            continue
        extracted = _extract_named_field_from_task_payload(nested, field)
        if extracted is not None:
            return extracted
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
        source_payload = _latest_task_result_payload(source_task)
        payload = source_payload
        if field:
            payload = _extract_named_field_from_task_payload(source_payload, field)
            if payload is None:
                if field == "destination":
                    payload = source_payload
                else:
                    return None, f"Task {source_task_id} output does not contain an explicit {field}."
        keyframe_id = _extract_keyframe_id_from_value(payload)
        if keyframe_id is None:
            return None, f"Task {source_task_id} output does not contain a keyframe destination."
        return keyframe_id, None
    if target_type == "position":
        return None, "position navigation targets are not executable until a go_to_position tool exists."
    return None, f"Unsupported navigation_action target type: {target_type or 'missing'}."


def resolve_navigation_action_position(
    current_task: Optional[TaskItem],
    tasks: dict[int, TaskItem],
) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    """Resolve a navigation_action target to a concrete map-frame position."""

    if not current_task or current_task.get("task_type") != "navigation_action":
        return None, "Task is not a navigation_action."
    target = current_task.get("target")
    if not isinstance(target, dict):
        return None, "navigation_action is missing structured target."

    target_type = str(target.get("type") or "").strip()
    if target_type == "position":
        destination = _extract_position_destination_from_value(target)
        if destination is None:
            return None, "navigation_action position target is missing position."
        return destination, None
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
        source_payload = _latest_task_result_payload(source_task)
        payload = source_payload
        if field:
            payload = _extract_named_field_from_task_payload(source_payload, field)
            if payload is None:
                if field == "destination":
                    payload = source_payload
                else:
                    return None, f"Task {source_task_id} output does not contain an explicit {field}."
        destination = _extract_position_destination_from_value(payload)
        if destination is None:
            return None, f"Task {source_task_id} output does not contain a position destination."
        return destination, None
    return None, f"Unsupported position target type: {target_type or 'missing'}."


def try_dispatch_structured_navigation_action(
    current_task: Optional[TaskItem],
    *,
    tasks: dict[int, TaskItem],
    tools: Sequence[BaseTool],
) -> Optional[dict[str, Any]]:
    """Dispatch navigation_action tasks directly from their structured target."""

    if not current_task or current_task.get("task_type") != "navigation_action":
        return None
    target = current_task.get("target") if isinstance(current_task, dict) else None
    if isinstance(target, dict):
        target = _refine_attached_image_keyframe_target(current_task, tasks, target)
    target_type = str((target or {}).get("type") or "").strip() if isinstance(target, dict) else ""
    if target_type == "position":
        position_destination, error = resolve_navigation_action_position(current_task, tasks)
        return _dispatch_position_navigation(current_task, tools, position_destination, error)
    if target_type == "semantic_keyframe":
        enabled, dry_run = _target_resolution_config()
        if enabled:
            resolver = TargetResolver(
                tools,
                background_result=_background_result_for_current_task(current_task),
            )
            result = (
                resolver.dry_run(current_task, target or {})
                if dry_run
                else resolver.resolve(current_task, target or {})
            )
            _log_target_resolution(result, current_task=current_task)
            if dry_run:
                _log_legacy_semantic_dispatch(
                    target_type="semantic_keyframe",
                    reason="target_resolution_dry_run",
                    current_task=current_task,
                )
                return _dispatch_semantic_keyframe_navigation(current_task, tools, target or {})
            if result.get("status") == "resolved" and isinstance(result.get("anchor"), dict):
                anchor_type = str(result["anchor"].get("anchor_type") or "")
                if anchor_type == "keyframe":
                    return _dispatch_keyframe_anchor_navigation(current_task, tools, result)
            if (
                result.get("status") == "needs_observation"
                and str((result.get("required_next_step") or {}).get("step_type") or "")
                == "needs_budgeted_live_localization"
            ):
                return _submit_resolution_needs_observation(
                    result,
                    target_type="semantic_keyframe",
                )
            return _submit_resolution_failure(result, target_type="semantic_keyframe")
        _log_legacy_semantic_dispatch(
            target_type="semantic_keyframe",
            reason="target_resolution_disabled",
            current_task=current_task,
        )
        return _dispatch_semantic_keyframe_navigation(current_task, tools, target or {})
    if target_type == "semantic_object":
        enabled, dry_run = _target_resolution_config()
        if enabled:
            resolver = TargetResolver(
                tools,
                background_result=_background_result_for_current_task(current_task),
            )
            result = (
                resolver.dry_run(current_task, target or {})
                if dry_run
                else resolver.resolve(current_task, target or {})
            )
            _log_target_resolution(result, current_task=current_task)
            if dry_run:
                _log_legacy_semantic_dispatch(
                    target_type="semantic_object",
                    reason="target_resolution_dry_run",
                    current_task=current_task,
                )
                return _dispatch_semantic_object_navigation(current_task, tools, target or {})
            if result.get("status") == "resolved" and isinstance(result.get("anchor"), dict):
                anchor_type = str(result["anchor"].get("anchor_type") or "")
                if anchor_type == "position":
                    return _dispatch_position_anchor_navigation(current_task, tools, result)
                if (
                    anchor_type == "keyframe"
                    and str((result.get("target_ref") or {}).get("source") or "") == "attached_image"
                    and str(result["anchor"].get("source") or "") == "attached_image_object_staging"
                ):
                    return _dispatch_keyframe_anchor_navigation(
                        current_task,
                        tools,
                        result,
                        target_type="semantic_object",
                    )
            if (
                result.get("status") == "needs_observation"
                and str((result.get("required_next_step") or {}).get("step_type") or "")
                == "needs_budgeted_live_localization"
            ):
                return _submit_resolution_needs_observation(
                    result,
                    target_type="semantic_object",
                )
            return _submit_resolution_failure(result, target_type="semantic_object")
        _log_legacy_semantic_dispatch(
            target_type="semantic_object",
            reason="target_resolution_disabled",
            current_task=current_task,
        )
        return _dispatch_semantic_object_navigation(current_task, tools, target or {})
    if target_type == "task_output":
        position_destination, position_error = resolve_navigation_action_position(current_task, tasks)
        if position_destination is not None:
            return _dispatch_position_navigation(current_task, tools, position_destination, None)

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
        raw_navigation = _invoke_tool(navigation_tool, tool_args)
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


def _semantic_object_description(current_task: TaskItem, target: dict[str, Any]) -> str:
    description = str(target.get("object_description") or "").strip()
    if description:
        return description
    return str(current_task.get("description") or "").strip()


def _dispatch_semantic_object_navigation(
    current_task: TaskItem,
    tools: Sequence[BaseTool],
    target: dict[str, Any],
) -> dict[str, Any]:
    target_source = _target_source(target, default="current_view")
    if target_source not in {"current_view", "arrived_scene", "upstream_result"}:
        summary = (
            "semantic_object target_source must be current_view, arrived_scene, "
            f"or upstream_result; got {target_source or 'missing'}."
        )
        failure = submit_task_result(summary=summary, failure_reason=summary)
        return {
            "summary": summary,
            "tool_name": "submit_task_result",
            "tool_trace": {
                "tool_calls": [{"name": "submit_task_result", "args": {"failure_reason": summary}}],
                "tool_results": [
                    {
                        "name": "submit_task_result",
                        "content": stringify_tool_content(failure),
                        "tool_call_id": None,
                    }
                ],
                "final_ai_content": summary,
            },
            "event_type": "task_failed",
        }
    object_inputs_from = target.get("inputs_from") or current_task.get("inputs_from")
    if target_source in {"arrived_scene", "upstream_result"} and not object_inputs_from:
        summary = (
            f"semantic_object target_source={target_source} requires inputs_from "
            "so the target scene/object is bound to upstream evidence."
        )
        failure = submit_task_result(summary=summary, failure_reason=summary)
        return {
            "summary": summary,
            "tool_name": "submit_task_result",
            "tool_trace": {
                "tool_calls": [{"name": "submit_task_result", "args": {"failure_reason": summary}}],
                "tool_results": [
                    {
                        "name": "submit_task_result",
                        "content": stringify_tool_content(failure),
                        "tool_call_id": None,
                    }
                ],
                "final_ai_content": summary,
            },
            "event_type": "task_failed",
        }
    if _contains_unresolved_template(target):
        summary = "semantic_object navigation target contains unresolved upstream template text."
        failure = submit_task_result(summary=summary, failure_reason=summary)
        return {
            "summary": summary,
            "tool_name": "submit_task_result",
            "tool_trace": {
                "tool_calls": [{"name": "submit_task_result", "args": {"failure_reason": summary}}],
                "tool_results": [
                    {
                        "name": "submit_task_result",
                        "content": stringify_tool_content(failure),
                        "tool_call_id": None,
                    }
                ],
                "final_ai_content": summary,
            },
            "event_type": "task_failed",
        }

    approach_tool = find_tool_by_name(tools, "approach_object_in_current_view")
    navigation_tool = find_tool_by_name(tools, "go_to_position")
    if approach_tool is None:
        summary = "Object localization tool approach_object_in_current_view is unavailable."
        return {
            "summary": summary,
            "tool_name": None,
            "tool_trace": {"tool_calls": [], "tool_results": [], "final_ai_content": summary},
            "event_type": "task_failed",
        }
    if navigation_tool is None:
        summary = "Navigation tool go_to_position is unavailable."
        return {
            "summary": summary,
            "tool_name": None,
            "tool_trace": {"tool_calls": [], "tool_results": [], "final_ai_content": summary},
            "event_type": "task_failed",
        }

    object_description = _semantic_object_description(current_task, target)
    if not object_description:
        summary = "semantic_object navigation target is missing object_description."
        failure = submit_task_result(summary=summary, failure_reason=summary)
        return {
            "summary": summary,
            "tool_name": "submit_task_result",
            "tool_trace": {
                "tool_calls": [{"name": "submit_task_result", "args": {"failure_reason": summary}}],
                "tool_results": [
                    {
                        "name": "submit_task_result",
                        "content": stringify_tool_content(failure),
                        "tool_call_id": None,
                    }
                ],
                "final_ai_content": summary,
            },
            "event_type": "task_failed",
        }

    approach_args: dict[str, Any] = {
        "target_description": object_description,
        "target_source": target_source,
    }
    if target.get("stop_distance_m") is not None:
        try:
            approach_args["stop_distance_m"] = float(target.get("stop_distance_m"))
        except Exception:
            pass
    timings: dict[str, float] = {}
    _log_structured_navigation(
        "Structured navigation: semantic_object grounding started "
        f"for '{object_description}'."
    )
    approach_start = time.perf_counter()
    try:
        raw_approach = _invoke_tool(approach_tool, approach_args)
    except Exception as exc:
        raw_approach = {
            "status": "error",
            "summary": "Object localization raised an exception.",
            "error": {"message": str(exc)},
        }
    timings["object_grounding_sec"] = _elapsed_since(approach_start)
    _log_structured_navigation(
        "Structured navigation: semantic_object grounding finished "
        f"in {timings['object_grounding_sec']:.3f}s."
    )

    position_destination = _extract_position_destination_from_value(raw_approach)
    if position_destination is None or not tool_result_status_ok(raw_approach):
        parsed = _parse_tool_payload(raw_approach)
        reason = ""
        if isinstance(parsed, dict):
            error = parsed.get("error") if isinstance(parsed.get("error"), dict) else {}
            reason = (
                str(error.get("message") or "").strip()
                or str(parsed.get("summary") or "").strip()
            )
        summary = reason or "Semantic object grounding did not produce a position destination."
        failure = submit_task_result(
            summary=summary,
            selected_object={
                "description": object_description,
                "target_type": "semantic_object",
            },
            failure_reason=summary,
        )
        return {
            "summary": summary,
            "tool_name": "approach_object_in_current_view",
            "tool_trace": {
                "tool_calls": [
                    {"name": "approach_object_in_current_view", "args": approach_args},
                    {"name": "submit_task_result", "args": {"failure_reason": summary}},
                ],
                "tool_results": [
                    {
                        "name": "approach_object_in_current_view",
                        "content": stringify_tool_content(raw_approach),
                        "tool_call_id": None,
                    },
                    {
                        "name": "submit_task_result",
                        "content": stringify_tool_content(failure),
                        "tool_call_id": None,
                    },
                ],
                "final_ai_content": summary,
                "timings_sec": timings,
            },
            "event_type": "task_failed",
        }

    destination = {
        "type": "position",
        "position": position_destination["position"],
        "yaw_deg": float(position_destination.get("yaw_deg") or 0.0),
    }
    selected_object = {
        "description": object_description,
        "target_type": "semantic_object",
        "target_source": target_source,
        "source": "approach_object_in_current_view",
    }
    submit_args = {
        "destination": destination,
        "selected_object": selected_object,
        "summary": f"Resolved semantic object target '{object_description}' to a position destination.",
    }
    submit_start = time.perf_counter()
    submitted = submit_task_result(**submit_args)
    timings["submit_task_result_sec"] = _elapsed_since(submit_start)
    position = destination["position"]
    nav_args = {
        "x": float(position[0]),
        "y": float(position[1]),
        "z": float(position[2]) if len(position) >= 3 else 0.0,
        "yaw_deg": float(destination.get("yaw_deg") or 0.0),
    }
    nav_start = time.perf_counter()
    try:
        raw_navigation = _invoke_tool(navigation_tool, nav_args)
    except Exception as exc:
        raw_navigation = {
            "status": "error",
            "summary": "Position navigation dispatch raised an exception.",
            "error": {"message": str(exc)},
        }
    timings["navigation_dispatch_sec"] = _elapsed_since(nav_start)
    _log_structured_navigation(
        "Structured navigation: semantic_object position dispatch finished "
        f"in {timings['navigation_dispatch_sec']:.3f}s."
    )

    tool_trace = {
        "tool_calls": [
            {"name": "approach_object_in_current_view", "args": approach_args},
            {"name": "submit_task_result", "args": submit_args},
            {"name": "go_to_position", "args": nav_args},
        ],
        "tool_results": [
            {
                "name": "approach_object_in_current_view",
                "content": stringify_tool_content(raw_approach),
                "tool_call_id": None,
            },
            {
                "name": "submit_task_result",
                "content": stringify_tool_content(submitted),
                "tool_call_id": None,
            },
            {
                "name": "go_to_position",
                "content": stringify_tool_content(raw_navigation),
                "tool_call_id": None,
            },
        ],
        "final_ai_content": (
            f"Resolved semantic object '{object_description}' and dispatched position navigation."
        ),
        "timings_sec": timings,
    }
    if tool_result_status_ok(raw_navigation):
        return {
            "summary": navigation_waiting_summary(current_task),
            "tool_name": "go_to_position",
            "tool_trace": tool_trace,
            "event_type": "task_waiting",
        }
    failure = find_tool_failure_message(tool_trace) or "Position navigation dispatch failed."
    return {
        "summary": failure,
        "tool_name": "go_to_position",
        "tool_trace": tool_trace,
        "event_type": "task_failed",
    }


def _dispatch_semantic_keyframe_navigation(
    current_task: TaskItem,
    tools: Sequence[BaseTool],
    target: dict[str, Any],
) -> dict[str, Any]:
    target_source = _target_source(target, default="scene_memory")
    if target_source == "attached_image":
        return _dispatch_attached_image_keyframe_navigation(current_task, tools, target)
    if target_source not in {"scene_memory", "explicit"}:
        summary = (
            "semantic_keyframe target_source must be scene_memory or attached_image; "
            f"got {target_source or 'missing'}."
        )
        failure = submit_task_result(summary=summary, failure_reason=summary)
        return {
            "summary": summary,
            "tool_name": "submit_task_result",
            "tool_trace": {
                "tool_calls": [{"name": "submit_task_result", "args": {"failure_reason": summary}}],
                "tool_results": [
                    {
                        "name": "submit_task_result",
                        "content": stringify_tool_content(failure),
                        "tool_call_id": None,
                    }
                ],
                "final_ai_content": summary,
            },
            "event_type": "task_failed",
        }

    search_tool = find_tool_by_name(tools, "search_requirement_on_keyframe_nodes")
    navigation_tool = find_tool_by_name(tools, "go_to_keyframe")
    if search_tool is None:
        summary = "Keyframe semantic search tool search_requirement_on_keyframe_nodes is unavailable."
        return {
            "summary": summary,
            "tool_name": None,
            "tool_trace": {"tool_calls": [], "tool_results": [], "final_ai_content": summary},
            "event_type": "task_failed",
        }
    if navigation_tool is None:
        summary = "Navigation tool go_to_keyframe is unavailable."
        return {
            "summary": summary,
            "tool_name": None,
            "tool_trace": {"tool_calls": [], "tool_results": [], "final_ai_content": summary},
            "event_type": "task_failed",
        }

    query = _semantic_keyframe_query(current_task, target)
    if not query:
        summary = "semantic_keyframe navigation target is missing query."
        failure = submit_task_result(
            summary=summary,
            failure_reason=summary,
        )
        return {
            "summary": summary,
            "tool_name": "submit_task_result",
            "tool_trace": {
                "tool_calls": [{"name": "submit_task_result", "args": {"failure_reason": summary}}],
                "tool_results": [
                    {
                        "name": "submit_task_result",
                        "content": stringify_tool_content(failure),
                        "tool_call_id": None,
                    }
                ],
                "final_ai_content": summary,
            },
            "event_type": "task_failed",
        }
    if _contains_unresolved_template({"query": query, "target": target}):
        summary = "semantic_keyframe navigation target contains unresolved upstream template text."
        failure = submit_task_result(summary=summary, failure_reason=summary)
        return {
            "summary": summary,
            "tool_name": "submit_task_result",
            "tool_trace": {
                "tool_calls": [{"name": "submit_task_result", "args": {"failure_reason": summary}}],
                "tool_results": [
                    {
                        "name": "submit_task_result",
                        "content": stringify_tool_content(failure),
                        "tool_call_id": None,
                    }
                ],
                "final_ai_content": summary,
            },
            "event_type": "task_failed",
        }

    timings: dict[str, float] = {}
    background_result = _background_result_for_current_task(current_task)
    background_keyframe_id = _recommended_keyframe_id_from_background(background_result)
    if background_keyframe_id is not None and isinstance(background_result, dict):
        raw_search = _semantic_keyframe_search_payload_from_background(
            query=query,
            background_result=background_result,
            recommended_keyframe_id=int(background_keyframe_id),
        )
        timings["keyframe_search_sec"] = 0.0
        search_args = {
            "requirement": query,
            "source": "background_preanalysis",
        }
        _log_structured_navigation(
            "Structured navigation: semantic_keyframe reused background "
            f"recommendation keyframe {background_keyframe_id} for '{query}'."
        )
    else:
        search_args = {"requirement": query}
        _log_structured_navigation(
            "Structured navigation: semantic_keyframe search started "
            f"for '{query}'."
        )
        search_start = time.perf_counter()
        try:
            raw_search = _invoke_tool(search_tool, search_args)
        except Exception as exc:
            raw_search = {
                "status": "error",
                "summary": "Semantic keyframe search raised an exception.",
                "error": {"message": str(exc)},
            }
        timings["keyframe_search_sec"] = _elapsed_since(search_start)
        _log_structured_navigation(
            "Structured navigation: semantic_keyframe search finished "
            f"in {timings['keyframe_search_sec']:.3f}s."
        )

    recommended_keyframe_id = _recommended_keyframe_id_from_search(raw_search)
    search_ok = tool_result_status_ok(raw_search)
    if not search_ok or recommended_keyframe_id is None:
        parsed = _parse_tool_payload(raw_search)
        data = parsed.get("data") if isinstance(parsed, dict) and isinstance(parsed.get("data"), dict) else {}
        reason = ""
        if isinstance(parsed, dict):
            error = parsed.get("error") if isinstance(parsed.get("error"), dict) else {}
            reason = (
                str(error.get("message") or "").strip()
                or str(parsed.get("summary") or "").strip()
            )
        summary = reason or "Semantic keyframe search did not produce a recommended destination."
        failure = submit_task_result(
            summary=summary,
            current_place_context={
                "target_type": "semantic_keyframe",
                "query": query,
                "selection_policy": target.get("selection_policy"),
                "resolution_status": data.get("resolution_status") or "failed",
                "candidate_keyframe_ids": _candidate_keyframe_ids(data)[:8],
            },
            failure_reason=summary,
        )
        return {
            "summary": summary,
            "tool_name": "search_requirement_on_keyframe_nodes",
            "tool_trace": {
                "tool_calls": [
                    {"name": "search_requirement_on_keyframe_nodes", "args": search_args},
                    {"name": "submit_task_result", "args": {"failure_reason": summary}},
                ],
                "tool_results": [
                    {
                        "name": "search_requirement_on_keyframe_nodes",
                        "content": stringify_tool_content(raw_search),
                        "tool_call_id": None,
                    },
                    {
                        "name": "submit_task_result",
                        "content": stringify_tool_content(failure),
                        "tool_call_id": None,
                    },
                ],
                "final_ai_content": summary,
                "timings_sec": timings,
            },
            "event_type": "task_failed",
        }
    destination = {"type": "keyframe", "keyframe_id": int(recommended_keyframe_id)}
    place_context = _semantic_keyframe_context(
        query=query,
        target=target,
        raw_search=raw_search,
        recommended_keyframe_id=int(recommended_keyframe_id),
    )
    submit_args = {
        "destination": destination,
        "current_place_context": place_context,
        "summary": (
            f"Resolved semantic navigation target to keyframe {recommended_keyframe_id}."
        ),
    }
    submit_start = time.perf_counter()
    submitted = submit_task_result(**submit_args)
    timings["submit_task_result_sec"] = _elapsed_since(submit_start)
    _log_structured_navigation(
        "Structured navigation: semantic_keyframe resolved to "
        f"keyframe {recommended_keyframe_id}; result submitted in "
        f"{timings['submit_task_result_sec']:.3f}s."
    )

    nav_args = {"keyframe_node_id": int(recommended_keyframe_id)}
    nav_start = time.perf_counter()
    try:
        raw_navigation = _invoke_tool(navigation_tool, nav_args)
    except Exception as exc:
        raw_navigation = {
            "status": "error",
            "summary": "Navigation dispatch raised an exception.",
            "error": {"message": str(exc)},
        }
    timings["navigation_dispatch_sec"] = _elapsed_since(nav_start)
    _log_structured_navigation(
        "Structured navigation: go_to_keyframe dispatch finished "
        f"in {timings['navigation_dispatch_sec']:.3f}s."
    )

    final_content = (
        f"Resolved semantic keyframe query '{query}' to keyframe "
        f"{recommended_keyframe_id} and dispatched navigation."
    )
    tool_trace = {
        "tool_calls": [
            {"name": "search_requirement_on_keyframe_nodes", "args": search_args},
            {"name": "submit_task_result", "args": submit_args},
            {"name": "go_to_keyframe", "args": nav_args},
        ],
        "tool_results": [
            {
                "name": "search_requirement_on_keyframe_nodes",
                "content": stringify_tool_content(raw_search),
                "tool_call_id": None,
            },
            {
                "name": "submit_task_result",
                "content": stringify_tool_content(submitted),
                "tool_call_id": None,
            },
            {
                "name": "go_to_keyframe",
                "content": stringify_tool_content(raw_navigation),
                "tool_call_id": None,
            },
        ],
        "final_ai_content": final_content,
        "timings_sec": timings,
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


def _dispatch_attached_image_keyframe_navigation(
    current_task: TaskItem,
    tools: Sequence[BaseTool],
    target: dict[str, Any],
) -> dict[str, Any]:
    match_tool = find_tool_by_name(tools, "match_attached_image_to_keyframes")
    navigation_tool = find_tool_by_name(tools, "go_to_keyframe")
    if match_tool is None:
        summary = "Attached-image keyframe matcher match_attached_image_to_keyframes is unavailable."
        return {
            "summary": summary,
            "tool_name": None,
            "tool_trace": {"tool_calls": [], "tool_results": [], "final_ai_content": summary},
            "event_type": "task_failed",
        }
    if navigation_tool is None:
        summary = "Navigation tool go_to_keyframe is unavailable."
        return {
            "summary": summary,
            "tool_name": None,
            "tool_trace": {"tool_calls": [], "tool_results": [], "final_ai_content": summary},
            "event_type": "task_failed",
        }

    image_ref = _task_image_ref(current_task, target)
    query = _semantic_keyframe_query(current_task, target)
    if not image_ref:
        summary = "attached-image semantic_keyframe target is missing image_refs."
        failure = submit_task_result(summary=summary, failure_reason=summary)
        return {
            "summary": summary,
            "tool_name": "submit_task_result",
            "tool_trace": {
                "tool_calls": [{"name": "submit_task_result", "args": {"failure_reason": summary}}],
                "tool_results": [
                    {
                        "name": "submit_task_result",
                        "content": stringify_tool_content(failure),
                        "tool_call_id": None,
                    }
                ],
                "final_ai_content": summary,
            },
            "event_type": "task_failed",
        }
    if _contains_unresolved_template({"query": query, "target": target}):
        summary = "attached-image semantic_keyframe target contains unresolved upstream template text."
        failure = submit_task_result(summary=summary, failure_reason=summary)
        return {
            "summary": summary,
            "tool_name": "submit_task_result",
            "tool_trace": {
                "tool_calls": [{"name": "submit_task_result", "args": {"failure_reason": summary}}],
                "tool_results": [
                    {
                        "name": "submit_task_result",
                        "content": stringify_tool_content(failure),
                        "tool_call_id": None,
                    }
                ],
                "final_ai_content": summary,
            },
            "event_type": "task_failed",
        }

    match_args = {
        "image_ref": image_ref,
        "query": query,
        "focus": _target_image_focus(target),
    }
    timings: dict[str, float] = {}
    _log_structured_navigation(
        "Structured navigation: attached-image semantic_keyframe match started "
        f"for image_ref='{image_ref}', focus='{match_args['focus']}', query='{query}'."
    )
    match_start = time.perf_counter()
    try:
        raw_match = _invoke_tool(match_tool, match_args)
    except TypeError:
        legacy_args = {"image_ref": image_ref, "query": query}
        raw_match = _invoke_tool(match_tool, legacy_args)
        match_args = legacy_args
    except Exception as exc:
        raw_match = {
            "status": "error",
            "summary": "Attached-image keyframe match raised an exception.",
            "error": {"message": str(exc)},
        }
    timings["attached_image_match_sec"] = _elapsed_since(match_start)
    _log_structured_navigation(
        "Structured navigation: attached-image semantic_keyframe match finished "
        f"in {timings['attached_image_match_sec']:.3f}s."
    )

    recommended_keyframe_id = _recommended_keyframe_id_from_search(raw_match)
    match_ok = tool_result_status_ok(raw_match)
    if not match_ok or recommended_keyframe_id is None:
        parsed = _parse_tool_payload(raw_match)
        data = parsed.get("data") if isinstance(parsed, dict) and isinstance(parsed.get("data"), dict) else {}
        reason = ""
        if isinstance(parsed, dict):
            error = parsed.get("error") if isinstance(parsed.get("error"), dict) else {}
            reason = (
                str(error.get("message") or "").strip()
                or str(parsed.get("summary") or "").strip()
            )
        summary = reason or "Attached-image keyframe matching did not produce a recommended destination."
        failure = submit_task_result(
            summary=summary,
            current_place_context={
                "target_type": "semantic_keyframe",
                "target_source": "attached_image",
                "image_focus": match_args.get("focus"),
                "image_ref": image_ref,
                "query": query,
                "resolution_status": data.get("resolution_status") or "failed",
                "candidate_keyframe_ids": _candidate_keyframe_ids(data)[:8],
            },
            failure_reason=summary,
        )
        return {
            "summary": summary,
            "tool_name": "match_attached_image_to_keyframes",
            "tool_trace": {
                "tool_calls": [
                    {"name": "match_attached_image_to_keyframes", "args": match_args},
                    {"name": "submit_task_result", "args": {"failure_reason": summary}},
                ],
                "tool_results": [
                    {
                        "name": "match_attached_image_to_keyframes",
                        "content": stringify_tool_content(raw_match),
                        "tool_call_id": None,
                    },
                    {
                        "name": "submit_task_result",
                        "content": stringify_tool_content(failure),
                        "tool_call_id": None,
                    },
                ],
                "final_ai_content": summary,
                "timings_sec": timings,
            },
            "event_type": "task_failed",
        }

    destination = {"type": "keyframe", "keyframe_id": int(recommended_keyframe_id)}
    place_context = _semantic_keyframe_context(
        query=query,
        target={**target, "selection_policy": target.get("selection_policy")},
        raw_search=raw_match,
        recommended_keyframe_id=int(recommended_keyframe_id),
    )
    place_context.update(
        {
            "target_source": "attached_image",
            "image_focus": match_args.get("focus"),
            "image_ref": image_ref,
        }
    )
    submit_args = {
        "destination": destination,
        "current_place_context": place_context,
        "summary": (
            f"Matched attached image target to keyframe {recommended_keyframe_id}."
        ),
    }
    submit_start = time.perf_counter()
    submitted = submit_task_result(**submit_args)
    timings["submit_task_result_sec"] = _elapsed_since(submit_start)

    nav_args = {"keyframe_node_id": int(recommended_keyframe_id)}
    nav_start = time.perf_counter()
    try:
        raw_navigation = _invoke_tool(navigation_tool, nav_args)
    except Exception as exc:
        raw_navigation = {
            "status": "error",
            "summary": "Navigation dispatch raised an exception.",
            "error": {"message": str(exc)},
        }
    timings["navigation_dispatch_sec"] = _elapsed_since(nav_start)
    _log_structured_navigation(
        "Structured navigation: attached-image go_to_keyframe dispatch finished "
        f"in {timings['navigation_dispatch_sec']:.3f}s."
    )

    tool_trace = {
        "tool_calls": [
            {"name": "match_attached_image_to_keyframes", "args": match_args},
            {"name": "submit_task_result", "args": submit_args},
            {"name": "go_to_keyframe", "args": nav_args},
        ],
        "tool_results": [
            {
                "name": "match_attached_image_to_keyframes",
                "content": stringify_tool_content(raw_match),
                "tool_call_id": None,
            },
            {
                "name": "submit_task_result",
                "content": stringify_tool_content(submitted),
                "tool_call_id": None,
            },
            {
                "name": "go_to_keyframe",
                "content": stringify_tool_content(raw_navigation),
                "tool_call_id": None,
            },
        ],
        "final_ai_content": (
            f"Matched attached image target to keyframe {recommended_keyframe_id} "
            "and dispatched navigation."
        ),
        "timings_sec": timings,
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


def _dispatch_position_navigation(
    current_task: Optional[TaskItem],
    tools: Sequence[BaseTool],
    position_destination: Optional[dict[str, Any]],
    error: Optional[str],
) -> dict[str, Any]:
    if error or position_destination is None:
        summary = error or "Position navigation target is unavailable."
        return {
            "summary": summary,
            "tool_name": None,
            "tool_trace": {"tool_calls": [], "tool_results": [], "final_ai_content": summary},
            "event_type": "task_failed",
        }
    navigation_tool = find_tool_by_name(tools, "go_to_position")
    if navigation_tool is None:
        summary = "Navigation tool go_to_position is unavailable."
        return {
            "summary": summary,
            "tool_name": None,
            "tool_trace": {"tool_calls": [], "tool_results": [], "final_ai_content": summary},
            "event_type": "task_failed",
        }
    position = position_destination["position"]
    tool_args = {
        "x": float(position[0]),
        "y": float(position[1]),
        "z": float(position[2]) if len(position) >= 3 else 0.0,
        "yaw_deg": float(position_destination.get("yaw_deg") or 0.0),
    }
    try:
        raw_navigation = _invoke_tool(navigation_tool, tool_args)
    except Exception as exc:
        raw_navigation = {
            "status": "error",
            "summary": "Position navigation dispatch raised an exception.",
            "error": {"message": str(exc)},
        }

    tool_trace = {
        "tool_calls": [{"name": "go_to_position", "args": tool_args}],
        "tool_results": [
            {
                "name": "go_to_position",
                "content": stringify_tool_content(raw_navigation),
                "tool_call_id": None,
            }
        ],
        "final_ai_content": (
            f"Dispatched structured navigation_action to position "
            f"[{tool_args['x']:.3f}, {tool_args['y']:.3f}, {tool_args['z']:.3f}]."
        ),
    }
    if tool_result_status_ok(raw_navigation):
        return {
            "summary": navigation_waiting_summary(current_task),
            "tool_name": "go_to_position",
            "tool_trace": tool_trace,
            "event_type": "task_waiting",
        }
    failure = find_tool_failure_message(tool_trace) or "Position navigation dispatch failed."
    return {
        "summary": failure,
        "tool_name": "go_to_position",
        "tool_trace": tool_trace,
        "event_type": "task_failed",
    }


__all__ = [
    "find_tool_by_name",
    "resolve_navigation_action_keyframe_id",
    "resolve_navigation_action_position",
    "tool_result_status_ok",
    "try_dispatch_structured_navigation_action",
]
