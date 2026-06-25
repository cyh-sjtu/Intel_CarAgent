"""Persistent run-memory storage and query helpers for the async agent."""

from __future__ import annotations

import json
import os
import re
import threading
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from .exports import write_memory_tables
from ..runtime.legacy_task_metadata import (
    legacy_object_kind,
    legacy_staging_kind,
    legacy_upstream_task_id,
)


OBSERVATION_TOOL_NAMES = {
    "analyse_on_current_image",
    "get_current_state",
}

MEMORY_VIEW_BUDGET_CHARS = {
    "summary_table": 12_000,
    "timeline": 12_000,
    "detail": 24_000,
}

ARTIFACT_PATH_KEYS = {
    "output_dir",
    "summary_json",
    "status_json",
    "approach_goal_json",
    "approach_debug_png",
    "debug_png",
    "mono_guard_json",
    "selected_grounding_json",
    "stereo_json",
    "segmentation_json",
}

DROP_FROM_LLM_MEMORY_KEYS = {
    "occupancy_grid",
    "map",
    "costmap",
    "grid",
    "raw_output",
    "raw_result",
    "raw_trace",
    "tool_results",
    "mask",
    "masks",
    "segmentation_mask",
    "depth_map",
    "disparity",
    "point_cloud",
    "image_data",
    "image_base64",
}


def _utc_now_iso() -> str:
    """Return one ISO timestamp for runtime-memory records."""

    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _truncate_text(value: Any, limit: int = 400) -> str:
    """Return one compact text preview for indexing and summaries."""

    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[: max(limit - 3, 0)].rstrip() + "..."


def _tokenize_query(value: Any) -> set[str]:
    """Return simple lowercase tokens for deterministic runtime-memory matching."""

    return {
        token.lower()
        for token in re.findall(r"[A-Za-z0-9_]+", str(value or ""))
        if token
    }


def _safe_int(value: Any) -> Optional[int]:
    """Coerce loose integer payloads without raising."""

    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _safe_float_triplet(value: Any) -> Optional[list[float]]:
    """Coerce one xyz-like sequence into a float triplet."""

    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return None
    try:
        return [float(value[0]), float(value[1]), float(value[2])]
    except Exception:
        return None


def _extract_tool_result_json(item: dict[str, Any], tool_name: str) -> Optional[dict[str, Any]]:
    """Return structured JSON content for one tool result embedded in task memory."""

    trace = item.get("tool_trace_excerpt")
    if not isinstance(trace, dict):
        return None
    for result in list(trace.get("tool_results") or []):
        if str(result.get("name") or "").strip() != tool_name:
            continue
        content = result.get("content")
        if isinstance(content, dict):
            return content
        try:
            parsed = json.loads(str(content or ""))
        except Exception:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _extract_structured_tool_data(raw_value: Any) -> Any:
    """Return the data field from one structured tool result payload."""

    parsed = raw_value
    if isinstance(raw_value, str):
        try:
            parsed = json.loads(raw_value)
        except Exception:
            return None
    if not isinstance(parsed, dict):
        return None
    if "data" in parsed and "status" in parsed:
        return parsed.get("data")
    return parsed


def _latest_tool_payload(tool_trace: dict[str, Any], tool_name: str) -> Any:
    """Return the latest payload for one tool name in a stored trace."""

    for tool_result in reversed(list(tool_trace.get("tool_results") or [])):
        if str(tool_result.get("name") or "").strip() != tool_name:
            continue
        return _extract_structured_tool_data(tool_result.get("content"))
    return None


def _observation_summary_from_trace(
    *,
    summary: str,
    tool_trace: dict[str, Any],
) -> Optional[str]:
    """Return an observation summary when execution actually used observation tools."""

    tool_names = {
        str(call.get("name") or "").strip()
        for call in list(tool_trace.get("tool_calls") or [])
        if isinstance(call, dict)
    }
    tool_names.update(
        str(result.get("name") or "").strip()
        for result in list(tool_trace.get("tool_results") or [])
        if isinstance(result, dict)
    )
    if not tool_names.intersection(OBSERVATION_TOOL_NAMES):
        return None

    image_payload = _latest_tool_payload(tool_trace, "analyse_on_current_image")
    if isinstance(image_payload, dict):
        answer = str(image_payload.get("answer") or "").strip()
        if answer:
            return _truncate_text(answer, 1000)

    final_ai_content = str(tool_trace.get("final_ai_content") or "").strip()
    if final_ai_content:
        return _truncate_text(final_ai_content, 1000)

    return _truncate_text(summary, 1000) if summary else None


def _observation_location_from_trace(
    tool_trace: dict[str, Any],
) -> tuple[Optional[list[float]], dict[str, Any]]:
    """Extract lightweight location details from observation tool payloads."""

    details: dict[str, Any] = {
        "observation_tools": sorted(
            {
                str(result.get("name") or "").strip()
                for result in list(tool_trace.get("tool_results") or [])
                if isinstance(result, dict)
                and str(result.get("name") or "").strip() in OBSERVATION_TOOL_NAMES
            }
        )
    }
    current_state = _latest_tool_payload(tool_trace, "get_current_state")
    position = None
    if isinstance(current_state, dict):
        position = _safe_float_triplet(current_state.get("position"))
        if current_state.get("status") is not None:
            details["current_status"] = current_state.get("status")
    return position, details


def _to_jsonable(value: Any) -> Any:
    """Convert one arbitrary runtime value into a JSON-safe representation."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        try:
            return _to_jsonable(value.tolist())
        except Exception:
            pass
    if hasattr(value, "content") and hasattr(value, "__class__"):
        return {
            "message_type": value.__class__.__name__,
            "content": _truncate_text(getattr(value, "content", "")),
            "name": getattr(value, "name", None),
        }
    return str(value)


def _coerce_positive_int(
    value: Any,
    *,
    default: int,
    minimum: int = 1,
    maximum: int = 50,
) -> tuple[int, Optional[str]]:
    """Coerce a loose integer value and return an optional warning."""

    try:
        resolved = int(value)
    except Exception:
        return default, f"Invalid integer value {value!r}; using {default}."
    if resolved < minimum:
        return minimum, f"Integer value {resolved} is below {minimum}; using {minimum}."
    if resolved > maximum:
        return maximum, f"Integer value {resolved} is above {maximum}; using {maximum}."
    return resolved, None


def _task_excerpt(task: dict[str, Any]) -> dict[str, Any]:
    """Build one compact task snapshot for memory indexing."""

    latest_result = (task.get("result") or [])[-1] if task.get("result") else None
    return _clean_empty_fields({
        "task_id": task.get("task_id"),
        "description": task.get("description"),
        "task_type": task.get("task_type"),
        "target": _to_jsonable(task.get("target")),
        "type": task.get("type"),
        "status": task.get("status"),
        "next_task_id": task.get("next_task_id"),
        "branches": _to_jsonable(task.get("branches")),
        "plan_id": task.get("plan_id"),
        "user_input_id": task.get("user_input_id"),
        "depends_on": task.get("depends_on", []),
        "inputs_from": _to_jsonable(task.get("inputs_from")),
        "outputs": _to_jsonable(task.get("outputs")),
        "selection_policy": task.get("selection_policy"),
        "image_refs": _to_jsonable(task.get("image_refs")),
        "wait_for_event": task.get("wait_for_event"),
        "latest_result": _to_jsonable(latest_result) if latest_result else None,
    })


def _record_time(record: dict[str, Any]) -> str:
    """Return the best chronological timestamp for a memory record."""

    return str(
        record.get("recorded_at")
        or record.get("created_at")
        or record.get("completed_at")
        or record.get("updated_at")
        or ""
    )


def _row_preview(value: Any, limit: int = 220) -> str:
    """Return a short user-facing preview for memory table rows."""

    if isinstance(value, dict):
        for key in ("summary", "response", "text", "description", "message", "plan_text"):
            if value.get(key):
                return _truncate_text(value.get(key), limit)
        return _truncate_text(json.dumps(value, ensure_ascii=False), limit)
    return _truncate_text(value, limit)


def _parse_json_like(value: Any) -> Any:
    """Parse JSON strings when possible; otherwise return the original value."""

    if isinstance(value, str):
        text = value.strip()
        if not text:
            return value
        try:
            return json.loads(text)
        except Exception:
            return value
    return value


def _json_char_count(value: Any) -> int:
    """Estimate JSON payload size in characters."""

    try:
        return len(json.dumps(value, ensure_ascii=False, default=str))
    except Exception:
        return len(str(value))


def _structured_tool_payload(result: dict[str, Any]) -> Any:
    """Return parsed content for one tool result without expanding raw payloads."""

    if not isinstance(result, dict):
        return None
    content = _parse_json_like(result.get("content"))
    if isinstance(content, dict):
        return content
    if result.get("observation") or result.get("summary") or result.get("status"):
        return {
            "status": result.get("status"),
            "summary": result.get("summary") or result.get("observation"),
        }
    return content


def _data_payload(payload: Any) -> Any:
    """Return structured data from normalized tool payloads."""

    parsed = _parse_json_like(payload)
    if isinstance(parsed, dict) and "data" in parsed and "status" in parsed:
        return parsed.get("data")
    return parsed


def _compact_args(args: Any) -> dict[str, Any]:
    """Return short argument previews for tool evidence."""

    if not isinstance(args, dict):
        return {"value": _truncate_text(args, 200)} if args is not None else {}
    compact: dict[str, Any] = {}
    for key, value in args.items():
        key_text = str(key)
        if key_text in DROP_FROM_LLM_MEMORY_KEYS:
            compact[key_text] = "<omitted>"
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            compact[key_text] = _truncate_text(value, 240) if isinstance(value, str) else value
        elif isinstance(value, (list, tuple)):
            compact[key_text] = [_to_jsonable(item) for item in list(value)[:8]]
            if len(value) > 8:
                compact[key_text].append({"truncated_count": len(value) - 8})
        elif isinstance(value, dict):
            compact[key_text] = _shrink_for_memory(value, text_limit=160, list_limit=6)
        else:
            compact[key_text] = _truncate_text(value, 180)
    return compact


def _extract_destination(value: Any, *, _depth: int = 0) -> Optional[dict[str, Any]]:
    """Extract one compact destination from known tool result shapes."""

    if _depth > 6:
        return None
    parsed = _parse_json_like(value)
    if isinstance(parsed, dict):
        if "destination" in parsed:
            destination = _extract_destination(parsed.get("destination"), _depth=_depth + 1)
            if destination is not None:
                return destination

        destination_type = str(parsed.get("type") or parsed.get("destination_type") or "").strip()
        if destination_type == "position" or parsed.get("position") is not None:
            position = _safe_float_triplet(parsed.get("position") or parsed.get("target_position"))
            yaw = (
                parsed.get("yaw")
                if parsed.get("yaw") is not None
                else parsed.get("yaw_deg")
                if parsed.get("yaw_deg") is not None
                else parsed.get("target_yaw_deg")
            )
            destination = {
                "type": "position",
                "position": position,
                "yaw": yaw,
            }
            source_keyframe_id = _safe_int(parsed.get("keyframe_id"))
            if source_keyframe_id is not None and destination_type != "keyframe":
                destination["source_keyframe_id"] = source_keyframe_id
            return destination
        if destination_type == "keyframe" or parsed.get("keyframe_id") is not None:
            keyframe_id = _safe_int(parsed.get("keyframe_id"))
            position = _safe_float_triplet(parsed.get("target_position"))
            return {
                "type": "keyframe",
                "keyframe_id": keyframe_id,
                "position": position,
            }

        keyframe_id = _safe_int(parsed.get("target_keyframe_id") or parsed.get("destination_keyframe_id"))
        target_position = _safe_float_triplet(
            parsed.get("target_position") or parsed.get("destination_position")
        )
        if keyframe_id is not None or target_position is not None:
            return {
                "type": "keyframe" if keyframe_id is not None else "position",
                "keyframe_id": keyframe_id,
                "position": target_position,
                "yaw": parsed.get("target_yaw_deg"),
            }

        for key in ("data", "result", "target", "navigation_goal", "recommended_destination"):
            destination = _extract_destination(parsed.get(key), _depth=_depth + 1)
            if destination is not None:
                return destination
    return None


def _merge_artifact_paths(*values: Any) -> dict[str, str]:
    """Extract known artifact paths from nested payloads."""

    artifacts: dict[str, str] = {}

    def visit(node: Any, depth: int = 0) -> None:
        if depth > 7:
            return
        parsed = _parse_json_like(node)
        if isinstance(parsed, dict):
            nested_artifacts = parsed.get("artifact_paths")
            if isinstance(nested_artifacts, dict):
                for key, value in nested_artifacts.items():
                    if value:
                        artifacts[str(key)] = str(value)
            for key, value in parsed.items():
                key_text = str(key)
                if key_text in ARTIFACT_PATH_KEYS and value:
                    artifacts[key_text] = str(value)
                elif isinstance(value, (dict, list, tuple)):
                    visit(value, depth + 1)
        elif isinstance(parsed, (list, tuple)):
            for item in list(parsed)[:40]:
                visit(item, depth + 1)

    for value in values:
        visit(value)
    return artifacts


def _extract_candidate_keyframe_ids(value: Any) -> list[int]:
    """Return candidate/recommended keyframe IDs from nested payloads."""

    ids: list[int] = []
    seen: set[int] = set()

    def add(raw: Any) -> None:
        resolved = _safe_int(raw)
        if resolved is None or resolved in seen:
            return
        seen.add(resolved)
        ids.append(resolved)

    def visit(node: Any, depth: int = 0) -> None:
        if depth > 6 or len(ids) >= 12:
            return
        parsed = _parse_json_like(node)
        if isinstance(parsed, dict):
            for key in (
                "candidate_keyframe_ids",
                "matched_keyframe_ids",
                "keyframe_ids",
                "candidate_keyframes",
                "candidates",
            ):
                value = parsed.get(key)
                if isinstance(value, (list, tuple)):
                    for item in value:
                        if isinstance(item, dict):
                            add(item.get("keyframe_id") or item.get("id") or item.get("kf_id"))
                        else:
                            add(item)
                elif value is not None:
                    add(value)
            for key in (
                "keyframe_id",
                "target_keyframe_id",
                "destination_keyframe_id",
                "recommended_keyframe_id",
                "best_keyframe_id",
                "id",
                "kf_id",
            ):
                if "keyframe" in key or key in {"kf_id"}:
                    add(parsed.get(key))
            for value in parsed.values():
                if isinstance(value, (dict, list, tuple)):
                    visit(value, depth + 1)
        elif isinstance(parsed, (list, tuple)):
            for item in list(parsed)[:40]:
                visit(item, depth + 1)

    visit(value)
    return ids[:12]


def _extract_failure_reason(payload: Any, result: dict[str, Any]) -> Optional[str]:
    """Return one compact failure reason from tool payloads."""

    parsed = _data_payload(payload)
    sources = []
    if isinstance(payload, dict):
        sources.append(payload)
    if isinstance(parsed, dict):
        sources.append(parsed)
    sources.append(result)
    for source in sources:
        if not isinstance(source, dict):
            continue
        for key in ("failure_reason", "reason", "error", "message"):
            value = source.get(key)
            if value:
                if isinstance(value, dict):
                    return _truncate_text(value.get("message") or value, 300)
                return _truncate_text(value, 300)
    return None


def _failure_reason_from_task_row(row: dict[str, Any]) -> Optional[str]:
    """Return one short task failure reason from compact task memory fields."""

    if str(row.get("status") or "").strip().lower() != "failed":
        return None
    for key in ("failure_reason", "error", "summary"):
        value = row.get(key)
        if value:
            if isinstance(value, dict):
                message = value.get("message") or value.get("reason") or value
                return _truncate_text(message, 320)
            return _truncate_text(value, 320)
    result = row.get("result")
    parsed_result = _parse_json_like(result)
    if isinstance(parsed_result, dict):
        for key in ("failure_reason", "error", "summary", "message"):
            value = parsed_result.get(key)
            if value:
                if isinstance(value, dict):
                    message = value.get("message") or value.get("reason") or value
                    return _truncate_text(message, 320)
                return _truncate_text(value, 320)
    return None


def _extract_key_metrics(tool_name: str, payload: Any, result: dict[str, Any]) -> dict[str, Any]:
    """Extract compact tool-specific metrics without raw maps/images."""

    data = _data_payload(payload)
    metrics: dict[str, Any] = {}
    source = data if isinstance(data, dict) else payload if isinstance(payload, dict) else {}
    if not isinstance(source, dict):
        source = {}

    if tool_name == "get_current_state":
        return {
            "position": _safe_float_triplet(source.get("position")),
            "orientation": _to_jsonable(source.get("orientation")),
            "status": source.get("status"),
            "source": source.get("source"),
        }

    if tool_name in {"go_to_keyframe", "go_to_position", "navigation_to_position"}:
        for key in (
            "target_keyframe_id",
            "target_position",
            "target_yaw_deg",
            "path_waypoint_count",
            "navigation_status",
        ):
            if source.get(key) is not None:
                metrics[key] = _to_jsonable(source.get(key))
        return metrics

    if "approach_object" in tool_name or tool_name == "resolve_object_from_attached_image":
        for key in (
            "depth_backend",
            "approach_status",
            "approach_reason",
            "mode",
            "candidate_count",
            "grid_source",
            "object_base_range_xy_m",
            "object_bearing_deg",
        ):
            if source.get(key) is not None:
                metrics[key] = _to_jsonable(source.get(key))
        for nested_key in ("approach_goal", "mono_guard", "stereo", "depth"):
            nested = source.get(nested_key)
            if isinstance(nested, dict):
                for key in ("status", "reason", "selected_source", "selected_depth_m", "valid_ratio", "valid_pixels"):
                    if nested.get(key) is not None:
                        metrics[f"{nested_key}_{key}"] = _to_jsonable(nested.get(key))
        return metrics

    for key in (
        "recommended_keyframe_id",
        "target_keyframe_id",
        "match_confidence",
        "recommendation_confidence",
        "candidate_count",
        "navigation_status",
    ):
        if source.get(key) is not None:
            metrics[key] = _to_jsonable(source.get(key))
    if result.get("status") is not None:
        metrics.setdefault("result_status", result.get("status"))
    return metrics


def _tool_summary_text(payload: Any, result: dict[str, Any]) -> str:
    """Build one short human-readable observation summary for tool evidence."""

    parsed = _data_payload(payload)
    candidates: list[Any] = []
    for source in (payload, parsed, result):
        if isinstance(source, dict):
            for key in ("summary", "observation", "answer", "message", "description"):
                if source.get(key):
                    candidates.append(source.get(key))
        elif source:
            candidates.append(source)
    for candidate in candidates:
        text = _truncate_text(candidate, 300)
        if text:
            return text
    return ""


def _normalize_compact_tool_evidence_item(item: Any) -> Optional[dict[str, Any]]:
    """Normalize one LLM-facing tool evidence item to the compact schema."""

    parsed = _parse_json_like(item)
    if not isinstance(parsed, dict):
        text = _truncate_text(parsed, 300)
        return {"summary": text} if text else None

    tool_name = str(parsed.get("name") or parsed.get("tool_name") or "").strip()
    compact: dict[str, Any] = {}
    if tool_name:
        compact["name"] = tool_name

    args = parsed.get("args_summary")
    if args is None:
        args = parsed.get("args")
    if args not in (None, "", [], {}):
        compact["args_summary"] = _compact_args(args)

    for key in ("status", "summary", "failure_reason"):
        value = parsed.get(key)
        if value in (None, "", [], {}):
            continue
        compact[key] = _truncate_text(value, 500 if key == "summary" else 220)

    destination = _extract_destination(parsed)
    if destination is not None:
        compact["destination"] = destination

    candidates = _extract_candidate_keyframe_ids(parsed)
    if candidates:
        compact["candidate_keyframe_ids"] = candidates

    artifacts = _merge_artifact_paths(parsed)
    if artifacts:
        compact["artifact_paths"] = artifacts

    metrics = parsed.get("key_metrics")
    if metrics in (None, "", [], {}):
        metrics = _extract_key_metrics(tool_name, parsed, parsed)
    if metrics not in (None, "", [], {}):
        compact["key_metrics"] = _shrink_for_memory(metrics, text_limit=160, list_limit=8)

    if compact:
        return compact

    shrunk = _shrink_for_memory(parsed, text_limit=180, list_limit=6)
    text = _truncate_text(json.dumps(shrunk, ensure_ascii=False), 500)
    return {"summary": text} if text else None


def _normalize_compact_tool_evidence(items: Any) -> list[dict[str, Any]]:
    """Return bounded compact tool evidence from any tool-evidence-like list."""

    if not isinstance(items, list):
        return []
    normalized: list[dict[str, Any]] = []
    for item in items[:8]:
        compact = _normalize_compact_tool_evidence_item(item)
        if compact:
            normalized.append(compact)
    if len(items) > 8:
        normalized.append({"summary": f"{len(items) - 8} additional tool calls omitted."})
    return normalized


def _compact_tool_evidence(tool_trace: Optional[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return compact tool evidence suitable for LLM-facing memory views."""

    if not isinstance(tool_trace, dict):
        return []
    if isinstance(tool_trace.get("tools"), list):
        return _normalize_compact_tool_evidence(tool_trace.get("tools"))
    calls = [call for call in list(tool_trace.get("tool_calls") or []) if isinstance(call, dict)]
    results = [result for result in list(tool_trace.get("tool_results") or []) if isinstance(result, dict)]
    summarized: list[dict[str, Any]] = []
    max_count = max(len(calls), len(results))
    for index in range(max_count):
        call = calls[index] if index < len(calls) else {}
        result = results[index] if index < len(results) else {}
        tool_name = str(call.get("name") or result.get("name") or "").strip()
        payload = _structured_tool_payload(result)
        status = result.get("status")
        if isinstance(payload, dict):
            status = status or payload.get("status")
        destination = None if tool_name == "get_current_state" else _extract_destination(payload)
        summary = {
            "name": tool_name,
            "args_summary": _compact_args(call.get("args") or {}),
            "status": status,
            "summary": _tool_summary_text(payload, result),
            "destination": destination,
            "candidate_keyframe_ids": _extract_candidate_keyframe_ids(payload),
            "artifact_paths": _merge_artifact_paths(payload),
            "failure_reason": _extract_failure_reason(payload, result),
            "key_metrics": _extract_key_metrics(tool_name, payload, result),
        }
        summarized.append(
            {key: value for key, value in summary.items() if value not in (None, "", [], {})}
        )
    return _normalize_compact_tool_evidence(summarized)


def _extract_tool_summary(tool_trace: Optional[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return compact tool evidence suitable for task memory detail views."""

    return _compact_tool_evidence(tool_trace)


def _extract_first_destination_from_evidence(evidence: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
    """Return the first destination from compact tool evidence."""

    for item in evidence:
        if isinstance(item, dict) and isinstance(item.get("destination"), dict):
            return item.get("destination")
    return None


def _extract_artifacts_from_evidence(evidence: list[dict[str, Any]]) -> dict[str, str]:
    """Merge artifact paths from compact tool evidence."""

    artifacts: dict[str, str] = {}
    for item in evidence:
        if isinstance(item, dict) and isinstance(item.get("artifact_paths"), dict):
            artifacts.update({str(key): str(value) for key, value in item["artifact_paths"].items()})
    return artifacts


def _extract_candidates_from_evidence(evidence: list[dict[str, Any]]) -> list[int]:
    """Merge candidate keyframe IDs from compact tool evidence."""

    ids: list[int] = []
    seen: set[int] = set()
    for item in evidence:
        if not isinstance(item, dict):
            continue
        for raw in list(item.get("candidate_keyframe_ids") or []):
            candidate_id = _safe_int(raw)
            if candidate_id is None or candidate_id in seen:
                continue
            seen.add(candidate_id)
            ids.append(candidate_id)
    return ids[:12]


def _compact_destination_for_table(value: Any) -> Optional[dict[str, Any]]:
    """Return a small destination shape for summary-table rows."""

    destination = _extract_destination(value)
    if destination is None:
        return None
    compact: dict[str, Any] = {"type": destination.get("type")}
    if destination.get("keyframe_id") is not None:
        compact["keyframe_id"] = destination.get("keyframe_id")
    position = _safe_float_triplet(destination.get("position"))
    if position is not None:
        compact["position"] = [round(item, 3) for item in position]
    if destination.get("yaw") is not None:
        try:
            compact["yaw"] = round(float(destination.get("yaw")), 2)
        except Exception:
            compact["yaw"] = destination.get("yaw")
    return {key: value for key, value in compact.items() if value is not None}


def _compact_task_result_value(value: Any) -> Any:
    """Compact task result storage so memory detail never embeds raw tool dumps."""

    parsed = _parse_json_like(value)
    if isinstance(parsed, list):
        return [_compact_task_result_value(item) for item in parsed[:6]]
    if not isinstance(parsed, dict):
        return _truncate_text(parsed, 500) if isinstance(parsed, str) else _to_jsonable(parsed)
    compact: dict[str, Any] = {}
    omitted_trace_summary: dict[str, Any] = {}
    raw_output = parsed.get("raw_output")
    if raw_output not in (None, "", [], {}):
        raw_payload = _parse_json_like(raw_output)
        omitted_trace_summary["omitted_chars"] = len(str(raw_output))
        if isinstance(raw_payload, dict):
            omitted_trace_summary["tool_evidence"] = _compact_tool_evidence(raw_payload)
            final_text = _truncate_text(raw_payload.get("final_ai_content"), 400)
            if final_text:
                omitted_trace_summary["final_ai_content"] = final_text
        else:
            omitted_trace_summary["preview"] = _truncate_text(raw_output, 300)
    for key in (
        "status",
        "summary",
        "response",
        "error",
        "message",
        "tool_name",
        "task_status",
        "navigation_status",
    ):
        if parsed.get(key) is not None:
            compact[key] = (
                _truncate_text(parsed.get(key), 700)
                if isinstance(parsed.get(key), str)
                else _to_jsonable(parsed.get(key))
            )
    destination = _extract_destination(parsed)
    if destination is not None:
        compact["destination"] = destination
    artifacts = _merge_artifact_paths(parsed)
    if artifacts:
        compact["artifact_paths"] = artifacts
    candidates = _extract_candidate_keyframe_ids(parsed)
    if candidates:
        compact["candidate_keyframe_ids"] = candidates
    if omitted_trace_summary:
        compact["omitted_trace_summary"] = {
            key: value
            for key, value in omitted_trace_summary.items()
            if value not in (None, "", [], {})
        }
    if not compact:
        shrunk = _shrink_for_memory(parsed)
        if shrunk:
            compact["summary"] = _truncate_text(json.dumps(shrunk, ensure_ascii=False), 700)
        else:
            compact["summary"] = "Raw task payload omitted from memory detail."
    return compact


def _shrink_for_memory(value: Any, *, text_limit: int = 240, list_limit: int = 12) -> Any:
    """Recursively shrink arbitrary values for memory-budget fallback."""

    parsed = _parse_json_like(value)
    if isinstance(parsed, str):
        return _truncate_text(parsed, text_limit)
    if parsed is None or isinstance(parsed, (int, float, bool)):
        return parsed
    if isinstance(parsed, Path):
        return str(parsed)
    if isinstance(parsed, dict):
        compact: dict[str, Any] = {}
        for key, item in parsed.items():
            key_text = str(key)
            if key_text in DROP_FROM_LLM_MEMORY_KEYS:
                continue
            if key_text == "candidate_keyframe_ids" and isinstance(item, (list, tuple)):
                candidate_ids = [
                    resolved
                    for resolved in (_safe_int(raw) for raw in list(item))
                    if resolved is not None
                ]
                compact[key_text] = candidate_ids[:list_limit]
                if len(candidate_ids) > list_limit:
                    compact["candidate_keyframe_ids_truncated_count"] = len(candidate_ids) - list_limit
                continue
            compact[key_text] = _shrink_for_memory(
                item,
                text_limit=text_limit,
                list_limit=list_limit,
            )
        return compact
    if isinstance(parsed, (list, tuple, set)):
        items = list(parsed)
        compact_items = [
            _shrink_for_memory(item, text_limit=text_limit, list_limit=list_limit)
            for item in items[:list_limit]
        ]
        if len(items) > list_limit:
            compact_items.append({"truncated_count": len(items) - list_limit})
        return compact_items
    return _truncate_text(parsed, text_limit)


def _summary_budget_item(item: dict[str, Any], *, text_limit: int) -> dict[str, Any]:
    """Keep one summary-table row as a tiny index entry."""

    compact: dict[str, Any] = {}
    for key in (
        "row_id",
        "scope",
        "time",
        "task_id",
        "status",
        "status_detail",
        "wait_for_event",
        "task_type",
        "result_kind",
        "keyframe_id",
        "order",
        "source",
        "mode",
        "version",
        "task_count",
        "success",
        "has_destination",
    ):
        if item.get(key) not in (None, "", [], {}):
            compact[key] = item.get(key)
    if item.get("intent_label"):
        compact["intent_label"] = _truncate_text(item.get("intent_label"), text_limit)
    if item.get("destination_summary"):
        compact["destination_summary"] = _truncate_text(item.get("destination_summary"), 80)
    if isinstance(item.get("candidate_summary"), dict):
        candidate_summary = dict(item["candidate_summary"])
        if isinstance(candidate_summary.get("ids"), list):
            candidate_summary["ids"] = candidate_summary["ids"][:6]
        compact["candidate_summary"] = _clean_empty_fields(candidate_summary)
    if isinstance(item.get("artifact_summary"), dict):
        compact["artifact_summary"] = item.get("artifact_summary")
    if isinstance(item.get("evidence_summary"), dict):
        evidence_summary = dict(item["evidence_summary"])
        if isinstance(evidence_summary.get("tools"), list):
            evidence_summary["tools"] = evidence_summary["tools"][:4]
        compact["evidence_summary"] = _clean_empty_fields(evidence_summary)
    if item.get("failure_reason"):
        compact["failure_reason"] = _truncate_text(item.get("failure_reason"), text_limit)
    if item.get("preview"):
        compact["preview"] = _truncate_text(item.get("preview"), text_limit)
    destination = _compact_destination_for_table(item.get("destination"))
    if destination:
        compact["destination"] = destination
    if item.get("position") is not None:
        position = _safe_float_triplet(item.get("position"))
        compact["position"] = [round(value, 3) for value in position] if position else item.get("position")
    candidate_ids = [
        resolved
        for resolved in (_safe_int(raw) for raw in list(item.get("candidate_keyframe_ids") or []))
        if resolved is not None
    ]
    if candidate_ids:
        compact["candidate_keyframe_ids"] = candidate_ids[:6]
        if len(candidate_ids) > 6:
            compact["candidate_keyframe_ids_truncated_count"] = len(candidate_ids) - 6
    return compact


def _clean_empty_fields(item: dict[str, Any]) -> dict[str, Any]:
    """Drop empty fields from one LLM-facing memory row."""

    return {
        key: value
        for key, value in item.items()
        if value not in (None, "", [], {})
    }


def _semantic_grounding_for_task(task: dict[str, Any]) -> Optional[str]:
    """Return the current semantic grounding label for one task."""

    target = task.get("target") if isinstance(task.get("target"), dict) else {}
    target_type = str(target.get("type") or "").strip()
    if target_type in {"semantic_keyframe", "semantic_object", "keyframe", "position"}:
        return target_type
    outputs = {str(item) for item in (task.get("outputs") or [])}
    if "current_place_context" in outputs and "destination" in outputs:
        return "semantic_keyframe"
    if "destination" in outputs and task.get("inputs_from"):
        return "semantic_object"

    if legacy_staging_kind(task):
        return "semantic_keyframe"
    if legacy_object_kind(task):
        return "semantic_object"
    return None


def _final_target_for_task(task: dict[str, Any]) -> Optional[str]:
    """Return a compact final target label without exposing legacy field names."""

    target = task.get("target") if isinstance(task.get("target"), dict) else {}
    for key in ("object_description", "query"):
        if target.get(key):
            return _truncate_text(target.get(key), 160)
    if task.get("primary_target"):
        return _truncate_text(task.get("primary_target"), 160)
    return None


def _semantic_upstream_task_ids(task: dict[str, Any]) -> list[int]:
    """Return upstream ids derived from dependencies and task-output inputs."""

    ids: list[int] = []

    def add(raw: Any) -> None:
        resolved = _safe_int(raw)
        if resolved is not None and resolved not in ids:
            ids.append(resolved)

    for raw_id in task.get("depends_on") or []:
        add(raw_id)
    target = task.get("target") if isinstance(task.get("target"), dict) else {}
    if target.get("type") == "task_output":
        add(target.get("task_id"))
    inputs_from = task.get("inputs_from")
    if isinstance(inputs_from, dict):
        for value in inputs_from.values():
            match = re.search(r"task(\d+)\.", str(value or ""))
            if match:
                add(match.group(1))

    add(legacy_upstream_task_id(task))
    return ids


def _intent_label(*values: Any, limit: int = 90) -> Optional[str]:
    """Return one short task/row intent label."""

    for value in values:
        text = _truncate_text(value, limit)
        if text:
            return text
    return None


def _destination_summary(value: Any) -> Optional[str]:
    """Return a short stable destination summary string."""

    destination = _compact_destination_for_table(value)
    if not destination:
        return None
    if destination.get("type") == "keyframe":
        keyframe_id = destination.get("keyframe_id")
        position = destination.get("position")
        return f"keyframe={keyframe_id}, pos={position}" if position else f"keyframe={keyframe_id}"
    if destination.get("type") == "position":
        position = destination.get("position")
        yaw = destination.get("yaw")
        if yaw is not None:
            return f"pos={position}, yaw={yaw}"
        return f"pos={position}"
    return json.dumps(destination, ensure_ascii=False)


def _candidate_summary(candidate_ids: Any) -> Optional[dict[str, Any]]:
    """Return a compact candidate-keyframe summary."""

    ids = [
        resolved
        for resolved in (_safe_int(raw) for raw in list(candidate_ids or []))
        if resolved is not None
    ]
    if not ids:
        return None
    return {
        "count": len(ids),
        "ids": ids[:8],
        **({"truncated_count": len(ids) - 8} if len(ids) > 8 else {}),
    }


def _artifact_summary(artifact_paths: Any) -> Optional[dict[str, Any]]:
    """Return artifact count and the most useful path keys without full paths."""

    if not isinstance(artifact_paths, dict) or not artifact_paths:
        return None
    preferred = [
        key
        for key in (
            "summary_json",
            "approach_goal_json",
            "status_json",
            "debug_png",
            "output_dir",
        )
        if artifact_paths.get(key)
    ]
    keys = preferred or sorted(str(key) for key in artifact_paths.keys())[:5]
    return {"count": len(artifact_paths), "keys": keys}


def _evidence_summary(tools: Any) -> Optional[dict[str, Any]]:
    """Return compact evidence-source facts for summary-table rows."""

    if not isinstance(tools, list):
        return None
    tool_names = [
        str(tool.get("name") or "").strip()
        for tool in tools
        if isinstance(tool, dict) and str(tool.get("name") or "").strip()
    ]
    if not tool_names:
        return None
    flags = []
    if any("approach_object" in name for name in tool_names):
        flags.append("object_approach")
    if any("keyframe" in name for name in tool_names):
        flags.append("keyframe")
    if any(name in {"get_current_state", "analyse_on_current_image"} for name in tool_names):
        flags.append("current_view")
    if any("attached_image" in name for name in tool_names):
        flags.append("attached_image")
    if any("distance" in name for name in tool_names):
        flags.append("metric")
    status_counts: dict[str, int] = {}
    issue_tools: list[str] = []
    issue_reasons: list[str] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        status = str(tool.get("status") or "").strip().lower()
        if status:
            status_counts[status] = status_counts.get(status, 0) + 1
        if status in {"blocked", "failed", "error", "timeout"}:
            issue_name = str(tool.get("name") or "").strip()
            if issue_name:
                issue_tools.append(issue_name)
            reason = _truncate_text(tool.get("failure_reason") or tool.get("summary"), 120)
            if reason:
                issue_reasons.append(reason)
    return _clean_empty_fields(
        {
            "tool_count": len(tool_names),
            "flags": flags,
            "status_counts": status_counts,
            "issue_tools": issue_tools[:4],
            "issue_reasons": issue_reasons[:3],
        }
    )


def _object_approach_status(tools: Any) -> dict[str, Any]:
    """Return status facts for object-approach evidence in one task row."""

    if not isinstance(tools, list):
        return {}
    object_tools = [
        tool
        for tool in tools
        if isinstance(tool, dict) and "approach_object" in str(tool.get("name") or "")
    ]
    if not object_tools:
        return {}
    statuses = [
        str(tool.get("status") or "").strip().lower()
        for tool in object_tools
        if str(tool.get("status") or "").strip()
    ]
    issue_statuses = {"blocked", "failed", "error", "timeout"}
    return {
        "count": len(object_tools),
        "statuses": statuses,
        "has_destination": any(_extract_destination(tool) for tool in object_tools),
        "has_issue": any(status in issue_statuses for status in statuses),
        "reasons": [
            _truncate_text(tool.get("failure_reason") or tool.get("summary"), 160)
            for tool in object_tools
            if str(tool.get("status") or "").strip().lower() in issue_statuses
            and _truncate_text(tool.get("failure_reason") or tool.get("summary"), 160)
        ][:3],
    }


def _destination_source_summary(row: dict[str, Any]) -> Optional[str]:
    """Return a compact source label for a row destination."""

    destination = _compact_destination_for_table(row.get("destination"))
    if not destination:
        return None
    tools = row.get("tools") if isinstance(row.get("tools"), list) else []
    object_status = _object_approach_status(tools)
    if object_status:
        if object_status.get("has_destination"):
            return "object_tool_verified"
        if object_status.get("has_issue"):
            return "fallback_after_object_tool_issue"
    if _extract_first_destination_from_evidence(tools):
        return "tool_evidence"
    return "task_result"


def _task_result_kind(row: dict[str, Any]) -> str:
    """Return one stable kind label for task summary-table selection."""

    status = str(row.get("status") or "").strip().lower()
    tools = row.get("tools") if isinstance(row.get("tools"), list) else []
    evidence = _evidence_summary(tools) or {}
    flags = set(evidence.get("flags") or [])
    destination = _compact_destination_for_table(row.get("destination"))
    has_destination = destination is not None
    task_type = str(row.get("task_type") or "").strip()
    if status == "failed":
        return "failed_task"
    semantic_grounding = str(row.get("semantic_grounding") or "").strip()
    if semantic_grounding == "semantic_keyframe":
        return "staging_keyframe_resolution"
    if semantic_grounding == "semantic_object":
        return "semantic_object_destination"
    if "object_approach" in flags and has_destination:
        object_status = _object_approach_status(tools)
        if object_status.get("has_issue") and not object_status.get("has_destination"):
            return "object_fallback_destination"
        return "object_destination"
    if task_type == "navigation_action" and destination:
        if destination.get("type") == "keyframe":
            return "keyframe_navigation"
        return "position_navigation"
    if "keyframe" in flags and row.get("candidate_keyframe_ids"):
        return "keyframe_resolution"
    if "metric" in flags:
        return "metric_analysis"
    if has_destination:
        return "destination_resolution"
    return task_type or "task"


def _summary_table_row(scope: str, row: dict[str, Any]) -> dict[str, Any]:
    """Render one stable summary-table index row."""

    base = {
        "row_id": row.get("row_id"),
        "scope": scope,
        "time": row.get("created_at") or row.get("arrived_at") or row.get("updated_at"),
        "task_id": row.get("task_id"),
    }
    if scope in {"conversation", "plan"}:
        base["plan_id"] = row.get("plan_id")
    if scope == "conversation":
        return _clean_empty_fields(
            {
                **base,
                "source": row.get("source"),
                "intent_label": _intent_label(row.get("input", {}).get("text"), limit=120),
                "response_count": len(list(row.get("agent_responses") or [])),
            }
        )
    if scope == "plan":
        return _clean_empty_fields(
            {
                **base,
                "mode": row.get("mode"),
                "version": row.get("version"),
                "task_count": row.get("task_count"),
                "intent_label": _intent_label(row.get("user_goal"), limit=120),
            }
        )
    if scope == "task":
        destination = _compact_destination_for_table(row.get("destination"))
        evidence = _evidence_summary(row.get("tools")) or {}
        tool_issue_reasons = list(evidence.get("issue_reasons") or [])
        background_failure = row.get("background_failure_reason")
        failure_reason = (
            _failure_reason_from_task_row(row)
            or background_failure
            or (tool_issue_reasons[0] if tool_issue_reasons else None)
        )
        status_detail = row.get("status_detail")
        if not status_detail and row.get("status") == "waiting" and row.get("wait_for_event"):
            status_detail = f"waiting_for_{row.get('wait_for_event')}"
        return _clean_empty_fields(
            {
                **base,
                "task_type": row.get("task_type"),
                "status": row.get("status"),
                "status_detail": status_detail,
                "wait_for_event": row.get("wait_for_event"),
                "success": str(row.get("status") or "").strip().lower() == "completed",
                "result_kind": _task_result_kind(row),
                "semantic_grounding": row.get("semantic_grounding"),
                "final_target": _truncate_text(row.get("final_target"), 120),
                "selection_policy": row.get("selection_policy"),
                "depends_on": row.get("depends_on"),
                "upstream_task_ids": row.get("upstream_task_ids"),
                "outputs": row.get("outputs"),
                "has_destination": destination is not None,
                "destination_source": _destination_source_summary(row),
                "intent_label": _intent_label(row.get("description"), row.get("summary"), limit=90),
                "destination_summary": _destination_summary(row.get("destination")),
                "candidate_summary": _candidate_summary(row.get("candidate_keyframe_ids")),
                "artifact_summary": _artifact_summary(row.get("artifact_paths")),
                "evidence_summary": evidence,
                "background_status": row.get("background_status"),
                "background_summary": _truncate_text(row.get("background_summary"), 180),
                "failure_reason": _truncate_text(failure_reason, 180) if failure_reason else None,
            }
        )
    if scope == "navigation":
        return _clean_empty_fields(
            {
                **base,
                "order": row.get("order"),
                "intent_label": _intent_label(row.get("description"), limit=120),
                "related_task_row_id": row.get("related_task_row_id"),
                "destination_summary": _destination_summary(
                    {
                        "type": "keyframe" if row.get("keyframe_id") is not None else "position",
                        "keyframe_id": row.get("keyframe_id"),
                        "position": row.get("position"),
                    }
                ),
                "source": row.get("source"),
            }
        )
    return _clean_empty_fields(
        {
            **base,
            "intent_label": _intent_label(row.get("summary"), limit=120),
            "keyframe_id": row.get("keyframe_id"),
            "anchor_id": row.get("anchor_id"),
            "source": row.get("source"),
        }
    )


def _timeline_summary(scope: str, row: dict[str, Any]) -> str:
    """Render one stable timeline sentence from structured memory fields."""

    if scope == "conversation":
        text = row.get("input", {}).get("text")
        responses = len(list(row.get("agent_responses") or []))
        return f"Conversation {row.get('row_id')}: {_truncate_text(text, 140)}; responses={responses}."
    if scope == "plan":
        return (
            f"Plan {row.get('plan_id')} v{row.get('version')} "
            f"mode={row.get('mode')} tasks={row.get('task_count')}: "
            f"{_truncate_text(row.get('user_goal'), 140)}"
        )
    if scope == "task":
        status_text = str(row.get("status") or "")
        if row.get("status_detail"):
            status_text = f"{status_text}/{row.get('status_detail')}"
        parts = [
            f"Task {row.get('task_id')} {status_text}",
            f"type={row.get('task_type')}",
            _truncate_text(row.get("description") or row.get("summary"), 120),
        ]
        if row.get("wait_for_event"):
            parts.append(f"wait_for_event={row.get('wait_for_event')}")
        if row.get("semantic_grounding"):
            parts.append(f"grounding={row.get('semantic_grounding')}")
        if row.get("final_target"):
            parts.append(f"target={_truncate_text(row.get('final_target'), 80)}")
        if row.get("depends_on"):
            parts.append(f"depends_on={row.get('depends_on')}")
        destination = _destination_summary(row.get("destination"))
        if destination:
            parts.append(f"destination={destination}")
        candidates = _candidate_summary(row.get("candidate_keyframe_ids"))
        if candidates:
            parts.append(f"candidates={candidates.get('ids')}")
        artifacts = _artifact_summary(row.get("artifact_paths"))
        if artifacts:
            parts.append(f"artifacts={artifacts.get('keys')}")
        return "; ".join(part for part in parts if part)
    if scope == "navigation":
        return (
            f"Navigation {row.get('order')}: {row.get('description')} "
            f"-> {_destination_summary({'type': 'keyframe' if row.get('keyframe_id') is not None else 'position', 'keyframe_id': row.get('keyframe_id'), 'position': row.get('position')})}."
        )
    return f"Observation {row.get('row_id')}: {_truncate_text(row.get('summary'), 160)}"


def _apply_memory_budget(
    items: list[dict[str, Any]],
    *,
    view: str,
    budget_chars: int,
) -> tuple[list[dict[str, Any]], bool, int]:
    """Apply a hard JSON-character budget to LLM-facing memory output."""

    estimated = _json_char_count(items)
    if estimated <= budget_chars:
        return deepcopy(items), False, estimated

    if view == "summary_table":
        for text_limit in (90, 60, 36):
            compact_summary = [
                _summary_budget_item(item, text_limit=text_limit)
                for item in items
            ]
            if _json_char_count(compact_summary) <= budget_chars:
                return compact_summary, True, _json_char_count(compact_summary)
        compact_summary = [
            _clean_empty_fields(
                {
                    "row_id": item.get("row_id"),
                    "scope": item.get("scope"),
                    "time": item.get("time"),
                    "task_id": item.get("task_id"),
                    "status": item.get("status"),
                    "result_kind": item.get("result_kind"),
                    "success": item.get("success"),
                    "intent_label": _truncate_text(item.get("intent_label"), 32),
                    "destination_summary": _truncate_text(item.get("destination_summary"), 48),
                    "candidate_summary": item.get("candidate_summary"),
                    "failure_reason": _truncate_text(item.get("failure_reason"), 72),
                }
            )
            for item in items
        ]
        if _json_char_count(compact_summary) <= budget_chars:
            return compact_summary, True, _json_char_count(compact_summary)

    text_limit = 220 if view == "detail" else 160
    list_limit = 10 if view == "detail" else 6
    shrunk = [
        _shrink_for_memory(item, text_limit=text_limit, list_limit=list_limit)
        for item in items
    ]
    if _json_char_count(shrunk) <= budget_chars:
        return shrunk, True, _json_char_count(shrunk)

    selected: list[dict[str, Any]] = []
    for item in shrunk:
        candidate = [*selected, item]
        if _json_char_count(candidate) > budget_chars:
            break
        selected.append(item)
    if selected:
        return selected, True, _json_char_count(selected)

    fallback = [
        _shrink_for_memory(items[0], text_limit=100, list_limit=4)
    ] if items else []
    return fallback, True, _json_char_count(fallback)


def _state_excerpt(state: dict[str, Any]) -> dict[str, Any]:
    """Build one compact state excerpt without storing the full message history."""

    tasks = state.get("tasks", {}) if isinstance(state, dict) else {}
    events = list(state.get("events", [])) if isinstance(state, dict) else []
    background_results = (
        dict(state.get("background_results", {})) if isinstance(state, dict) else {}
    )

    task_items = []
    if isinstance(tasks, dict):
        for task_id in sorted(tasks):
            task = tasks[task_id]
            if isinstance(task, dict):
                task_items.append(_task_excerpt(task))

    background_excerpt = {}
    for task_id, result in background_results.items():
        if isinstance(result, dict):
            background_excerpt[str(task_id)] = {
                "status": result.get("status"),
                "summary": _truncate_text(result.get("summary", ""), 220),
                "latest_tool_name": result.get("latest_tool_name"),
                "candidate_keyframe_ids": list(
                    result.get("candidate_keyframe_ids", [])
                )[:8],
                "recommended_keyframe_id": result.get("recommended_keyframe_id"),
                "recommendation_confidence": result.get(
                    "recommendation_confidence"
                ),
            }
        else:
            background_excerpt[str(task_id)] = {"status": "completed", "summary": _truncate_text(result, 220)}

    return {
        "current_task_id": state.get("current_task_id"),
        "current_plan_id": state.get("current_plan_id"),
        "error_message": state.get("error_message"),
        "next_action": _to_jsonable(state.get("next_action")),
        "task_count": len(task_items),
        "tasks": task_items,
        "recent_events": _to_jsonable(events[-8:]),
        "background_results": background_excerpt,
        "turn_response_items": _to_jsonable(state.get("turn_response_items", [])),
        "turn_response_type": state.get("turn_response_type"),
        "turn_response_text": state.get("turn_response_text"),
        "user_facing_response": state.get("user_facing_response"),
    }


def _record_thread_matches(record: dict[str, Any], thread_id: Optional[str]) -> bool:
    """Return True when a memory record belongs to the restored thread scope."""

    if not thread_id:
        return True
    record_thread_id = str(record.get("thread_id") or "").strip()
    return not record_thread_id or record_thread_id == str(thread_id)


def _task_identity(plan_id: Any, task_id: Any) -> Optional[tuple[str, int]]:
    """Return the plan/task identity used when filtering restored memory."""

    resolved_task_id = _safe_int(task_id)
    if resolved_task_id is None:
        return None
    return (str(plan_id or "").strip(), resolved_task_id)


def _completed_task_identity_sets(
    data: dict[str, Any],
    *,
    thread_id: Optional[str],
) -> tuple[set[tuple[str, int]], set[int], set[str]]:
    """Collect task identities that represent completed task history."""

    identities: set[tuple[str, int]] = set()
    task_ids: set[int] = set()
    plan_ids: set[str] = set()
    for item in list(data.get("task_results") or []):
        if not isinstance(item, dict) or not _record_thread_matches(item, thread_id):
            continue
        status = str(item.get("status") or "").strip().lower()
        event_type = str(item.get("event_type") or "").strip()
        if status != "completed" and event_type != "task_completed":
            continue
        identity = _task_identity(item.get("plan_id"), item.get("task_id"))
        if identity is None:
            continue
        identities.add(identity)
        task_ids.add(identity[1])
        if identity[0]:
            plan_ids.add(identity[0])
    return identities, task_ids, plan_ids


def _filter_completed_turn_records(
    turns: list[Any],
    *,
    thread_id: Optional[str],
) -> list[dict[str, Any]]:
    """Keep only user/system turns that reached a completed result."""

    scoped_turns = [
        dict(item)
        for item in turns
        if isinstance(item, dict) and _record_thread_matches(item, thread_id)
    ]
    if not scoped_turns:
        return []

    start_indices = [
        index
        for index, turn in enumerate(scoped_turns)
        if str(turn.get("status") or "").strip() == "started"
    ]
    keep_indices: set[int] = set()
    for ordinal, start_index in enumerate(start_indices):
        next_start = (
            start_indices[ordinal + 1]
            if ordinal + 1 < len(start_indices)
            else len(scoped_turns)
        )
        start_thread_id = str(scoped_turns[start_index].get("thread_id") or "default")
        completed_index = None
        for candidate_index in range(start_index + 1, next_start):
            candidate = scoped_turns[candidate_index]
            if str(candidate.get("thread_id") or "default") != start_thread_id:
                continue
            if str(candidate.get("status") or "").strip() == "completed":
                completed_index = candidate_index
                break
        if completed_index is not None:
            keep_indices.add(start_index)
            keep_indices.add(completed_index)

    for index, turn in enumerate(scoped_turns):
        if str(turn.get("status") or "").strip() == "completed":
            keep_indices.add(index)

    return [scoped_turns[index] for index in sorted(keep_indices)]


RESTORE_FAILED_TASK_REASON = (
    "Session was restored with the active plan cleared; this task was not resumed."
)


def _latest_state_tasks_from_threads(
    data: dict[str, Any],
    *,
    thread_id: Optional[str],
) -> tuple[dict[tuple[str, int], dict[str, Any]], set[str]]:
    """Return the latest task excerpts and active plan ids from thread snapshots."""

    tasks_by_identity: dict[tuple[str, int], dict[str, Any]] = {}
    active_plan_ids: set[str] = set()
    threads = data.get("threads") if isinstance(data.get("threads"), dict) else {}
    for raw_thread_id, item in threads.items():
        if not isinstance(item, dict):
            continue
        if thread_id and str(raw_thread_id) != str(thread_id):
            continue
        state_excerpt = item.get("state_excerpt")
        if not isinstance(state_excerpt, dict):
            continue
        current_plan_id = str(state_excerpt.get("current_plan_id") or "").strip()
        if current_plan_id:
            active_plan_ids.add(current_plan_id)
        for task in list(state_excerpt.get("tasks") or []):
            if not isinstance(task, dict):
                continue
            identity = _task_identity(
                task.get("plan_id") or current_plan_id,
                task.get("task_id"),
            )
            if identity is not None:
                tasks_by_identity[identity] = dict(task)
    return tasks_by_identity, active_plan_ids


def _restore_failed_task_identities(
    data: dict[str, Any],
    *,
    thread_id: Optional[str],
    completed_task_identities: set[tuple[str, int]],
) -> tuple[set[tuple[str, int]], dict[tuple[str, int], dict[str, Any]]]:
    """Find active-plan tasks that should be preserved as failed on restore."""

    state_tasks, active_plan_ids = _latest_state_tasks_from_threads(
        data,
        thread_id=thread_id,
    )
    failed: set[tuple[str, int]] = set()
    for identity, task in state_tasks.items():
        status = str(task.get("status") or "").strip().lower()
        if identity in completed_task_identities or status == "completed":
            continue
        if identity[0] in active_plan_ids or status in {"pending", "in_progress", "running", "waiting"}:
            failed.add(identity)

    for plan in list(data.get("plans") or []):
        if not isinstance(plan, dict) or not _record_thread_matches(plan, thread_id):
            continue
        plan_id = str(plan.get("plan_id") or "").strip()
        if plan_id not in active_plan_ids:
            continue
        for task in list(plan.get("tasks") or []):
            if not isinstance(task, dict):
                continue
            identity = _task_identity(task.get("plan_id") or plan_id, task.get("task_id"))
            if identity is not None and identity not in completed_task_identities:
                failed.add(identity)
                state_tasks.setdefault(identity, {**task, "plan_id": plan_id})
    return failed, state_tasks


def _task_with_restore_failure(
    task: dict[str, Any],
    *,
    plan_id: str,
) -> dict[str, Any]:
    """Return a task excerpt marked as failed because restore cleared its plan."""

    failed_task = dict(task)
    failed_task.setdefault("plan_id", plan_id)
    failed_task["status"] = "failed"
    failed_task["terminal_reason"] = RESTORE_FAILED_TASK_REASON
    failed_task["failure_reason"] = RESTORE_FAILED_TASK_REASON
    return failed_task


def _filter_restored_plan_records(
    plans: list[Any],
    *,
    thread_id: Optional[str],
    completed_task_identities: set[tuple[str, int]],
    completed_plan_ids: set[str],
    restore_failed_task_identities: set[tuple[str, int]],
    state_tasks_by_identity: dict[tuple[str, int], dict[str, Any]],
) -> list[dict[str, Any]]:
    """Keep restored plan records and mark active unfinished tasks as failed."""

    restored_plans: list[dict[str, Any]] = []
    for item in plans:
        if not isinstance(item, dict) or not _record_thread_matches(item, thread_id):
            continue
        plan_id = str(item.get("plan_id") or "").strip()
        has_restored_failed_tasks = any(
            identity[0] == plan_id for identity in restore_failed_task_identities
        )
        if plan_id and plan_id not in completed_plan_ids and not has_restored_failed_tasks:
            continue
        restored_tasks: list[dict[str, Any]] = []
        for task in list(item.get("tasks") or []):
            if not isinstance(task, dict):
                continue
            identity = _task_identity(task.get("plan_id") or plan_id, task.get("task_id"))
            if identity in completed_task_identities:
                restored_task = dict(state_tasks_by_identity.get(identity, task))
                restored_task.setdefault("plan_id", plan_id)
                restored_task["status"] = "completed"
                restored_tasks.append(restored_task)
            elif identity in restore_failed_task_identities:
                restored_tasks.append(
                    _task_with_restore_failure(
                        state_tasks_by_identity.get(identity, task),
                        plan_id=plan_id,
                    )
                )
        if not restored_tasks:
            continue
        restored = dict(item)
        restored["tasks"] = restored_tasks
        restored["task_count"] = len(restored_tasks)
        first_task_ids = {
            _safe_int(task.get("task_id"))
            for task in restored_tasks
            if _safe_int(task.get("task_id")) is not None
        }
        if _safe_int(restored.get("first_task_id")) not in first_task_ids:
            restored["first_task_id"] = restored_tasks[0].get("task_id")
        restored_plans.append(restored)
    return restored_plans


def _task_scoped_record_is_completed(
    record: dict[str, Any],
    *,
    completed_task_identities: set[tuple[str, int]],
    completed_task_ids: set[int],
    restore_failed_task_identities: set[tuple[str, int]] | None = None,
    allow_taskless: bool = False,
) -> bool:
    """Return True when one restored record belongs to kept task history."""

    identity = _task_identity(record.get("plan_id"), record.get("task_id"))
    if identity in completed_task_identities:
        return True
    if restore_failed_task_identities and identity in restore_failed_task_identities:
        return True
    task_id = _safe_int(record.get("task_id"))
    record_plan_id = str(record.get("plan_id") or "").strip()
    if task_id is not None and not record_plan_id:
        return task_id in completed_task_ids
    return allow_taskless


def _restore_failed_task_result(
    task: dict[str, Any],
    *,
    thread_id: Optional[str],
    identity: tuple[str, int],
) -> dict[str, Any]:
    """Create a task-result memory row for a task failed by checkpoint restore."""

    failed_task = _task_with_restore_failure(task, plan_id=identity[0])
    return {
        "thread_id": thread_id,
        "task_id": identity[1],
        "description": failed_task.get("description"),
        "status": "failed",
        "status_detail": "failed_after_session_restore",
        "wait_for_event": failed_task.get("wait_for_event"),
        "task_type": failed_task.get("task_type"),
        "target": _to_jsonable(failed_task.get("target")),
        "type": failed_task.get("type"),
        "plan_id": identity[0] or failed_task.get("plan_id"),
        "user_input_id": failed_task.get("user_input_id"),
        "depends_on": _to_jsonable(failed_task.get("depends_on", [])),
        "inputs_from": _to_jsonable(failed_task.get("inputs_from")),
        "outputs": _to_jsonable(failed_task.get("outputs")),
        "selection_policy": failed_task.get("selection_policy"),
        "image_refs": _to_jsonable(failed_task.get("image_refs")),
        "result": [],
        "origin": "session_checkpoint_restore",
        "error": RESTORE_FAILED_TASK_REASON,
        "event_type": "task_failed",
        "summary": RESTORE_FAILED_TASK_REASON,
        "recorded_at": _utc_now_iso(),
    }


def _filter_restored_run_memory_data(
    data: dict[str, Any],
    *,
    thread_id: Optional[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return a run-memory snapshot with completed history and restored failures."""

    completed_identities, completed_task_ids, completed_plan_ids = (
        _completed_task_identity_sets(data, thread_id=thread_id)
    )
    restore_failed_identities, state_tasks_by_identity = _restore_failed_task_identities(
        data,
        thread_id=thread_id,
        completed_task_identities=completed_identities,
    )
    kept_task_identities = completed_identities | restore_failed_identities
    kept_task_ids = {
        identity[1]
        for identity in kept_task_identities
    }

    def scoped_records(bucket: str) -> list[dict[str, Any]]:
        return [
            dict(item)
            for item in list(data.get(bucket) or [])
            if isinstance(item, dict) and _record_thread_matches(item, thread_id)
        ]

    task_results = [
        item
        for item in scoped_records("task_results")
        if _task_scoped_record_is_completed(
            item,
            completed_task_identities=completed_identities,
            completed_task_ids=kept_task_ids,
            restore_failed_task_identities=restore_failed_identities,
        )
        and (
            str(item.get("status") or "").strip().lower() == "completed"
            or str(item.get("event_type") or "").strip() == "task_completed"
            or _task_identity(item.get("plan_id"), item.get("task_id")) in restore_failed_identities
        )
    ]
    existing_task_result_identities = {
        identity
        for identity in (
            _task_identity(item.get("plan_id"), item.get("task_id"))
            for item in task_results
        )
        if identity is not None
    }
    for identity in sorted(restore_failed_identities):
        if identity in existing_task_result_identities:
            continue
        task = state_tasks_by_identity.get(identity)
        if not isinstance(task, dict):
            continue
        task_results.append(
            _restore_failed_task_result(
                task,
                thread_id=thread_id,
                identity=identity,
            )
        )

    tool_traces = [
        item
        for item in scoped_records("tool_traces")
        if _task_scoped_record_is_completed(
            item,
            completed_task_identities=completed_identities,
            completed_task_ids=kept_task_ids,
            restore_failed_task_identities=restore_failed_identities,
        )
    ]
    background_updates = [
        item
        for item in scoped_records("background_updates")
        if _task_scoped_record_is_completed(
            {
                "task_id": item.get("task_id")
                or (item.get("record") if isinstance(item.get("record"), dict) else {}).get("task_id"),
                "plan_id": item.get("plan_id")
                or (item.get("record") if isinstance(item.get("record"), dict) else {}).get("plan_id"),
            },
            completed_task_identities=completed_identities,
            completed_task_ids=kept_task_ids,
            restore_failed_task_identities=restore_failed_identities,
        )
    ]
    observations = [
        item
        for item in scoped_records("observations")
        if _task_scoped_record_is_completed(
            item,
            completed_task_identities=completed_identities,
            completed_task_ids=kept_task_ids,
            restore_failed_task_identities=restore_failed_identities,
            allow_taskless=True,
        )
    ]
    events = []
    for item in scoped_records("events"):
        if _task_scoped_record_is_completed(
            item,
            completed_task_identities=completed_identities,
            completed_task_ids=kept_task_ids,
            restore_failed_task_identities=restore_failed_identities,
        ):
            events.append(item)
            continue
        plan_id = str(item.get("plan_id") or "").strip()
        if plan_id and (
            plan_id in completed_plan_ids
            or any(identity[0] == plan_id for identity in restore_failed_identities)
        ):
            events.append(item)
    responses = [
        item
        for item in scoped_records("responses")
        if _task_scoped_record_is_completed(
            item,
            completed_task_identities=completed_identities,
            completed_task_ids=kept_task_ids,
            restore_failed_task_identities=restore_failed_identities,
            allow_taskless=True,
        )
    ]

    restored = {
        "threads": {},
        "turns": _filter_completed_turn_records(
            list(data.get("turns") or []),
            thread_id=thread_id,
        ),
        "events": events,
        "plans": _filter_restored_plan_records(
            list(data.get("plans") or []),
            thread_id=thread_id,
            completed_task_identities=completed_identities,
            completed_plan_ids=completed_plan_ids,
            restore_failed_task_identities=restore_failed_identities,
            state_tasks_by_identity=state_tasks_by_identity,
        ),
        "task_results": task_results,
        "tool_traces": tool_traces,
        "background_updates": background_updates,
        "observations": observations,
        "stream_updates": [],
        "responses": responses,
        "checkpoints": scoped_records("checkpoints"),
    }
    summary = {
        "completed_task_count": len(completed_identities),
        "restore_failed_task_count": len(restore_failed_identities),
        "plan_count": len(restored["plans"]),
        "task_result_count": len(task_results),
        "navigation_source_count": len(events) + len(task_results),
    }
    return restored, summary


class AsyncAgentRunMemory:
    """Store full-session runtime records and expose structured query helpers."""

    def __init__(
        self,
        *,
        session_id: Optional[str] = None,
        session_dir: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        self._lock = threading.RLock()
        self.session_id = str(session_id or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self.session_dir = Path(session_dir) if session_dir else None
        self._latest_thread_id = "default"
        self.snapshot_path = (
            self.session_dir / "run_memory.json" if self.session_dir is not None else None
        )
        self._data: dict[str, Any] = {
            "session": {
                "session_id": self.session_id,
                "created_at": _utc_now_iso(),
                "session_dir": str(self.session_dir) if self.session_dir else None,
                "metadata": _to_jsonable(metadata or {}),
            },
            "threads": {},
            "turns": [],
            "events": [],
            "plans": [],
            "task_results": [],
            "tool_traces": [],
            "background_updates": [],
            "observations": [],
            "stream_updates": [],
            "responses": [],
            "checkpoints": [],
        }
        self._flush_locked()

    def _flush_locked(self) -> None:
        """Persist the current in-memory snapshot to disk when a session dir exists."""

        if self.snapshot_path is None:
            return

        self.snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.snapshot_path.with_name(
            f"{self.snapshot_path.stem}_{os.getpid()}.tmp"
        )
        temp_path.write_text(
            json.dumps(self._data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temp_path.replace(self.snapshot_path)
        self._export_memory_tables_locked()

    def _export_memory_tables_locked(self) -> None:
        """Persist the five simplified query tables for human review."""

        if self.session_dir is None:
            return
        write_memory_tables(
            self.session_dir,
            {
                "conversation": self._conversation_memory_rows_locked(),
                "plan": self._plan_memory_rows_locked(),
                "task": self._task_memory_rows_locked(),
                "navigation": self._navigation_memory_rows_locked(),
                "observation": self._observation_memory_rows_locked(),
            },
        )

    def _append_record(self, bucket: str, record: dict[str, Any]) -> None:
        """Append one record to the named bucket and persist the snapshot."""

        with self._lock:
            entry = _to_jsonable(record)
            if "recorded_at" not in entry:
                entry["recorded_at"] = _utc_now_iso()
            self._data.setdefault(bucket, []).append(entry)
            self._flush_locked()

    def update_session_metadata(self, **metadata: Any) -> None:
        """Merge additional session metadata into the top-level run snapshot."""

        with self._lock:
            existing = dict(self._data.get("session", {}).get("metadata", {}))
            existing.update(_to_jsonable(metadata))
            self._data["session"]["metadata"] = existing
            self._data["session"]["updated_at"] = _utc_now_iso()
            self._flush_locked()

    def restore_from_snapshot(
        self,
        snapshot_path: str | Path,
        *,
        thread_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Load a previous run-memory snapshot while keeping runtime state idle."""

        source_path = Path(snapshot_path).expanduser()
        raw_data = json.loads(source_path.read_text(encoding="utf-8"))
        if not isinstance(raw_data, dict):
            raise ValueError(f"Run-memory snapshot is not an object: {source_path}")

        restored_records, restore_summary = _filter_restored_run_memory_data(
            raw_data,
            thread_id=thread_id,
        )
        source_session = raw_data.get("session") if isinstance(raw_data.get("session"), dict) else {}

        with self._lock:
            current_session = deepcopy(self._data.get("session", {}))
            metadata = dict(current_session.get("metadata", {}))
            metadata["restored_run_memory"] = {
                "source_path": str(source_path),
                "source_session_id": source_session.get("session_id"),
                "source_created_at": source_session.get("created_at"),
                "restored_at": _utc_now_iso(),
                **restore_summary,
            }
            current_session["metadata"] = _to_jsonable(metadata)
            current_session["updated_at"] = _utc_now_iso()

            self._data = {
                "session": current_session,
                "threads": restored_records.get("threads", {}),
                "turns": restored_records.get("turns", []),
                "events": restored_records.get("events", []),
                "plans": restored_records.get("plans", []),
                "task_results": restored_records.get("task_results", []),
                "tool_traces": restored_records.get("tool_traces", []),
                "background_updates": restored_records.get("background_updates", []),
                "observations": restored_records.get("observations", []),
                "stream_updates": restored_records.get("stream_updates", []),
                "responses": restored_records.get("responses", []),
                "checkpoints": restored_records.get("checkpoints", []),
            }
            self._flush_locked()

        return {
            "source_path": str(source_path),
            "source_session_id": source_session.get("session_id"),
            **restore_summary,
        }

    def record_turn_start(self, *, thread_id: str, role: str, message: str) -> None:
        """Record the start of one new user or system turn."""

        self._latest_thread_id = str(thread_id or "default")
        self._append_record(
            "turns",
            {
                "thread_id": thread_id,
                "role": role,
                "message": message,
                "status": "started",
            },
        )

    def record_turn_result(self, turn_result: dict[str, Any]) -> None:
        """Record the final structured result for one completed turn."""

        thread_id = str(turn_result.get("thread_id") or "default")
        self._latest_thread_id = thread_id
        final_entry = {
            "thread_id": thread_id,
            "role": turn_result.get("role"),
            "message": turn_result.get("message"),
            "response_items": turn_result.get("response_items", []),
            "turn_response_type": turn_result.get("turn_response_type"),
            "turn_response_text": turn_result.get("turn_response_text"),
            "visited_nodes": turn_result.get("visited_nodes", []),
            "step_trace": turn_result.get("step_trace", []),
            "saw_plan_node": turn_result.get("saw_plan_node"),
            "saw_navigation_activity": turn_result.get("saw_navigation_activity"),
            "status": "completed",
        }
        self._append_record("turns", final_entry)
        self.sync_thread_state(thread_id=thread_id, state=turn_result.get("state", {}))
        response_items = list(turn_result.get("response_items", []) or [])
        if response_items:
            for item in response_items:
                self._record_turn_response_item(
                    thread_id=thread_id,
                    role=str(turn_result.get("role") or "unknown"),
                    item=item,
                )
            return

        response_text = str(turn_result.get("turn_response_text") or "").strip()
        if response_text:
            self.record_response(
                thread_id=thread_id,
                response=response_text,
                role=str(turn_result.get("role") or "unknown"),
            )

    def sync_thread_state(self, *, thread_id: str, state: dict[str, Any]) -> None:
        """Store the latest compact state snapshot for one thread."""

        self._latest_thread_id = str(thread_id or "default")
        with self._lock:
            self._data.setdefault("threads", {})
            self._data["threads"][thread_id] = {
                "thread_id": thread_id,
                "updated_at": _utc_now_iso(),
                "state_excerpt": _state_excerpt(state or {}),
            }
            self._flush_locked()

    def record_stream_update(
        self,
        *,
        thread_id: str,
        node_name: str,
        step_summary: dict[str, Any],
    ) -> None:
        """Record one streamed node update emitted during a turn."""

        self._append_record(
            "stream_updates",
            {
                "thread_id": thread_id,
                "node_name": node_name,
                "step_summary": step_summary,
            },
        )
        for item in list((step_summary or {}).get("turn_response_items", []) or []):
            if isinstance(item, dict):
                self._record_turn_response_item(
                    thread_id=thread_id,
                    role="system",
                    item=item,
                )

    def record_event(
        self,
        event: dict[str, Any],
        *,
        thread_id: Optional[str] = None,
        stage: Optional[str] = None,
    ) -> None:
        """Record one structured workflow event."""

        payload = dict(event or {})
        if thread_id:
            payload["thread_id"] = thread_id
        if stage:
            payload["stage"] = stage
        self._append_record("events", payload)

    def record_user_input(
        self,
        user_input: dict[str, Any],
        *,
        thread_id: Optional[str] = None,
    ) -> None:
        """Record one normalized user-input item as a checkpoint entry."""

        record = dict(user_input or {})
        if thread_id:
            record["thread_id"] = thread_id
        self._append_record("checkpoints", {"kind": "user_input", "data": record})

    def record_plan(
        self,
        *,
        plan_id: str,
        plan_mode: str,
        user_input_id: Optional[str],
        tasks: dict[int, dict[str, Any]],
        plan_text: Optional[str] = None,
        first_task_id: Optional[int] = None,
        thread_id: Optional[str] = None,
    ) -> None:
        """Record one plan snapshot and its compact task layout."""

        task_items = []
        for task_id in sorted(tasks):
            task = tasks[task_id]
            if isinstance(task, dict):
                task_items.append(_task_excerpt(task))
        self._append_record(
            "plans",
            {
                "thread_id": thread_id,
                "plan_id": plan_id,
                "plan_mode": plan_mode,
                "user_input_id": user_input_id,
                "first_task_id": first_task_id,
                "task_count": len(task_items),
                "tasks": task_items,
                "plan_text": _truncate_text(plan_text, 1200) if plan_text else None,
            },
        )

    def record_task_result(
        self,
        *,
        task: dict[str, Any],
        event_type: str,
        summary: str,
        tool_trace: Optional[dict[str, Any]] = None,
        thread_id: Optional[str] = None,
    ) -> None:
        """Record one task outcome with optional tool-trace evidence."""

        status_detail = None
        if task.get("status") == "waiting" and task.get("wait_for_event"):
            status_detail = f"waiting_for_{task.get('wait_for_event')}"
        entry = {
            "thread_id": thread_id,
            "task_id": task.get("task_id"),
            "description": task.get("description"),
            "status": task.get("status"),
            "status_detail": status_detail,
            "wait_for_event": task.get("wait_for_event"),
            "task_type": task.get("task_type"),
            "target": _to_jsonable(task.get("target")),
            "type": task.get("type"),
            "plan_id": task.get("plan_id"),
            "user_input_id": task.get("user_input_id"),
            "depends_on": _to_jsonable(task.get("depends_on", [])),
            "inputs_from": _to_jsonable(task.get("inputs_from")),
            "outputs": _to_jsonable(task.get("outputs")),
            "selection_policy": task.get("selection_policy"),
            "image_refs": _to_jsonable(task.get("image_refs")),
            "result": _compact_task_result_value(task.get("result", [])),
            "origin": task.get("origin"),
            "error": task.get("error"),
            "event_type": event_type,
            "summary": summary,
        }
        if tool_trace:
            tool_evidence = _compact_tool_evidence(tool_trace)
            entry["tool_evidence"] = tool_evidence
            entry["tool_trace_excerpt"] = {
                "tools": tool_evidence,
                "final_ai_content": _truncate_text(tool_trace.get("final_ai_content", ""), 500),
            }
        self._append_record("task_results", entry)
        if tool_trace:
            observation_summary = _observation_summary_from_trace(
                summary=summary,
                tool_trace=tool_trace,
            )
            if observation_summary:
                position, details = _observation_location_from_trace(tool_trace)
                self.record_observation(
                    summary=observation_summary,
                    task_id=_safe_int(task.get("task_id")),
                    plan_id=task.get("plan_id"),
                    thread_id=thread_id,
                    position=position,
                    source="tool_trace",
                    details={
                        **details,
                        "task_description": task.get("description"),
                        "task_type": task.get("task_type"),
                    },
                )

    def record_observation(
        self,
        *,
        summary: str,
        task_id: Optional[int] = None,
        plan_id: Optional[str] = None,
        thread_id: Optional[str] = None,
        keyframe_id: Optional[int] = None,
        position: Optional[list[float]] = None,
        anchor_id: Optional[str] = None,
        source: str = "task_result",
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        """Record one narrow observation memory item."""

        self._append_record(
            "observations",
            {
                "thread_id": thread_id,
                "plan_id": plan_id,
                "task_id": task_id,
                "summary": summary,
                "keyframe_id": keyframe_id,
                "position": _safe_float_triplet(position),
                "anchor_id": anchor_id,
                "source": source,
                "details": _to_jsonable(details or {}),
            },
        )

    def record_tool_trace(
        self,
        *,
        task: dict[str, Any],
        tool_trace: dict[str, Any],
        thread_id: Optional[str] = None,
    ) -> None:
        """Record the full tool trace for one executed task."""

        self._append_record(
            "tool_traces",
            {
                "thread_id": thread_id,
                "task_id": task.get("task_id"),
                "description": task.get("description"),
                "plan_id": task.get("plan_id"),
                "user_input_id": task.get("user_input_id"),
                "tool_trace": tool_trace,
            },
        )

    def record_background_update(
        self,
        *,
        task_id: int,
        task_description: str,
        record: dict[str, Any],
        worker_name: Optional[str] = None,
        thread_id: Optional[str] = None,
    ) -> None:
        """Record one background-analysis lifecycle update."""

        self._append_record(
            "background_updates",
            {
                "thread_id": thread_id,
                "worker_name": worker_name,
                "task_id": task_id,
                "task_description": task_description,
                "record": record,
            },
        )

    def record_response(
        self,
        *,
        thread_id: str,
        response: str,
        role: str = "assistant",
        response_type: str = "result",
        response_id: Optional[str] = None,
        task_id: Optional[int] = None,
        source_event_type: Optional[str] = None,
    ) -> None:
        """Record one final user-facing response."""

        self._append_record(
            "responses",
            {
                "thread_id": thread_id,
                "role": role,
                "response": response,
                "response_type": response_type,
                "response_id": response_id,
                "task_id": task_id,
                "source_event_type": source_event_type,
            },
        )

    def _record_turn_response_item(
        self,
        *,
        thread_id: str,
        role: str,
        item: dict[str, Any],
    ) -> None:
        """Persist one streamed/final turn response item, deduped by response id."""

        response_text = str(item.get("response_text") or "").strip()
        if not response_text:
            return

        response_id = str(item.get("response_id") or "").strip() or None
        if response_id:
            with self._lock:
                for response in self._data.get("responses", []):
                    if response.get("response_id") == response_id:
                        return

        self.record_response(
            thread_id=thread_id,
            response=response_text,
            role=role,
            response_type=str(item.get("response_type") or "result"),
            response_id=response_id,
            task_id=item.get("task_id"),
            source_event_type=str(item.get("source_event_type") or "").strip() or None,
        )

    def snapshot(self) -> dict[str, Any]:
        """Return one deep copy of the full in-memory snapshot."""

        with self._lock:
            return deepcopy(self._data)

    def _route_item_from_task_result(self, item: dict[str, Any]) -> Optional[dict[str, Any]]:
        """Build one confirmed route item from a completed navigation task result."""

        if str(item.get("event_type") or "").strip() != "task_completed":
            return None
        description = str(item.get("description") or "").strip()
        if not description:
            return None
        if str(item.get("task_type") or "").strip() != "navigation_action":
            return None

        enriched_item = self._attach_tool_trace_excerpt_to_task_item_locked(item)
        evidence = list(enriched_item.get("tool_evidence") or [])
        keyframe_id = None
        position = None
        for tool in evidence:
            if not isinstance(tool, dict):
                continue
            if str(tool.get("name") or "").strip() != "go_to_keyframe":
                continue
            metrics = tool.get("key_metrics") if isinstance(tool.get("key_metrics"), dict) else {}
            destination = tool.get("destination") if isinstance(tool.get("destination"), dict) else {}
            keyframe_id = _safe_int(
                metrics.get("target_keyframe_id") or destination.get("keyframe_id")
            )
            position = _safe_float_triplet(
                metrics.get("target_position") or destination.get("position")
            )
            break
        if keyframe_id is None and position is None:
            tool_data = _extract_tool_result_json(enriched_item, "go_to_keyframe")
            data = tool_data.get("data") if isinstance(tool_data, dict) else None
            if not isinstance(data, dict):
                data = {}
            keyframe_id = _safe_int(data.get("target_keyframe_id"))
            position = _safe_float_triplet(data.get("target_position"))
        if keyframe_id is None and position is None:
            return None

        return {
            "plan_id": item.get("plan_id"),
            "task_id": item.get("task_id"),
            "description": description,
            "destination_keyframe_id": keyframe_id,
            "destination_position": position,
            "arrived_at": item.get("recorded_at"),
            "source_event_id": None,
            "source": "task_result",
        }

    def _route_item_from_navigation_arrival_event(
        self,
        event: dict[str, Any],
    ) -> Optional[dict[str, Any]]:
        """Build one confirmed route item from a navigation_arrived event."""

        if str(event.get("type") or event.get("event_family") or "").strip() != "navigation_arrived":
            return None
        payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
        description = str(
            payload.get("destination_description")
            or payload.get("description")
            or payload.get("summary")
            or ""
        ).strip()
        if not description:
            return None

        keyframe_id = _safe_int(payload.get("destination_keyframe_id"))
        position = _safe_float_triplet(payload.get("destination_position"))
        if keyframe_id is None and position is None:
            return None

        return {
            "plan_id": event.get("plan_id"),
            "task_id": event.get("task_id"),
            "description": description,
            "destination_keyframe_id": keyframe_id,
            "destination_position": position,
            "arrived_at": event.get("recorded_at") or event.get("created_at"),
            "source_event_id": event.get("event_id"),
            "source": "event",
        }

    def _confirmed_route_items_locked(self) -> list[dict[str, Any]]:
        """Return deduplicated confirmed route arrivals in chronological order."""

        raw_items: list[dict[str, Any]] = []
        for event in self._data.get("events", []):
            route_item = self._route_item_from_navigation_arrival_event(event)
            if route_item is not None:
                raw_items.append(route_item)
        for item in self._data.get("task_results", []):
            route_item = self._route_item_from_task_result(item)
            if route_item is not None:
                raw_items.append(route_item)

        raw_items = sorted(raw_items, key=lambda item: str(item.get("arrived_at") or ""))
        deduped_by_identity: dict[tuple[Any, Any, Any, str], dict[str, Any]] = {}
        for item in raw_items:
            identity = (
                item.get("plan_id"),
                item.get("task_id"),
                item.get("destination_keyframe_id"),
                json.dumps(item.get("destination_position"), ensure_ascii=False),
            )
            existing = deduped_by_identity.get(identity)
            if existing is None:
                deduped_by_identity[identity] = dict(item)
                continue
            if item.get("source") == "event":
                existing.update({key: value for key, value in item.items() if value is not None})

        route_items = sorted(
            deduped_by_identity.values(),
            key=lambda item: str(item.get("arrived_at") or ""),
        )
        for index, item in enumerate(route_items, start=1):
            item["order"] = index
        return route_items

    def _attach_tool_trace_excerpt_to_task_item_locked(
        self,
        item: dict[str, Any],
    ) -> dict[str, Any]:
        """Attach the latest tool trace excerpt for matching task records."""

        enriched = dict(item)
        task_id = _safe_int(item.get("task_id"))
        if task_id is None:
            return enriched
        plan_id = str(item.get("plan_id") or "").strip()
        description = str(item.get("description") or "").strip()
        for trace_item in reversed(self._data.get("tool_traces", [])):
            if _safe_int(trace_item.get("task_id")) != task_id:
                continue
            if plan_id and str(trace_item.get("plan_id") or "").strip() != plan_id:
                continue
            if (
                description
                and str(trace_item.get("description") or "").strip()
                and str(trace_item.get("description") or "").strip() != description
            ):
                continue
            trace_payload = trace_item.get("tool_trace")
            if isinstance(trace_payload, dict):
                tool_evidence = _compact_tool_evidence(trace_payload)
                enriched["tool_evidence"] = tool_evidence
                enriched["tool_trace_excerpt"] = {
                    "tools": tool_evidence,
                    "final_ai_content": _truncate_text(
                        trace_payload.get("final_ai_content"),
                        500,
                    ),
                }
            break
        return enriched

    def _conversation_memory_rows_locked(self) -> list[dict[str, Any]]:
        """Return conversation memory rows grouped by user/system turn."""

        turns = list(self._data.get("turns", []))
        responses = list(self._data.get("responses", []))
        rows: list[dict[str, Any]] = []
        start_indices = [
            index
            for index, turn in enumerate(turns)
            if str(turn.get("status") or "").strip() == "started"
        ]
        if not start_indices:
            start_indices = list(range(len(turns)))

        for ordinal, turn_index in enumerate(start_indices, start=1):
            turn = turns[turn_index]
            next_start_index = (
                start_indices[ordinal] if ordinal < len(start_indices) else len(turns)
            )
            thread_id = str(turn.get("thread_id") or "default")
            started_at = _record_time(turn)
            next_started_at = (
                _record_time(turns[next_start_index])
                if next_start_index < len(turns)
                else ""
            )
            completed_turn = None
            for candidate in turns[turn_index + 1 : next_start_index]:
                if (
                    str(candidate.get("thread_id") or "default") == thread_id
                    and str(candidate.get("status") or "").strip() == "completed"
                ):
                    completed_turn = candidate
                    break

            assigned_responses = []
            for response in responses:
                if str(response.get("thread_id") or "default") != thread_id:
                    continue
                response_time = _record_time(response)
                if started_at and response_time and response_time < started_at:
                    continue
                if next_started_at and response_time and response_time >= next_started_at:
                    continue
                assigned_responses.append(
                    {
                        "type": response.get("response_type") or "result",
                        "text": response.get("response") or "",
                        "trigger": response.get("source_event_type"),
                        "plan_id": response.get("plan_id"),
                        "task_id": response.get("task_id"),
                        "created_at": response.get("recorded_at"),
                    }
                )

            role = str(turn.get("role") or "system").strip() or "system"
            rows.append(
                {
                    "row_id": f"conversation_{ordinal}",
                    "turn_id": f"turn_{ordinal}",
                    "thread_id": thread_id,
                    "source": "user" if role == "user" else "system",
                    "input": {
                        "type": "user_message" if role == "user" else "system_event",
                        "text": turn.get("message") or "",
                        "created_at": turn.get("recorded_at"),
                    },
                    "agent_responses": assigned_responses,
                    "status": (completed_turn or turn).get("status") or "started",
                    "created_at": turn.get("recorded_at"),
                    "updated_at": (completed_turn or turn).get("recorded_at"),
                }
            )
        return rows

    def _plan_memory_rows_locked(self) -> list[dict[str, Any]]:
        """Return simplified plan-version memory rows."""

        rows: list[dict[str, Any]] = []
        versions_by_plan: dict[str, int] = {}
        for index, plan in enumerate(self._data.get("plans", []), start=1):
            plan_id = str(plan.get("plan_id") or "")
            versions_by_plan[plan_id] = versions_by_plan.get(plan_id, 0) + 1
            tasks = []
            for task in list(plan.get("tasks") or []):
                if not isinstance(task, dict):
                    continue
                tasks.append(
                    {
                        "task_id": task.get("task_id"),
                        "description": task.get("description"),
                        "task_type": task.get("task_type") or task.get("type"),
                        "target": _to_jsonable(task.get("target")),
                        "status": task.get("status"),
                        "terminal_reason": task.get("terminal_reason"),
                        "failure_reason": task.get("failure_reason"),
                        "wait_for_event": task.get("wait_for_event"),
                        "depends_on": _to_jsonable(task.get("depends_on", [])),
                        "next_task_id": task.get("next_task_id"),
                        "branches": _to_jsonable(task.get("branches")),
                    }
                )
            rows.append(
                {
                    "row_id": f"plan_{index}",
                    "plan_id": plan.get("plan_id"),
                    "version": versions_by_plan[plan_id],
                    "mode": plan.get("plan_mode"),
                    "user_turn_id": plan.get("user_input_id"),
                    "user_goal": plan.get("plan_text"),
                    "entry_task_id": plan.get("first_task_id"),
                    "task_count": plan.get("task_count"),
                    "tasks": tasks,
                    "thread_id": plan.get("thread_id"),
                    "created_at": plan.get("recorded_at"),
                }
            )
        return rows

    def _latest_background_record_for_task_locked(
        self,
        task_id: Any,
        description: Any,
    ) -> Optional[dict[str, Any]]:
        """Return the newest compact background record matching one task row."""

        resolved_task_id = _safe_int(task_id)
        description_text = str(description or "").strip()
        matches: list[dict[str, Any]] = []
        for item in self._data.get("background_updates", []):
            if not isinstance(item, dict):
                continue
            record = item.get("record")
            if not isinstance(record, dict):
                continue
            if resolved_task_id is not None and _safe_int(record.get("task_id")) != resolved_task_id:
                continue
            record_description = str(
                record.get("task_description")
                or item.get("task_description")
                or ""
            ).strip()
            if description_text and record_description and record_description != description_text:
                continue
            merged = dict(record)
            merged.setdefault("recorded_at", item.get("recorded_at"))
            merged.setdefault("worker_name", item.get("worker_name"))
            matches.append(merged)
        if not matches:
            return None
        matches.sort(key=_record_time)
        return matches[-1]

    def _task_memory_rows_locked(self) -> list[dict[str, Any]]:
        """Return simplified task execution memory rows."""

        rows: list[dict[str, Any]] = []
        for index, item in enumerate(self._data.get("task_results", []), start=1):
            if not isinstance(item, dict):
                continue
            enriched = self._attach_tool_trace_excerpt_to_task_item_locked(item)
            tool_trace = enriched.get("tool_trace_excerpt")
            result = enriched.get("result")
            if isinstance(result, list) and len(result) == 1:
                result = result[0]
            compact_result = _compact_task_result_value(result)
            tool_evidence = list(enriched.get("tool_evidence") or _extract_tool_summary(tool_trace))
            destination = _extract_destination(compact_result) or _extract_first_destination_from_evidence(tool_evidence)
            artifact_paths = _merge_artifact_paths(compact_result)
            artifact_paths.update(_extract_artifacts_from_evidence(tool_evidence))
            candidate_keyframe_ids = _extract_candidate_keyframe_ids(compact_result)
            if not candidate_keyframe_ids:
                candidate_keyframe_ids = _extract_candidates_from_evidence(tool_evidence)
            background_record = self._latest_background_record_for_task_locked(
                enriched.get("task_id"),
                enriched.get("description"),
            )
            rows.append(
                {
                    "row_id": f"task_{index}",
                    "task_id": enriched.get("task_id"),
                    "plan_id": enriched.get("plan_id"),
                    "turn_id": enriched.get("thread_id"),
                    "thread_id": enriched.get("thread_id"),
                    "task_type": enriched.get("task_type") or enriched.get("type"),
                    "description": enriched.get("description"),
                    "status": enriched.get("status"),
                    "status_detail": enriched.get("status_detail"),
                    "wait_for_event": enriched.get("wait_for_event"),
                    "depends_on": _to_jsonable(enriched.get("depends_on", [])),
                    "inputs_from": _to_jsonable(enriched.get("inputs_from")),
                    "outputs": _to_jsonable(enriched.get("outputs")),
                    "semantic_grounding": _semantic_grounding_for_task(enriched),
                    "final_target": _final_target_for_task(enriched),
                    "upstream_task_ids": _semantic_upstream_task_ids(enriched),
                    "selection_policy": enriched.get("selection_policy"),
                    "image_refs": _to_jsonable(enriched.get("image_refs")),
                    "event_type": enriched.get("event_type"),
                    "summary": enriched.get("summary"),
                    "result": compact_result,
                    "destination": destination,
                    "candidate_keyframe_ids": candidate_keyframe_ids,
                    "artifact_paths": artifact_paths,
                    "tools": tool_evidence,
                    "background_status": (
                        background_record.get("status") if background_record else None
                    ),
                    "background_summary": (
                        background_record.get("summary") if background_record else None
                    ),
                    "background_failure_reason": (
                        background_record.get("failure_reason") if background_record else None
                    ),
                    "background_candidate_keyframe_ids": (
                        background_record.get("candidate_keyframe_ids")
                        if background_record
                        else None
                    ),
                    "origin": enriched.get("origin"),
                    "error": enriched.get("error"),
                    "failure_reason": _failure_reason_from_task_row(
                        {
                            "status": enriched.get("status"),
                            "summary": enriched.get("summary"),
                            "result": compact_result,
                            "error": enriched.get("error"),
                        }
                    )
                    if str(enriched.get("status") or "").strip().lower() == "failed"
                    else None,
                    "created_at": enriched.get("recorded_at"),
                }
            )
        return rows

    def _navigation_memory_rows_locked(self) -> list[dict[str, Any]]:
        """Return route/anchor memory rows derived from confirmed arrivals."""

        rows: list[dict[str, Any]] = []
        task_row_by_identity: dict[tuple[str, Optional[int]], str] = {}
        for task_row in self._task_memory_rows_locked():
            task_row_id = str(task_row.get("row_id") or "").strip()
            if not task_row_id:
                continue
            identity = (
                str(task_row.get("plan_id") or "").strip(),
                _safe_int(task_row.get("task_id")),
            )
            task_row_by_identity.setdefault(identity, task_row_id)
        for index, item in enumerate(self._confirmed_route_items_locked(), start=1):
            position = _safe_float_triplet(item.get("destination_position"))
            arrived_at_text = item.get("arrived_at")
            task_identity = (
                str(item.get("plan_id") or "").strip(),
                _safe_int(item.get("task_id")),
            )
            rows.append(
                {
                    "row_id": f"navigation_{index}",
                    "order": item.get("order") or index,
                    "anchor_id": f"anchor_{index}",
                    "label": item.get("description"),
                    "description": item.get("description"),
                    "plan_id": item.get("plan_id"),
                    "task_id": item.get("task_id"),
                    "related_task_row_id": task_row_by_identity.get(task_identity),
                    "keyframe_id": item.get("destination_keyframe_id"),
                    "position": position or item.get("destination_position"),
                    "arrived_at": arrived_at_text,
                    "source": item.get("source"),
                    "source_event_id": item.get("source_event_id"),
                    "created_at": arrived_at_text,
                }
            )
        return rows

    def _observation_memory_rows_locked(self) -> list[dict[str, Any]]:
        """Return narrow observation memory rows."""

        rows: list[dict[str, Any]] = []
        for index, item in enumerate(self._data.get("observations", []), start=1):
            if not isinstance(item, dict):
                continue
            rows.append(
                {
                    "row_id": f"observation_{index}",
                    "summary": item.get("summary"),
                    "plan_id": item.get("plan_id"),
                    "task_id": item.get("task_id"),
                    "turn_id": item.get("thread_id"),
                    "thread_id": item.get("thread_id"),
                    "keyframe_id": item.get("keyframe_id"),
                    "position": item.get("position"),
                    "anchor_id": item.get("anchor_id"),
                    "source": item.get("source"),
                    "details": _to_jsonable(item.get("details") or {}),
                    "created_at": item.get("recorded_at"),
                }
            )
        return rows

    def _memory_scope_rows_locked(self, scope: str) -> list[dict[str, Any]]:
        """Return raw simplified rows for one memory scope."""

        if scope == "conversation":
            return self._conversation_memory_rows_locked()
        if scope == "plan":
            return self._plan_memory_rows_locked()
        if scope == "task":
            return self._task_memory_rows_locked()
        if scope == "navigation":
            return self._navigation_memory_rows_locked()
        if scope == "observation":
            return self._observation_memory_rows_locked()
        return []

    def _filter_memory_rows(
        self,
        rows: list[dict[str, Any]],
        *,
        time: str,
        plan_id: str,
        turn_id: str,
        task_id: int,
        row_id: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        """Apply the intentionally small query_memory filter set."""

        selected = list(rows)
        if row_id:
            selected = [
                row for row in selected if str(row.get("row_id") or "") == row_id
            ]
            selected = sorted(selected, key=_record_time)
            return selected[:limit]
        if plan_id:
            selected = [row for row in selected if str(row.get("plan_id") or "") == plan_id]
        if turn_id:
            selected = [
                row
                for row in selected
                if str(row.get("turn_id") or "") == turn_id
                or str(row.get("thread_id") or "") == turn_id
            ]
        if task_id >= 0:
            selected = [
                row for row in selected if _safe_int(row.get("task_id")) == int(task_id)
            ]
        selected = sorted(selected, key=_record_time)
        if time == "recent":
            return selected[-limit:]
        return selected[:limit]

    def _render_memory_rows(
        self,
        *,
        scope: str,
        view: str,
        rows: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Render simplified memory rows into one of the supported views."""

        if view == "detail":
            return deepcopy(rows)
        if view == "timeline":
            rendered = []
            for row in rows:
                when = row.get("created_at") or row.get("arrived_at") or row.get("updated_at")
                rendered.append(
                    {
                        "row_id": row.get("row_id"),
                        "time": when,
                        "scope": scope,
                        "summary": _timeline_summary(scope, row),
                        "plan_id": row.get("plan_id"),
                        "task_id": row.get("task_id"),
                        "related_task_row_id": row.get("related_task_row_id"),
                    }
                )
            return rendered

        return [_summary_table_row(scope, row) for row in rows]

    def query_memory(
        self,
        *,
        scope: str = "all",
        view: str = "summary_table",
        query: str = "",
        time: str = "all",
        plan_id: str = "",
        turn_id: str = "",
        task_id: int = -1,
        row_id: str = "",
        limit: int = 50,
    ) -> dict[str, Any]:
        """Query simplified conversation/plan/task/navigation/observation memory."""

        warnings: list[str] = []
        valid_scopes = {"conversation", "plan", "task", "navigation", "observation", "all"}
        normalized_scope = str(scope or "all").strip().lower()
        if normalized_scope not in valid_scopes:
            warnings.append(f"Invalid scope {scope!r}; normalized to 'all'.")
            normalized_scope = "all"
        valid_views = {"summary_table", "timeline", "detail"}
        normalized_view = str(view or "summary_table").strip().lower()
        if normalized_view not in valid_views:
            warnings.append(f"Invalid view {view!r}; normalized to 'summary_table'.")
            normalized_view = "summary_table"
        normalized_time = str(time or "recent").strip().lower()
        if normalized_time not in {"recent", "all"}:
            warnings.append(f"Invalid time {time!r}; normalized to 'all'.")
            normalized_time = "all"
        resolved_limit, limit_warning = _coerce_positive_int(
            limit,
            default=50,
            minimum=1,
            maximum=50,
        )
        if limit_warning:
            warnings.append(limit_warning)
        try:
            resolved_task_id = int(task_id) if task_id is not None else -1
        except Exception:
            warnings.append(f"Invalid task_id {task_id!r}; normalized to -1.")
            resolved_task_id = -1

        normalized_query = str(query or "").strip()
        normalized_plan_id = str(plan_id or "").strip()
        normalized_turn_id = str(turn_id or "").strip()
        normalized_row_id = str(row_id or "").strip()
        scopes = (
            ["conversation", "plan", "task", "navigation", "observation"]
            if normalized_scope == "all"
            else [normalized_scope]
        )

        with self._lock:
            rendered_items: list[dict[str, Any]] = []
            raw_count = 0
            scoped_row_count = 0
            for scoped_name in scopes:
                rows = self._memory_scope_rows_locked(scoped_name)
                scoped_row_count += len(rows)
                selected = self._filter_memory_rows(
                    rows,
                    time=normalized_time,
                    plan_id=normalized_plan_id,
                    turn_id=normalized_turn_id,
                    task_id=resolved_task_id,
                    row_id=normalized_row_id,
                    limit=resolved_limit,
                )
                raw_count += len(selected)
                rendered = self._render_memory_rows(
                    scope=scoped_name,
                    view=normalized_view,
                    rows=selected,
                )
                if normalized_scope == "all":
                    for item in rendered:
                        item.setdefault("scope", scoped_name)
                rendered_items.extend(rendered)
        if scoped_row_count > 0 and raw_count == 0:
            if normalized_row_id:
                row_scope_hint = normalized_row_id.split("_", 1)[0]
                if (
                    row_scope_hint in valid_scopes
                    and row_scope_hint != "all"
                    and row_scope_hint != normalized_scope
                ):
                    warnings.append(
                        "row_id belongs to scope `{hint}`, but query scope is `{scope}`. "
                        "Use scope=`{hint}` for that row, or use related_task_row_id if "
                        "a navigation row points to task evidence.".format(
                            hint=row_scope_hint,
                            scope=normalized_scope,
                        )
                    )
                else:
                    warnings.append(
                        "row_id matched no current-session memory rows; verify the row_id from summary_table/timeline."
                    )
            elif normalized_plan_id:
                warnings.append(
                    "plan_id filter matched no current-session memory rows; remove plan_id unless the user asked about that exact plan."
                )
            elif resolved_task_id >= 0:
                warnings.append(
                    "task_id matched no rows; task_id is plan-local, so prefer row_id from summary_table/timeline."
                )

        rendered_items = sorted(
            rendered_items,
            key=lambda item: str(item.get("time") or item.get("created_at") or ""),
        )
        if normalized_time == "recent":
            rendered_items = rendered_items[-resolved_limit:]
        else:
            rendered_items = rendered_items[:resolved_limit]
        budget_chars = MEMORY_VIEW_BUDGET_CHARS[normalized_view]
        rendered_items, was_truncated, estimated_chars = _apply_memory_budget(
            rendered_items,
            view=normalized_view,
            budget_chars=budget_chars,
        )
        if was_truncated:
            warnings.append(f"memory_{normalized_view}_truncated")

        return {
            "status": "ok",
            "summary": "Memory query returned {count} item(s) from scope `{scope}` as `{view}`.".format(
                count=len(rendered_items),
                scope=normalized_scope,
                view=normalized_view,
            ),
            "scope": normalized_scope,
            "view": normalized_view,
            "query": normalized_query,
            "time": normalized_time,
            "plan_id": normalized_plan_id or None,
            "turn_id": normalized_turn_id or None,
            "task_id": resolved_task_id if resolved_task_id >= 0 else None,
            "row_id": normalized_row_id or None,
            "normalized_args": {
                "scope": normalized_scope,
                "view": normalized_view,
                "query": normalized_query,
                "time": normalized_time,
                "plan_id": normalized_plan_id,
                "turn_id": normalized_turn_id,
                "task_id": resolved_task_id,
                "row_id": normalized_row_id,
                "limit": resolved_limit,
            },
            "warnings": warnings,
            "truncated": was_truncated,
            "budget_chars": budget_chars,
            "estimated_chars": estimated_chars,
            "raw_match_count": raw_count,
            "items": deepcopy(rendered_items),
        }

__all__ = ["AsyncAgentRunMemory"]
