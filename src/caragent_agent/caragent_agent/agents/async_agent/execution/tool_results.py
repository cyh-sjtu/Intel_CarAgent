"""Shared helpers for the structured async-agent tool result contract."""

from __future__ import annotations

import json
import re
from typing import Any, Optional


def dedupe_ints(values: list[Any] | tuple[Any, ...], *, limit: int = 24) -> list[int]:
    """Return unique integer values in first-seen order."""

    seen: set[int] = set()
    result: list[int] = []
    for value in values:
        try:
            item = int(value)
        except Exception:
            continue
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
        if len(result) >= limit:
            break
    return result


def parse_json_like_payload(raw_value: Any) -> Any:
    """Parse a tool payload that may be a JSON string or already structured."""

    if isinstance(raw_value, str):
        try:
            return json.loads(raw_value)
        except Exception:
            return None
    return raw_value


def extract_keyframe_ids_from_payload(raw_value: Any) -> list[int]:
    """Extract candidate keyframe IDs from common search/tool payload shapes."""

    parsed = parse_json_like_payload(raw_value)
    candidates: list[Any] = []

    def visit(value: Any, *, allow_plain_id_list: bool = False) -> None:
        if isinstance(value, dict):
            for key in (
                "matched_keyframe_ids",
                "keyframe_ids",
                "candidate_keyframe_ids",
                "recommended_keyframe_ids",
            ):
                raw_ids = value.get(key)
                if isinstance(raw_ids, (list, tuple)):
                    candidates.extend(raw_ids)
            for key in ("kf_id", "keyframe_id", "keyframe_node_id"):
                if key in value:
                    candidates.append(value.get(key))
            for nested in value.values():
                if isinstance(nested, (dict, list, tuple)):
                    visit(nested, allow_plain_id_list=False)
        elif isinstance(value, (list, tuple)):
            if allow_plain_id_list and all(
                not isinstance(item, (dict, list, tuple)) for item in value
            ):
                candidates.extend(value)
            else:
                for item in value:
                    if isinstance(item, (dict, list, tuple)):
                        visit(item, allow_plain_id_list=False)

    visit(parsed, allow_plain_id_list=True)

    if not candidates and isinstance(raw_value, str):
        patterns = (
            r"(?:kf|keyframe|keyframe[_\s-]*id)\s*[:#]?\s*(\d+)",
            r"\bKF\s*(\d+)\b",
        )
        for pattern in patterns:
            candidates.extend(re.findall(pattern, raw_value, flags=re.IGNORECASE))

    return dedupe_ints(candidates)


def merge_candidate_keyframes(
    existing: list[Any] | tuple[Any, ...],
    new_values: list[Any] | tuple[Any, ...],
    *,
    limit: int = 24,
) -> list[int]:
    """Merge background candidate IDs from multiple partial observations."""

    return dedupe_ints([*list(existing or []), *list(new_values or [])], limit=limit)


def is_structured_tool_result(value: Any) -> bool:
    """Return True when the payload matches the unified tool result contract."""

    return (
        isinstance(value, dict)
        and "status" in value
        and "summary" in value
        and "data" in value
        and "provenance" in value
    )


def extract_structured_tool_result(value: Any) -> Optional[dict[str, Any]]:
    """Read one structured tool result from raw content when possible."""

    if is_structured_tool_result(value):
        return value

    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except Exception:
            return None
        return parsed if is_structured_tool_result(parsed) else None

    return None


def extract_tool_result_status(value: Any) -> str:
    """Return the normalized status for one tool payload."""

    structured_result = extract_structured_tool_result(value)
    if structured_result is not None:
        return str(structured_result.get("status") or "").strip().lower()
    return "ok"


def extract_tool_result_data(value: Any) -> Any:
    """Return the structured data field when the tool uses the unified contract."""

    structured_result = extract_structured_tool_result(value)
    if structured_result is None:
        return None
    return structured_result.get("data")


__all__ = [
    "dedupe_ints",
    "extract_keyframe_ids_from_payload",
    "extract_structured_tool_result",
    "merge_candidate_keyframes",
    "parse_json_like_payload",
    "extract_tool_result_data",
    "extract_tool_result_status",
    "is_structured_tool_result",
]
