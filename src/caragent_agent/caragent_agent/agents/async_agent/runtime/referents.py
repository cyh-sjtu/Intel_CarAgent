"""Navigation result extraction helpers for the async agent."""

from __future__ import annotations

import json
import re
from typing import Any, Optional

from ..execution.tool_results import extract_structured_tool_result
from .types import TaskItem


def _parse_tool_trace(raw_output: Optional[str]) -> Optional[dict[str, Any]]:
    """Parse one stored tool trace payload when it is valid JSON."""

    if not raw_output:
        return None
    try:
        parsed = json.loads(raw_output)
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


def _extract_structured_result_data(tool_result: dict[str, Any]) -> Any:
    """Return the structured data field from one stored tool result when present."""

    structured_result = extract_structured_tool_result(tool_result.get("content"))
    if structured_result is None:
        return None
    return structured_result.get("data")


def _extract_position_triplet(raw_value: Any) -> Optional[list[float]]:
    """Normalize arbitrary position payloads into one 3D coordinate list."""

    if not isinstance(raw_value, (list, tuple)) or len(raw_value) < 3:
        return None
    try:
        return [float(raw_value[0]), float(raw_value[1]), float(raw_value[2])]
    except Exception:
        return None


def extract_selected_keyframe_id_from_raw_output(raw_output: Optional[str]) -> Optional[int]:
    """Recover the latest dispatched keyframe id from one stored tool trace."""

    tool_trace = _parse_tool_trace(raw_output)
    if tool_trace is None:
        return None

    for tool_call in reversed(list(tool_trace.get("tool_calls", []))):
        if str(tool_call.get("name") or "") != "go_to_keyframe":
            continue
        keyframe_id = tool_call.get("args", {}).get("keyframe_node_id")
        try:
            return int(keyframe_id)
        except Exception:
            continue
    return None


def extract_selected_position_from_raw_output(raw_output: Optional[str]) -> Optional[list[float]]:
    """Recover the latest dispatched destination position from one stored tool trace."""

    tool_trace = _parse_tool_trace(raw_output)
    if tool_trace is None:
        return None

    selected_keyframe_id = extract_selected_keyframe_id_from_raw_output(raw_output)
    for tool_result in reversed(list(tool_trace.get("tool_results", []))):
        tool_name = str(tool_result.get("name") or "")
        if tool_name == "go_to_keyframe":
            result_data = _extract_structured_result_data(tool_result)
            if not isinstance(result_data, dict):
                continue
            position = _extract_position_triplet(
                result_data.get("target_position")
            )
            if position is not None:
                return position
            continue

        if tool_name == "go_to_position":
            result_data = _extract_structured_result_data(tool_result)
            if not isinstance(result_data, dict):
                continue
            position = _extract_position_triplet(result_data.get("target_position"))
            if position is not None:
                return position
            continue

        if tool_name != "get_keyframe_nodes_info" or selected_keyframe_id is None:
            continue

        result_data = _extract_structured_result_data(tool_result)
        if isinstance(result_data, dict):
            nodes = result_data.get("nodes")
            if isinstance(nodes, dict):
                node_payload = nodes.get(str(selected_keyframe_id))
                if node_payload is None:
                    node_payload = nodes.get(selected_keyframe_id)
                if isinstance(node_payload, dict):
                    position = _extract_position_triplet(node_payload.get("position"))
                    if position is not None:
                        return position

        content = str(tool_result.get("content") or "")
        keyframe_pattern = re.compile(
            rf"{re.escape(str(selected_keyframe_id))}\s*:\s*\{{.*?'position': array\(\[\s*([-0-9.eE]+)\s*,\s*([-0-9.eE]+)\s*,\s*([-0-9.eE]+)\s*\]\)",
            re.DOTALL,
        )
        match = keyframe_pattern.search(content)
        if not match:
            continue
        try:
            return [
                float(match.group(1)),
                float(match.group(2)),
                float(match.group(3)),
            ]
        except Exception:
            continue
    return None


__all__ = [
    "extract_selected_keyframe_id_from_raw_output",
    "extract_selected_position_from_raw_output",
]
