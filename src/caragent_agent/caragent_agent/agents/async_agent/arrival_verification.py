"""Arrival verification helpers for user-facing guidance.

This module is intentionally a side channel: it does not dispatch navigation,
create tasks, or alter task routing.  It only inspects the just-arrived scene
and returns a compact record plus a Chinese message for the user.
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional, Sequence

from langchain_core.tools import BaseTool

from caragent_agent.agents.async_agent.execution.runtime_tool_context import (
    runtime_tool_context,
)
from caragent_agent.config.config import config


def is_arrival_verification_enabled() -> bool:
    nav_cfg = config.get("navigation") if isinstance(config.get("navigation"), dict) else {}
    return bool(nav_cfg.get("arrival_verification_enabled", True))


def should_skip_arrival_verification_for_staging(
    arrived_task: dict[str, Any],
    tasks: dict[int, dict[str, Any]],
) -> bool:
    """Return True when this keyframe arrival is only staging for object grounding."""

    next_task_id = arrived_task.get("next_task_id")
    try:
        next_task_id = int(next_task_id)
    except Exception:
        return False
    next_task = tasks.get(next_task_id)
    if not isinstance(next_task, dict):
        return False
    if str(next_task.get("task_type") or "").strip() != "navigation_action":
        return False
    target = next_task.get("target") if isinstance(next_task.get("target"), dict) else {}
    if str(target.get("type") or "").strip() != "semantic_object":
        return False
    source = str(target.get("target_source") or "").strip()
    if source in {"arrived_scene", "upstream_result"}:
        return True
    inputs_from = target.get("inputs_from") or next_task.get("inputs_from")
    if isinstance(inputs_from, (list, tuple)):
        for item in inputs_from:
            if isinstance(item, dict):
                try:
                    if int(item.get("task_id")) == int(arrived_task.get("task_id")):
                        return True
                except Exception:
                    continue
    if isinstance(inputs_from, dict):
        for value in inputs_from.values():
            try:
                if int(value) == int(arrived_task.get("task_id")):
                    return True
            except Exception:
                continue
    return False


def semantic_arrival_target(task: dict[str, Any]) -> tuple[str, str]:
    """Return (target_type, target_label) when the task is worth verifying."""

    target = task.get("target") if isinstance(task.get("target"), dict) else {}
    target_type = str(target.get("type") or "").strip()
    latest = latest_task_result(task)
    current_place_context = (
        latest.get("current_place_context")
        if isinstance(latest.get("current_place_context"), dict)
        else {}
    )
    if target_type not in {"semantic_keyframe", "semantic_object", "task_output"}:
        destination = latest.get("destination") if isinstance(latest.get("destination"), dict) else {}
        if not destination.get("display_label") and not destination.get("user_query"):
            return "", ""
        target_type = str(destination.get("target_type") or "semantic_keyframe")
    label = str(
        current_place_context.get("display_label")
        or current_place_context.get("user_query")
        or current_place_context.get("query")
        or current_place_context.get("target_text")
        or target.get("display_label")
        or target.get("user_query")
        or target.get("object_description")
        or target.get("query")
        or task.get("display_label")
        or task.get("user_query")
        or task.get("description")
        or ""
    ).strip()
    label = clean_target_label(label)
    return target_type, label


def latest_task_result(task: dict[str, Any]) -> dict[str, Any]:
    result = task.get("result")
    if isinstance(result, list) and result:
        item = result[-1]
        return item if isinstance(item, dict) else {}
    return result if isinstance(result, dict) else {}


def clean_target_label(value: Any) -> str:
    text = str(value or "").strip()
    text = re.sub(r"\s+", " ", text)
    for prefix in (
        "Navigate to ",
        "Go to ",
        "Guide me to ",
        "Take me to ",
        "Approach ",
        "Find ",
    ):
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix):].strip()
            break
    return text.strip(" 。.!?！？")[:48] or "目标"


def build_arrival_verification(
    *,
    arrived_task: dict[str, Any],
    tasks: dict[int, dict[str, Any]],
    tools: Sequence[BaseTool],
    event_payload: dict[str, Any],
) -> Optional[dict[str, Any]]:
    if not is_arrival_verification_enabled():
        return None
    if should_skip_arrival_verification_for_staging(arrived_task, tasks):
        return None
    nav_cfg = config.get("navigation") if isinstance(config.get("navigation"), dict) else {}
    if bool(nav_cfg.get("arrival_verification_semantic_only", True)):
        target_type, target_label = semantic_arrival_target(arrived_task)
        if not target_type or not target_label:
            return None
    else:
        target_type, target_label = semantic_arrival_target(arrived_task)
        if not target_label:
            target_label = clean_target_label(arrived_task.get("description"))
    if target_type == "semantic_object":
        return verify_object_arrival(arrived_task, tasks, tools, event_payload, target_label)
    return verify_keyframe_arrival(arrived_task, tools, event_payload, target_label, target_type)


def verify_keyframe_arrival(
    arrived_task: dict[str, Any],
    tools: Sequence[BaseTool],
    event_payload: dict[str, Any],
    target_label: str,
    target_type: str,
) -> dict[str, Any]:
    analyzer = find_tool_by_name(tools, "analyse_on_current_image")
    image_ref = event_payload.get("arrival_image_ref") if isinstance(event_payload, dict) else {}
    result: dict[str, Any] = {
        "type": "arrival_verification",
        "target_label": target_label,
        "target_type": target_type or "semantic_keyframe",
        "seen": None,
        "confidence": "unknown",
        "reason": "",
        "image_ref_id": image_ref.get("image_ref_id") if isinstance(image_ref, dict) else None,
    }
    if analyzer is None:
        result.update(
            {
                "seen": None,
                "confidence": "unavailable",
                "reason": "current image analyzer unavailable",
                "message": f"已经到达附近。我会停在这里，但暂时无法自动确认是否看到{target_label}。",
            }
        )
        return result

    question = (
        "You are verifying arrival for a blind user. In the current robot camera image, "
        f"can you see the target '{target_label}'? Reply in compact JSON with keys: "
        "seen (true/false), direction_hint (front/front-left/front-right/left/right/unknown), "
        "distance_hint (very close/near/far/unknown), reason (short Chinese explanation). "
        "Do not mention keyframes, task ids, tools, or internal state."
    )
    raw = invoke_tool(analyzer, {"question": question})
    payload = parse_tool_payload(raw)
    answer = extract_current_image_answer(payload)
    result["raw_tool_result"] = compact_jsonable(payload)
    seen = infer_seen(answer)
    direction = infer_direction(answer)
    result["seen"] = seen
    result["direction_hint"] = direction
    result["reason"] = answer[:240]
    if seen is True:
        result["confidence"] = "vlm_confirmed"
        result["message"] = f"已经到达附近。我看到{target_label}了，在你{direction_text(direction)}。"
    elif seen is False:
        result["confidence"] = "vlm_not_seen"
        result["message"] = f"已经到达检索到的位置，但当前画面里没有确认看到{target_label}。可能被遮挡，或者位置有偏差。"
    else:
        result["confidence"] = "uncertain"
        result["message"] = f"已经到达附近。我已停下，但当前画面不能稳定确认{target_label}。"
    return result


def verify_object_arrival(
    arrived_task: dict[str, Any],
    tasks: dict[int, dict[str, Any]],
    tools: Sequence[BaseTool],
    event_payload: dict[str, Any],
    target_label: str,
) -> dict[str, Any]:
    approach_tool = find_tool_by_name(tools, "approach_object_in_current_view")
    image_ref = event_payload.get("arrival_image_ref") if isinstance(event_payload, dict) else {}
    nav_cfg = config.get("navigation") if isinstance(config.get("navigation"), dict) else {}
    stop_distance = float(nav_cfg.get("object_arrival_verify_stop_distance_m", 0.8) or 0.8)
    near_threshold = float(nav_cfg.get("object_arrival_near_threshold_m", 1.2) or 1.2)
    result: dict[str, Any] = {
        "type": "arrival_verification",
        "target_label": target_label,
        "target_type": "semantic_object",
        "seen": None,
        "confidence": "unknown",
        "reason": "",
        "image_ref_id": image_ref.get("image_ref_id") if isinstance(image_ref, dict) else None,
    }
    if approach_tool is None:
        return fallback_vlm_object_arrival(tools, result, target_label)

    raw = invoke_tool_with_arrival_context(
        approach_tool,
        {
            "target_description": target_label,
            "target_source": "current_view",
            "stop_distance_m": stop_distance,
        },
        arrived_task=arrived_task,
        tasks=tasks,
        event_payload=event_payload,
    )
    payload = parse_tool_payload(raw)
    data = payload.get("data") if isinstance(payload, dict) and isinstance(payload.get("data"), dict) else {}
    result["raw_tool_result"] = compact_jsonable(payload)
    distance = extract_object_distance_m(data)
    if distance is not None:
        result["distance_m"] = round(float(distance), 2)
    direction = infer_direction(json.dumps(data, ensure_ascii=False))
    if direction:
        result["direction_hint"] = direction
    status = str(payload.get("status") or "").strip().lower() if isinstance(payload, dict) else ""
    if status in {"ok", "success", "succeeded"}:
        result["seen"] = True
        result["confidence"] = "detected"
        if distance is not None:
            if float(distance) <= near_threshold:
                result["message"] = f"已经到达。我检测到{target_label}，在你{direction_text(direction)}，距离大约 {float(distance):.1f} 米。"
            else:
                result["message"] = f"已经到达附近。我检测到{target_label}，但看起来还有些距离，大约 {float(distance):.1f} 米。"
        else:
            result["message"] = f"已经到达。我检测到{target_label}，在你{direction_text(direction)}。"
        return result
    if status == "partial":
        return fallback_vlm_object_arrival(tools, result, target_label, tool_payload=payload)
    return fallback_vlm_object_arrival(tools, result, target_label, tool_payload=payload)


def fallback_vlm_object_arrival(
    tools: Sequence[BaseTool],
    base_result: dict[str, Any],
    target_label: str,
    *,
    tool_payload: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    analyzer = find_tool_by_name(tools, "analyse_on_current_image")
    if analyzer is None:
        reason = extract_error_reason(tool_payload) or "current image analyzer unavailable"
        base_result.update(
            {
                "seen": False,
                "confidence": "unavailable",
                "reason": reason,
                "message": f"已经到达附近，但当前没有稳定检测到{target_label}。{reason}",
            }
        )
        return base_result
    question = (
        "The object detector did not provide a reliable confirmation. Be conservative: "
        f"look at the current robot camera image and verify whether the exact target '{target_label}' "
        "is visible and very close to the robot. Reply in compact JSON only with keys: "
        "seen (true/false), near (true/false), direction_hint "
        "(front/front-left/front-right/left/right/unknown), reason (short Chinese explanation). "
        "Set seen=true only if the target itself is clearly visible. Set near=true only if the target "
        "appears within arm reach or only a close partial view is visible. If the target is absent, "
        "ambiguous, merely plausible, or only expected from navigation, set both false. Do not mention internal tools."
    )
    payload = parse_tool_payload(invoke_tool(analyzer, {"question": question}))
    answer = extract_current_image_answer(payload)
    base_result["vlm_fallback"] = compact_jsonable(payload)
    base_result["reason"] = answer[:240] or extract_error_reason(tool_payload)
    direction = infer_direction(answer)
    if direction:
        base_result["direction_hint"] = direction
    near = infer_near(answer)
    seen = infer_seen(answer)
    strong_close_evidence = has_strong_close_evidence(answer)
    if seen is True and (near is True or strong_close_evidence):
        base_result["seen"] = True
        base_result["confidence"] = "vlm_close_fallback"
        base_result["message"] = (
            f"已经到达。我没有稳定检测到{target_label}，但当前画面显示你已经在{target_label}附近，"
            "可能距离太近或只拍到局部。"
        )
    else:
        base_result["seen"] = False
        base_result["confidence"] = "not_confirmed"
        reason = base_result.get("reason") or extract_error_reason(tool_payload)
        if reason:
            base_result["message"] = f"已经到达附近，但当前画面没有确认看到{target_label}。{reason}"
        else:
            base_result["message"] = f"已经到达附近，但当前画面没有确认看到{target_label}。可能被遮挡、角度不对，或者检索位置有偏差。"
    return base_result


def find_tool_by_name(tools: Sequence[BaseTool], name: str) -> Optional[BaseTool]:
    for tool in tools:
        if str(getattr(tool, "name", "") or "").strip() == name:
            return tool
    return None


def invoke_tool(tool: BaseTool, args: dict[str, Any]) -> Any:
    invoke = getattr(tool, "invoke", None)
    if callable(invoke):
        try:
            return invoke(args)
        except TypeError:
            return invoke(**args)
    execute = getattr(tool, "execute", None)
    if callable(execute):
        return execute(**args)
    raise TypeError(f"Tool {getattr(tool, 'name', tool)} is not invokable.")


def invoke_tool_with_arrival_context(
    tool: BaseTool,
    args: dict[str, Any],
    *,
    arrived_task: dict[str, Any],
    tasks: dict[int, dict[str, Any]],
    event_payload: dict[str, Any],
) -> Any:
    """Invoke a perception tool as a fresh arrival-verification pass."""

    verification_task = dict(arrived_task or {})
    base_plan_id = str(
        verification_task.get("plan_id")
        or event_payload.get("plan_id")
        or "arrival_verification"
    ).strip()
    suffix = str(
        event_payload.get("arrival_event_id")
        or event_payload.get("navigation_token")
        or event_payload.get("content")
        or "current"
    ).strip()
    verification_task["plan_id"] = f"{base_plan_id}:arrival_verification:{suffix[:80]}"
    context = {
        "current_task": verification_task,
        "tasks": dict(tasks or {}),
        "arrival_verification": True,
    }
    with runtime_tool_context(context):
        return invoke_tool(tool, args)


def parse_tool_payload(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {"raw": value}
        except Exception:
            return {"raw": value}
    return {"raw": str(value)}


def extract_current_image_answer(payload: dict[str, Any]) -> str:
    data = payload.get("data") if isinstance(payload.get("data"), dict) else {}
    for key in ("answer", "description", "summary", "raw"):
        if data.get(key):
            return str(data.get(key)).strip()
        if payload.get(key):
            return str(payload.get(key)).strip()
    return ""


def infer_seen(text: str) -> Optional[bool]:
    lowered = str(text or "").lower()
    if re.search(r'"seen"\s*:\s*false|\bseen\s*[:=]\s*false|\bno\b|没有看到|未看到|不能确认|无法确认|不确定|看不清|未确认', lowered):
        return False
    if re.search(r'"seen"\s*:\s*true|\bseen\s*[:=]\s*true|clearly visible|明确可见|清楚看到|看到了|能看到|可见', lowered):
        return True
    return None


def infer_near(text: str) -> Optional[bool]:
    lowered = str(text or "").lower()
    if re.search(r'"near"\s*:\s*false|\bnear\s*[:=]\s*false|far|较远|很远|不在附近', lowered):
        return False
    if re.search(r'"near"\s*:\s*true|\bnear\s*[:=]\s*true|very close|within arm reach|贴近|局部|近距离|伸手|触手可及', lowered):
        return True
    return None


def has_strong_close_evidence(text: str) -> bool:
    lowered = str(text or "").lower()
    return bool(
        re.search(
            r"very close|within arm reach|贴近|局部|近距离|伸手|触手可及|只拍到局部|占据画面",
            lowered,
        )
    )


def infer_direction(text: str) -> str:
    lowered = str(text or "").lower()
    if "front-right" in lowered or "前方偏右" in lowered or "右前" in lowered:
        return "front-right"
    if "front-left" in lowered or "前方偏左" in lowered or "左前" in lowered:
        return "front-left"
    if "right" in lowered or "右侧" in lowered or "偏右" in lowered:
        return "right"
    if "left" in lowered or "左侧" in lowered or "偏左" in lowered:
        return "left"
    if "front" in lowered or "前方" in lowered or "正前" in lowered:
        return "front"
    return "unknown"


def direction_text(direction: str) -> str:
    return {
        "front": "前方不远处",
        "front-left": "前方偏左不远处",
        "front-right": "前方偏右不远处",
        "left": "左侧附近",
        "right": "右侧附近",
    }.get(str(direction or ""), "附近")


def extract_object_distance_m(data: dict[str, Any]) -> Optional[float]:
    candidates: list[Any] = []
    for key in (
        "distance_m",
        "distance_meters",
        "target_distance_m",
        "selected_depth_m",
        "median_depth_m",
        "mono_guard_selected_depth_m",
    ):
        candidates.append(data.get(key))
    checks = data.get("checks") if isinstance(data.get("checks"), dict) else {}
    for key in ("range_xy_m", "distance_m", "selected_depth_m", "mono_guard_selected_depth_m"):
        candidates.append(checks.get(key))
    selected = data.get("selected_object") if isinstance(data.get("selected_object"), dict) else {}
    for key in ("distance_m", "depth_m", "range_xy_m"):
        candidates.append(selected.get(key))
    for value in candidates:
        try:
            if value is not None:
                return float(value)
        except Exception:
            continue
    return None


def extract_error_reason(payload: Optional[dict[str, Any]]) -> str:
    if not isinstance(payload, dict):
        return ""
    error = payload.get("error") if isinstance(payload.get("error"), dict) else {}
    return str(error.get("message") or payload.get("summary") or payload.get("raw") or "").strip()


def compact_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        compact: dict[str, Any] = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= 24:
                compact["..."] = "truncated"
                break
            compact[str(key)] = compact_jsonable(item)
        return compact
    if isinstance(value, (list, tuple)):
        items = [compact_jsonable(item) for item in list(value)[:24]]
        if len(value) > 24:
            items.append("truncated")
        return items
    if isinstance(value, (str, int, float, bool)) or value is None:
        text = str(value) if isinstance(value, str) else value
        return text[:800] if isinstance(text, str) else text
    return str(value)[:800]
