"""Interaction-profile and voice-guidance helpers for the async agent.

The guidance layer is intentionally user-interface facing.  It does not feed
back into planning, task routing, or navigation control.
"""

from __future__ import annotations

import time
import re
from datetime import datetime, timezone
from typing import Any, Optional, TypedDict
from typing_extensions import NotRequired

from caragent_agent.config.config import config


class InteractionProfile(TypedDict, total=False):
    """User-facing interaction profile selected by configuration."""

    profile_id: str
    display_name: str
    language: str
    voice_enabled_default: bool
    speak_agent_replies: bool
    speak_guidance_events: bool
    response_role_enabled: bool
    response_role: str
    progress_reminder_interval_sec: float
    demo_safety_notice: str


class GuidanceEvent(TypedDict):
    """One UI/voice guidance event emitted from stable runtime transitions."""

    event_id: str
    event_type: str
    text: str
    created_at: str
    priority: str
    interrupt: bool
    dedupe_key: str
    task_id: NotRequired[int]
    source_event_id: NotRequired[str]
    source_event_type: NotRequired[str]
    payload: NotRequired[dict[str, Any]]


DEFAULT_INTERACTION_PROFILE: InteractionProfile = {
    "profile_id": "semantic_guide_demo",
    "display_name": "语音提示辅助导引演示",
    "language": "zh",
    "voice_enabled_default": True,
    "speak_agent_replies": True,
    "speak_guidance_events": True,
    "response_role_enabled": True,
    "response_role": "blind_assistance_companion",
    "progress_reminder_interval_sec": 18.0,
    "demo_safety_notice": "演示模式下请在旁站人员看护下缓慢跟随，必要时立即停止。",
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_guidance_id() -> str:
    return f"guidance_{time.time_ns()}"


def clean_destination_label(value: Any) -> str:
    """Clean request verbs from a destination label used for display/voice."""

    label = str(value or "").strip()
    if not label:
        return ""
    label = re.sub(r"\s+", " ", label)
    label = label.strip(" \t\r\n。.!?！？")
    replacements = (
        r"^我想(?:带你?去|帮我)?(?:去找|找|寻找|去|到|前往|靠近)\s*",
        r"^请(?:你)?(?:帮我)?\s*",
        r"^(?:帮我)?(?:带我去找|带我去|带我到|去找|找|寻找|去|到|前往|靠近)\s*",
    )
    for pattern in replacements:
        label = re.sub(pattern, "", label, flags=re.IGNORECASE).strip()
    return label[:48]


def get_interaction_profile() -> InteractionProfile:
    """Return the configured interaction profile with safe defaults."""

    raw_profile = config.get("interaction_profile")
    if isinstance(raw_profile, str):
        return {
            **DEFAULT_INTERACTION_PROFILE,
            "profile_id": raw_profile.strip() or DEFAULT_INTERACTION_PROFILE["profile_id"],
        }
    if isinstance(raw_profile, dict):
        return {
            **DEFAULT_INTERACTION_PROFILE,
            **{
                str(key): value
                for key, value in raw_profile.items()
                if value is not None
            },
        }
    return dict(DEFAULT_INTERACTION_PROFILE)


def _profile_enabled(profile: Optional[InteractionProfile] = None) -> bool:
    active = profile or get_interaction_profile()
    return str(active.get("profile_id") or "").strip().lower() != "none"


def destination_label(task: Optional[dict[str, Any]]) -> str:
    """Return a compact Chinese destination label for UI guidance."""

    if not task:
        return "目的地"
    task_label = clean_destination_label(task.get("display_label") or task.get("user_query") or "")
    if task_label:
        return task_label
    target = task.get("target") if isinstance(task.get("target"), dict) else {}
    if target.get("keyframe_id") is not None:
        return f"关键帧 {target.get('keyframe_id')}"
    position = target.get("position")
    if isinstance(position, list) and len(position) >= 2:
        try:
            return "坐标 ({:.2f}, {:.2f})".format(float(position[0]), float(position[1]))
        except Exception:
            return "目标坐标"
    query = str(
        target.get("display_label")
        or target.get("user_query")
        or target.get("query")
        or target.get("object_description")
        or ""
    ).strip()
    if query:
        return clean_destination_label(query) or query[:48]
    description = str(task.get("description") or "").strip()
    return clean_destination_label(description) or (description[:48] if description else "目的地")


def navigation_waiting_text(task: Optional[dict[str, Any]]) -> str:
    """Chinese user-facing text for a dispatched navigation command."""

    label = destination_label(task)
    target = task.get("target") if isinstance((task or {}).get("target"), dict) else {}
    target_type = str(target.get("type") or "").strip().lower()
    if target_type == "semantic_object":
        return f"正在靠近目标物体“{label}”。"
    if target_type == "semantic_keyframe":
        return f"正在前往目标关键帧“{label}”。"
    return f"正在前往目标地点“{label}”。"


def navigation_arrival_text(task: Optional[dict[str, Any]]) -> str:
    """Chinese user-facing text for a confirmed navigation arrival."""

    label = destination_label(task)
    target = task.get("target") if isinstance((task or {}).get("target"), dict) else {}
    target_type = str(target.get("type") or "").strip().lower()
    if target_type == "semantic_object":
        return f"已靠近目标物体“{label}”。"
    if target_type == "semantic_keyframe":
        return f"已到达目标关键帧“{label}”。"
    return f"已到达目标地点“{label}”附近。"


def plan_created_text(task_count: int = 0) -> str:
    """Chinese progress text for a newly prepared task plan."""

    if task_count > 0:
        return f"已生成计划，包含 {task_count} 个任务。"
    return "已生成计划。"


def plan_finished_text() -> str:
    """Chinese progress text for a completed plan when no richer result exists."""

    return "这次任务完成了。我会停在这里等你。"


def task_cancelled_text(task_description: str) -> str:
    """Chinese progress text for cancellation."""

    if task_description:
        return f"我已停止当前任务：{task_description}。请先停在原地。"
    return "我已停止当前任务。请先停在原地。"


def _friendly_failure_reason(summary: str) -> str:
    """Map internal failure summaries to short user-facing Chinese reasons."""

    normalized = " ".join(str(summary or "").strip().split())
    if not normalized:
        return ""
    lowered = normalized.lower()
    if "background_job_watchdog_timeout" in lowered or "watchdog" in lowered:
        return "目标定位用时过长，暂时没有得到可靠位置"
    if "vlm_timeout" in lowered or "timed out" in lowered or "timeout" in lowered:
        return "视觉理解超时，暂时没有确认目标位置"
    if "no_detection" in lowered or "no candidate" in lowered:
        return "当前画面里没有稳定检测到目标"
    if "current camera image is unavailable" in lowered or "current_image_unavailable" in lowered:
        return "当前相机画面不可用"
    if "right stereo image" in lowered or "current_right_image_unavailable" in lowered:
        return "当前双目右图不可用，无法可靠估计距离"
    if "laserscan" in lowered or "current_scan_unavailable" in lowered:
        return "当前雷达数据不可用，无法可靠估计距离"
    if "object localization" in lowered or "object approach" in lowered:
        return "目标物体定位没有得到可靠结果"
    if any(token in normalized for token in ("{", "}", "[", "]", "_", "/home/", "Traceback", "Exception")):
        return "内部定位流程没有得到可靠结果"
    return normalized


def failure_text(summary: str = "") -> str:
    """Chinese error text for unresolved or failed tasks."""

    normalized = _friendly_failure_reason(summary)
    if normalized:
        return f"这个目标我暂时没法可靠完成，原因是：{normalized}。请你换一种说法，或请旁站人员帮忙确认。"
    return "这个目标我暂时没法可靠确认位置。请你换一种说法，或请旁站人员帮忙确认。"


def build_guidance_event(
    *,
    event_type: str,
    text: str,
    priority: str = "normal",
    interrupt: bool = False,
    dedupe_key: Optional[str] = None,
    task_id: Optional[int] = None,
    source_event: Optional[dict[str, Any]] = None,
    payload: Optional[dict[str, Any]] = None,
) -> Optional[GuidanceEvent]:
    """Build one guidance event for the UI voice layer."""

    normalized_text = str(text or "").strip()
    normalized_type = str(event_type or "").strip()
    if not normalized_text or not normalized_type or not _profile_enabled():
        return None

    item: GuidanceEvent = {
        "event_id": _new_guidance_id(),
        "event_type": normalized_type,
        "text": normalized_text,
        "created_at": _now_iso(),
        "priority": str(priority or "normal").strip() or "normal",
        "interrupt": bool(interrupt),
        "dedupe_key": dedupe_key or normalized_type,
    }
    if task_id is not None:
        try:
            item["task_id"] = int(task_id)
        except Exception:
            pass
    if source_event:
        source_id = str(source_event.get("event_id") or "").strip()
        source_type = str(source_event.get("type") or "").strip()
        if source_id:
            item["source_event_id"] = source_id
        if source_type:
            item["source_event_type"] = source_type
    if payload:
        item["payload"] = payload
    return item


def append_guidance_event(
    state: dict[str, Any],
    event: Optional[GuidanceEvent],
    *,
    limit: int = 80,
) -> dict[str, Any]:
    """Append a guidance event to state with conservative dedupe."""

    if event is None:
        return state

    existing = [
        dict(item)
        for item in list(state.get("guidance_events") or [])
        if isinstance(item, dict)
    ]
    event_id = str(event.get("event_id") or "").strip()
    dedupe_key = str(event.get("dedupe_key") or "").strip()
    if event_id and any(str(item.get("event_id") or "") == event_id for item in existing):
        return state
    if dedupe_key and any(
        str(item.get("dedupe_key") or "").strip() == dedupe_key
        for item in existing
    ):
        return state
    existing.append(dict(event))
    state["guidance_events"] = existing[-limit:]
    return state


def append_guidance(
    state: dict[str, Any],
    *,
    event_type: str,
    text: str,
    priority: str = "normal",
    interrupt: bool = False,
    dedupe_key: Optional[str] = None,
    task_id: Optional[int] = None,
    source_event: Optional[dict[str, Any]] = None,
    payload: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Build and append one UI guidance event."""

    return append_guidance_event(
        state,
        build_guidance_event(
            event_type=event_type,
            text=text,
            priority=priority,
            interrupt=interrupt,
            dedupe_key=dedupe_key,
            task_id=task_id,
            source_event=source_event,
            payload=payload,
        ),
    )


__all__ = [
    "GuidanceEvent",
    "InteractionProfile",
    "append_guidance",
    "append_guidance_event",
    "build_guidance_event",
    "destination_label",
    "failure_text",
    "get_interaction_profile",
    "navigation_arrival_text",
    "navigation_waiting_text",
    "plan_created_text",
    "plan_finished_text",
    "task_cancelled_text",
]
