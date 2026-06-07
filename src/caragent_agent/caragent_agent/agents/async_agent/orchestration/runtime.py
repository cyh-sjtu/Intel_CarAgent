"""Runtime-state and ingest-event helpers for the async agent."""

from __future__ import annotations

import hashlib
import itertools
import math
import re
from datetime import datetime
from typing import Any, Optional, Sequence

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from ..runtime.referents import (
    extract_selected_keyframe_id_from_raw_output,
    extract_selected_position_from_raw_output,
)
from ..runtime.types import (
    EventItem,
    TaskItem,
    TaskStatus,
    UserInputItem,
)

_STRUCTURED_ID_COUNTER = itertools.count()


def now_iso() -> str:
    """Return a stable ISO timestamp string for state records."""

    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def new_structured_id(prefix: str) -> str:
    """Generate a simple sortable identifier for events, plans, and inputs."""

    return (
        f"{prefix}_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}_"
        f"{next(_STRUCTURED_ID_COUNTER):06d}"
    )


def get_message_id(
    message: BaseMessage | None,
    *,
    fallback_prefix: str,
    message_history: Optional[Sequence[BaseMessage]] = None,
) -> str:
    """Extract a message id when available, otherwise synthesize one."""

    if message is not None:
        message_id = getattr(message, "id", None)
        if message_id:
            return str(message_id)
        if message_history:
            for index, candidate in enumerate(message_history):
                if candidate is message:
                    return f"{fallback_prefix}_{message.__class__.__name__.lower()}_{index}"
        message_content = getattr(message, "content", "")
        digest = hashlib.sha1(
            f"{message.__class__.__name__}:{message_content}".encode("utf-8")
        ).hexdigest()[:12]
        return f"{fallback_prefix}_{digest}"
    return new_structured_id(fallback_prefix)


def new_runtime_task_id(tasks: dict[int, TaskItem]) -> int:
    """Allocate a negative task id for runtime-inserted tasks."""

    negative_ids = [task_id for task_id in tasks if task_id < 0]
    if not negative_ids:
        return -1
    return min(negative_ids) - 1


def task_status_is_terminal(status: Optional[TaskStatus]) -> bool:
    """Return True when a task status represents a terminal lifecycle state."""

    return status in {"completed", "failed", "cancelled"}


def merge_background_results(left: dict, right: dict) -> dict:
    """Merge background results emitted by parallel workers into one mapping."""

    if not left:
        return right if right else {}
    if not right:
        return left
    return {**left, **right}


def message_is_navigation_arrival(message: BaseMessage | None) -> bool:
    """Return True when a system message reports navigation arrival."""

    return (
        isinstance(message, SystemMessage)
        and "arrived at destination" in str(message.content).lower()
    )


def resolve_navigation_task_id(
    preferred_task_id: Optional[int],
    tasks: dict[int, TaskItem],
) -> Optional[int]:
    """Resolve the waiting navigation task that should consume an arrival event."""

    if (
        preferred_task_id is not None
        and preferred_task_id in tasks
        and tasks[preferred_task_id].get("wait_for_event") == "navigation_arrived"
    ):
        return preferred_task_id

    for task_id, task in tasks.items():
        if (
            task.get("status") == "waiting"
            and task.get("wait_for_event") == "navigation_arrived"
        ):
            return task_id

    if preferred_task_id is not None and preferred_task_id in tasks:
        return preferred_task_id

    return None


def parse_navigation_arrival_position(content: Any) -> Optional[list[float]]:
    """Parse controller text like 'Arrived at destination [x, y, z]'."""

    text = str(content or "")
    match = re.search(r"arrived\s+at\s+destination\s*\[([^\]]+)\]", text, re.IGNORECASE)
    if not match:
        return None
    parts = [part.strip() for part in match.group(1).split(",")]
    if len(parts) < 3:
        return None
    try:
        return [float(parts[0]), float(parts[1]), float(parts[2])]
    except Exception:
        return None


def _safe_float_triplet(value: Any) -> Optional[list[float]]:
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return None
    try:
        return [float(value[0]), float(value[1]), float(value[2])]
    except Exception:
        return None


def _position_distance(left: Sequence[float], right: Sequence[float]) -> float:
    return math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(left[:3], right[:3])))


def build_pending_navigation_snapshot(
    task: TaskItem,
    *,
    task_id: int,
    created_at: Optional[str] = None,
) -> dict[str, Any]:
    """Capture an in-flight navigation target so replanning does not orphan arrival."""

    snapshot: dict[str, Any] = {
        "task_id": task_id,
        "plan_id": task.get("plan_id"),
        "user_input_id": task.get("user_input_id"),
        "description": str(task.get("description") or "").strip(),
        "created_at": created_at or now_iso(),
    }
    latest_result = (
        (task.get("result") or [])[-1]
        if task.get("result")
        else None
    )
    latest_raw_output = latest_result.get("raw_output") if latest_result else None
    keyframe_id = extract_selected_keyframe_id_from_raw_output(latest_raw_output)
    position = extract_selected_position_from_raw_output(latest_raw_output)
    if keyframe_id is not None:
        snapshot["destination_keyframe_id"] = keyframe_id
    if position is not None:
        snapshot["destination_position"] = position
    if latest_result is not None:
        snapshot["tool_name"] = latest_result.get("tool_name")
        snapshot["waiting_summary"] = latest_result.get("summary")
    return snapshot


def build_navigation_arrival_event(
    message: BaseMessage,
    *,
    messages: Sequence[BaseMessage],
    tasks: dict[int, TaskItem],
    current_task_id: Optional[int],
    active_navigation: Optional[dict[str, Any]] = None,
    pending_navigation: Optional[dict[str, Any]] = None,
    match_tolerance_meters: float = 2.0,
) -> EventItem:
    """Convert a raw system arrival message into a matched or diagnostic event."""

    navigation = active_navigation if isinstance(active_navigation, dict) else None
    message_id = get_message_id(
        message,
        fallback_prefix="system_message",
        message_history=messages,
    )
    content = str(message.content)
    reported_position = parse_navigation_arrival_position(content)
    destination_position = (
        _safe_float_triplet(navigation.get("destination_position"))
        if navigation is not None
        else None
    )
    match_distance: Optional[float] = None
    is_matched = False
    if reported_position is not None and destination_position is not None:
        match_distance = _position_distance(reported_position, destination_position)
        is_matched = match_distance <= float(match_tolerance_meters)

    event_type = "navigation_arrived" if is_matched else "navigation_arrival_unmatched"
    event: EventItem = {
        "event_id": new_structured_id("event"),
        "type": event_type,  # type: ignore[typeddict-item]
        "source": "system",
        "created_at": now_iso(),
        "message_id": message_id,
        "payload": {
            "summary": (
                "Navigation arrival matched active navigation."
                if is_matched
                else "Navigation arrival did not match active navigation."
            ),
            "content": content,
            "reported_position": reported_position,
            "destination_position": destination_position,
            "match_tolerance_meters": match_tolerance_meters,
        },
    }
    if match_distance is not None:
        event["payload"]["match_distance_meters"] = match_distance

    if navigation is not None:
        navigation_task_id = navigation.get("task_id")
        try:
            event["task_id"] = int(navigation_task_id)
        except Exception:
            pass
        if navigation.get("plan_id"):
            event["plan_id"] = navigation.get("plan_id")
        if navigation.get("user_input_id"):
            event["user_input_id"] = navigation.get("user_input_id")
        event["payload"]["destination_description"] = str(
            navigation.get("description") or ""
        ).strip()
        if navigation.get("destination_keyframe_id") is not None:
            event["payload"]["destination_keyframe_id"] = navigation.get(
                "destination_keyframe_id"
            )
        if destination_position is not None:
            event["payload"]["destination_position"] = destination_position
        if event.get("task_id") not in tasks:
            event["payload"]["orphaned_waiting_navigation"] = True
    if navigation is None:
        event["payload"]["unmatched_reason"] = "missing_active_navigation"
    elif reported_position is None:
        event["payload"]["unmatched_reason"] = "missing_reported_position"
    elif destination_position is None:
        event["payload"]["unmatched_reason"] = "missing_destination_position"
    elif not is_matched:
        event["payload"]["unmatched_reason"] = "position_mismatch"
    return event


def build_user_input_received_event(
    message: HumanMessage,
    *,
    messages: Sequence[BaseMessage],
    user_inputs: Sequence[UserInputItem],
) -> tuple[EventItem, list[UserInputItem]]:
    """Convert a raw human message into a structured user-input event."""

    message_id = get_message_id(
        message,
        fallback_prefix="user_message",
        message_history=messages,
    )
    updated_user_inputs = list(user_inputs)
    existing_user_input = next(
        (
            item
            for item in reversed(updated_user_inputs)
            if item.get("message_id") == message_id
        ),
        None,
    )

    if existing_user_input is None:
        existing_user_input = {
            "user_input_id": new_structured_id("user_input"),
            "message_id": message_id,
            "content": str(message.content),
            "created_at": now_iso(),
        }
        updated_user_inputs.append(existing_user_input)

    event: EventItem = {
        "event_id": new_structured_id("event"),
        "type": "user_input_received",
        "source": "user",
        "created_at": now_iso(),
        "message_id": message_id,
        "user_input_id": existing_user_input["user_input_id"],
        "payload": {
            "content": existing_user_input["content"],
        },
    }
    return event, updated_user_inputs


def build_replanning_cancellation_events(
    tasks: dict[int, TaskItem],
    *,
    cancelling_user_input_id: Optional[str],
) -> list[EventItem]:
    """Create audit events that explicitly cancel the currently active task graph."""

    created_at = now_iso()
    cancellation_events: list[EventItem] = []
    cancelled_task_ids_by_plan: dict[str, list[int]] = {}

    for task_id, task in sorted(tasks.items()):
        if task_status_is_terminal(task.get("status")):
            continue

        task_plan_id = task.get("plan_id")
        if task_plan_id:
            cancelled_task_ids_by_plan.setdefault(task_plan_id, []).append(task_id)

        task_event: EventItem = {
            "event_id": new_structured_id("event"),
            "type": "task_cancelled",
            "source": "system",
            "created_at": created_at,
            "task_id": task_id,
            "payload": {
                "summary": "Task cancelled because a new user request triggered replanning.",
                "plan_id": task_plan_id,
                "task_description": task.get("description"),
                "cancelled_by_user_input_id": cancelling_user_input_id,
            },
        }
        if task.get("user_input_id"):
            task_event["user_input_id"] = task["user_input_id"]
        cancellation_events.append(task_event)

    for plan_id, cancelled_task_ids in sorted(cancelled_task_ids_by_plan.items()):
        cancellation_events.append(
            {
                "event_id": new_structured_id("event"),
                "type": "plan_cancelled",
                "source": "system",
                "created_at": created_at,
                "user_input_id": cancelling_user_input_id,
                "payload": {
                    "summary": "Plan cancelled because a new user request triggered replanning.",
                    "plan_id": plan_id,
                    "cancelled_task_ids": cancelled_task_ids,
                },
            }
        )

    return cancellation_events
