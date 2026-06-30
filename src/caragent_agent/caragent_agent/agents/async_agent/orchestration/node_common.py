"""Shared helpers for async-agent graph node factories."""

from __future__ import annotations

from typing import Any, Optional

from caragent_agent.agents.async_agent.runtime.types import (
    AsyncAgentState,
    BackgroundAnalysisItem,
    TaskItem,
)


IGNORED_STATE_FIELDS = {"active_task", "awaiting_task_completion", "is_require_planning"}


def _strip_ignored_state_fields(state: AsyncAgentState) -> AsyncAgentState:
    """Remove unsupported state keys so they do not propagate through returns."""

    cleaned_state = dict(state)
    for field_name in IGNORED_STATE_FIELDS:
        cleaned_state.pop(field_name, None)
    return cleaned_state  # type: ignore[return-value]


def _get_current_task(
    tasks: dict[int, TaskItem],
    current_task_id: Optional[int],
) -> Optional[TaskItem]:
    """Resolve the authoritative current task from current_task_id."""

    if current_task_id is None:
        return None
    return tasks.get(current_task_id)


def _find_user_input_item(
    state: AsyncAgentState,
    user_input_id: Optional[str],
) -> Optional[dict[str, Any]]:
    """Return the matching user-input record from state when available."""

    normalized_user_input_id = str(user_input_id or "").strip()
    if not normalized_user_input_id:
        return None

    for item in reversed(list(state.get("user_inputs", []))):
        if str(item.get("user_input_id") or "").strip() != normalized_user_input_id:
            continue
        return {
            "user_input_id": item.get("user_input_id"),
            "content": item.get("content"),
            "created_at": item.get("created_at"),
            "original_content": item.get("original_content"),
            "resolved_referent_id": item.get("resolved_referent_id"),
        }
    return None


def _background_cache_status(
    bg_result: BackgroundAnalysisItem | str | None,
) -> str:
    """Return a compact lifecycle label for one cached background-analysis entry."""

    if isinstance(bg_result, dict):
        return str(bg_result.get("status") or "completed")
    if isinstance(bg_result, str) and bg_result.strip():
        return "completed"
    return "missing"


def _record_run_memory_event(
    run_memory: Optional[Any],
    event: dict[str, Any],
    *,
    stage: str,
) -> None:
    """Record one workflow event in run memory when the recorder exists."""

    if run_memory is None:
        return
    try:
        run_memory.record_event(event, stage=stage)
    except Exception:
        pass
