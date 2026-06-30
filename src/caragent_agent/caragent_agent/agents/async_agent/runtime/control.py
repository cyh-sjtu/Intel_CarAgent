"""Runtime-control helpers for foreground/background coordination."""

from __future__ import annotations

import threading
from typing import Any, Optional

from caragent_agent.config.runtime_profiles import resolve_runtime_profile


def build_runtime_control(config: dict[str, Any]) -> dict[str, Any]:
    """Create the shared runtime-control object used by graph nodes."""

    profile = resolve_runtime_profile(config)
    control: dict[str, Any] = {
        "active_plan_id": None,
        "background_generation": 0,
        "background_enabled": False,
        "foreground_current_task_id": None,
        "latest_foreground_task_id": None,
        "foreground_started_task_ids": set(),
        "resolved_decision_branches": {},
        "background_start_policy": str(
            profile.get("background_start_policy") or "after_first_navigation_dispatch"
        ),
        "speculative_branch_preanalysis": bool(
            profile.get("speculative_branch_preanalysis", False)
        ),
    }
    get_background_claim_lock(control)
    get_background_ready_event(control)
    if control["background_start_policy"] == "immediate_after_plan":
        set_background_enabled(control, True)
    return control


def task_processing_key(task_id: int, plan_id: Optional[str]) -> str:
    """Build a stable worker-coordination key for one task within one plan scope."""

    return f"{plan_id or 'runtime'}:{task_id}"


def get_background_claim_lock(control: dict[str, Any]) -> threading.Lock:
    """Return the lock that protects background task claiming."""

    lock = control.get("background_claim_lock")
    if isinstance(lock, threading.Lock().__class__):
        return lock
    lock = threading.Lock()
    control["background_claim_lock"] = lock
    return lock


def get_background_ready_event(control: dict[str, Any]) -> threading.Event:
    """Return the event used to wake background workers when analysis may start."""

    ready_event = control.get("background_ready_event")
    if isinstance(ready_event, threading.Event):
        return ready_event
    ready_event = threading.Event()
    if bool(control.get("background_enabled", False)):
        ready_event.set()
    control["background_ready_event"] = ready_event
    return ready_event


def set_background_enabled(control: dict[str, Any], enabled: bool) -> None:
    """Synchronize the background-enabled flag with its wakeup event."""

    control["background_enabled"] = bool(enabled)
    ready_event = get_background_ready_event(control)
    if enabled:
        ready_event.set()
    else:
        ready_event.clear()


def wake_background_waiters(control: dict[str, Any]) -> None:
    """Wake background workers so they can observe generation/plan changes."""

    get_background_ready_event(control).set()


def bump_background_generation(control: dict[str, Any]) -> int:
    """Advance the shared background generation token and return the new value."""

    current_value = int(control.get("background_generation", 0) or 0)
    next_value = current_value + 1
    control["background_generation"] = next_value
    wake_background_waiters(control)
    return next_value


def deactivate_runtime_plan(control: dict[str, Any]) -> None:
    """Invalidate background work for the current plan and clear its active token."""

    control["active_plan_id"] = None
    control["resolved_decision_branches"] = {}
    control["foreground_started_task_ids"] = set()
    control["foreground_current_task_id"] = None
    control["latest_foreground_task_id"] = None
    set_background_enabled(control, False)
    bump_background_generation(control)


def activate_runtime_plan(control: dict[str, Any], *, plan_id: str) -> int:
    """Activate one plan as the sole valid background-work scope."""

    control["active_plan_id"] = plan_id
    control["resolved_decision_branches"] = {}
    control["foreground_started_task_ids"] = set()
    control["foreground_current_task_id"] = None
    control["latest_foreground_task_id"] = None
    set_background_enabled(
        control,
        str(control.get("background_start_policy") or "")
        == "immediate_after_plan",
    )
    return bump_background_generation(control)


def record_foreground_task(control: dict[str, Any], task_id: int) -> None:
    """Record that foreground execution has started or reached one task."""

    control["foreground_current_task_id"] = task_id
    control["latest_foreground_task_id"] = task_id
    started_tasks = control.setdefault("foreground_started_task_ids", set())
    if hasattr(started_tasks, "add"):
        started_tasks.add(int(task_id))


def record_decision_branch(
    control: dict[str, Any],
    *,
    decision_task_id: int,
    branch: Any,
    target_task_id: int,
    plan_id: Optional[str],
) -> None:
    """Record the selected branch for background path selection."""

    resolved_decisions = control.setdefault("resolved_decision_branches", {})
    if isinstance(resolved_decisions, dict):
        resolved_decisions[int(decision_task_id)] = {
            "branch": branch,
            "target_task_id": int(target_task_id),
            "plan_id": plan_id,
        }


__all__ = [
    "activate_runtime_plan",
    "build_runtime_control",
    "bump_background_generation",
    "deactivate_runtime_plan",
    "get_background_claim_lock",
    "get_background_ready_event",
    "record_decision_branch",
    "record_foreground_task",
    "set_background_enabled",
    "task_processing_key",
    "wake_background_waiters",
]
