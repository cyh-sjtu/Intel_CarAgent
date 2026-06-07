"""User-facing response helpers for the async agent."""

from __future__ import annotations

from typing import Any, Callable, Optional

from ..execution.support import apply_turn_response
from ..execution.support import normalize_turn_response_items
from ..runtime.types import AsyncAgentState, TaskItem


def collect_navigation_route_facts(
    tasks: dict[int, TaskItem],
    *,
    plan_id: Optional[str],
    user_input_id: Optional[str],
    task_destination_label: Callable[[Optional[TaskItem]], str],
    is_navigation_action: Callable[[Optional[TaskItem]], bool],
    navigation_state_for_task: Callable[[Optional[TaskItem]], Optional[str]],
) -> Optional[dict[str, Any]]:
    """Summarize route progress across navigation tasks for response synthesis."""

    navigation_steps: list[dict[str, Any]] = []
    for task_id in sorted(tasks):
        task = tasks[task_id]
        if plan_id and task.get("plan_id") != plan_id:
            continue
        if user_input_id and task.get("user_input_id") != user_input_id:
            continue
        if not is_navigation_action(task):
            continue

        navigation_steps.append(
            {
                "task_id": task_id,
                "description": task.get("description"),
                "destination_label": task_destination_label(task),
                "navigation_state": navigation_state_for_task(task),
            }
        )

    if not navigation_steps:
        return None

    current_step = next(
        (
            step
            for step in reversed(navigation_steps)
            if step.get("navigation_state") in {"in_transit", "arrived"}
        ),
        navigation_steps[-1],
    )
    route_status = "completed"
    if any(step.get("navigation_state") == "in_transit" for step in navigation_steps):
        route_status = "in_progress"
    elif any(step.get("navigation_state") not in {"arrived", None} for step in navigation_steps):
        route_status = "mixed"

    return {
        "navigation_task_count": len(navigation_steps),
        "route_status": route_status,
        "current_destination": current_step.get("destination_label"),
        "completed_destinations": [
            step["destination_label"]
            for step in navigation_steps
            if step.get("navigation_state") == "arrived"
        ],
        "pending_destinations": [
            step["destination_label"]
            for step in navigation_steps
            if step.get("navigation_state") == "in_transit"
        ],
        "navigation_steps": navigation_steps,
    }


def fallback_navigation_user_facing_response(
    tasks: dict[int, TaskItem],
    *,
    plan_id: Optional[str],
    user_input_id: Optional[str],
    task_destination_label: Callable[[Optional[TaskItem]], str],
    is_navigation_action: Callable[[Optional[TaskItem]], bool],
    navigation_state_for_task: Callable[[Optional[TaskItem]], Optional[str]],
) -> str:
    """Return a deterministic navigation-aware answer when synthesis is unavailable."""

    route_facts = collect_navigation_route_facts(
        tasks,
        plan_id=plan_id,
        user_input_id=user_input_id,
        task_destination_label=task_destination_label,
        is_navigation_action=is_navigation_action,
        navigation_state_for_task=navigation_state_for_task,
    )
    if not route_facts:
        return ""

    destination = str(route_facts.get("current_destination") or "").strip()
    if not destination:
        return ""

    route_status = route_facts.get("route_status")
    step_count = int(route_facts.get("navigation_task_count", 0) or 0)

    if route_status == "in_progress":
        if step_count > 1:
            return f"The tour is in progress. The robot is now on its way to {destination}."
        return f"The robot is now on its way to {destination}."

    if route_status == "completed":
        if step_count > 1:
            return f"My tour has been completed. I am now at {destination}."
        return f"I have arrived at {destination}."

    return ""


def finish_plan_without_user_response(
    state: AsyncAgentState,
    current_state_tasks: dict[int, TaskItem],
    *,
    logger: Optional[Any],
    shared_runtime_control: Optional[dict[str, Any]],
    deactivate_runtime_plan: Callable[[dict[str, Any]], None],
) -> AsyncAgentState:
    """Finish the current plan without synthesizing an extra final answer."""

    if logger:
        logger.log_foreground(
            "Orchestrate: All tasks in the plan have been completed."
        )
    if shared_runtime_control is not None:
        deactivate_runtime_plan(shared_runtime_control)

    response_items = normalize_turn_response_items(state.get("turn_response_items"))
    headline_type = str(state.get("turn_response_type") or "none")
    headline_text = state.get("turn_response_text")
    headline_id = state.get("turn_response_id")
    user_facing_response = state.get("user_facing_response")
    user_facing_response_id = state.get("user_facing_response_id")

    result_state = {
        **state,
        "tasks": {},
        "current_task_id": None,
        "current_plan_id": None,
        "next_action": {"type": "idle"},
        "messages": state["messages"],
        "turn_response_items": response_items,
        "turn_response_type": headline_type,
        "turn_response_text": headline_text,
        "turn_response_id": headline_id,
        "user_facing_response": user_facing_response,
        "user_facing_response_id": user_facing_response_id,
    }
    if any(item.get("response_type") in {"result", "error"} for item in response_items):
        return result_state
    return apply_turn_response(
        result_state,
        response_text="The plan is complete.",
        response_type="progress",
        source_event_type="plan_finished",
    )


__all__ = [
    "collect_navigation_route_facts",
    "fallback_navigation_user_facing_response",
    "finish_plan_without_user_response",
]
