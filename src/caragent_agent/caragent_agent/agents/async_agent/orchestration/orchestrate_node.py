"""Orchestrate node for event consumption and task lifecycle control."""

from __future__ import annotations

from typing import Any, Callable, Optional, Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage

from caragent_agent.agents.async_agent.execution.context import is_navigation_action
from caragent_agent.agents.async_agent.execution.support import (
    append_task_result,
    apply_user_facing_response,
    navigation_arrival_summary,
    navigation_state_for_task,
    task_destination_label,
)
from caragent_agent.agents.async_agent.orchestration.handlers import (
    evaluate_decision_and_proceed as _evaluate_decision_and_proceed_impl,
    finish_inserted_task as _finish_inserted_task_impl,
    handle_navigation_arrived_event as _handle_navigation_arrived_event_impl,
    handle_navigation_arrival_unmatched_event,
    handle_plan_cancelled_event,
    handle_plan_created_event,
    handle_task_cancelled_event,
    handle_task_completed_event as _handle_task_completed_event_impl,
    handle_task_failed_event as _handle_task_failed_event_impl,
    handle_task_waiting_event,
    handle_user_input_received_event as _handle_user_input_received_event_impl,
    mark_event_processed_from_context,
    normalize_next_action_value,
    proceed_to_next_task as _proceed_to_next_task_impl,
    proceed_to_next_task_from_context as _proceed_to_next_task_from_context_impl,
)
from caragent_agent.agents.async_agent.orchestration.node_common import (
    _get_current_task,
    _strip_ignored_state_fields,
)
from caragent_agent.agents.async_agent.orchestration.runtime import (
    build_pending_navigation_snapshot,
    build_replanning_cancellation_events,
    new_runtime_task_id,
    new_structured_id,
    now_iso,
)
from caragent_agent.agents.async_agent.planning.helpers import (
    build_decision_context,
    classify_current_plan_relation,
    parse_decision_choice,
    resolve_default_branch,
)
from caragent_agent.agents.async_agent.planning.prompting import (
    AGENT_PROMPTS,
    classify_planning_requirement,
    is_require_planning,
)
from caragent_agent.agents.async_agent.response.user_response import (
    fallback_navigation_user_facing_response as _fallback_navigation_user_facing_response_impl,
    finish_plan_without_user_response as _finish_plan_without_user_response_impl,
)
from caragent_agent.agents.async_agent.runtime.control import deactivate_runtime_plan
from caragent_agent.agents.async_agent.runtime.console import Colors
from caragent_agent.agents.async_agent.runtime.types import (
    AsyncAgentState,
    EventItem,
    EventType,
    OrchestrateContext,
    TaskItem,
)


_ORIGINAL_IS_REQUIRE_PLANNING = is_require_planning


def _planning_requirement_for_orchestrate(
    user_input: str,
    llm: BaseChatModel,
) -> dict[str, Any]:
    """Return structured planning metadata for orchestration."""

    if is_require_planning is not _ORIGINAL_IS_REQUIRE_PLANNING:
        requires_planning = bool(
            is_require_planning(user_input, llm, agent_prompts=AGENT_PROMPTS)
        )
        if requires_planning:
            return {"requires_planning": True}
        try:
            return dict(
                classify_planning_requirement(
                    user_input,
                    llm,
                    agent_prompts=AGENT_PROMPTS,
                )
            )
        except Exception:
            return {
                "requires_planning": False,
                "task_type": "llm_action",
            }

    return dict(
        classify_planning_requirement(
            user_input,
            llm,
            agent_prompts=AGENT_PROMPTS,
        )
    )

def _fallback_navigation_user_facing_response(
    tasks: dict[int, TaskItem],
    *,
    plan_id: Optional[str],
    user_input_id: Optional[str],
) -> str:
    """Return a deterministic navigation-aware answer when synthesis is unavailable."""

    return _fallback_navigation_user_facing_response_impl(
        tasks,
        plan_id=plan_id,
        user_input_id=user_input_id,
        task_destination_label=task_destination_label,
        is_navigation_action=is_navigation_action,
        navigation_state_for_task=navigation_state_for_task,
    )


def _finish_plan_without_user_response(
    state: AsyncAgentState,
    current_state_tasks: dict[int, TaskItem],
    *,
    logger: Optional[Any],
    shared_runtime_control: Optional[dict[str, Any]] = None,
) -> AsyncAgentState:
    """Finish the current plan without synthesizing an extra final response."""

    return _finish_plan_without_user_response_impl(
        state,
        current_state_tasks,
        logger=logger,
        shared_runtime_control=shared_runtime_control,
        deactivate_runtime_plan=deactivate_runtime_plan,
    )


def _finish_inserted_task(
    state: AsyncAgentState,
    finished_task_id: int,
    current_state_tasks: dict[int, TaskItem],
    *,
    error_message: Optional[str] = None,
) -> AsyncAgentState:
    """Remove a runtime-inserted task and resume its parent context if any."""

    return _finish_inserted_task_impl(
        state,
        finished_task_id,
        current_state_tasks,
        error_message=error_message,
    )


def _evaluate_decision_and_proceed(
    state: AsyncAgentState,
    decision_task: TaskItem,
    current_state_tasks: dict[int, TaskItem],
    *,
    llm: BaseChatModel,
    logger: Optional[Any],
    messages: Sequence[BaseMessage],
    events: Sequence[EventItem],
    shared_runtime_control: Optional[dict[str, Any]] = None,
    run_memory: Optional[Any] = None,
) -> AsyncAgentState:
    """Evaluate a decision task and continue to the chosen branch."""

    return _evaluate_decision_and_proceed_impl(
        state,
        decision_task,
        current_state_tasks,
        llm=llm,
        logger=logger,
        messages=messages,
        events=events,
        shared_runtime_control=shared_runtime_control,
        run_memory=run_memory,
        resolve_default_branch=resolve_default_branch,
        build_decision_context=build_decision_context,
        parse_decision_choice=parse_decision_choice,
        append_task_result=append_task_result,
        proceed_to_next_task_fn=_proceed_to_next_task,
        new_structured_id=new_structured_id,
    )


def _proceed_to_next_task(
    state: AsyncAgentState,
    next_id: Optional[int],
    current_state_tasks: dict[int, TaskItem],
    *,
    llm: BaseChatModel,
    logger: Optional[Any],
    messages: Sequence[BaseMessage],
    events: Sequence[EventItem],
    shared_runtime_control: Optional[dict[str, Any]] = None,
    run_memory: Optional[Any] = None,
) -> AsyncAgentState:
    """Advance to the next task or finish the plan."""

    return _proceed_to_next_task_impl(
        state,
        next_id,
        current_state_tasks,
        llm=llm,
        logger=logger,
        messages=messages,
        events=events,
        shared_runtime_control=shared_runtime_control,
        run_memory=run_memory,
        append_task_result=append_task_result,
        finish_plan_without_user_response_fn=_finish_plan_without_user_response,
        evaluate_decision_and_proceed_fn=_evaluate_decision_and_proceed,
        new_structured_id=new_structured_id,
    )


def _proceed_to_next_task_from_context(
    state: AsyncAgentState,
    next_id: Optional[int],
    current_state_tasks: dict[int, TaskItem],
    context: OrchestrateContext,
) -> AsyncAgentState:
    """Advance task execution using the shared orchestrate context."""

    return _proceed_to_next_task_from_context_impl(
        state,
        next_id,
        current_state_tasks,
        context,
        proceed_to_next_task_fn=_proceed_to_next_task,
    )


def _handle_user_input_received_event(
    event: EventItem,
    context: OrchestrateContext,
) -> AsyncAgentState:
    """Register user input and decide whether it should plan or execute directly."""

    if context["logger"]:
        context["logger"].log_foreground("Resume to Reasoning for new user input.")
        print(
            f"{Colors.ORCHESTRATE}Resume to Reasoning for new user input.{Colors.RESET}"
        )
    return _handle_user_input_received_event_impl(
        event,
        context,
        is_require_planning_fn=_planning_requirement_for_orchestrate,
        relation_classifier_fn=classify_current_plan_relation,
        build_replanning_cancellation_events_fn=build_replanning_cancellation_events,
        deactivate_runtime_plan=deactivate_runtime_plan,
        new_runtime_task_id=new_runtime_task_id,
        now_iso=now_iso,
    )


def _handle_task_completed_event(
    event: EventItem,
    context: OrchestrateContext,
) -> AsyncAgentState:
    """Finalize a completed task and advance the plan when possible."""

    return _handle_task_completed_event_impl(
        event,
        context,
        finish_inserted_task_fn=_finish_inserted_task,
        proceed_to_next_task_from_context_fn=_proceed_to_next_task_from_context,
        apply_user_facing_response=apply_user_facing_response,
    )


def _handle_task_failed_event(
    event: EventItem,
    context: OrchestrateContext,
) -> AsyncAgentState:
    """Handle task failure and terminate or unwind runtime-inserted tasks."""

    return _handle_task_failed_event_impl(
        event,
        context,
        finish_inserted_task_fn=_finish_inserted_task,
        apply_user_facing_response=apply_user_facing_response,
        deactivate_runtime_plan=deactivate_runtime_plan,
    )


def _handle_navigation_arrived_event(
    event: EventItem,
    context: OrchestrateContext,
) -> AsyncAgentState:
    """Resume the workflow after a navigation arrival event."""

    return _handle_navigation_arrived_event_impl(
        event,
        context,
        finish_inserted_task_fn=_finish_inserted_task,
        navigation_arrival_summary=navigation_arrival_summary,
        fallback_navigation_user_facing_response_fn=_fallback_navigation_user_facing_response,
        append_task_result=append_task_result,
        proceed_to_next_task_from_context_fn=_proceed_to_next_task_from_context,
        new_structured_id=new_structured_id,
        apply_user_facing_response=apply_user_facing_response,
    )

def create_orchestrate_node(
    llm: BaseChatModel,
    shared_background_results: dict,
    shared_processing_tasks: set,
    shared_runtime_control: Optional[dict[str, Any]] = None,
    logger: Optional[Any] = None,
    run_memory: Optional[Any] = None,
):
    """Dispatch user input, manage task lifecycle, and gate routing decisions."""

    def orchestrate_node(state: AsyncAgentState) -> AsyncAgentState:
        """Process pending events or choose the next plan/execute transition."""

        state = _strip_ignored_state_fields(state)
        if shared_runtime_control is None:
            runtime_control: dict[str, Any] = {}
        else:
            runtime_control = shared_runtime_control
        messages = state.get("messages", [])
        tasks = state.get("tasks", {})
        current_task_id = state.get("current_task_id")
        current_task = _get_current_task(tasks, current_task_id)
        events = list(state.get("events", []))
        processed_event_ids = state.get("processed_event_ids", [])
        next_action = state.get("next_action", {"type": "idle"})

        pending_events = [
            event
            for event in events
            if event.get("event_id") not in processed_event_ids
        ]

        if pending_events:
            event = pending_events[0]
            event_type = event.get("type")
            event_task_id = event.get("task_id")
            context: OrchestrateContext = {
                "state": state,
                "tasks": tasks,
                "current_task_id": current_task_id,
                "processed_event_ids": processed_event_ids,
                "next_action": next_action,
                "llm": llm,
                "logger": logger,
                "messages": messages,
                "events": events,
                "shared_background_results": shared_background_results,
                "shared_processing_tasks": shared_processing_tasks,
                "shared_runtime_control": runtime_control,
                "run_memory": run_memory,
            }
            event_handlers: dict[EventType, Callable[[EventItem, OrchestrateContext], AsyncAgentState]] = {
                "user_input_received": _handle_user_input_received_event,
                "plan_created": handle_plan_created_event,
                "task_completed": _handle_task_completed_event,
                "task_failed": _handle_task_failed_event,
                "task_waiting": lambda event, context: handle_task_waiting_event(
                    event,
                    context,
                    build_pending_navigation_snapshot_fn=build_pending_navigation_snapshot,
                ),
                "task_cancelled": handle_task_cancelled_event,
                "plan_cancelled": handle_plan_cancelled_event,
                "navigation_arrived": _handle_navigation_arrived_event,
                "navigation_arrival_unmatched": handle_navigation_arrival_unmatched_event,
            }

            if logger:
                logger.log_foreground(
                    f"Orchestrate: Processing structured event '{event_type}'"
                )
                print(
                    f"{Colors.ORCHESTRATE}Orchestrate: Processing structured event:{Colors.RESET} {event_type}"
                )
                logger.log_foreground(
                    "Orchestrate Debug: event_id={event_id}, task_id={task_id}, payload={payload}".format(
                        event_id=event.get("event_id"),
                        task_id=event_task_id,
                        payload=event.get("payload", {}),
                    )
                )

            handler = event_handlers.get(event_type)
            if handler is not None:
                return handler(event, context)

            return mark_event_processed_from_context({**state}, event, context)

        if current_task and current_task.get("type") == "decision":
            return _evaluate_decision_and_proceed(
                state,
                current_task,
                tasks,
                llm=llm,
                logger=logger,
                messages=messages,
                events=events,
                shared_runtime_control=shared_runtime_control,
                run_memory=run_memory,
            )

        return {
            **state,
            "next_action": normalize_next_action_value(None, next_action),
        }

    return orchestrate_node
