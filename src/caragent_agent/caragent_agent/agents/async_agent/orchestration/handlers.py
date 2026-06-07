"""Event-handler and orchestration transition helpers for the async agent."""

from __future__ import annotations

import json
from typing import Any, Callable, Optional, Sequence

from langchain_core.messages import AIMessage, HumanMessage

from ..execution.support import (
    apply_turn_response,
    clear_turn_response,
    derive_headline_turn_response,
    normalize_turn_response_items,
)
from ..runtime.types import (
    AsyncAgentState,
    EventItem,
    NextAction,
    OrchestrateContext,
    TaskItem,
    TurnResponseType,
)
from ..runtime.control import record_decision_branch, set_background_enabled


def normalize_next_action_value(
    action: Optional[NextAction],
    fallback_next_action: Optional[NextAction],
) -> NextAction:
    """Return a valid next_action payload with an idle fallback."""

    if action:
        return action
    if isinstance(fallback_next_action, dict) and fallback_next_action.get("type"):
        return fallback_next_action
    return {"type": "idle"}


def normalize_turn_response_type(value: Any) -> Optional[TurnResponseType]:
    """Normalize free-form response type payloads into the supported contract."""

    normalized = str(value or "").strip().lower()
    if normalized in {"result", "progress", "error", "none"}:
        return normalized  # type: ignore[return-value]
    return None


def get_payload_turn_response(
    payload: Optional[dict[str, Any]],
) -> tuple[Optional[TurnResponseType], Optional[str]]:
    """Extract one turn reply from an event payload."""

    normalized_payload = payload or {}
    response_text = str(
        normalized_payload.get("turn_response_text")
        or normalized_payload.get("user_facing_response")
        or ""
    ).strip()
    response_type = normalize_turn_response_type(
        normalized_payload.get("turn_response_type")
    )
    if response_text and response_type is None:
        response_type = "result"
    if not response_text:
        return response_type, None
    return response_type, response_text


def get_payload_turn_response_items(
    payload: Optional[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Extract all response items carried by one payload."""

    normalized_payload = payload or {}
    response_items = normalize_turn_response_items(
        normalized_payload.get("turn_response_items")
    )
    if response_items:
        return response_items

    response_type, response_text = get_payload_turn_response(normalized_payload)
    if not response_text:
        return []
    return normalize_turn_response_items(
        [
            {
                "response_type": response_type or "result",
                "response_text": response_text,
                "response_id": normalized_payload.get("turn_response_id")
                or normalized_payload.get("user_facing_response_id"),
                "source_event_type": normalized_payload.get("event_type"),
                "task_id": normalized_payload.get("task_id"),
            }
        ]
    )


def build_user_facing_decision_progress(
    *,
    reason_text: str,
    chosen_branch: str,
    next_task: Optional[TaskItem],
) -> Optional[str]:
    """Return a concise progress note for a resolved branch decision."""

    normalized_reason = str(reason_text or "").strip()
    lowered_reason = normalized_reason.lower()
    internal_markers = (
        "upstream task",
        "source_task",
        "task id",
        "task_id",
        "structured evidence",
    )
    if normalized_reason and not any(marker in lowered_reason for marker in internal_markers):
        return normalized_reason

    next_description = str(next_task.get("description") or "").strip() if next_task else ""
    if next_description:
        return f"Based on the completed observations, I will continue with: {next_description}."

    normalized_branch = str(chosen_branch or "").replace("_", " ").strip()
    if normalized_branch:
        return f"Based on the completed observations, I selected the {normalized_branch} route."

    return None


def build_plan_created_progress(event_payload: Optional[dict[str, Any]]) -> str:
    """Return a lightweight progress note after a plan is created."""

    payload = event_payload or {}
    task_ids = payload.get("task_ids")
    task_count = len(task_ids) if isinstance(task_ids, list) else 0
    if task_count > 0:
        return f"I have prepared a {task_count}-step plan and will start the first step now."
    return "I have prepared a plan and will start the first step now."


def build_plan_finished_progress(state: AsyncAgentState) -> Optional[str]:
    """Return a plan-complete note only when no richer reply was emitted this turn."""

    response_items = normalize_turn_response_items(state.get("turn_response_items"))
    if response_items:
        return None
    return "The plan is complete."


def apply_task_cancelled_progress(
    state: AsyncAgentState,
    event: EventItem,
    tasks: dict[int, TaskItem],
) -> AsyncAgentState:
    """Append a visible progress item for one cancelled task when it has a label."""

    task_id = event.get("task_id")
    task = tasks.get(task_id) if task_id is not None else None
    task_description = str(
        (task or {}).get("description")
        or (event.get("payload", {}) or {}).get("task_description")
        or ""
    ).strip()
    if not task_description:
        return state

    return apply_turn_response(
        state,
        response_text=f"\u5df2\u53d6\u6d88\u4efb\u52a1: {task_description}",
        response_type="progress",
        source_event_type=str(event.get("type") or ""),
        task_id=task_id if isinstance(task_id, int) else None,
    )


def ensure_turn_response_fields(result_state: AsyncAgentState) -> AsyncAgentState:
    """Normalize turn response fields after one orchestration transition."""

    response_items = normalize_turn_response_items(result_state.get("turn_response_items"))
    response_text = str(
        result_state.get("turn_response_text")
        or result_state.get("user_facing_response")
        or ""
    ).strip()
    response_type = normalize_turn_response_type(result_state.get("turn_response_type"))
    response_id = (
        result_state.get("turn_response_id")
        or result_state.get("user_facing_response_id")
    )

    if response_items:
        headline_type, headline_text, headline_id = derive_headline_turn_response(
            response_items
        )
        result_state["turn_response_items"] = response_items
        result_state["turn_response_type"] = headline_type
        result_state["turn_response_text"] = headline_text
        result_state["turn_response_id"] = headline_id
        result_state["user_facing_response"] = headline_text
        result_state["user_facing_response_id"] = headline_id
        return result_state

    if response_text:
        if response_type is None or response_type == "none":
            response_type = "result"
        result_state["turn_response_items"] = normalize_turn_response_items(
            [
                {
                    "response_type": response_type,
                    "response_text": response_text,
                    "response_id": response_id,
                }
            ]
        )
        result_state["turn_response_type"] = response_type
        result_state["turn_response_text"] = response_text
        result_state["turn_response_id"] = response_id
        result_state["user_facing_response"] = response_text
        result_state["user_facing_response_id"] = response_id
        return result_state

    result_state.setdefault("turn_response_items", [])
    result_state.setdefault("turn_response_type", "none")
    result_state.setdefault("turn_response_text", None)
    result_state.setdefault("turn_response_id", None)
    result_state.setdefault("user_facing_response", None)
    result_state.setdefault("user_facing_response_id", None)
    return result_state


def mark_event_processed(
    result_state: AsyncAgentState,
    event: EventItem,
    *,
    processed_event_ids: Sequence[str],
    fallback_next_action: Optional[NextAction],
    logger: Optional[Any],
    next_action_override: Optional[NextAction] = None,
) -> AsyncAgentState:
    """Record an event as processed and ensure next_action remains well-formed."""

    updated_processed_ids = list(
        result_state.get("processed_event_ids", processed_event_ids)
    )
    event_id = event.get("event_id")
    if event_id and event_id not in updated_processed_ids:
        updated_processed_ids.append(event_id)

    result_state = ensure_turn_response_fields(result_state)
    result_state["processed_event_ids"] = updated_processed_ids
    if next_action_override is not None:
        result_state["next_action"] = normalize_next_action_value(
            next_action_override,
            fallback_next_action,
        )
    elif "next_action" not in result_state:
        result_state["next_action"] = normalize_next_action_value(
            None,
            fallback_next_action,
        )

    if logger:
        logger.log_foreground(
            "Orchestrate Debug: consumed event_id={event_id}, type={event_type}, next_action={next_action_value}, processed_count={processed_count}".format(
                event_id=event.get("event_id"),
                event_type=event.get("type"),
                next_action_value=result_state.get("next_action"),
                processed_count=len(updated_processed_ids),
            )
        )

    return result_state


def finish_inserted_task(
    state: AsyncAgentState,
    finished_task_id: int,
    current_state_tasks: dict[int, TaskItem],
    *,
    error_message: Optional[str] = None,
) -> AsyncAgentState:
    """Remove a runtime-inserted task and resume its parent context if any."""

    finished_task = current_state_tasks.get(finished_task_id)
    remaining_tasks = {
        task_id: task
        for task_id, task in current_state_tasks.items()
        if task_id != finished_task_id
    }

    parent_task_id = (
        finished_task.get("parent_task_id")
        if finished_task is not None
        else None
    )

    result_state: AsyncAgentState = {
        **state,
        "tasks": remaining_tasks,
        "next_action": {"type": "idle"},
    }

    if error_message is not None:
        result_state["error_message"] = error_message

    if parent_task_id is not None and parent_task_id in remaining_tasks:
        parent_task = remaining_tasks[parent_task_id]
        result_state["current_task_id"] = parent_task_id
        if parent_task.get("status") == "waiting":
            result_state["next_action"] = {"type": "idle"}
        else:
            result_state["next_action"] = (
                {"type": "idle"}
                if parent_task.get("type") == "decision"
                else {"type": "execute", "task_id": parent_task_id}
            )
        return result_state

    result_state["current_task_id"] = None
    return result_state


def evaluate_decision_and_proceed(
    state: AsyncAgentState,
    decision_task: TaskItem,
    current_state_tasks: dict[int, TaskItem],
    *,
    llm: Any,
    logger: Optional[Any],
    messages: Sequence[Any],
    events: Sequence[EventItem],
    shared_runtime_control: Optional[dict[str, Any]],
    run_memory: Optional[Any],
    resolve_default_branch: Callable[[TaskItem, dict[str, int]], Optional[str]],
    build_decision_context: Callable[..., dict[str, Any]],
    parse_decision_choice: Callable[[str, dict[str, int], Optional[str]], tuple[Optional[str], str]],
    append_task_result: Callable[..., None],
    proceed_to_next_task_fn: Callable[..., AsyncAgentState],
    new_structured_id: Callable[[str], str],
) -> AsyncAgentState:
    """Evaluate a decision task and continue to the chosen branch."""

    branches = decision_task.get("branches") or {}
    if not branches:
        if logger:
            logger.log_foreground(
                f"Orchestrate: Decision task {decision_task['task_id']} has no branches."
            )
        return clear_turn_response(
            {
                **state,
                "error_message": f"Decision task {decision_task['task_id']} has no branches.",
                "next_action": {"type": "idle"},
            }
        )

    if logger:
        logger.log_foreground(
            f"Orchestrate: Evaluating decision: {decision_task.get('description')}"
        )

    default_branch = resolve_default_branch(decision_task, branches)
    decision_context = build_decision_context(
        decision_task,
        current_state_tasks,
        events,
        messages,
        state=state,
        run_memory=run_memory,
        logger=logger,
    )
    prompt_decision_context = {
        key: value
        for key, value in decision_context.items()
        if not str(key).startswith("_")
    }
    routing_prompt = (
        decision_task.get("routing_prompt")
        or decision_task.get("condition")
        or decision_task.get("description")
        or "Choose the best branch."
    )
    decision_prompt = (
        "You are a routing decision engine for a task graph.\n"
        "Choose exactly one branch label from the available branches.\n"
        "Use only the current decision context: upstream task results first, then related current-run events.\n"
        "If the evidence is insufficient or ambiguous, choose the configured default branch.\n\n"
        "The reason must be concise and user-facing. Do not mention task ids, upstream task, JSON fields, or internal workflow names.\n\n"
        f"Routing instruction:\n{routing_prompt}\n\n"
        f"Available branches: {list(branches.keys())}\n"
        f"Default branch: {default_branch}\n\n"
        "Return JSON only with this schema:\n"
        '{"branch": "<branch label>", "reason": "<short reason>"}'
        "\n\n"
        f"Decision context:\n{json.dumps(prompt_decision_context, ensure_ascii=False, indent=2)}"
    )

    response = llm.invoke([HumanMessage(content=decision_prompt)])
    raw_result = str(getattr(response, "content", "") or "").strip()
    chosen_branch, parsed_reason = parse_decision_choice(
        raw_result,
        branches,
        default_branch,
    )
    if chosen_branch is None:
        error_message = (
            f"Decision task {decision_task['task_id']} could not resolve a valid branch."
        )
        if logger:
            logger.log_foreground(error_message)
        return clear_turn_response(
            {
                **state,
                "error_message": error_message,
                "next_action": {"type": "idle"},
            }
        )

    reason_text = parsed_reason or f"Selected branch '{chosen_branch}'."
    if logger:
        logger.log_foreground(
            f"Orchestrate: Decision result branch={chosen_branch}"
        )

    next_id = branches.get(chosen_branch)
    if shared_runtime_control is not None:
        record_decision_branch(
            shared_runtime_control,
            decision_task_id=int(decision_task["task_id"]),
            branch=chosen_branch,
            target_task_id=int(next_id),
            plan_id=decision_task.get("plan_id") or state.get("current_plan_id"),
        )
    decision_task["status"] = "completed"
    user_facing_decision_progress = build_user_facing_decision_progress(
        reason_text=reason_text,
        chosen_branch=chosen_branch,
        next_task=current_state_tasks.get(next_id),
    )
    append_task_result(
        decision_task,
        event_id=new_structured_id("event"),
        summary=reason_text,
        raw_output=raw_result,
        decision=chosen_branch,
    )
    try:
        if run_memory is not None:
            run_memory.record_task_result(
                task=decision_task,
                event_type="task_completed",
                summary=reason_text,
                tool_trace={
                    "tool_calls": [],
                    "tool_results": [],
                    "final_ai_content": raw_result,
                },
            )
    except Exception:
        pass

    result_state = proceed_to_next_task_fn(
        state,
        next_id,
        current_state_tasks,
        llm=llm,
        logger=logger,
        messages=messages,
        events=events,
        shared_runtime_control=shared_runtime_control,
        run_memory=run_memory,
    )
    if user_facing_decision_progress:
        result_state = apply_turn_response(
            result_state,
            response_text=user_facing_decision_progress,
            response_type="progress",
            source_event_type="decision_resolved",
            task_id=int(decision_task["task_id"]),
        )
    return result_state


def proceed_to_next_task(
    state: AsyncAgentState,
    next_id: Optional[int],
    current_state_tasks: dict[int, TaskItem],
    *,
    llm: Any,
    logger: Optional[Any],
    messages: Sequence[Any],
    events: Sequence[EventItem],
    shared_runtime_control: Optional[dict[str, Any]],
    run_memory: Optional[Any],
    append_task_result: Callable[..., None],
    finish_plan_without_user_response_fn: Callable[..., AsyncAgentState],
    evaluate_decision_and_proceed_fn: Callable[..., AsyncAgentState],
    new_structured_id: Callable[[str], str],
) -> AsyncAgentState:
    """Advance to the next task or finish the plan."""

    if next_id is not None and next_id in current_state_tasks:
        next_task = current_state_tasks[next_id]
        next_task["status"] = "in_progress"
        if logger:
            logger.log_foreground(
                f"Orchestrate: Proceeding to task {next_id} ({next_task['description']})"
            )

        if next_task.get("type") == "decision":
            next_task["task_type"] = "decision"
            return evaluate_decision_and_proceed_fn(
                state,
                next_task,
                current_state_tasks,
                llm=llm,
                logger=logger,
                messages=messages,
                events=events,
                shared_runtime_control=shared_runtime_control,
                run_memory=run_memory,
            )

        return {
            **state,
            "tasks": current_state_tasks,
            "current_task_id": next_id,
            "next_action": {"type": "execute", "task_id": next_id}
            if next_task.get("type") != "decision"
            else {"type": "idle"},
        }

    return finish_plan_without_user_response_fn(
        state,
        current_state_tasks,
        logger=logger,
        shared_runtime_control=shared_runtime_control,
    )


def mark_event_processed_from_context(
    result_state: AsyncAgentState,
    event: EventItem,
    context: OrchestrateContext,
) -> AsyncAgentState:
    """Mark an event processed using the shared orchestrate context."""

    return mark_event_processed(
        result_state,
        event,
        processed_event_ids=context["processed_event_ids"],
        fallback_next_action=context["next_action"],
        logger=context["logger"],
    )


def proceed_to_next_task_from_context(
    state: AsyncAgentState,
    next_id: Optional[int],
    current_state_tasks: dict[int, TaskItem],
    context: OrchestrateContext,
    *,
    proceed_to_next_task_fn: Callable[..., AsyncAgentState],
) -> AsyncAgentState:
    """Advance task execution using the shared orchestrate context."""

    return proceed_to_next_task_fn(
        state,
        next_id,
        current_state_tasks,
        llm=context["llm"],
        logger=context["logger"],
        messages=context["messages"],
        events=context["events"],
        shared_runtime_control=context["shared_runtime_control"],
        run_memory=context.get("run_memory"),
    )


def handle_plan_created_event(
    event: EventItem,
    context: OrchestrateContext,
) -> AsyncAgentState:
    """Activate the first planned task after a plan is created."""

    state = context["state"]
    tasks = context["tasks"]
    current_task_id = context["current_task_id"]

    first_task_id = event.get("payload", {}).get("first_task_id", current_task_id)
    if first_task_id is not None and first_task_id in tasks:
        first_task = tasks[first_task_id]
        if first_task.get("status") == "pending":
            first_task["status"] = "in_progress"

        result_state = {
            **state,
            "tasks": tasks,
            "current_task_id": first_task_id,
            "current_plan_id": event.get("payload", {}).get("plan_id"),
            "next_action": {"type": "execute", "task_id": first_task_id}
            if first_task.get("type") != "decision"
            else {"type": "idle"},
        }
        result_state = apply_turn_response(
            result_state,
            response_text=build_plan_created_progress(event.get("payload", {})),
            response_type="progress",
            source_event_type=str(event.get("type") or ""),
        )
        return mark_event_processed_from_context(result_state, event, context)

    return mark_event_processed_from_context(
        {
            **state,
            "next_action": {"type": "idle"},
        },
        event,
        context,
    )


def handle_task_cancelled_event(
    event: EventItem,
    context: OrchestrateContext,
) -> AsyncAgentState:
    """Consume task-cancelled events and surface visible cancellation progress."""

    result_state = apply_task_cancelled_progress(
        {**context["state"]},
        event,
        context["tasks"],
    )
    return mark_event_processed_from_context(result_state, event, context)


def handle_plan_cancelled_event(
    event: EventItem,
    context: OrchestrateContext,
) -> AsyncAgentState:
    """Consume audit-only plan-cancelled events."""

    return mark_event_processed_from_context({**context["state"]}, event, context)


def handle_task_completed_event(
    event: EventItem,
    context: OrchestrateContext,
    *,
    finish_inserted_task_fn: Callable[..., AsyncAgentState],
    proceed_to_next_task_from_context_fn: Callable[..., AsyncAgentState],
    apply_user_facing_response: Callable[[AsyncAgentState, Optional[str]], AsyncAgentState],
) -> AsyncAgentState:
    """Finalize a completed task and advance the plan when possible."""

    state = context["state"]
    tasks = context["tasks"]
    current_task_id = context["current_task_id"]

    completed_task_id = event.get("task_id")
    event_payload = event.get("payload", {})
    response_items = get_payload_turn_response_items(event_payload)
    response_type, user_facing_response = get_payload_turn_response(event_payload)
    if completed_task_id is None:
        completed_task_id = current_task_id

    if completed_task_id is not None and completed_task_id in tasks:
        if tasks[completed_task_id].get("inserted"):
            result_state = finish_inserted_task_fn(
                state,
                completed_task_id,
                tasks,
            )
            if response_items:
                for item in response_items:
                    result_state = apply_user_facing_response(
                        result_state,
                        str(item.get("response_text") or "").strip() or None,
                        response_type=item.get("response_type"),
                        source_event_type=str(event.get("type") or ""),
                        task_id=completed_task_id,
                    )
            else:
                result_state = apply_user_facing_response(
                    result_state,
                    user_facing_response,
                    response_type=response_type,
                    source_event_type=str(event.get("type") or ""),
                    task_id=completed_task_id,
                )
            return mark_event_processed_from_context(result_state, event, context)

        tasks[completed_task_id]["status"] = "completed"
        next_id = tasks[completed_task_id].get("next_task_id")
        result_state = proceed_to_next_task_from_context_fn(
            state,
            next_id,
            tasks,
            context,
        )
        if response_items:
            for item in response_items:
                result_state = apply_user_facing_response(
                    result_state,
                    str(item.get("response_text") or "").strip() or None,
                    response_type=item.get("response_type"),
                    source_event_type=str(event.get("type") or ""),
                    task_id=completed_task_id,
                )
        else:
            result_state = apply_user_facing_response(
                result_state,
                user_facing_response,
                response_type=response_type,
                source_event_type=str(event.get("type") or ""),
                task_id=completed_task_id,
            )
        return mark_event_processed_from_context(result_state, event, context)

    return mark_event_processed_from_context({**state}, event, context)


def handle_task_waiting_event(
    event: EventItem,
    context: OrchestrateContext,
    *,
    build_pending_navigation_snapshot_fn: Optional[Callable[..., dict[str, Any]]] = None,
) -> AsyncAgentState:
    """Put the current task into waiting mode until an external event arrives."""

    state = context["state"]
    tasks = context["tasks"]
    current_task_id = context["current_task_id"]
    response_items = get_payload_turn_response_items(event.get("payload", {}))
    response_type, response_text = get_payload_turn_response(event.get("payload", {}))

    waiting_task_id = event.get("task_id")
    if waiting_task_id is None:
        waiting_task_id = current_task_id

    if waiting_task_id is not None and waiting_task_id in tasks:
        tasks[waiting_task_id]["status"] = "waiting"
        runtime_control = context.get("shared_runtime_control")
        if runtime_control is not None:
            set_background_enabled(runtime_control, True)
        if (
            str(event.get("payload", {}).get("tool_name") or "") == "go_to_keyframe"
            and build_pending_navigation_snapshot_fn is not None
        ):
            active_navigation = build_pending_navigation_snapshot_fn(
                tasks[waiting_task_id],
                task_id=waiting_task_id,
                created_at=str(event.get("created_at") or ""),
            )
            state["active_navigation"] = active_navigation
            state["pending_navigation"] = active_navigation

    result_state: AsyncAgentState = {
        **state,
        "tasks": tasks,
        "next_action": {"type": "idle"},
    }
    if response_items:
        for item in response_items:
            result_state = apply_turn_response(
                result_state,
                response_text=str(item.get("response_text") or "").strip() or None,
                response_type=item.get("response_type"),
                source_event_type=str(event.get("type") or ""),
                task_id=waiting_task_id,
            )
    elif response_text:
        result_state = apply_turn_response(
            result_state,
            response_text=response_text,
            response_type=response_type,
            source_event_type=str(event.get("type") or ""),
            task_id=waiting_task_id,
        )

    return mark_event_processed_from_context(
        {
            **result_state,
        },
        event,
        context,
    )


def handle_task_failed_event(
    event: EventItem,
    context: OrchestrateContext,
    *,
    finish_inserted_task_fn: Callable[..., AsyncAgentState],
    apply_user_facing_response: Callable[[AsyncAgentState, Optional[str]], AsyncAgentState],
    deactivate_runtime_plan: Callable[[dict[str, Any]], None],
) -> AsyncAgentState:
    """Handle task failure and terminate or unwind runtime-inserted tasks."""

    state = context["state"]
    tasks = context["tasks"]
    current_task_id = context["current_task_id"]

    failed_task_id = event.get("task_id")
    failure_summary = event.get("payload", {}).get(
        "summary", "Task execution failed."
    )
    event_payload = event.get("payload", {})
    response_items = get_payload_turn_response_items(event_payload)
    response_type, user_facing_response = get_payload_turn_response(event_payload)

    if failed_task_id is None:
        failed_task_id = current_task_id

    if failed_task_id is not None and failed_task_id in tasks:
        if tasks[failed_task_id].get("inserted"):
            result_state = finish_inserted_task_fn(
                state,
                failed_task_id,
                tasks,
                error_message=failure_summary,
            )
            if response_items:
                for item in response_items:
                    result_state = apply_user_facing_response(
                        result_state,
                        str(item.get("response_text") or "").strip() or None,
                        response_type=item.get("response_type"),
                        source_event_type=str(event.get("type") or ""),
                        task_id=failed_task_id,
                    )
            else:
                result_state = apply_user_facing_response(
                    result_state,
                    user_facing_response,
                    response_type=response_type,
                    source_event_type=str(event.get("type") or ""),
                    task_id=failed_task_id,
                )
            result_state["messages"] = state["messages"] + [
                AIMessage(content=failure_summary)
            ]
            return mark_event_processed_from_context(result_state, event, context)

        tasks[failed_task_id]["status"] = "failed"
        tasks[failed_task_id]["terminal_reason"] = failure_summary
        deactivate_runtime_plan(context["shared_runtime_control"])

    result_state = {
        **state,
        "tasks": tasks,
        "current_task_id": None,
        "current_plan_id": None,
        "error_message": failure_summary,
        "next_action": {"type": "idle"},
        "messages": state["messages"] + [AIMessage(content=failure_summary)],
    }
    if response_items:
        for item in response_items:
            result_state = apply_user_facing_response(
                result_state,
                str(item.get("response_text") or "").strip() or None,
                response_type=item.get("response_type"),
                source_event_type=str(event.get("type") or ""),
                task_id=failed_task_id,
            )
    else:
        result_state = apply_user_facing_response(
            result_state,
            user_facing_response,
            response_type=response_type,
            source_event_type=str(event.get("type") or ""),
            task_id=failed_task_id,
        )
    return mark_event_processed_from_context(result_state, event, context)


def handle_navigation_arrived_event(
    event: EventItem,
    context: OrchestrateContext,
    *,
    finish_inserted_task_fn: Callable[..., AsyncAgentState],
    navigation_arrival_summary: Callable[[Optional[TaskItem]], str],
    fallback_navigation_user_facing_response_fn: Callable[..., str],
    append_task_result: Callable[..., None],
    proceed_to_next_task_from_context_fn: Callable[..., AsyncAgentState],
    new_structured_id: Callable[[str], str],
    apply_user_facing_response: Callable[[AsyncAgentState, Optional[str]], AsyncAgentState],
) -> AsyncAgentState:
    """Resume the workflow after a navigation arrival event."""

    state = context["state"]
    tasks = context["tasks"]
    current_task_id = context["current_task_id"]
    run_memory = context.get("run_memory")

    arrived_task_id = event.get("task_id")
    if arrived_task_id is None:
        arrived_task_id = current_task_id

    if arrived_task_id is not None and arrived_task_id in tasks:
        arrived_task_status = str(tasks[arrived_task_id].get("status") or "").lower()
        if arrived_task_status == "completed":
            return mark_event_processed_from_context({**state}, event, context)

    active_navigation = state.get("active_navigation")
    arrived_task_can_complete = False
    if arrived_task_id is not None and arrived_task_id in tasks:
        candidate_task = tasks[arrived_task_id]
        arrived_task_can_complete = (
            candidate_task.get("status") == "waiting"
            and candidate_task.get("wait_for_event") == "navigation_arrived"
        )

    if arrived_task_id is not None and arrived_task_id in tasks and arrived_task_can_complete:
        arrived_task = tasks[arrived_task_id]
        latest_result = (
            (arrived_task.get("result") or [])[-1]
            if arrived_task.get("result")
            else None
        )
        arrival_summary = navigation_arrival_summary(arrived_task)
        if arrived_task.get("inserted"):
            arrived_task["status"] = "completed"
            if run_memory:
                try:
                    run_memory.record_task_result(
                        task=arrived_task,
                        event_type="task_completed",
                        summary=arrival_summary,
                    )
                except Exception:
                    pass
            result_state = apply_user_facing_response(
                finish_inserted_task_fn(
                    state,
                    arrived_task_id,
                    tasks,
                ),
                arrival_summary,
                response_type="result",
                source_event_type=str(event.get("type") or ""),
                task_id=arrived_task_id,
            )
            result_state.pop("active_navigation", None)
            result_state.pop("pending_navigation", None)
            return mark_event_processed_from_context(result_state, event, context)

        arrived_task["status"] = "completed"
        arrival_payload = event.get("payload", {})
        arrival_payload["destination_description"] = str(
            arrived_task.get("description") or ""
        ).strip()
        latest_summary = str(latest_result.get("summary") or "").strip() if latest_result else ""
        if arrival_summary and arrival_summary != latest_summary:
            append_task_result(
                arrived_task,
                event_id=event.get("event_id", new_structured_id("event")),
                summary=arrival_summary,
                raw_output=json.dumps(arrival_payload, ensure_ascii=False),
                tool_name=latest_result.get("tool_name") if latest_result else None,
            )
        if run_memory:
            try:
                run_memory.record_task_result(
                    task=arrived_task,
                    event_type="task_completed",
                    summary=arrival_summary,
                )
            except Exception:
                pass
        next_id = arrived_task.get("next_task_id")
        user_facing_arrival_summary = fallback_navigation_user_facing_response_fn(
            tasks,
            plan_id=arrived_task.get("plan_id") or state.get("current_plan_id"),
            user_input_id=arrived_task.get("user_input_id"),
        ) or arrival_summary
        result_state = proceed_to_next_task_from_context_fn(
            apply_user_facing_response(
                state,
                user_facing_arrival_summary,
                response_type="result",
                source_event_type=str(event.get("type") or ""),
                task_id=arrived_task_id,
            ),
            next_id,
            tasks,
            context,
        )
        result_state.pop("active_navigation", None)
        result_state.pop("pending_navigation", None)
        return mark_event_processed_from_context(result_state, event, context)

    if isinstance(active_navigation, dict) and event.get("task_id") == active_navigation.get("task_id"):
        result_state = {
            **state,
            "active_navigation": None,
            "pending_navigation": None,
        }
        result_state.pop("pending_navigation", None)
        return mark_event_processed_from_context(result_state, event, context)

    return mark_event_processed_from_context({**state}, event, context)


def handle_navigation_arrival_unmatched_event(
    event: EventItem,
    context: OrchestrateContext,
) -> AsyncAgentState:
    """Record unmatched controller arrivals without completing or advancing tasks."""

    state = context["state"]
    logger = context.get("logger")
    payload = event.get("payload", {})
    if logger:
        logger.log_foreground(
            "Orchestrate: Ignoring unmatched navigation arrival; reason={reason}, reported={reported}, destination={destination}".format(
                reason=payload.get("unmatched_reason"),
                reported=payload.get("reported_position"),
                destination=payload.get("destination_position"),
            )
        )
    return mark_event_processed_from_context({**state}, event, context)


def _coerce_planning_requirement_decision(value: Any) -> dict[str, Any]:
    """Normalize planning-gate outputs."""

    if isinstance(value, dict):
        return {**value, "requires_planning": bool(value.get("requires_planning"))}
    return {"requires_planning": bool(value)}


def handle_user_input_received_event(
    event: EventItem,
    context: OrchestrateContext,
    *,
    is_require_planning_fn: Callable[..., bool],
    relation_classifier_fn: Callable[..., dict[str, Any]],
    build_replanning_cancellation_events_fn: Callable[..., list[EventItem]],
    deactivate_runtime_plan: Callable[[dict[str, Any]], None],
    new_runtime_task_id: Callable[[dict[int, TaskItem]], int],
    now_iso: Callable[[], str],
) -> AsyncAgentState:
    """Register user input and decide whether it should plan or execute directly."""

    state = context["state"]
    tasks = context["tasks"]
    current_task_id = context["current_task_id"]
    logger = context["logger"]
    shared_background_results = context["shared_background_results"]
    shared_processing_tasks = context["shared_processing_tasks"]
    shared_runtime_control = context["shared_runtime_control"]
    run_memory = context.get("run_memory")

    user_input = str(event.get("payload", {}).get("content", ""))
    user_input_id = event.get("user_input_id")
    user_inputs = list(state.get("user_inputs", []))
    if logger:
        logger.log_foreground(
            "Orchestrate: keeping user_input_id={user_input_id} unchanged; any referent or history lookup must be planned/executed as an explicit task.".format(
                user_input_id=user_input_id
            )
        )

    planning_decision = _coerce_planning_requirement_decision(
        is_require_planning_fn(user_input, context["llm"])
    )
    requires_plan = bool(planning_decision.get("requires_planning"))

    if requires_plan:
        relation_decision = relation_classifier_fn(
            state=state,
            tasks=tasks,
            current_task_id=current_task_id,
            llm=context["llm"],
            logger=logger,
            user_input=user_input,
            user_input_id=user_input_id,
            run_memory=run_memory,
        )
        relation_action = relation_decision.get("action", "new_plan")
        if relation_action == "unsupported_edit":
            reason = str(relation_decision.get("reason") or "").strip()
            response_text = (
                "I cannot safely apply that plan edit without risking the current plan. "
                "I will keep the current task running and leave the plan unchanged."
            )
            if reason:
                response_text += f" Reason: {reason}"
            result_state = apply_turn_response(
                clear_turn_response(
                    {
                        **state,
                        "user_inputs": user_inputs,
                        "next_action": {"type": "idle"},
                    }
                ),
                response_text=response_text,
                response_type="error",
                source_event_type="plan_edit_rejected",
            )
            return mark_event_processed_from_context(result_state, event, context)

        if current_task_id is not None and tasks and relation_action != "new_plan":
            result_state = clear_turn_response(
                {
                    **state,
                    "user_inputs": user_inputs,
                    "next_action": {
                        "type": "plan",
                        "user_input_id": user_input_id,
                        "plan_mode": relation_action,
                        "anchor_task_id": current_task_id,
                    },
                }
            )
            return mark_event_processed_from_context(result_state, event, context)

        cancellation_events = build_replanning_cancellation_events_fn(
            tasks,
            cancelling_user_input_id=user_input_id,
        )
        cancellation_response_state: AsyncAgentState = {}
        for cancel_event in cancellation_events:
            if cancel_event.get("type") == "task_cancelled":
                cancellation_response_state = apply_task_cancelled_progress(
                    cancellation_response_state,
                    cancel_event,
                    tasks,
                )
        deactivate_runtime_plan(shared_runtime_control)
        shared_background_results.clear()
        shared_processing_tasks.clear()

        result_state = clear_turn_response(
            {
                **state,
                "current_task_id": None,
                "current_plan_id": None,
                "tasks": {},
                "background_results": {},
                "user_inputs": user_inputs,
                "events": list(state.get("events", [])) + cancellation_events,
                "processed_event_ids": list(state.get("processed_event_ids", []))
                + [cancel_event["event_id"] for cancel_event in cancellation_events],
                "next_action": {
                    "type": "plan",
                    "user_input_id": user_input_id,
                },
            }
        )
        result_state = {
            **result_state,
            "turn_response_items": cancellation_response_state.get(
                "turn_response_items",
                [],
            ),
            "turn_response_type": cancellation_response_state.get(
                "turn_response_type",
                "none",
            ),
            "turn_response_text": cancellation_response_state.get(
                "turn_response_text",
            ),
            "turn_response_id": cancellation_response_state.get("turn_response_id"),
            "user_facing_response": cancellation_response_state.get(
                "user_facing_response"
            ),
            "user_facing_response_id": cancellation_response_state.get(
                "user_facing_response_id"
            ),
        }
        return mark_event_processed_from_context(result_state, event, context)

    simple_task_id = new_runtime_task_id(tasks)
    parent_task_id = current_task_id if current_task_id in tasks else None
    simple_task: TaskItem = {
        "task_id": simple_task_id,
        "task_type": str(planning_decision.get("task_type") or "llm_action"),
        "type": "action",
        "description": user_input,
        "status": "in_progress",
        "next_task_id": None,
        "condition": None,
        "branches": None,
        "user_input_id": user_input_id,
        "depends_on": [],
        "result": [],
        "created_at": now_iso(),
        "updated_at": now_iso(),
        "inserted": True,
        "parent_task_id": parent_task_id,
    }
    result_state = clear_turn_response(
        {
            **state,
            "tasks": {
                **tasks,
                simple_task_id: simple_task,
            },
            "current_task_id": simple_task_id,
            "user_inputs": user_inputs,
            "next_action": {"type": "execute", "task_id": simple_task_id},
        }
    )
    return mark_event_processed_from_context(result_state, event, context)


__all__ = [
    "evaluate_decision_and_proceed",
    "finish_inserted_task",
    "handle_plan_cancelled_event",
    "handle_plan_created_event",
    "handle_navigation_arrived_event",
    "handle_task_cancelled_event",
    "handle_task_completed_event",
    "handle_task_failed_event",
    "handle_task_waiting_event",
    "handle_user_input_received_event",
    "mark_event_processed",
    "mark_event_processed_from_context",
    "normalize_next_action_value",
    "proceed_to_next_task",
    "proceed_to_next_task_from_context",
]
