"""Planning node for creating and editing async-agent task plans."""

from __future__ import annotations

import json
from typing import Any, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from caragent_agent.agents.async_agent.execution.context import truncate_context_text
from caragent_agent.agents.async_agent.execution.support import (
    apply_turn_response,
    clear_turn_response,
)
from caragent_agent.agents.async_agent.orchestration.node_common import (
    _record_run_memory_event,
    _strip_ignored_state_fields,
)
from caragent_agent.agents.async_agent.orchestration.runtime import (
    get_message_id,
    new_structured_id,
    now_iso,
)
from caragent_agent.agents.async_agent.planning.helpers import (
    build_plan_editing_messages,
)
from caragent_agent.agents.async_agent.planning.plan_edit import (
    PlanEditError,
    apply_plan_edit,
    parse_plan_edit_from_response,
)
from caragent_agent.agents.async_agent.planning.prompting import AGENT_PROMPTS
from caragent_agent.agents.async_agent.planning.task_graph import (
    parse_planned_tasks_from_response,
)
from caragent_agent.agents.async_agent.runtime.control import (
    activate_runtime_plan,
    bump_background_generation,
)
from caragent_agent.agents.async_agent.runtime.console import Colors
from caragent_agent.agents.async_agent.runtime.types import AsyncAgentState, TaskItem


PLAN_EDIT_PROGRESS_TEXT = {
    "insert_after_current": "I have updated the plan and inserted the requested follow-up steps.",
    "replan_future_after_current": "I have replanned the remaining future steps.",
}

def _log_plan_error(
    logger: Optional[Any],
    *,
    error_msg: str,
    error: BaseException,
    plan_mode: Optional[str],
    plan_id: Optional[str],
    user_input_id: Optional[str],
    next_action: Optional[dict[str, Any]],
    plan_text: str = "",
) -> None:
    """Write enough planning failure context to reconstruct bad LLM edits."""

    if logger is None:
        return

    anchor_task_id = None
    if isinstance(next_action, dict):
        anchor_task_id = next_action.get("anchor_task_id")

    logger.log_foreground(
        "Plan Error: {error_msg}; error_type={error_type}; plan_mode={plan_mode}; "
        "plan_id={plan_id}; user_input_id={user_input_id}; anchor_task_id={anchor_task_id}".format(
            error_msg=error_msg,
            error_type=type(error).__name__,
            plan_mode=plan_mode,
            plan_id=plan_id,
            user_input_id=user_input_id,
            anchor_task_id=anchor_task_id,
        )
    )
    if plan_text:
        logger.log_foreground(
            "Plan Error Debug: planner_output_preview={preview}".format(
                preview=truncate_context_text(plan_text, limit=1200)
            )
        )

def _invalidate_background_after_plan_edit(
    *,
    shared_runtime_control: Optional[dict[str, Any]],
    shared_background_results: Optional[dict],
    shared_processing_tasks: Optional[set],
    logger: Optional[Any] = None,
    updated_tasks: Optional[dict[int, TaskItem]] = None,
) -> None:
    """Invalidate in-flight/background cache entries after an in-place plan edit."""

    preserved_results: dict[int, Any] = {}
    preserved_decision_branches: dict[int, Any] = {}
    if shared_background_results is not None:
        if updated_tasks:
            for task_id, bg_result in list(shared_background_results.items()):
                task = updated_tasks.get(task_id)
                if task is None or not isinstance(bg_result, dict):
                    continue
                # Only completed records are safe to carry across a plan edit.
                # Running partials belong to the old background generation; if
                # preserved, they block the new generation from recomputing the
                # same future resolver while still being non-actionable for the
                # foreground executor.
                if str(bg_result.get("status") or "").strip().lower() != "completed":
                    continue
                task_description = str(task.get("description") or "").strip()
                bg_description = str(bg_result.get("task_description") or "").strip()
                if task_description and task_description == bg_description:
                    preserved_results[task_id] = bg_result
        shared_background_results.clear()
        shared_background_results.update(preserved_results)
    if shared_processing_tasks is not None:
        # A plan edit bumps background_generation below. Old in-flight workers may
        # still finish, but their final writes are stale; clear claims so the new
        # generation can re-run still-future destination resolvers.
        shared_processing_tasks.clear()
    if shared_runtime_control is not None:
        if updated_tasks:
            active_plan_id = shared_runtime_control.get("active_plan_id")
            raw_decisions = shared_runtime_control.get("resolved_decision_branches")
            if isinstance(raw_decisions, dict):
                for raw_decision_id, raw_record in raw_decisions.items():
                    try:
                        decision_id = int(raw_decision_id)
                    except Exception:
                        continue
                    decision_task = updated_tasks.get(decision_id)
                    if not decision_task or decision_task.get("type") != "decision":
                        continue
                    if active_plan_id and decision_task.get("plan_id") != active_plan_id:
                        continue
                    if isinstance(raw_record, dict):
                        branch_label = raw_record.get("branch")
                        raw_target_id = raw_record.get("target_task_id")
                        record_plan_id = raw_record.get("plan_id")
                    else:
                        branch_label = None
                        raw_target_id = raw_record
                        record_plan_id = None
                    if record_plan_id and decision_task.get("plan_id") != record_plan_id:
                        continue
                    try:
                        target_id = int(raw_target_id)
                    except Exception:
                        continue
                    if target_id not in updated_tasks:
                        continue
                    branches = decision_task.get("branches") or {}
                    branch_still_matches = False
                    if branch_label is not None:
                        try:
                            branch_still_matches = int(branches.get(str(branch_label))) == target_id
                        except Exception:
                            branch_still_matches = False
                    else:
                        for raw_branch_target in branches.values():
                            try:
                                if int(raw_branch_target) == target_id:
                                    branch_still_matches = True
                                    break
                            except Exception:
                                continue
                    if branch_still_matches:
                        preserved_decision_branches[decision_id] = raw_record
        shared_runtime_control["resolved_decision_branches"] = preserved_decision_branches
        bump_background_generation(shared_runtime_control)
        if logger:
            logger.log_foreground(
                "Plan Edit: invalidated background generation after in-place plan edit"
            )
            if preserved_results:
                logger.log_foreground(
                    "Plan Edit: preserved background results for unchanged tasks: {task_ids}".format(
                        task_ids=sorted(preserved_results)
                    )
                )
            if preserved_decision_branches:
                logger.log_foreground(
                    "Plan Edit: preserved resolved decision branches for unchanged decisions: {task_ids}".format(
                        task_ids=sorted(preserved_decision_branches)
                    )
                )

def create_plan_node(
    llm: BaseChatModel,
    shared_runtime_control: Optional[dict[str, Any]] = None,
    shared_background_results: Optional[dict] = None,
    shared_processing_tasks: Optional[set] = None,
    logger: Optional[Any] = None,
    run_memory: Optional[Any] = None,
):
    """Generate a structured task plan from the latest user intent."""

    def plan_node(state: AsyncAgentState) -> AsyncAgentState:
        """Build a new plan or edit suffix tasks based on the requested plan mode."""

        state = _strip_ignored_state_fields(state)
        if logger:
            logger.log_foreground("Plan: Generating task plan")
            print(f"{Colors.PLAN}Plan: Generating task plan{Colors.RESET}")
        messages = state.get("messages", [])
        next_action = state.get("next_action", {"type": "idle"})
        user_inputs = list(state.get("user_inputs", []))
        existing_events = list(state.get("events", []))

        requested_user_input_id = next_action.get("user_input_id")
        selected_user_input = None
        if requested_user_input_id:
            selected_user_input = next(
                (
                    item
                    for item in reversed(user_inputs)
                    if item.get("user_input_id") == requested_user_input_id
                ),
                None,
            )

        user_message = next(
            (m for m in reversed(messages) if isinstance(m, HumanMessage)), None
        )
        if not user_message and not selected_user_input:
            return state

        if selected_user_input is None and user_message is not None:
            user_message_id = get_message_id(
                user_message,
                fallback_prefix="user_message",
                message_history=messages,
            )
            selected_user_input = next(
                (
                    item
                    for item in reversed(user_inputs)
                    if item.get("message_id") == user_message_id
                ),
                None,
            )
            if selected_user_input is None:
                selected_user_input = {
                    "user_input_id": new_structured_id("user_input"),
                    "message_id": user_message_id,
                    "content": user_message.content,
                    "created_at": now_iso(),
                }
                user_inputs.append(selected_user_input)

        if selected_user_input is None:
            return state

        user_input_id = selected_user_input["user_input_id"]
        created_at = now_iso()
        plan_mode = next_action.get("plan_mode", "new_plan")
        anchor_task_id = next_action.get("anchor_task_id")
        current_plan_id = state.get("current_plan_id")
        current_task_id = state.get("current_task_id")
        current_task = (
            state.get("tasks", {}).get(current_task_id)
            if current_task_id is not None
            else None
        )

        editable_plan_modes = {
            "insert_after_current",
            "replan_future_after_current",
        }

        plan_id = (
            current_plan_id
            if plan_mode in editable_plan_modes
            and current_plan_id
            else new_structured_id("plan")
        )

        if logger:
            logger.log_foreground(
                "Plan Debug: planning for user_input_id={user_input_id}, plan_id={plan_id}, mode={plan_mode}".format(
                    user_input_id=user_input_id,
                    plan_id=plan_id,
                    plan_mode=plan_mode,
                )
            )
            print(
                f"{Colors.PLAN}Plan Debug:{Colors.RESET} "
                f"user_input_id={user_input_id} plan_id={plan_id} mode={plan_mode}"
            )

        if plan_mode in editable_plan_modes:
            planning_messages = build_plan_editing_messages(
                user_request=selected_user_input["content"],
                current_task=current_task,
                tasks=state.get("tasks", {}),
                current_plan_id=current_plan_id,
                plan_mode=plan_mode,
            )
        else:
            planning_messages = [
                SystemMessage(content=AGENT_PROMPTS.get("plan_system", "")),
                HumanMessage(content=selected_user_input["content"]),
            ]

        plan_text = ""

        try:
            response = llm.invoke(planning_messages)
            plan_text = response.content.strip()

            if plan_mode in editable_plan_modes:
                existing_tasks = state.get("tasks", {})
                effective_anchor_task_id = (
                    anchor_task_id
                    if anchor_task_id is not None
                    else current_task_id
                )
                if (
                    effective_anchor_task_id is None
                    or effective_anchor_task_id not in existing_tasks
                ):
                    effective_anchor_task_id = current_task_id

                protected_task_ids = {
                    task_id
                    for task_id in [
                        effective_anchor_task_id,
                        current_task_id,
                    ]
                    if task_id is not None
                }
                parsed_edit = parse_plan_edit_from_response(
                    plan_text,
                    fallback_edit_type=plan_mode,
                )
                updated_tasks, touched_task_ids = apply_plan_edit(
                    existing_tasks,
                    edit=parsed_edit,
                    plan_mode=plan_mode,
                    plan_id=plan_id,
                    user_input_id=user_input_id,
                    created_at=created_at,
                    now_iso=now_iso,
                    anchor_task_id=effective_anchor_task_id,
                    protected_task_ids=protected_task_ids,
                )
                edit_log = "Plan Edit: applied high-level edit; touched tasks={touched}".format(
                    touched=sorted(touched_task_ids),
                )
                try:
                    run_memory.record_plan(
                        plan_id=plan_id,
                        plan_mode=plan_mode,
                        user_input_id=user_input_id,
                        tasks=updated_tasks,
                        plan_text=plan_text,
                        first_task_id=effective_anchor_task_id,
                    )
                except Exception:
                    pass
                if logger:
                    logger.log_foreground(edit_log)
                _invalidate_background_after_plan_edit(
                    shared_runtime_control=shared_runtime_control,
                    shared_background_results=shared_background_results,
                    shared_processing_tasks=shared_processing_tasks,
                    logger=logger,
                    updated_tasks=updated_tasks,
                )
                result_state = clear_turn_response(
                    {
                        **state,
                        "tasks": updated_tasks,
                        "current_plan_id": plan_id,
                        "background_results": {},
                        "user_inputs": user_inputs,
                        "next_action": {"type": "idle"},
                        "messages": state["messages"],
                    }
                )
                return apply_turn_response(
                    result_state,
                    response_text=PLAN_EDIT_PROGRESS_TEXT.get(
                        plan_mode,
                        "I have updated the plan.",
                    ),
                    response_type="progress",
                    source_event_type="plan_edited",
                )

            parsed_tasks, first_task_id = parse_planned_tasks_from_response(
                plan_text,
                plan_id=plan_id,
                user_input_id=user_input_id,
                created_at=created_at,
            )

            tasks = parsed_tasks

            if logger:
                logger.log_foreground(
                    f"Plan: Generated task plan with {len(tasks)} steps."
                )
                print(
                    f"{Colors.PLAN}Plan: Generated task plan with {len(tasks)} steps.{Colors.RESET}"
                )
                logger.log_foreground(f"Plan: Generated {len(tasks)} tasks")
                print(
                    f"{Colors.PLAN}Plan: Generated {len(tasks)} tasks{Colors.RESET}"
                )
                for t_id, task in tasks.items():
                    logger.log_foreground(
                        f"  Task {t_id}: {task.get('task_type')} - {task['description']}"
                    )
                    logger.log_foreground(
                        "  Plan Debug: task_id={task_id}, plan_id={plan_id}, user_input_id={user_input_id}, depends_on={depends_on}, task_type={task_type}".format(
                            task_id=t_id,
                            plan_id=task.get("plan_id"),
                            user_input_id=task.get("user_input_id"),
                            depends_on=task.get("depends_on", []),
                            task_type=task.get("task_type"),
                        )
                    )

            if logger:
                logger.log_foreground("Plan Details:")
            for t_id, task in tasks.items():
                if logger:
                    logger.log_foreground(
                        "  - Task ID: {task_id}, Type: {task_type}, Description: {task_desc}, Status: {status}".format(
                            task_id=task["task_id"],
                            task_type=task.get("task_type") or task.get("type"),
                            task_desc=task["description"],
                            status=task["status"],
                        )
                    )

            plan_event = {
                "event_id": new_structured_id("event"),
                "type": "plan_created",
                "source": "planner",
                "created_at": created_at,
                "user_input_id": user_input_id,
                "payload": {
                    "plan_id": plan_id,
                    "first_task_id": first_task_id,
                    "task_ids": sorted(tasks.keys()),
                },
            }

            if logger:
                logger.log_foreground(
                    "Plan Debug: emitted event_id={event_id}, type=plan_created, first_task_id={first_task_id}, task_ids={task_ids}".format(
                        event_id=plan_event["event_id"],
                        first_task_id=first_task_id,
                        task_ids=plan_event["payload"]["task_ids"],
                    )
                )
                print(
                    f"{Colors.PLAN}Plan Debug:{Colors.RESET} emitted "
                    f"event_id={plan_event['event_id']} first_task_id={first_task_id}"
                )

            if shared_runtime_control is not None:
                activate_runtime_plan(shared_runtime_control, plan_id=plan_id)
                if logger:
                    logger.log_foreground(
                        "Background Policy: start_policy={policy}, speculative_branches={branches}".format(
                            policy=shared_runtime_control.get(
                                "background_start_policy",
                                "after_first_navigation_dispatch",
                            ),
                            branches=bool(
                                shared_runtime_control.get(
                                    "speculative_branch_preanalysis",
                                    False,
                                )
                            ),
                        )
                    )

            try:
                run_memory.record_plan(
                    plan_id=plan_id,
                    plan_mode=plan_mode,
                    user_input_id=user_input_id,
                    tasks=tasks,
                    plan_text=plan_text,
                    first_task_id=first_task_id,
                )
            except Exception:
                pass
            _record_run_memory_event(
                run_memory,
                plan_event,
                stage="plan",
            )

            return clear_turn_response(
                {
                    **state,
                    "tasks": tasks,
                    "current_task_id": first_task_id,
                    "current_plan_id": plan_id,
                    "user_inputs": user_inputs,
                    "events": existing_events + [plan_event],
                    "next_action": {"type": "idle"},
                    "messages": state["messages"],
                }
            )

        except json.JSONDecodeError as e:
            error_msg = f"Failed to parse task plan: {str(e)}"
            _log_plan_error(
                logger,
                error_msg=error_msg,
                error=e,
                plan_mode=plan_mode,
                plan_id=plan_id,
                user_input_id=user_input_id,
                next_action=next_action,
                plan_text=plan_text,
            )

            return clear_turn_response(
                {
                    **state,
                    "error_message": error_msg,
                    "next_action": {"type": "idle"},
                    "messages": state["messages"] + [AIMessage(content=error_msg)],
                }
            )
        except PlanEditError as e:
            error_msg = f"Failed to apply plan edit: {str(e)}"
            _log_plan_error(
                logger,
                error_msg=error_msg,
                error=e,
                plan_mode=plan_mode,
                plan_id=plan_id,
                user_input_id=user_input_id,
                next_action=next_action,
                plan_text=plan_text,
            )

            return clear_turn_response(
                {
                    **state,
                    "error_message": error_msg,
                    "next_action": {"type": "idle"},
                    "messages": state["messages"] + [AIMessage(content=error_msg)],
                }
            )
        except Exception as e:
            error_msg = f"Failed to generate or apply task plan: {str(e)}"
            _log_plan_error(
                logger,
                error_msg=error_msg,
                error=e,
                plan_mode=plan_mode,
                plan_id=plan_id,
                user_input_id=user_input_id,
                next_action=next_action,
                plan_text=plan_text,
            )

            return clear_turn_response(
                {
                    **state,
                    "error_message": error_msg,
                    "next_action": {"type": "idle"},
                    "messages": state["messages"] + [AIMessage(content=error_msg)],
                }
            )

    return plan_node
