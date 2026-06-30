"""Planning and decision-support helpers for the async agent node layer."""

from __future__ import annotations

import json
from typing import Any, Optional, Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from ..execution.context import (
    summarize_task_for_execution_context,
)
from .prompting import AGENT_PROMPTS
from .task_graph import extract_json_block
from ..runtime.types import (
    AsyncAgentState,
    EventItem,
    TaskItem,
)


def collect_plan_scope_snapshot(
    tasks: dict[int, TaskItem],
    *,
    plan_id: Optional[str],
    current_task_id: Optional[int],
) -> list[dict[str, Any]]:
    """Build a compact plan snapshot for relation classification and editing."""

    snapshot: list[dict[str, Any]] = []
    for task_id in sorted(tasks):
        task = tasks[task_id]
        if plan_id and task.get("plan_id") != plan_id:
            continue
        latest_result = (task.get("result") or [])[-1] if task.get("result") else None
        snapshot.append(
            {
                "task_id": task_id,
                "description": task.get("description"),
                "task_type": task.get("task_type") or task.get("type"),
                "status": task.get("status"),
                "is_current": task_id == current_task_id,
                "next_task_id": task.get("next_task_id"),
                "branches": task.get("branches"),
                "image_refs": task.get("image_refs", []),
                "wait_for_event": task.get("wait_for_event"),
                "latest_result_summary": latest_result.get("summary")
                if latest_result
                else None,
            }
        )
    return snapshot


def classify_current_plan_relation(
    *,
    state: AsyncAgentState,
    tasks: dict[int, TaskItem],
    current_task_id: Optional[int],
    llm: BaseChatModel,
    logger: Optional[Any],
    user_input: str,
    user_input_id: Optional[str],
    run_memory: Optional[Any] = None,
) -> dict[str, Any]:
    """Classify how a new user request should interact with the active plan."""

    current_plan_id = state.get("current_plan_id")
    if current_plan_id is None or current_task_id is None or current_task_id not in tasks:
        return {"action": "new_plan", "reason": "No active plan to edit."}

    current_task = tasks.get(current_task_id)
    plan_snapshot = collect_plan_scope_snapshot(
        tasks,
        plan_id=current_plan_id,
        current_task_id=current_task_id,
    )
    recent_events = [
        {
            "event_id": event.get("event_id"),
            "type": event.get("type"),
            "task_id": event.get("task_id"),
            "payload": event.get("payload", {}),
        }
        for event in list(state.get("events", []))[-8:]
    ]
    classifier_prompt = (
        "You classify how a new user request should modify the current task plan.\n"
        "Choose exactly one action from:\n"
        "- `new_plan`: discard the current plan and create a brand-new one.\n"
        "- `insert_after_current`: keep the current task and existing plan, but insert new future work immediately after the current task.\n"
        "- `replan_future_after_current`: keep the current task, but regenerate all future tasks after it.\n"
        "- `unsupported_edit`: refuse the edit and leave the current plan/task unchanged.\n\n"
        "Policy:\n"
        "- Use `new_plan` only when the user clearly wants to abandon/replace the active mission, stop the current sequence, or start a separate mission now.\n"
        "- Use `insert_after_current` only for simple add-on requests like 'after the current task, also do X, then continue' where the old future continuation should remain.\n"
        "- Use `replan_future_after_current` for deleting, replacing, reordering, branch changes, or any broader change to future work after the current task.\n"
        "- Use `unsupported_edit` when the request asks to mutate the current in-flight task without allowing interruption, edit completed work, or cannot be turned into a safe future replan.\n"
        "- If unsure whether the current task would be changed, choose `unsupported_edit` instead of guessing.\n\n"
        "Return JSON only with this schema:\n"
        '{"action":"<one action>","reason":"<short reason>"}'
        "\n\n"
        f"New user request:\n{user_input}\n\n"
        f"User input id:\n{user_input_id}\n\n"
        f"Current task:\n{json.dumps(summarize_task_for_execution_context(current_task), ensure_ascii=False, indent=2)}\n\n"
        f"Current plan snapshot:\n{json.dumps(plan_snapshot, ensure_ascii=False, indent=2)}\n\n"
        f"Recent events:\n{json.dumps(recent_events, ensure_ascii=False, indent=2)}"
    )

    result: dict[str, Any]
    try:
        response = llm.invoke([HumanMessage(content=classifier_prompt)])
        parsed = json.loads(extract_json_block(str(response.content).strip()))
        action = parsed.get("action")
        if action in {
            "new_plan",
            "insert_after_current",
            "replan_future_after_current",
            "unsupported_edit",
        }:
            result = {
                "action": action,
                "reason": str(parsed.get("reason") or "").strip(),
            }
        else:
            result = {
                "action": "new_plan",
                "reason": "Fallback to a fresh plan because relation classification returned an invalid action.",
            }
    except Exception as exc:
        if logger:
            logger.log_foreground(
                f"Orchestrate: plan-relation classification failed, falling back to new_plan: {exc}"
            )
        result = {
            "action": "new_plan",
            "reason": "Fallback to a fresh plan because relation classification was unavailable.",
        }

    return result


def build_plan_editing_messages(
    *,
    user_request: str,
    current_task: Optional[TaskItem],
    tasks: dict[int, TaskItem],
    current_plan_id: Optional[str],
    plan_mode: str,
) -> list[BaseMessage]:
    """Build planner messages for dynamic plan edits."""

    plan_snapshot = collect_plan_scope_snapshot(
        tasks,
        plan_id=current_plan_id,
        current_task_id=current_task.get("task_id") if current_task else None,
    )
    edit_instruction = {
        "insert_after_current": (
            "Return only the new subgraph that should be inserted immediately after the current task. The runtime will reconnect it to the original continuation, so do not repeat old future tasks."
        ),
        "replan_future_after_current": (
            "Return the complete new future subgraph that should run after the current task. The runtime will remove the old future suffix after the current task and connect this new future subgraph, so include every future task that should still happen."
        ),
    }.get(
        plan_mode,
        "Return only the new or replacement task subgraph.",
    )

    return [
        SystemMessage(
            content=(
                AGENT_PROMPTS.get("plan_system", "")
                + "\n\nPLAN EDITING MODE:\n"
                + f"- Mode: {plan_mode}\n"
                + f"- {edit_instruction}\n"
                + "- Do not return low-level graph operations such as add_edge, remove_edge, set_sequence, or delete_nodes.\n"
                + "- Return JSON only with this schema: {\"edit_type\":\"insert_after_current|replan_future_after_current\",\"rationale\":\"short reason\",\"resume\":\"original_next|none\",\"tasks\":[...]}.\n"
                + "- `tasks` contains only the new inserted/future subgraph, not completed tasks and not the current task.\n"
                + "- The current task is protected. Do not try to edit, cancel, replace, or include it in the returned tasks.\n"
                + "- Task format: {\"task_id\":101,\"description\":\"...\",\"task_type\":\"llm_action\",\"next_task_id\":102,\"depends_on\":[],\"image_refs\":[\"latest\"]}.\n"
                + "- Supported task_type values: llm_action, navigation_action, decision.\n"
                + "- Include image_refs only on tasks that must inspect an attached user image.\n"
                + "- For navigation_action, include a structured target when the destination is already resolved: {\"type\":\"keyframe\",\"keyframe_id\":5}, {\"type\":\"position\",\"position\":[x,y,z]}, or {\"type\":\"task_output\",\"task_id\":101,\"field\":\"destination\"}.\n"
                + "- For insert_after_current, new task ids only need to be unique inside `tasks`; the runtime will remap them. Use those local ids consistently in next_task_id, branches, depends_on, and target.task_id.\n"
                + "- For replan_future_after_current, preserve existing future task ids from the plan snapshot for tasks that still represent the same work. Use fresh ids only for genuinely new future tasks.\n"
                + "- For insert_after_current, return just the inserted subgraph and use `resume:\"original_next\"` unless the user explicitly wants to terminate the old continuation; do not copy the original continuation into `tasks`.\n"
                + "- For replan_future_after_current, return the complete future plan after the current task and use `resume:\"none\"`; include old future tasks only when they still satisfy the updated user request, and omit tasks the user removed or replaced.\n"
                + "- For branch changes under replan_future_after_current, return a complete coherent future branch structure from the current task onward instead of a branch patch.\n"
                + "- Do not add a terminal task whose only purpose is to report, respond, or tell the user the final answer.\n"
            )
        ),
        HumanMessage(
            content=(
                f"Current task:\n{json.dumps(summarize_task_for_execution_context(current_task), ensure_ascii=False, indent=2)}\n\n"
                f"Current plan snapshot:\n{json.dumps(plan_snapshot, ensure_ascii=False, indent=2)}\n\n"
                f"Plan edit request:\n{user_request}"
            )
        ),
    ]


def resolve_default_branch(
    decision_task: TaskItem,
    branches: dict[str, int],
) -> Optional[str]:
    """Choose the fallback branch for a decision task."""

    configured_default = decision_task.get("default_branch")
    if configured_default in branches:
        return configured_default
    if "no" in branches:
        return "no"
    if branches:
        return next(iter(branches))
    return None


def _normalize_task_ids(task_ids: Any) -> list[int]:
    """Return stable integer task ids while preserving order."""

    normalized_ids: list[int] = []
    raw_task_ids = task_ids if isinstance(task_ids, (list, tuple, set)) else [task_ids]
    for task_id in raw_task_ids:
        try:
            normalized_task_id = int(task_id)
        except (TypeError, ValueError):
            continue
        if normalized_task_id not in normalized_ids:
            normalized_ids.append(normalized_task_id)
    return normalized_ids


def _expand_dependency_ids(
    tasks: dict[int, TaskItem],
    direct_dependency_ids: Sequence[int],
) -> list[int]:
    """Include transitive upstream tasks so compound decisions keep all signals."""

    expanded_ids: list[int] = []
    visiting: set[int] = set()

    def _visit(task_id: int) -> None:
        if task_id in visiting:
            return
        visiting.add(task_id)
        for upstream_id in _normalize_task_ids(tasks.get(task_id, {}).get("depends_on", [])):
            _visit(upstream_id)
        visiting.discard(task_id)
        if task_id not in expanded_ids:
            expanded_ids.append(task_id)

    for dependency_id in direct_dependency_ids:
        _visit(dependency_id)

    return expanded_ids


def build_decision_context(
    decision_task: TaskItem,
    current_state_tasks: dict[int, TaskItem],
    events: Sequence[EventItem],
    messages: Sequence[BaseMessage],
    *,
    state: Optional[AsyncAgentState] = None,
    run_memory: Optional[Any] = None,
    logger: Optional[Any] = None,
) -> dict[str, Any]:
    """Build the minimal current-plan context needed to route one decision task."""

    del messages, state, run_memory, logger
    decision_task_id_raw = decision_task.get("task_id")
    try:
        decision_task_id = (
            int(decision_task_id_raw) if decision_task_id_raw is not None else None
        )
    except (TypeError, ValueError):
        decision_task_id = None

    direct_dependency_ids = _normalize_task_ids(decision_task.get("depends_on", []))
    if not direct_dependency_ids:
        direct_dependency_ids = [
            task_id
            for task_id, task in sorted(current_state_tasks.items())
            if task_id != decision_task_id
            and task.get("status") in {"completed", "waiting", "failed"}
        ]
    dependency_ids = _expand_dependency_ids(current_state_tasks, direct_dependency_ids)

    upstream_tasks = []
    for task_id in dependency_ids:
        task = current_state_tasks.get(task_id)
        if not task:
            continue
        upstream_tasks.append(
            {
                "task_id": task_id,
                "dependency_role": "direct" if task_id in direct_dependency_ids else "transitive",
                "description": task.get("description"),
                "status": task.get("status"),
                "latest_result": (task.get("result") or [])[-1]
                if task.get("result")
                else None,
            }
        )

    branches = decision_task.get("branches") or {}
    branch_options = []
    for branch_name, branch_target in branches.items():
        target_task = current_state_tasks.get(branch_target)
        branch_options.append(
            {
                "branch": branch_name,
                "task_id": branch_target,
                "target_description": target_task.get("description")
                if target_task
                else None,
            }
        )

    related_events = []
    dependency_id_set = set(dependency_ids)
    for event in events:
        try:
            event_task_id = int(event.get("task_id")) if event.get("task_id") is not None else None
        except (TypeError, ValueError):
            event_task_id = None
        if event_task_id not in dependency_id_set and event_task_id != decision_task_id:
            continue
        related_events.append(
            {
                "event_id": event.get("event_id"),
                "type": event.get("type"),
                "task_id": event.get("task_id"),
                "payload": event.get("payload", {}),
            }
        )

    return {
        "decision_task": {
            "task_id": decision_task.get("task_id"),
            "description": decision_task.get("description"),
            "condition": decision_task.get("condition"),
            "routing_prompt": decision_task.get("routing_prompt"),
            "default_branch": decision_task.get("default_branch"),
            "direct_depends_on": direct_dependency_ids,
            "context_depends_on": dependency_ids,
        },
        "branch_options": branch_options,
        "upstream_tasks": upstream_tasks,
        "related_events": related_events[-6:],
    }


def parse_decision_choice(
    raw_content: str,
    branches: dict[str, int],
    default_branch: Optional[str],
) -> tuple[Optional[str], str]:
    """Parse an LLM routing response into a valid branch choice."""

    cleaned = extract_json_block(raw_content)
    parsed_reason = ""

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            branch = parsed.get("branch")
            if isinstance(branch, str) and branch in branches:
                parsed_reason = str(parsed.get("reason") or "").strip()
                return branch, parsed_reason
    except Exception:
        pass

    stripped = cleaned.strip()
    if stripped in branches:
        return stripped, ""

    lowered = stripped.lower()
    normalized_map = {branch.lower(): branch for branch in branches}
    if lowered in normalized_map:
        return normalized_map[lowered], ""

    if default_branch in branches:
        return default_branch, f"Invalid branch response: {raw_content}"

    return None, f"Invalid branch response: {raw_content}"
