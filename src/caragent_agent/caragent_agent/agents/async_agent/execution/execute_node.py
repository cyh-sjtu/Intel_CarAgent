"""Execute node for running current async-agent tasks."""

from __future__ import annotations

import json
import traceback
from typing import Any, Optional, Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.prebuilt.tool_node import ToolNode

from caragent_agent.agents.async_agent.execution.context import (
    build_background_reference,
    build_execution_guide,
    build_tool_catalog,
    is_navigation_action,
    prepare_context_bundle,
    task_depends_on_query_result,
    truncate_context_text,
)
from caragent_agent.agents.async_agent.execution.navigation_actions import (
    try_dispatch_structured_navigation_action as _try_dispatch_structured_navigation_action,
)
from caragent_agent.agents.async_agent.execution.support import (
    append_task_result,
    apply_user_facing_response,
    build_precision_support_tools,
    build_task_turn_response_type,
    build_task_user_facing_response,
    count_successful_navigation_commands,
    extract_tool_trace,
    find_tool_failure_message,
    issued_navigation_command,
    navigation_arrival_summary,
    navigation_waiting_summary,
)
from caragent_agent.agents.async_agent.orchestration.node_common import (
    _get_current_task,
    _record_run_memory_event,
    _strip_ignored_state_fields,
)
from caragent_agent.agents.async_agent.orchestration.runtime import new_structured_id, now_iso
from caragent_agent.agents.async_agent.planning.prompting import AGENT_PROMPTS
from caragent_agent.agents.async_agent.planning.task_graph import get_task_progress_context
from caragent_agent.agents.async_agent.runtime.control import (
    record_foreground_task,
    set_background_enabled,
)
from caragent_agent.agents.async_agent.runtime.console import Colors
from caragent_agent.agents.async_agent.runtime.types import (
    AsyncAgentState,
    BackgroundAnalysisItem,
    EventItem,
    TaskItem,
)
from caragent_agent.third_party.from_langgraph.react_agent import create_react_agent


NAVIGATION_TOOL_NAMES = {"go_to_keyframe"}


def _is_destination_resolver_for_navigation(
    current_task: Optional[TaskItem],
    tasks: dict[int, TaskItem],
) -> bool:
    """Return True when an llm_action feeds a following navigation_action target."""

    if not current_task or current_task.get("task_type") != "llm_action":
        return False
    try:
        current_task_id = int(current_task.get("task_id"))
    except Exception:
        return False

    for task in tasks.values():
        if not isinstance(task, dict):
            continue
        if str(task.get("task_type") or "").strip() != "navigation_action":
            continue
        target = task.get("target")
        if not isinstance(target, dict):
            continue
        if target.get("type") != "task_output" or target.get("field") != "destination":
            continue
        try:
            if int(target.get("task_id")) == current_task_id:
                return True
        except Exception:
            continue
    return False


def _try_complete_destination_resolver_from_background(
    current_task: Optional[TaskItem],
    *,
    tasks: dict[int, TaskItem],
    background_result: BackgroundAnalysisItem | str | None,
) -> Optional[dict[str, Any]]:
    """Use completed destination-resolver background output as the task result."""

    if not _is_destination_resolver_for_navigation(current_task, tasks):
        return None
    if not isinstance(background_result, dict):
        return None
    if str(background_result.get("status") or "").strip().lower() != "completed":
        return None

    raw_keyframe_id = background_result.get("recommended_keyframe_id")
    try:
        keyframe_id = int(raw_keyframe_id)
    except Exception:
        return None

    reason = truncate_context_text(
        background_result.get("recommendation_reason")
        or background_result.get("summary")
        or background_result.get("final_output"),
        limit=320,
    )
    destination = {"destination": {"type": "keyframe", "keyframe_id": keyframe_id}}
    destination_json = json.dumps(destination, ensure_ascii=False)
    summary = (
        f"Resolved destination from background preanalysis: keyframe {keyframe_id}."
    )
    if reason:
        summary += f" Reason: {reason}"
    final_ai_content = f"{summary}\n\n{destination_json}"
    synthetic_tool_payload = {
        "status": "ok",
        "summary": summary,
        "data": {
            "source": "background_preanalysis",
            "destination": destination["destination"],
            "recommended_keyframe_id": keyframe_id,
            "recommendation_confidence": background_result.get(
                "recommendation_confidence"
            ),
            "candidate_keyframe_ids": background_result.get("candidate_keyframe_ids"),
        },
    }

    return {
        "event_type": "task_completed",
        "summary": final_ai_content,
        "tool_name": "background_preanalysis",
        "tool_trace": {
            "tool_calls": [],
            "tool_results": [
                {
                    "name": "background_preanalysis",
                    "content": json.dumps(synthetic_tool_payload, ensure_ascii=False),
                    "tool_call_id": None,
                }
            ],
            "final_ai_content": final_ai_content,
        },
    }


def _build_navigation_memory_context_for_resolver(
    current_task: Optional[TaskItem],
    *,
    tasks: dict[int, TaskItem],
    run_memory: Optional[Any],
) -> str:
    """Inject a compact navigation table for destination resolvers."""

    if run_memory is None:
        return ""
    if not _is_destination_resolver_for_navigation(current_task, tasks):
        return ""
    try:
        table = run_memory.query_memory(
            scope="navigation",
            view="summary_table",
            query=str((current_task or {}).get("description") or ""),
            time="all",
            limit=50,
        )
    except Exception:
        return ""

    items = list(table.get("items") or [])
    if not items:
        return (
            "\n--- NAVIGATION MEMORY TABLE ---\n"
            "No prior navigation anchors are available for this session.\n"
            "Proceed to scene-memory search if the destination still needs resolving.\n"
            "--- END NAVIGATION MEMORY TABLE ---\n"
        )

    compact_rows: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        compact_rows.append(
            {
                "row_id": item.get("row_id"),
                "description": item.get("preview"),
                "keyframe_id": item.get("keyframe_id"),
                "position": item.get("position"),
                "order": item.get("order"),
                "time": item.get("time"),
            }
        )

    if not compact_rows:
        return ""

    return (
        "\n--- NAVIGATION MEMORY TABLE ---\n"
        "This destination resolver has already been given the compact navigation summary table.\n"
        "Use these rows only for visited-place reuse. If no row clearly matches, do not query other memory scopes; proceed to scene-memory search.\n"
        f"{json.dumps(compact_rows, ensure_ascii=False, indent=2)}\n"
        "--- END NAVIGATION MEMORY TABLE ---\n"
    )


def _tools_for_current_execute_task(
    current_task: Optional[TaskItem],
    execution_tools: Sequence[BaseTool],
) -> list[BaseTool]:
    """Apply the minimal tool boundary for the new task schema."""

    task_type = str((current_task or {}).get("task_type") or "").strip()
    if task_type == "navigation_action":
        return [
            tool
            for tool in execution_tools
            if str(getattr(tool, "name", "") or "").strip() in NAVIGATION_TOOL_NAMES
        ]
    return [
        tool
        for tool in execution_tools
        if str(getattr(tool, "name", "") or "").strip() not in NAVIGATION_TOOL_NAMES
    ]

def create_execute_node(
    llm: BaseChatModel,
    tools: Sequence[BaseTool],
    tool_node: ToolNode,
    shared_background_results: dict,
    shared_runtime_control: Optional[dict[str, Any]] = None,
    logger: Optional[Any] = None,
    run_memory: Optional[Any] = None,
):
    """Execute the current task using a ReAct-style agent with shared context."""

    del tool_node
    execution_tools = list(tools) + build_precision_support_tools(tools)

    def _build_execute_prompt(
        *,
        current_task: Optional[TaskItem],
        selected_execution_context_packet: dict[str, Any],
        background_context: str,
    ) -> str:
        """Assemble the fixed execute prompt from deterministic sections."""
        allowed_tools = _tools_for_current_execute_task(current_task, execution_tools)
        tool_catalog_text = build_tool_catalog(allowed_tools)
        current_task_payload: dict[str, Any] | str = (
            selected_execution_context_packet.get("current_task")
            if current_task is not None
            else "No active task"
        )
        execution_contract = build_execution_guide(
            current_task,
            selected_execution_context_packet,
        )
        prompt_lines = [
            "You are a sub-task executor.",
            "TASK CONTRACT:",
            "Solve only the current task. Do not implicitly complete future tasks.",
            "MINIMAL EXECUTION CONTEXT:",
            json.dumps(current_task_payload, ensure_ascii=False, indent=2),
            "EXECUTION CONTRACT:",
            execution_contract,
            "CONTINUITY CONTEXT:",
            json.dumps(selected_execution_context_packet, ensure_ascii=False, indent=2),
            "ALLOWED TOOLS:",
            tool_catalog_text,
        ]
        if is_navigation_action(current_task):
            prompt_lines.extend(
                [
                    "NAVIGATION RULES:",
                    "This is a navigation task.",
                    "After a successful navigation command, stop immediately and wait for arrival.",
                    "Do not answer downstream perception or reporting questions in this task.",
                ]
            )
        if selected_execution_context_packet.get("arrival_context") is not None:
            prompt_lines.append(
                "Arrival context is only a reference anchor for this task."
            )
        if selected_execution_context_packet.get("upstream_tasks"):
            prompt_lines.append(
                "Reuse upstream evidence when it already resolves the current task."
            )
        prompt_lines.extend(
            [
                "EXECUTION RULES:",
                "- Treat continuity context as small working memory, not full history.",
                "- If historical facts are needed, call query_memory with the narrowest scope and view.",
                "- Do not answer historical navigation, task, plan, conversation, or observation questions by guessing from prompt context.",
                "- When a deterministic helper tool is available for numeric computation, use it instead of mental arithmetic.",
            ]
        )
        if background_context:
            prompt_lines.append(background_context.rstrip())
        return "\n".join(prompt_lines) + "\n"

    def prepare_execute_inputs(state: AsyncAgentState) -> dict[str, Any]:
        """Collect current task, context bundle, guards, and prompt inputs."""

        messages = state.get("messages", [])
        tasks = state.get("tasks", {})
        current_task_id = state.get("current_task_id")
        current_task = _get_current_task(tasks, current_task_id)
        existing_events = list(state.get("events", []))

        candidate_background_result: BackgroundAnalysisItem | str | None = None
        if current_task and current_task.get("task_id", -1) >= 0:
            if not task_depends_on_query_result(current_task, tasks):
                candidate_background_result = shared_background_results.get(
                    current_task["task_id"]
                )
                if not candidate_background_result:
                    candidate_background_result = state.get("background_results", {}).get(
                        current_task["task_id"]
                    )
            else:
                if shared_background_results.pop(current_task["task_id"], None) is not None:
                    if logger:
                        logger.log_foreground(
                            "Execute: Ignoring background analysis for task {task_id} because its target depends on an upstream query result.".format(
                                task_id=current_task["task_id"],
                            )
                        )

        background_context = build_background_reference(candidate_background_result)
        prepared_context = prepare_context_bundle(
            state,
            current_task,
            run_memory=run_memory,
        )
        selected_execution_context_packet = dict(
            prepared_context.get("selected_execution_context_packet") or {}
        )
        plan_context = _build_execute_prompt(
            current_task=current_task,
            selected_execution_context_packet=selected_execution_context_packet,
            background_context=background_context,
        )
        navigation_memory_context = _build_navigation_memory_context_for_resolver(
            current_task,
            tasks=tasks,
            run_memory=run_memory,
        )
        if navigation_memory_context:
            plan_context = plan_context + navigation_memory_context

        return {
            "messages": messages,
            "tasks": tasks,
            "current_task_id": current_task_id,
            "current_task": current_task,
            "existing_events": existing_events,
            "selected_execution_context_packet": selected_execution_context_packet,
            "plan_context": plan_context,
            "background_result": candidate_background_result,
        }

    def run_execute_agent(execute_inputs: dict[str, Any]) -> dict[str, Any]:
        """Run structured navigation dispatch or a ReAct pass."""

        current_task = execute_inputs.get("current_task")
        allowed_tools = _tools_for_current_execute_task(current_task, execution_tools)
        agent_messages: list[BaseMessage] = []

        deterministic_outcome = _try_dispatch_structured_navigation_action(
            current_task,
            tasks=dict(execute_inputs.get("tasks") or {}),
            tools=allowed_tools,
        )
        if deterministic_outcome is not None:
            if logger and current_task:
                logger.log_foreground(
                    "Execute: Handled structured navigation_action for task {task_id} without ReAct planning.".format(
                        task_id=current_task.get("task_id")
                    )
                )
            print(
                f"{Colors.REACT}Execute: Handled structured navigation action{Colors.RESET}"
            )
            return {
                "agent_messages": agent_messages,
                "deterministic_outcome": deterministic_outcome,
                "execution_error": None,
            }
        if is_navigation_action(current_task):
            return {
                "agent_messages": agent_messages,
                "deterministic_outcome": {
                    "event_type": "task_failed",
                    "summary": "Structured navigation_action could not be dispatched from its target contract.",
                    "tool_name": None,
                    "tool_trace": {},
                },
                "execution_error": None,
            }

        background_outcome = _try_complete_destination_resolver_from_background(
            current_task,
            tasks=dict(execute_inputs.get("tasks") or {}),
            background_result=execute_inputs.get("background_result"),
        )
        if background_outcome is not None:
            if logger and current_task:
                logger.log_foreground(
                    "Execute: Used completed background preanalysis for destination resolver task {task_id}.".format(
                        task_id=current_task.get("task_id")
                    )
                )
            return {
                "agent_messages": agent_messages,
                "deterministic_outcome": background_outcome,
                "execution_error": None,
            }

        execution_error: Optional[Exception] = None
        react_agent = create_react_agent(
            model=llm,
            tools=allowed_tools,
            prompt=AGENT_PROMPTS.get("react_system", ""),
            logger=logger.log_foreground if logger else None,
        )

        def stream_executor_pass(system_prompt: str) -> None:
            react_input = {
                "messages": [
                    SystemMessage(content=system_prompt),
                ]
            }
            for chunk in react_agent.stream(react_input, stream_mode="values"):
                if "messages" not in chunk:
                    continue
                for msg in chunk["messages"]:
                    if msg in agent_messages:
                        continue
                    agent_messages.append(msg)
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        for tool_call in msg.tool_calls:
                            if logger:
                                logger.log_foreground(
                                    f"Execute: Tool Call to {tool_call['name']} with args: {tool_call['args']}"
                                )
                            print(
                                f"{Colors.TOOL}Tool Call:{Colors.RESET} {tool_call['name']} with args: {tool_call['args']}"
                            )
                    if isinstance(msg, ToolMessage):
                        if logger:
                            logger.log_foreground(
                                f"Execute: Tool Result from {msg.name}: {str(msg.content)[:200]}..."
                            )
                        print(
                            f"{Colors.TOOL}Tool Result:{Colors.RESET} {msg.name} returned: {str(msg.content)[:200]}..."
                        )

        try:
            stream_executor_pass(str(execute_inputs.get("plan_context") or ""))
        except Exception as exc:
            print(f"{Colors.REACT}Execute Error:{Colors.RESET} {str(exc)}")
            traceback.print_exc()
            execution_error = exc

        return {
            "agent_messages": agent_messages,
            "deterministic_outcome": None,
            "execution_error": execution_error,
        }

    def classify_execute_result(
        execute_inputs: dict[str, Any],
        execute_run: dict[str, Any],
    ) -> dict[str, Any]:
        """Normalize execute outputs into task_waiting/task_completed/task_failed."""

        current_task = execute_inputs.get("current_task")
        deterministic_outcome = execute_run.get("deterministic_outcome")
        if deterministic_outcome is not None:
            tool_trace = dict(deterministic_outcome.get("tool_trace", {}))
            navigation_command_count = count_successful_navigation_commands(
                tool_trace,
                navigation_tool_names=NAVIGATION_TOOL_NAMES,
            )
            event_type = "task_waiting" if navigation_command_count > 0 else "task_completed"
            if deterministic_outcome.get("event_type"):
                event_type = str(deterministic_outcome.get("event_type"))
            return {
                "agent_messages": list(execute_run.get("agent_messages") or []),
                "tool_trace": tool_trace,
                "event_type": event_type,
                "summary": str(
                    deterministic_outcome.get("summary")
                    or navigation_arrival_summary(current_task)
                ),
                "primary_tool_name": deterministic_outcome.get("tool_name"),
                "navigation_command_count": navigation_command_count,
            }

        agent_messages = list(execute_run.get("agent_messages") or [])
        execution_error = execute_run.get("execution_error")
        tool_trace = extract_tool_trace(agent_messages)
        failure_summary = find_tool_failure_message(tool_trace)
        navigation_command_count = count_successful_navigation_commands(
            tool_trace,
            navigation_tool_names=NAVIGATION_TOOL_NAMES,
        )
        final_ai_content = str(tool_trace.get("final_ai_content") or "").strip()

        if execution_error is not None:
            event_type = "task_failed"
            summary = f"Task execution failed with exception: {str(execution_error)}"
        elif navigation_command_count > 1:
            event_type = "task_failed"
            summary = (
                "Task execution issued multiple navigation commands in a single task. "
                "Each task must resolve to exactly one destination before waiting for arrival."
            )
        elif issued_navigation_command(
            tool_trace,
            navigation_tool_names=NAVIGATION_TOOL_NAMES,
        ):
            event_type = "task_waiting"
            summary = navigation_waiting_summary(current_task)
        elif failure_summary:
            event_type = "task_failed"
            summary = failure_summary
        else:
            event_type = "task_completed"
            summary = (
                final_ai_content
                or "Task completed successfully."
            )

        primary_tool_name = None
        if tool_trace.get("tool_calls"):
            primary_tool_name = tool_trace["tool_calls"][-1].get("name")
        elif tool_trace.get("tool_results"):
            primary_tool_name = tool_trace["tool_results"][-1].get("name")

        return {
            "agent_messages": agent_messages,
            "tool_trace": tool_trace,
            "event_type": event_type,
            "summary": summary,
            "primary_tool_name": primary_tool_name,
            "navigation_command_count": navigation_command_count,
        }

    def execute_node(state: AsyncAgentState) -> AsyncAgentState:
        """Run the executor for the current task and emit a structured execution event."""

        state = _strip_ignored_state_fields(state)
        if logger:
            logger.log_foreground("Execute: Starting task execution")
            print(f"{Colors.REACT}Execute: Starting task execution{Colors.RESET}")
        messages = state.get("messages", [])
        tasks = state.get("tasks", {})
        current_task_id = state.get("current_task_id")
        current_task = _get_current_task(tasks, current_task_id)
        existing_events = list(state.get("events", []))
        execution_error: Optional[Exception] = None

        if current_task:
            if (
                shared_runtime_control is not None
                and current_task_id is not None
                and current_task.get("plan_id") == state.get("current_plan_id")
            ):
                record_foreground_task(shared_runtime_control, int(current_task_id))
            progress_context = get_task_progress_context(
                tasks,
                current_task_id=current_task_id,
                current_plan_id=state.get("current_plan_id"),
            )
            task_label = (
                f"task #{current_task_id}"
                if current_task_id is not None and current_task_id >= 0
                else "runtime task"
            )
            progress_label = ""
            if progress_context is not None:
                progress_label = (
                    f" (step {progress_context['position']}/{progress_context['total']})"
                )
            if logger:
                logger.log_foreground(
                    f"Execute: Executing {task_label}{progress_label}: {current_task['description']}"
                )
                print(
                    f"{Colors.REACT}Execute: Executing {task_label}{progress_label}:{Colors.RESET} {current_task['description']}"
                )
        else:
            user_message = next(
                (m for m in reversed(messages) if isinstance(m, HumanMessage)), None
            )
            if not user_message:
                return state
            if logger:
                logger.log_foreground(
                    f"Execute: Executing simple task: {user_message.content[:100]}..."
                )
                print(
                    f"{Colors.REACT}Execute: Executing simple task:{Colors.RESET} {user_message.content[:100]}..."
                )

        try:
            execute_inputs = prepare_execute_inputs(state)
        except Exception:
            execute_inputs = {
                "messages": messages,
                "tasks": tasks,
                "current_task_id": current_task_id,
                "current_task": current_task,
                "existing_events": existing_events,
                "plan_context": "Task plan context unavailable.",
            }

        task_ref = None
        if current_task_id is not None and current_task_id in tasks:
            task_ref = tasks[current_task_id]
            task_ref["status"] = "running"
            task_ref["updated_at"] = now_iso()

        execute_run = run_execute_agent(execute_inputs)
        classified_result = classify_execute_result(execute_inputs, execute_run)
        agent_messages = list(classified_result.get("agent_messages") or [])
        tool_trace = dict(classified_result.get("tool_trace") or {})
        navigation_command_count = int(
            classified_result.get("navigation_command_count") or 0
        )
        if navigation_command_count > 0 and shared_runtime_control is not None:
            set_background_enabled(shared_runtime_control, True)

        emitted_event_id = new_structured_id("event")
        event_type = str(classified_result.get("event_type") or "task_completed")
        summary = str(
            classified_result.get("summary") or "Task completed successfully."
        )
        primary_tool_name = classified_result.get("primary_tool_name")

        raw_output = json.dumps(tool_trace, ensure_ascii=False)
        user_facing_response = build_task_user_facing_response(
            current_task,
            event_type=event_type,
            summary=summary,
        )
        turn_response_type = build_task_turn_response_type(
            current_task,
            event_type=event_type,
            summary=summary,
        )

        if task_ref is not None:
            if event_type == "task_waiting":
                task_ref["status"] = "waiting"
                task_ref["wait_for_event"] = "navigation_arrived"
            elif event_type == "task_failed":
                task_ref["status"] = "failed"
                task_ref["terminal_reason"] = summary
            else:
                task_ref["status"] = "completed"
            append_task_result(
                task_ref,
                event_id=emitted_event_id,
                summary=summary,
                raw_output=raw_output,
                tool_name=primary_tool_name,
            )
            try:
                run_memory.record_task_result(
                    task=task_ref,
                    event_type=event_type,
                    summary=summary,
                    tool_trace=tool_trace,
                )
                run_memory.record_tool_trace(
                    task=task_ref,
                    tool_trace=tool_trace,
                )
            except Exception:
                pass

        if logger:
            logger.log_foreground(
                "Execute Debug: classified event_type={event_type}, task_id={task_id}, tool_calls={tool_calls}, navigation_command_count={navigation_command_count}, summary={summary}".format(
                    event_type=event_type,
                    task_id=current_task_id if task_ref is not None else -1,
                    tool_calls=[call.get("name") for call in tool_trace.get("tool_calls", [])],
                    navigation_command_count=navigation_command_count,
                    summary=summary,
                )
            )
            print(
                f"{Colors.REACT}Execute Debug:{Colors.RESET} "
                f"event_type={event_type} task_id={current_task_id if task_ref is not None else -1}"
            )

        execution_event: EventItem = {
            "event_id": emitted_event_id,
            "type": event_type,
            "source": "executor",
            "created_at": now_iso(),
            "task_id": current_task_id if task_ref is not None else -1,
            "payload": {
                "summary": summary,
                "tool_name": primary_tool_name,
            },
        }
        if user_facing_response:
            execution_event["payload"]["user_facing_response"] = user_facing_response
        if turn_response_type:
            execution_event["payload"]["turn_response_type"] = turn_response_type
        if task_ref and task_ref.get("user_input_id"):
            execution_event["user_input_id"] = task_ref["user_input_id"]
        _record_run_memory_event(
            run_memory,
            execution_event,
            stage="execute",
        )

        result_state: AsyncAgentState = {
            **state,
            "tasks": tasks,
            "events": existing_events + [execution_event],
            "next_action": {"type": "idle"},
            "messages": state["messages"] + agent_messages,
        }
        return apply_user_facing_response(
            result_state,
            user_facing_response,
            response_type=turn_response_type,
        )

    return execute_node
