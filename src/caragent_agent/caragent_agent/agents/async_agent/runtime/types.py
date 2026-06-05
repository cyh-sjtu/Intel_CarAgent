"""Shared type definitions for the async agent runtime."""

from __future__ import annotations

from typing import Any, Literal, Optional, Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, NotRequired, TypedDict


def merge_background_results(left: dict, right: dict) -> dict:
    """Merge background results emitted by parallel workers into one mapping."""

    if not left:
        return right if right else {}
    if not right:
        return left
    return {**left, **right}


def merge_pending_controller_messages(left: list, right: Any) -> list:
    """Merge externally injected controller messages while allowing ingest to replace."""

    if isinstance(right, dict) and right.get("__replace__") is True:
        replacement = right.get("items", [])
        return list(replacement) if isinstance(replacement, list) else []

    base = list(left or [])
    incoming = list(right or []) if isinstance(right, list) else []
    if isinstance(right, list) and not incoming:
        return []
    if not incoming:
        return base

    seen_ids = {
        str(item.get("message_id") or item.get("id") or "").strip()
        for item in base
        if isinstance(item, dict)
    }
    for item in incoming:
        if not isinstance(item, dict):
            continue
        message_id = str(item.get("message_id") or item.get("id") or "").strip()
        if message_id and message_id in seen_ids:
            continue
        base.append(dict(item))
        if message_id:
            seen_ids.add(message_id)
    return base


TaskStatus = Literal[
    "pending",
    "in_progress",
    "running",
    "waiting",
    "completed",
    "failed",
    "cancelled",
]


TaskType = Literal["llm_action", "navigation_action", "decision"]


NavigationTargetType = Literal["keyframe", "position", "task_output"]


class NavigationTarget(TypedDict):
    """Structured target contract for deterministic navigation actions."""

    type: NavigationTargetType
    keyframe_id: NotRequired[int]
    position: NotRequired[list[float]]
    task_id: NotRequired[int]
    field: NotRequired[str]


EventType = Literal[
    "user_input_received",
    "plan_created",
    "task_completed",
    "task_failed",
    "task_waiting",
    "task_cancelled",
    "plan_cancelled",
    "navigation_arrived",
    "navigation_arrival_unmatched",
]

TurnResponseType = Literal["result", "progress", "error", "none"]


BackgroundAnalysisStatus = Literal["running", "completed", "failed"]

NavigationGroundingStage = Literal[
    "started",
    "memory_hit",
    "candidate_seed",
    "candidate_pack",
    "target_decision",
]


class TaskResultItem(TypedDict):
    """Append-only execution history entry for a task."""

    event_id: str
    summary: str
    created_at: str
    raw_output: NotRequired[str]
    decision: NotRequired[str]
    tool_name: NotRequired[str]


class UserInputItem(TypedDict):
    """Structured record for a single user request."""

    user_input_id: str
    message_id: str
    content: str
    created_at: str
    original_content: NotRequired[str]
    resolved_referent_id: NotRequired[str]
    resolved_referent_ids: NotRequired[list[str]]
    resolution_note: NotRequired[str]


class EventItem(TypedDict):
    """Structured event consumed by the orchestrator."""

    event_id: str
    type: EventType
    source: Literal["planner", "executor", "system", "user"]
    created_at: str
    payload: dict[str, Any]
    message_id: NotRequired[str]
    task_id: NotRequired[int]
    user_input_id: NotRequired[str]


class BackgroundAnalysisItem(TypedDict):
    """Structured background-analysis cache entry for one future task."""

    task_id: int
    task_description: str
    status: BackgroundAnalysisStatus
    started_at: str
    updated_at: str
    summary: NotRequired[str]
    notes: NotRequired[list[str]]
    tool_observations: NotRequired[list[str]]
    latest_tool_name: NotRequired[str]
    latest_tool_output: NotRequired[str]
    grounding_stage: NotRequired[NavigationGroundingStage]
    target_text: NotRequired[str]
    evidence_source: NotRequired[Literal["memory", "background", "foreground_tool"]]
    truth_mode: NotRequired[
        Literal["live_verified", "historical_grounded", "background_hypothesis"]
    ]
    candidate_keyframe_ids: NotRequired[list[int]]
    candidate_keyframes: NotRequired[list[dict[str, Any]]]
    recommended_keyframe_id: NotRequired[int]
    recommendation_confidence: NotRequired[float]
    recommendation_reason: NotRequired[str]
    final_output: NotRequired[str]
    completed_at: NotRequired[str]
    error: NotRequired[str]


class TurnResponseItem(TypedDict):
    """One user-visible reply emitted during a single graph turn."""

    response_id: str
    response_type: TurnResponseType
    response_text: str
    created_at: str
    source_event_type: NotRequired[str]
    task_id: NotRequired[int]


class NextAction(TypedDict):
    """Explicit routing decision produced by orchestrate."""

    type: Literal["idle", "plan", "execute"]
    task_id: NotRequired[int]
    user_input_id: NotRequired[str]
    plan_mode: NotRequired[
        Literal[
            "new_plan",
            "insert_after_current",
            "replan_future_after_current",
            "unsupported_edit",
        ]
    ]
    anchor_task_id: NotRequired[int]


class TaskItem(TypedDict):
    """Single task item in the plan."""

    task_id: int
    task_type: TaskType
    description: str
    status: TaskStatus
    next_task_id: Optional[int]
    condition: Optional[str]
    branches: Optional[dict[str, int]]
    target: NotRequired[NavigationTarget]
    # Runtime graph code still mirrors this coarse action/decision label for
    # UI and graph serialization; task_type remains the semantic source.
    type: NotRequired[Literal["action", "decision"]]
    routing_prompt: NotRequired[str]
    default_branch: NotRequired[str]
    plan_id: NotRequired[str]
    user_input_id: NotRequired[str]
    depends_on: NotRequired[list[int]]
    parent_task_id: NotRequired[int]
    wait_for_event: NotRequired[EventType]
    result: NotRequired[list[TaskResultItem]]
    created_at: NotRequired[str]
    updated_at: NotRequired[str]
    inserted: NotRequired[bool]
    terminal_reason: NotRequired[str]


class AsyncAgentState(TypedDict):
    """State for the async task-graph agent."""

    messages: Annotated[Sequence[BaseMessage], add_messages]
    tasks: dict[int, TaskItem]
    current_task_id: Optional[int]
    error_message: Optional[str]
    background_results: Annotated[
        dict[int, BackgroundAnalysisItem | str],
        merge_background_results,
    ]
    events: NotRequired[list[EventItem]]
    processed_event_ids: NotRequired[list[str]]
    next_action: NotRequired[NextAction]
    user_inputs: NotRequired[list[UserInputItem]]
    _pending_controller_messages: NotRequired[
        Annotated[list[dict[str, Any]], merge_pending_controller_messages]
    ]
    active_navigation: NotRequired[dict[str, Any]]
    pending_navigation: NotRequired[dict[str, Any]]
    current_plan_id: NotRequired[Optional[str]]
    turn_response_items: NotRequired[list[TurnResponseItem]]
    turn_response_type: NotRequired[Optional[TurnResponseType]]
    turn_response_text: NotRequired[Optional[str]]
    turn_response_id: NotRequired[Optional[str]]
    user_facing_response: NotRequired[Optional[str]]
    user_facing_response_id: NotRequired[Optional[str]]


class OrchestrateContext(TypedDict):
    """Shared runtime context passed to orchestrate event handlers."""

    state: AsyncAgentState
    tasks: dict[int, TaskItem]
    current_task_id: Optional[int]
    processed_event_ids: Sequence[str]
    next_action: Optional[NextAction]
    llm: BaseChatModel
    logger: Optional[Any]
    messages: Sequence[BaseMessage]
    events: Sequence[EventItem]
    shared_background_results: dict[int, BackgroundAnalysisItem | str]
    shared_processing_tasks: set[str]
    shared_runtime_control: dict[str, Any]
    run_memory: NotRequired[Optional[Any]]


__all__ = [
    "AsyncAgentState",
    "BackgroundAnalysisItem",
    "BackgroundAnalysisStatus",
    "EventItem",
    "EventType",
    "NavigationTarget",
    "NextAction",
    "NavigationGroundingStage",
    "OrchestrateContext",
    "TaskItem",
    "TaskResultItem",
    "TaskStatus",
    "TaskType",
    "TurnResponseItem",
    "TurnResponseType",
    "UserInputItem",
]
