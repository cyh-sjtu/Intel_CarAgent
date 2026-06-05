"""Background processing support: result storage, coordination, and analysis."""

from __future__ import annotations

import json
import re
import threading
import time
from typing import Any, Callable, Optional, Sequence

from caragent_agent.agents.async_agent.execution.context import (
    should_preanalyze_future_task,
    truncate_context_text,
)
from caragent_agent.agents.async_agent.execution.support import stringify_tool_content

from caragent_agent.agents.async_agent.runtime.control import (
    get_background_claim_lock,
    task_processing_key,
)
from caragent_agent.agents.async_agent.orchestration.runtime import now_iso
from caragent_agent.agents.async_agent.planning.task_graph import (
    collect_ordered_task_ids_for_plan,
    collect_plan_root_task_ids,
)
from caragent_agent.agents.async_agent.runtime.types import (
    AsyncAgentState,
    BackgroundAnalysisItem,
    NavigationGroundingStage,
    TaskItem,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END
from caragent_agent.agents.async_agent.execution.tool_results import (
    dedupe_ints as _dedupe_ints,
    extract_keyframe_ids_from_payload as _extract_keyframe_ids_from_payload,
    merge_candidate_keyframes as _merge_candidate_keyframes,
    parse_json_like_payload as _parse_json_like_payload,
)
from caragent_agent.agents.async_agent.planning.prompting import AGENT_PROMPTS
from caragent_agent.third_party.from_langgraph.react_agent import create_react_agent
from caragent_agent.utils.llm_handler import UnifiedLLMClient


class BackgroundYieldToForeground(Exception):
    """Internal control-flow signal for yielding claimed background work."""


def _speculative_branch_preanalysis_from_control(
    shared_runtime_control: Optional[dict[str, Any]],
) -> bool:
    """Return whether background may preanalyze unresolved decision branches."""

    if shared_runtime_control is None:
        return False
    return bool(shared_runtime_control.get("speculative_branch_preanalysis", False))


def _latest_foreground_task_id(
    state: AsyncAgentState,
    shared_runtime_control: Optional[dict[str, Any]],
) -> Optional[int]:
    """Return the newest foreground task id known to local or shared state."""

    current_task_id = state.get("current_task_id")
    if shared_runtime_control is None:
        return current_task_id
    raw_value = shared_runtime_control.get("latest_foreground_task_id")
    if raw_value is None:
        raw_value = shared_runtime_control.get("foreground_current_task_id")
    try:
        return int(raw_value) if raw_value is not None else current_task_id
    except Exception:
        return current_task_id


def _foreground_has_claimed_task(
    task_id: int,
    state: AsyncAgentState,
    shared_runtime_control: Optional[dict[str, Any]],
) -> bool:
    """Return True when foreground has started this task or passed it in plan order."""

    latest_task_id = _latest_foreground_task_id(state, shared_runtime_control)
    if latest_task_id is not None:
        if task_id == latest_task_id:
            return True
        ordered_task_ids = collect_ordered_task_ids_for_plan(
            state.get("tasks", {}),
            plan_id=state.get("current_plan_id"),
        )
        if task_id in ordered_task_ids and latest_task_id in ordered_task_ids:
            return ordered_task_ids.index(task_id) <= ordered_task_ids.index(
                latest_task_id
            )
    if shared_runtime_control is None:
        return False
    started_tasks = shared_runtime_control.get("foreground_started_task_ids")
    return hasattr(started_tasks, "__contains__") and task_id in started_tasks


def _resolved_decision_branch_records(
    shared_runtime_control: Optional[dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    """Return normalized decision-id to selected branch records."""

    if shared_runtime_control is None:
        return {}
    resolved: dict[int, dict[str, Any]] = {}
    raw_items = shared_runtime_control.get("resolved_decision_branches")
    if not isinstance(raw_items, dict):
        return resolved
    for raw_decision_id, raw_record in raw_items.items():
        try:
            decision_id = int(raw_decision_id)
        except Exception:
            continue
        if isinstance(raw_record, dict):
            raw_target_id = raw_record.get("target_task_id")
            branch_label = raw_record.get("branch")
            record_plan_id = raw_record.get("plan_id")
        else:
            raw_target_id = raw_record
            branch_label = None
            record_plan_id = None
        try:
            target_id = int(raw_target_id) if raw_target_id is not None else None
        except Exception:
            target_id = None
        if target_id is not None:
            resolved[decision_id] = {
                "branch": branch_label,
                "target_task_id": target_id,
                "plan_id": record_plan_id,
            }
    return resolved


def _resolved_branch_target_is_valid(
    decision_task_id: int,
    selected_record: Optional[dict[str, Any]],
    tasks: dict[int, TaskItem],
    *,
    current_plan_id: Optional[str],
) -> bool:
    """Return True when a remembered branch target still matches the live task graph."""

    if not isinstance(selected_record, dict):
        return False
    selected_target = selected_record.get("target_task_id")
    if selected_target is None or selected_target not in tasks:
        return False
    decision_task = tasks.get(decision_task_id)
    if not decision_task or decision_task.get("type") != "decision":
        return False
    task_plan_id = decision_task.get("plan_id")
    if current_plan_id is not None and task_plan_id != current_plan_id:
        return False
    record_plan_id = selected_record.get("plan_id")
    if record_plan_id is not None and task_plan_id != record_plan_id:
        return False
    branches = decision_task.get("branches") or {}
    branch_label = selected_record.get("branch")
    if branch_label is not None:
        try:
            return int(branches.get(str(branch_label))) == int(selected_target)
        except Exception:
            return False
    for raw_target in branches.values():
        try:
            if int(raw_target) == int(selected_target):
                return True
        except Exception:
            continue
    return False


def _selected_path_task_ids_for_background(
    tasks: dict[int, TaskItem],
    *,
    state: AsyncAgentState,
    shared_runtime_control: Optional[dict[str, Any]],
    allow_speculative_branches: bool,
) -> tuple[list[int], bool]:
    """Walk future task ids and report whether an unresolved decision blocks progress."""

    current_plan_id = state.get("current_plan_id")
    foreground_task_id = _latest_foreground_task_id(state, shared_runtime_control)
    resolved_records = _resolved_decision_branch_records(shared_runtime_control)
    visited: set[int] = set()
    ordered: list[int] = []
    stopped_on_unresolved_decision = False

    def push_task_path(start_task_id: Optional[int]) -> None:
        nonlocal stopped_on_unresolved_decision
        task_id = start_task_id
        while task_id is not None and task_id in tasks and task_id not in visited:
            task = tasks[task_id]
            if current_plan_id is not None and task.get("plan_id") != current_plan_id:
                return
            visited.add(task_id)
            ordered.append(task_id)

            branches = task.get("branches") or {}
            if task.get("type") == "decision" and branches:
                selected_record = resolved_records.get(task_id)
                if _resolved_branch_target_is_valid(
                    task_id,
                    selected_record,
                    tasks,
                    current_plan_id=current_plan_id,
                ):
                    task_id = selected_record["target_task_id"]
                    continue
                if allow_speculative_branches:
                    for branch_target in sorted(set(branches.values())):
                        push_task_path(branch_target)
                else:
                    stopped_on_unresolved_decision = True
                return

            task_id = task.get("next_task_id")

    start_task_ids: list[Optional[int]] = []
    if foreground_task_id is not None and foreground_task_id in tasks:
        foreground_task = tasks[foreground_task_id]
        if foreground_task.get("type") == "decision":
            selected_record = resolved_records.get(foreground_task_id)
            start_task_ids.append(
                selected_record["target_task_id"]
                if _resolved_branch_target_is_valid(
                    foreground_task_id,
                    selected_record,
                    tasks,
                    current_plan_id=current_plan_id,
                )
                else None
            )
        else:
            start_task_ids.append(foreground_task.get("next_task_id"))
    else:
        start_task_ids.extend(collect_plan_root_task_ids(tasks, plan_id=current_plan_id))

    for start_task_id in start_task_ids:
        push_task_path(start_task_id)
    if not ordered and not stopped_on_unresolved_decision:
        for root_task_id in collect_plan_root_task_ids(tasks, plan_id=current_plan_id):
            push_task_path(root_task_id)
    return ordered, stopped_on_unresolved_decision


def select_background_target_task(
    state: AsyncAgentState,
    *,
    worker_id: int,
    total_workers: int,
    shared_background_results: dict,
    shared_processing_tasks: set[str],
    shared_runtime_control: Optional[dict[str, Any]],
) -> Optional[TaskItem]:
    """Choose the next future background task using path-aware shared-queue policy."""

    if len(shared_processing_tasks) >= max(1, int(total_workers)):
        return None

    tasks = state.get("tasks", {})
    current_plan_id = state.get("current_plan_id")
    allow_speculative_branches = _speculative_branch_preanalysis_from_control(
        shared_runtime_control
    )
    path_task_ids, _ = _selected_path_task_ids_for_background(
        tasks,
        state=state,
        shared_runtime_control=shared_runtime_control,
        allow_speculative_branches=allow_speculative_branches,
    )
    if not path_task_ids and allow_speculative_branches:
        path_task_ids = sorted(tasks)
    plan_order = collect_ordered_task_ids_for_plan(tasks, plan_id=current_plan_id)
    plan_order_index = {task_id: index for index, task_id in enumerate(plan_order)}
    path_task_ids = sorted(
        dict.fromkeys(path_task_ids),
        key=lambda task_id: plan_order_index.get(task_id, len(plan_order_index)),
    )

    eligible_task_ids: list[int] = []
    for t_id in path_task_ids:
        task = tasks[t_id]
        task_plan_id = task.get("plan_id")
        processing_key = task_processing_key(t_id, task_plan_id)
        if (
            not _foreground_has_claimed_task(t_id, state, shared_runtime_control)
            and task["type"] == "action"
            and should_preanalyze_future_task(task, tasks)
            and task_plan_id == current_plan_id
            and t_id not in shared_background_results
            and processing_key not in shared_processing_tasks
        ):
            eligible_task_ids.append(t_id)

    for t_id in eligible_task_ids:
        return tasks[t_id]

    return None


def claim_next_background_task(
    state: AsyncAgentState,
    *,
    worker_id: int,
    total_workers: int,
    shared_background_results: dict,
    shared_processing_tasks: set[str],
    shared_runtime_control: Optional[dict[str, Any]],
) -> Optional[TaskItem]:
    """Atomically claim the next eligible background task from the shared queue."""

    if shared_runtime_control is None:
        target_task = select_background_target_task(
            state,
            worker_id=worker_id,
            total_workers=total_workers,
            shared_background_results=shared_background_results,
            shared_processing_tasks=shared_processing_tasks,
            shared_runtime_control=shared_runtime_control,
        )
        if target_task is not None:
            shared_processing_tasks.add(
                task_processing_key(
                    int(target_task["task_id"]),
                    target_task.get("plan_id"),
                )
            )
        return target_task

    claim_lock = get_background_claim_lock(shared_runtime_control)
    with claim_lock:
        if len(shared_processing_tasks) >= max(1, int(total_workers)):
            return None
        target_task = select_background_target_task(
            state,
            worker_id=worker_id,
            total_workers=total_workers,
            shared_background_results=shared_background_results,
            shared_processing_tasks=shared_processing_tasks,
            shared_runtime_control=shared_runtime_control,
        )
        if target_task is None:
            return None
        processing_key = task_processing_key(
            int(target_task["task_id"]),
            target_task.get("plan_id"),
        )
        if processing_key in shared_processing_tasks:
            return None
        shared_processing_tasks.add(processing_key)
        return target_task


def background_selection_blocked_by_unresolved_decision(
    state: AsyncAgentState,
    *,
    shared_runtime_control: Optional[dict[str, Any]],
) -> bool:
    """Return True when background should wait for a decision branch to resolve."""

    tasks = state.get("tasks", {})
    allow_speculative_branches = _speculative_branch_preanalysis_from_control(
        shared_runtime_control
    )
    _, blocked = _selected_path_task_ids_for_background(
        tasks,
        state=state,
        shared_runtime_control=shared_runtime_control,
        allow_speculative_branches=allow_speculative_branches,
    )
    return blocked


GROUNDING_STAGE_RANK: dict[str, int] = {
    "started": 0,
    "memory_hit": 1,
    "candidate_seed": 2,
    "candidate_pack": 3,
    "target_decision": 4,
}


def _normalize_grounding_stage(value: Any) -> NavigationGroundingStage:
    """Return the most specific known navigation-grounding artifact stage."""

    normalized = str(value or "").strip()
    if normalized in GROUNDING_STAGE_RANK:
        return normalized  # type: ignore[return-value]
    return "started"


def _max_grounding_stage(*values: Any) -> NavigationGroundingStage:
    """Return the highest-ranked grounding stage among the supplied values."""

    best_stage: NavigationGroundingStage = "started"
    best_rank = GROUNDING_STAGE_RANK[best_stage]
    for value in values:
        stage = _normalize_grounding_stage(value)
        rank = GROUNDING_STAGE_RANK[stage]
        if rank > best_rank:
            best_stage = stage
            best_rank = rank
    return best_stage


def _looks_like_raw_tool_call_text(text: Any) -> bool:
    """Return True for model-emitted raw tool-call markup that should not be user context."""

    compact_text = str(text or "").strip().lower()
    if not compact_text:
        return False
    return (
        "<function=" in compact_text
        or "</tool_call>" in compact_text
        or compact_text.startswith("tool call:")
    )


def _append_background_note(
    existing_items: Sequence[str],
    candidate_text: Any,
    *,
    limit: int = 240,
    max_items: int = 5,
) -> list[str]:
    """Append one unique compact note to a bounded background-analysis list."""

    compact_text = truncate_context_text(candidate_text, limit=limit)
    if not compact_text or _looks_like_raw_tool_call_text(compact_text):
        return list(existing_items)

    updated_items = list(existing_items)
    if compact_text in updated_items:
        return updated_items

    updated_items.append(compact_text)
    return updated_items[-max_items:]


def _extract_candidate_keyframes_from_payload(raw_value: Any) -> list[dict[str, Any]]:
    """Extract compact candidate keyframe records from get_keyframe_nodes_info output."""

    parsed = _parse_json_like_payload(raw_value)
    if not isinstance(parsed, dict):
        return []

    data = parsed.get("data")
    if isinstance(data, dict) and isinstance(data.get("nodes"), dict):
        nodes = data.get("nodes")
    elif isinstance(parsed.get("nodes"), dict):
        nodes = parsed.get("nodes")
    else:
        return []

    candidates: list[dict[str, Any]] = []
    for raw_key, raw_node in nodes.items():
        if not isinstance(raw_node, dict):
            continue
        try:
            keyframe_id = int(raw_node.get("kf_id", raw_key))
        except Exception:
            continue
        item: dict[str, Any] = {"keyframe_id": keyframe_id}
        for source_key, target_key in (
            ("name", "name"),
            ("position", "position"),
            ("semantics", "semantics"),
        ):
            if source_key in raw_node:
                item[target_key] = raw_node.get(source_key)
        candidates.append(item)

    return candidates


def _merge_candidate_keyframe_records(
    existing: Sequence[dict[str, Any]],
    new_values: Sequence[dict[str, Any]],
    *,
    limit: int = 12,
) -> list[dict[str, Any]]:
    """Merge compact candidate keyframe records by keyframe id."""

    by_id: dict[int, dict[str, Any]] = {}
    order: list[int] = []
    for item in [*list(existing or []), *list(new_values or [])]:
        if not isinstance(item, dict):
            continue
        raw_id = item.get("keyframe_id", item.get("kf_id"))
        try:
            keyframe_id = int(raw_id)
        except Exception:
            continue
        if keyframe_id not in by_id:
            order.append(keyframe_id)
            by_id[keyframe_id] = {"keyframe_id": keyframe_id}
        for key, value in item.items():
            if value not in (None, "", [], {}):
                by_id[keyframe_id][key] = value
        if len(order) >= limit:
            break
    return [by_id[keyframe_id] for keyframe_id in order[:limit]]


def _extract_recommended_keyframe_from_text(
    text: Any,
    candidate_ids: Sequence[int],
) -> tuple[Optional[int], Optional[str], Optional[float]]:
    """Infer a recommended keyframe ID from final background prose when explicit."""

    compact_text = str(text or "")
    if not compact_text.strip():
        return None, None, None

    normalized_text = re.sub(r"[*_`]", "", compact_text)
    normalized_text = normalized_text.replace("\u2013", "-").replace("\u2014", "-")
    recommendation_patterns = (
        r"(?:best|strongest|top|primary|clearest|clear winner|most definitive|recommended|recommendation)[^\n.]{0,140}?(?:candidate)?[^\n.]{0,80}?(?:KF|keyframe)\s*#?\s*(\d+)",
        r"(?:KF|keyframe)\s*#?\s*(\d+)[^\n.]{0,160}?(?:best candidate|strongest candidate|top candidate|primary candidate|best|strongest|top|primary|recommended|most direct|direct match|exact match|clearest|most definitive|definitive match)",
        r"(?:navigate toward|navigate to|go to)[^\n.]{0,140}?(?:KF|keyframe)\s*#?\s*(\d+)",
    )
    for pattern in recommendation_patterns:
        match = re.search(pattern, normalized_text, flags=re.IGNORECASE)
        if not match:
            continue
        try:
            keyframe_id = int(match.group(1))
        except Exception:
            continue
        reason = truncate_context_text(match.group(0), limit=180)
        return keyframe_id, reason, 0.9

    if len(candidate_ids) == 1:
        keyframe_id = int(candidate_ids[0])
        return keyframe_id, "Only one background candidate keyframe was found.", 0.72

    return None, None, None


def _extract_recommendation_from_background_output(
    output: Any,
    candidate_ids: Sequence[int],
) -> tuple[list[int], Optional[int], Optional[str], Optional[float]]:
    """Extract final candidates plus an optional recommendation from background prose."""

    merged_candidates = _merge_candidate_keyframes(
        candidate_ids,
        _extract_keyframe_ids_from_payload(output),
    )
    (
        recommended_keyframe_id,
        recommendation_reason,
        recommendation_confidence,
    ) = _extract_recommended_keyframe_from_text(output, merged_candidates)
    if recommended_keyframe_id is not None:
        merged_candidates = _merge_candidate_keyframes(
            [recommended_keyframe_id],
            merged_candidates,
        )
    elif merged_candidates:
        recommended_keyframe_id = int(merged_candidates[0])
        recommendation_reason = (
            "Using the first ranked background candidate because no explicit "
            "recommendation sentence was parsed."
        )
        recommendation_confidence = 0.72
    return (
        merged_candidates,
        recommended_keyframe_id,
        recommendation_reason,
        recommendation_confidence,
    )


def _build_background_result_record(
    *,
    task_id: int,
    task_description: str,
    status: str,
    started_at: str,
    summary: Optional[str] = None,
    notes: Optional[Sequence[str]] = None,
    tool_observations: Optional[Sequence[str]] = None,
    latest_tool_name: Optional[str] = None,
    latest_tool_output: Optional[str] = None,
    grounding_stage: Optional[str] = None,
    target_text: Optional[str] = None,
    evidence_source: Optional[str] = None,
    truth_mode: Optional[str] = None,
    candidate_keyframe_ids: Optional[Sequence[int]] = None,
    candidate_keyframes: Optional[Sequence[dict[str, Any]]] = None,
    recommended_keyframe_id: Optional[int] = None,
    recommendation_confidence: Optional[float] = None,
    recommendation_reason: Optional[str] = None,
    final_output: Optional[str] = None,
    error: Optional[str] = None,
) -> BackgroundAnalysisItem:
    """Build one structured background-analysis cache record for a future task."""

    record: BackgroundAnalysisItem = {
        "task_id": task_id,
        "task_description": task_description,
        "status": status,
        "started_at": started_at,
        "updated_at": now_iso(),
    }
    if summary:
        record["summary"] = summary
    if notes:
        record["notes"] = list(notes)
    if tool_observations:
        record["tool_observations"] = list(tool_observations)
    if latest_tool_name:
        record["latest_tool_name"] = latest_tool_name
    if latest_tool_output:
        record["latest_tool_output"] = latest_tool_output
    resolved_stage = _max_grounding_stage(grounding_stage)
    normalized_candidates = _dedupe_ints(list(candidate_keyframe_ids or []))
    if normalized_candidates:
        record["candidate_keyframe_ids"] = normalized_candidates
        resolved_stage = _max_grounding_stage(resolved_stage, "candidate_seed")
    normalized_candidate_records = _merge_candidate_keyframe_records(
        [],
        list(candidate_keyframes or []),
    )
    if normalized_candidate_records:
        record["candidate_keyframes"] = normalized_candidate_records
        resolved_stage = _max_grounding_stage(resolved_stage, "candidate_pack")
    if recommended_keyframe_id is not None:
        try:
            record["recommended_keyframe_id"] = int(recommended_keyframe_id)
            resolved_stage = _max_grounding_stage(resolved_stage, "target_decision")
        except Exception:
            pass
    if recommendation_confidence is not None:
        try:
            record["recommendation_confidence"] = float(recommendation_confidence)
        except Exception:
            pass
    if recommendation_reason:
        record["recommendation_reason"] = recommendation_reason
    if resolved_stage != "started":
        record["grounding_stage"] = resolved_stage
    elif grounding_stage:
        record["grounding_stage"] = _normalize_grounding_stage(grounding_stage)
    if target_text:
        record["target_text"] = target_text
    if evidence_source in {"memory", "background", "foreground_tool"}:
        record["evidence_source"] = evidence_source  # type: ignore[typeddict-item]
    if truth_mode in {"live_verified", "historical_grounded", "background_hypothesis"}:
        record["truth_mode"] = truth_mode  # type: ignore[typeddict-item]
    if final_output:
        record["final_output"] = final_output
        record["completed_at"] = now_iso()
    if error:
        record["error"] = error
        if status == "failed":
            record["completed_at"] = now_iso()
    return record


def _extract_background_text_from_message(message: BaseMessage) -> Optional[str]:
    """Return compact human-readable evidence from one background-agent message."""

    if isinstance(message, AIMessage):
        text = truncate_context_text(message.content, limit=320)
        if _looks_like_raw_tool_call_text(text):
            return None
        return text
    if isinstance(message, ToolMessage):
        content_preview = truncate_context_text(
            stringify_tool_content(message.content),
            limit=260,
        )
        if not content_preview:
            return None
        tool_name = str(getattr(message, "name", "") or "tool")
        return f"{tool_name}: {content_preview}"
    return None


def _filter_tools_by_capability(
    tools: Sequence[BaseTool],
    *,
    names: set[str],
    tags: set[str],
) -> list[BaseTool]:
    """Return tools whose explicit name or capability tag is allowed."""

    filtered: list[BaseTool] = []
    for tool in tools:
        tool_name = str(getattr(tool, "name", "") or "")
        tool_tags = set(getattr(tool, "tags", None) or [])
        if tool_name in names or tool_tags.intersection(tags):
            filtered.append(tool)
    return filtered


def _tools_for_background_task(
    task: Optional[TaskItem],
    tasks: dict[int, TaskItem],
    tools: Sequence[BaseTool],
) -> list[BaseTool]:
    """Return background-safe tools for destination resolver preanalysis."""

    if not should_preanalyze_future_task(task, tasks):
        return []
    return _filter_tools_by_capability(
        tools,
        names={
            "search_requirement_on_keyframe_nodes",
            "search_keywords_on_keyframe_nodes",
            "get_keyframe_nodes_info",
            "analyse_on_each_kf_images",
        },
        tags={"scene_memory_search"},
    )


def _background_recommendation_is_actionable(
    bg_result: BackgroundAnalysisItem | str | None,
) -> bool:
    """Return True when background preanalysis contains a completed navigation target."""

    if not isinstance(bg_result, dict):
        return False
    stage = _normalize_grounding_stage(bg_result.get("grounding_stage"))
    if (
        str(bg_result.get("status") or "").strip().lower() != "completed"
        and stage != "target_decision"
    ):
        return False
    if bg_result.get("recommended_keyframe_id") is None:
        return False
    confidence = bg_result.get("recommendation_confidence")
    if confidence is None:
        return True
    try:
        return float(confidence) >= 0.7
    except Exception:
        return True


class BackgroundResultStore:
    """Persist scoped background-analysis records and release task claims safely."""

    def __init__(
        self,
        *,
        task: TaskItem,
        node_name: str,
        active_generation: Optional[int],
        shared_background_results: dict,
        shared_processing_tasks: set[str],
        shared_runtime_control: Optional[dict[str, Any]],
        logger: Optional[Any],
        run_memory: Optional[Any],
    ) -> None:
        self.task = task
        self.task_id = int(task["task_id"])
        self.description = str(task.get("description") or "")
        self.plan_id = task.get("plan_id")
        self.node_name = node_name
        self.active_generation = active_generation
        self.shared_background_results = shared_background_results
        self.shared_processing_tasks = shared_processing_tasks
        self.shared_runtime_control = shared_runtime_control
        self.logger = logger
        self.run_memory = run_memory
        self.processing_key = task_processing_key(self.task_id, self.plan_id)
        self.started_at = now_iso()

    def scope_is_current(self) -> bool:
        """Return True when this result still belongs to the active plan generation."""

        if self.shared_runtime_control is None:
            return True
        return (
            self.shared_runtime_control.get("active_plan_id") == self.plan_id
            and int(
                self.shared_runtime_control.get("background_generation", 0) or 0
            )
            == self.active_generation
        )

    def store(self, record: BackgroundAnalysisItem) -> bool:
        """Persist one partial or final background-analysis record when current."""

        if not self.scope_is_current():
            if self.logger:
                self.logger.log_background(
                    f"[{self.node_name} - Thread] Discarded stale background result for task {self.task_id}"
                )
            return False
        self.shared_background_results[self.task_id] = record
        try:
            self.run_memory.record_background_update(
                task_id=self.task_id,
                task_description=self.description,
                record=record,
                worker_name=self.node_name,
            )
        except Exception:
            pass
        return True

    def release_claim(self) -> None:
        """Release this task claim without deleting marks from a newer generation."""

        if self.processing_key not in self.shared_processing_tasks:
            return
        if self.shared_runtime_control is not None and int(
            self.shared_runtime_control.get("background_generation", 0) or 0
        ) != self.active_generation:
            return
        self.shared_processing_tasks.remove(self.processing_key)


class BackgroundForegroundCoordinator:
    """Handle foreground/background ownership and generation checks."""

    def __init__(
        self,
        *,
        state: AsyncAgentState,
        task_id: int,
        shared_runtime_control: Optional[dict[str, Any]],
    ) -> None:
        self.state = state
        self.task_id = task_id
        self.shared_runtime_control = shared_runtime_control

    def foreground_has_claimed_task(self) -> bool:
        """Return True when foreground has started this task or later."""

        return _foreground_has_claimed_task(
            self.task_id,
            self.state,
            self.shared_runtime_control,
        )

    def should_yield(self, grounding_stage: Optional[str]) -> bool:
        """Return True when foreground owns unfinished grounding work."""

        if not self.foreground_has_claimed_task():
            return False
        return (
            GROUNDING_STAGE_RANK[_normalize_grounding_stage(grounding_stage)]
            < GROUNDING_STAGE_RANK["target_decision"]
        )

    def maybe_yield(
        self,
        *,
        store: BackgroundResultStore,
        latest_summary: Optional[str],
        partial_notes: Sequence[str],
        partial_tool_observations: Sequence[str],
        latest_tool_name: Optional[str],
        latest_tool_output: Optional[str],
        grounding_stage: Optional[str],
        candidate_keyframe_ids: Sequence[int],
        candidate_keyframes: Sequence[dict[str, Any]],
        recommended_keyframe_id: Optional[int],
        recommendation_confidence: Optional[float],
        recommendation_reason: Optional[str],
    ) -> None:
        """Persist partial work and stop when foreground has claimed the task."""

        if not self.should_yield(grounding_stage):
            return
        if store.logger:
            store.logger.log_background(
                f"[{store.node_name} - Thread] Yielding task {store.task_id} to foreground; stage={grounding_stage}"
            )
        store.store(
            _build_background_result_record(
                task_id=store.task_id,
                task_description=store.description,
                status="completed" if candidate_keyframe_ids else "running",
                started_at=store.started_at,
                summary=latest_summary or "Background yielded to foreground.",
                notes=partial_notes,
                tool_observations=partial_tool_observations,
                latest_tool_name=latest_tool_name,
                latest_tool_output=latest_tool_output,
                grounding_stage=grounding_stage,
                target_text=store.description,
                evidence_source="background",
                truth_mode="background_hypothesis",
                candidate_keyframe_ids=candidate_keyframe_ids,
                candidate_keyframes=candidate_keyframes,
                recommended_keyframe_id=recommended_keyframe_id,
                recommendation_confidence=recommendation_confidence,
                recommendation_reason=(
                    recommendation_reason
                    or "Background yielded because foreground started this task."
                ),
            )
        )
        raise BackgroundYieldToForeground()


def _build_background_seed_context(
    *,
    state: AsyncAgentState,
    task: TaskItem,
    node_name: str,
    run_memory: Optional[Any],
) -> str:
    """Return optional background seed context from explicit task dependencies."""

    del state, node_name
    if run_memory is None:
        return ""
    try:
        table = run_memory.query_memory(
            scope="navigation",
            view="summary_table",
            query=str(task.get("description") or ""),
            time="all",
            limit=50,
        )
    except Exception:
        return ""

    items = list(table.get("items") or [])
    if not items:
        return (
            "--- NAVIGATION MEMORY TABLE ---\n"
            "No prior navigation anchors are available for this session.\n"
            "If the destination still needs resolving, use scene-memory search.\n"
            "--- END NAVIGATION MEMORY TABLE ---\n\n"
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
        "--- NAVIGATION MEMORY TABLE ---\n"
        "Use these rows first for visited-place reuse. If no row clearly matches, use scene-memory search.\n"
        f"{json.dumps(compact_rows, ensure_ascii=False, indent=2)}\n"
        "--- END NAVIGATION MEMORY TABLE ---\n\n"
    )


def _build_background_task_prompt(
    *,
    task_description: str,
    background_seed_context: str,
) -> str:
    """Build the prompt used by the scene-memory background analyzer."""

    return (
        "You are pre-analyzing a future destination-resolver task.\n\n"
        f"Task Description: {task_description}\n\n"
        f"{background_seed_context}"
        "Resolve the destination as far as possible from navigation memory and stored scene data. "
        "Remember: You cannot access current state or navigate. "
        "You do not know the robot's true live viewpoint, even if an earlier task navigated somewhere. "
        "For destination resolver tasks, first reuse a matching navigation-memory anchor if one exists; otherwise use scene-memory search and metadata. "
        "If a concrete keyframe is identified, state it clearly as the recommended destination keyframe. "
        "If the task involves analyzing the current image or state, what is visible right now, or what is true 'at that time', you MUST return directly and DO NOT call any tools. "
        "For those real-time tasks, DO NOT answer the question itself, DO NOT say the result is confirmed, and DO NOT give a final yes/no judgment. "
        "Instead, clearly say that real-time verification is required by the executor. "
        "You may optionally list historical reference keyframes, but you must label them as reference only and not tied to the current live viewpoint.\n\n"
        "OUTPUT REQUIREMENTS:\n"
        "- If you mention candidate keyframes, always include each keyframe ID.\n"
        "- For each candidate, include a concise semantic description or visual description, not just coordinates or names.\n"
        "- Briefly explain why the candidate matches the task.\n"
        "- If one candidate is clearly best, state that explicitly.\n"
        "- If navigation memory provides a clear match, prefer it over repeated scene-memory search.\n"
        "- Do not return coordinates-only lists unless semantic descriptions are truly unavailable.\n"
        "- For destination resolver tasks, provide a recommended keyframe when possible.\n"
        "- For real-time observation tasks, begin with a short note that real-time verification is required."
    )


def run_background_analysis(
    *,
    state: AsyncAgentState,
    task_copy: TaskItem,
    llm: BaseChatModel,
    tools: Sequence[BaseTool],
    background_prompt: str,
    store: BackgroundResultStore,
    coordinator: BackgroundForegroundCoordinator,
    logger: Optional[Any],
    run_memory: Optional[Any],
    react_agent_factory: Optional[Callable[..., Any]] = None,
) -> None:
    """Analyze one future task off-thread and cache incremental records."""

    desc = store.description
    background_tools = _tools_for_background_task(
        task_copy,
        state.get("tasks", {}),
        tools,
    )
    if not background_tools:
        return

    if logger:
        logger.log_background(
            f"[{store.node_name} - Thread] Pre-analyzing task {store.task_id}: {desc[:80]}..."
        )

    agent_factory = react_agent_factory or create_react_agent
    background_agent = agent_factory(
        model=llm,
        tools=background_tools,
        prompt=background_prompt,
        logger=logger.log_background if logger else None,
    )

    task_prompt = _build_background_task_prompt(
        task_description=desc,
        background_seed_context=_build_background_seed_context(
            state=state,
            task=task_copy,
            node_name=store.node_name,
            run_memory=run_memory,
        ),
    )

    agent_start_time = time.time()
    partial_notes: list[str] = []
    partial_tool_observations: list[str] = []
    candidate_keyframe_ids: list[int] = []
    candidate_keyframes: list[dict[str, Any]] = []
    recommended_keyframe_id: Optional[int] = None
    recommendation_confidence: Optional[float] = None
    recommendation_reason: Optional[str] = None
    grounding_stage: NavigationGroundingStage = "started"
    latest_summary: Optional[str] = "Background analysis started."
    latest_tool_name: Optional[str] = None
    latest_tool_output: Optional[str] = None
    agent_messages: list[BaseMessage] = []

    def maybe_yield_to_foreground() -> None:
        coordinator.maybe_yield(
            store=store,
            latest_summary=latest_summary,
            partial_notes=partial_notes,
            partial_tool_observations=partial_tool_observations,
            latest_tool_name=latest_tool_name,
            latest_tool_output=latest_tool_output,
            grounding_stage=grounding_stage,
            candidate_keyframe_ids=candidate_keyframe_ids,
            candidate_keyframes=candidate_keyframes,
            recommended_keyframe_id=recommended_keyframe_id,
            recommendation_confidence=recommendation_confidence,
            recommendation_reason=recommendation_reason,
        )

    store.store(
        _build_background_result_record(
            task_id=store.task_id,
            task_description=desc,
            status="running",
            started_at=store.started_at,
            summary=latest_summary,
            grounding_stage=grounding_stage,
            target_text=desc,
            evidence_source="background",
            truth_mode="background_hypothesis",
        )
    )

    with UnifiedLLMClient.request_priority("background"):
        if hasattr(background_agent, "stream"):
            for chunk in background_agent.stream(
                {"messages": [HumanMessage(content=task_prompt)]},
                stream_mode="values",
            ):
                if not store.scope_is_current():
                    return
                maybe_yield_to_foreground()
                if "messages" not in chunk:
                    continue

                updated = False
                for msg in chunk["messages"]:
                    if msg in agent_messages:
                        continue
                    agent_messages.append(msg)

                    if (
                        isinstance(msg, AIMessage)
                        and msg.content
                        and not _looks_like_raw_tool_call_text(msg.content)
                    ):
                        latest_summary = (
                            truncate_context_text(msg.content, limit=420)
                            or latest_summary
                        )
                        partial_notes = _append_background_note(
                            partial_notes,
                            msg.content,
                        )
                        updated = True
                    elif isinstance(msg, ToolMessage):
                        latest_tool_name = str(msg.name or "")
                        latest_tool_output = truncate_context_text(
                            stringify_tool_content(msg.content),
                            limit=260,
                        )
                        partial_tool_observations = _append_background_note(
                            partial_tool_observations,
                            _extract_background_text_from_message(msg),
                            limit=260,
                        )
                        candidate_keyframe_ids = _merge_candidate_keyframes(
                            candidate_keyframe_ids,
                            _extract_keyframe_ids_from_payload(msg.content),
                        )
                        candidate_keyframes = _merge_candidate_keyframe_records(
                            candidate_keyframes,
                            _extract_candidate_keyframes_from_payload(msg.content),
                        )
                        grounding_stage = _max_grounding_stage(
                            grounding_stage,
                            "candidate_pack"
                            if candidate_keyframes
                            else "candidate_seed"
                            if candidate_keyframe_ids
                            else "started",
                        )
                        updated = True

                if updated:
                    store.store(
                        _build_background_result_record(
                            task_id=store.task_id,
                            task_description=desc,
                            status="running",
                            started_at=store.started_at,
                            summary=latest_summary,
                            notes=partial_notes,
                            tool_observations=partial_tool_observations,
                            latest_tool_name=latest_tool_name,
                            latest_tool_output=latest_tool_output,
                            grounding_stage=grounding_stage,
                            target_text=desc,
                            evidence_source="background",
                            truth_mode="background_hypothesis",
                            candidate_keyframe_ids=candidate_keyframe_ids,
                            candidate_keyframes=candidate_keyframes,
                            recommended_keyframe_id=recommended_keyframe_id,
                            recommendation_confidence=recommendation_confidence,
                            recommendation_reason=recommendation_reason,
                        )
                    )
                    maybe_yield_to_foreground()
        else:
            result = background_agent.invoke(
                {"messages": [HumanMessage(content=task_prompt)]}
            )
            if "messages" in result:
                for msg in result["messages"]:
                    if msg in agent_messages:
                        continue
                    agent_messages.append(msg)
                    if (
                        isinstance(msg, AIMessage)
                        and msg.content
                        and not _looks_like_raw_tool_call_text(msg.content)
                    ):
                        latest_summary = (
                            truncate_context_text(msg.content, limit=420)
                            or latest_summary
                        )
                        partial_notes = _append_background_note(
                            partial_notes,
                            msg.content,
                        )
                    elif isinstance(msg, ToolMessage):
                        latest_tool_name = str(msg.name or "")
                        latest_tool_output = truncate_context_text(
                            stringify_tool_content(msg.content),
                            limit=260,
                        )
                        partial_tool_observations = _append_background_note(
                            partial_tool_observations,
                            _extract_background_text_from_message(msg),
                            limit=260,
                        )
                        candidate_keyframe_ids = _merge_candidate_keyframes(
                            candidate_keyframe_ids,
                            _extract_keyframe_ids_from_payload(msg.content),
                        )
                        candidate_keyframes = _merge_candidate_keyframe_records(
                            candidate_keyframes,
                            _extract_candidate_keyframes_from_payload(msg.content),
                        )
                        grounding_stage = _max_grounding_stage(
                            grounding_stage,
                            "candidate_pack"
                            if candidate_keyframes
                            else "candidate_seed"
                            if candidate_keyframe_ids
                            else "started",
                        )
            maybe_yield_to_foreground()

    execution_time = time.time() - agent_start_time

    output = ""
    for msg in reversed(agent_messages):
        if (
            isinstance(msg, AIMessage)
            and msg.content
            and not _looks_like_raw_tool_call_text(msg.content)
        ):
            output = str(msg.content)
            break

    if not output:
        output = "Background analysis completed but no specific findings."

    if logger:
        logger.log_background(
            f"[{store.node_name} - Thread] Completed task {store.task_id} in {execution_time:.2f}s"
        )

    latest_summary = truncate_context_text(output, limit=420) or latest_summary
    partial_notes = _append_background_note(partial_notes, output)
    (
        candidate_keyframe_ids,
        recommended_keyframe_id,
        recommendation_reason,
        recommendation_confidence,
    ) = _extract_recommendation_from_background_output(
        output,
        candidate_keyframe_ids,
    )
    if logger and recommended_keyframe_id is not None:
        logger.log_background(
            f"[{store.node_name} - Thread] Recommended keyframe {recommended_keyframe_id} for task {store.task_id}"
        )
    grounding_stage = _max_grounding_stage(
        grounding_stage,
        "target_decision" if recommended_keyframe_id is not None else None,
    )
    store.store(
        _build_background_result_record(
            task_id=store.task_id,
            task_description=desc,
            status="completed",
            started_at=store.started_at,
            summary=latest_summary,
            notes=partial_notes,
            tool_observations=partial_tool_observations,
            latest_tool_name=latest_tool_name,
            latest_tool_output=latest_tool_output,
            grounding_stage=grounding_stage,
            target_text=desc,
            evidence_source="background",
            truth_mode="background_hypothesis",
            candidate_keyframe_ids=candidate_keyframe_ids,
            candidate_keyframes=candidate_keyframes,
            recommended_keyframe_id=recommended_keyframe_id,
            recommendation_confidence=recommendation_confidence,
            recommendation_reason=recommendation_reason,
            final_output=output,
        )
    )


def create_background_worker_node(
    worker_id: int,
    total_workers: int,
    llm: BaseChatModel,
    tools: Sequence[BaseTool],
    shared_background_results: dict,
    shared_processing_tasks: set[str],
    shared_runtime_control: Optional[dict[str, Any]] = None,
    logger: Optional[Any] = None,
    run_memory: Optional[Any] = None,
    react_agent_factory: Optional[Callable[..., Any]] = None,
):
    """Spawn a background worker that pre-processes upcoming tasks in parallel."""

    background_prompt = AGENT_PROMPTS.get("background_system", "")

    node_name = f"bg_worker_{worker_id}"

    def background_worker_node(state: AsyncAgentState) -> AsyncAgentState:
        """Launch one background thread for one available destination resolver."""

        if logger:
            logger.log_background(f"Worker {worker_id}: processing background tasks")

        current_plan_id = state.get("current_plan_id")
        target_task = claim_next_background_task(
            state,
            worker_id=worker_id,
            total_workers=total_workers,
            shared_background_results=shared_background_results,
            shared_processing_tasks=shared_processing_tasks,
            shared_runtime_control=shared_runtime_control,
        )
        if target_task is None:
            return {}

        active_generation = None
        if shared_runtime_control is not None:
            active_generation = int(
                shared_runtime_control.get("background_generation", 0) or 0
            )

        def analyze_background_task(task_copy: TaskItem) -> None:
            """Run one claimed background analysis task."""

            store = BackgroundResultStore(
                task=task_copy,
                node_name=node_name,
                active_generation=active_generation,
                shared_background_results=shared_background_results,
                shared_processing_tasks=shared_processing_tasks,
                shared_runtime_control=shared_runtime_control,
                logger=logger,
                run_memory=run_memory,
            )
            coordinator = BackgroundForegroundCoordinator(
                state=state,
                task_id=store.task_id,
                shared_runtime_control=shared_runtime_control,
            )
            try:
                run_background_analysis(
                    state=state,
                    task_copy=task_copy,
                    llm=llm,
                    tools=tools,
                    background_prompt=background_prompt,
                    store=store,
                    coordinator=coordinator,
                    logger=logger,
                    run_memory=run_memory,
                    react_agent_factory=react_agent_factory,
                )
            except BackgroundYieldToForeground:
                pass
            except Exception as e:
                if logger:
                    logger.log_background(
                        f"[{node_name} - Thread] Error analyzing task {store.task_id}: {str(e)}"
                    )
                store.store(
                    _build_background_result_record(
                        task_id=store.task_id,
                        task_description=store.description,
                        status="failed",
                        started_at=store.started_at,
                        summary="Background analysis failed.",
                        error=f"Error in background analysis: {str(e)}",
                    )
                )
            finally:
                store.release_claim()

        def run_background_loop() -> None:
            """Analyze at most one available resolver task for this scheduling pass."""

            if shared_runtime_control is not None:
                if shared_runtime_control.get("active_plan_id") != current_plan_id:
                    return
                if (
                    int(shared_runtime_control.get("background_generation", 0) or 0)
                    != active_generation
                ):
                    return
            analyze_background_task(target_task)

        thread = threading.Thread(target=run_background_loop)
        thread.daemon = True
        thread.start()

        if logger:
            logger.log_background(
                f"[{node_name}] Started background thread for task {target_task['task_id']}"
            )

        return {}

    return background_worker_node


def create_bg_router(
    bg_node_name,
    num_background_workers,
    shared_background_results,
    shared_processing_tasks,
    shared_runtime_control: Optional[dict[str, Any]] = None,
) -> Callable[[AsyncAgentState], str]:
    """Create a router that re-schedules one background worker while work remains."""

    def route_background_worker(state: AsyncAgentState) -> str:
        """Return the worker node name when shared background work remains."""

        worker_id = int(bg_node_name.split("_")[-1])
        total_workers = num_background_workers

        if select_background_target_task(
            state,
            worker_id=worker_id,
            total_workers=total_workers,
            shared_background_results=shared_background_results,
            shared_processing_tasks=shared_processing_tasks,
            shared_runtime_control=shared_runtime_control,
        ):
            return bg_node_name

        return END

    return route_background_worker


__all__ = [
    "BackgroundYieldToForeground",
    "BackgroundResultStore",
    "BackgroundForegroundCoordinator",
    "GROUNDING_STAGE_RANK",
    "_foreground_has_claimed_task",
    "_background_recommendation_is_actionable",
    "_build_background_result_record",
    "_extract_recommendation_from_background_output",
    "_max_grounding_stage",
    "_merge_candidate_keyframe_records",
    "_normalize_grounding_stage",
    "background_selection_blocked_by_unresolved_decision",
    "claim_next_background_task",
    "create_background_worker_node",
    "create_bg_router",
    "run_background_analysis",
    "select_background_target_task",
]
