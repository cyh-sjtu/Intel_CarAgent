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
from caragent_agent.agents.async_agent.execution.tool_call_budget import (
    execute_tool_budget_context,
)

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
from caragent_agent.agents.async_agent.runtime.legacy_task_metadata import (
    has_legacy_grounding_metadata,
    legacy_object_kind,
    legacy_staging_kind,
    legacy_upstream_task_id,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END
from caragent_agent.agents.async_agent.execution.tool_results import (
    dedupe_ints as _dedupe_ints,
    extract_structured_tool_result as _extract_structured_tool_result,
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
        if (
            _task_is_semantic_object_grounding(task)
            and _find_staging_keyframe_for_object_task(task, tasks) is None
        ):
            continue
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


def _json_from_text(text: Any) -> dict[str, Any] | None:
    clean = str(text or "").strip()
    if not clean:
        return None
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", clean, re.DOTALL | re.IGNORECASE)
    candidates = [fenced.group(1).strip()] if fenced else []
    candidates.append(clean)
    for candidate in candidates:
        start_index = candidate.find("{")
        while start_index >= 0:
            depth = 0
            in_string = False
            escape = False
            for index in range(start_index, len(candidate)):
                char = candidate[index]
                if escape:
                    escape = False
                    continue
                if char == "\\":
                    escape = True
                    continue
                if char == '"':
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            parsed = json.loads(candidate[start_index : index + 1])
                        except Exception:
                            break
                        if isinstance(parsed, dict):
                            return parsed
                        break
            start_index = candidate.find("{", start_index + 1)
    return None


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

    parsed = _json_from_text(normalized_text)
    if isinstance(parsed, dict):
        destination = parsed.get("destination")
        if isinstance(destination, dict) and destination.get("type") == "keyframe":
            try:
                keyframe_id = int(destination.get("keyframe_id"))
                reason = truncate_context_text(
                    parsed.get("recommendation_reason")
                    or parsed.get("reason")
                    or "Structured background destination JSON.",
                    limit=180,
                )
                return keyframe_id, reason, 0.96
            except Exception:
                pass
        for key in ("recommended_keyframe_id", "keyframe_id"):
            if parsed.get(key) is None:
                continue
            try:
                keyframe_id = int(parsed.get(key))
                reason = truncate_context_text(
                    parsed.get("recommendation_reason")
                    or parsed.get("reason")
                    or f"Structured background recommendation field `{key}`.",
                    limit=180,
                )
                return keyframe_id, reason, 0.94
            except Exception:
                continue

    recommendation_patterns = (
        r"(?:recommended\s+keyframe|recommended\s+destination|recommendation)\s*[:#-]?\s*(?:\s|\n|.){0,120}?(?:KF|keyframe)\s*#?\s*(\d+)",
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
    recommended_destination: Optional[dict[str, Any]] = None,
    destination_type: Optional[str] = None,
    object_preanalysis: Optional[dict[str, Any]] = None,
    failure_reason: Optional[str] = None,
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
    if isinstance(recommended_destination, dict):
        record["recommended_destination"] = recommended_destination
        record["destination_type"] = str(destination_type or recommended_destination.get("type") or "")
        resolved_stage = _max_grounding_stage(resolved_stage, "target_decision")
    elif destination_type:
        record["destination_type"] = str(destination_type)
    if isinstance(object_preanalysis, dict):
        record["object_preanalysis"] = object_preanalysis
    if failure_reason:
        record["failure_reason"] = str(failure_reason)
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


def _record_waiting_object_preanalysis_if_needed(
    *,
    state: AsyncAgentState,
    shared_background_results: dict,
    shared_processing_tasks: set[str],
    shared_runtime_control: Optional[dict[str, Any]],
    worker_name: str,
    run_memory: Optional[Any],
    logger: Optional[Any],
) -> None:
    """Record that an object preanalysis is waiting for its staging keyframe."""

    if run_memory is None or shared_runtime_control is None:
        return

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

    recorded = shared_runtime_control.setdefault(
        "background_waiting_object_preanalysis_keys",
        set(),
    )
    if not hasattr(recorded, "__contains__") or not hasattr(recorded, "add"):
        recorded = set(recorded if isinstance(recorded, (list, tuple, set)) else [])
        shared_runtime_control["background_waiting_object_preanalysis_keys"] = recorded

    for t_id in path_task_ids:
        task = tasks.get(t_id)
        if not isinstance(task, dict):
            continue
        task_plan_id = task.get("plan_id")
        processing_key = task_processing_key(t_id, task_plan_id)
        if (
            task.get("type") != "action"
            or task_plan_id != current_plan_id
            or t_id in shared_background_results
            or processing_key in shared_processing_tasks
            or processing_key in recorded
            or _foreground_has_claimed_task(t_id, state, shared_runtime_control)
            or not should_preanalyze_future_task(task, tasks)
            or not _task_is_semantic_object_grounding(task)
            or _find_staging_keyframe_for_object_task(task, tasks) is not None
        ):
            continue

        desc = str(task.get("description") or "")
        record = _build_background_result_record(
            task_id=int(t_id),
            task_description=desc,
            status="waiting",
            started_at=now_iso(),
            summary=(
                "Semantic object background preanalysis is waiting for the staging "
                "keyframe resolver/navigation result."
            ),
            grounding_stage="started",
            target_text=desc,
            evidence_source="background",
            truth_mode="background_hypothesis",
            failure_reason="waiting_for_staging_keyframe",
            recommendation_reason=(
                "Historical object preanalysis requires a resolved staging keyframe "
                "before it can inspect stored stereo keyframe images."
            ),
        )
        try:
            run_memory.record_background_update(
                task_id=int(t_id),
                task_description=desc,
                record=record,
                worker_name=worker_name,
            )
            recorded.add(processing_key)
            if logger:
                logger.log_background(
                    f"[{worker_name}] Object preanalysis for task {t_id} is waiting for staging keyframe."
                )
        except Exception as exc:
            if logger:
                logger.log_background(
                    f"[{worker_name}] Failed to record waiting object preanalysis for task {t_id}: {exc}"
                )
        return


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
    """Return background-safe tools for semantic target preanalysis."""

    if not should_preanalyze_future_task(task, tasks):
        return []
    return _filter_tools_by_capability(
        tools,
        names={
            "search_requirement_on_keyframe_nodes",
            "search_keywords_on_keyframe_nodes",
            "get_keyframe_nodes_info",
            "analyse_on_each_kf_images",
            "preanalyze_object_on_keyframe",
        },
        tags={"scene_memory_search", "object_preanalysis"},
    )


def _background_recommendation_is_actionable(
    bg_result: BackgroundAnalysisItem | str | None,
) -> bool:
    """Return True when background preanalysis contains a completed navigation target."""

    if not isinstance(bg_result, dict):
        return False
    if isinstance(bg_result.get("recommended_destination"), dict):
        return str(bg_result.get("status") or "").strip().lower() == "completed"
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


def _task_metadata_marks_semantic_object(task: Optional[TaskItem]) -> bool:
    """Return True when current-schema task metadata identifies semantic object grounding."""

    target = (task or {}).get("target")
    if (
        isinstance(target, dict)
        and str(target.get("type") or "").strip() == "semantic_object"
    ):
        source = str(target.get("target_source") or "").strip()
        if source == "current_view":
            return False
        return True
    return legacy_object_kind(task)


def _legacy_description_marks_semantic_object(task: Optional[TaskItem]) -> bool:
    """Return True only for older plans that lack semantic object metadata."""

    target = (task or {}).get("target")
    if isinstance(target, dict):
        return False
    if has_legacy_grounding_metadata(task):
        return False
    text = str((task or {}).get("description") or "").lower()
    return (
        "approach_object_in_current_view" in text
        or "semantic object" in text
        or "object level" in text
        or "visible target" in text and "object" in text
    )


def _task_is_semantic_object_grounding(task: Optional[TaskItem]) -> bool:
    if _task_metadata_marks_semantic_object(task):
        return True
    return _legacy_description_marks_semantic_object(task)


def _latest_task_result_payload(task: Optional[TaskItem]) -> Any:
    if not task:
        return None
    for result in reversed(list(task.get("result", []) or [])):
        if isinstance(result, dict):
            direct_payload: dict[str, Any] = {}
            for key in ("destination", "target", "current_place_context"):
                if result.get(key) not in (None, "", [], {}):
                    direct_payload[key] = result.get(key)
            if direct_payload:
                return direct_payload
        raw_output = result.get("raw_output")
        if not raw_output:
            continue
        try:
            trace = json.loads(raw_output)
        except Exception:
            trace = None
        if isinstance(trace, dict):
            final_text = str(trace.get("final_ai_content") or "")
            parsed = _json_from_text(final_text)
            if isinstance(parsed, dict):
                return parsed
            for tool_result in reversed(list(trace.get("tool_results", []) or [])):
                structured = _extract_structured_tool_result(tool_result.get("content"))
                if structured is None:
                    continue
                data = structured.get("data")
                if data is not None:
                    return data
    return None


def _extract_keyframe_destination(value: Any) -> Optional[int]:
    if isinstance(value, dict):
        if value.get("type") == "keyframe" and value.get("keyframe_id") is not None:
            try:
                return int(value.get("keyframe_id"))
            except Exception:
                return None
        for direct_key in ("target_keyframe_id", "recommended_keyframe_id"):
            if value.get(direct_key) is not None:
                try:
                    return int(value.get(direct_key))
                except Exception:
                    pass
        for key in ("destination", "target", "data"):
            found = _extract_keyframe_destination(value.get(key))
            if found is not None:
                return found
        for nested in value.values():
            if isinstance(nested, (dict, list, tuple)):
                found = _extract_keyframe_destination(nested)
                if found is not None:
                    return found
    elif isinstance(value, (list, tuple)):
        for item in value:
            found = _extract_keyframe_destination(item)
            if found is not None:
                return found
    return None


def _task_ids_from_inputs_from(value: Any) -> list[int]:
    task_ids: list[int] = []

    def add(raw_value: Any) -> None:
        try:
            task_id = int(raw_value)
        except Exception:
            return
        if task_id not in task_ids:
            task_ids.append(task_id)

    def visit(item: Any) -> None:
        if isinstance(item, dict):
            if item.get("task_id") is not None:
                add(item.get("task_id"))
            for nested in item.values():
                visit(nested)
            return
        if isinstance(item, (list, tuple, set)):
            for nested in item:
                visit(nested)
            return
        match = re.search(r"task\s*(\d+)", str(item or ""), flags=re.IGNORECASE)
        if match:
            add(match.group(1))

    visit(value)
    return task_ids


def _find_staging_keyframe_for_object_task(
    task: TaskItem,
    tasks: dict[int, TaskItem],
) -> Optional[int]:
    target = task.get("target")
    if isinstance(target, dict):
        inputs_from = target.get("inputs_from")
        for task_id in _task_ids_from_inputs_from(inputs_from):
            source_task = tasks.get(task_id)
            found = _extract_keyframe_destination(_latest_task_result_payload(source_task))
            if found is not None:
                return found
    legacy_upstream_id = legacy_upstream_task_id(task)
    if legacy_upstream_id is not None:
        staging_task = tasks.get(legacy_upstream_id)
        found = _extract_keyframe_destination(_latest_task_result_payload(staging_task))
        if found is not None:
            return found
        target = staging_task.get("target") if isinstance(staging_task, dict) else None
        if isinstance(target, dict):
            found = _extract_keyframe_destination(target)
            if found is not None:
                return found
    for dep_id in reversed(list(task.get("depends_on", []) or [])):
        dep_task = tasks.get(dep_id)
        if not isinstance(dep_task, dict):
            continue
        target = dep_task.get("target")
        if dep_task.get("task_type") == "navigation_action" and isinstance(target, dict):
            if target.get("type") == "keyframe":
                try:
                    return int(target.get("keyframe_id"))
                except Exception:
                    pass
            if target.get("type") == "task_output":
                try:
                    source_task = tasks.get(int(target.get("task_id")))
                except Exception:
                    source_task = None
                found = _extract_keyframe_destination(_latest_task_result_payload(source_task))
                if found is not None:
                    return found
        found = _extract_keyframe_destination(_latest_task_result_payload(dep_task))
        if found is not None:
            return found
    return None


def _find_tool_by_name(tools: Sequence[BaseTool], name: str) -> Optional[BaseTool]:
    for tool in tools:
        if str(getattr(tool, "name", "") or "").strip() == name:
            return tool
    return None


def _structured_tool_data(raw_result: Any) -> Any:
    structured = _extract_structured_tool_result(raw_result)
    if structured is not None:
        return structured.get("data")
    if isinstance(raw_result, dict) and "data" in raw_result:
        return raw_result.get("data")
    return raw_result


def _structured_tool_status(raw_result: Any) -> str:
    structured = _extract_structured_tool_result(raw_result)
    if structured is not None:
        return str(structured.get("status") or "").strip().lower()
    if isinstance(raw_result, dict):
        return str(raw_result.get("status") or "").strip().lower()
    return ""


def _structured_tool_error(raw_result: Any) -> str:
    structured = _extract_structured_tool_result(raw_result)
    payload = structured if structured is not None else raw_result if isinstance(raw_result, dict) else {}
    error = payload.get("error") if isinstance(payload, dict) else None
    if isinstance(error, dict):
        return str(error.get("message") or error.get("code") or "").strip()
    return str(error or "").strip()


def _try_historical_object_preanalysis(
    *,
    state: AsyncAgentState,
    task_copy: TaskItem,
    tools: Sequence[BaseTool],
    store: "BackgroundResultStore",
    logger: Optional[Any],
) -> bool:
    """Run deterministic keyframe-image object preanalysis when staging is known."""

    if not _task_is_semantic_object_grounding(task_copy):
        return False
    tasks = state.get("tasks", {})
    staging_keyframe_id = _find_staging_keyframe_for_object_task(task_copy, tasks)
    if staging_keyframe_id is None:
        return False
    tool = _find_tool_by_name(tools, "preanalyze_object_on_keyframe")
    if tool is None:
        return False
    desc = str(task_copy.get("description") or "")
    target = task_copy.get("target")
    object_description = desc
    stop_distance_m = 0.8
    if isinstance(target, dict):
        object_description = str(target.get("object_description") or desc).strip() or desc
        if target.get("stop_distance_m") is not None:
            try:
                stop_distance_m = float(target.get("stop_distance_m"))
            except Exception:
                stop_distance_m = 0.8
    if logger:
        logger.log_background(
            f"[{store.node_name} - Thread] Historical object preanalysis on keyframe {staging_keyframe_id} for task {store.task_id}"
        )
    started_at = store.started_at
    store.store(
        _build_background_result_record(
            task_id=store.task_id,
            task_description=desc,
            status="running",
            started_at=started_at,
            summary=f"Historical object preanalysis started on keyframe {staging_keyframe_id}.",
            grounding_stage="candidate_pack",
            target_text=desc,
            evidence_source="background",
            truth_mode="background_hypothesis",
            candidate_keyframe_ids=[staging_keyframe_id],
        )
    )
    try:
        raw_result = tool.invoke(
            {
                "keyframe_id": int(staging_keyframe_id),
                "object_description": object_description,
                "stop_distance_m": stop_distance_m,
            }
        )
    except Exception as exc:
        raw_result = {
            "status": "error",
            "summary": "Historical object preanalysis raised an exception.",
            "data": None,
            "error": {"message": str(exc)},
            "provenance": {"source_type": "scene_memory"},
        }
    data = _structured_tool_data(raw_result)
    status = _structured_tool_status(raw_result)
    destination = data.get("destination") if isinstance(data, dict) else None
    failure_reason = _structured_tool_error(raw_result)
    if not failure_reason and isinstance(data, dict):
        approach = data.get("approach") if isinstance(data.get("approach"), dict) else {}
        failure_reason = str(approach.get("reason") or data.get("status") or "").strip()
    if status == "ok" and isinstance(destination, dict):
        if logger:
            logger.log_background(
                "[{node} - Thread] Historical object preanalysis completed for task {task_id}; destination={destination}".format(
                    node=store.node_name,
                    task_id=store.task_id,
                    destination=destination,
                )
            )
        store.store(
            _build_background_result_record(
                task_id=store.task_id,
                task_description=desc,
                status="completed",
                started_at=started_at,
                summary=f"Historical object preanalysis produced a position destination from keyframe {staging_keyframe_id}.",
                grounding_stage="target_decision",
                target_text=desc,
                evidence_source="background",
                truth_mode="background_hypothesis",
                candidate_keyframe_ids=[staging_keyframe_id],
                recommended_destination=destination,
                destination_type=str(destination.get("type") or "position"),
                object_preanalysis=data if isinstance(data, dict) else None,
                recommendation_reason=f"Static semantic object preanalysis on historical keyframe {staging_keyframe_id}.",
                final_output=f"Recommended semantic object destination from keyframe {staging_keyframe_id}.",
            )
        )
        return True
    if logger:
        logger.log_background(
            "[{node} - Thread] Historical object preanalysis failed for task {task_id}; keyframe={keyframe_id}; reason={reason}".format(
                node=store.node_name,
                task_id=store.task_id,
                keyframe_id=staging_keyframe_id,
                reason=failure_reason or "destination_unavailable",
            )
        )
    store.store(
        _build_background_result_record(
            task_id=store.task_id,
            task_description=desc,
            status="failed",
            started_at=started_at,
            summary=f"Historical object preanalysis did not produce a destination from keyframe {staging_keyframe_id}.",
            grounding_stage="candidate_pack",
            target_text=desc,
            evidence_source="background",
            truth_mode="background_hypothesis",
            candidate_keyframe_ids=[staging_keyframe_id],
            object_preanalysis=data if isinstance(data, dict) else None,
            failure_reason=failure_reason or "destination_unavailable",
            error=failure_reason or "destination_unavailable",
        )
    )
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

    def task_is_staging_keyframe_grounding(self) -> bool:
        """Return True for background work that can feed later object preanalysis."""

        task = self.state.get("tasks", {}).get(self.task_id)
        if not isinstance(task, dict):
            return False
        target = task.get("target")
        if (
            task.get("task_type") == "navigation_action"
            and isinstance(target, dict)
            and str(target.get("type") or "").strip() == "semantic_keyframe"
        ):
            return True
        return legacy_staging_kind(task)

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
        if (
            self.task_is_staging_keyframe_grounding()
            and (candidate_keyframe_ids or candidate_keyframes)
            and recommended_keyframe_id is None
        ):
            if store.logger:
                store.logger.log_background(
                    f"[{store.node_name} - Thread] Continuing staging preanalysis for task {store.task_id}; "
                    f"foreground has started, but candidate evidence is available and downstream tasks can reuse it."
                )
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
        "You are pre-analyzing a future semantic grounding task.\n\n"
        f"Task Description: {task_description}\n\n"
        f"{background_seed_context}"
        "Prepare grounding evidence as far as possible from navigation memory and stored scene data. "
        "Remember: You cannot access current state or navigate. "
        "You do not know the robot's true live viewpoint, even if an earlier task navigated somewhere. "
        "For reusable semantic destination signals, first reuse a matching navigation-memory anchor if one exists; otherwise use scene-memory search and metadata. "
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
        "- For semantic keyframe grounding, provide a recommended keyframe when possible.\n"
        "- If one keyframe is clearly best, include a compact recommendation payload if possible: "
        "{\"recommended_keyframe_id\":<id>,\"recommendation_reason\":\"short reason\"}. "
        "The payload must match your prose recommendation and is background evidence, not a movement command.\n"
        "- If several candidates remain plausible and none is clearly best, do not invent a final destination; "
        "return candidates and explain what foreground verification should compare.\n"
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

    if _task_is_semantic_object_grounding(task_copy):
        if _try_historical_object_preanalysis(
            state=state,
            task_copy=task_copy,
            tools=background_tools,
            store=store,
            logger=logger,
        ):
            return
        return

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

    with UnifiedLLMClient.request_priority("background"), execute_tool_budget_context():
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
        """Launch one background thread for one available semantic target."""

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
            _record_waiting_object_preanalysis_if_needed(
                state=state,
                shared_background_results=shared_background_results,
                shared_processing_tasks=shared_processing_tasks,
                shared_runtime_control=shared_runtime_control,
                worker_name=node_name,
                run_memory=run_memory,
                logger=logger,
            )
            return {}
        if logger:
            logger.log_background(
                "Worker {worker_id}: claimed background task {task_id}; target_type={target_type}; outputs={outputs}".format(
                    worker_id=worker_id,
                    task_id=target_task.get("task_id"),
                    target_type=(
                        (target_task.get("target") or {}).get("type")
                        if isinstance(target_task.get("target"), dict)
                        else ""
                    ),
                    outputs=target_task.get("outputs"),
                )
            )

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
            """Analyze at most one available semantic target for this scheduling pass."""

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
