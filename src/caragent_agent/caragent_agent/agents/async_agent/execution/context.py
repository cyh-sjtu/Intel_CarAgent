"""Execution-context helpers for the async agent."""

from __future__ import annotations

import json
from typing import Any, Optional, Sequence

from langchain_core.tools import BaseTool

from ..runtime.types import (
    AsyncAgentState,
    BackgroundAnalysisItem,
    EventItem,
    TaskItem,
)
from ..runtime.legacy_task_metadata import (
    legacy_object_kind,
    legacy_staging_kind,
)

def truncate_context_text(value: Any, limit: int = 400) -> Optional[str]:
    """Return a compact string for prompt context payloads."""

    if value is None:
        return None

    text = str(value).strip()
    if not text:
        return None

    if len(text) <= limit:
        return text

    return f"{text[:limit]}..."


def _looks_like_raw_tool_call_text(value: Any) -> bool:
    """Return True for raw model tool-call markup that should not become context."""

    text = str(value or "").strip().lower()
    return bool(
        text
        and (
            "<function=" in text
            or "</tool_call>" in text
            or text.startswith("tool call:")
        )
    )


def _background_text_is_useful(value: Any) -> bool:
    """Return True when background text is meaningful enough for executor injection."""

    text = str(value or "").strip()
    if not text or _looks_like_raw_tool_call_text(text):
        return False

    lowered = text.lower()
    if lowered in {
        "background analysis started.",
        "background analysis started",
        "background analysis completed but no specific findings.",
    }:
        return False
    if '"status": "error"' in lowered or "'status': 'error'" in lowered:
        return False
    if " search failed." in lowered or "analysis failed." in lowered:
        return False
    return True


def build_task_scoped_user_input_content(
    current_task: Optional[TaskItem],
    current_user_input: Optional[dict[str, Any]],
) -> Optional[str]:
    """Build a task-localized user-input view so execute does not inherit future goals."""

    task_description = str(
        (current_task or {}).get("description") or ""
    ).strip()
    if task_description:
        return task_description

    user_input_content = str(
        (current_user_input or {}).get("content") or ""
    ).strip()
    return user_input_content or None


def _has_query_dependency(
    task: Optional[TaskItem],
    tasks: dict[int, TaskItem],
) -> bool:
    """Return True when any upstream dependency is a query-style llm_action."""

    if not task:
        return False
    for dep_id in task.get("depends_on", []):
        dep_task = tasks.get(dep_id)
        if not dep_task:
            continue
        if dep_task.get("task_type") == "llm_action":
            return True
    return False


def _navigation_action_resolver_task_id(task: Optional[TaskItem]) -> Optional[int]:
    """Return the source task id used by a structured navigation target."""

    if not task or task.get("task_type") != "navigation_action":
        return None
    target = task.get("target")
    if not isinstance(target, dict):
        return None
    if target.get("type") != "task_output" or target.get("field") != "destination":
        return None
    try:
        return int(target.get("task_id"))
    except Exception:
        return None


def is_navigation_action(task: Optional[TaskItem]) -> bool:
    """Return True only for structured navigation_action tasks."""

    if not task:
        return False

    task_type = str(task.get("task_type") or "").strip().lower()
    return task_type == "navigation_action"


def task_requires_current_state(task: Optional[TaskItem]) -> bool:
    """Return False in the simplified schema; tool choice is executor-driven."""

    del task
    return False


def task_requires_current_position_distance(task: Optional[TaskItem]) -> bool:
    """Return False in the simplified schema; tool choice is executor-driven."""

    del task
    return False


def task_requires_current_view(task: Optional[TaskItem]) -> bool:
    """Return False in the simplified schema; tool choice is executor-driven."""

    del task
    return False


def task_mentions_history_reference(task: Optional[TaskItem]) -> bool:
    """Return False in the simplified schema; memory is a core executor tool."""

    del task
    return False


def task_depends_on_query_result(
    task: Optional[TaskItem],
    tasks: dict[int, TaskItem],
) -> bool:
    """Return True when planner metadata says a task target comes from upstream."""

    return _has_query_dependency(task, tasks)


def background_result_is_reusable_for_task(
    bg_result: Optional[BackgroundAnalysisItem | str],
) -> bool:
    """Return True when cached background output is useful executor evidence."""

    if not isinstance(bg_result, dict):
        return False
    if isinstance(bg_result.get("recommended_destination"), dict):
        return True
    if bg_result.get("recommended_keyframe_id") is not None:
        return True
    if bg_result.get("failure_reason") or bg_result.get("object_preanalysis"):
        return True
    return False


def get_default_execute_context_keys(
    current_task: Optional[TaskItem],
) -> tuple[str, ...]:
    """Return the minimal context keys every execute task may see."""

    del current_task
    return (
        "current_task",
        "current_plan_id",
        "current_user_input",
        "upstream_tasks",
        "arrival_context",
        "plan_blackboard",
    )


def should_preanalyze_future_task(
    current_task: Optional[TaskItem],
    tasks: dict[int, TaskItem],
) -> bool:
    """Return True for future tasks that can benefit from early grounding."""

    if not current_task:
        return False
    target = current_task.get("target")
    if (
        current_task.get("task_type") == "navigation_action"
        and isinstance(target, dict)
        and str(target.get("type") or "").strip() == "semantic_object"
    ):
        if str(target.get("target_source") or "").strip() == "current_view":
            return False
        return True
    if (
        current_task.get("task_type") == "navigation_action"
        and isinstance(target, dict)
        and str(target.get("type") or "").strip() == "semantic_keyframe"
    ):
        source = str(target.get("target_source") or "").strip()
        return source in {"scene_memory", "explicit", ""}
    if current_task.get("task_type") != "llm_action":
        return False
    outputs = {str(value).strip() for value in current_task.get("outputs", []) or []}
    is_semantic_object = (
        "destination" in outputs
        and bool(current_task.get("inputs_from"))
        and current_task.get("target") in (None, "", [], {})
    )
    if not is_semantic_object:
        is_semantic_object = legacy_object_kind(current_task)
    if not is_semantic_object and not current_task.get("target"):
        description = str(current_task.get("description") or "").lower()
        is_semantic_object = (
            "approach_object_in_current_view" in description
            or "semantic object" in description
            or "object level" in description
        )
    current_task_id = current_task.get("task_id")
    for task in tasks.values():
        if _navigation_action_resolver_task_id(task) == current_task_id:
            return True
    if is_semantic_object:
        return True
    return False


def prepare_context_bundle(
    state: AsyncAgentState,
    current_task: Optional[TaskItem],
    *,
    run_memory: Optional[Any] = None,
) -> dict[str, Any]:
    """Build the executor's minimal dependency-driven context package."""

    del run_memory

    selected_context_packet = build_execution_context_snapshot(
        state,
        current_task,
    )
    context_keys = get_default_execute_context_keys(current_task)
    selected_context_packet = filter_execution_context_packet(
        selected_context_packet,
        context_keys,
    )

    return {
        "selected_execution_context_packet": selected_context_packet,
        "context_keys": context_keys,
        "include_background_reference": False,
    }


def filter_execution_context_packet(
    packet: dict[str, Any],
    selected_keys: Sequence[str],
) -> dict[str, Any]:
    """Return one filtered execution packet without rebuilding the full context."""

    allowed_keys = {
        str(key or "").strip()
        for key in selected_keys
        if str(key or "").strip()
    }
    return {
        key: value
        for key, value in packet.items()
        if key in allowed_keys
    }


def summarize_task_for_execution_context(task: Optional[TaskItem]) -> Optional[dict[str, Any]]:
    """Build a compact task summary suitable for executor prompt context."""

    if not task:
        return None

    latest_result = (task.get("result") or [])[-1] if task.get("result") else None
    compact_result = None
    if latest_result:
        compact_result = {
            "summary": latest_result.get("summary"),
            "tool_name": latest_result.get("tool_name"),
            "decision": latest_result.get("decision"),
            "raw_output_excerpt": truncate_context_text(
                latest_result.get("raw_output"),
                limit=320,
            ),
        }

    return {
        "task_id": task.get("task_id"),
        "task_type": task.get("task_type"),
        "target": task.get("target"),
        "inputs_from": task.get("inputs_from", {}),
        "outputs": task.get("outputs", []),
        "image_refs": task.get("image_refs", []),
        "primary_target": task.get("primary_target"),
        "scene_context": task.get("scene_context"),
        "selection_policy": task.get("selection_policy"),
        "type": task.get("type"),
        "description": task.get("description"),
        "status": task.get("status"),
        "plan_id": task.get("plan_id"),
        "user_input_id": task.get("user_input_id"),
        "depends_on": task.get("depends_on", []),
        "wait_for_event": task.get("wait_for_event"),
        "terminal_reason": task.get("terminal_reason"),
        "latest_result": compact_result,
    }


def _parse_json_like_context_payload(value: Any) -> Any:
    """Parse a JSON-ish value for compact blackboard extraction."""

    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return value
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    start = text.find("{")
    while start >= 0:
        depth = 0
        in_string = False
        escape = False
        for index in range(start, len(text)):
            char = text[index]
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
                        return json.loads(text[start : index + 1])
                    except Exception:
                        break
        start = text.find("{", start + 1)
    return value


def _extract_named_signal(value: Any, signal_name: str, *, _depth: int = 0) -> Any:
    """Extract a compact named output signal from common task/tool payload shapes."""

    if value is None or _depth > 8:
        return None
    parsed = _parse_json_like_context_payload(value)
    if parsed is not value:
        return _extract_named_signal(parsed, signal_name, _depth=_depth + 1)
    if isinstance(value, list):
        for item in reversed(value):
            found = _extract_named_signal(item, signal_name, _depth=_depth + 1)
            if found is not None:
                return found
        return None
    if not isinstance(value, dict):
        return None
    if value.get(signal_name) not in (None, "", [], {}):
        return value.get(signal_name)
    if signal_name == "destination":
        destination = value.get("destination")
        if isinstance(destination, dict):
            return destination
        if isinstance(value.get("data"), dict):
            destination = value["data"].get("destination")
            if isinstance(destination, dict):
                return destination
    for key in (
        "data",
        "result",
        "record",
        "target",
        "content",
        "summary",
        "raw_output",
        "final_ai_content",
        "tool_results",
    ):
        nested = value.get(key)
        if nested is value:
            continue
        found = _extract_named_signal(nested, signal_name, _depth=_depth + 1)
        if found is not None:
            return found
    return None


def _compact_signal_value(value: Any, *, limit: int = 800) -> Any:
    """Keep blackboard signals compact enough for prompt context."""

    if isinstance(value, str):
        return truncate_context_text(value, limit=limit)
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    try:
        text = json.dumps(value, ensure_ascii=False)
    except Exception:
        text = str(value)
    if len(text) <= limit:
        return value
    return {
        "truncated": True,
        "preview": truncate_context_text(text, limit=limit),
    }


def _latest_task_result(task: Optional[TaskItem]) -> Optional[dict[str, Any]]:
    if not task:
        return None
    results = list(task.get("result") or [])
    if not results:
        return None
    latest = results[-1]
    return latest if isinstance(latest, dict) else {"summary": latest}


def _task_declared_outputs(task: TaskItem) -> list[str]:
    declared = [
        str(value).strip()
        for value in task.get("outputs", []) or []
        if str(value).strip()
    ]
    if declared:
        return declared
    target = task.get("target")
    if task.get("task_type") == "navigation_action" and isinstance(target, dict):
        target_type = str(target.get("type") or "").strip()
        if target_type == "semantic_keyframe":
            return ["destination", "current_place_context"]
        if target_type == "semantic_object":
            return ["destination", "selected_object"]
    if legacy_staging_kind(task):
        return ["destination", "current_place_context"]
    if legacy_object_kind(task):
        return ["destination", "selected_object"]
    return []


def build_plan_blackboard(
    tasks: dict[int, TaskItem],
    *,
    current_plan_id: Optional[str],
) -> dict[str, Any]:
    """Build current-plan working memory from declared task outputs."""

    by_task: dict[str, dict[str, Any]] = {}
    latest: dict[str, Any] = {}
    for task_id in sorted(tasks):
        task = tasks[task_id]
        if current_plan_id and task.get("plan_id") != current_plan_id:
            continue
        if task.get("status") not in {"completed", "waiting"}:
            continue
        result = _latest_task_result(task)
        if result is None:
            continue
        outputs = _task_declared_outputs(task)
        task_signals: dict[str, Any] = {}
        for output_name in outputs:
            value = _extract_named_signal(result, output_name)
            if value is None and output_name == "current_place_context":
                destination = _extract_named_signal(result, "destination")
                if destination is not None:
                    value = {"destination": destination, "source_task_id": task_id}
            if value is None:
                continue
            compact = _compact_signal_value(value)
            task_signals[output_name] = compact
            latest[output_name] = {
                "source_task_id": task_id,
                "value": compact,
            }
        if task_signals:
            by_task[str(task_id)] = {
                "description": truncate_context_text(task.get("description"), limit=180),
                "outputs": task_signals,
            }

    return {
        "by_task": by_task,
        "latest": latest,
    }


def find_relevant_arrival_context(
    current_task: Optional[TaskItem],
    tasks: dict[int, TaskItem],
    events: Sequence[EventItem],
) -> Optional[dict[str, Any]]:
    """Find the most relevant navigation-arrival context for the current task."""

    if not current_task:
        return None

    dependency_ids = list(current_task.get("depends_on", []))
    preferred_task_ids = [
        task_id
        for task_id in dependency_ids
        if is_navigation_action(tasks.get(task_id))
    ]

    for event in reversed(list(events)):
        if event.get("type") != "navigation_arrived":
            continue

        event_task_id = event.get("task_id")
        if preferred_task_ids and event_task_id not in preferred_task_ids:
            continue

        arrival_task = tasks.get(event_task_id) if event_task_id is not None else None
        if preferred_task_ids or arrival_task is not None:
            return {
                "arrived_after_task": summarize_task_for_execution_context(arrival_task),
                "arrival_event": {
                    "event_id": event.get("event_id"),
                    "summary": event.get("payload", {}).get("summary"),
                    "content": event.get("payload", {}).get("content"),
                    "created_at": event.get("created_at"),
                },
            }

    if not preferred_task_ids:
        return None

    last_nav_task = tasks.get(preferred_task_ids[-1])
    if last_nav_task is None:
        return None

    return {
        "arrived_after_task": summarize_task_for_execution_context(last_nav_task),
        "arrival_event": None,
    }


def find_recent_navigation_anchor(events: Sequence[EventItem]) -> Optional[dict[str, Any]]:
    """Return the latest arrival event as a generic navigation anchor for later tasks."""

    for event in reversed(list(events)):
        if event.get("type") != "navigation_arrived":
            continue

        payload = event.get("payload", {})
        return {
            "task_id": event.get("task_id"),
            "event_id": event.get("event_id"),
            "created_at": event.get("created_at"),
            "summary": payload.get("summary"),
            "content": payload.get("content"),
            "destination_description": payload.get("destination_description"),
            "destination_keyframe_id": payload.get("destination_keyframe_id"),
            "destination_position": payload.get("destination_position"),
        }

    return None


def build_execution_context_snapshot(
    state: AsyncAgentState,
    current_task: Optional[TaskItem],
    *,
    run_memory: Optional[Any] = None,
) -> dict[str, Any]:
    """Build the executor's small state-derived working-memory snapshot."""

    del run_memory

    tasks = state.get("tasks", {})
    events = list(state.get("events", []))
    user_inputs = list(state.get("user_inputs", []))

    current_plan_id = (
        current_task.get("plan_id")
        if current_task and current_task.get("plan_id")
        else state.get("current_plan_id")
    )
    current_user_input_id = current_task.get("user_input_id") if current_task else None

    current_user_input = None
    if current_user_input_id:
        current_user_input = next(
            (
                item
                for item in reversed(user_inputs)
                if item.get("user_input_id") == current_user_input_id
            ),
            None,
        )

    scoped_user_input_content = build_task_scoped_user_input_content(
        current_task,
        current_user_input,
    )

    dependency_ids = list(current_task.get("depends_on", [])) if current_task else []
    upstream_tasks = [
        summarize_task_for_execution_context(tasks.get(task_id))
        for task_id in dependency_ids
        if tasks.get(task_id) is not None
    ]

    arrival_context = find_relevant_arrival_context(current_task, tasks, events)

    packet = {
        "current_task": summarize_task_for_execution_context(current_task),
        "current_plan_id": current_plan_id,
        "current_user_input": {
            "user_input_id": current_user_input.get("user_input_id"),
            "content": truncate_context_text(scoped_user_input_content, limit=240),
            "attached_images": list(current_user_input.get("attached_images") or []),
        }
        if current_user_input and scoped_user_input_content
        else None,
        "arrival_context": arrival_context,
        "upstream_tasks": upstream_tasks,
        "plan_blackboard": build_plan_blackboard(
            tasks,
            current_plan_id=current_plan_id,
        ),
    }

    return packet


def build_execution_context_packet(
    state: AsyncAgentState,
    current_task: Optional[TaskItem],
    *,
    include_keys: Optional[Sequence[str]] = None,
    run_memory: Optional[Any] = None,
) -> dict[str, Any]:
    """Build a filtered execution context packet."""

    snapshot = build_execution_context_snapshot(
        state,
        current_task,
        run_memory=run_memory,
    )
    if include_keys is None:
        return snapshot
    return filter_execution_context_packet(snapshot, include_keys)


def build_tool_catalog(tools: Sequence[BaseTool]) -> str:
    """Build a compact catalog of currently available tools and their descriptions."""

    catalog_lines: list[str] = []
    for tool in tools:
        tool_name = str(getattr(tool, "name", tool.__class__.__name__)).strip()
        tool_description = truncate_context_text(
            getattr(tool, "description", None),
            limit=500,
        ) or "No description provided."
        catalog_lines.append(f"- {tool_name}: {tool_description}")

    return "\n".join(catalog_lines) if catalog_lines else "- No tools available."


def build_execution_guide(
    current_task: Optional[TaskItem],
    execution_context_packet: dict[str, Any],
    evidence_bundle: Optional[dict[str, Any]] = None,
) -> str:
    """Build one compact base execution guide for deterministic execution."""

    del evidence_bundle

    guide_lines = [
        "1. Solve only the current active task and treat other context as supporting evidence.",
        "2. Start from the small working context packet; it is not full history.",
        "3. If more information is needed, choose the smallest set of tools whose descriptions best match the task.",
        "4. Prefer tools that return deterministic values for math, coordinates, or other precision-sensitive results.",
        "5. Prefer live-state or live-observation tools when the task depends on the robot's current physical state.",
        "6. When a tool reports that a physical action has been dispatched successfully, treat that as progress on the current task and respond consistently with that state.",
        "7. For prior navigation, task history, plan edits, conversation, or observations, call query_memory with the narrowest scope and view; do not infer historical facts from working context alone.",
        "8. query_memory reads memory tables; its query argument is only an intent note, not a hard search filter. Inspect summary_table/timeline rows yourself, then request detail by row_id when relevant.",
        "9. For historical or reusable destinations, first read navigation memory summary_table. It is the primary source for visited places because it contains keyframes, positions, and route anchors.",
        "10. For navigation destination reuse, inspect only navigation detail by row_id. Do not use conversation, plan, task, or observation memory as navigation-destination evidence.",
        "11. If reusable navigation memory does not provide a concrete keyframe/position, use scene-memory search.",
        "12. For scene-memory destination search, map the intent into 3-6 concrete object/place phrases or aliases, search with those concrete variants in one request when possible, then inspect keyframe metadata to verify the target.",
        "13. If metadata verifies one candidate that satisfies the requested target, choose it and finish. If several candidates satisfy the same target, pick one directly instead of launching additional searches.",
        "14. Use keyframe image analysis only when metadata is insufficient or visual confirmation is genuinely needed.",
        "15. Tool budget for keyframe matching: normally call one scene-memory search/match tool, then inspect metadata; analyze at most three candidate images. Do not keep searching for a perfect match.",
        "16. For attached-image matching, use match_attached_image_to_keyframes with the task's image_focus. Trust its resolved recommended_keyframe_id when resolution_status is resolved; candidate evidence is for explanation/debugging.",
        "17. For attached-image object targets, treat the image as approximate scene context. Prefer staging keyframes where the target object is closer, clearer, more complete, and less occluded rather than matching every background detail.",
        "18. If current_task.target.target_source is current_view, use live/current-view tools and do not wait for historical background preanalysis.",
        "19. If current_task.target.target_source is arrived_scene or upstream_result, start from current_task.target.inputs_from and upstream task results. Do not re-select a target object that an upstream task already selected unless the signal is missing or contradictory.",
        "20. If this task is semantic object localization and an upstream selected_object is available, localize that selected object; do not repeat broad keyframe search or choose a different object.",
        "21. Respect current_task.outputs as the task boundary. If outputs does not include destination, produce only the requested observation/selection/context signals and leave destination resolution to the downstream task.",
        "22. Finish with a concise result for this task only.",
    ]

    if current_task is not None:
        task_description = str(current_task.get("description") or "").strip()
        if task_description:
            guide_lines.insert(0, f"Current task goal: {task_description}")

        current_task_id = current_task.get("task_id")
        if current_task_id is not None:
            for task in execution_context_packet.get("upcoming_tasks", []):
                if not isinstance(task, dict):
                    continue
                target = task.get("target")
                if not isinstance(target, dict):
                    continue
                if str(target.get("type") or "").strip() != "task_output":
                    continue
                try:
                    target_task_id = int(target.get("task_id"))
                    active_task_id = int(current_task_id)
                except Exception:
                    continue
                if target_task_id != active_task_id:
                    continue
                upcoming_descriptions = " ".join(
                    str(task.get("description") or "")
                    for task in execution_context_packet.get("upcoming_tasks", [])
                    if isinstance(task, dict)
                ).lower()
                staging_for_semantic_object = (
                    "approach_object_in_current_view" in upcoming_descriptions
                    or "semantic_object" in upcoming_descriptions
                    or "object localization" in upcoming_descriptions
                )
                staging_constraint_note = (
                    " If this task provides the staging destination for later semantic object localization, keep the final target and reference constraints in the scene-memory search phrases so the chosen keyframe gives a clear, close, localizable view of that final target. Prefer candidates where the final object appears reasonably large, complete, sharp, and unoccluded enough for detection, segmentation, and stereo depth; do not choose a keyframe merely because the broad landmark matches if the target object is tiny, far away, at the image edge, or only barely visible. For ordinary broad place navigation, keep the search broad and do not invent object constraints."
                    if staging_for_semantic_object
                    else ""
                )
                guide_lines.append(
                    "This task produces a reusable destination signal for a following navigation_action. Follow this workflow: (1) call query_memory with scope='navigation' and view='summary_table' to inspect reusable visited-place anchors; (2) if a navigation row clearly matches, call navigation detail for that row and reuse its keyframe/position; (3) if navigation memory returns no matching anchor, stop using runtime memory and ground the target from scene memory: map the target into concrete object/place phrases; search once with those phrases, retrieve candidate keyframe metadata with positions, and choose one candidate whose best_semantic_chunk/semantic_excerpt verifies the requested target; retrieval_score is only an initial ranking signal, not answer confidence; (4) analyze candidate images only when metadata is insufficient, and analyze at most three candidates; (5) if multiple returned keyframes satisfy the same target, prefer the keyframe where the requested target appears closer, larger, clearer, more complete, more centered, less occluded, and localizable enough for later object perception; do not treat this as current-to-keyframe route distance; (6) do not query conversation, plan, task, or observation memory, and do not call live current-state/current-image tools for this reusable destination signal. Submit the result with submit_task_result(destination={...}, summary=...)."
                    + staging_constraint_note
                )
                guide_lines.append(
                    "Reusable destination boundary: do not call navigation tools such as go_to_keyframe/go_to_position in this task, even if they appear tempting. Also do not invent a tool named destination. Submit the structured destination; the following navigation_action task will dispatch movement."
                )
                guide_lines.append(
                    "If this task uses an attached image, remember that the image is not a perfect-match template. For object targets, choose the keyframe where the target object is most visible and useful for later live object localization; minor scene-detail mismatches are acceptable."
                )
                if current_task.get("primary_target") or current_task.get("selection_policy"):
                    guide_lines.append(
                        "Structured task intent: primary_target={primary_target}; selection_policy={selection_policy}; scene_context={scene_context}. Use this intent to choose among candidate keyframes returned by tools.".format(
                            primary_target=current_task.get("primary_target") or "",
                            selection_policy=current_task.get("selection_policy") or "",
                            scene_context=current_task.get("scene_context") or "",
                        )
                    )
                guide_lines.append(
                    "For semantic destination search, use the pure target object/place as the search text. Keep spatial constraints and reference landmarks that identify the final target or improve the staging viewpoint, such as 'left of the elevator entrance' for a later semantic object fire-extinguisher-box approach. Treat phrases like 'from there', 'across the hall', 'near the previous stop', or 'after that' as route context, not required search keywords, unless the target is otherwise ambiguous."
                )
                break

    if execution_context_packet.get("arrival_context") is not None:
        guide_lines.append(
            "Arrival context is available as historical evidence that may help ground the current task."
        )

    if task_requires_current_state(current_task):
        guide_lines.append(
            "This task requires live robot state. Call get_current_state before answering; do not answer from memory alone."
        )

    if task_requires_current_position_distance(current_task):
        guide_lines.append(
            "If the task asks for distance from the current position, first get the live current position, then call calculate_distance_between_positions. Do not estimate the distance mentally."
        )

    return "\n".join(guide_lines)


def summarize_background_reference(
    bg_result: Optional[BackgroundAnalysisItem | str],
) -> Optional[str]:
    """Convert raw background cache content into a compact executor-facing summary."""

    if not bg_result:
        return None

    if isinstance(bg_result, str):
        return truncate_context_text(bg_result, limit=1200) if _background_text_is_useful(bg_result) else None

    if not isinstance(bg_result, dict):
        text = str(bg_result)
        return truncate_context_text(text, limit=1200) if _background_text_is_useful(text) else None

    status = str(bg_result.get("status") or "completed")
    summary = (
        truncate_context_text(bg_result.get("summary"), limit=420)
        if _background_text_is_useful(bg_result.get("summary"))
        else None
    )
    final_output = (
        truncate_context_text(bg_result.get("final_output"), limit=520)
        if _background_text_is_useful(bg_result.get("final_output"))
        else None
    )
    error = truncate_context_text(bg_result.get("error"), limit=320)
    candidate_keyframe_ids = [
        int(value)
        for value in list(bg_result.get("candidate_keyframe_ids", []))
        if str(value).strip().lstrip("-").isdigit()
    ]
    recommended_keyframe_id = bg_result.get("recommended_keyframe_id")
    recommended_destination = bg_result.get("recommended_destination")
    recommendation_reason = truncate_context_text(
        bg_result.get("recommendation_reason"),
        limit=220,
    )
    failure_reason = truncate_context_text(bg_result.get("failure_reason"), limit=260)
    recommendation_confidence = bg_result.get("recommendation_confidence")
    notes = [
        note
        for note in list(bg_result.get("notes", []))[-3:]
        if _background_text_is_useful(note)
    ]
    tool_observations = [
        observation
        for observation in list(bg_result.get("tool_observations", []))[-3:]
        if _background_text_is_useful(observation)
    ]

    candidate_keyframe_lines: list[str] = []
    for candidate in list(bg_result.get("candidate_keyframes", []))[:5]:
        if not isinstance(candidate, dict):
            continue
        candidate_id = candidate.get("kf_id") or candidate.get("keyframe_id")
        candidate_summary = truncate_context_text(
            candidate.get("semantics")
            or candidate.get("summary")
            or candidate.get("description"),
            limit=180,
        )
        if candidate_id is not None and candidate_summary:
            candidate_keyframe_lines.append(
                f"- keyframe {candidate_id}: {candidate_summary}"
            )

    if not (summary or final_output or notes or tool_observations or candidate_keyframe_lines):
        object_preanalysis = bg_result.get("object_preanalysis")
        if (
            candidate_keyframe_ids
            or recommended_keyframe_id is not None
            or isinstance(bg_result.get("recommended_destination"), dict)
            or failure_reason
            or isinstance(object_preanalysis, dict)
        ):
            lines = [f"Background analysis status: {status}."]
            if candidate_keyframe_ids:
                lines.append(
                    "Candidate keyframes: "
                    + ", ".join(str(item) for item in candidate_keyframe_ids[:8])
                )
            if recommended_keyframe_id is not None:
                lines.append(f"Recommended keyframe: {recommended_keyframe_id}")
            if isinstance(bg_result.get("recommended_destination"), dict):
                lines.append(
                    "Recommended destination: "
                    + truncate_context_text(bg_result.get("recommended_destination"), limit=360)
                )
            if failure_reason:
                lines.append(f"Background failure reason: {failure_reason}")
            if isinstance(object_preanalysis, dict):
                paths = object_preanalysis.get("paths") if isinstance(object_preanalysis.get("paths"), dict) else {}
                summary_json = paths.get("summary_json") or object_preanalysis.get("summary_json")
                if summary_json:
                    lines.append(f"Object preanalysis summary_json: {summary_json}")
            return "\n".join(lines)
        return None

    lines: list[str] = [f"Background analysis status: {status}."]
    if candidate_keyframe_ids:
        lines.append(
            "Candidate keyframes: "
            + ", ".join(str(item) for item in candidate_keyframe_ids[:8])
        )
    if candidate_keyframe_lines:
        lines.append("Candidate keyframe notes:")
        lines.extend(candidate_keyframe_lines)
    if recommended_keyframe_id is not None:
        recommendation_line = f"Recommended keyframe: {recommended_keyframe_id}"
        if recommendation_confidence is not None:
            recommendation_line += f" (confidence {recommendation_confidence})"
        if recommendation_reason:
            recommendation_line += f" - {recommendation_reason}"
        lines.append(recommendation_line)
    if isinstance(recommended_destination, dict):
        lines.append(
            "Recommended destination: "
            + truncate_context_text(recommended_destination, limit=360)
        )
    if failure_reason:
        lines.append(f"Background failure reason: {failure_reason}")
    object_preanalysis = bg_result.get("object_preanalysis")
    if isinstance(object_preanalysis, dict):
        paths = object_preanalysis.get("paths") if isinstance(object_preanalysis.get("paths"), dict) else {}
        summary_json = paths.get("summary_json") or object_preanalysis.get("summary_json")
        if summary_json:
            lines.append(f"Object preanalysis summary_json: {summary_json}")
    if summary:
        lines.append(f"Current summary: {summary}")
    if final_output and final_output != summary:
        lines.append(f"Final synthesized note: {final_output}")
    if notes:
        lines.append("Background model conclusions so far:")
        for note in notes:
            compact_note = truncate_context_text(note, limit=240)
            if compact_note:
                lines.append(f"- {compact_note}")
    if tool_observations:
        lines.append("Background tool results seen so far:")
        for observation in tool_observations:
            compact_observation = truncate_context_text(observation, limit=260)
            if compact_observation:
                lines.append(f"- {compact_observation}")
    if error:
        lines.append(f"Background error: {error}")

    return "\n".join(lines)


def build_background_reference(
    bg_result: Optional[BackgroundAnalysisItem | str],
) -> str:
    """Build a generic positive prompt block for background precomputation."""

    summarized_reference = summarize_background_reference(bg_result)
    if not summarized_reference:
        return ""

    return (
        "\n--- BACKGROUND REFERENCE ---\n"
        "The system prepared the following reference material while this task was pending:\n"
        f"{summarized_reference}\n\n"
        "This material may be partial if background analysis was still in progress when execution began.\n"
        "Treat background tool results and background model conclusions as prior evidence from the same scene-memory tools available to you.\n"
        "Reuse them when they already identify plausible candidates or a destination. Do not repeat the same search/metadata call unless the background evidence is missing, failed, or directly conflicts with the current task.\n"
        "--- END BACKGROUND REFERENCE ---\n"
    )


__all__ = [
    "background_result_is_reusable_for_task",
    "build_background_reference",
    "build_execution_context_snapshot",
    "build_execution_guide",
    "build_tool_catalog",
    "filter_execution_context_packet",
    "find_relevant_arrival_context",
    "find_recent_navigation_anchor",
    "is_navigation_action",
    "prepare_context_bundle",
    "should_preanalyze_future_task",
    "summarize_background_reference",
    "summarize_task_for_execution_context",
    "task_depends_on_query_result",
    "truncate_context_text",
]
