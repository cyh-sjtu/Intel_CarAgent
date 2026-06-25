"""Pure task-graph and planner-parsing helpers for the async agent."""

from __future__ import annotations

import json
import re
from typing import Any, Optional, Sequence

from ..runtime.types import TaskItem


def extract_json_block(text: str) -> str:
    """Extract a JSON object from raw text when fenced blocks are used."""

    stripped = text.strip()
    if "```json" in stripped:
        return stripped.split("```json", 1)[1].split("```", 1)[0].strip()
    if "```" in stripped:
        return stripped.split("```", 1)[1].split("```", 1)[0].strip()
    return stripped


def normalize_text_for_matching(text: str) -> str:
    """Normalize free-form text for forgiving task-description matching."""

    lowered = text.lower()
    lowered = re.sub(r"[^a-z0-9]+", " ", lowered)
    return " ".join(part for part in lowered.split() if part)


def normalize_task_type(value: Any) -> str:
    """Normalize planner task_type using only the new task schema."""

    normalized = str(value or "").strip().lower()
    if normalized in {"llm_action", "navigation_action", "decision"}:
        return normalized

    return "llm_action"


def normalize_task_type_from_planner(task_info: dict[str, Any]) -> str:
    """Normalize a planner task using only the new task_type contract."""

    return normalize_task_type(task_info.get("task_type"))


def derive_task_dependencies(tasks: dict[int, TaskItem]) -> dict[int, list[int]]:
    """Derive graph-inbound task ids without mutating explicit depends_on."""

    dependency_map: dict[int, list[int]] = {task_id: [] for task_id in tasks}

    for task_id, task in tasks.items():
        next_task_id = task.get("next_task_id")
        if next_task_id is not None and next_task_id in dependency_map:
            dependency_map[next_task_id].append(task_id)

        for branch_target in (task.get("branches") or {}).values():
            if branch_target is not None and branch_target in dependency_map:
                dependency_map[branch_target].append(task_id)

    return dependency_map


def count_task_inbound_references(tasks: dict[int, TaskItem]) -> dict[int, int]:
    """Count how many task-graph edges point at each task."""

    inbound_counts = {task_id: 0 for task_id in tasks}
    for task in tasks.values():
        next_task_id = task.get("next_task_id")
        if next_task_id in inbound_counts:
            inbound_counts[next_task_id] += 1

        for branch_target in (task.get("branches") or {}).values():
            if branch_target in inbound_counts:
                inbound_counts[branch_target] += 1

    return inbound_counts


def collect_ordered_task_ids_for_plan(
    tasks: dict[int, TaskItem],
    *,
    plan_id: Optional[str],
) -> list[int]:
    """Return a stable dependency-respecting order for tasks in one plan scope."""

    scoped_tasks = {
        task_id: task
        for task_id, task in tasks.items()
        if plan_id is None or task.get("plan_id") == plan_id
    }
    if not scoped_tasks:
        return []

    inbound_counts = count_task_inbound_references(scoped_tasks)
    ordered_task_ids: list[int] = []
    remaining_inbound = dict(inbound_counts)
    outgoing_edges: dict[int, list[int]] = {task_id: [] for task_id in scoped_tasks}

    for task_id, task in scoped_tasks.items():
        next_task_id = task.get("next_task_id")
        if next_task_id is not None and next_task_id in scoped_tasks:
            outgoing_edges[task_id].append(next_task_id)

        for _, branch_target in sorted((task.get("branches") or {}).items()):
            if (
                branch_target is not None
                and branch_target in scoped_tasks
                and branch_target not in outgoing_edges[task_id]
            ):
                outgoing_edges[task_id].append(branch_target)

    ready_task_ids = sorted(
        task_id
        for task_id, inbound_count in remaining_inbound.items()
        if inbound_count == 0
    )
    if not ready_task_ids:
        ready_task_ids = sorted(scoped_tasks)

    while ready_task_ids:
        task_id = ready_task_ids.pop(0)
        if task_id in ordered_task_ids:
            continue

        ordered_task_ids.append(task_id)

        for child_task_id in outgoing_edges.get(task_id, []):
            remaining_inbound[child_task_id] -= 1
            if remaining_inbound[child_task_id] == 0:
                ready_task_ids.append(child_task_id)
        ready_task_ids.sort()

    for task_id in sorted(scoped_tasks):
        if task_id not in ordered_task_ids:
            ordered_task_ids.append(task_id)

    return ordered_task_ids


def get_task_progress_context(
    tasks: dict[int, TaskItem],
    *,
    current_task_id: Optional[int],
    current_plan_id: Optional[str],
) -> Optional[dict[str, Any]]:
    """Compute a human-friendly execution position for the current task."""

    if current_task_id is None or current_task_id not in tasks:
        return None

    current_task = tasks[current_task_id]
    if current_task.get("inserted"):
        ordered_task_ids = [current_task_id]
    else:
        scoped_plan_id = current_task.get("plan_id") or current_plan_id
        ordered_task_ids = collect_ordered_task_ids_for_plan(
            tasks,
            plan_id=scoped_plan_id,
        )
        if not ordered_task_ids:
            ordered_task_ids = [current_task_id]

    if current_task_id not in ordered_task_ids:
        ordered_task_ids.append(current_task_id)

    return {
        "task_id": current_task_id,
        "position": ordered_task_ids.index(current_task_id) + 1,
        "total": len(ordered_task_ids),
        "ordered_task_ids": ordered_task_ids,
    }


def slugify_branch_label(text: str, *, fallback: str) -> str:
    """Convert free-form text into a compact branch label."""

    lowered = text.lower()
    lowered = re.sub(r"[^a-z0-9]+", "_", lowered)
    lowered = lowered.strip("_")
    if not lowered:
        return fallback

    parts = [part for part in lowered.split("_") if part]
    if not parts:
        return fallback

    return "_".join(parts[:8])


def make_unique_branch_label(label: str, used_labels: set[str]) -> str:
    """Ensure branch labels stay unique within one decision task."""

    if label not in used_labels:
        used_labels.add(label)
        return label

    suffix = 2
    candidate = f"{label}_{suffix}"
    while candidate in used_labels:
        suffix += 1
        candidate = f"{label}_{suffix}"
    used_labels.add(candidate)
    return candidate


def decision_condition_text(task: TaskItem) -> str:
    """Extract the most useful human-readable condition for a decision task."""

    condition_text = (
        task.get("condition")
        or task.get("routing_prompt")
        or task.get("description")
        or ""
    )
    normalized = str(condition_text).strip()
    lowered = normalized.lower()
    for prefix in (
        "check if ",
        "check whether ",
        "determine if ",
        "determine whether ",
        "decide if ",
        "decide whether ",
    ):
        if lowered.startswith(prefix):
            normalized = normalized[len(prefix) :].strip()
            break

    return normalized.rstrip(".?")


def target_branch_label(
    target_task_id: int,
    tasks: dict[int, TaskItem],
    *,
    fallback_prefix: str,
) -> str:
    """Build a stable branch label from the branch target task."""

    target_task = tasks.get(target_task_id)
    if target_task is not None:
        description = str(target_task.get("description") or "").strip()
        if description:
            return slugify_branch_label(
                description,
                fallback=f"{fallback_prefix}_{target_task_id}",
            )
    return f"{fallback_prefix}_{target_task_id}"


def _coerce_branch_target_id(raw_target: Any) -> Optional[int]:
    """Normalize branch targets to integer task ids.

    Planner output occasionally wraps branch targets as
    {"next_task_id": 5}.  Runtime graph code expects branches to be
    label -> int, so unwrap that structural variant here instead of letting a
    dict flow into graph traversal.
    """

    if isinstance(raw_target, dict):
        raw_target = raw_target.get("next_task_id") or raw_target.get("task_id")
    try:
        return int(raw_target)
    except Exception:
        return None


def normalize_task_branches(tasks: dict[int, TaskItem]) -> None:
    """Ensure every task's branches map labels to valid integer task ids."""

    for task_id, task in tasks.items():
        raw_branches = task.get("branches")
        if not isinstance(raw_branches, dict):
            if raw_branches is not None:
                task["branches"] = None
            continue
        valid_branches: dict[str, int] = {}
        for raw_label, raw_target in raw_branches.items():
            target_id = _coerce_branch_target_id(raw_target)
            if target_id is None or target_id not in tasks or target_id == task_id:
                continue
            valid_branches[str(raw_label).strip() or f"branch_{target_id}"] = target_id
        task["branches"] = valid_branches or None


def repair_missing_decision_branches(tasks: dict[int, TaskItem]) -> None:
    """Infer missing decision branch entries from explicit downstream deps.

    The planner prompt requires decision tasks to provide `branches`, but LLM
    output can omit that field while still encoding branch roots via
    `depends_on: [decision_task_id]`.  Repair only that structural omission:
    no keyword matching, no semantic branch choice.
    """

    for task_id, task in tasks.items():
        if task.get("task_type") != "decision":
            continue

        raw_branches = task.get("branches")
        valid_branches: dict[str, int] = {}
        if isinstance(raw_branches, dict):
            for raw_label, raw_target in raw_branches.items():
                target_id = _coerce_branch_target_id(raw_target)
                if target_id is None:
                    continue
                if target_id in tasks and target_id != task_id:
                    valid_branches[str(raw_label).strip() or f"branch_{target_id}"] = target_id
        if valid_branches:
            task["branches"] = valid_branches
            continue

        inferred_targets: list[int] = []
        for candidate_id, candidate in tasks.items():
            if candidate_id == task_id:
                continue
            try:
                dependencies = {int(value) for value in candidate.get("depends_on", []) or []}
            except Exception:
                dependencies = set()
            if task_id in dependencies:
                inferred_targets.append(candidate_id)

        inferred_branches: dict[str, int] = {}
        used_labels: set[str] = set()
        for index, target_id in enumerate(sorted(set(inferred_targets)), start=1):
            label = target_branch_label(
                target_id,
                tasks,
                fallback_prefix=f"branch_{index}",
            )
            inferred_branches[make_unique_branch_label(label, used_labels)] = target_id

        if inferred_branches:
            task["branches"] = inferred_branches
            if task.get("default_branch") not in inferred_branches:
                task["default_branch"] = next(iter(inferred_branches))


def parse_planned_tasks_from_response(
    plan_text: str,
    *,
    plan_id: str,
    user_input_id: str,
    created_at: str,
) -> tuple[dict[int, TaskItem], Optional[int]]:
    """Parse planner JSON into normalized task records."""

    cleaned_plan_text = extract_json_block(plan_text)
    plan_data = json.loads(cleaned_plan_text)
    parsed_tasks: dict[int, TaskItem] = {}
    first_task_id: Optional[int] = None

    for index, task_info in enumerate(plan_data.get("tasks", [])):
        task_id = task_info["task_id"]
        if index == 0:
            first_task_id = task_id
        task_type = normalize_task_type_from_planner(task_info)

        parsed_tasks[task_id] = {
            "task_id": task_id,
            "task_type": task_type,
            "type": "decision" if task_type == "decision" else "action",
            "description": task_info["description"],
            "status": "pending",
            "next_task_id": task_info.get("next_task_id"),
            "condition": task_info.get("condition"),
            "branches": task_info.get("branches"),
            "routing_prompt": task_info.get("routing_prompt"),
            "default_branch": task_info.get("default_branch"),
            "plan_id": plan_id,
            "user_input_id": user_input_id,
            "depends_on": list(task_info.get("depends_on") or []),
            "result": [],
            "created_at": created_at,
            "updated_at": created_at,
        }
        if isinstance(task_info.get("outputs"), list):
            parsed_tasks[task_id]["outputs"] = [
                str(value).strip()
                for value in task_info["outputs"]
                if str(value).strip()
            ]
        if isinstance(task_info.get("inputs_from"), dict):
            parsed_tasks[task_id]["inputs_from"] = dict(task_info["inputs_from"])
        if isinstance(task_info.get("target"), dict):
            parsed_tasks[task_id]["target"] = dict(task_info["target"])
        if isinstance(task_info.get("image_refs"), list):
            parsed_tasks[task_id]["image_refs"] = [
                str(value).strip()
                for value in task_info["image_refs"]
                if str(value).strip()
            ]
        for optional_key in (
            "scene_context",
            "selection_policy",
        ):
            optional_value = task_info.get(optional_key)
            if optional_value is not None and str(optional_value).strip():
                parsed_tasks[task_id][optional_key] = str(optional_value).strip()

    if first_task_id not in parsed_tasks:
        first_task_id = min(parsed_tasks) if parsed_tasks else None

    normalize_task_branches(parsed_tasks)
    repair_missing_decision_branches(parsed_tasks)

    return parsed_tasks, first_task_id


def remap_plan_task_ids(
    tasks_to_remap: dict[int, TaskItem],
    existing_tasks: dict[int, TaskItem],
    *,
    first_task_id: Optional[int] = None,
) -> tuple[dict[int, TaskItem], Optional[int]]:
    """Remap planner-local task ids so they can be merged into an existing plan."""

    if not tasks_to_remap:
        return {}, None

    next_positive_id = max([task_id for task_id in existing_tasks if task_id > 0] + [0]) + 1
    task_id_map: dict[int, int] = {}
    for previous_task_id in sorted(tasks_to_remap):
        task_id_map[previous_task_id] = next_positive_id
        next_positive_id += 1

    remapped_tasks: dict[int, TaskItem] = {}
    for previous_task_id, task in tasks_to_remap.items():
        new_task_id = task_id_map[previous_task_id]
        remapped_task: TaskItem = {**task, "task_id": new_task_id}
        next_task_id = task.get("next_task_id")
        remapped_task["next_task_id"] = (
            task_id_map[next_task_id]
            if next_task_id is not None and next_task_id in task_id_map
            else next_task_id
        )
        branches = task.get("branches") or None
        if branches is not None:
            remapped_task["branches"] = {
                branch_label: task_id_map.get(target_id, target_id)
                for branch_label, target_id in branches.items()
            }
        remapped_depends_on = []
        for dependency_id in task.get("depends_on", []) or []:
            remapped_depends_on.append(task_id_map.get(dependency_id, dependency_id))
        remapped_task["depends_on"] = remapped_depends_on
        target = task.get("target")
        if isinstance(target, dict):
            remapped_target = dict(target)
            if remapped_target.get("type") == "task_output":
                source_task_id = remapped_target.get("task_id")
                if source_task_id in task_id_map:
                    remapped_target["task_id"] = task_id_map[source_task_id]
            remapped_task["target"] = remapped_target
        if isinstance(task.get("image_refs"), list):
            remapped_task["image_refs"] = list(task.get("image_refs") or [])
        if isinstance(task.get("outputs"), list):
            remapped_task["outputs"] = list(task.get("outputs") or [])
        if isinstance(task.get("inputs_from"), dict):
            remapped_task["inputs_from"] = dict(task.get("inputs_from") or {})
        for optional_key in (
            "scene_context",
            "selection_policy",
        ):
            if task.get(optional_key) is not None:
                remapped_task[optional_key] = str(task.get(optional_key) or "")
        remapped_tasks[new_task_id] = remapped_task

    source_first_task_id = first_task_id if first_task_id in task_id_map else None
    if source_first_task_id is None and tasks_to_remap:
        source_first_task_id = next(iter(tasks_to_remap))
    remapped_first_task_id = (
        task_id_map.get(source_first_task_id)
        if source_first_task_id is not None
        else None
    )
    return remapped_tasks, remapped_first_task_id


def collect_reachable_future_task_ids(
    tasks: dict[int, TaskItem],
    *,
    start_task_ids: Sequence[Optional[int]],
    plan_id: Optional[str],
) -> set[int]:
    """Collect future tasks reachable from one or more plan entry points."""

    reachable_task_ids: set[int] = set()
    stack = [task_id for task_id in start_task_ids if task_id is not None]

    while stack:
        task_id = stack.pop()
        if task_id in reachable_task_ids or task_id not in tasks:
            continue
        task = tasks[task_id]
        if plan_id and task.get("plan_id") != plan_id:
            continue
        reachable_task_ids.add(task_id)
        next_task_id = task.get("next_task_id")
        if next_task_id is not None:
            stack.append(next_task_id)
        for branch_target in (task.get("branches") or {}).values():
            if branch_target is not None:
                stack.append(branch_target)

    return reachable_task_ids


def find_plan_leaf_task_ids(tasks: dict[int, TaskItem]) -> list[int]:
    """Return task ids that have no outgoing edge inside the provided task graph."""

    leaf_task_ids: list[int] = []
    for task_id, task in tasks.items():
        branches = task.get("branches") or {}
        if task.get("next_task_id") is None and not branches:
            leaf_task_ids.append(task_id)
    return sorted(leaf_task_ids)


def recompute_dependencies_for_all_tasks(tasks: dict[int, TaskItem]) -> None:
    """Normalize explicit depends_on lists without deriving control-flow edges."""

    for task in tasks.values():
        task["depends_on"] = list(task.get("depends_on") or [])


def collect_plan_root_task_ids(
    tasks: dict[int, TaskItem],
    *,
    plan_id: Optional[str],
) -> list[int]:
    """Return true graph roots for one plan scope without forcing disconnected tasks in."""

    scoped_tasks = {
        task_id: task
        for task_id, task in tasks.items()
        if plan_id is None or task.get("plan_id") == plan_id
    }
    if not scoped_tasks:
        return []

    inbound_counts = count_task_inbound_references(scoped_tasks)
    root_task_ids = sorted(
        task_id for task_id, inbound_count in inbound_counts.items() if inbound_count == 0
    )
    if root_task_ids:
        return root_task_ids

    return sorted(scoped_tasks)


__all__ = [
    "collect_ordered_task_ids_for_plan",
    "collect_plan_root_task_ids",
    "collect_reachable_future_task_ids",
    "count_task_inbound_references",
    "decision_condition_text",
    "derive_task_dependencies",
    "extract_json_block",
    "find_plan_leaf_task_ids",
    "get_task_progress_context",
    "make_unique_branch_label",
    "normalize_task_type",
    "normalize_task_type_from_planner",
    "normalize_text_for_matching",
    "parse_planned_tasks_from_response",
    "recompute_dependencies_for_all_tasks",
    "repair_missing_decision_branches",
    "remap_plan_task_ids",
    "slugify_branch_label",
    "target_branch_label",
]
