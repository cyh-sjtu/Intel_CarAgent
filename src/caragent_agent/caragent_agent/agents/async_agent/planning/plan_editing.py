"""Plan-editing helpers for in-place async-agent plan mutation."""

from __future__ import annotations

from typing import Callable, Optional

from .task_graph import (
    collect_reachable_future_task_ids,
    find_plan_leaf_task_ids,
    normalize_text_for_matching,
    recompute_dependencies_for_all_tasks,
)
from ..runtime.types import TaskItem


def resolve_future_target_task_id(
    tasks: dict[int, TaskItem],
    *,
    anchor_task_id: int,
    target_task_id: Optional[int],
    target_task_description: Optional[str],
) -> Optional[int]:
    """Resolve one future task relative to the anchor using id or fuzzy description match."""

    if anchor_task_id not in tasks:
        return None

    anchor_task = tasks[anchor_task_id]
    plan_id = anchor_task.get("plan_id")
    future_task_ids = collect_reachable_future_task_ids(
        tasks,
        start_task_ids=[anchor_task.get("next_task_id")],
        plan_id=plan_id,
    )
    if not future_task_ids:
        return None

    if target_task_id in future_task_ids:
        return target_task_id

    if not target_task_description:
        return None

    expected_text = normalize_text_for_matching(target_task_description)
    for future_task_id in sorted(future_task_ids):
        description = str(tasks[future_task_id].get("description") or "")
        normalized_description = normalize_text_for_matching(description)
        if expected_text and (
            expected_text in normalized_description
            or normalized_description in expected_text
        ):
            return future_task_id

    return None


def descriptions_likely_describe_same_operation(
    left: Optional[str],
    right: Optional[str],
) -> bool:
    """Heuristically detect when two task descriptions describe the same underlying step."""

    left_normalized = normalize_text_for_matching(str(left or ""))
    right_normalized = normalize_text_for_matching(str(right or ""))
    if not left_normalized or not right_normalized:
        return False

    if left_normalized in right_normalized or right_normalized in left_normalized:
        return True

    left_tokens = {token for token in left_normalized.split() if len(token) >= 3}
    right_tokens = {token for token in right_normalized.split() if len(token) >= 3}
    shared_tokens = left_tokens & right_tokens

    return len(shared_tokens) >= 4


def expand_targeted_replacement_start_task_id(
    tasks: dict[int, TaskItem],
    *,
    anchor_task_id: int,
    target_task_id: int,
    replacement_tasks: dict[int, TaskItem],
    replacement_first_task_id: Optional[int],
    count_task_inbound_references: Callable[[dict[int, TaskItem]], dict[int, int]],
) -> int:
    """Expand a targeted replacement backward when the new root subsumes a unique linear predecessor."""

    if target_task_id not in tasks or replacement_first_task_id is None:
        return target_task_id

    replacement_root = replacement_tasks.get(replacement_first_task_id)
    target_task = tasks.get(target_task_id)
    if replacement_root is None or target_task is None:
        return target_task_id

    if target_task.get("type") != "decision" or replacement_root.get("type") != "decision":
        return target_task_id

    inbound_counts = count_task_inbound_references(tasks)
    current_target_task_id = target_task_id

    while True:
        predecessor_task_ids = [
            task_id
            for task_id, task in tasks.items()
            if task.get("next_task_id") == current_target_task_id
        ]
        if len(predecessor_task_ids) != 1:
            break

        predecessor_task_id = predecessor_task_ids[0]
        if predecessor_task_id == anchor_task_id or predecessor_task_id not in tasks:
            break

        predecessor_task = tasks[predecessor_task_id]
        if predecessor_task.get("type") != "action":
            break
        if predecessor_task.get("branches"):
            break
        if inbound_counts.get(current_target_task_id, 0) != 1:
            break
        if not descriptions_likely_describe_same_operation(
            predecessor_task.get("description"),
            replacement_root.get("description"),
        ):
            break

        current_target_task_id = predecessor_task_id

    return current_target_task_id


def apply_targeted_future_task_replacement(
    tasks: dict[int, TaskItem],
    *,
    target_task_id: int,
    replacement_tasks: dict[int, TaskItem],
    replacement_first_task_id: Optional[int],
    now_iso: Callable[[], str],
) -> dict[int, TaskItem]:
    """Replace one future task/subgraph while preserving only explicitly reused old nodes."""

    if target_task_id not in tasks:
        return tasks

    target_task = tasks[target_task_id]
    plan_id = target_task.get("plan_id")
    old_subtree_task_ids = collect_reachable_future_task_ids(
        tasks,
        start_task_ids=[target_task_id],
        plan_id=plan_id,
    )

    original_old_subtree_tasks: dict[int, TaskItem] = {
        task_id: dict(tasks[task_id])
        for task_id in old_subtree_task_ids
        if task_id in tasks
    }
    updated_tasks: dict[int, TaskItem] = {
        task_id: dict(task)
        for task_id, task in tasks.items()
        if task_id not in old_subtree_task_ids
    }

    for task_id, task in list(updated_tasks.items()):
        if task.get("next_task_id") == target_task_id:
            task["next_task_id"] = replacement_first_task_id
            task["updated_at"] = now_iso()

        branches = task.get("branches") or None
        if branches is not None and target_task_id in branches.values():
            task["branches"] = {
                branch_label: (
                    replacement_first_task_id
                    if branch_target == target_task_id
                    else branch_target
                )
                for branch_label, branch_target in branches.items()
                if branch_target != target_task_id or replacement_first_task_id is not None
            }
            task["updated_at"] = now_iso()

    for task in replacement_tasks.values():
        updated_tasks[task["task_id"]] = task

    preserved_old_entry_ids: set[int] = set()
    for task in updated_tasks.values():
        next_task_id = task.get("next_task_id")
        if next_task_id in old_subtree_task_ids:
            preserved_old_entry_ids.add(next_task_id)

        for branch_target in (task.get("branches") or {}).values():
            if branch_target in old_subtree_task_ids:
                preserved_old_entry_ids.add(branch_target)

    preserved_old_task_ids = collect_reachable_future_task_ids(
        original_old_subtree_tasks,
        start_task_ids=sorted(preserved_old_entry_ids),
        plan_id=plan_id,
    )

    for preserved_task_id in preserved_old_task_ids:
        original_task = original_old_subtree_tasks.get(preserved_task_id)
        if original_task is not None:
            updated_tasks[preserved_task_id] = dict(original_task)

    recompute_dependencies_for_all_tasks(updated_tasks)
    return updated_tasks


def apply_insert_after_current(
    tasks: dict[int, TaskItem],
    *,
    anchor_task_id: int,
    inserted_tasks: dict[int, TaskItem],
    inserted_first_task_id: Optional[int],
    now_iso: Callable[[], str],
) -> dict[int, TaskItem]:
    """Insert a planned subgraph immediately after the anchor task."""

    if anchor_task_id not in tasks or not inserted_tasks or inserted_first_task_id is None:
        return tasks

    updated_tasks = dict(tasks)
    anchor_task = updated_tasks[anchor_task_id]
    original_next_task_id = anchor_task.get("next_task_id")
    anchor_task["next_task_id"] = inserted_first_task_id
    anchor_task["updated_at"] = now_iso()

    for task in inserted_tasks.values():
        updated_tasks[task["task_id"]] = task

    for leaf_task_id in find_plan_leaf_task_ids(inserted_tasks):
        updated_tasks[leaf_task_id]["next_task_id"] = original_next_task_id
        updated_tasks[leaf_task_id]["updated_at"] = now_iso()

    recompute_dependencies_for_all_tasks(updated_tasks)
    return updated_tasks


def apply_replace_remaining_plan(
    tasks: dict[int, TaskItem],
    *,
    anchor_task_id: int,
    replacement_tasks: dict[int, TaskItem],
    replacement_first_task_id: Optional[int],
    target_task_id: Optional[int],
    target_task_description: Optional[str],
    now_iso: Callable[[], str],
    count_task_inbound_references: Callable[[dict[int, TaskItem]], dict[int, int]],
) -> dict[int, TaskItem]:
    """Replace the reachable future suffix, or one targeted future subgraph, after the anchor task."""

    if anchor_task_id not in tasks:
        return tasks

    resolved_target_task_id = resolve_future_target_task_id(
        tasks,
        anchor_task_id=anchor_task_id,
        target_task_id=target_task_id,
        target_task_description=target_task_description,
    )
    if resolved_target_task_id is not None:
        target_task = tasks.get(resolved_target_task_id)
        replacement_root = (
            replacement_tasks.get(replacement_first_task_id)
            if replacement_first_task_id is not None
            else None
        )
        if (
            target_task is None
            or target_task.get("type") != "decision"
            or replacement_root is None
            or replacement_root.get("type") == "decision"
        ):
            expanded_target_task_id = expand_targeted_replacement_start_task_id(
                tasks,
                anchor_task_id=anchor_task_id,
                target_task_id=resolved_target_task_id,
                replacement_tasks=replacement_tasks,
                replacement_first_task_id=replacement_first_task_id,
                count_task_inbound_references=count_task_inbound_references,
            )
            return apply_targeted_future_task_replacement(
                tasks,
                target_task_id=expanded_target_task_id,
                replacement_tasks=replacement_tasks,
                replacement_first_task_id=replacement_first_task_id,
                now_iso=now_iso,
            )

    anchor_task = tasks[anchor_task_id]
    plan_id = anchor_task.get("plan_id")
    suffix_start_ids = [anchor_task.get("next_task_id")]
    replaced_task_ids = collect_reachable_future_task_ids(
        tasks,
        start_task_ids=suffix_start_ids,
        plan_id=plan_id,
    )

    updated_tasks = {
        task_id: task
        for task_id, task in tasks.items()
        if task_id not in replaced_task_ids
    }
    if anchor_task_id not in updated_tasks:
        return tasks

    updated_tasks[anchor_task_id]["next_task_id"] = replacement_first_task_id
    updated_tasks[anchor_task_id]["updated_at"] = now_iso()

    for task in replacement_tasks.values():
        updated_tasks[task["task_id"]] = task

    recompute_dependencies_for_all_tasks(updated_tasks)
    return updated_tasks


def apply_delete_future_task(
    tasks: dict[int, TaskItem],
    *,
    current_task_id: Optional[int],
    current_plan_id: Optional[str],
    target_task_id: Optional[int],
    target_task_description: Optional[str],
    now_iso: Callable[[], str],
) -> dict[int, TaskItem]:
    """Delete one identifiable future task from the active plan."""

    if current_task_id is None or current_task_id not in tasks or current_plan_id is None:
        return tasks

    current_task = tasks[current_task_id]
    future_task_ids = collect_reachable_future_task_ids(
        tasks,
        start_task_ids=[current_task.get("next_task_id")],
        plan_id=current_plan_id,
    )
    if not future_task_ids:
        return tasks

    resolved_target_task_id = None
    if target_task_id in future_task_ids:
        resolved_target_task_id = target_task_id
    elif target_task_description:
        expected_text = normalize_text_for_matching(target_task_description)
        for future_task_id in sorted(future_task_ids):
            description = str(tasks[future_task_id].get("description") or "")
            normalized_description = normalize_text_for_matching(description)
            if expected_text and (
                expected_text in normalized_description
                or normalized_description in expected_text
            ):
                resolved_target_task_id = future_task_id
                break

    if resolved_target_task_id is None or resolved_target_task_id not in tasks:
        return tasks

    deleted_task = tasks[resolved_target_task_id]
    replacement_next_task_id = deleted_task.get("next_task_id")
    updated_tasks = {
        task_id: task
        for task_id, task in tasks.items()
        if task_id != resolved_target_task_id
    }

    for task in updated_tasks.values():
        if task.get("next_task_id") == resolved_target_task_id:
            task["next_task_id"] = replacement_next_task_id
            task["updated_at"] = now_iso()
        branches = task.get("branches") or None
        if branches is not None and resolved_target_task_id in branches.values():
            task["branches"] = {
                branch_label: (
                    replacement_next_task_id
                    if branch_target == resolved_target_task_id
                    else branch_target
                )
                for branch_label, branch_target in branches.items()
                if branch_target != resolved_target_task_id or replacement_next_task_id is not None
            }
            task["updated_at"] = now_iso()

    recompute_dependencies_for_all_tasks(updated_tasks)
    return updated_tasks


__all__ = [
    "apply_delete_future_task",
    "apply_insert_after_current",
    "apply_replace_remaining_plan",
    "apply_targeted_future_task_replacement",
    "descriptions_likely_describe_same_operation",
    "expand_targeted_replacement_start_task_id",
    "resolve_future_target_task_id",
]
