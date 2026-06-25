"""Small plan-edit helpers for insert and future replan."""

from __future__ import annotations

import json
from typing import Any, Literal, Optional, Sequence, TypedDict

from .plan_graph import MutablePlanGraph, validate_plan_graph
from .task_graph import extract_json_block, normalize_task_type
from ..runtime.types import TaskItem


class PlanEditError(ValueError):
    """Raised when a high-level plan edit is malformed or unsafe to apply."""


PlanEditMode = Literal["insert_after_current", "replan_future_after_current"]


class ParsedPlanEdit(TypedDict):
    """High-level plan edit emitted by the planner."""

    edit_type: str
    tasks: list[dict[str, Any]]
    rationale: str
    resume: str


def parse_plan_edit_from_response(
    plan_text: str,
    *,
    fallback_edit_type: str,
) -> ParsedPlanEdit:
    """Parse an insert/future-replan response."""

    cleaned_plan_text = extract_json_block(plan_text)
    plan_data = json.loads(cleaned_plan_text)
    if not isinstance(plan_data, dict):
        raise PlanEditError("Plan edit response must be a JSON object.")
    if "ops" in plan_data:
        raise PlanEditError("Plan edit response must use high-level `tasks`, not `ops`.")

    raw_tasks = plan_data.get("tasks")
    if not isinstance(raw_tasks, list):
        raise PlanEditError("Plan edit response must contain a `tasks` list.")

    return {
        "edit_type": str(plan_data.get("edit_type") or fallback_edit_type).strip(),
        "tasks": [task for task in raw_tasks if isinstance(task, dict)],
        "rationale": str(plan_data.get("rationale") or "").strip(),
        "resume": str(plan_data.get("resume") or "original_next").strip(),
    }


def apply_plan_edit(
    tasks: dict[int, TaskItem],
    *,
    edit: ParsedPlanEdit,
    plan_mode: PlanEditMode,
    plan_id: str,
    user_input_id: str,
    created_at: str,
    now_iso: Any,
    anchor_task_id: Optional[int],
    protected_task_ids: Optional[set[int]] = None,
) -> tuple[dict[int, TaskItem], set[int]]:
    """Apply either a simple insert or a full future replan after the anchor."""

    raw_tasks = edit.get("tasks", [])
    if not raw_tasks:
        raise PlanEditError("Plan edit must include at least one task.")

    graph = MutablePlanGraph.from_tasks(tasks)
    protected_task_ids = protected_task_ids or set()
    preserved_task_ids = _preservable_future_task_ids(
        graph,
        plan_id=plan_id,
        anchor_task_id=anchor_task_id,
        protected_task_ids=protected_task_ids,
    ) if plan_mode == "replan_future_after_current" else set()
    node_id_map = _build_node_id_map(
        raw_tasks,
        existing_tasks=tasks,
        preserved_task_ids=preserved_task_ids,
    )
    removed_future_task_ids: set[int] = set()
    if plan_mode == "replan_future_after_current":
        removed_future_task_ids = remove_future_after_anchor(
            graph,
            plan_id=plan_id,
            anchor_task_id=anchor_task_id,
            protected_task_ids=protected_task_ids,
        )
    touched_task_ids = _add_compiled_tasks(
        graph,
        raw_tasks,
        node_id_map=node_id_map,
        plan_id=plan_id,
        user_input_id=user_input_id,
        created_at=created_at,
    )
    new_root_id, new_leaf_ids = _subgraph_entry_and_leaves(
        raw_tasks,
        node_id_map=node_id_map,
    )
    if new_root_id is None:
        raise PlanEditError("Plan edit did not produce an entry task.")

    if plan_mode == "insert_after_current":
        _insert_after_anchor(
            graph,
            anchor_task_id=anchor_task_id,
            new_root_id=new_root_id,
            new_leaf_ids=new_leaf_ids,
            resume=edit.get("resume"),
            touched_task_ids=touched_task_ids,
        )
    elif plan_mode == "replan_future_after_current":
        connect_future_after_anchor(
            graph,
            anchor_task_id=anchor_task_id,
            new_root_id=new_root_id,
            touched_task_ids=touched_task_ids,
        )
        touched_task_ids.update(removed_future_task_ids)
    else:
        raise PlanEditError(f"Unsupported plan edit mode: {plan_mode}")

    exported_tasks = _validate_and_export(graph, plan_id=plan_id)
    timestamp = now_iso()
    for task_id in touched_task_ids:
        if task_id in exported_tasks:
            exported_tasks[task_id]["updated_at"] = timestamp
    return exported_tasks, touched_task_ids


def remove_future_after_anchor(
    graph: MutablePlanGraph,
    *,
    plan_id: str,
    anchor_task_id: Optional[int],
    protected_task_ids: set[int],
) -> set[int]:
    """Remove the entire old future suffix after one anchor task."""

    if anchor_task_id is None or anchor_task_id not in graph.nodes:
        raise PlanEditError("replan_future_after_current requires a valid anchor task.")
    old_future_roots = [
        int(edge["target"])
        for edge in graph.outgoing_edges(anchor_task_id)
    ]
    replaced_task_ids = graph.reachable_task_ids(
        start_task_ids=old_future_roots,
        plan_id=plan_id,
    )
    if protected_task_ids & replaced_task_ids:
        raise PlanEditError("Plan edit cannot replace protected current/completed tasks.")
    graph.remove_nodes(replaced_task_ids)
    graph.remove_outgoing_edges(anchor_task_id)
    return replaced_task_ids


def connect_future_after_anchor(
    graph: MutablePlanGraph,
    *,
    anchor_task_id: Optional[int],
    new_root_id: int,
    touched_task_ids: set[int],
) -> None:
    """Connect one newly compiled future subgraph after the anchor task."""

    if anchor_task_id is None or anchor_task_id not in graph.nodes:
        raise PlanEditError("replan_future_after_current requires a valid anchor task.")
    graph.set_sequence_edge(anchor_task_id, new_root_id)
    touched_task_ids.update({anchor_task_id, new_root_id})


def _coerce_task_id(raw_value: Any, *, field_name: str) -> int:
    try:
        return int(raw_value)
    except (TypeError, ValueError) as exc:
        raise PlanEditError(f"Plan edit field `{field_name}` must be an integer.") from exc


def _next_available_task_id(existing_tasks: dict[int, TaskItem]) -> int:
    return max([task_id for task_id in existing_tasks if task_id > 0] + [0]) + 1


def _build_node_id_map(
    raw_tasks: Sequence[dict[str, Any]],
    *,
    existing_tasks: dict[int, TaskItem],
    preserved_task_ids: Optional[set[int]] = None,
) -> dict[int, int]:
    preserved_task_ids = preserved_task_ids or set()
    next_task_id = _next_available_task_id(existing_tasks)
    node_id_map: dict[int, int] = {}
    assigned_ids: set[int] = set()
    for raw_task in raw_tasks:
        local_task_id = _coerce_task_id(
            raw_task.get("task_id"),
            field_name="task.task_id",
        )
        if local_task_id not in node_id_map:
            if local_task_id in preserved_task_ids and local_task_id not in assigned_ids:
                node_id_map[local_task_id] = local_task_id
                assigned_ids.add(local_task_id)
                continue
            while next_task_id in assigned_ids or next_task_id in existing_tasks:
                next_task_id += 1
            node_id_map[local_task_id] = next_task_id
            assigned_ids.add(next_task_id)
            next_task_id += 1
    return node_id_map


def _preservable_future_task_ids(
    graph: MutablePlanGraph,
    *,
    plan_id: str,
    anchor_task_id: Optional[int],
    protected_task_ids: set[int],
) -> set[int]:
    """Return existing future ids that a replan may keep stable."""

    if anchor_task_id is None or anchor_task_id not in graph.nodes:
        return set()
    future_roots = [int(edge["target"]) for edge in graph.outgoing_edges(anchor_task_id)]
    future_task_ids = graph.reachable_task_ids(
        start_task_ids=future_roots,
        plan_id=plan_id,
    )
    return future_task_ids - protected_task_ids


def _resolve_task_reference(
    raw_task_id: Any,
    *,
    node_id_map: dict[int, int],
    field_name: str,
) -> int:
    task_id = _coerce_task_id(raw_task_id, field_name=field_name)
    return node_id_map.get(task_id, task_id)


def _normalize_new_task(
    raw_task: dict[str, Any],
    *,
    task_id: int,
    plan_id: str,
    user_input_id: str,
    created_at: str,
    node_id_map: dict[int, int],
) -> TaskItem:
    description = str(raw_task.get("description") or "").strip()
    if not description:
        raise PlanEditError("Plan edit task requires description.")

    raw_depends_on = raw_task.get("depends_on") or []
    if not isinstance(raw_depends_on, list):
        raw_depends_on = [raw_depends_on]
    depends_on = [
        _resolve_task_reference(
            dependency_id,
            node_id_map=node_id_map,
            field_name="task.depends_on",
        )
        for dependency_id in raw_depends_on
    ]

    task_type = normalize_task_type(raw_task.get("task_type"))
    normalized_task: TaskItem = {
        "task_id": task_id,
        "task_type": task_type,  # type: ignore[typeddict-item]
        "type": "decision" if task_type == "decision" else "action",
        "description": description,
        "status": "pending",
        "next_task_id": None,
        "condition": raw_task.get("condition"),
        "branches": None,
        "routing_prompt": raw_task.get("routing_prompt"),
        "default_branch": raw_task.get("default_branch"),
        "plan_id": plan_id,
        "user_input_id": user_input_id,
        "depends_on": depends_on,
        "result": [],
        "created_at": created_at,
        "updated_at": created_at,
    }
    if isinstance(raw_task.get("target"), dict):
        target = dict(raw_task["target"])
        if target.get("type") == "task_output" and target.get("task_id") is not None:
            target["task_id"] = _resolve_task_reference(
                target.get("task_id"),
                node_id_map=node_id_map,
                field_name="task.target.task_id",
            )
        normalized_task["target"] = target
    if isinstance(raw_task.get("image_refs"), list):
        normalized_task["image_refs"] = [
            str(value).strip()
            for value in raw_task["image_refs"]
            if str(value).strip()
        ]
    if isinstance(raw_task.get("outputs"), list):
        normalized_task["outputs"] = [
            str(value).strip()
            for value in raw_task["outputs"]
            if str(value).strip()
        ]
    if isinstance(raw_task.get("inputs_from"), dict):
        normalized_task["inputs_from"] = dict(raw_task["inputs_from"])
    for optional_key in (
        "scene_context",
        "selection_policy",
    ):
        optional_value = raw_task.get(optional_key)
        if optional_value is not None and str(optional_value).strip():
            normalized_task[optional_key] = str(optional_value).strip()
    return normalized_task


def _add_compiled_tasks(
    graph: MutablePlanGraph,
    raw_tasks: Sequence[dict[str, Any]],
    *,
    node_id_map: dict[int, int],
    plan_id: str,
    user_input_id: str,
    created_at: str,
) -> set[int]:
    touched_task_ids: set[int] = set()
    for raw_task in raw_tasks:
        local_task_id = _coerce_task_id(
            raw_task.get("task_id"),
            field_name="task.task_id",
        )
        runtime_task_id = node_id_map[local_task_id]
        graph.add_raw_task(
            _normalize_new_task(
                raw_task,
                task_id=runtime_task_id,
                plan_id=plan_id,
                user_input_id=user_input_id,
                created_at=created_at,
                node_id_map=node_id_map,
            )
        )
        touched_task_ids.add(runtime_task_id)

    for raw_task in raw_tasks:
        local_task_id = _coerce_task_id(
            raw_task.get("task_id"),
            field_name="task.task_id",
        )
        source_id = node_id_map[local_task_id]
        next_task_id = raw_task.get("next_task_id")
        if next_task_id is not None:
            target_id = _resolve_task_reference(
                next_task_id,
                node_id_map=node_id_map,
                field_name="task.next_task_id",
            )
            if target_id not in graph.nodes:
                raise PlanEditError("Plan edit task next_task_id references a missing task.")
            graph.set_sequence_edge(source_id, target_id)
            touched_task_ids.update({source_id, target_id})
        branches = raw_task.get("branches")
        if isinstance(branches, dict):
            for label, raw_target_id in branches.items():
                target_id = _resolve_task_reference(
                    raw_target_id,
                    node_id_map=node_id_map,
                    field_name="task.branches",
                )
                if target_id not in graph.nodes:
                    raise PlanEditError("Plan edit task branch references a missing task.")
                graph.set_branch_edge(source_id, str(label), target_id)
                touched_task_ids.update({source_id, target_id})
    return touched_task_ids


def _subgraph_entry_and_leaves(
    raw_tasks: Sequence[dict[str, Any]],
    *,
    node_id_map: dict[int, int],
) -> tuple[Optional[int], list[int]]:
    source_ids: set[int] = set()
    target_ids: set[int] = set()
    runtime_task_ids: list[int] = []
    for raw_task in raw_tasks:
        local_task_id = _coerce_task_id(
            raw_task.get("task_id"),
            field_name="task.task_id",
        )
        runtime_task_id = node_id_map[local_task_id]
        runtime_task_ids.append(runtime_task_id)
        if raw_task.get("next_task_id") is not None:
            source_ids.add(runtime_task_id)
            target_ids.add(
                _resolve_task_reference(
                    raw_task.get("next_task_id"),
                    node_id_map=node_id_map,
                    field_name="task.next_task_id",
                )
            )
        branches = raw_task.get("branches")
        if isinstance(branches, dict):
            for raw_target_id in branches.values():
                source_ids.add(runtime_task_id)
                target_ids.add(
                    _resolve_task_reference(
                        raw_target_id,
                        node_id_map=node_id_map,
                        field_name="task.branches",
                    )
                )
    roots = [task_id for task_id in runtime_task_ids if task_id not in target_ids]
    leaves = [task_id for task_id in runtime_task_ids if task_id not in source_ids]
    return roots[0] if roots else None, leaves or runtime_task_ids[-1:]


def _sequence_successor(
    graph: MutablePlanGraph,
    source_task_id: int,
) -> Optional[int]:
    for edge in graph.outgoing_edges(source_task_id, kind="sequence"):
        return int(edge["target"])
    return None


def _insert_after_anchor(
    graph: MutablePlanGraph,
    *,
    anchor_task_id: Optional[int],
    new_root_id: int,
    new_leaf_ids: Sequence[int],
    resume: str,
    touched_task_ids: set[int],
) -> None:
    if anchor_task_id is None or anchor_task_id not in graph.nodes:
        raise PlanEditError("insert_after_current requires a valid anchor task.")
    original_next_task_id = _sequence_successor(graph, anchor_task_id)
    graph.set_sequence_edge(anchor_task_id, new_root_id)
    touched_task_ids.update({anchor_task_id, new_root_id})
    if resume != "none" and original_next_task_id is not None:
        for leaf_task_id in new_leaf_ids:
            graph.set_sequence_edge(leaf_task_id, original_next_task_id)
            touched_task_ids.update({leaf_task_id, original_next_task_id})


def _validate_and_export(
    graph: MutablePlanGraph,
    *,
    plan_id: str,
) -> dict[int, TaskItem]:
    exported_tasks = graph.to_tasks()
    errors = [
        issue
        for issue in validate_plan_graph(exported_tasks, plan_id=plan_id)
        if issue["severity"] == "error"
    ]
    if errors:
        raise PlanEditError(f"Plan edit produced an invalid graph: {errors[0]}")
    return exported_tasks


__all__ = [
    "connect_future_after_anchor",
    "ParsedPlanEdit",
    "PlanEditError",
    "PlanEditMode",
    "apply_plan_edit",
    "parse_plan_edit_from_response",
    "remove_future_after_anchor",
]
