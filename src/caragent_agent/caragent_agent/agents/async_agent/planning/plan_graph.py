"""Read-only PlanGraph adapter for async-agent task dictionaries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Literal, Optional, Sequence

import networkx as nx
from typing_extensions import NotRequired, TypedDict

from ..runtime.types import TaskItem


PlanEdgeKind = Literal["sequence", "branch"]
PlanIssueSeverity = Literal["error", "warning"]


class PlanGraphEdge(TypedDict):
    """One explicit control-flow edge derived from a TaskItem field."""

    source: int
    target: int
    kind: PlanEdgeKind
    label: NotRequired[str]


class PlanGraphIssue(TypedDict):
    """One validation issue found in a task graph."""

    severity: PlanIssueSeverity
    code: str
    task_id: int
    message: str
    details: dict[str, Any]


@dataclass
class MutablePlanGraph:
    """Mutable node/edge plan graph that can export back to TaskItem dictionaries."""

    nodes: dict[int, TaskItem]
    edges: list[PlanGraphEdge]

    @classmethod
    def from_tasks(
        cls,
        tasks: dict[int, TaskItem],
        *,
        plan_id: Optional[str] = None,
    ) -> "MutablePlanGraph":
        """Build a mutable graph from the current task-dict runtime format."""

        scoped_nodes = {
            task_id: dict(task)
            for task_id, task in tasks.items()
            if plan_id is None or task.get("plan_id") == plan_id
        }
        return cls(nodes=scoped_nodes, edges=iter_plan_edges(scoped_nodes))

    def copy(self) -> "MutablePlanGraph":
        """Return an independent mutable graph copy."""

        return MutablePlanGraph(
            nodes={task_id: dict(task) for task_id, task in self.nodes.items()},
            edges=[dict(edge) for edge in self.edges],
        )

    def has_node(self, task_id: Optional[int]) -> bool:
        """Return True when the graph contains the task id."""

        return task_id is not None and task_id in self.nodes

    def add_task(self, task: TaskItem) -> None:
        """Add or replace one task node and import its embedded outgoing edges."""

        task_id = int(task["task_id"])
        self.nodes[task_id] = dict(task)
        self.remove_outgoing_edges(task_id)
        for edge in iter_plan_edges({task_id: task}):
            self.edges.append(edge)

    def add_raw_task(self, task: TaskItem) -> None:
        """Add one task node without importing embedded edge fields."""

        self.nodes[int(task["task_id"])] = dict(task)

    def add_raw_tasks(self, tasks: dict[int, TaskItem]) -> None:
        """Add several task nodes without importing embedded edge fields."""

        for task_id in sorted(tasks):
            self.add_raw_task(tasks[task_id])

    def add_tasks(self, tasks: dict[int, TaskItem]) -> None:
        """Add several task nodes."""

        for task_id in sorted(tasks):
            self.add_task(tasks[task_id])

    def remove_nodes(self, task_ids: Iterable[int]) -> None:
        """Remove nodes and every edge touching them."""

        removed = {int(task_id) for task_id in task_ids}
        for task_id in removed:
            self.nodes.pop(task_id, None)
        for task in self.nodes.values():
            task["depends_on"] = [
                dependency_id
                for dependency_id in list(task.get("depends_on", []))
                if dependency_id not in removed
            ]
        self.edges = [
            edge
            for edge in self.edges
            if edge["source"] not in removed and edge["target"] not in removed
        ]

    def remove_outgoing_edges(
        self,
        source_task_id: int,
        *,
        kind: Optional[PlanEdgeKind] = None,
    ) -> None:
        """Remove outgoing edges from one source, optionally limited by kind."""

        self.edges = [
            edge
            for edge in self.edges
            if edge["source"] != source_task_id
            or (kind is not None and edge["kind"] != kind)
        ]

    def remove_edge(
        self,
        *,
        source_task_id: int,
        target_task_id: Optional[int] = None,
        kind: Optional[PlanEdgeKind] = None,
        label: Optional[str] = None,
    ) -> None:
        """Remove explicit edges matching the provided fields."""

        self.edges = [
            edge
            for edge in self.edges
            if not (
                edge["source"] == source_task_id
                and (target_task_id is None or edge["target"] == target_task_id)
                and (kind is None or edge["kind"] == kind)
                and (label is None or str(edge.get("label") or "") == label)
            )
        ]

    def set_sequence_edge(
        self,
        source_task_id: int,
        target_task_id: Optional[int],
    ) -> None:
        """Replace one node's sequence edge."""

        self.remove_outgoing_edges(source_task_id, kind="sequence")
        if target_task_id is not None:
            self.edges.append(
                {
                    "source": source_task_id,
                    "target": int(target_task_id),
                    "kind": "sequence",
                }
            )

    def replace_edge_target(
        self,
        previous_target_task_id: int,
        new_target_task_id: Optional[int],
    ) -> None:
        """Retarget inbound edges, dropping them when the new target is None."""

        updated_edges: list[PlanGraphEdge] = []
        for edge in self.edges:
            if edge["target"] != previous_target_task_id:
                updated_edges.append(edge)
                continue
            if new_target_task_id is None:
                continue
            updated_edge = dict(edge)
            updated_edge["target"] = int(new_target_task_id)
            updated_edges.append(updated_edge)
        self.edges = updated_edges

    def reachable_task_ids(
        self,
        *,
        start_task_ids: Sequence[Optional[int]],
        plan_id: Optional[str] = None,
    ) -> set[int]:
        """Return nodes reachable from the given graph entry points."""

        reachable: set[int] = set()
        stack = [task_id for task_id in start_task_ids if task_id is not None]

        while stack:
            task_id = int(stack.pop())
            if task_id in reachable or task_id not in self.nodes:
                continue
            task = self.nodes[task_id]
            if plan_id is not None and task.get("plan_id") != plan_id:
                continue
            reachable.add(task_id)
            for edge in self.outgoing_edges(task_id):
                if edge["target"] not in reachable:
                    stack.append(edge["target"])

        return reachable

    def outgoing_edges(
        self,
        source_task_id: int,
        *,
        kind: Optional[PlanEdgeKind] = None,
    ) -> list[PlanGraphEdge]:
        """Return outgoing edges for one source."""

        return [
            edge
            for edge in self.edges
            if edge["source"] == source_task_id
            and (kind is None or edge["kind"] == kind)
        ]

    def add_edge(self, edge: PlanGraphEdge) -> None:
        """Append one explicit edge."""

        self.edges.append(dict(edge))

    def set_branch_edge(
        self,
        source_task_id: int,
        label: str,
        target_task_id: int,
    ) -> None:
        """Replace one labeled branch edge from a decision node."""

        self.remove_edge(
            source_task_id=source_task_id,
            kind="branch",
            label=label,
        )
        self.edges.append(
            {
                "source": source_task_id,
                "target": int(target_task_id),
                "kind": "branch",
                "label": label,
            }
        )

    def leaf_task_ids(
        self,
        *,
        task_ids: Optional[Iterable[int]] = None,
    ) -> list[int]:
        """Return task ids with no outgoing edge inside the selected node set."""

        selected = (
            set(self.nodes)
            if task_ids is None
            else {int(task_id) for task_id in task_ids}
        )
        leaves: list[int] = []
        for task_id in sorted(selected):
            has_outgoing = any(
                edge["source"] == task_id and edge["target"] in selected
                for edge in self.edges
            )
            if not has_outgoing:
                leaves.append(task_id)
        return leaves

    def to_tasks(self) -> dict[int, TaskItem]:
        """Export control-flow edges while preserving explicit dependencies."""

        tasks: dict[int, TaskItem] = {}
        for task_id, task in self.nodes.items():
            exported_task: TaskItem = {
                **task,
                "next_task_id": None,
                "branches": None,
                "depends_on": list(task.get("depends_on", [])),
            }
            tasks[task_id] = exported_task

        for edge in self.edges:
            source_id = int(edge["source"])
            target_id = int(edge["target"])
            if source_id not in tasks:
                continue
            if edge["kind"] == "sequence":
                tasks[source_id]["next_task_id"] = target_id
            elif edge["kind"] == "branch":
                branches = dict(tasks[source_id].get("branches") or {})
                branches[str(edge.get("label") or f"branch_{target_id}")] = target_id
                tasks[source_id]["branches"] = branches

        return dict(sorted(tasks.items()))


def _scoped_tasks(
    tasks: dict[int, TaskItem],
    *,
    plan_id: Optional[str],
) -> dict[int, TaskItem]:
    """Return tasks in the requested plan scope."""

    return {
        task_id: task
        for task_id, task in tasks.items()
        if plan_id is None or task.get("plan_id") == plan_id
    }


def _coerce_task_id(raw_value: Any) -> Optional[int]:
    """Normalize a loose task id payload into an int when possible."""

    if raw_value is None:
        return None
    try:
        return int(raw_value)
    except Exception:
        return None


def iter_plan_edges(
    tasks: dict[int, TaskItem],
    plan_id: Optional[str] = None,
) -> list[PlanGraphEdge]:
    """Return explicit sequence/branch edges derived from task fields."""

    scoped_tasks = _scoped_tasks(tasks, plan_id=plan_id)
    edges: list[PlanGraphEdge] = []
    for source_id, task in sorted(scoped_tasks.items()):
        next_task_id = _coerce_task_id(task.get("next_task_id"))
        if next_task_id is not None:
            edges.append(
                {
                    "source": source_id,
                    "target": next_task_id,
                    "kind": "sequence",
                }
            )

        for branch_label, branch_target in sorted((task.get("branches") or {}).items()):
            target_id = _coerce_task_id(branch_target)
            if target_id is None:
                continue
            edges.append(
                {
                    "source": source_id,
                    "target": target_id,
                    "kind": "branch",
                    "label": str(branch_label),
                }
            )

    return edges


def build_plan_graph(
    tasks: dict[int, TaskItem],
    plan_id: Optional[str] = None,
) -> nx.DiGraph:
    """Build a DiGraph from valid in-scope task edges without mutating tasks."""

    scoped_tasks = _scoped_tasks(tasks, plan_id=plan_id)
    graph = nx.DiGraph()
    for task_id, task in sorted(scoped_tasks.items()):
        graph.add_node(task_id, task=dict(task))

    for edge in iter_plan_edges(scoped_tasks):
        if edge["target"] not in scoped_tasks:
            continue
        graph.add_edge(
            edge["source"],
            edge["target"],
            kind=edge["kind"],
            label=edge.get("label"),
        )

    return graph


def _issue(
    *,
    severity: PlanIssueSeverity,
    code: str,
    task_id: int,
    message: str,
    details: Optional[dict[str, Any]] = None,
) -> PlanGraphIssue:
    """Create a normalized validation issue."""

    return {
        "severity": severity,
        "code": code,
        "task_id": task_id,
        "message": message,
        "details": details or {},
    }


def _actual_depends_on(task: TaskItem) -> list[int]:
    """Return sorted declared depends_on values, ignoring invalid payloads."""

    result: list[int] = []
    for raw_value in task.get("depends_on", []):
        task_id = _coerce_task_id(raw_value)
        if task_id is not None:
            result.append(task_id)
    return sorted(set(result))


def validate_plan_graph(
    tasks: dict[int, TaskItem],
    plan_id: Optional[str] = None,
) -> list[PlanGraphIssue]:
    """Validate task graph shape without changing runtime behavior."""

    scoped_tasks = _scoped_tasks(tasks, plan_id=plan_id)
    issues: list[PlanGraphIssue] = []

    for edge in iter_plan_edges(scoped_tasks):
        source_id = edge["source"]
        target_id = edge["target"]
        if source_id == target_id:
            issues.append(
                _issue(
                    severity="error",
                    code="self_loop",
                    task_id=source_id,
                    message=f"Task {source_id} points to itself.",
                    details={"edge": edge},
                )
            )
        if target_id not in scoped_tasks:
            issues.append(
                _issue(
                    severity="error",
                    code="dangling_edge",
                    task_id=source_id,
                    message=f"Task {source_id} points to missing task {target_id}.",
                    details={"edge": edge, "plan_id": plan_id},
                )
            )

    graph = build_plan_graph(scoped_tasks)
    if graph.number_of_nodes() == 0:
        return issues

    if not nx.is_directed_acyclic_graph(graph):
        cycles = list(nx.simple_cycles(graph))
        issues.append(
            _issue(
                severity="error",
                code="cycle",
                task_id=int(cycles[0][0]) if cycles and cycles[0] else -1,
                message="Plan graph contains a cycle.",
                details={"cycles": cycles},
            )
        )

    root_ids = sorted(node_id for node_id, degree in graph.in_degree() if degree == 0)
    if root_ids:
        primary_root_id = int(root_ids[0])
        reachable_ids = {
            primary_root_id,
            *[int(node_id) for node_id in nx.descendants(graph, primary_root_id)],
        }
        for task_id in sorted(set(graph.nodes) - reachable_ids):
            issues.append(
                _issue(
                    severity="warning",
                    code="unreachable_task",
                    task_id=int(task_id),
                    message=(
                        f"Task {task_id} is not reachable from primary root "
                        f"{primary_root_id}."
                    ),
                    details={"primary_root_id": primary_root_id, "root_ids": root_ids},
                )
            )

    for task_id, task in sorted(scoped_tasks.items()):
        for dependency_id in _actual_depends_on(task):
            if dependency_id not in scoped_tasks:
                issues.append(
                    _issue(
                        severity="error",
                        code="dangling_dependency",
                        task_id=task_id,
                        message=(
                            f"Task {task_id} depends on missing task {dependency_id}."
                        ),
                        details={"dependency_id": dependency_id, "plan_id": plan_id},
                    )
                )

    return issues


def summarize_plan_graph(
    tasks: dict[int, TaskItem],
    plan_id: Optional[str] = None,
) -> dict[str, Any]:
    """Return a compact, visualization-friendly graph summary."""

    graph = build_plan_graph(tasks, plan_id=plan_id)
    issues = validate_plan_graph(tasks, plan_id=plan_id)
    root_ids = sorted(int(node_id) for node_id, degree in graph.in_degree() if degree == 0)
    leaf_ids = sorted(
        int(node_id)
        for node_id, degree in graph.out_degree()
        if degree == 0
    )
    return {
        "plan_id": plan_id,
        "node_count": graph.number_of_nodes(),
        "edge_count": graph.number_of_edges(),
        "root_task_ids": root_ids,
        "leaf_task_ids": leaf_ids,
        "is_dag": nx.is_directed_acyclic_graph(graph),
        "issues": issues,
    }


__all__ = [
    "PlanGraphEdge",
    "PlanGraphIssue",
    "build_plan_graph",
    "iter_plan_edges",
    "summarize_plan_graph",
    "validate_plan_graph",
]
