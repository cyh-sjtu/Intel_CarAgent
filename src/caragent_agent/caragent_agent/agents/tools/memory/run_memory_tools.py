"""Unified read-only tool for querying async-agent runtime memory."""

from __future__ import annotations

from typing import Any

from caragent_agent.agents.tools.base.tool_base import ToolBase


class QueryMemoryTool(ToolBase):
    """Query the simplified async-agent memory tables."""

    def __init__(self) -> None:
        super().__init__(
            name="query_memory",
            description="""
                Read simplified current-session memory tables.

                Use this as the main memory tool. It exposes five scopes:
                conversation, plan, task, navigation, and observation. Prefer
                summary_table or timeline first so you can inspect the table
                yourself, then call detail with row_id/task_id/plan_id when
                needed. Use scope=all only when the information type is unclear. Views:
                summary_table for candidate scanning, timeline for ordered
                history, and detail for the full selected row.

                Args:
                    scope (str): conversation, plan, task, navigation,
                        observation, or all.
                    view (str): summary_table, timeline, or detail.
                    query (str): Optional natural-language note describing what
                        you are looking for. It is echoed back for your
                        reasoning, not used as a hard text filter.
                    time (str): all or recent.
                    plan_id (str): Optional plan filter.
                    turn_id (str): Optional turn/thread filter.
                    task_id (int): Optional task filter; use -1 to ignore.
                    row_id (str): Optional exact row id from a previous query.
                    limit (int): Maximum number of items to return.
            """,
            capability_tags=("memory", "runtime_memory", "background_safe"),
        )

    def execute(
        self,
        scope: str = "all",
        view: str = "summary_table",
        query: str = "",
        time: str = "recent",
        plan_id: str = "",
        turn_id: str = "",
        task_id: int = -1,
        row_id: str = "",
        limit: int = 10,
    ) -> dict[str, Any]:
        """Return simplified scoped memory matches as a normalized tool result."""

        if self.run_memory is None:
            return self.blocked(
                "Run memory is unavailable, so session memory cannot be queried.",
                error={
                    "code": "run_memory_unavailable",
                    "message": "No run-memory object is attached to query_memory.",
                },
                provenance={"source_type": "run_memory"},
            )

        try:
            result = self.run_memory.query_memory(
                scope=scope,
                view=view,
                query=query,
                time=time,
                plan_id=plan_id,
                turn_id=turn_id,
                task_id=task_id,
                row_id=row_id,
                limit=limit,
            )
        except Exception as exc:
            return self.error_result(
                "Memory query failed.",
                error={
                    "code": "query_memory_exception",
                    "message": str(exc),
                },
                provenance={"source_type": "run_memory"},
            )

        data = {
            key: value
            for key, value in result.items()
            if key not in {"status", "summary", "error"}
        }
        warnings = list(data.get("warnings") or [])
        summary = str(
            result.get("summary")
            or f"Memory query returned {len(list(data.get('items') or []))} item(s)."
        )
        provenance = {
            "source_type": "run_memory",
            "scope": data.get("scope"),
            "view": data.get("view"),
        }

        if warnings:
            return self.partial(
                summary,
                data=data,
                error={
                    "code": "query_memory_normalized_args",
                    "warnings": warnings,
                },
                provenance=provenance,
            )

        return self.ok(summary, data=data, provenance=provenance)


__all__ = ["QueryMemoryTool"]
