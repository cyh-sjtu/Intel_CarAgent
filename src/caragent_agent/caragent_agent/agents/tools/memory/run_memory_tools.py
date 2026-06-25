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
                summary_table or timeline first, then call detail only for the
                selected row_id/task_id/plan_id you need. Use scope=all only
                when the information type is unclear.

                Views:
                - summary_table is an index table. It has stable compact fields
                  such as intent_label, destination_summary, candidate_summary,
                  artifact_summary, status, task_type, result_kind,
                  semantic_grounding, final_target, destination_source,
                  evidence_summary, and background_status.
                  Use it to choose the relevant row; do not expect full
                  reasoning evidence here.
                - timeline is an ordered event narrative generated from
                  structured fields. Use it to understand task/navigation order,
                  dependencies, semantic keyframe/object grounding flow, and
                  arrivals.
                - detail is compact selected-row evidence for reasoning.
                  It preserves new-agent task metadata such as target,
                  inputs_from, outputs, image_refs, compact tool_evidence,
                  destinations, candidates, and artifact paths. It is not a raw
                  trace dump.

                Important lookup rule: after reading summary_table or timeline,
                use row_id for detail lookup whenever possible. task_id is only
                unique inside one plan and can repeat across plans in the same
                session, so task_id-only detail queries may select the wrong
                row or no row. Use task_id together with plan_id only when you
                intentionally mean that exact plan-local task. row_id is a
                session-scoped exact pointer; if you have row_id, use it by
                itself for detail.

                Scope rule: row_id values belong to their returned scope. A
                plan row_id such as plan_1 is not a task row_id; do not use it
                for scope=task detail. If a detail query returns no items, do
                not repeat the identical query. Go back to the relevant
                summary_table/timeline, choose a valid row_id from that scope,
                or continue with scene-memory/tool evidence.

                Plan filters are exact filters inside the current run session.
                Do not pass the current task/plan_id by habit. For current
                session memory questions such as "previous target", "what
                happened earlier", or any request that is not explicitly
                limited to the current plan, leave plan_id empty and search the
                relevant scope first with time=all. This tool does not retrieve
                other saved sessions unless such a snapshot has been explicitly
                loaded as the current run memory by the host program.

                Detail is intentionally not a raw tool-trace dump. Tool results
                are compressed into tool_evidence fields such as destination,
                candidate keyframe ids, artifact paths, failure reason, and key
                metrics. If you need full debugging payloads, use the returned
                artifact paths or offline run-memory/tool-trace files instead
                of expecting query_memory to return large JSON, maps, images, or
                costmaps.

                Navigation rows may include related_task_row_id. Use that exact
                task row_id with scope=task/detail when you need the tool
                evidence behind one visited navigation anchor. Do not pass a
                navigation row_id to scope=task; row_id prefixes identify their
                owning scope.

                Treat result_kind and destination_source as evidence labels, not
                hard guarantees. In particular, semantic_object_fallback_destination
                or fallback_after_semantic_object_issue means the task produced a
                usable destination after semantic object tooling was blocked or
                failed; it is not the same as a verified object-tool destination.
                Check evidence_summary.status_counts and background_status
                before saying background preanalysis or object localization
                succeeded.

                Args:
                    scope (str): conversation, plan, task, navigation,
                        observation, or all.
                    view (str): summary_table, timeline, or detail.
                    query (str): Optional natural-language note describing what
                        you are looking for. It is echoed back for your
                        reasoning, not used as a hard text filter.
                    time (str): all or recent. Use all for normal current-session
                        memory questions; recent only for explicitly recent
                        context.
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
        time: str = "all",
        plan_id: str = "",
        turn_id: str = "",
        task_id: int = -1,
        row_id: str = "",
        limit: int = 50,
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
