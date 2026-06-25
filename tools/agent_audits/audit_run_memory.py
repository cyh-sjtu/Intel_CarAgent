"""Offline audit helper for async-agent run-memory snapshots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from import_stubs import install_offline_import_stubs

install_offline_import_stubs()

from caragent_agent.agents.async_agent.memory.run_memory import AsyncAgentRunMemory


def _json_size(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=False, default=str))


def _contains_any(value: Any, needles: tuple[str, ...]) -> list[str]:
    text = json.dumps(value, ensure_ascii=False, default=str)
    return [needle for needle in needles if needle in text]


def load_snapshot(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_memory_from_snapshot(snapshot: dict[str, Any]) -> AsyncAgentRunMemory:
    session = snapshot.get("session") if isinstance(snapshot.get("session"), dict) else {}
    memory = AsyncAgentRunMemory(
        session_id=str(session.get("session_id") or "offline_audit"),
        session_dir=None,
        metadata={"source": "offline_audit"},
    )
    memory._data = snapshot
    return memory


def rebuild_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Rebuild one old snapshot through the current compact memory writer."""

    source_memory = build_memory_from_snapshot(snapshot)
    session = snapshot.get("session") if isinstance(snapshot.get("session"), dict) else {}
    rebuilt = AsyncAgentRunMemory(
        session_id=str(session.get("session_id") or "rebuilt_run_memory"),
        session_dir=None,
        metadata={
            **(session.get("metadata") if isinstance(session.get("metadata"), dict) else {}),
            "rebuilt_from": "offline_snapshot",
        },
    )

    with rebuilt._lock:
        rebuilt._data["session"] = {
            **dict(session),
            "rebuilt_at": rebuilt._data["session"].get("created_at"),
            "metadata": {
                **(session.get("metadata") if isinstance(session.get("metadata"), dict) else {}),
                "rebuilt_from": "offline_snapshot",
            },
        }
        for bucket in (
            "threads",
            "turns",
            "events",
            "plans",
            "background_updates",
            "observations",
            "stream_updates",
            "responses",
            "checkpoints",
            "tool_traces",
        ):
            value = snapshot.get(bucket, {} if bucket == "threads" else [])
            rebuilt._data[bucket] = json.loads(json.dumps(value, ensure_ascii=False, default=str))

        rebuilt_tasks: list[dict[str, Any]] = []
        for item in list(snapshot.get("task_results") or []):
            if not isinstance(item, dict):
                continue
            enriched = source_memory._attach_tool_trace_excerpt_to_task_item_locked(item)
            tool_trace = enriched.get("tool_trace_excerpt")
            result = enriched.get("result")
            compact_result = source_memory._data  # keep attribute access lint quiet
            del compact_result
            from caragent_agent.agents.async_agent.memory.run_memory import (
                _compact_task_result_value,
                _compact_tool_evidence,
            )

            rebuilt_item = {
                "thread_id": enriched.get("thread_id"),
                "task_id": enriched.get("task_id"),
                "description": enriched.get("description"),
                "status": enriched.get("status"),
                "task_type": enriched.get("task_type"),
                "target": enriched.get("target"),
                "type": enriched.get("type"),
                "plan_id": enriched.get("plan_id"),
                "user_input_id": enriched.get("user_input_id"),
                "depends_on": enriched.get("depends_on", []),
                "result": _compact_task_result_value(result),
                "origin": enriched.get("origin"),
                "error": enriched.get("error"),
                "event_type": enriched.get("event_type"),
                "summary": enriched.get("summary"),
                "recorded_at": enriched.get("recorded_at"),
            }
            evidence = _compact_tool_evidence(tool_trace)
            if evidence:
                rebuilt_item["tool_evidence"] = evidence
                rebuilt_item["tool_trace_excerpt"] = {
                    "tools": evidence,
                    "final_ai_content": (
                        tool_trace.get("final_ai_content")
                        if isinstance(tool_trace, dict)
                        else None
                    ),
                }
            rebuilt_tasks.append(
                {
                    key: value
                    for key, value in rebuilt_item.items()
                    if value not in (None, "", [], {})
                }
            )
        rebuilt._data["task_results"] = rebuilt_tasks

    return rebuilt.snapshot()


def rebuild_snapshot_file(snapshot_path: Path, output_path: Path) -> dict[str, Any]:
    """Rebuild a snapshot file and write the compact run-memory JSON."""

    rebuilt = rebuild_snapshot(load_snapshot(snapshot_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(rebuilt, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return rebuilt


def audit_snapshot(snapshot_path: Path, output_path: Path | None = None) -> dict[str, Any]:
    snapshot = load_snapshot(snapshot_path)
    memory = build_memory_from_snapshot(snapshot)
    scopes = ("conversation", "plan", "task", "navigation", "observation")
    views = ("summary_table", "timeline", "detail")
    queries: dict[str, Any] = {}
    warnings: list[str] = []
    leaks: dict[str, list[str]] = {}
    for scope in scopes:
        for view in views:
            result = memory.query_memory(scope=scope, view=view, time="all", limit=50)
            key = f"{scope}.{view}"
            queries[key] = {
                "item_count": len(result.get("items") or []),
                "estimated_chars": result.get("estimated_chars"),
                "budget_chars": result.get("budget_chars"),
                "truncated": result.get("truncated"),
                "warnings": result.get("warnings"),
                "items": result.get("items"),
            }
            if result.get("warnings"):
                warnings.extend([f"{key}: {item}" for item in result["warnings"]])
            found = _contains_any(
                result.get("items"),
                (
                    "occupancy_grid",
                    "depth_map",
                    "image_base64",
                    "raw_output",
                    "tool_results",
                    "[100, 100, 100",
                ),
            )
            if found:
                leaks[key] = found

    task_detail = queries["task.detail"]["items"]
    task_destinations = [
        {
            "row_id": item.get("row_id"),
            "task_id": item.get("task_id"),
            "description": item.get("description"),
            "destination": item.get("destination"),
            "candidate_keyframe_ids": item.get("candidate_keyframe_ids"),
            "artifact_paths": item.get("artifact_paths"),
        }
        for item in task_detail
        if isinstance(item, dict)
        and (
            item.get("destination")
            or item.get("candidate_keyframe_ids")
            or item.get("artifact_paths")
        )
    ]
    report = {
        "snapshot_path": str(snapshot_path),
        "snapshot_size_chars": _json_size(snapshot),
        "bucket_counts": {
            key: len(value) if isinstance(value, list) else len(value) if isinstance(value, dict) else 0
            for key, value in snapshot.items()
        },
        "warnings": warnings,
        "leaks": leaks,
        "task_destinations": task_destinations,
        "queries": queries,
    }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return report


def probe_memory_questions(snapshot_path: Path) -> dict[str, Any]:
    """Run deterministic probes that mimic the intended query_memory workflow."""

    memory = build_memory_from_snapshot(load_snapshot(snapshot_path))
    task_table = memory.query_memory(scope="task", view="summary_table", time="all", limit=50)
    navigation_table = memory.query_memory(scope="navigation", view="summary_table", time="all", limit=50)
    task_items = list(task_table.get("items") or [])
    probes: list[dict[str, Any]] = []

    def add_probe(name: str, row: dict[str, Any] | None, *, expect_destination: bool = False) -> None:
        if row is None:
            probes.append({"name": name, "status": "failed", "reason": "row_not_found"})
            return
        detail = memory.query_memory(
            scope="task",
            view="detail",
            row_id=str(row.get("row_id") or ""),
            time="all",
            limit=50,
        )
        item = (detail.get("items") or [{}])[0]
        destination = item.get("destination") if isinstance(item, dict) else None
        artifacts = item.get("artifact_paths") if isinstance(item, dict) else None
        candidates = item.get("candidate_keyframe_ids") if isinstance(item, dict) else None
        text = json.dumps(detail, ensure_ascii=False, default=str)
        leaks = _contains_any(
            detail.get("items"),
            (
                "occupancy_grid",
                "depth_map",
                "image_base64",
                "raw_output",
                "tool_results",
            ),
        )
        ok = not leaks and (not expect_destination or bool(destination))
        probes.append(
            {
                "name": name,
                "status": "ok" if ok else "failed",
                "row_id": row.get("row_id"),
                "task_id": row.get("task_id"),
                "detail_chars": len(text),
                "detail_truncated": detail.get("truncated"),
                "destination": destination,
                "candidate_keyframe_ids": candidates,
                "artifact_keys": sorted(list(artifacts.keys())) if isinstance(artifacts, dict) else [],
                "leaks": leaks,
            }
        )

    object_rows = []
    for row in task_items:
        evidence = row.get("evidence_summary") if isinstance(row.get("evidence_summary"), dict) else {}
        flags = set(evidence.get("flags") or [])
        artifact_count = (
            row.get("artifact_summary", {}).get("count")
            if isinstance(row.get("artifact_summary"), dict)
            else 0
        )
        if "object_approach" in flags or artifact_count:
            object_rows.append(row)
    keyframe_rows = [
        row
        for row in task_items
        if "keyframe=" in str(row.get("destination_summary") or "")
    ]
    candidate_rows = [
        row
        for row in task_items
        if isinstance(row.get("candidate_summary"), dict)
        and row["candidate_summary"].get("count")
    ]
    add_probe("latest_object_destination_detail", object_rows[-1] if object_rows else None, expect_destination=True)
    add_probe("first_keyframe_navigation_detail", keyframe_rows[0] if keyframe_rows else None, expect_destination=True)
    add_probe("latest_candidate_search_detail", candidate_rows[-1] if candidate_rows else None)

    return {
        "snapshot_path": str(snapshot_path),
        "task_summary": {
            "item_count": len(task_items),
            "estimated_chars": task_table.get("estimated_chars"),
            "budget_chars": task_table.get("budget_chars"),
            "truncated": task_table.get("truncated"),
            "warnings": task_table.get("warnings"),
        },
        "navigation_summary": {
            "item_count": len(navigation_table.get("items") or []),
            "estimated_chars": navigation_table.get("estimated_chars"),
            "budget_chars": navigation_table.get("budget_chars"),
            "truncated": navigation_table.get("truncated"),
            "warnings": navigation_table.get("warnings"),
        },
        "probes": probes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit_snapshot(args.snapshot, args.output)
    print(
        json.dumps(
            {
                "snapshot_path": report["snapshot_path"],
                "snapshot_size_chars": report["snapshot_size_chars"],
                "bucket_counts": report["bucket_counts"],
                "warnings": report["warnings"],
                "leaks": report["leaks"],
                "task_destinations": report["task_destinations"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
