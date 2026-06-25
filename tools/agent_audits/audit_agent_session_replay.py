"""Offline audit for one async-agent run-memory session.

This script does not execute robot actions. It reads a saved run_memory.json
snapshot and produces a compact report about plan structure, tool usage,
background preanalysis, object approach artifacts, and memory size risks.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


NAV_TARGET_TYPES = {"keyframe", "position", "task_output"}
KEYFRAME_SEARCH_TOOLS = {
    "requirement_search",
    "search_requirement_on_keyframe_nodes",
    "keyword_search",
    "search_keywords_on_keyframe_nodes",
    "match_attached_image_to_keyframes",
}
KEYFRAME_IMAGE_ANALYSIS_TOOLS = {
    "analyse_on_each_kf_images",
    "co_analyse_on_kf_images",
}
OBJECT_TOOLS = {
    "approach_object_in_current_view",
    "preanalyze_object_on_keyframe",
    "resolve_object_from_attached_image",
}
CAPTURE_TOOLS = {"capture_current_view"}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _json_len(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=False, default=str))


def _short(value: Any, limit: int = 160) -> str:
    text = str(value or "").replace("\n", " ").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _safe_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _iter_tasks(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for plan in _safe_list(snapshot.get("plans")):
        if not isinstance(plan, dict):
            continue
        for task in _safe_list(plan.get("tasks")):
            if isinstance(task, dict):
                item = dict(task)
                item.setdefault("plan_id", plan.get("plan_id"))
                tasks.append(item)
    return tasks


def _iter_tool_events(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for trace_item in _safe_list(snapshot.get("tool_traces")):
        if not isinstance(trace_item, dict):
            continue
        trace = _safe_dict(trace_item.get("tool_trace"))
        raw_results = (
            trace.get("tool_results")
            or trace.get("tools")
            or trace.get("tool_calls")
            or []
        )
        for idx, result in enumerate(_safe_list(raw_results)):
            if not isinstance(result, dict):
                continue
            name = (
                result.get("tool_name")
                or result.get("name")
                or result.get("tool")
                or result.get("called_tool")
            )
            content = result.get("content")
            if content is None:
                content = result.get("result")
            parsed = _parse_content(content)
            if not name and isinstance(parsed, dict):
                name = parsed.get("tool_name") or parsed.get("name")
            events.append(
                {
                    "task_id": trace_item.get("task_id"),
                    "plan_id": trace_item.get("plan_id"),
                    "description": trace_item.get("description"),
                    "index": idx,
                    "tool_name": str(name or "unknown"),
                    "raw": result,
                    "parsed": parsed,
                }
            )
    for task_result in _safe_list(snapshot.get("task_results")):
        if not isinstance(task_result, dict):
            continue
        for idx, evidence in enumerate(_safe_list(task_result.get("tool_evidence"))):
            if not isinstance(evidence, dict):
                continue
            name = evidence.get("tool_name") or evidence.get("name")
            events.append(
                {
                    "task_id": task_result.get("task_id"),
                    "plan_id": task_result.get("plan_id"),
                    "description": task_result.get("description"),
                    "index": idx,
                    "tool_name": str(name or "unknown"),
                    "raw": evidence,
                    "parsed": evidence,
                    "source": "tool_evidence",
                }
            )
    return events


def _parse_content(content: Any) -> Any:
    if isinstance(content, (dict, list)):
        return content
    if not isinstance(content, str):
        return content
    text = content.strip()
    if not text:
        return text
    try:
        return json.loads(text)
    except Exception:
        return text


def _extract_tool_payload(event: dict[str, Any]) -> dict[str, Any]:
    parsed = event.get("parsed")
    if isinstance(parsed, dict):
        data = parsed.get("data")
        if isinstance(data, dict):
            return data
        return parsed
    raw = event.get("raw")
    if isinstance(raw, dict):
        data = raw.get("data")
        if isinstance(data, dict):
            return data
        return raw
    return {}


def _valid_navigation_target(target: Any) -> tuple[bool, str]:
    if not isinstance(target, dict):
        return False, "target_not_object"
    target_type = str(target.get("type") or "").strip()
    if target_type not in NAV_TARGET_TYPES:
        return False, f"unsupported_target_type:{target_type or '<missing>'}"
    if target_type == "keyframe" and target.get("keyframe_id") is None:
        return False, "missing_keyframe_id"
    if target_type == "position":
        position = target.get("position")
        if not isinstance(position, list) or len(position) < 2:
            return False, "missing_position"
    if target_type == "task_output" and target.get("task_id") is None:
        return False, "missing_task_id"
    return True, "ok"


def _plan_findings(snapshot: dict[str, Any]) -> list[str]:
    findings: list[str] = []
    tasks = _iter_tasks(snapshot)
    if not tasks:
        findings.append("WARN no_plan_tasks: run_memory contains no recorded planner tasks.")
        return findings
    by_plan: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for task in tasks:
        by_plan[str(task.get("plan_id") or "<none>")].append(task)
    for plan_id, plan_tasks in by_plan.items():
        nav_count = 0
        resolver_count = 0
        for task in plan_tasks:
            task_type = str(task.get("task_type") or "").strip()
            desc = str(task.get("description") or "")
            if task_type == "navigation_action":
                nav_count += 1
                ok, reason = _valid_navigation_target(task.get("target"))
                if not ok:
                    findings.append(
                        f"ERROR invalid_navigation_target plan={plan_id} task={task.get('task_id')} reason={reason} desc={_short(desc)}"
                    )
                deps = _safe_list(task.get("depends_on"))
                target = _safe_dict(task.get("target"))
                if target.get("type") == "task_output" and target.get("task_id") not in deps:
                    findings.append(
                        f"WARN navigation_missing_dependency plan={plan_id} task={task.get('task_id')} target_task={target.get('task_id')}"
                    )
            elif task_type == "llm_action":
                desc_lower = desc.lower()
                if (
                    task.get("resolver_kind")
                    or "destination" in desc_lower
                    or "resolve " in desc_lower
                    or "localization" in desc_lower
                    or "localisation" in desc_lower
                ):
                    resolver_count += 1
            if "after arrival" in desc.lower() or "after arriving" in desc.lower():
                findings.append(
                    f"INFO arrival_relative_description plan={plan_id} task={task.get('task_id')} desc={_short(desc)}"
                )
        if nav_count and resolver_count == 0:
            findings.append(
                f"WARN nav_without_recorded_resolver plan={plan_id}: navigation tasks exist but no resolver-like llm_action was recorded."
            )
    return findings


def _tool_findings(events: list[dict[str, Any]]) -> list[str]:
    findings: list[str] = []
    counts = Counter(event.get("tool_name") for event in events)
    if not counts:
        findings.append("WARN no_tool_events: no tool trace/evidence found.")
        return findings

    search_count = sum(counts.get(name, 0) for name in KEYFRAME_SEARCH_TOOLS)
    image_analysis_count = sum(counts.get(name, 0) for name in KEYFRAME_IMAGE_ANALYSIS_TOOLS)
    if search_count > 4:
        findings.append(f"WARN many_keyframe_search_calls: {search_count} calls; inspect whether the task kept chasing perfect matches.")
    if image_analysis_count > 3:
        findings.append(f"WARN many_keyframe_image_analysis_calls: {image_analysis_count} calls; prefer compact keyframe semantics unless candidates are ambiguous.")

    by_task: dict[Any, Counter[str]] = defaultdict(Counter)
    for event in events:
        by_task[event.get("task_id")][event.get("tool_name")] += 1
    for task_id, task_counts in by_task.items():
        repeated = {name: count for name, count in task_counts.items() if count >= 3}
        if repeated:
            findings.append(f"INFO repeated_tools task={task_id}: {dict(repeated)}")
    return findings


def _background_findings(snapshot: dict[str, Any]) -> list[str]:
    findings: list[str] = []
    updates = _safe_list(snapshot.get("background_updates"))
    if not updates:
        findings.append("INFO no_background_updates: no background preanalysis records were saved.")
        return findings
    statuses = Counter()
    for item in updates:
        if not isinstance(item, dict):
            continue
        record = _safe_dict(item.get("record"))
        status = str(record.get("status") or item.get("status") or "unknown")
        statuses[status] += 1
        if record.get("failure_reason"):
            findings.append(
                f"INFO background_failure task={item.get('task_id')} reason={_short(record.get('failure_reason'))}"
            )
        if record.get("recommended_destination"):
            findings.append(
                f"OK background_destination task={item.get('task_id')} type={record.get('destination_type') or 'unknown'}"
            )
    findings.insert(0, f"INFO background_status_counts: {dict(statuses)}")
    return findings


def _object_findings(events: list[dict[str, Any]]) -> list[str]:
    findings: list[str] = []
    object_events = [
        event for event in events
        if str(event.get("tool_name") or "") in OBJECT_TOOLS
    ]
    if not object_events:
        findings.append("INFO no_object_tool_events: no semantic object localization tools were recorded.")
        return findings
    for event in object_events:
        payload = _extract_tool_payload(event)
        metrics = _safe_dict(payload.get("key_metrics"))
        artifact_paths = _safe_dict(payload.get("artifact_paths"))
        backend = payload.get("depth_backend") or metrics.get("depth_backend")
        approach_status = payload.get("approach_status") or payload.get("status")
        parts = [
            f"task={event.get('task_id')}",
            f"tool={event.get('tool_name')}",
            f"status={approach_status}",
        ]
        if backend:
            parts.append(f"backend={backend}")
        if metrics.get("mono_guard_selected_source"):
            parts.append(f"mono_guard={metrics.get('mono_guard_selected_source')}")
            parts.append(f"reason={metrics.get('mono_guard_reason')}")
        if artifact_paths.get("summary_json"):
            parts.append(f"summary_json={artifact_paths.get('summary_json')}")
        findings.append("INFO object_tool " + " ".join(parts))
    return findings


def _capture_findings(events: list[dict[str, Any]]) -> list[str]:
    findings: list[str] = []
    captures = [
        event for event in events
        if str(event.get("tool_name") or "") in CAPTURE_TOOLS
    ]
    if not captures:
        findings.append("INFO no_capture_events: no current-view capture tool was recorded.")
        return findings
    for event in captures:
        payload = _extract_tool_payload(event)
        findings.append(
            "OK capture "
            f"task={event.get('task_id')} path={payload.get('path')} note={_short(payload.get('note'), 80)}"
        )
    return findings


def _memory_findings(snapshot: dict[str, Any]) -> list[str]:
    findings: list[str] = []
    sizes = {
        key: _json_len(snapshot.get(key))
        for key in (
            "plans",
            "task_results",
            "tool_traces",
            "background_updates",
            "observations",
            "responses",
        )
    }
    findings.append(f"INFO memory_bucket_chars: {sizes}")
    for item in _safe_list(snapshot.get("task_results")):
        if not isinstance(item, dict):
            continue
        size = _json_len(item)
        if size > 24_000:
            findings.append(
                f"WARN large_task_result task={item.get('task_id')} chars={size}; check compact tool evidence."
            )
    return findings


def _summary(snapshot: dict[str, Any], events: list[dict[str, Any]]) -> list[str]:
    session = _safe_dict(snapshot.get("session"))
    tool_counts = Counter(event.get("tool_name") for event in events)
    return [
        f"session_id: {session.get('session_id')}",
        f"session_dir: {session.get('session_dir')}",
        f"plans: {len(_safe_list(snapshot.get('plans')))}",
        f"task_results: {len(_safe_list(snapshot.get('task_results')))}",
        f"tool_events: {len(events)}",
        f"background_updates: {len(_safe_list(snapshot.get('background_updates')))}",
        f"observations: {len(_safe_list(snapshot.get('observations')))}",
        f"top_tools: {dict(tool_counts.most_common(8))}",
    ]


def build_report(snapshot: dict[str, Any]) -> dict[str, Any]:
    events = _iter_tool_events(snapshot)
    return {
        "summary": _summary(snapshot, events),
        "plan_quality": _plan_findings(snapshot),
        "tool_usage": _tool_findings(events),
        "background": _background_findings(snapshot),
        "object_approach": _object_findings(events),
        "capture": _capture_findings(events),
        "memory": _memory_findings(snapshot),
    }


def _format_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = ["# Agent Session Replay Audit", ""]
    for title, items in report.items():
        lines.append(f"## {title.replace('_', ' ').title()}")
        if isinstance(items, list):
            for item in items:
                lines.append(f"- {item}")
        else:
            lines.append("```json")
            lines.append(json.dumps(items, ensure_ascii=False, indent=2))
            lines.append("```")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _find_latest_run_memory(search_root: Path) -> Path:
    candidates = sorted(
        search_root.glob("**/run_memory.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No run_memory.json found under {search_root}")
    return candidates[0]


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit one CarAgent async-agent run_memory session.")
    parser.add_argument("--run-memory", type=Path, help="Path to run_memory.json.")
    parser.add_argument("--search-root", type=Path, default=Path("logs"), help="Search root used when --run-memory is omitted.")
    parser.add_argument("--output", type=Path, help="Optional report path (.md or .json).")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of markdown.")
    args = parser.parse_args()

    run_memory = args.run_memory or _find_latest_run_memory(args.search_root)
    snapshot = _load_json(run_memory)
    report = build_report(snapshot)
    report["source"] = {"run_memory": str(run_memory)}

    if args.json or (args.output and args.output.suffix.lower() == ".json"):
        text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    else:
        text = _format_markdown(report)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
