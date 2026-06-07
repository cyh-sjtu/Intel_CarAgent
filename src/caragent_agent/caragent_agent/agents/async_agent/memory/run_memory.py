"""Persistent run-memory storage and query helpers for the async agent."""

from __future__ import annotations

import json
import os
import re
import threading
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from .exports import write_memory_tables


OBSERVATION_TOOL_NAMES = {
    "analyse_on_current_image",
    "get_current_state",
}


def _utc_now_iso() -> str:
    """Return one ISO timestamp for runtime-memory records."""

    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _truncate_text(value: Any, limit: int = 400) -> str:
    """Return one compact text preview for indexing and summaries."""

    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[: max(limit - 3, 0)].rstrip() + "..."


def _tokenize_query(value: Any) -> set[str]:
    """Return simple lowercase tokens for deterministic runtime-memory matching."""

    return {
        token.lower()
        for token in re.findall(r"[A-Za-z0-9_]+", str(value or ""))
        if token
    }


def _safe_int(value: Any) -> Optional[int]:
    """Coerce loose integer payloads without raising."""

    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _safe_float_triplet(value: Any) -> Optional[list[float]]:
    """Coerce one xyz-like sequence into a float triplet."""

    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return None
    try:
        return [float(value[0]), float(value[1]), float(value[2])]
    except Exception:
        return None


def _extract_tool_result_json(item: dict[str, Any], tool_name: str) -> Optional[dict[str, Any]]:
    """Return structured JSON content for one tool result embedded in task memory."""

    trace = item.get("tool_trace_excerpt")
    if not isinstance(trace, dict):
        return None
    for result in list(trace.get("tool_results") or []):
        if str(result.get("name") or "").strip() != tool_name:
            continue
        content = result.get("content")
        if isinstance(content, dict):
            return content
        try:
            parsed = json.loads(str(content or ""))
        except Exception:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _extract_structured_tool_data(raw_value: Any) -> Any:
    """Return the data field from one structured tool result payload."""

    parsed = raw_value
    if isinstance(raw_value, str):
        try:
            parsed = json.loads(raw_value)
        except Exception:
            return None
    if not isinstance(parsed, dict):
        return None
    if "data" in parsed and "status" in parsed:
        return parsed.get("data")
    return parsed


def _latest_tool_payload(tool_trace: dict[str, Any], tool_name: str) -> Any:
    """Return the latest payload for one tool name in a stored trace."""

    for tool_result in reversed(list(tool_trace.get("tool_results") or [])):
        if str(tool_result.get("name") or "").strip() != tool_name:
            continue
        return _extract_structured_tool_data(tool_result.get("content"))
    return None


def _observation_summary_from_trace(
    *,
    summary: str,
    tool_trace: dict[str, Any],
) -> Optional[str]:
    """Return an observation summary when execution actually used observation tools."""

    tool_names = {
        str(call.get("name") or "").strip()
        for call in list(tool_trace.get("tool_calls") or [])
        if isinstance(call, dict)
    }
    tool_names.update(
        str(result.get("name") or "").strip()
        for result in list(tool_trace.get("tool_results") or [])
        if isinstance(result, dict)
    )
    if not tool_names.intersection(OBSERVATION_TOOL_NAMES):
        return None

    image_payload = _latest_tool_payload(tool_trace, "analyse_on_current_image")
    if isinstance(image_payload, dict):
        answer = str(image_payload.get("answer") or "").strip()
        if answer:
            return _truncate_text(answer, 1000)

    final_ai_content = str(tool_trace.get("final_ai_content") or "").strip()
    if final_ai_content:
        return _truncate_text(final_ai_content, 1000)

    return _truncate_text(summary, 1000) if summary else None


def _observation_location_from_trace(
    tool_trace: dict[str, Any],
) -> tuple[Optional[list[float]], dict[str, Any]]:
    """Extract lightweight location details from observation tool payloads."""

    details: dict[str, Any] = {
        "observation_tools": sorted(
            {
                str(result.get("name") or "").strip()
                for result in list(tool_trace.get("tool_results") or [])
                if isinstance(result, dict)
                and str(result.get("name") or "").strip() in OBSERVATION_TOOL_NAMES
            }
        )
    }
    current_state = _latest_tool_payload(tool_trace, "get_current_state")
    position = None
    if isinstance(current_state, dict):
        position = _safe_float_triplet(current_state.get("position"))
        if current_state.get("status") is not None:
            details["current_status"] = current_state.get("status")
    return position, details


def _to_jsonable(value: Any) -> Any:
    """Convert one arbitrary runtime value into a JSON-safe representation."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        try:
            return _to_jsonable(value.tolist())
        except Exception:
            pass
    if hasattr(value, "content") and hasattr(value, "__class__"):
        return {
            "message_type": value.__class__.__name__,
            "content": _truncate_text(getattr(value, "content", "")),
            "name": getattr(value, "name", None),
        }
    return str(value)


def _coerce_positive_int(
    value: Any,
    *,
    default: int,
    minimum: int = 1,
    maximum: int = 50,
) -> tuple[int, Optional[str]]:
    """Coerce a loose integer value and return an optional warning."""

    try:
        resolved = int(value)
    except Exception:
        return default, f"Invalid integer value {value!r}; using {default}."
    if resolved < minimum:
        return minimum, f"Integer value {resolved} is below {minimum}; using {minimum}."
    if resolved > maximum:
        return maximum, f"Integer value {resolved} is above {maximum}; using {maximum}."
    return resolved, None


def _task_excerpt(task: dict[str, Any]) -> dict[str, Any]:
    """Build one compact task snapshot for memory indexing."""

    latest_result = (task.get("result") or [])[-1] if task.get("result") else None
    return {
        "task_id": task.get("task_id"),
        "description": task.get("description"),
        "task_type": task.get("task_type"),
        "target": _to_jsonable(task.get("target")),
        "type": task.get("type"),
        "status": task.get("status"),
        "next_task_id": task.get("next_task_id"),
        "branches": _to_jsonable(task.get("branches")),
        "plan_id": task.get("plan_id"),
        "user_input_id": task.get("user_input_id"),
        "depends_on": task.get("depends_on", []),
        "wait_for_event": task.get("wait_for_event"),
        "latest_result": _to_jsonable(latest_result) if latest_result else None,
    }


def _record_time(record: dict[str, Any]) -> str:
    """Return the best chronological timestamp for a memory record."""

    return str(
        record.get("recorded_at")
        or record.get("created_at")
        or record.get("completed_at")
        or record.get("updated_at")
        or ""
    )


def _row_preview(value: Any, limit: int = 220) -> str:
    """Return a short user-facing preview for memory table rows."""

    if isinstance(value, dict):
        for key in ("summary", "response", "text", "description", "message", "plan_text"):
            if value.get(key):
                return _truncate_text(value.get(key), limit)
        return _truncate_text(json.dumps(value, ensure_ascii=False), limit)
    return _truncate_text(value, limit)


def _extract_tool_summary(tool_trace: Optional[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return compact tool evidence suitable for task memory detail views."""

    if not isinstance(tool_trace, dict):
        return []
    calls = list(tool_trace.get("tool_calls") or [])
    results = list(tool_trace.get("tool_results") or [])
    summarized: list[dict[str, Any]] = []
    for index, call in enumerate(calls):
        if not isinstance(call, dict):
            continue
        result = results[index] if index < len(results) and isinstance(results[index], dict) else {}
        summarized.append(
            {
                "name": call.get("name"),
                "args": _to_jsonable(call.get("args") or {}),
                "status": result.get("status") or result.get("content", {}).get("status")
                if isinstance(result.get("content"), dict)
                else result.get("status"),
                "observation": _truncate_text(
                    result.get("observation")
                    or result.get("summary")
                    or result.get("content")
                    or "",
                    300,
                ),
                "result": _to_jsonable(result.get("content") or result),
            }
        )
    return summarized


def _state_excerpt(state: dict[str, Any]) -> dict[str, Any]:
    """Build one compact state excerpt without storing the full message history."""

    tasks = state.get("tasks", {}) if isinstance(state, dict) else {}
    events = list(state.get("events", [])) if isinstance(state, dict) else []
    background_results = (
        dict(state.get("background_results", {})) if isinstance(state, dict) else {}
    )

    task_items = []
    if isinstance(tasks, dict):
        for task_id in sorted(tasks):
            task = tasks[task_id]
            if isinstance(task, dict):
                task_items.append(_task_excerpt(task))

    background_excerpt = {}
    for task_id, result in background_results.items():
        if isinstance(result, dict):
            background_excerpt[str(task_id)] = {
                "status": result.get("status"),
                "summary": _truncate_text(result.get("summary", ""), 220),
                "latest_tool_name": result.get("latest_tool_name"),
                "candidate_keyframe_ids": list(
                    result.get("candidate_keyframe_ids", [])
                )[:8],
                "recommended_keyframe_id": result.get("recommended_keyframe_id"),
                "recommendation_confidence": result.get(
                    "recommendation_confidence"
                ),
            }
        else:
            background_excerpt[str(task_id)] = {"status": "completed", "summary": _truncate_text(result, 220)}

    return {
        "current_task_id": state.get("current_task_id"),
        "current_plan_id": state.get("current_plan_id"),
        "error_message": state.get("error_message"),
        "next_action": _to_jsonable(state.get("next_action")),
        "task_count": len(task_items),
        "tasks": task_items,
        "recent_events": _to_jsonable(events[-8:]),
        "background_results": background_excerpt,
        "turn_response_items": _to_jsonable(state.get("turn_response_items", [])),
        "turn_response_type": state.get("turn_response_type"),
        "turn_response_text": state.get("turn_response_text"),
        "user_facing_response": state.get("user_facing_response"),
    }


class AsyncAgentRunMemory:
    """Store full-session runtime records and expose structured query helpers."""

    def __init__(
        self,
        *,
        session_id: Optional[str] = None,
        session_dir: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        self._lock = threading.RLock()
        self.session_id = str(session_id or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self.session_dir = Path(session_dir) if session_dir else None
        self._latest_thread_id = "default"
        self.snapshot_path = (
            self.session_dir / "run_memory.json" if self.session_dir is not None else None
        )
        self._data: dict[str, Any] = {
            "session": {
                "session_id": self.session_id,
                "created_at": _utc_now_iso(),
                "session_dir": str(self.session_dir) if self.session_dir else None,
                "metadata": _to_jsonable(metadata or {}),
            },
            "threads": {},
            "turns": [],
            "events": [],
            "plans": [],
            "task_results": [],
            "tool_traces": [],
            "background_updates": [],
            "observations": [],
            "stream_updates": [],
            "responses": [],
            "checkpoints": [],
        }
        self._flush_locked()

    def _flush_locked(self) -> None:
        """Persist the current in-memory snapshot to disk when a session dir exists."""

        if self.snapshot_path is None:
            return

        self.snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.snapshot_path.with_name(
            f"{self.snapshot_path.stem}_{os.getpid()}.tmp"
        )
        temp_path.write_text(
            json.dumps(self._data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temp_path.replace(self.snapshot_path)
        self._export_memory_tables_locked()

    def _export_memory_tables_locked(self) -> None:
        """Persist the five simplified query tables for human review."""

        if self.session_dir is None:
            return
        write_memory_tables(
            self.session_dir,
            {
                "conversation": self._conversation_memory_rows_locked(),
                "plan": self._plan_memory_rows_locked(),
                "task": self._task_memory_rows_locked(),
                "navigation": self._navigation_memory_rows_locked(),
                "observation": self._observation_memory_rows_locked(),
            },
        )

    def _append_record(self, bucket: str, record: dict[str, Any]) -> None:
        """Append one record to the named bucket and persist the snapshot."""

        with self._lock:
            entry = _to_jsonable(record)
            if "recorded_at" not in entry:
                entry["recorded_at"] = _utc_now_iso()
            self._data.setdefault(bucket, []).append(entry)
            self._flush_locked()

    def update_session_metadata(self, **metadata: Any) -> None:
        """Merge additional session metadata into the top-level run snapshot."""

        with self._lock:
            existing = dict(self._data.get("session", {}).get("metadata", {}))
            existing.update(_to_jsonable(metadata))
            self._data["session"]["metadata"] = existing
            self._data["session"]["updated_at"] = _utc_now_iso()
            self._flush_locked()

    def record_turn_start(self, *, thread_id: str, role: str, message: str) -> None:
        """Record the start of one new user or system turn."""

        self._latest_thread_id = str(thread_id or "default")
        self._append_record(
            "turns",
            {
                "thread_id": thread_id,
                "role": role,
                "message": message,
                "status": "started",
            },
        )

    def record_turn_result(self, turn_result: dict[str, Any]) -> None:
        """Record the final structured result for one completed turn."""

        thread_id = str(turn_result.get("thread_id") or "default")
        self._latest_thread_id = thread_id
        final_entry = {
            "thread_id": thread_id,
            "role": turn_result.get("role"),
            "message": turn_result.get("message"),
            "response_items": turn_result.get("response_items", []),
            "turn_response_type": turn_result.get("turn_response_type"),
            "turn_response_text": turn_result.get("turn_response_text"),
            "visited_nodes": turn_result.get("visited_nodes", []),
            "step_trace": turn_result.get("step_trace", []),
            "saw_plan_node": turn_result.get("saw_plan_node"),
            "saw_navigation_activity": turn_result.get("saw_navigation_activity"),
            "status": "completed",
        }
        self._append_record("turns", final_entry)
        self.sync_thread_state(thread_id=thread_id, state=turn_result.get("state", {}))
        response_items = list(turn_result.get("response_items", []) or [])
        if response_items:
            for item in response_items:
                self._record_turn_response_item(
                    thread_id=thread_id,
                    role=str(turn_result.get("role") or "unknown"),
                    item=item,
                )
            return

        response_text = str(turn_result.get("turn_response_text") or "").strip()
        if response_text:
            self.record_response(
                thread_id=thread_id,
                response=response_text,
                role=str(turn_result.get("role") or "unknown"),
            )

    def sync_thread_state(self, *, thread_id: str, state: dict[str, Any]) -> None:
        """Store the latest compact state snapshot for one thread."""

        self._latest_thread_id = str(thread_id or "default")
        with self._lock:
            self._data.setdefault("threads", {})
            self._data["threads"][thread_id] = {
                "thread_id": thread_id,
                "updated_at": _utc_now_iso(),
                "state_excerpt": _state_excerpt(state or {}),
            }
            self._flush_locked()

    def record_stream_update(
        self,
        *,
        thread_id: str,
        node_name: str,
        step_summary: dict[str, Any],
    ) -> None:
        """Record one streamed node update emitted during a turn."""

        self._append_record(
            "stream_updates",
            {
                "thread_id": thread_id,
                "node_name": node_name,
                "step_summary": step_summary,
            },
        )
        for item in list((step_summary or {}).get("turn_response_items", []) or []):
            if isinstance(item, dict):
                self._record_turn_response_item(
                    thread_id=thread_id,
                    role="system",
                    item=item,
                )

    def record_event(
        self,
        event: dict[str, Any],
        *,
        thread_id: Optional[str] = None,
        stage: Optional[str] = None,
    ) -> None:
        """Record one structured workflow event."""

        payload = dict(event or {})
        if thread_id:
            payload["thread_id"] = thread_id
        if stage:
            payload["stage"] = stage
        self._append_record("events", payload)

    def record_user_input(
        self,
        user_input: dict[str, Any],
        *,
        thread_id: Optional[str] = None,
    ) -> None:
        """Record one normalized user-input item as a checkpoint entry."""

        record = dict(user_input or {})
        if thread_id:
            record["thread_id"] = thread_id
        self._append_record("checkpoints", {"kind": "user_input", "data": record})

    def record_plan(
        self,
        *,
        plan_id: str,
        plan_mode: str,
        user_input_id: Optional[str],
        tasks: dict[int, dict[str, Any]],
        plan_text: Optional[str] = None,
        first_task_id: Optional[int] = None,
        thread_id: Optional[str] = None,
    ) -> None:
        """Record one plan snapshot and its compact task layout."""

        task_items = []
        for task_id in sorted(tasks):
            task = tasks[task_id]
            if isinstance(task, dict):
                task_items.append(_task_excerpt(task))
        self._append_record(
            "plans",
            {
                "thread_id": thread_id,
                "plan_id": plan_id,
                "plan_mode": plan_mode,
                "user_input_id": user_input_id,
                "first_task_id": first_task_id,
                "task_count": len(task_items),
                "tasks": task_items,
                "plan_text": _truncate_text(plan_text, 1200) if plan_text else None,
            },
        )

    def record_task_result(
        self,
        *,
        task: dict[str, Any],
        event_type: str,
        summary: str,
        tool_trace: Optional[dict[str, Any]] = None,
        thread_id: Optional[str] = None,
    ) -> None:
        """Record one task outcome with optional tool-trace evidence."""

        entry = {
            "thread_id": thread_id,
            "task_id": task.get("task_id"),
            "description": task.get("description"),
            "status": task.get("status"),
            "task_type": task.get("task_type"),
            "target": _to_jsonable(task.get("target")),
            "type": task.get("type"),
            "plan_id": task.get("plan_id"),
            "user_input_id": task.get("user_input_id"),
            "depends_on": _to_jsonable(task.get("depends_on", [])),
            "result": _to_jsonable(task.get("result", [])),
            "origin": task.get("origin"),
            "error": task.get("error"),
            "event_type": event_type,
            "summary": summary,
        }
        if tool_trace:
            entry["tool_trace_excerpt"] = {
                "tool_calls": tool_trace.get("tool_calls", []),
                "tool_results": tool_trace.get("tool_results", []),
                "final_ai_content": _truncate_text(tool_trace.get("final_ai_content", ""), 500),
            }
        self._append_record("task_results", entry)
        if tool_trace:
            observation_summary = _observation_summary_from_trace(
                summary=summary,
                tool_trace=tool_trace,
            )
            if observation_summary:
                position, details = _observation_location_from_trace(tool_trace)
                self.record_observation(
                    summary=observation_summary,
                    task_id=_safe_int(task.get("task_id")),
                    plan_id=task.get("plan_id"),
                    thread_id=thread_id,
                    position=position,
                    source="tool_trace",
                    details={
                        **details,
                        "task_description": task.get("description"),
                        "task_type": task.get("task_type"),
                    },
                )

    def record_observation(
        self,
        *,
        summary: str,
        task_id: Optional[int] = None,
        plan_id: Optional[str] = None,
        thread_id: Optional[str] = None,
        keyframe_id: Optional[int] = None,
        position: Optional[list[float]] = None,
        anchor_id: Optional[str] = None,
        source: str = "task_result",
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        """Record one narrow observation memory item."""

        self._append_record(
            "observations",
            {
                "thread_id": thread_id,
                "plan_id": plan_id,
                "task_id": task_id,
                "summary": summary,
                "keyframe_id": keyframe_id,
                "position": _safe_float_triplet(position),
                "anchor_id": anchor_id,
                "source": source,
                "details": _to_jsonable(details or {}),
            },
        )

    def record_tool_trace(
        self,
        *,
        task: dict[str, Any],
        tool_trace: dict[str, Any],
        thread_id: Optional[str] = None,
    ) -> None:
        """Record the full tool trace for one executed task."""

        self._append_record(
            "tool_traces",
            {
                "thread_id": thread_id,
                "task_id": task.get("task_id"),
                "description": task.get("description"),
                "plan_id": task.get("plan_id"),
                "user_input_id": task.get("user_input_id"),
                "tool_trace": tool_trace,
            },
        )

    def record_background_update(
        self,
        *,
        task_id: int,
        task_description: str,
        record: dict[str, Any],
        worker_name: Optional[str] = None,
        thread_id: Optional[str] = None,
    ) -> None:
        """Record one background-analysis lifecycle update."""

        self._append_record(
            "background_updates",
            {
                "thread_id": thread_id,
                "worker_name": worker_name,
                "task_id": task_id,
                "task_description": task_description,
                "record": record,
            },
        )

    def record_response(
        self,
        *,
        thread_id: str,
        response: str,
        role: str = "assistant",
        response_type: str = "result",
        response_id: Optional[str] = None,
        task_id: Optional[int] = None,
        source_event_type: Optional[str] = None,
    ) -> None:
        """Record one final user-facing response."""

        self._append_record(
            "responses",
            {
                "thread_id": thread_id,
                "role": role,
                "response": response,
                "response_type": response_type,
                "response_id": response_id,
                "task_id": task_id,
                "source_event_type": source_event_type,
            },
        )

    def _record_turn_response_item(
        self,
        *,
        thread_id: str,
        role: str,
        item: dict[str, Any],
    ) -> None:
        """Persist one streamed/final turn response item, deduped by response id."""

        response_text = str(item.get("response_text") or "").strip()
        if not response_text:
            return

        response_id = str(item.get("response_id") or "").strip() or None
        if response_id:
            with self._lock:
                for response in self._data.get("responses", []):
                    if response.get("response_id") == response_id:
                        return

        self.record_response(
            thread_id=thread_id,
            response=response_text,
            role=role,
            response_type=str(item.get("response_type") or "result"),
            response_id=response_id,
            task_id=item.get("task_id"),
            source_event_type=str(item.get("source_event_type") or "").strip() or None,
        )

    def snapshot(self) -> dict[str, Any]:
        """Return one deep copy of the full in-memory snapshot."""

        with self._lock:
            return deepcopy(self._data)

    def _route_item_from_task_result(self, item: dict[str, Any]) -> Optional[dict[str, Any]]:
        """Build one confirmed route item from a completed navigation task result."""

        if str(item.get("event_type") or "").strip() != "task_completed":
            return None
        description = str(item.get("description") or "").strip()
        if not description:
            return None
        if str(item.get("task_type") or "").strip() != "navigation_action":
            return None

        enriched_item = self._attach_tool_trace_excerpt_to_task_item_locked(item)
        tool_data = _extract_tool_result_json(enriched_item, "go_to_keyframe")
        data = tool_data.get("data") if isinstance(tool_data, dict) else None
        if not isinstance(data, dict):
            data = {}
        keyframe_id = _safe_int(data.get("target_keyframe_id"))
        position = _safe_float_triplet(data.get("target_position"))
        if keyframe_id is None and position is None:
            return None

        return {
            "plan_id": item.get("plan_id"),
            "task_id": item.get("task_id"),
            "description": description,
            "destination_keyframe_id": keyframe_id,
            "destination_position": position,
            "arrived_at": item.get("recorded_at"),
            "source_event_id": None,
            "source": "task_result",
        }

    def _route_item_from_navigation_arrival_event(
        self,
        event: dict[str, Any],
    ) -> Optional[dict[str, Any]]:
        """Build one confirmed route item from a navigation_arrived event."""

        if str(event.get("type") or event.get("event_family") or "").strip() != "navigation_arrived":
            return None
        payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
        description = str(
            payload.get("destination_description")
            or payload.get("description")
            or payload.get("summary")
            or ""
        ).strip()
        if not description:
            return None

        keyframe_id = _safe_int(payload.get("destination_keyframe_id"))
        position = _safe_float_triplet(payload.get("destination_position"))
        if keyframe_id is None and position is None:
            return None

        return {
            "plan_id": event.get("plan_id"),
            "task_id": event.get("task_id"),
            "description": description,
            "destination_keyframe_id": keyframe_id,
            "destination_position": position,
            "arrived_at": event.get("recorded_at") or event.get("created_at"),
            "source_event_id": event.get("event_id"),
            "source": "event",
        }

    def _confirmed_route_items_locked(self) -> list[dict[str, Any]]:
        """Return deduplicated confirmed route arrivals in chronological order."""

        raw_items: list[dict[str, Any]] = []
        for event in self._data.get("events", []):
            route_item = self._route_item_from_navigation_arrival_event(event)
            if route_item is not None:
                raw_items.append(route_item)
        for item in self._data.get("task_results", []):
            route_item = self._route_item_from_task_result(item)
            if route_item is not None:
                raw_items.append(route_item)

        raw_items = sorted(raw_items, key=lambda item: str(item.get("arrived_at") or ""))
        deduped_by_identity: dict[tuple[Any, Any, Any, str], dict[str, Any]] = {}
        for item in raw_items:
            identity = (
                item.get("plan_id"),
                item.get("task_id"),
                item.get("destination_keyframe_id"),
                json.dumps(item.get("destination_position"), ensure_ascii=False),
            )
            existing = deduped_by_identity.get(identity)
            if existing is None:
                deduped_by_identity[identity] = dict(item)
                continue
            if item.get("source") == "event":
                existing.update({key: value for key, value in item.items() if value is not None})

        route_items = sorted(
            deduped_by_identity.values(),
            key=lambda item: str(item.get("arrived_at") or ""),
        )
        for index, item in enumerate(route_items, start=1):
            item["order"] = index
        return route_items

    def _attach_tool_trace_excerpt_to_task_item_locked(
        self,
        item: dict[str, Any],
    ) -> dict[str, Any]:
        """Attach the latest tool trace excerpt for matching task records."""

        enriched = dict(item)
        task_id = _safe_int(item.get("task_id"))
        if task_id is None:
            return enriched
        plan_id = str(item.get("plan_id") or "").strip()
        description = str(item.get("description") or "").strip()
        for trace_item in reversed(self._data.get("tool_traces", [])):
            if _safe_int(trace_item.get("task_id")) != task_id:
                continue
            if plan_id and str(trace_item.get("plan_id") or "").strip() != plan_id:
                continue
            if (
                description
                and str(trace_item.get("description") or "").strip()
                and str(trace_item.get("description") or "").strip() != description
            ):
                continue
            trace_payload = trace_item.get("tool_trace")
            if isinstance(trace_payload, dict):
                enriched["tool_trace_excerpt"] = {
                    "tool_calls": list(trace_payload.get("tool_calls", []))[:6],
                    "tool_results": list(trace_payload.get("tool_results", []))[:6],
                    "final_ai_content": _truncate_text(
                        trace_payload.get("final_ai_content"),
                        500,
                    ),
                }
            break
        return enriched

    def _conversation_memory_rows_locked(self) -> list[dict[str, Any]]:
        """Return conversation memory rows grouped by user/system turn."""

        turns = list(self._data.get("turns", []))
        responses = list(self._data.get("responses", []))
        rows: list[dict[str, Any]] = []
        start_indices = [
            index
            for index, turn in enumerate(turns)
            if str(turn.get("status") or "").strip() == "started"
        ]
        if not start_indices:
            start_indices = list(range(len(turns)))

        for ordinal, turn_index in enumerate(start_indices, start=1):
            turn = turns[turn_index]
            next_start_index = (
                start_indices[ordinal] if ordinal < len(start_indices) else len(turns)
            )
            thread_id = str(turn.get("thread_id") or "default")
            started_at = _record_time(turn)
            next_started_at = (
                _record_time(turns[next_start_index])
                if next_start_index < len(turns)
                else ""
            )
            completed_turn = None
            for candidate in turns[turn_index + 1 : next_start_index]:
                if (
                    str(candidate.get("thread_id") or "default") == thread_id
                    and str(candidate.get("status") or "").strip() == "completed"
                ):
                    completed_turn = candidate
                    break

            assigned_responses = []
            for response in responses:
                if str(response.get("thread_id") or "default") != thread_id:
                    continue
                response_time = _record_time(response)
                if started_at and response_time and response_time < started_at:
                    continue
                if next_started_at and response_time and response_time >= next_started_at:
                    continue
                assigned_responses.append(
                    {
                        "type": response.get("response_type") or "result",
                        "text": response.get("response") or "",
                        "trigger": response.get("source_event_type"),
                        "plan_id": response.get("plan_id"),
                        "task_id": response.get("task_id"),
                        "created_at": response.get("recorded_at"),
                    }
                )

            role = str(turn.get("role") or "system").strip() or "system"
            rows.append(
                {
                    "row_id": f"conversation_{ordinal}",
                    "turn_id": f"turn_{ordinal}",
                    "thread_id": thread_id,
                    "source": "user" if role == "user" else "system",
                    "input": {
                        "type": "user_message" if role == "user" else "system_event",
                        "text": turn.get("message") or "",
                        "created_at": turn.get("recorded_at"),
                    },
                    "agent_responses": assigned_responses,
                    "status": (completed_turn or turn).get("status") or "started",
                    "created_at": turn.get("recorded_at"),
                    "updated_at": (completed_turn or turn).get("recorded_at"),
                }
            )
        return rows

    def _plan_memory_rows_locked(self) -> list[dict[str, Any]]:
        """Return simplified plan-version memory rows."""

        rows: list[dict[str, Any]] = []
        versions_by_plan: dict[str, int] = {}
        for index, plan in enumerate(self._data.get("plans", []), start=1):
            plan_id = str(plan.get("plan_id") or "")
            versions_by_plan[plan_id] = versions_by_plan.get(plan_id, 0) + 1
            tasks = []
            for task in list(plan.get("tasks") or []):
                if not isinstance(task, dict):
                    continue
                tasks.append(
                    {
                        "task_id": task.get("task_id"),
                        "description": task.get("description"),
                        "task_type": task.get("task_type") or task.get("type"),
                        "target": _to_jsonable(task.get("target")),
                        "depends_on": _to_jsonable(task.get("depends_on", [])),
                        "next_task_id": task.get("next_task_id"),
                        "branches": _to_jsonable(task.get("branches")),
                    }
                )
            rows.append(
                {
                    "row_id": f"plan_{index}",
                    "plan_id": plan.get("plan_id"),
                    "version": versions_by_plan[plan_id],
                    "mode": plan.get("plan_mode"),
                    "user_turn_id": plan.get("user_input_id"),
                    "user_goal": plan.get("plan_text"),
                    "entry_task_id": plan.get("first_task_id"),
                    "task_count": plan.get("task_count"),
                    "tasks": tasks,
                    "thread_id": plan.get("thread_id"),
                    "created_at": plan.get("recorded_at"),
                }
            )
        return rows

    def _task_memory_rows_locked(self) -> list[dict[str, Any]]:
        """Return simplified task execution memory rows."""

        rows: list[dict[str, Any]] = []
        for index, item in enumerate(self._data.get("task_results", []), start=1):
            if not isinstance(item, dict):
                continue
            enriched = self._attach_tool_trace_excerpt_to_task_item_locked(item)
            tool_trace = enriched.get("tool_trace_excerpt")
            result = enriched.get("result")
            if isinstance(result, list) and len(result) == 1:
                result = result[0]
            rows.append(
                {
                    "row_id": f"task_{index}",
                    "task_id": enriched.get("task_id"),
                    "plan_id": enriched.get("plan_id"),
                    "turn_id": enriched.get("thread_id"),
                    "thread_id": enriched.get("thread_id"),
                    "task_type": enriched.get("task_type") or enriched.get("type"),
                    "description": enriched.get("description"),
                    "status": enriched.get("status"),
                    "depends_on": _to_jsonable(enriched.get("depends_on", [])),
                    "event_type": enriched.get("event_type"),
                    "summary": enriched.get("summary"),
                    "result": _to_jsonable(result),
                    "tools": _extract_tool_summary(tool_trace),
                    "origin": enriched.get("origin"),
                    "error": enriched.get("error"),
                    "created_at": enriched.get("recorded_at"),
                }
            )
        return rows

    def _navigation_memory_rows_locked(self) -> list[dict[str, Any]]:
        """Return route/anchor memory rows derived from confirmed arrivals."""

        rows: list[dict[str, Any]] = []
        for index, item in enumerate(self._confirmed_route_items_locked(), start=1):
            rows.append(
                {
                    "row_id": f"navigation_{index}",
                    "order": item.get("order") or index,
                    "anchor_id": f"anchor_{index}",
                    "label": item.get("description"),
                    "description": item.get("description"),
                    "plan_id": item.get("plan_id"),
                    "task_id": item.get("task_id"),
                    "keyframe_id": item.get("destination_keyframe_id"),
                    "position": item.get("destination_position"),
                    "arrived_at": item.get("arrived_at"),
                    "source": item.get("source"),
                    "source_event_id": item.get("source_event_id"),
                    "created_at": item.get("arrived_at"),
                }
            )
        return rows

    def _observation_memory_rows_locked(self) -> list[dict[str, Any]]:
        """Return narrow observation memory rows."""

        rows: list[dict[str, Any]] = []
        for index, item in enumerate(self._data.get("observations", []), start=1):
            if not isinstance(item, dict):
                continue
            rows.append(
                {
                    "row_id": f"observation_{index}",
                    "summary": item.get("summary"),
                    "plan_id": item.get("plan_id"),
                    "task_id": item.get("task_id"),
                    "turn_id": item.get("thread_id"),
                    "thread_id": item.get("thread_id"),
                    "keyframe_id": item.get("keyframe_id"),
                    "position": item.get("position"),
                    "anchor_id": item.get("anchor_id"),
                    "source": item.get("source"),
                    "details": _to_jsonable(item.get("details") or {}),
                    "created_at": item.get("recorded_at"),
                }
            )
        return rows

    def _memory_scope_rows_locked(self, scope: str) -> list[dict[str, Any]]:
        """Return raw simplified rows for one memory scope."""

        if scope == "conversation":
            return self._conversation_memory_rows_locked()
        if scope == "plan":
            return self._plan_memory_rows_locked()
        if scope == "task":
            return self._task_memory_rows_locked()
        if scope == "navigation":
            return self._navigation_memory_rows_locked()
        if scope == "observation":
            return self._observation_memory_rows_locked()
        return []

    def _filter_memory_rows(
        self,
        rows: list[dict[str, Any]],
        *,
        time: str,
        plan_id: str,
        turn_id: str,
        task_id: int,
        row_id: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        """Apply the intentionally small query_memory filter set."""

        selected = list(rows)
        if plan_id:
            selected = [row for row in selected if str(row.get("plan_id") or "") == plan_id]
        if turn_id:
            selected = [
                row
                for row in selected
                if str(row.get("turn_id") or "") == turn_id
                or str(row.get("thread_id") or "") == turn_id
            ]
        if task_id >= 0:
            selected = [
                row for row in selected if _safe_int(row.get("task_id")) == int(task_id)
            ]
        if row_id:
            selected = [row for row in selected if str(row.get("row_id") or "") == row_id]
        selected = sorted(selected, key=_record_time)
        if time == "recent":
            return selected[-limit:]
        return selected[:limit]

    def _render_memory_rows(
        self,
        *,
        scope: str,
        view: str,
        rows: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Render simplified memory rows into one of the supported views."""

        if view == "detail":
            return deepcopy(rows)
        if view == "timeline":
            rendered = []
            for row in rows:
                when = row.get("created_at") or row.get("arrived_at") or row.get("updated_at")
                if scope == "conversation":
                    text = row.get("input", {}).get("text")
                    responses = [
                        response.get("text")
                        for response in list(row.get("agent_responses") or [])
                        if isinstance(response, dict) and response.get("text")
                    ]
                    summary = " | ".join([_truncate_text(text, 160), *[_truncate_text(item, 160) for item in responses]]).strip(" |")
                elif scope == "plan":
                    summary = f"{row.get('mode')} plan {row.get('plan_id')} v{row.get('version')} with {row.get('task_count')} task(s)."
                elif scope == "task":
                    summary = f"Task {row.get('task_id')} {row.get('status')}: {_row_preview(row.get('summary') or row.get('description'))}"
                elif scope == "navigation":
                    summary = f"Arrived at {row.get('description')} (keyframe={row.get('keyframe_id')}, position={row.get('position')})."
                else:
                    summary = _row_preview(row.get("summary") or row)
                rendered.append(
                    {
                        "row_id": row.get("row_id"),
                        "time": when,
                        "scope": scope,
                        "summary": summary,
                        "plan_id": row.get("plan_id"),
                        "task_id": row.get("task_id"),
                    }
                )
            return rendered

        rendered = []
        for row in rows:
            if scope == "conversation":
                preview = row.get("input", {}).get("text")
                response_count = len(list(row.get("agent_responses") or []))
                extra = {"source": row.get("source"), "response_count": response_count}
            elif scope == "plan":
                preview = row.get("user_goal")
                extra = {"mode": row.get("mode"), "version": row.get("version"), "task_count": row.get("task_count")}
            elif scope == "task":
                preview = row.get("summary") or row.get("description")
                extra = {"status": row.get("status"), "task_type": row.get("task_type"), "origin": row.get("origin")}
            elif scope == "navigation":
                preview = row.get("description")
                extra = {"order": row.get("order"), "keyframe_id": row.get("keyframe_id"), "position": row.get("position")}
            else:
                preview = row.get("summary")
                extra = {"keyframe_id": row.get("keyframe_id"), "anchor_id": row.get("anchor_id")}
            rendered.append(
                {
                    "row_id": row.get("row_id"),
                    "scope": scope,
                    "time": row.get("created_at") or row.get("arrived_at") or row.get("updated_at"),
                    "preview": _truncate_text(preview, 240),
                    "plan_id": row.get("plan_id"),
                    "task_id": row.get("task_id"),
                    **extra,
                }
            )
        return rendered

    def query_memory(
        self,
        *,
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
        """Query simplified conversation/plan/task/navigation/observation memory."""

        warnings: list[str] = []
        valid_scopes = {"conversation", "plan", "task", "navigation", "observation", "all"}
        normalized_scope = str(scope or "all").strip().lower()
        if normalized_scope not in valid_scopes:
            warnings.append(f"Invalid scope {scope!r}; normalized to 'all'.")
            normalized_scope = "all"
        valid_views = {"summary_table", "timeline", "detail"}
        normalized_view = str(view or "summary_table").strip().lower()
        if normalized_view not in valid_views:
            warnings.append(f"Invalid view {view!r}; normalized to 'summary_table'.")
            normalized_view = "summary_table"
        normalized_time = str(time or "recent").strip().lower()
        if normalized_time not in {"recent", "all"}:
            warnings.append(f"Invalid time {time!r}; normalized to 'all'.")
            normalized_time = "all"
        resolved_limit, limit_warning = _coerce_positive_int(
            limit,
            default=50,
            minimum=1,
            maximum=50,
        )
        if limit_warning:
            warnings.append(limit_warning)
        try:
            resolved_task_id = int(task_id) if task_id is not None else -1
        except Exception:
            warnings.append(f"Invalid task_id {task_id!r}; normalized to -1.")
            resolved_task_id = -1

        normalized_query = str(query or "").strip()
        normalized_plan_id = str(plan_id or "").strip()
        normalized_turn_id = str(turn_id or "").strip()
        normalized_row_id = str(row_id or "").strip()
        scopes = (
            ["conversation", "plan", "task", "navigation", "observation"]
            if normalized_scope == "all"
            else [normalized_scope]
        )

        with self._lock:
            rendered_items: list[dict[str, Any]] = []
            raw_count = 0
            for scoped_name in scopes:
                rows = self._memory_scope_rows_locked(scoped_name)
                selected = self._filter_memory_rows(
                    rows,
                    time=normalized_time,
                    plan_id=normalized_plan_id,
                    turn_id=normalized_turn_id,
                    task_id=resolved_task_id,
                    row_id=normalized_row_id,
                    limit=resolved_limit,
                )
                raw_count += len(selected)
                rendered = self._render_memory_rows(
                    scope=scoped_name,
                    view=normalized_view,
                    rows=selected,
                )
                if normalized_scope == "all":
                    for item in rendered:
                        item.setdefault("scope", scoped_name)
                rendered_items.extend(rendered)

        rendered_items = sorted(
            rendered_items,
            key=lambda item: str(item.get("time") or item.get("created_at") or ""),
        )
        if normalized_time == "recent":
            rendered_items = rendered_items[-resolved_limit:]
        else:
            rendered_items = rendered_items[:resolved_limit]

        return {
            "status": "ok",
            "summary": "Memory query returned {count} item(s) from scope `{scope}` as `{view}`.".format(
                count=len(rendered_items),
                scope=normalized_scope,
                view=normalized_view,
            ),
            "scope": normalized_scope,
            "view": normalized_view,
            "query": normalized_query,
            "time": normalized_time,
            "plan_id": normalized_plan_id or None,
            "turn_id": normalized_turn_id or None,
            "task_id": resolved_task_id if resolved_task_id >= 0 else None,
            "row_id": normalized_row_id or None,
            "normalized_args": {
                "scope": normalized_scope,
                "view": normalized_view,
                "query": normalized_query,
                "time": normalized_time,
                "plan_id": normalized_plan_id,
                "turn_id": normalized_turn_id,
                "task_id": resolved_task_id,
                "row_id": normalized_row_id,
                "limit": resolved_limit,
            },
            "warnings": warnings,
            "raw_match_count": raw_count,
            "items": deepcopy(rendered_items),
        }

__all__ = ["AsyncAgentRunMemory"]
