"""Minimal local web UI for AsyncAgent message input and live plan inspection."""

from __future__ import annotations

import argparse
import base64
import errno
import json
import mimetypes
import re
import threading
import time
from collections import deque
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from caragent_agent.agents.async_agent import AsyncAgent
from caragent_agent.agents.async_agent.execution.support import (
    normalize_turn_response_items,
)
from caragent_agent.agents.async_agent.planning.plan_graph import (
    iter_plan_edges,
    summarize_plan_graph,
    validate_plan_graph,
)
from caragent_agent.agents.async_agent.planning.task_graph import (
    collect_ordered_task_ids_for_plan,
    get_task_progress_context,
)
from caragent_agent.config.config import config
from caragent_agent.config.runtime_paths import (
    get_default_scene_dataset_dir,
    normalize_runtime_path,
)
from caragent_agent.impression_graph.scene_memory import SceneMemory
from caragent_agent.io_adapters import (
    adapt_turn_result_language,
    current_controller_image,
    describe_image_for_navigation,
    image_from_data_url,
    image_to_data_url,
    normalize_language,
    prepare_user_message_for_agent,
)


CLIENT_DISCONNECT_ERRNOS = {
    errno.EPIPE,
    errno.ECONNRESET,
    getattr(errno, "ECONNABORTED", 103),
    10053,
    10054,
}


def _is_client_disconnect_error(exc: BaseException) -> bool:
    """Return True for normal browser disconnects during HTTP writes."""

    if isinstance(exc, (BrokenPipeError, ConnectionResetError, ConnectionAbortedError)):
        return True
    raw_errno = getattr(exc, "errno", None)
    try:
        return int(raw_errno) in CLIENT_DISCONNECT_ERRNOS
    except Exception:
        return False
from caragent_agent.agents.async_agent.guidance import get_interaction_profile

DEFAULT_DATASET_DIR = get_default_scene_dataset_dir()
LOG_ENTRY_START_RE = re.compile(r"^\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}\]")
CHECKPOINT_SCHEMA_VERSION = 1
CHECKPOINT_STATE_KEYS = (
    "tasks",
    "current_task_id",
    "current_plan_id",
    "next_action",
    "background_results",
    "events",
    "processed_event_ids",
    "user_inputs",
    "active_navigation",
    "pending_navigation",
    "navigation_arrival_receipts",
    "guidance_events",
    "turn_response_items",
    "turn_response_type",
    "turn_response_text",
    "turn_response_id",
    "user_facing_response",
    "user_facing_response_id",
    "error_message",
)
ACTIVE_TASK_STATUSES = {"pending", "in_progress", "running", "waiting"}


def file_to_data_url(path: Path) -> str:
    """Encode one local image file as a browser data URL."""

    mime_type = mimetypes.guess_type(str(path))[0] or "image/jpeg"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{data}"


def _safe_checkpoint_name(thread_id: str) -> str:
    """Return a filesystem-safe checkpoint stem for one LangGraph thread."""

    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(thread_id or "").strip())
    return safe_name.strip("._") or "web_console"


def _default_checkpoint_path(thread_id: str) -> Path:
    """Return the default persisted Web UI checkpoint path."""

    log_dir = normalize_runtime_path(config.get("log_dir", "logs"))
    return log_dir / "web_checkpoints" / f"{_safe_checkpoint_name(thread_id)}.json"


def _json_safe(value: Any) -> Any:
    """Convert runtime values into a compact JSON-safe checkpoint payload."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]

    content = getattr(value, "content", None)
    if content is not None:
        return {
            "type": value.__class__.__name__,
            "content": str(content),
        }
    return str(value)


def _int_keyed_dict(value: Any) -> dict[int, Any]:
    """Coerce JSON-loaded string keys back to int keys where possible."""

    if not isinstance(value, dict):
        return {}

    result: dict[int, Any] = {}
    for raw_key, item in value.items():
        try:
            key = int(raw_key)
        except (TypeError, ValueError):
            continue
        result[key] = item
    return result


def _state_for_checkpoint(state: dict[str, Any]) -> dict[str, Any]:
    """Keep only UI-relevant state fields in the persisted checkpoint."""

    if not isinstance(state, dict):
        return {}
    return {
        key: _json_safe(state.get(key))
        for key in CHECKPOINT_STATE_KEYS
        if key in state
    }


def _normalize_checkpoint_state(state: Any) -> dict[str, Any]:
    """Normalize JSON-loaded state into the runtime shape used by the UI."""

    if not isinstance(state, dict):
        return {}
    normalized = dict(state)
    normalized["tasks"] = _int_keyed_dict(normalized.get("tasks"))
    normalized["background_results"] = _int_keyed_dict(
        normalized.get("background_results")
    )
    next_action = normalized.get("next_action")
    if not isinstance(next_action, dict):
        normalized["next_action"] = {"type": "idle"}
    return normalized


def _state_has_thread_content(state: dict[str, Any]) -> bool:
    """Return True when a live in-process graph state should take precedence."""

    if not isinstance(state, dict) or not state:
        return False
    if state.get("messages") or state.get("tasks"):
        return True
    if state.get("current_plan_id") or state.get("current_task_id") is not None:
        return True
    if state.get("events") or state.get("user_inputs"):
        return True
    return False


def _state_has_active_plan_residue(state: dict[str, Any]) -> bool:
    """Return True when a restored checkpoint contains in-flight task state."""

    if not isinstance(state, dict):
        return False
    if state.get("current_plan_id") or state.get("current_task_id") is not None:
        return True
    next_action = state.get("next_action")
    if isinstance(next_action, dict) and next_action.get("type") not in (None, "", "idle"):
        return True
    if state.get("active_navigation") or state.get("pending_navigation"):
        return True

    tasks = state.get("tasks")
    if not isinstance(tasks, dict):
        return False
    for task in tasks.values():
        if not isinstance(task, dict):
            continue
        if str(task.get("status") or "").strip().lower() in ACTIVE_TASK_STATUSES:
            return True
    return False


def _clear_resumed_runtime_state(state: dict[str, Any]) -> dict[str, Any]:
    """Drop restored in-flight plan/task state so a restarted robot starts idle."""

    if not isinstance(state, dict):
        return {"tasks": {}, "current_task_id": None, "current_plan_id": None, "next_action": {"type": "idle"}}

    return {
        "tasks": {},
        "current_task_id": None,
        "current_plan_id": None,
        "background_results": {},
        "events": [],
        "processed_event_ids": [],
        "user_inputs": [],
        "active_navigation": {},
        "pending_navigation": {},
        "navigation_arrival_receipts": [],
        "guidance_events": [],
        "next_action": {"type": "idle"},
        "turn_response_items": [],
        "turn_response_type": "none",
        "turn_response_text": "",
        "user_facing_response": "",
    }


def _normalize_checkpoint_turns(turns: Any) -> list[dict[str, Any]]:
    """Load visible completed chat turns and drop interrupted in-flight turns."""

    if not isinstance(turns, list):
        return []

    normalized: list[dict[str, Any]] = []
    for item in turns[-80:]:
        if not isinstance(item, dict):
            continue
        turn = dict(item)
        if str(turn.get("status") or "").strip().lower() == "running":
            continue
        normalized.append(turn)
    return normalized


def _checkpoint_payload_has_resume_content(payload: dict[str, Any]) -> bool:
    """Return True when a checkpoint should ask the user to resume or start fresh."""

    if not isinstance(payload, dict):
        return False
    if _normalize_checkpoint_turns(payload.get("turn_history")):
        return True
    checkpoint_state = _normalize_checkpoint_state(
        payload.get("state_cache") or payload.get("state") or {}
    )
    return _state_has_thread_content(checkpoint_state)


def _run_memory_record_matches(record: dict[str, Any], thread_id: str) -> bool:
    """Return True when one run-memory row belongs to this Web thread."""

    record_thread_id = str(record.get("thread_id") or "").strip()
    return not record_thread_id or record_thread_id == str(thread_id)


def _run_memory_state_for_checkpoint(data: dict[str, Any], thread_id: str) -> dict[str, Any]:
    """Build a UI checkpoint state cache from a run-memory thread excerpt."""

    threads = data.get("threads") if isinstance(data.get("threads"), dict) else {}
    thread = threads.get(thread_id) if isinstance(threads.get(thread_id), dict) else {}
    state_excerpt = (
        thread.get("state_excerpt") if isinstance(thread.get("state_excerpt"), dict) else {}
    )
    tasks = {}
    for task in list(state_excerpt.get("tasks") or []):
        if not isinstance(task, dict):
            continue
        task_id = task.get("task_id")
        if task_id is not None:
            tasks[str(task_id)] = dict(task)
    return {
        "tasks": tasks,
        "current_task_id": state_excerpt.get("current_task_id"),
        "current_plan_id": state_excerpt.get("current_plan_id"),
        "background_results": state_excerpt.get("background_results") or {},
        "events": state_excerpt.get("recent_events") or [],
        "processed_event_ids": [],
        "user_inputs": [],
        "active_navigation": {},
        "pending_navigation": {},
        "navigation_arrival_receipts": [],
        "guidance_events": state_excerpt.get("guidance_events") or [],
        "next_action": state_excerpt.get("next_action") or {"type": "idle"},
        "turn_response_items": state_excerpt.get("turn_response_items") or [],
        "turn_response_type": state_excerpt.get("turn_response_type") or "none",
        "turn_response_text": state_excerpt.get("turn_response_text") or "",
        "user_facing_response": state_excerpt.get("user_facing_response") or "",
    }


def _run_memory_turn_history(data: dict[str, Any], thread_id: str) -> list[dict[str, Any]]:
    """Convert completed run-memory turns into visible Web turn history."""

    completed = [
        item
        for item in list(data.get("turns") or [])
        if isinstance(item, dict)
        and _run_memory_record_matches(item, thread_id)
        and str(item.get("status") or "").strip() == "completed"
    ]
    history: list[dict[str, Any]] = []
    for index, item in enumerate(completed[-80:], start=1):
        role = str(item.get("role") or "user")
        history.append(
            {
                "turn_id": index,
                "role": role,
                "role_label": "System" if role == "system" else "User Instruction",
                "source": "restored-run-memory",
                "content": str(item.get("message") or ""),
                "response": str(item.get("turn_response_text") or ""),
                "response_items": item.get("response_items") or [],
                "created_at": item.get("recorded_at") or "",
                "updated_at": item.get("recorded_at") or "",
                "finished_at": item.get("recorded_at") or "",
                "visited_nodes": item.get("visited_nodes") or [],
                "step_trace": item.get("step_trace") or [],
                "status": "completed",
                "live_node": "",
                "turn_response_type": str(item.get("turn_response_type") or ""),
                "saw_plan_node": bool(item.get("saw_plan_node")),
                "saw_navigation_activity": bool(item.get("saw_navigation_activity")),
            }
        )
    return history


def _checkpoint_payload_from_run_memory_snapshot(
    snapshot_path: Path,
    *,
    thread_id: str,
) -> dict[str, Any]:
    """Build a checkpoint payload from one persisted run-memory snapshot."""

    data = json.loads(snapshot_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Run-memory snapshot is not an object: {snapshot_path}")
    now = _now_display()
    history = _run_memory_turn_history(data, thread_id)
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "thread_id": thread_id,
        "saved_at": now,
        "updated_at": now,
        "input_language": "zh",
        "output_language": "zh",
        "turn_counter": len(history),
        "turn_history": history,
        "state_cache": _run_memory_state_for_checkpoint(data, thread_id),
        "run_memory_snapshot_path": str(snapshot_path),
        "latest_error": None,
    }


APP_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>CarAgent Plan Console</title>
  <style>
    :root {
      --bg: #f4efe7;
      --panel: rgba(255, 252, 247, 0.92);
      --panel-strong: #fffdf9;
      --line: #d7caba;
      --text: #231b14;
      --muted: #756657;
      --accent: #b95c37;
      --accent-deep: #8f3f20;
      --ok: #2f7d4d;
      --warn: #9c6a14;
      --wait: #376ea6;
      --fail: #a23535;
      --shadow: 0 18px 40px rgba(62, 41, 23, 0.12);
      --radius: 18px;
      --mono: "Consolas", "SFMono-Regular", monospace;
      --sans: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      font-family: var(--sans);
      color: var(--text);
      background:
        radial-gradient(circle at top left, rgba(255,255,255,0.7), transparent 28%),
        linear-gradient(160deg, #efe4d2 0%, #f5f0e9 45%, #e8dccf 100%);
      min-height: 100vh;
    }

    .shell {
      max-width: 1480px;
      margin: 0 auto;
      padding: 24px;
    }

    .hero {
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: flex-start;
      margin-bottom: 18px;
      padding: 22px 24px;
      border: 1px solid rgba(137, 103, 71, 0.18);
      border-radius: 24px;
      background: linear-gradient(135deg, rgba(255,255,255,0.78), rgba(246,235,221,0.92));
      box-shadow: var(--shadow);
    }

    .hero h1 {
      margin: 0;
      font-size: clamp(28px, 4vw, 40px);
      line-height: 1;
      letter-spacing: 0.02em;
    }

    .hero p {
      margin: 10px 0 0;
      color: var(--muted);
      max-width: 680px;
    }

    .meta-chip-row {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      justify-content: flex-end;
    }

    .meta-chip {
      padding: 10px 14px;
      border-radius: 999px;
      background: rgba(255,255,255,0.8);
      border: 1px solid rgba(137, 103, 71, 0.16);
      color: var(--muted);
      font-size: 13px;
    }

    .session-choice {
      display: none;
      gap: 12px;
      align-items: center;
      justify-content: space-between;
      padding: 12px;
      border-radius: 8px;
      border: 1px solid rgba(56, 116, 93, 0.18);
      background: rgba(244,255,249,0.9);
    }

    .session-choice-main {
      min-width: 0;
      display: grid;
      gap: 4px;
    }

    .session-choice-main strong {
      font-size: 13px;
      color: var(--ok);
    }

    .session-choice-main span {
      color: var(--muted);
      font-size: 12px;
      overflow-wrap: anywhere;
    }

    .session-resume-select {
      width: min(520px, 100%);
      max-width: 100%;
      margin-top: 4px;
      border-radius: 8px;
      border: 1px solid rgba(56, 116, 93, 0.18);
      background: rgba(255,255,255,0.9);
      color: var(--text);
      padding: 8px 10px;
      font-size: 13px;
    }

    .session-choice-actions {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      justify-content: flex-end;
    }

    .layout {
      display: grid;
      grid-template-columns: minmax(360px, 1.15fr) minmax(340px, 0.85fr);
      gap: 18px;
    }

    .panel {
      border-radius: var(--radius);
      border: 1px solid rgba(137, 103, 71, 0.16);
      background: var(--panel);
      box-shadow: var(--shadow);
      overflow: hidden;
      min-height: 0;
    }

    .panel-head {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: center;
      padding: 16px 18px;
      border-bottom: 1px solid rgba(137, 103, 71, 0.12);
      background: rgba(255,255,255,0.58);
    }

    .panel-head h2 {
      margin: 0;
      font-size: 17px;
    }

    .panel-head .sub {
      color: var(--muted);
      font-size: 13px;
    }

    .conversation {
      display: grid;
      grid-template-rows: auto 1fr auto;
      min-height: 78vh;
    }

    .summary-grid {
      display: grid;
      grid-template-columns: repeat(4, minmax(120px, 1fr));
      gap: 12px;
      padding: 16px 18px 0;
    }

    .summary-card {
      padding: 14px;
      border-radius: 16px;
      background: var(--panel-strong);
      border: 1px solid rgba(137, 103, 71, 0.12);
    }

    .summary-card .k {
      font-size: 12px;
      color: var(--muted);
      margin-bottom: 8px;
    }

    .summary-card .v {
      font-size: 18px;
      font-weight: 700;
      word-break: break-word;
    }

    .guide-panel {
      margin: 14px 18px 0;
      padding: 14px;
      border-radius: 14px;
      background: linear-gradient(135deg, rgba(242,250,246,0.96), rgba(255,255,255,0.86));
      border: 1px solid rgba(47, 125, 77, 0.18);
      display: grid;
      gap: 12px;
    }

    .guide-panel-head {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: center;
    }

    .guide-title {
      font-weight: 700;
      color: var(--ok);
    }

    .guide-status {
      color: var(--muted);
      font-size: 13px;
    }

    .guide-latest {
      min-height: 42px;
      padding: 12px;
      border-radius: 10px;
      background: rgba(255,255,255,0.82);
      border: 1px solid rgba(47, 125, 77, 0.12);
      font-size: 15px;
      line-height: 1.45;
    }

    .guide-actions {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }

    .guide-actions .btn {
      min-height: 40px;
    }

    .history {
      padding: 18px;
      overflow: auto;
      display: flex;
      flex-direction: column;
      gap: 14px;
    }

    .turn {
      padding: 14px 16px;
      border-radius: 18px;
      border: 1px solid rgba(137, 103, 71, 0.1);
      background: rgba(255,255,255,0.8);
    }

    .turn.user {
      background: linear-gradient(135deg, rgba(255,247,240,0.98), rgba(248,235,225,0.92));
    }

    .turn.system {
      background: linear-gradient(135deg, rgba(241,247,255,0.98), rgba(226,237,248,0.92));
    }

    .turn.running {
      border-style: dashed;
      border-color: rgba(185, 92, 55, 0.28);
    }

    .turn.failed {
      border-color: rgba(162, 53, 53, 0.22);
      background: linear-gradient(135deg, rgba(255,248,248,0.98), rgba(252,236,236,0.92));
    }

    .turn-head {
      display: flex;
      justify-content: space-between;
      gap: 10px;
      margin-bottom: 8px;
      font-size: 13px;
      color: var(--muted);
    }

    .turn-role {
      font-weight: 700;
      color: var(--text);
    }

    .turn-badges {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
    }

    .turn-badge {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 4px 10px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 700;
      background: rgba(35, 27, 20, 0.06);
      color: var(--muted);
    }

    .turn-badge.running { color: var(--accent-deep); }
    .turn-badge.completed { color: var(--ok); }
    .turn-badge.failed { color: var(--fail); }

    .turn-content {
      white-space: pre-wrap;
      line-height: 1.55;
    }

    .turn-answer {
      margin-top: 10px;
      padding-top: 10px;
      border-top: 1px dashed rgba(137, 103, 71, 0.18);
      color: var(--text);
      white-space: pre-wrap;
      line-height: 1.55;
    }

    .turn-answer strong {
      color: var(--accent-deep);
      display: block;
      margin-bottom: 6px;
    }

    .turn-answer + .turn-answer {
      margin-top: 8px;
      padding-top: 8px;
      border-top: 1px dashed rgba(137, 103, 71, 0.12);
    }

    .turn-trace {
      margin-top: 12px;
      padding: 12px;
      border-radius: 14px;
      background: rgba(255,255,255,0.68);
      border: 1px solid rgba(137, 103, 71, 0.1);
      display: flex;
      flex-direction: column;
      gap: 8px;
    }

    .turn-trace-title {
      font-size: 12px;
      font-weight: 700;
      letter-spacing: 0.04em;
      text-transform: uppercase;
      color: var(--muted);
    }

    .trace-row {
      display: grid;
      grid-template-columns: auto 1fr;
      gap: 10px;
      align-items: start;
    }

    .trace-node {
      padding: 4px 9px;
      border-radius: 999px;
      background: rgba(185, 92, 55, 0.1);
      color: var(--accent-deep);
      font-size: 11px;
      font-family: var(--mono);
      font-weight: 700;
      white-space: nowrap;
    }

    .trace-text {
      color: var(--muted);
      font-size: 13px;
      line-height: 1.45;
    }

    .turn-note {
      margin-top: 10px;
      padding: 10px 12px;
      border-radius: 12px;
      font-size: 13px;
      line-height: 1.45;
      background: rgba(255,255,255,0.66);
      border: 1px solid rgba(137, 103, 71, 0.1);
      color: var(--muted);
    }

    .turn-note.error {
      background: rgba(162, 53, 53, 0.08);
      border-color: rgba(162, 53, 53, 0.16);
      color: var(--fail);
    }

    .composer {
      padding: 16px 18px 18px;
      border-top: 1px solid rgba(137, 103, 71, 0.12);
      background: rgba(255,255,255,0.58);
    }

    .composer-top {
      display: grid;
      grid-template-columns: 140px repeat(4, auto);
      gap: 10px;
      align-items: center;
      margin-bottom: 10px;
    }

    .composer textarea,
    .composer select,
    .composer button {
      font: inherit;
    }

    .composer textarea,
    .composer select {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 14px;
      background: rgba(255,255,255,0.92);
      color: var(--text);
      padding: 12px 14px;
    }

    .composer textarea {
      min-height: 110px;
      resize: vertical;
      line-height: 1.5;
    }

    .btn {
      border: 0;
      border-radius: 14px;
      padding: 12px 16px;
      cursor: pointer;
      font-weight: 700;
      transition: transform 120ms ease, opacity 120ms ease;
    }

    .btn:hover { transform: translateY(-1px); }
    .btn:disabled { opacity: 0.65; cursor: wait; }

    .btn-primary {
      background: linear-gradient(135deg, var(--accent), var(--accent-deep));
      color: white;
    }

    .btn-secondary {
      background: rgba(255,255,255,0.9);
      border: 1px solid rgba(137, 103, 71, 0.16);
      color: var(--text);
    }

    .composer-note {
      display: flex;
      flex-direction: column;
      gap: 4px;
      margin-bottom: 10px;
      padding: 12px 14px;
      border-radius: 14px;
      border: 1px solid rgba(137, 103, 71, 0.12);
      background: rgba(255,255,255,0.84);
    }

    .composer-note strong {
      font-size: 13px;
    }

    .composer-note span {
      font-size: 13px;
      color: var(--muted);
      line-height: 1.45;
    }

    .composer-note.ready strong,
    .composer-note.active_plan strong {
      color: var(--ok);
    }

    .composer-note.waiting strong {
      color: var(--wait);
    }

    .composer-note.busy strong,
    .composer-note.settling strong {
      color: var(--accent-deep);
    }

    .io-toolbar {
      display: grid;
      grid-template-columns: repeat(5, minmax(0, auto));
      gap: 10px;
      align-items: center;
      margin-bottom: 10px;
    }

    .io-toolbar label {
      display: flex;
      gap: 8px;
      align-items: center;
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
    }

    .io-toolbar input[type="file"] {
      max-width: 210px;
      font-size: 12px;
    }

    .image-preview {
      display: none;
      gap: 12px;
      align-items: flex-start;
      margin-bottom: 10px;
      padding: 10px;
      border: 1px solid rgba(137, 103, 71, 0.12);
      border-radius: 14px;
      background: rgba(255,255,255,0.72);
    }

    .agent-capture-preview {
      display: none;
      gap: 12px;
      align-items: flex-start;
      margin-bottom: 10px;
      padding: 10px;
      border: 1px solid rgba(56, 116, 93, 0.16);
      border-radius: 14px;
      background: rgba(244,255,249,0.76);
    }

    .image-preview img {
      width: 132px;
      height: 88px;
      object-fit: cover;
      border-radius: 10px;
      border: 1px solid rgba(137, 103, 71, 0.16);
      background: white;
    }

    .agent-capture-preview img {
      width: 132px;
      height: 88px;
      object-fit: cover;
      border-radius: 10px;
      border: 1px solid rgba(56, 116, 93, 0.18);
      background: white;
    }

    .image-preview-text {
      flex: 1;
      min-width: 0;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.45;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
    }

    .image-preview-body {
      flex: 1;
      min-width: 0;
      display: flex;
      flex-direction: column;
      gap: 6px;
    }

    .image-preview-status {
      display: inline-flex;
      align-self: flex-start;
      padding: 4px 8px;
      border-radius: 999px;
      background: rgba(137, 103, 71, 0.1);
      color: var(--muted);
      font-size: 11px;
      font-weight: 800;
      text-transform: uppercase;
    }

    .image-preview-status.attached {
      background: rgba(56, 116, 93, 0.12);
      color: var(--ok);
    }

    .image-preview-actions {
      display: flex;
      flex-direction: column;
      gap: 8px;
      align-items: flex-end;
    }

    .image-preview-actions .mini-btn {
      white-space: nowrap;
    }

    .mini-btn.icon-btn {
      width: 30px;
      height: 30px;
      padding: 0;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      font-size: 16px;
      line-height: 1;
    }

    .side {
      display: grid;
      grid-template-rows: auto auto auto;
      gap: 18px;
      min-height: 78vh;
    }

    .status-strip {
      padding: 16px 18px;
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
    }

    .status-pill {
      padding: 8px 12px;
      border-radius: 999px;
      font-size: 13px;
      background: rgba(255,255,255,0.88);
      border: 1px solid rgba(137, 103, 71, 0.14);
      color: var(--muted);
    }

    .tasks {
      padding: 16px 18px 18px;
      overflow: auto;
      display: flex;
      flex-direction: column;
      gap: 12px;
    }

    .plan-toolbar {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      padding: 0 18px 4px;
    }

    .view-toggle {
      display: inline-flex;
      gap: 4px;
      padding: 4px;
      border-radius: 999px;
      border: 1px solid rgba(137, 103, 71, 0.16);
      background: rgba(255,255,255,0.72);
    }

    .view-toggle button {
      border: 0;
      border-radius: 999px;
      padding: 7px 12px;
      background: transparent;
      color: var(--muted);
      cursor: pointer;
      font-size: 12px;
      font-weight: 700;
    }

    .view-toggle button.active {
      background: var(--accent);
      color: #fffaf2;
      box-shadow: 0 6px 14px rgba(185, 92, 55, 0.18);
    }

    .plan-toolbar-note {
      color: var(--muted);
      font-size: 12px;
    }

    .plan-graph {
      display: none;
      padding: 16px 18px 18px;
      overflow: auto;
    }

    .graph-summary {
      display: grid;
      grid-template-columns: repeat(4, minmax(86px, 1fr));
      gap: 10px;
      margin-bottom: 14px;
    }

    .graph-chip {
      padding: 10px 12px;
      border-radius: 14px;
      border: 1px solid rgba(137, 103, 71, 0.12);
      background: rgba(255,255,255,0.82);
    }

    .graph-chip .k {
      color: var(--muted);
      font-size: 11px;
      margin-bottom: 5px;
    }

    .graph-chip .v {
      font-weight: 800;
      font-size: 15px;
    }

    .graph-section {
      margin-top: 14px;
    }

    .graph-section-title {
      margin: 0 0 8px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 800;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }

    .graph-node-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 10px;
    }

    .graph-canvas-wrap {
      border-radius: 18px;
      border: 1px solid rgba(137, 103, 71, 0.14);
      background:
        radial-gradient(circle at 18px 18px, rgba(137,103,71,0.08) 1px, transparent 1px),
        linear-gradient(135deg, rgba(255,255,255,0.9), rgba(248,239,228,0.76));
      background-size: 22px 22px, auto;
      overflow: auto;
      min-height: 280px;
      box-shadow: inset 0 1px 0 rgba(255,255,255,0.72);
    }

    .graph-svg {
      display: block;
      min-width: 100%;
    }

    .svg-edge {
      fill: none;
      stroke: rgba(117, 102, 87, 0.54);
      stroke-width: 2.2;
      cursor: pointer;
      transition: stroke 140ms ease, stroke-width 140ms ease;
    }

    .svg-edge.branch {
      stroke: rgba(55, 110, 166, 0.64);
      stroke-dasharray: 8 5;
    }

    .svg-edge:hover,
    .svg-edge.selected {
      stroke: var(--accent);
      stroke-width: 3.4;
    }

    .svg-edge-label {
      cursor: pointer;
    }

    .svg-edge-label rect {
      fill: rgba(255,255,255,0.88);
      stroke: rgba(137, 103, 71, 0.18);
    }

    .svg-edge-label text {
      font-size: 11px;
      fill: var(--muted);
      font-weight: 700;
    }

    .svg-edge-label:hover rect,
    .svg-edge-label.selected rect {
      fill: rgba(255,248,240,0.96);
      stroke: var(--accent);
    }

    .svg-node {
      cursor: pointer;
    }

    .svg-node rect {
      fill: rgba(255,255,255,0.92);
      stroke: rgba(137, 103, 71, 0.22);
      stroke-width: 1.3;
      filter: drop-shadow(0 8px 16px rgba(62, 41, 23, 0.10));
      transition: stroke 140ms ease, stroke-width 140ms ease, fill 140ms ease;
    }

    .svg-node.current rect {
      stroke: var(--accent);
      stroke-width: 2.4;
      fill: rgba(255,248,240,0.98);
    }

    .svg-node:hover rect,
    .svg-node.selected rect {
      stroke: var(--accent-deep);
      stroke-width: 2.6;
    }

    .svg-node-title {
      font-size: 13px;
      font-weight: 800;
      fill: var(--text);
    }

    .svg-node-sub {
      font-size: 11px;
      fill: var(--muted);
      font-weight: 700;
    }

    .graph-detail {
      margin-top: 12px;
      padding: 14px;
      border-radius: 16px;
      border: 1px solid rgba(137, 103, 71, 0.14);
      background: rgba(255,255,255,0.82);
    }

    .graph-detail-title {
      font-weight: 850;
      margin-bottom: 8px;
    }

    .graph-detail-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px 12px;
      color: var(--muted);
      font-size: 13px;
    }

    .graph-detail-grid strong {
      color: var(--text);
    }

    .graph-node {
      padding: 12px;
      border-radius: 16px;
      border: 1px solid rgba(137, 103, 71, 0.14);
      background: rgba(255,255,255,0.86);
    }

    .graph-node.current {
      border-color: rgba(185, 92, 55, 0.42);
      box-shadow: 0 0 0 3px rgba(185, 92, 55, 0.08);
    }

    .graph-node-title {
      font-weight: 800;
      line-height: 1.35;
      margin-bottom: 8px;
    }

    .edge-list,
    .issue-list {
      display: flex;
      flex-direction: column;
      gap: 8px;
    }

    .edge-row,
    .issue-row {
      display: grid;
      grid-template-columns: auto 1fr auto;
      align-items: center;
      gap: 10px;
      padding: 10px 12px;
      border-radius: 14px;
      border: 1px solid rgba(137, 103, 71, 0.12);
      background: rgba(255,255,255,0.78);
      font-size: 13px;
    }

    .edge-arrow {
      font-family: var(--mono);
      font-weight: 800;
      color: var(--accent-deep);
    }

    .issue-row.error {
      border-color: rgba(162, 53, 53, 0.22);
      background: rgba(255,248,248,0.9);
    }

    .issue-row.warning {
      border-color: rgba(156, 106, 20, 0.22);
      background: rgba(255,249,238,0.9);
    }

    .task {
      padding: 14px 16px;
      border-radius: 18px;
      background: rgba(255,255,255,0.86);
      border: 1px solid rgba(137, 103, 71, 0.14);
    }

    .task.current {
      border-color: rgba(185, 92, 55, 0.4);
      box-shadow: 0 0 0 3px rgba(185, 92, 55, 0.08);
    }

    .task-head {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: flex-start;
      margin-bottom: 8px;
    }

    .task-title {
      font-weight: 700;
      line-height: 1.45;
    }

    .task-meta {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-bottom: 10px;
    }

    .tag {
      border-radius: 999px;
      padding: 5px 10px;
      font-size: 12px;
      font-weight: 700;
      background: rgba(35, 27, 20, 0.06);
      color: var(--muted);
    }

    .tag.pending { color: var(--muted); }
    .tag.in_progress, .tag.running { color: var(--accent-deep); }
    .tag.waiting { color: var(--wait); }
    .tag.completed { color: var(--ok); }
    .tag.failed, .tag.cancelled { color: var(--fail); }

    .task-detail {
      color: var(--muted);
      font-size: 13px;
      line-height: 1.5;
      white-space: pre-wrap;
    }

    .side {
      display: grid;
      grid-template-rows: minmax(300px, 1fr) minmax(440px, 1.4fr);
      gap: 18px;
    }

    .logs {
      padding: 16px 18px 18px;
      overflow: auto;
      min-height: 440px;
      max-height: 58vh;
      background: rgba(255,255,255,0.52);
    }

    .console-toolbar {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      padding: 0 18px 12px;
      color: var(--muted);
      font-size: 12px;
    }

    .console-actions {
      display: flex;
      align-items: center;
      gap: 10px;
    }

    .console-indicator {
      font-family: var(--mono);
      font-size: 11px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: var(--accent-deep);
    }

    .mini-btn {
      border: 1px solid rgba(137, 103, 71, 0.18);
      background: rgba(255,255,255,0.76);
      color: var(--ink);
      border-radius: 999px;
      padding: 6px 12px;
      cursor: pointer;
      font-size: 12px;
      transition: transform 140ms ease, background 140ms ease;
    }

    .mini-btn:hover {
      transform: translateY(-1px);
      background: rgba(255,255,255,0.92);
    }

    .console-pre {
      margin: 0;
      min-height: 220px;
      padding: 12px 14px;
      border-radius: 14px;
      border: 1px solid rgba(137, 103, 71, 0.12);
      background: rgba(34, 28, 24, 0.96);
      color: #f6ead9;
      white-space: pre-wrap;
      word-break: break-word;
      font-size: 12px;
      line-height: 1.5;
      font-family: var(--mono);
      box-shadow: inset 0 1px 0 rgba(255,255,255,0.04);
    }

    .log-line {
      display: block;
      white-space: pre-wrap;
      margin-bottom: 6px;
    }

    .log-line.log-dim { color: #c8b8a4; }
    .log-line.log-ingest { color: #e4b15f; }
    .log-line.log-orchestrate { color: #89d1ff; }
    .log-line.log-plan { color: #8de0b5; }
    .log-line.log-execute { color: #ffd27d; }
    .log-line.log-tool { color: #d8a9ff; }
    .log-line.log-response { color: #f2a6a6; }

    .log-line.log-warning {
      color: #ffb86e;
      font-weight: 600;
    }

    .log-line.log-error {
      color: #ff8f8f;
      font-weight: 600;
    }

    .empty {
      padding: 18px;
      color: var(--muted);
      text-align: center;
    }

    .error-banner {
      display: none;
      margin: 0 18px 14px;
      padding: 12px 14px;
      border-radius: 14px;
      background: rgba(162, 53, 53, 0.08);
      border: 1px solid rgba(162, 53, 53, 0.18);
      color: var(--fail);
      white-space: pre-wrap;
    }

    @media (max-width: 1120px) {
      .layout {
        grid-template-columns: 1fr;
      }

      .conversation,
      .side {
        min-height: auto;
      }
    }

    @media (max-width: 720px) {
      .shell { padding: 14px; }
      .hero { padding: 18px; }
      .summary-grid { grid-template-columns: repeat(2, minmax(120px, 1fr)); }
      .composer-top { grid-template-columns: 1fr 1fr; }
      .guide-panel-head { align-items: flex-start; flex-direction: column; }
      .guide-actions { display: grid; grid-template-columns: 1fr; }
      .session-choice { align-items: stretch; flex-direction: column; }
      .session-choice-actions { justify-content: stretch; }
      .session-choice-actions .btn { flex: 1; }
    }
  </style>
</head>
<body>
  <div class="shell">
    <section class="hero">
      <div>
        <h1>CarAgent 导引控制台</h1>
        <p>在左侧发送指令，右侧查看实时计划、任务进度和运行日志。</p>
      </div>
      <div class="meta-chip-row">
        <div class="meta-chip">会话: <span id="thread-id">-</span></div>
        <div class="meta-chip">更新: <span id="updated-at">-</span></div>
        <div class="meta-chip">记忆: <span id="checkpoint-status">-</span></div>
      </div>
    </section>

    <div class="layout">
      <section class="panel conversation">
        <div>
          <div class="panel-head">
            <div>
              <h2>对话</h2>
              <div class="sub">用户指令、控制器事件和 Agent 回复会显示在这里。</div>
            </div>
          </div>
          <div class="summary-grid">
            <div class="summary-card"><div class="k">当前计划</div><div class="v" id="summary-plan">-</div></div>
            <div class="summary-card"><div class="k">当前任务</div><div class="v" id="summary-task">-</div></div>
            <div class="summary-card"><div class="k">下一动作</div><div class="v" id="summary-next-action">idle</div></div>
            <div class="summary-card"><div class="k">任务数量</div><div class="v" id="summary-task-count">0</div></div>
          </div>
          <div class="guide-panel" aria-live="polite">
            <div class="guide-panel-head">
              <div>
                <div class="guide-title">语音播报</div>
                <div class="guide-status" id="guide-status">语音会按配置自动启用，也可以手动静音。</div>
              </div>
              <div class="guide-actions">
                <button class="btn btn-primary" type="button" id="guide-enable-btn">启用语音</button>
                <button class="btn btn-secondary" type="button" id="guide-mute-btn">静音</button>
                <button class="btn btn-secondary" type="button" id="guide-replay-btn">重播</button>
              </div>
            </div>
            <div class="guide-latest" id="guide-latest">暂无播报内容。</div>
          </div>
          <div class="session-choice" id="session-choice">
            <div class="session-choice-main">
              <strong>发现已保存会话</strong>
              <span id="session-choice-detail">请选择继续旧会话还是开启新会话。</span>
              <select class="session-resume-select" id="session-resume-select"></select>
            </div>
            <div class="session-choice-actions">
              <button class="btn btn-primary" type="button" id="resume-session-btn">继续会话</button>
              <button class="btn btn-secondary" type="button" id="new-session-btn">新会话</button>
            </div>
          </div>
        </div>

        <div class="error-banner" id="error-banner"></div>
        <div class="history" id="history"></div>

        <form class="composer" id="composer">
          <div class="composer-top">
            <div class="meta-chip">发送用户指令</div>
            <button class="btn btn-secondary" type="button" id="clear-btn">清空</button>
            <button class="btn btn-secondary" type="button" id="refresh-btn">刷新</button>
            <button class="btn btn-primary" type="submit" id="send-btn">发送</button>
          </div>
          <div class="composer-note ready" id="composer-note">
            <strong>可以输入</strong>
            <span>发送新的指令即可，控制器到达事件会自动处理。</span>
          </div>
          <div class="io-toolbar">
            <label>输入
              <select id="input-language">
                <option value="zh">中文</option>
                <option value="en">English</option>
              </select>
            </label>
            <label>回复
              <select id="output-language">
                <option value="zh">中文</option>
                <option value="en">English</option>
              </select>
            </label>
            <button class="btn btn-secondary" type="button" id="capture-btn">拍照</button>
            <input type="file" id="image-upload" accept="image/*">
            <button class="btn btn-secondary" type="button" id="describe-image-btn">描述图片</button>
          </div>
          <div class="image-preview" id="image-preview">
            <img id="image-preview-img" alt="Selected or captured frame">
            <div class="image-preview-body">
              <div class="image-preview-status" id="image-preview-status">preview only</div>
              <div class="image-preview-text" id="image-preview-text"></div>
            </div>
            <div class="image-preview-actions">
              <button class="mini-btn icon-btn" type="button" id="clear-image-btn" title="Clear image preview">×</button>
              <button class="mini-btn" type="button" id="attach-image-btn" disabled>附加</button>
              <button class="mini-btn" type="button" id="download-image-btn" disabled>下载</button>
            </div>
          </div>
          <div class="agent-capture-preview" id="agent-capture-preview">
            <img id="agent-capture-img" alt="Latest agent capture">
            <div class="image-preview-body">
              <div class="image-preview-status">agent capture</div>
              <div class="image-preview-text" id="agent-capture-text"></div>
            </div>
            <div class="image-preview-actions">
              <button class="mini-btn" type="button" id="use-agent-capture-btn" disabled>作为输入</button>
              <button class="mini-btn" type="button" id="download-agent-capture-btn" disabled>下载</button>
            </div>
          </div>
          <textarea id="message-input" placeholder="输入用户指令。控制器到达事件会自动处理。"></textarea>
        </form>
      </section>

      <section class="side">
        <div class="panel">
          <div class="panel-head">
            <div>
              <h2>计划快照</h2>
              <div class="sub">当前计划、活动任务、等待状态和任务顺序。</div>
            </div>
          </div>
          <div class="status-strip" id="status-strip"></div>
          <div class="plan-toolbar">
            <div class="view-toggle" aria-label="Plan view mode">
              <button class="active" type="button" id="plan-view-list">列表</button>
              <button type="button" id="plan-view-graph">图</button>
            </div>
            <div class="plan-toolbar-note">图模式只用于查看任务依赖。</div>
          </div>
          <div class="tasks" id="tasks"></div>
          <div class="plan-graph" id="plan-graph"></div>
        </div>

        <div class="panel">
          <div class="panel-head">
            <div>
              <h2>实时日志</h2>
              <div class="sub">汇总工作流、工具和控制器日志。</div>
            </div>
          </div>
          <div class="console-toolbar">
            <div>滚动条靠近底部时会自动跟随最新日志。</div>
            <div class="console-actions">
              <span class="console-indicator" id="console-follow-indicator">跟随最新</span>
              <button class="mini-btn" type="button" id="console-jump-btn">跳到最新</button>
            </div>
          </div>
          <div class="logs" id="logs"></div>
        </div>
      </section>
    </div>
  </div>

  <script>
    const historyEl = document.getElementById("history");
    const tasksEl = document.getElementById("tasks");
    const planGraphEl = document.getElementById("plan-graph");
    const planViewListBtnEl = document.getElementById("plan-view-list");
    const planViewGraphBtnEl = document.getElementById("plan-view-graph");
    const logsEl = document.getElementById("logs");
    const statusStripEl = document.getElementById("status-strip");
    const errorBannerEl = document.getElementById("error-banner");
    const sessionChoiceEl = document.getElementById("session-choice");
    const sessionChoiceDetailEl = document.getElementById("session-choice-detail");
    const sessionResumeSelectEl = document.getElementById("session-resume-select");
    const resumeSessionBtnEl = document.getElementById("resume-session-btn");
    const newSessionBtnEl = document.getElementById("new-session-btn");
    const formEl = document.getElementById("composer");
    const messageInputEl = document.getElementById("message-input");
    const sendBtnEl = document.getElementById("send-btn");
    const clearBtnEl = document.getElementById("clear-btn");
    const refreshBtnEl = document.getElementById("refresh-btn");
    const composerNoteEl = document.getElementById("composer-note");
    const consoleJumpBtnEl = document.getElementById("console-jump-btn");
    const consoleFollowIndicatorEl = document.getElementById("console-follow-indicator");
    const inputLanguageEl = document.getElementById("input-language");
    const outputLanguageEl = document.getElementById("output-language");
    const captureBtnEl = document.getElementById("capture-btn");
    const imageUploadEl = document.getElementById("image-upload");
    const describeImageBtnEl = document.getElementById("describe-image-btn");
    const imagePreviewEl = document.getElementById("image-preview");
    const imagePreviewImgEl = document.getElementById("image-preview-img");
    const imagePreviewStatusEl = document.getElementById("image-preview-status");
    const imagePreviewTextEl = document.getElementById("image-preview-text");
    const attachImageBtnEl = document.getElementById("attach-image-btn");
    const clearImageBtnEl = document.getElementById("clear-image-btn");
    const downloadImageBtnEl = document.getElementById("download-image-btn");
    const agentCapturePreviewEl = document.getElementById("agent-capture-preview");
    const agentCaptureImgEl = document.getElementById("agent-capture-img");
    const agentCaptureTextEl = document.getElementById("agent-capture-text");
    const useAgentCaptureBtnEl = document.getElementById("use-agent-capture-btn");
    const downloadAgentCaptureBtnEl = document.getElementById("download-agent-capture-btn");
    const guideStatusEl = document.getElementById("guide-status");
    const guideLatestEl = document.getElementById("guide-latest");
    const guideEnableBtnEl = document.getElementById("guide-enable-btn");
    const guideMuteBtnEl = document.getElementById("guide-mute-btn");
    const guideReplayBtnEl = document.getElementById("guide-replay-btn");
    let refreshTimer = null;
    let refreshInFlight = false;
    let logAutoFollow = true;
    let historyAutoFollow = true;
    let planViewMode = "list";
    let selectedGraphItem = null;
    let selectedImageDataUrl = "";
    let previewImageDataUrl = "";
    let selectedImageDownloadName = "caragent_capture.jpg";
    let previewImageDownloadName = "caragent_capture.jpg";
    let latestAgentCapture = null;
    let languageInitialized = false;
    let sessionChoiceRequired = false;
    let guideVoiceEnabled = false;
    let guideMuted = false;
    let guideLastEvent = null;
    let guideLastSpokenText = "";
    let guideInitializedFromProfile = false;
    let guidanceHistoryInitialized = false;
    let responseSpeechHistoryInitialized = false;
    const spokenGuidanceIds = new Set();
    const spokenResponseIds = new Set();

    function escapeHtml(text) {
      const normalized = text === null || text === undefined ? "" : String(text);
      return normalized
        .split("&").join("&amp;")
        .split("<").join("&lt;")
        .split(">").join("&gt;");
    }

    function formatJsonInline(value) {
      if (value === null || value === undefined || value === "") {
        return "-";
      }
      if (typeof value === "string") {
        return value;
      }
      return JSON.stringify(value);
    }

    function isNearBottom(element, threshold = 24) {
      if (!element) {
        return true;
      }
      return element.scrollHeight - element.scrollTop - element.clientHeight <= threshold;
    }

    function updateConsoleFollowIndicator() {
      consoleFollowIndicatorEl.textContent = logAutoFollow ? "跟随最新" : "滚动锁定";
    }

    function scheduleRefresh(delayMs) {
      if (refreshTimer) {
        window.clearTimeout(refreshTimer);
      }
      refreshTimer = window.setTimeout(() => {
        refreshState();
      }, delayMs);
    }

    async function request(path, options = {}) {
      const response = await fetch(path, {
        headers: { "Content-Type": "application/json" },
        ...options,
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.error || "Request failed");
      }
      return data;
    }

    function speechSupported() {
      return "speechSynthesis" in window && "SpeechSynthesisUtterance" in window;
    }

    function updateGuideControls() {
      if (!speechSupported()) {
        guideStatusEl.textContent = "当前浏览器不支持语音合成，回复仍会以文字显示。";
        guideEnableBtnEl.disabled = true;
        guideMuteBtnEl.disabled = true;
        guideReplayBtnEl.disabled = !guideLastEvent;
        return;
      }
      if (!guideVoiceEnabled) {
        guideStatusEl.textContent = "语音当前关闭。需要播报时请点击启用语音。";
      } else if (guideMuted) {
        guideStatusEl.textContent = "语音已静音，回复仍会以文字显示。";
      } else {
        guideStatusEl.textContent = "语音已启用。Agent 回复和关键导引信息会播报。";
      }
      guideEnableBtnEl.textContent = guideVoiceEnabled ? "测试语音" : "启用语音";
      guideEnableBtnEl.disabled = false;
      guideMuteBtnEl.textContent = guideMuted ? "取消静音" : "静音";
      guideReplayBtnEl.disabled = !guideLastEvent;
    }

    function initializeGuideProfile(state) {
      if (guideInitializedFromProfile) {
        return;
      }
      const profile = state && state.interaction_profile ? state.interaction_profile : {};
      guideVoiceEnabled = Boolean(profile.voice_enabled_default);
      guideMuted = false;
      guideInitializedFromProfile = true;
      updateGuideControls();
    }

    function speakGuidance(text, options = {}) {
      const cleanText = String(text || "").trim();
      if (!cleanText || !speechSupported() || !guideVoiceEnabled || guideMuted) {
        return;
      }
      if (options.interrupt) {
        window.speechSynthesis.cancel();
      }
      const utterance = new SpeechSynthesisUtterance(cleanText);
      utterance.lang = "zh-CN";
      utterance.rate = 0.95;
      utterance.pitch = 1.0;
      window.speechSynthesis.speak(utterance);
    }

    function renderGuidance(guidanceEvents) {
      const events = Array.isArray(guidanceEvents) ? guidanceEvents : [];
      const profile = (window.__carAgentLatestState || {}).interaction_profile || {};
      const speakGuidanceEvents = profile.speak_guidance_events === true;
      const requiredSpeechEvents = new Set([
        "request_received",
        "plan_created",
        "plan_updated",
        "navigation_start",
        "arrival",
        "arrival_verification",
        "failed",
        "cancelled",
        "stuck",
      ]);
      if (!events.length) {
        guideLatestEl.textContent = "暂无播报内容。";
        guidanceHistoryInitialized = true;
        updateGuideControls();
        return;
      }
      const latest = events[events.length - 1];
      guideLastEvent = latest;
      guideLatestEl.textContent = latest.text || "暂无播报内容。";
      const newEvents = events.filter((event) => {
        const eventId = String(event.event_id || "");
        return eventId && !spokenGuidanceIds.has(eventId);
      });
      newEvents.forEach((event) => {
        spokenGuidanceIds.add(String(event.event_id || ""));
        const eventType = String(event.event_type || "");
        const mustSpeak = requiredSpeechEvents.has(eventType);
        if (!guidanceHistoryInitialized || (!speakGuidanceEvents && !mustSpeak)) {
          return;
        }
        const priority = String(event.priority || "normal");
        speakGuidance(event.text || "", {
          interrupt: Boolean(event.interrupt) || priority === "high" || priority === "critical",
        });
      });
      guidanceHistoryInitialized = true;
      updateGuideControls();
    }

    function visibleResponseItemsForSpeech(turns) {
      const items = [];
      const profile = (window.__carAgentLatestState || {}).interaction_profile || {};
      if (profile.speak_agent_replies === false) {
        return items;
      }
      (Array.isArray(turns) ? turns : []).forEach((turn) => {
        if (turn.status && turn.status !== "completed") {
          return;
        }
        const responseItems = Array.isArray(turn.response_items) ? turn.response_items : [];
        responseItems.forEach((item, index) => {
          if (String(item.response_type || "") === "progress") {
            return;
          }
          const text = String(item.response_text || "").trim();
          if (!text) {
            return;
          }
          const responseId = String(
            item.response_id
            || [turn.turn_id, item.response_type || "reply", index, text].join(":")
          );
          items.push({
            id: responseId,
            text,
            type: String(item.response_type || "result"),
          });
        });
      });
      return items;
    }

    function speakNewResponseItems(turns) {
      const items = visibleResponseItemsForSpeech(turns);
      items.forEach((item) => {
        if (spokenResponseIds.has(item.id)) {
          return;
        }
        spokenResponseIds.add(item.id);
        guideLastSpokenText = item.text;
        guideLastEvent = {
          event_id: item.id,
          event_type: "agent_reply",
          text: item.text,
          priority: item.type === "error" ? "high" : "normal",
        };
        guideLatestEl.textContent = item.text;
        if (!responseSpeechHistoryInitialized) {
          return;
        }
        speakGuidance(item.text, {
          interrupt: item.type === "error",
        });
      });
      responseSpeechHistoryInitialized = true;
    }

    function renderStepTrace(stepTrace) {
      const trace = Array.isArray(stepTrace) ? stepTrace.slice(-6) : [];
      if (!trace.length) {
        return "";
      }

      const rows = trace.map((step) => {
        const parts = [];
        if (step.latest_event && step.latest_event.summary) {
          parts.push(step.latest_event.summary);
        }
        if (step.focus_task && step.focus_task.description) {
          parts.push(step.focus_task.description);
        }
        if (!parts.length && step.current_plan_id) {
          parts.push("plan " + step.current_plan_id + " updated");
        }
        const traceText = parts.length ? parts.join(" | ") : "State updated";
        return `
          <div class="trace-row">
            <div class="trace-node">${escapeHtml(step.node || "node")}</div>
            <div class="trace-text">${escapeHtml(traceText)}</div>
          </div>
        `;
      }).join("");

      return `
        <div class="turn-trace">
          <div class="turn-trace-title">Workflow</div>
          ${rows}
        </div>
      `;
    }

    function renderHistory(turns) {
      if (!turns.length) {
        historyEl.innerHTML = '<div class="empty">暂无消息。</div>';
        responseSpeechHistoryInitialized = true;
        return;
      }
      const shouldStick = historyAutoFollow || isNearBottom(historyEl, 36);
      historyEl.innerHTML = turns.map((turn) => renderTurn(turn)).join("");
      speakNewResponseItems(turns);
      if (shouldStick) {
        historyEl.scrollTop = historyEl.scrollHeight;
      }
    }

    function isSystemStatusTurn(turn) {
      const source = String(turn.source || "");
      return source === "controller" || source === "controller-watchdog";
    }

    function renderTurn(turn) {
      const status = turn.status || "completed";
      const statusLabel = {
        running: "Running",
        completed: "Completed",
        failed: "Failed",
      }[status] || status;
      const liveNodeBadge = turn.live_node && status === "running"
        ? '<span class="turn-badge running">' + escapeHtml(turn.live_node) + "</span>"
        : "";
      const systemStatusBadge = turn.display_kind === "system_status"
        ? '<span class="turn-badge">status update</span>'
        : "";
      const imageBadge = turn.image_attached
        ? '<span class="turn-badge">image attached</span>'
        : "";
      const responseItems = Array.isArray(turn.response_items) ? turn.response_items : [];
      const legacyResponse = turn.response && !responseItems.length
        ? [{ response_text: turn.response, response_type: turn.response_type || "result" }]
        : responseItems;
      const responseHtml = legacyResponse.map((item, index) => {
        const labelMap = {
          result: "CarAgent 回复",
          progress: "CarAgent 进展",
          error: "CarAgent 提醒",
        };
        const itemType = item.response_type || "result";
        const itemLabel = labelMap[itemType] || "CarAgent 回复";
        const suffix = legacyResponse.length > 1 ? " #" + (index + 1) : "";
        return '<div class="turn-answer"><strong>' + escapeHtml(itemLabel + suffix) + '</strong>' + escapeHtml(item.response_text || "") + "</div>";
      }).join("");
      const traceHtml = renderStepTrace(turn.step_trace || []);
      const noteHtml = !legacyResponse.length && status === "running"
        ? '<div class="turn-note">任务仍在执行中，页面会自动刷新进展。</div>'
        : "";
      const errorHtml = turn.error
        ? '<div class="turn-note error">' + escapeHtml(turn.error) + "</div>"
        : "";
      const agentMessageHtml = turn.agent_message
        ? '<div class="turn-note"><strong>Agent 输入</strong><span>' + escapeHtml(turn.agent_message) + "</span></div>"
        : "";
      const traceHtmlForTurn = isSystemStatusTurn(turn) ? "" : traceHtml;

      return [
        '<article class="turn ' + escapeHtml(turn.role) + ' ' + escapeHtml(status) + '">',
        '  <div class="turn-head">',
        '    <div class="turn-badges"><span class="turn-role">' + escapeHtml(turn.role_label) + '</span><span class="turn-badge">' + escapeHtml(turn.source) + '</span><span class="turn-badge ' + escapeHtml(status) + '">' + escapeHtml(statusLabel) + "</span>" + liveNodeBadge + systemStatusBadge + imageBadge + "</div>",
        '    <div>' + escapeHtml(turn.created_at) + "</div>",
        "  </div>",
        '  <div class="turn-content">' + escapeHtml(turn.content) + "</div>",
        agentMessageHtml,
        traceHtmlForTurn,
        responseHtml,
        noteHtml,
        errorHtml,
        "</article>",
      ].join("");
    }

    function renderTasks(tasks) {
      if (!tasks.length) {
        tasksEl.innerHTML = '<div class="empty">No active tasks.</div>';
        return;
      }

      tasksEl.innerHTML = tasks.map((task) => `
        <article class="task ${task.is_current ? "current" : ""}">
          <div class="task-head">
            <div class="task-title">${escapeHtml(task.title)}</div>
            <div class="tag ${escapeHtml(task.status)}">${escapeHtml(task.status)}</div>
          </div>
          <div class="task-meta">
            ${task.sequence_label ? `<span class="tag">${escapeHtml(task.sequence_label)}</span>` : ""}
            <span class="tag">${escapeHtml(task.type)}</span>
            ${task.wait_for_event ? `<span class="tag">wait: ${escapeHtml(task.wait_for_event)}</span>` : ""}
            ${task.is_inserted ? `<span class="tag">inserted</span>` : ""}
          </div>
          <div class="task-detail">${escapeHtml(task.detail)}</div>
        </article>
      `).join("");
    }

    function applyPlanViewMode() {
      const showGraph = planViewMode === "graph";
      tasksEl.style.display = showGraph ? "none" : "flex";
      planGraphEl.style.display = showGraph ? "block" : "none";
      planViewListBtnEl.classList.toggle("active", !showGraph);
      planViewGraphBtnEl.classList.toggle("active", showGraph);
    }

    function formatTaskRef(taskId) {
      return taskId === null || taskId === undefined ? "-" : "#" + escapeHtml(taskId);
    }

    function truncateText(text, maxLength) {
      const normalized = String(text || "").replace(/\s+/g, " ").trim();
      if (normalized.length <= maxLength) {
        return normalized;
      }
      return normalized.slice(0, Math.max(0, maxLength - 3)).trimEnd() + "...";
    }

    function buildGraphLayout(nodes, edges) {
      const nodeMap = new Map(nodes.map((node) => [Number(node.task_id), node]));
      const validEdges = edges.filter((edge) => nodeMap.has(Number(edge.source)) && nodeMap.has(Number(edge.target)));
      const indegree = new Map(nodes.map((node) => [Number(node.task_id), 0]));
      validEdges.forEach((edge) => {
        const target = Number(edge.target);
        indegree.set(target, (indegree.get(target) || 0) + 1);
      });

      const levels = new Map(nodes.map((node) => [Number(node.task_id), 0]));
      const roots = nodes.filter((node) => node.is_root || (indegree.get(Number(node.task_id)) || 0) === 0);
      roots.forEach((node) => levels.set(Number(node.task_id), 0));

      for (let pass = 0; pass < nodes.length; pass += 1) {
        let changed = false;
        validEdges.forEach((edge) => {
          const source = Number(edge.source);
          const target = Number(edge.target);
          const nextLevel = (levels.get(source) || 0) + 1;
          if (nextLevel > (levels.get(target) || 0)) {
            levels.set(target, nextLevel);
            changed = true;
          }
        });
        if (!changed) break;
      }

      const grouped = new Map();
      nodes.forEach((node) => {
        const level = levels.get(Number(node.task_id)) || 0;
        if (!grouped.has(level)) grouped.set(level, []);
        grouped.get(level).push(node);
      });
      grouped.forEach((items) => items.sort((a, b) => Number(a.task_id) - Number(b.task_id)));

      const nodeWidth = 210;
      const nodeHeight = 82;
      const columnGap = 238;
      const rowGap = 132;
      const marginX = 42;
      const marginY = 34;
      const positioned = new Map();
      const maxLevel = Math.max(0, ...Array.from(grouped.keys()));
      let maxRows = 1;

      grouped.forEach((items, level) => {
        maxRows = Math.max(maxRows, items.length);
        items.forEach((node, index) => {
          positioned.set(Number(node.task_id), {
            ...node,
            x: marginX + index * columnGap,
            y: marginY + level * rowGap,
            width: nodeWidth,
            height: nodeHeight,
          });
        });
      });

      return {
        nodes: Array.from(positioned.values()),
        edges: validEdges,
        nodeMap: positioned,
        width: marginX * 2 + nodeWidth + (maxRows - 1) * columnGap,
        height: marginY * 2 + nodeHeight + maxLevel * rowGap,
      };
    }

    function renderGraphDetail(type, item) {
      const detailEl = document.getElementById("graph-detail");
      if (!detailEl || !item) {
        return;
      }

      if (type === "edge") {
        detailEl.innerHTML = `
          <div class="graph-detail-title">Edge ${formatTaskRef(item.source)} -&gt; ${formatTaskRef(item.target)}</div>
          <div class="graph-detail-grid">
            <div><strong>kind</strong><br>${escapeHtml(item.kind || "edge")}</div>
            <div><strong>branch label</strong><br>${escapeHtml(item.label || "-")}</div>
            <div><strong>source</strong><br>${formatTaskRef(item.source)}</div>
            <div><strong>target</strong><br>${formatTaskRef(item.target)}</div>
          </div>
        `;
        return;
      }

      detailEl.innerHTML = `
        <div class="graph-detail-title">${formatTaskRef(item.task_id)} ${escapeHtml(item.description || "")}</div>
        <div class="graph-detail-grid">
          <div><strong>status</strong><br>${escapeHtml(item.status || "pending")}</div>
          <div><strong>type</strong><br>${escapeHtml(item.type || "action")}</div>
          <div><strong>plan_id</strong><br>${escapeHtml(item.plan_id || "-")}</div>
          <div><strong>next_task_id</strong><br>${formatJsonInline(item.next_task_id)}</div>
          <div><strong>depends_on</strong><br>${escapeHtml(formatJsonInline(item.depends_on || []))}</div>
          <div><strong>branches</strong><br>${escapeHtml(formatJsonInline(item.branches || {}))}</div>
          <div><strong>wait_for_event</strong><br>${escapeHtml(item.wait_for_event || "-")}</div>
          <div><strong>latest_result</strong><br>${escapeHtml(item.latest_result_summary || "-")}</div>
        </div>
      `;
    }

    function selectGraphItem(type, item, element) {
      selectedGraphItem = { type, item };
      planGraphEl.querySelectorAll(".selected").forEach((selectedEl) => selectedEl.classList.remove("selected"));
      if (element) {
        element.classList.add("selected");
        const edgeIndex = element.getAttribute("data-edge-index");
        if (edgeIndex !== null) {
          planGraphEl.querySelectorAll(`[data-edge-index="${edgeIndex}"]`).forEach((edgeEl) => edgeEl.classList.add("selected"));
        }
      }
      renderGraphDetail(type, item);
    }

    function renderPlanGraph(planGraph) {
      const summary = (planGraph || {}).summary || {};
      const nodes = Array.isArray((planGraph || {}).nodes) ? planGraph.nodes : [];
      const edges = Array.isArray((planGraph || {}).edges) ? planGraph.edges : [];
      const issues = Array.isArray((planGraph || {}).issues) ? planGraph.issues : [];
      const previousCanvas = planGraphEl.querySelector(".graph-canvas-wrap");
      const previousScrollLeft = previousCanvas ? previousCanvas.scrollLeft : 0;
      const previousScrollTop = previousCanvas ? previousCanvas.scrollTop : 0;
      const previousSelection = selectedGraphItem;

      if (!nodes.length) {
        planGraphEl.innerHTML = '<div class="empty">No active plan graph.</div>';
        return;
      }

      const summaryHtml = [
        ["Nodes", summary.node_count || nodes.length || 0],
        ["Edges", summary.edge_count || edges.length || 0],
        ["DAG", summary.is_dag === false ? "no" : "yes"],
        ["Issues", issues.length],
      ].map(([label, value]) => `
        <div class="graph-chip">
          <div class="k">${escapeHtml(label)}</div>
          <div class="v">${escapeHtml(value)}</div>
        </div>
      `).join("");

      const layout = buildGraphLayout(nodes, edges);
      const markerId = "plan-arrow-" + Math.random().toString(36).slice(2);
      const edgeSvg = layout.edges.map((edge, index) => {
        const source = layout.nodeMap.get(Number(edge.source));
        const target = layout.nodeMap.get(Number(edge.target));
        if (!source || !target) return "";
        const x1 = source.x + source.width / 2;
        const y1 = source.y + source.height;
        const x2 = target.x + target.width / 2;
        const y2 = target.y;
        const bend = Math.max(48, Math.abs(y2 - y1) / 2);
        const path = `M ${x1} ${y1} C ${x1} ${y1 + bend}, ${x2} ${y2 - bend}, ${x2} ${y2}`;
        const label = edge.label || edge.kind || "";
        const labelX = (x1 + x2) / 2 - 38;
        const labelY = (y1 + y2) / 2 - 11;
        return `
          <path class="svg-edge ${escapeHtml(edge.kind || "sequence")}" data-edge-index="${index}" d="${path}" marker-end="url(#${markerId})"></path>
          ${label ? `
            <g class="svg-edge-label" data-edge-index="${index}" transform="translate(${labelX}, ${labelY})">
              <rect rx="10" ry="10" width="76" height="22"></rect>
              <text x="38" y="15" text-anchor="middle">${escapeHtml(truncateText(label, 12))}</text>
            </g>
          ` : ""}
        `;
      }).join("");

      const nodeSvg = layout.nodes.map((node) => `
        <g class="svg-node ${node.is_current ? "current" : ""}" data-node-id="${escapeHtml(node.task_id)}" transform="translate(${node.x}, ${node.y})">
          <rect rx="16" ry="16" width="${node.width}" height="${node.height}"></rect>
          <text class="svg-node-title" x="16" y="25">${formatTaskRef(node.task_id)} ${escapeHtml(truncateText(node.description || "", 22))}</text>
          <text class="svg-node-sub" x="16" y="49">${escapeHtml(node.status || "pending")} | ${escapeHtml(node.type || "action")}</text>
          <text class="svg-node-sub" x="16" y="67">${node.is_root ? "root" : ""}${node.is_root && node.is_leaf ? " | " : ""}${node.is_leaf ? "leaf" : ""}</text>
        </g>
      `).join("");

      const topologyHtml = `
        <div class="graph-canvas-wrap" id="graph-canvas-wrap">
          <svg class="graph-svg" width="${layout.width}" height="${layout.height}" viewBox="0 0 ${layout.width} ${layout.height}" role="img" aria-label="Plan topology graph">
            <defs>
              <marker id="${markerId}" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth">
                <path d="M0,0 L0,6 L8,3 z" fill="rgba(117, 102, 87, 0.74)"></path>
              </marker>
            </defs>
            ${edgeSvg}
            ${nodeSvg}
          </svg>
        </div>
      `;

      const issuesHtml = issues.length
        ? issues.map((issue) => `
          <div class="issue-row ${escapeHtml(issue.severity || "warning")}">
            <span class="tag ${escapeHtml(issue.severity || "warning")}">${escapeHtml(issue.severity || "warning")}</span>
            <span>${escapeHtml(issue.message || issue.code || "Plan graph issue")}</span>
            <span class="tag">${escapeHtml(issue.code || "")}</span>
          </div>
        `).join("")
        : '<div class="empty">No PlanGraph validation issues.</div>';

      planGraphEl.innerHTML = `
        <div class="graph-summary">${summaryHtml}</div>
        <div class="graph-section">
          <h3 class="graph-section-title">Topology</h3>
          ${topologyHtml}
          <div class="graph-detail" id="graph-detail"></div>
        </div>
        <div class="graph-section">
          <h3 class="graph-section-title">Validation</h3>
          <div class="issue-list">${issuesHtml}</div>
        </div>
      `;

      const nodeById = new Map(nodes.map((node) => [String(node.task_id), node]));
      planGraphEl.querySelectorAll(".svg-node").forEach((nodeEl) => {
        nodeEl.addEventListener("click", () => {
          const node = nodeById.get(nodeEl.getAttribute("data-node-id"));
          if (node) {
            selectGraphItem("node", node, nodeEl);
          }
        });
      });
      planGraphEl.querySelectorAll("[data-edge-index]").forEach((edgeEl) => {
        edgeEl.addEventListener("click", () => {
          const edge = layout.edges[Number(edgeEl.getAttribute("data-edge-index"))];
          if (edge) {
            selectGraphItem("edge", edge, edgeEl);
          }
        });
      });

      const canvasEl = document.getElementById("graph-canvas-wrap");
      if (canvasEl) {
        canvasEl.scrollLeft = previousScrollLeft;
        canvasEl.scrollTop = previousScrollTop;
      }

      if (previousSelection && previousSelection.type === "edge") {
        const selectedEdge = layout.edges.find((edge) =>
          String(edge.source) === String(previousSelection.item.source)
          && String(edge.target) === String(previousSelection.item.target)
          && String(edge.kind || "") === String(previousSelection.item.kind || "")
          && String(edge.label || "") === String(previousSelection.item.label || "")
        );
        if (selectedEdge) {
          const selectedIndex = layout.edges.indexOf(selectedEdge);
          const selectedEdgeEl = planGraphEl.querySelector(`[data-edge-index="${selectedIndex}"]`);
          selectGraphItem("edge", selectedEdge, selectedEdgeEl);
          return;
        }
      }

      if (previousSelection && previousSelection.type === "node") {
        const selectedNode = nodes.find((node) => String(node.task_id) === String(previousSelection.item.task_id));
        if (selectedNode) {
          const selectedNodeEl = planGraphEl.querySelector(`.svg-node[data-node-id="${selectedNode.task_id}"]`);
          selectGraphItem("node", selectedNode, selectedNodeEl);
          return;
        }
      }

      const preferredNode = nodes.find((node) => node.is_current) || nodes[0];
      if (preferredNode) {
        const preferredNodeEl = planGraphEl.querySelector(`.svg-node[data-node-id="${preferredNode.task_id}"]`);
        selectGraphItem("node", preferredNode, preferredNodeEl);
      }
    }

    function renderEvents(events) {
      if (!events.length) {
        eventsEl.innerHTML = '<div class="empty">No recent events.</div>';
        return;
      }

      eventsEl.innerHTML = events.map((event) => `
        <article class="event">
          <div class="event-type">${escapeHtml(event.type)}</div>
          <div>${escapeHtml(event.summary || "-")}</div>
          <div class="sub" style="margin-top:8px;color:#756657;font-size:12px;">
            ${escapeHtml(event.created_at)}${event.task_id !== null ? ` · task #${escapeHtml(event.task_id)}` : ""}
          </div>
        </article>
      `).join("");
    }

    function renderStatusStrip(state) {
      const pills = [
        `current_plan_id: ${formatJsonInline(state.current_plan_id)}`,
        `current_task_id: ${formatJsonInline(state.current_task_id)}`,
        `next_action: ${formatJsonInline(state.next_action_type)}`,
        `user_facing_response: ${state.user_facing_response ? "ready" : "empty"}`,
      ];
      statusStripEl.innerHTML = pills.map((text) => `<span class="status-pill">${escapeHtml(text)}</span>`).join("");
    }

    function renderSummary(state, payload) {
      document.getElementById("thread-id").textContent = payload.thread_id || "-";
      document.getElementById("updated-at").textContent = payload.updated_at || "-";
      document.getElementById("summary-plan").textContent = state.current_plan_id || "-";
      document.getElementById("summary-task").textContent = state.current_task_label || "-";
      document.getElementById("summary-next-action").textContent = state.next_action_type || "idle";
      document.getElementById("summary-task-count").textContent = String(state.visible_task_count || 0);
    }

    function renderError(errorText) {
      if (!errorText) {
        errorBannerEl.style.display = "none";
        errorBannerEl.textContent = "";
        return;
      }
      errorBannerEl.style.display = "block";
      errorBannerEl.textContent = errorText;
    }

    function setSelectValueIfAvailable(selectEl, value) {
      const normalized = String(value || "");
      if (!normalized) return;
      for (const option of selectEl.options) {
        if (option.value === normalized) {
          selectEl.value = normalized;
          return;
        }
      }
    }

    function initializeLanguageSelectors(payload) {
      if (languageInitialized) return;
      setSelectValueIfAvailable(inputLanguageEl, payload.input_language || "zh");
      setSelectValueIfAvailable(outputLanguageEl, payload.output_language || "zh");
      languageInitialized = true;
    }

    function renderPayload(payload) {
      initializeLanguageSelectors(payload);
      renderSummary(payload.state, payload);
      renderStatusStrip(payload.state);
      renderHistory(payload.conversation_history || payload.turn_history || []);
      renderTasks(payload.state.tasks || []);
      renderPlanGraph(payload.state.plan_graph || {});
      applyPlanViewMode();
      renderError(payload.latest_error || "");
    }

    async function refreshState() {
      const payload = await request("/api/state");
      renderPayload(payload);
    }

    async function sendMessage(message, role) {
      sendBtnEl.disabled = true;
      try {
        const payload = await request("/api/message", {
          method: "POST",
          body: JSON.stringify({ message, role }),
        });
        renderPayload(payload);
      } finally {
        sendBtnEl.disabled = false;
      }
    }

    // Override the legacy render helpers below with state-driven live UI behavior.
    function classifyLogLine(line) {
      const text = String(line || "");
      if (!text.trim()) {
        return "log-dim";
      }
      if (/error|traceback|exception|failed/i.test(text)) {
        return "log-error";
      }
      if (/warning|warn/i.test(text)) {
        return "log-warning";
      }
      if (/Tool |Invoking tool|Tool Result|tool_call/i.test(text)) {
        return "log-tool";
      }
      if (/^.*\bExecute\b|React Agent/i.test(text)) {
        return "log-execute";
      }
      if (/^.*\bPlan\b/i.test(text)) {
        return "log-plan";
      }
      if (/^.*\bOrchestrate\b|Route After Orchestrate/i.test(text)) {
        return "log-orchestrate";
      }
      if (/^.*\bIngest\b/i.test(text)) {
        return "log-ingest";
      }
      if (/^.*\bResponse\b/i.test(text)) {
        return "log-response";
      }
      return "log-dim";
    }

    function renderEvents(events) {
      if (!events.length) {
        eventsEl.innerHTML = '<div class="empty">No recent events.</div>';
        return;
      }

      eventsEl.innerHTML = events.map((event) => `
        <article class="event">
          <div class="event-type">${escapeHtml(event.type)}</div>
          <div>${escapeHtml(event.summary || "-")}</div>
          <div class="sub" style="margin-top:8px;color:#756657;font-size:12px;">
            ${escapeHtml(event.created_at || "-")}
            ${event.source ? ` | ${escapeHtml(event.source)}` : ""}
            ${event.task_id !== null && event.task_id !== undefined ? ` | task #${escapeHtml(event.task_id)}` : ""}
          </div>
        </article>
      `).join("");
    }

    function renderStatusStrip(state) {
      const pills = [
        `status: ${formatJsonInline(state.agent_status)}`,
        `plan: ${formatJsonInline(state.current_plan_id)}`,
        `task: ${formatJsonInline(state.current_task_id)}`,
        `next action: ${formatJsonInline(state.next_action_type)}`,
        `input mode: ${formatJsonInline((state.input_window || {}).mode || "ready")}`,
        `response: ${state.user_facing_response ? "ready" : "empty"}`,
      ];
      statusStripEl.innerHTML = pills.map((text) => `<span class="status-pill">${escapeHtml(text)}</span>`).join("");
    }

    function renderLogs(consoleEntries) {
      const entries = Array.isArray(consoleEntries)
        ? consoleEntries
        : String(consoleEntries || "").split(/\\r?\\n/).filter((line) => line.trim());
      if (!entries.length) {
        logsEl.innerHTML = '<div class="empty">No log output is available for this session yet.</div>';
        return;
      }
      const shouldStick = logAutoFollow || isNearBottom(logsEl, 36);
      const html = entries.map((entry) => {
        const text = String(entry || "");
        const firstLine = text.split(/\\r?\\n/, 1)[0] || "";
        const cls = classifyLogLine(firstLine);
        return `<span class="log-line ${cls}">${escapeHtml(text)}</span>`;
      }).join("");
      logsEl.innerHTML = `<pre class="console-pre">${html}</pre>`;
      if (shouldStick) {
        logsEl.scrollTop = logsEl.scrollHeight;
        logAutoFollow = true;
      }
      updateConsoleFollowIndicator();
    }

    function renderSessionChoice(checkpoint) {
      sessionChoiceRequired = Boolean(checkpoint && checkpoint.choice_required);
      if (!sessionChoiceRequired) {
        sessionChoiceEl.style.display = "none";
        sessionResumeSelectEl.innerHTML = "";
        resumeSessionBtnEl.disabled = false;
        newSessionBtnEl.disabled = false;
        return;
      }
      const previousSelection = sessionResumeSelectEl.value || "";
      const sessions = Array.isArray(checkpoint.available_sessions)
        ? checkpoint.available_sessions
        : [];
      const preferredPath = previousSelection
        || checkpoint.pending_run_memory_snapshot_path
        || (sessions[0] && sessions[0].path)
        || "";
      sessionResumeSelectEl.innerHTML = "";
      sessions.forEach((item) => {
        const option = document.createElement("option");
        option.value = item.path || "";
        option.textContent = item.label || item.path || "run_memory.json";
        sessionResumeSelectEl.appendChild(option);
      });
      if (preferredPath) {
        sessionResumeSelectEl.value = preferredPath;
      }
      sessionResumeSelectEl.style.display = sessions.length ? "block" : "none";
      const savedAt = checkpoint.pending_saved_at || checkpoint.saved_at || "";
      const sourcePath = sessionResumeSelectEl.value || checkpoint.pending_run_memory_snapshot_path || "";
      const detailParts = [];
      if (savedAt) detailParts.push("Saved: " + savedAt);
      if (sourcePath) detailParts.push("Memory: " + sourcePath);
      sessionChoiceDetailEl.textContent = detailParts.length
        ? detailParts.join(" | ")
        : "Resume the saved run memory or start clean.";
      sessionChoiceEl.style.display = "flex";
      resumeSessionBtnEl.disabled = false;
      newSessionBtnEl.disabled = false;
    }

    function renderSummary(state, payload) {
      document.getElementById("thread-id").textContent = payload.thread_id || "-";
      document.getElementById("updated-at").textContent = payload.updated_at || "-";
      const checkpoint = payload.checkpoint || {};
      const checkpointStatusEl = document.getElementById("checkpoint-status");
      if (checkpointStatusEl) {
        checkpointStatusEl.textContent = checkpoint.choice_required
          ? "choose"
          : checkpoint.run_memory_restored
          ? "restored"
          : checkpoint.mode === "new"
            ? "new"
          : checkpoint.enabled
            ? "saved"
            : "off";
      }
      renderSessionChoice(checkpoint);
      document.getElementById("summary-plan").textContent = state.current_plan_id || "-";
      document.getElementById("summary-task").textContent = state.current_task_label || state.agent_status || "-";
      document.getElementById("summary-next-action").textContent = state.processing ? "processing" : (state.next_action_type || "idle");
      document.getElementById("summary-task-count").textContent = String(state.visible_task_count || 0);
    }

    function updateComposerState(state) {
      const inputWindow = state.input_window || {};
      const blockedBySessionChoice = Boolean(sessionChoiceRequired);
      const locked = Boolean(inputWindow.locked) || blockedBySessionChoice;
      const mode = inputWindow.mode || "ready";
      const noteTitle = blockedBySessionChoice
        ? "选择会话"
        : inputWindow.title || "可以输入";
      const noteDetail = blockedBySessionChoice
        ? "请先继续已保存会话，或开启一个新会话。"
        : inputWindow.detail || "准备好后可以发送新的指令。";
      const placeholder = "输入用户指令。控制器到达事件会自动处理。";

      window.__carAgentLatestState = state;
      composerNoteEl.className = "composer-note " + escapeHtml(blockedBySessionChoice ? "waiting" : mode);
      composerNoteEl.innerHTML = "<strong>" + escapeHtml(noteTitle) + "</strong><span>" + escapeHtml(noteDetail) + "</span>";
      messageInputEl.disabled = locked;
      messageInputEl.placeholder = placeholder;
      sendBtnEl.disabled = locked || (!messageInputEl.value.trim() && !selectedImageDataUrl);
      clearBtnEl.disabled = state.processing || blockedBySessionChoice;
      refreshBtnEl.disabled = refreshInFlight;
      captureBtnEl.disabled = blockedBySessionChoice;
      imageUploadEl.disabled = blockedBySessionChoice;
      describeImageBtnEl.disabled = blockedBySessionChoice;
    }

    function renderPayload(payload) {
      const state = payload.state || {};
      window.__carAgentLatestState = state;
      initializeGuideProfile(state);
      renderSummary(state, payload);
      renderStatusStrip(state);
      renderGuidance(state.guidance_events || []);
      renderHistory(payload.conversation_history || payload.turn_history || []);
      renderTasks(state.tasks || []);
      renderPlanGraph(state.plan_graph || {});
      applyPlanViewMode();
      renderLogs(payload.console_entries || payload.console_output || "");
      renderLatestAgentCapture(payload.latest_agent_capture || null);
      updateComposerState(state);
      renderError(payload.latest_error || "");
    }

    async function refreshState() {
      if (refreshInFlight) {
        return;
      }

      refreshInFlight = true;
      refreshBtnEl.disabled = true;
      try {
        const payload = await request("/api/state");
        renderPayload(payload);
        const state = payload.state || {};
        const inputWindow = state.input_window || {};
        const nextDelay = state.processing
          ? 350
          : inputWindow.mode === "waiting"
            ? 800
            : 1400;
        scheduleRefresh(nextDelay);
      } catch (error) {
        renderError(error.message || String(error));
        scheduleRefresh(1800);
      } finally {
        refreshInFlight = false;
        const latestState = window.__carAgentLatestState || null;
        if (latestState) {
          updateComposerState(latestState);
        } else {
          refreshBtnEl.disabled = false;
        }
      }
    }

    function dataUrlExtension(dataUrl) {
      const match = String(dataUrl || "").match(/^data:image\/([^;,]+)/i);
      if (!match) return "jpg";
      const kind = match[1].toLowerCase();
      if (kind === "jpeg") return "jpg";
      return kind.replace(/[^a-z0-9]/g, "") || "jpg";
    }

    function safeDownloadName(name, dataUrl) {
      const fallback = "caragent_capture." + dataUrlExtension(dataUrl);
      const clean = String(name || "").trim().split(/[\\/]/).pop() || fallback;
      if (/\.[a-z0-9]{2,5}$/i.test(clean)) return clean;
      return clean + "." + dataUrlExtension(dataUrl);
    }

    function syncImagePreviewControls() {
      const previewUrl = previewImageDataUrl || selectedImageDataUrl || "";
      const attached = Boolean(selectedImageDataUrl && previewUrl === selectedImageDataUrl);
      if (previewUrl) {
        imagePreviewImgEl.src = previewUrl;
      } else {
        imagePreviewImgEl.removeAttribute("src");
      }
      imagePreviewStatusEl.textContent = attached ? "attached to next message" : "preview only";
      imagePreviewStatusEl.classList.toggle("attached", attached);
      attachImageBtnEl.disabled = !previewUrl;
      attachImageBtnEl.textContent = attached ? "Detach" : "Attach";
      clearImageBtnEl.disabled = !previewUrl;
      downloadImageBtnEl.disabled = !previewUrl;
      imagePreviewEl.style.display = previewUrl || imagePreviewTextEl.textContent ? "flex" : "none";
    }

    function clearImagePreview() {
      selectedImageDataUrl = "";
      previewImageDataUrl = "";
      selectedImageDownloadName = "caragent_capture.jpg";
      previewImageDownloadName = "caragent_capture.jpg";
      imageUploadEl.value = "";
      imagePreviewTextEl.textContent = "";
      syncImagePreviewControls();
      updateComposerState(window.__carAgentLatestState || { input_window: { locked: false, mode: "ready" }, processing: false });
    }

    function showImagePreview(dataUrl, text, attach, downloadName) {
      if (dataUrl !== undefined) {
        previewImageDataUrl = dataUrl || "";
        if (previewImageDataUrl) {
          previewImageDownloadName = safeDownloadName(downloadName, previewImageDataUrl);
        }
      }
      const attachToMessage = attach === true;
      if (attachToMessage) {
        selectedImageDataUrl = previewImageDataUrl || "";
        selectedImageDownloadName = safeDownloadName(downloadName, selectedImageDataUrl);
      }
      imagePreviewTextEl.textContent = text || "";
      syncImagePreviewControls();
    }

    function togglePreviewAttachment() {
      const previewUrl = previewImageDataUrl || selectedImageDataUrl || "";
      if (!previewUrl) return;
      if (selectedImageDataUrl && previewUrl === selectedImageDataUrl) {
        selectedImageDataUrl = "";
      } else {
        selectedImageDataUrl = previewUrl;
        selectedImageDownloadName = safeDownloadName(previewImageDownloadName, selectedImageDataUrl);
      }
      syncImagePreviewControls();
      updateComposerState(window.__carAgentLatestState || { input_window: { locked: false, mode: "ready" }, processing: false });
    }

    function renderLatestAgentCapture(capture) {
      if (!capture || !capture.image_data_url) {
        latestAgentCapture = null;
        agentCapturePreviewEl.style.display = "none";
        return;
      }
      const previousPath = latestAgentCapture && latestAgentCapture.path;
      latestAgentCapture = capture;
      agentCaptureImgEl.src = capture.image_data_url;
      agentCaptureTextEl.textContent = capture.path
        ? `Latest photo saved by agent tool.\nSaved on board: ${capture.path}`
        : "Latest photo saved by agent tool.";
      downloadAgentCaptureBtnEl.disabled = false;
      useAgentCaptureBtnEl.disabled = false;
      agentCapturePreviewEl.style.display = "flex";
      if (capture.path && capture.path !== previousPath && !previewImageDataUrl && !selectedImageDataUrl) {
        agentCapturePreviewEl.scrollIntoView({ block: "nearest", behavior: "smooth" });
      }
    }

    function readSelectedImageAsDataUrl() {
      const file = imageUploadEl.files && imageUploadEl.files[0];
      if (!file) {
        return Promise.resolve("");
      }
      return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(String(reader.result || ""));
        reader.onerror = () => reject(reader.error || new Error("Image read failed"));
        reader.readAsDataURL(file);
      });
    }

    async function captureCurrentImage() {
      captureBtnEl.disabled = true;
      try {
        const payload = await request("/api/current-image");
        if (!payload.image_data_url) {
          throw new Error(payload.error || "Current image is unavailable");
        }
        const sourceText = payload.path
          ? `已拍摄当前机器人画面，暂未附加到消息。\n板端保存位置：${payload.path}`
          : "已拍摄当前机器人画面，暂未附加到消息。";
        const captureName = payload.path ? payload.path.split("/").pop() : "";
        selectedImageDataUrl = "";
        showImagePreview(payload.image_data_url, sourceText, false, captureName);
        imageUploadEl.value = "";
        updateComposerState(window.__carAgentLatestState || { input_window: { locked: false, mode: "ready" }, processing: false });
      } catch (error) {
        renderError(error.message || String(error));
      } finally {
        captureBtnEl.disabled = false;
      }
    }

    async function describeSelectedImage() {
      describeImageBtnEl.disabled = true;
      try {
        const dataUrl = selectedImageDataUrl || previewImageDataUrl || await readSelectedImageAsDataUrl();
        if (!dataUrl) {
          throw new Error("Choose an image file first, or capture the current camera frame for preview.");
        }
        showImagePreview(dataUrl, "Describing image for navigation search...", false);
        const payload = await request("/api/upload-image", {
          method: "POST",
          body: JSON.stringify({
            image_data_url: dataUrl,
            input_language: inputLanguageEl.value,
            output_language: outputLanguageEl.value,
            submit: false,
          }),
        });
        const description = payload.description || "";
        showImagePreview(dataUrl, description, false);
        const prefix = outputLanguageEl.value === "zh" || inputLanguageEl.value === "zh"
          ? "\u8fd9\u662f\u4e00\u5f20\u76ee\u6807\u56fe\u7247\u751f\u6210\u7684\u7d27\u51d1\u68c0\u7d22\u63cf\u8ff0\u3002\u8bf7\u5728\u573a\u666f\u8bb0\u5fc6\u91cc\u5feb\u901f\u627e\u5230\u6700\u63a5\u8fd1\u7684\u5019\u9009\u5173\u952e\u5e27\u5e76\u5bfc\u822a\u8fc7\u53bb\uff1b\u4e0d\u8981\u8ffd\u6c42\u5b8c\u7f8e\u9010\u9879\u5339\u914d\uff0c\u5982\u679c\u591a\u4e2a\u5019\u9009\u5dee\u4e0d\u591a\u5c31\u9009\u6700\u597d\u7684\u4e00\u4e2a\uff1a"
          : "This is a compact search description generated from a target image. Quickly find the closest matching candidate keyframe in scene memory and navigate there; do not require a perfect detail-by-detail match, and if several candidates are close, choose the best one:";
        messageInputEl.value = prefix + "\\n" + description;
        updateComposerState(window.__carAgentLatestState || { input_window: { locked: false, mode: "ready" }, processing: false });
      } catch (error) {
        renderError(error.message || String(error));
      } finally {
        describeImageBtnEl.disabled = false;
      }
    }

    async function sendMessage(message) {
      sendBtnEl.disabled = true;
      messageInputEl.disabled = true;
      try {
        const imageDataUrl = selectedImageDataUrl || await readSelectedImageAsDataUrl();
        const payload = await request("/api/message", {
          method: "POST",
          body: JSON.stringify({
            message,
            role: "user",
            input_language: inputLanguageEl.value,
            output_language: outputLanguageEl.value,
            image_data_url: imageDataUrl,
          }),
        });
        renderPayload(payload);
        messageInputEl.value = "";
        selectedImageDataUrl = "";
        previewImageDataUrl = "";
        selectedImageDownloadName = "caragent_capture.jpg";
        previewImageDownloadName = "caragent_capture.jpg";
        imageUploadEl.value = "";
        showImagePreview("", "");
        scheduleRefresh(200);
        return payload;
      } catch (error) {
        renderError(error.message || String(error));
        scheduleRefresh(1200);
        throw error;
      } finally {
        const latestState = window.__carAgentLatestState || null;
        if (latestState) {
          updateComposerState(latestState);
        } else {
          sendBtnEl.disabled = false;
          messageInputEl.disabled = false;
        }
      }
    }

    formEl.addEventListener("submit", async (event) => {
      event.preventDefault();
      if (sendBtnEl.disabled) return;
      const message = messageInputEl.value.trim();
      if (!message && !selectedImageDataUrl && !(imageUploadEl.files && imageUploadEl.files[0])) return;
      try {
        await sendMessage(message);
      } catch (error) {
        // Error state is already rendered above.
      }
    });

    captureBtnEl.addEventListener("click", captureCurrentImage);
    describeImageBtnEl.addEventListener("click", describeSelectedImage);
    attachImageBtnEl.addEventListener("click", togglePreviewAttachment);
    clearImageBtnEl.addEventListener("click", clearImagePreview);
    downloadImageBtnEl.addEventListener("click", () => {
      const dataUrl = previewImageDataUrl || selectedImageDataUrl || imagePreviewImgEl.src || "";
      if (!dataUrl) return;
      const link = document.createElement("a");
      link.href = dataUrl;
      link.download = safeDownloadName(previewImageDownloadName || selectedImageDownloadName, dataUrl);
      document.body.appendChild(link);
      link.click();
      link.remove();
    });
    downloadAgentCaptureBtnEl.addEventListener("click", () => {
      if (!latestAgentCapture || !latestAgentCapture.image_data_url) return;
      const link = document.createElement("a");
      link.href = latestAgentCapture.image_data_url;
      link.download = safeDownloadName(latestAgentCapture.name || "agent_capture.jpg", latestAgentCapture.image_data_url);
      document.body.appendChild(link);
      link.click();
      link.remove();
    });
    useAgentCaptureBtnEl.addEventListener("click", () => {
      if (!latestAgentCapture || !latestAgentCapture.image_data_url) return;
      showImagePreview(
        latestAgentCapture.image_data_url,
        latestAgentCapture.path
          ? `Agent 拍摄画面已附加到下一条消息。\n板端保存位置：${latestAgentCapture.path}`
          : "Agent 拍摄画面已附加到下一条消息。",
        true,
        latestAgentCapture.name || "agent_capture.jpg"
      );
      imageUploadEl.value = "";
      updateComposerState(window.__carAgentLatestState || { input_window: { locked: false, mode: "ready" }, processing: false });
    });
    imageUploadEl.addEventListener("change", async () => {
      try {
        const dataUrl = await readSelectedImageAsDataUrl();
        if (dataUrl) {
          const file = imageUploadEl.files && imageUploadEl.files[0];
          showImagePreview(dataUrl, "图片已附加。输入指令后点击发送，或点击描述图片。", true, file ? file.name : "");
          const latestState = window.__carAgentLatestState || {
            input_window: { locked: false, mode: "ready" },
            processing: false,
          };
          updateComposerState(latestState);
        }
      } catch (error) {
        renderError(error.message || String(error));
      }
    });

    messageInputEl.addEventListener("input", () => {
      const latestState = window.__carAgentLatestState || {
        input_window: {
          mode: "ready",
          title: "可以输入",
          detail: "可以发送新的指令。",
          locked: false,
        },
        processing: false,
      };
      updateComposerState(latestState);
    });

    logsEl.addEventListener("scroll", () => {
      logAutoFollow = isNearBottom(logsEl, 36);
      updateConsoleFollowIndicator();
    });

    historyEl.addEventListener("scroll", () => {
      historyAutoFollow = isNearBottom(historyEl, 36);
    });

    consoleJumpBtnEl.addEventListener("click", () => {
      logsEl.scrollTop = logsEl.scrollHeight;
      logAutoFollow = true;
      updateConsoleFollowIndicator();
    });

    async function chooseSessionMode(mode) {
      resumeSessionBtnEl.disabled = true;
      newSessionBtnEl.disabled = true;
      try {
        const body = { mode };
        if (mode === "resume" && sessionResumeSelectEl.value) {
          body.run_memory_snapshot_path = sessionResumeSelectEl.value;
        }
        const payload = await request("/api/session", {
          method: "POST",
          body: JSON.stringify(body),
        });
        renderPayload(payload);
      } catch (error) {
        renderError(error.message || String(error));
        resumeSessionBtnEl.disabled = false;
        newSessionBtnEl.disabled = false;
      }
    }

    resumeSessionBtnEl.addEventListener("click", () => chooseSessionMode("resume"));
    newSessionBtnEl.addEventListener("click", () => chooseSessionMode("new"));

    guideEnableBtnEl.addEventListener("click", () => {
      guideVoiceEnabled = true;
      guideMuted = false;
      speakGuidance("语音播报已就绪。", { interrupt: true });
      updateGuideControls();
    });

    guideMuteBtnEl.addEventListener("click", () => {
      guideMuted = !guideMuted;
      if (guideMuted && speechSupported()) {
        window.speechSynthesis.cancel();
      }
      updateGuideControls();
    });

    guideReplayBtnEl.addEventListener("click", () => {
      if (guideLastSpokenText || guideLastEvent) {
        speakGuidance(guideLastSpokenText || guideLastEvent.text || "", { interrupt: true });
      }
      updateGuideControls();
    });

    planViewListBtnEl.addEventListener("click", () => {
      planViewMode = "list";
      applyPlanViewMode();
    });

    planViewGraphBtnEl.addEventListener("click", () => {
      planViewMode = "graph";
      applyPlanViewMode();
    });

    document.getElementById("refresh-btn").addEventListener("click", async () => {
      await refreshState();
    });
    document.getElementById("clear-btn").addEventListener("click", async () => {
      clearBtnEl.disabled = true;
      try {
        const payload = await request("/api/clear", { method: "POST", body: "{}" });
        renderPayload(payload);
      } catch (error) {
        renderError(error.message || String(error));
      } finally {
        clearBtnEl.disabled = false;
        const latestState = window.__carAgentLatestState || null;
        if (latestState) {
          updateComposerState(latestState);
        }
      }
    });
    window.__carAgentLatestState = null;
    updateConsoleFollowIndicator();
    updateGuideControls();
    applyPlanViewMode();
    refreshState();
  </script>
</body>
</html>
"""


LITE_APP_HTML = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CarAgent 导引助手</title>
<style>
  :root{--bg:#eef2f7;--panel:#fff;--ink:#172033;--muted:#64748b;--line:#d7dde8;--blue:#1f6fd1}
  *{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font-family:Inter,system-ui,-apple-system,"Segoe UI",sans-serif}
  .shell{height:100vh;display:grid;grid-template-rows:auto 1fr auto;max-width:860px;margin:0 auto;background:var(--panel);border-left:1px solid var(--line);border-right:1px solid var(--line)}
  header{padding:14px 18px;background:#102033;color:white;display:flex;align-items:center;gap:12px}h1{font-size:19px;margin:0}.status{font-size:13px;color:#dbeafe;margin-left:auto}
  .chat{overflow:auto;padding:18px;display:flex;flex-direction:column;gap:12px;background:linear-gradient(#f8fafc,#eef2f7)}
  .bubble{max-width:78%;border-radius:16px;padding:12px 14px;line-height:1.55;box-shadow:0 4px 14px rgba(15,23,42,.06);white-space:pre-wrap}
  .user{align-self:flex-end;background:#dbeafe}.agent{align-self:flex-start;background:white;border:1px solid var(--line)}.event{align-self:center;background:#fff7ed;color:#8a4b0a;font-size:13px;padding:7px 12px;border-radius:999px}
  .composer{border-top:1px solid var(--line);padding:12px;background:white}.preview{display:none;margin-bottom:8px}.preview.show{display:flex;gap:8px;align-items:center}.preview img{width:88px;height:66px;object-fit:cover;border-radius:8px;border:1px solid var(--line)}
  textarea{width:100%;min-height:70px;resize:vertical;border:1px solid var(--line);border-radius:10px;padding:10px;font-size:16px}
  .teleop{border-top:1px solid var(--line);padding:12px;background:#f8fafc;display:grid;grid-template-columns:96px 1fr 88px;gap:14px;align-items:center}
  .teleop label{font-size:13px;color:var(--muted);display:block;margin-bottom:8px}
  .stick{display:flex;flex-direction:column;align-items:center;gap:6px}.stick input{width:160px;accent-color:var(--blue);touch-action:none}
  .stick.vertical input{transform:rotate(-90deg);margin:48px 0}
  .teleop-actions{display:flex;flex-direction:column;gap:8px}.teleop .danger{background:#fee2e2;border-color:#fecaca;color:#991b1b;font-weight:700}.teleop .active{background:#dcfce7;border-color:#86efac;color:#14532d}
  .row{display:flex;gap:8px;align-items:center;margin-top:8px}button,label.btn{border:1px solid var(--line);background:#eef2f7;border-radius:8px;padding:9px 12px;font-size:15px;cursor:pointer}button.primary{background:var(--blue);border-color:var(--blue);color:white}.voice{color:var(--muted);font-size:13px}.spacer{flex:1}input[type=file]{display:none}
  @media(max-width:640px){.shell{border:0}.bubble{max-width:88%}}
</style>
</head>
<body>
<div class="shell">
  <header><h1>CarAgent 导引助手</h1><div class="status" id="status">准备就绪</div></header>
  <main class="chat" id="chat"></main>
  <section class="teleop" aria-label="手动遥控">
    <div class="stick vertical"><label>前后</label><input id="teleop-linear" type="range" min="-100" max="100" value="0" step="1"></div>
    <div class="stick"><label>左右转向</label><input id="teleop-angular" type="range" min="-100" max="100" value="0" step="1"><div class="voice" id="teleop-status">遥控松手即停</div></div>
    <div class="teleop-actions"><button type="button" id="speed-btn">低速</button><button type="button" class="danger" id="estop-btn">急停</button></div>
  </section>
  <form class="composer" id="form">
    <div class="preview" id="preview"><img id="preview-img" alt="待发送图片"><span id="preview-name"></span><button type="button" onclick="clearImage()">移除</button></div>
    <textarea id="message" placeholder="请输入你的问题或导引需求"></textarea>
    <div class="row">
      <label class="btn" for="image">选择图片</label><input id="image" type="file" accept="image/*">
      <button type="button" id="voice-btn">测试语音</button>
      <button type="button" id="mute-btn">静音</button>
      <button type="button" id="replay-btn">重播</button>
      <span class="spacer"></span>
      <button class="primary" type="submit">发送</button>
    </div>
    <div class="voice" id="voice-status">语音默认开启；浏览器可能需要先点击“测试语音”。</div>
  </form>
</div>
<script>
const $=(id)=>document.getElementById(id);
let imageDataUrl="", imageName="", seenTurnIds=new Set(), seenResponses=new Set(), seenEvents=new Set();
let voiceEnabled=true, muted=false, voiceUnlocked=false, pendingSpeech="", lastSpoken="", firstLoad=true;
let teleopMode="normal", teleopTimer=null, teleopActive=false, teleopInFlight=false, teleopQueued=false, teleopQueuedStop=false;
let liteRefreshTimer=null, liteRefreshInFlight=false, fastRefreshUntil=0;
let spokenTexts=[], spokenGuidanceEvents=[];
function speechSupported(){return "speechSynthesis" in window && "SpeechSynthesisUtterance" in window}
function setVoiceStatus(text){$("voice-status").textContent=text}
function pickZhVoice(){const voices=speechSupported()?speechSynthesis.getVoices():[]; return voices.find(v=>/zh|cmn|mandarin|chinese/i.test(`${v.lang} ${v.name}`))||voices[0]||null}
function makeUtterance(text){const u=new SpeechSynthesisUtterance(text); u.lang="zh-CN"; const v=pickZhVoice(); if(v)u.voice=v; u.rate=.95; u.pitch=1; u.onstart=()=>setVoiceStatus("正在播报"); u.onend=()=>setVoiceStatus("语音已启用"); u.onerror=(ev)=>setVoiceStatus(`语音播报失败：${ev.error||"浏览器拦截"}`); return u}
function unlockSpeech(playTest=true){if(!speechSupported()){setVoiceStatus("当前浏览器不支持语音合成，请换用手机 Chrome、Edge 或 Safari。"); return false} voiceEnabled=true; muted=false; voiceUnlocked=true; speechSynthesis.resume(); if(playTest){speechSynthesis.cancel(); speechSynthesis.speak(makeUtterance("语音播报已就绪。")); lastSpoken="语音播报已就绪。"} if(pendingSpeech){const text=pendingSpeech; pendingSpeech=""; setTimeout(()=>speak(text,true),160)} $("mute-btn").textContent="静音"; return true}
function normalizedSpeechText(text){return String(text||"").replace(/[“”"'‘’。！？!?,，、；;：:\\s]/g,"").trim()}
function recentlySpokenSimilar(text){const n=normalizedSpeechText(text); if(!n)return true; return spokenTexts.some(item=>{const old=normalizedSpeechText(item.text); if(!old)return false; return n===old||n.includes(old)||old.includes(n)})}
function rememberSpoken(text){spokenTexts.push({text:String(text||""),at:Date.now()}); const cutoff=Date.now()-45000; spokenTexts=spokenTexts.filter(item=>item.at>=cutoff).slice(-24)}
function speak(text, interrupt=false, force=false){const t=String(text||"").trim(); if(!t||!voiceEnabled||muted)return; if(!force&&recentlySpokenSimilar(t))return; if(!speechSupported()){setVoiceStatus("当前浏览器不支持语音合成。"); return} if(!voiceUnlocked){pendingSpeech=t; setVoiceStatus("手机浏览器需要先点击“测试语音”解锁播报。"); return} speechSynthesis.resume(); if(interrupt) speechSynthesis.cancel(); speechSynthesis.speak(makeUtterance(t)); lastSpoken=t; rememberSpoken(t)}
function bubble(cls,text){const el=document.createElement("div"); el.className="bubble "+cls; el.textContent=text; $("chat").appendChild(el); $("chat").scrollTop=$("chat").scrollHeight}
function eventBubble(text){const el=document.createElement("div"); el.className="event"; el.textContent=text; $("chat").appendChild(el); $("chat").scrollTop=$("chat").scrollHeight}
async function request(path, options={}){const r=await fetch(path,{headers:{"Content-Type":"application/json"},...options}); const d=await r.json(); if(!r.ok) throw new Error(d.error||"请求失败"); return d}
function teleopLimits(){return teleopMode==="slow"?{linear:.06,angular:.20}:{linear:.18,angular:.50}}
function teleopPayload(stop=false){const lim=teleopLimits(); const linear=stop?0:(Number($("teleop-linear").value||0)/100)*lim.linear; const angular=stop?0:-(Number($("teleop-angular").value||0)/100)*lim.angular; return stop?{command:"stop"}:{linear,angular,mode:teleopMode}}
function dashboardTeleopUrl(){return `${location.protocol}//${location.hostname}:8234/api/teleop`}
async function postTeleop(payload){const body=JSON.stringify(payload); try{const r=await fetch(dashboardTeleopUrl(),{method:"POST",headers:{"Content-Type":"application/json"},body}); const d=await r.json(); if(!r.ok) throw new Error(d.error||"遥控请求失败"); return d}catch(_){const r=await fetch("/api/teleop",{method:"POST",headers:{"Content-Type":"application/json"},body}); const d=await r.json(); if(!r.ok) throw new Error(d.error||"遥控请求失败"); return d}}
function queueTeleop(stop=false){teleopQueued=true; teleopQueuedStop=!!stop; if(!teleopInFlight) flushTeleop()}
async function flushTeleop(){if(!teleopQueued)return; teleopInFlight=true; const stop=teleopQueuedStop; teleopQueued=false; teleopQueuedStop=false; try{await postTeleop(teleopPayload(stop)); $("teleop-status").textContent=stop?"已停车":"手动遥控中"}catch(e){$("teleop-status").textContent=e.message||String(e)} finally{teleopInFlight=false; if(teleopQueued || (teleopActive&&!teleopTimer)) flushTeleop()}}
function startTeleopLoop(ev){if(ev&&ev.preventDefault)ev.preventDefault(); if(ev&&ev.currentTarget&&ev.currentTarget.setPointerCapture&&ev.pointerId!==undefined){try{ev.currentTarget.setPointerCapture(ev.pointerId)}catch(_){}} teleopActive=true; queueTeleop(false); if(!teleopTimer)teleopTimer=setInterval(()=>queueTeleop(false),80)}
function stopTeleopLoop(ev){if(ev&&ev.preventDefault)ev.preventDefault(); teleopActive=false; if(teleopTimer){clearInterval(teleopTimer); teleopTimer=null} $("teleop-linear").value=0; $("teleop-angular").value=0; queueTeleop(true)}
function handleTeleopInput(){const moving=Math.abs(Number($("teleop-linear").value||0))+Math.abs(Number($("teleop-angular").value||0))>0; if(moving){if(!teleopActive)startTeleopLoop(); else queueTeleop(false)}else if(teleopActive){stopTeleopLoop()}}
function requiredEvent(type){return ["request_received","plan_created","plan_updated","navigation_start","arrival","arrival_verification","failed","cancelled","stuck"].includes(type)}
function navigationEventType(type){return ["navigation_start","arrival","arrival_verification","failed","cancelled","stuck"].includes(String(type||""))}
function rememberGuidanceEvent(type,text){spokenGuidanceEvents.push({type:String(type||""),text:String(text||""),at:Date.now()}); const cutoff=Date.now()-20000; spokenGuidanceEvents=spokenGuidanceEvents.filter(item=>item.at>=cutoff).slice(-24)}
function guidanceTextSimilarToFinal(text){const n=normalizedSpeechText(text); if(!n)return false; const cutoff=Date.now()-20000; return spokenGuidanceEvents.some(item=>{if(item.at<cutoff||!navigationEventType(item.type))return false; const old=normalizedSpeechText(item.text); if(!old)return false; return n===old||n.includes(old)||old.includes(n)})}
function hasCjk(text){return /[\u3400-\u9fff]/.test(String(text||""))}
function wantsChinese(payload, turn){return String((turn&&turn.output_language)||payload.output_language||"zh").toLowerCase().startsWith("zh")}
function finalTurnText(payload, t){const completed=String(t.status||"")==="completed"; if(!completed)return ""; const responseItems=Array.isArray(t.response_items)?t.response_items:[]; const visibleItems=responseItems.filter(item=>!["progress"].includes(String(item.response_type||""))); const finalText=String(t.response||t.turn_response_text||"").trim(); if(finalText&&visibleItems.length)return finalText; const fallback=visibleItems.length?String(visibleItems[visibleItems.length-1].response_text||"").trim():""; if(wantsChinese(payload,t)&&fallback&&!hasCjk(fallback))return ""; return fallback}
function render(payload){
  const state=payload.state||{};
  $("status").textContent=state.processing?"正在处理":"准备就绪";
  const pendingFinals=[];
  const turns=payload.conversation_history||[];
  turns.forEach(t=>{
    const tid=String(t.turn_id||"");
    if(tid&&!seenTurnIds.has(tid)){
      seenTurnIds.add(tid);
      if(t.role==="user") bubble("user", t.content||"");
    }
    const txt=finalTurnText(payload,t);
    const id=String(t.response_id||t.turn_response_id||`${tid}:final:${txt}`);
    if(txt&&!seenResponses.has(id)){
      seenResponses.add(id);
      pendingFinals.push({text:txt,isError:String(t.turn_response_type||"")==="error"});
    }
  });
  (state.guidance_events||[]).forEach(ev=>{
    const id=String(ev.event_id||"");
    if(id&&seenEvents.has(id))return;
    const type=String(ev.event_type||"");
    const text=String(ev.text||"").trim();
    if(!text||!requiredEvent(type))return;
    if(String(payload.output_language||"zh").toLowerCase().startsWith("zh")&&!hasCjk(text))return;
    if(id)seenEvents.add(id);
    eventBubble(text);
    if(!firstLoad){
      speak(text,["failed","cancelled","stuck"].includes(type),true);
      rememberGuidanceEvent(type,text);
    }
  });
  pendingFinals.forEach(item=>{
    const suppressNavFinal=!item.isError&&guidanceTextSimilarToFinal(item.text);
    if(suppressNavFinal)return;
    bubble("agent",item.text);
    if(!firstLoad)speak(item.text,item.isError);
  });
  firstLoad=false;
}
function requestFastRefresh(ms=8000){fastRefreshUntil=Math.max(fastRefreshUntil,Date.now()+ms); scheduleLiteRefresh(120)}
function nextRefreshDelay(payload){const state=(payload&&payload.state)||{}; const inputWindow=state.input_window||{}; if(Date.now()<fastRefreshUntil)return 200; return state.processing?350:(inputWindow.mode==="waiting"?600:1200)}
function scheduleLiteRefresh(delayMs){if(liteRefreshTimer)clearTimeout(liteRefreshTimer); liteRefreshTimer=setTimeout(()=>refresh(),Math.max(200,Number(delayMs)||1200))}
async function refresh(){if(liteRefreshInFlight){scheduleLiteRefresh(250); return} liteRefreshInFlight=true; try{const payload=await request("/api/lite-state"); render(payload); scheduleLiteRefresh(nextRefreshDelay(payload))}catch(e){$("status").textContent=e.message||String(e); scheduleLiteRefresh(1800)}finally{liteRefreshInFlight=false}}
async function startLite(){try{let payload=await request("/api/lite-state"); render(payload); const checkpoint=payload.checkpoint||{}; if(checkpoint.choice_required){payload=await request("/api/session",{method:"POST",body:JSON.stringify({mode:"new"})}); render(payload)} scheduleLiteRefresh(450)}catch(e){$("status").textContent=e.message||String(e); scheduleLiteRefresh(1800)}}
function clearImage(){imageDataUrl=""; imageName=""; $("preview").classList.remove("show"); $("image").value=""}
$("image").addEventListener("change",()=>{const f=$("image").files&&$("image").files[0]; if(!f)return; const reader=new FileReader(); reader.onload=()=>{imageDataUrl=String(reader.result||""); imageName=f.name; $("preview-img").src=imageDataUrl; $("preview-name").textContent=f.name; $("preview").classList.add("show")}; reader.readAsDataURL(f)});
$("form").addEventListener("submit",async(ev)=>{ev.preventDefault(); const msg=$("message").value.trim(); if(!msg&&!imageDataUrl)return; const imagePayload=imageDataUrl; speak("已收到你的指令。",true,true); $("message").value=""; clearImage(); requestFastRefresh(20000); try{const payload=await request("/api/message",{method:"POST",body:JSON.stringify({message:msg,input_language:"zh",output_language:"zh",image_data_url:imagePayload})}); render(payload); requestFastRefresh(12000)}catch(e){bubble("agent",e.message||String(e)); speak(e.message||String(e),true); scheduleLiteRefresh(1200)}});
$("voice-btn").onclick=()=>unlockSpeech(true);
$("mute-btn").onclick=()=>{muted=!muted; $("mute-btn").textContent=muted?"取消静音":"静音"; setVoiceStatus(muted?"语音已静音":"语音已启用"); if(muted&&speechSupported()) speechSynthesis.cancel()};
$("replay-btn").onclick=()=>{unlockSpeech(false); speak(lastSpoken||pendingSpeech,true)};
if(speechSupported()){speechSynthesis.onvoiceschanged=()=>pickZhVoice()}
$("speed-btn").onclick=()=>{teleopMode=teleopMode==="slow"?"normal":"slow"; $("speed-btn").textContent=teleopMode==="slow"?"更低速":"低速"; $("speed-btn").classList.toggle("active",teleopMode==="slow")};
$("estop-btn").onclick=()=>stopTeleopLoop();
["teleop-linear","teleop-angular"].forEach(id=>{const el=$(id); if(window.PointerEvent){el.addEventListener("pointerdown",startTeleopLoop); el.addEventListener("pointerup",stopTeleopLoop); el.addEventListener("pointercancel",stopTeleopLoop)}else{el.addEventListener("touchstart",startTeleopLoop,{passive:false}); el.addEventListener("touchend",stopTeleopLoop,{passive:false}); el.addEventListener("mousedown",startTeleopLoop); window.addEventListener("mouseup",stopTeleopLoop)} el.addEventListener("input",handleTeleopInput); el.addEventListener("change",handleTeleopInput)});
window.addEventListener("blur",()=>{if(teleopActive)stopTeleopLoop()});
startLite();
</script>
</body>
</html>"""


SIM_VIEW_HTML = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>CarAgent 仿真可视化</title>
<style>
:root{color-scheme:light;--bg:#f7f8fb;--panel:#fff;--ink:#172033;--muted:#69758a;--line:#d9dfeb;--blue:#2563eb;--green:#16a34a;--red:#dc2626;--amber:#d97706}
*{box-sizing:border-box}body{margin:0;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","Microsoft YaHei",Arial,sans-serif;background:var(--bg);color:var(--ink)}
.app{min-height:100vh;display:grid;grid-template-rows:auto 1fr}
header{display:flex;gap:12px;align-items:center;justify-content:space-between;padding:12px 16px;background:#fff;border-bottom:1px solid var(--line)}
h1{font-size:18px;margin:0;font-weight:700}.meta{font-size:13px;color:var(--muted)}
.actions{display:flex;gap:8px;align-items:center}a,button{border:1px solid var(--line);background:#fff;border-radius:8px;color:var(--ink);padding:8px 12px;text-decoration:none;font:inherit;cursor:pointer}button.primary{background:var(--blue);border-color:var(--blue);color:#fff}
main{display:grid;grid-template-columns:minmax(0,1fr) 360px;gap:12px;padding:12px;min-height:0}
.map-panel,.side{background:var(--panel);border:1px solid var(--line);border-radius:10px;min-height:0}
.map-panel{display:grid;grid-template-rows:auto 1fr;overflow:hidden}.toolbar{display:flex;gap:8px;align-items:center;padding:10px;border-bottom:1px solid var(--line)}
.toolbar input{flex:1;min-width:120px;border:1px solid var(--line);border-radius:8px;padding:10px 12px;font:inherit}
.stage{position:relative;min-height:360px}.stage canvas{width:100%;height:100%;display:block;background:#fbfcff}
.legend{position:absolute;left:12px;bottom:12px;background:rgba(255,255,255,.92);border:1px solid var(--line);border-radius:8px;padding:8px 10px;font-size:12px;color:var(--muted);display:flex;gap:12px;flex-wrap:wrap}.dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:5px;vertical-align:-1px}.dot.kf{background:#94a3b8}.dot.robot{background:var(--blue)}.dot.dest{background:var(--red)}.dot.path{background:var(--green)}
.side{display:grid;grid-template-rows:auto auto 1fr;overflow:hidden}.card{padding:12px;border-bottom:1px solid var(--line)}.label{font-size:12px;color:var(--muted);margin-bottom:5px}.value{font-size:15px;line-height:1.45}.status{font-weight:700}.events{overflow:auto;padding:8px 12px}.event{padding:9px 0;border-bottom:1px solid #eef2f7}.event .type{font-size:12px;color:var(--muted)}.event .text{font-size:14px;line-height:1.4}
@media(max-width:860px){main{grid-template-columns:1fr;grid-template-rows:60vh minmax(320px,40vh)}.side{min-height:320px}header{align-items:flex-start;flex-direction:column}.actions{width:100%;justify-content:space-between}.stage{min-height:0}}
</style>
</head>
<body>
<div class="app">
  <header>
    <div><h1>CarAgent 仿真可视化</h1><div class="meta" id="meta">连接中...</div></div>
    <div class="actions"><a href="/lite" target="_blank">Lite UI</a><button id="reset-view">重置视图</button></div>
  </header>
  <main>
    <section class="map-panel">
      <div class="toolbar">
        <input id="command" placeholder="输入仿真指令" autocomplete="off" />
        <button class="primary" id="send">发送</button>
      </div>
      <div class="stage">
        <canvas id="map"></canvas>
        <div class="legend">
          <span><i class="dot kf"></i>关键帧</span>
          <span><i class="dot robot"></i>小车</span>
          <span><i class="dot dest"></i>目标</span>
          <span><i class="dot path"></i>轨迹</span>
        </div>
      </div>
    </section>
    <aside class="side">
      <div class="card"><div class="label">状态</div><div class="value status" id="status">-</div></div>
      <div class="card"><div class="label">导航</div><div class="value" id="nav">-</div></div>
      <div class="events" id="events"></div>
    </aside>
  </main>
</div>
<script>
const $=id=>document.getElementById(id);
const canvas=$("map"), ctx=canvas.getContext("2d");
let latest=null, trail=[], lastRobot=null, viewBounds=null, manualBounds=false, sending=false;
function req(path,opt){return fetch(path,Object.assign({headers:{"Content-Type":"application/json"}},opt||{})).then(r=>{if(!r.ok)throw new Error(r.status+" "+r.statusText);return r.json()})}
function pos2(p){return Array.isArray(p)&&p.length>=2?[Number(p[0]),Number(p[1])]:null}
function dist(a,b){return Math.hypot(a[0]-b[0],a[1]-b[1])}
function resize(){const box=canvas.parentElement.getBoundingClientRect();const dpr=window.devicePixelRatio||1;canvas.width=Math.max(320,Math.floor(box.width*dpr));canvas.height=Math.max(260,Math.floor(box.height*dpr));canvas.style.width=box.width+"px";canvas.style.height=box.height+"px";ctx.setTransform(dpr,0,0,dpr,0,0);draw()}
function computeBounds(data){const pts=[];(data.keyframes||[]).forEach(k=>{const p=pos2(k.position);if(p)pts.push(p)});const rp=pos2(data.robot&&data.robot.position);if(rp)pts.push(rp);const dp=pos2(data.navigation&&data.navigation.destination_position);if(dp)pts.push(dp);trail.forEach(p=>pts.push(p));if(!pts.length)return {minX:-2,maxX:2,minY:-2,maxY:2};let minX=pts[0][0],maxX=pts[0][0],minY=pts[0][1],maxY=pts[0][1];pts.forEach(p=>{minX=Math.min(minX,p[0]);maxX=Math.max(maxX,p[0]);minY=Math.min(minY,p[1]);maxY=Math.max(maxY,p[1])});const pad=Math.max(0.8,(maxX-minX+maxY-minY)*0.06);return {minX:minX-pad,maxX:maxX+pad,minY:minY-pad,maxY:maxY+pad}}
function project(p,b){const w=canvas.clientWidth,h=canvas.clientHeight;const sx=w/Math.max(.1,b.maxX-b.minX),sy=h/Math.max(.1,b.maxY-b.minY);const s=Math.min(sx,sy)*0.9;const cx=(b.minX+b.maxX)/2,cy=(b.minY+b.maxY)/2;return [w/2+(p[0]-cx)*s,h/2-(p[1]-cy)*s]}
function line(points,b,color,width){if(points.length<2)return;ctx.beginPath();points.forEach((p,i)=>{const q=project(p,b);if(i)ctx.lineTo(q[0],q[1]);else ctx.moveTo(q[0],q[1])});ctx.strokeStyle=color;ctx.lineWidth=width;ctx.lineJoin="round";ctx.lineCap="round";ctx.stroke()}
function dot(p,b,r,color,stroke,label){const q=project(p,b);ctx.beginPath();ctx.arc(q[0],q[1],r,0,Math.PI*2);ctx.fillStyle=color;ctx.fill();if(stroke){ctx.strokeStyle=stroke;ctx.lineWidth=2;ctx.stroke()}if(label){ctx.fillStyle="#344054";ctx.font="12px sans-serif";ctx.fillText(label,q[0]+r+4,q[1]-r-2)}}
function draw(){ctx.clearRect(0,0,canvas.clientWidth,canvas.clientHeight);if(!latest)return;const b=manualBounds&&viewBounds?viewBounds:computeBounds(latest);if(!manualBounds)viewBounds=b;ctx.strokeStyle="#e5eaf3";ctx.lineWidth=1;for(let i=0;i<8;i++){const x=canvas.clientWidth*i/7;ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,canvas.clientHeight);ctx.stroke();const y=canvas.clientHeight*i/7;ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(canvas.clientWidth,y);ctx.stroke()}line(trail,b,"#16a34a",3);(latest.keyframes||[]).forEach(k=>{const p=pos2(k.position);if(p)dot(p,b,3.5,"#94a3b8",null,String(k.kf_id))});const nav=latest.navigation||{};const dp=pos2(nav.destination_position);if(dp){if(lastRobot)line([lastRobot,dp],b,"#dc2626",2);dot(dp,b,8,"#dc2626","#fff","目标")}const rp=pos2(latest.robot&&latest.robot.position);if(rp){dot(rp,b,9,"#2563eb","#fff","小车")}}
function render(data){latest=data;const rp=pos2(data.robot&&data.robot.position);if(rp){if(!lastRobot||dist(lastRobot,rp)>0.02){trail.push(rp);trail=trail.slice(-80);lastRobot=rp}}$("meta").textContent=`${data.updated_at||""} · ${data.simulation_mode?"虚拟导航":"实机/非仿真"} · ${data.keyframes.length} 个关键帧`;$("status").textContent=(data.robot&&data.robot.status)||data.agent_status||"-";const nav=data.navigation||{};$("nav").textContent=nav.active?`${nav.kind||"导航"} -> ${nav.label||nav.description||"目标"}`:"暂无导航";const box=$("events");box.innerHTML=(data.guidance_events||[]).slice(-14).reverse().map(e=>`<div class="event"><div class="type">${e.event_type||""}</div><div class="text">${e.text||""}</div></div>`).join("")||'<div class="event"><div class="text">暂无事件</div></div>';draw()}
async function refresh(){try{const data=await req("/api/sim-state");render(data)}catch(e){$("status").textContent=e.message||String(e)}setTimeout(refresh,650)}
async function send(){const msg=$("command").value.trim();if(!msg||sending)return;sending=true;$("command").value="";try{await req("/api/message",{method:"POST",body:JSON.stringify({message:msg,input_language:"zh",output_language:"zh"})})}catch(e){$("status").textContent=e.message||String(e)}finally{sending=false;setTimeout(refresh,250)}}
$("send").onclick=send;$("command").addEventListener("keydown",e=>{if(e.key==="Enter")send()});$("reset-view").onclick=()=>{manualBounds=false;viewBounds=null;trail=[];lastRobot=null;draw()};window.addEventListener("resize",resize);resize();refresh();
</script>
</body>
</html>"""


def _now_display() -> str:
    """Return a human-readable local timestamp for UI updates and turn history."""

    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _normalize_task_detail(task: dict[str, Any]) -> str:
    """Render compact task metadata as a multi-line detail block for task cards."""

    details: list[str] = []
    if task.get("plan_id"):
        details.append(f"plan_id: {task['plan_id']}")
    if task.get("next_task_id") is not None:
        details.append(f"next_task_id: {task['next_task_id']}")
    if task.get("depends_on"):
        details.append(f"depends_on: {task['depends_on']}")
    if task.get("latest_result_summary"):
        details.append(f"latest_result: {task['latest_result_summary']}")
    object_debug = task.get("object_approach_debug")
    if isinstance(object_debug, dict):
        if object_debug.get("output_dir"):
            details.append(f"object_output_dir: {object_debug['output_dir']}")
        if object_debug.get("summary_json"):
            details.append(f"object_summary_json: {object_debug['summary_json']}")
        if object_debug.get("stages"):
            details.append(f"object_stages: {object_debug['stages']}")
    if task.get("terminal_reason"):
        details.append(f"terminal_reason: {task['terminal_reason']}")
    return "\n".join(details)


def _extract_object_approach_debug(latest_result: dict[str, Any] | None) -> dict[str, Any] | None:
    if not latest_result:
        return None
    raw_output = latest_result.get("raw_output")
    if not raw_output:
        return None
    try:
        trace = json.loads(str(raw_output))
    except Exception:
        return None
    tool_results = trace.get("tool_results") if isinstance(trace, dict) else None
    if not isinstance(tool_results, list):
        return None
    for tool_result in reversed(tool_results):
        if not isinstance(tool_result, dict):
            continue
        if str(tool_result.get("name") or "") != "approach_object_in_current_view":
            continue
        content = tool_result.get("content")
        if isinstance(content, str):
            try:
                content = json.loads(content)
            except Exception:
                continue
        if not isinstance(content, dict):
            continue
        data = content.get("data")
        if not isinstance(data, dict):
            continue
        paths = data.get("paths") if isinstance(data.get("paths"), dict) else {}
        stages = data.get("stages") if isinstance(data.get("stages"), list) else []
        stage_text = " -> ".join(
            f"{stage.get('name')}:{stage.get('status')}"
            for stage in stages
            if isinstance(stage, dict) and stage.get("name")
        )
        return {
            "output_dir": data.get("output_dir") or paths.get("output_dir"),
            "summary_json": data.get("summary_json") or paths.get("summary_json"),
            "stages": stage_text,
        }
    return None


class AsyncAgentWebApp:
    """Serve one AsyncAgent instance behind a tiny browser-based console."""

    def __init__(
        self,
        agent: AsyncAgent,
        thread_id: str,
        *,
        resume_checkpoint: bool | None = None,
        checkpoint_path: Path | str | None = None,
        clear_resumed_plan: bool | None = None,
    ):
        """Initialize web-session state and cached agent state."""

        self.agent = agent
        self.thread_id = thread_id
        self.io_cfg = config.get("io", {})
        web_cfg = config.get("web_ui", {}) or {}
        self.last_input_language = normalize_language(
            self.io_cfg.get("input_language", "zh"),
            fallback="zh",
        )
        self.last_output_language = normalize_language(
            self.io_cfg.get("output_language", "zh"),
            fallback="zh",
        )
        self.lock = threading.RLock()
        self.turn_history: list[dict[str, Any]] = []
        self.latest_error: str | None = None
        self.updated_at = _now_display()
        self.turn_counter = 0
        self.processing_turn_id: int | None = None
        self.controller_arrival_turn_id: int | None = None
        self.checkpoint_enabled = bool(
            web_cfg.get("session_checkpoint_enabled", True)
            if resume_checkpoint is None
            else resume_checkpoint
        )
        self.clear_resumed_plan = bool(
            web_cfg.get("clear_resumed_plan", True)
            if clear_resumed_plan is None
            else clear_resumed_plan
        )
        configured_checkpoint_path = (
            checkpoint_path
            if checkpoint_path is not None
            else web_cfg.get("session_checkpoint_path")
        )
        self.checkpoint_path = (
            normalize_runtime_path(configured_checkpoint_path)
            if configured_checkpoint_path
            else _default_checkpoint_path(thread_id)
        )
        self.checkpoint_loaded = False
        self.checkpoint_loaded_at = ""
        self.checkpoint_saved_at = ""
        self.checkpoint_error: str | None = None
        self.checkpoint_runtime_cleared = False
        self.restored_checkpoint_active = False
        self.checkpoint_prompt_on_start = bool(
            web_cfg.get("session_checkpoint_prompt_on_start", True)
        )
        self.checkpoint_pending_payload: dict[str, Any] | None = None
        self.checkpoint_choice_required = False
        self.checkpoint_mode = "off" if not self.checkpoint_enabled else "new"
        self.checkpoint_available = False
        self.checkpoint_archived_path = ""
        self.run_memory_restored = False
        self.run_memory_restore_summary: dict[str, Any] = {}
        self.run_memory_restore_error: str | None = None
        live_state = self.agent.get_thread_state(thread_id) or {}
        self.state_cache: dict[str, Any] = live_state
        self.max_log_lines = 28
        workspace_root = Path((config.get("paths") or {}).get("workspace_root", "/home/car/caragent_ws"))
        self.user_image_dir = workspace_root / "perception_outputs" / "agent_user_images"
        self.agent_capture_dir = workspace_root / "perception_outputs" / "agent_captures"
        checkpoint_payload = self._load_checkpoint_payload()
        if checkpoint_payload is not None:
            checkpoint_has_resume_content = _checkpoint_payload_has_resume_content(
                checkpoint_payload
            )
            self.checkpoint_available = checkpoint_has_resume_content
            self.checkpoint_saved_at = str(
                checkpoint_payload.get("saved_at")
                or checkpoint_payload.get("updated_at")
                or ""
            )
            if (
                checkpoint_has_resume_content
                and
                self.checkpoint_prompt_on_start
                and not _state_has_thread_content(live_state)
            ):
                self.checkpoint_pending_payload = checkpoint_payload
                self.checkpoint_choice_required = True
                self.checkpoint_mode = "pending"
            elif checkpoint_has_resume_content:
                self._apply_checkpoint_payload(checkpoint_payload, live_state=live_state)
        if (
            self.checkpoint_enabled
            and self.checkpoint_prompt_on_start
            and not self.checkpoint_choice_required
            and not self.checkpoint_loaded
            and not _state_has_thread_content(live_state)
            and self._available_run_memory_sessions(limit=1)
        ):
            self.checkpoint_available = True
            self.checkpoint_choice_required = True
            self.checkpoint_mode = "pending"
        if hasattr(self.agent, "add_controller_arrival_turn_listener"):
            self.agent.add_controller_arrival_turn_listener(
                self._handle_background_controller_arrival_turn
            )

    def _load_checkpoint_payload(self) -> dict[str, Any] | None:
        """Load the persisted Web UI checkpoint when one exists."""

        if not self.checkpoint_enabled:
            return None
        if not self.checkpoint_path.exists() or not self.checkpoint_path.is_file():
            return None
        try:
            payload = json.loads(self.checkpoint_path.read_text(encoding="utf-8"))
        except Exception as exc:
            self.checkpoint_error = f"Failed to load session checkpoint: {exc}"
            return None
        if not isinstance(payload, dict):
            self.checkpoint_error = "Session checkpoint payload is not an object."
            return None
        return payload

    def _apply_checkpoint_payload(
        self,
        payload: dict[str, Any],
        *,
        live_state: dict[str, Any],
    ) -> None:
        """Restore visible session data without reviving interrupted work."""

        if str(payload.get("thread_id") or "") not in {"", self.thread_id}:
            return

        self.checkpoint_loaded = True
        self.checkpoint_available = True
        self.checkpoint_pending_payload = None
        self.checkpoint_choice_required = False
        self.checkpoint_mode = "resumed"
        self.checkpoint_loaded_at = _now_display()
        self.checkpoint_saved_at = str(payload.get("saved_at") or payload.get("updated_at") or "")
        self.turn_history = _normalize_checkpoint_turns(payload.get("turn_history"))
        if self.turn_history:
            self.turn_counter = max(
                int(turn.get("turn_id") or 0)
                for turn in self.turn_history
                if isinstance(turn, dict)
            )
        self.last_input_language = normalize_language(
            payload.get("input_language") or self.last_input_language,
            fallback="zh",
        )
        self.last_output_language = normalize_language(
            payload.get("output_language") or self.last_output_language,
            fallback="zh",
        )

        if _state_has_thread_content(live_state):
            self.state_cache = live_state
            self.restored_checkpoint_active = False
            self.checkpoint_runtime_cleared = False
            return

        checkpoint_state = _normalize_checkpoint_state(
            payload.get("state_cache") or payload.get("state") or {}
        )
        if self.clear_resumed_plan and _state_has_active_plan_residue(checkpoint_state):
            checkpoint_state = _clear_resumed_runtime_state(checkpoint_state)
            self.checkpoint_runtime_cleared = True
        self.state_cache = checkpoint_state
        self._restore_run_memory_from_checkpoint(payload)
        self.restored_checkpoint_active = bool(self.turn_history or checkpoint_state)

    def _checkpoint_view(self) -> dict[str, Any]:
        """Return checkpoint metadata for the browser payload."""

        return {
            "enabled": self.checkpoint_enabled,
            "path": str(self.checkpoint_path) if self.checkpoint_enabled else "",
            "available": self.checkpoint_available,
            "choice_required": self.checkpoint_choice_required,
            "mode": self.checkpoint_mode,
            "loaded": self.checkpoint_loaded,
            "loaded_at": self.checkpoint_loaded_at,
            "saved_at": self.checkpoint_saved_at,
            "pending_saved_at": str(
                (self.checkpoint_pending_payload or {}).get("saved_at")
                or (self.checkpoint_pending_payload or {}).get("updated_at")
                or ""
            ),
            "pending_run_memory_snapshot_path": str(
                (self.checkpoint_pending_payload or {}).get("run_memory_snapshot_path")
                or ""
            ),
            "available_sessions": self._available_run_memory_sessions(),
            "archived_path": self.checkpoint_archived_path,
            "restored": self.restored_checkpoint_active,
            "runtime_cleared": self.checkpoint_runtime_cleared,
            "error": self.checkpoint_error,
            "run_memory_restored": self.run_memory_restored,
            "run_memory_restore_summary": self.run_memory_restore_summary,
            "run_memory_restore_error": self.run_memory_restore_error,
        }

    def _run_memory_snapshot_path(self) -> str:
        """Return the current run-memory JSON path when logging is enabled."""

        run_memory = getattr(self.agent, "run_memory", None)
        snapshot_path = getattr(run_memory, "snapshot_path", None)
        return str(snapshot_path) if snapshot_path else ""

    def _available_run_memory_sessions(self, limit: int = 30) -> list[dict[str, Any]]:
        """Return recent persisted run-memory snapshots for the UI resume picker."""

        log_dir = normalize_runtime_path(config.get("log_dir", "logs"))
        if not log_dir.exists():
            return []
        sessions: list[dict[str, Any]] = []
        for path in log_dir.glob("session_*/run_memory.json"):
            if not path.is_file():
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(data, dict):
                    continue
                if not (
                    data.get("turns")
                    or data.get("plans")
                    or data.get("task_results")
                    or data.get("threads")
                ):
                    continue
                stat = path.stat()
                session = data.get("session") if isinstance(data.get("session"), dict) else {}
                session_id = str(session.get("session_id") or path.parent.name)
                plan_count = len(list(data.get("plans") or []))
                task_result_count = len(list(data.get("task_results") or []))
                turn_count = len(list(data.get("turns") or []))
                updated = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
                label = (
                    f"{session_id} | {updated} | "
                    f"plans={plan_count} tasks={task_result_count} turns={turn_count}"
                )
                sessions.append(
                    {
                        "path": str(path),
                        "label": label,
                        "session_id": session_id,
                        "updated_at": updated,
                        "plan_count": plan_count,
                        "task_result_count": task_result_count,
                        "turn_count": turn_count,
                        "mtime": stat.st_mtime,
                    }
                )
            except Exception:
                continue
        sessions.sort(key=lambda item: float(item.get("mtime") or 0.0), reverse=True)
        return sessions[:limit]

    def _restore_run_memory_from_checkpoint(self, payload: dict[str, Any]) -> None:
        """Attach the previous run-memory snapshot to this fresh Agent instance."""

        source_path = str(payload.get("run_memory_snapshot_path") or "").strip()
        if not source_path:
            return

        current_snapshot_path = self._run_memory_snapshot_path()
        if current_snapshot_path and Path(source_path) == Path(current_snapshot_path):
            return

        run_memory = getattr(self.agent, "run_memory", None)
        restore_from_snapshot = getattr(run_memory, "restore_from_snapshot", None)
        if not callable(restore_from_snapshot):
            self.run_memory_restore_error = "Current Agent run_memory cannot restore snapshots."
            return

        try:
            self.run_memory_restore_summary = restore_from_snapshot(
                source_path,
                thread_id=self.thread_id,
            )
            self.run_memory_restored = True
            self.run_memory_restore_error = None
        except Exception as exc:
            self.run_memory_restore_error = f"Failed to restore run memory: {exc}"

    def _save_checkpoint_locked(self, state: dict[str, Any] | None = None) -> None:
        """Persist the current visible session snapshot to disk."""

        if not self.checkpoint_enabled:
            return
        if self.checkpoint_choice_required:
            return

        now = _now_display()
        payload = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "thread_id": self.thread_id,
            "session_mode": self.checkpoint_mode,
            "saved_at": now,
            "updated_at": self.updated_at,
            "input_language": self.last_input_language,
            "output_language": self.last_output_language,
            "turn_counter": self.turn_counter,
            "turn_history": _json_safe(_normalize_checkpoint_turns(self.turn_history)),
            "state_cache": _state_for_checkpoint(state or self.state_cache),
            "run_memory_snapshot_path": self._run_memory_snapshot_path(),
            "latest_error": self.latest_error,
        }

        try:
            self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            temp_path = self.checkpoint_path.with_name(
                f"{self.checkpoint_path.stem}_{time.time_ns()}.tmp"
            )
            temp_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            temp_path.replace(self.checkpoint_path)
            self.checkpoint_saved_at = now
            self.checkpoint_error = None
        except Exception as exc:
            self.checkpoint_error = f"Failed to save session checkpoint: {exc}"

    def _drop_restored_runtime_snapshot_locked(self) -> None:
        """Discard restored display-only state before starting a fresh real turn."""

        if not self.restored_checkpoint_active:
            return
        live_state = self.agent.get_thread_state(self.thread_id) or {}
        self.state_cache = live_state if _state_has_thread_content(live_state) else {}
        self.restored_checkpoint_active = False
        self.checkpoint_runtime_cleared = False

    def _archive_checkpoint_file_locked(self) -> str:
        """Move the saved checkpoint out of the default load path."""

        if not self.checkpoint_path.exists():
            return ""
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archived = self.checkpoint_path.with_name(
            f"{self.checkpoint_path.stem}.archived_{stamp}{self.checkpoint_path.suffix}"
        )
        self.checkpoint_path.replace(archived)
        return str(archived)

    def _start_new_session_locked(self) -> None:
        """Start from an empty visible session and archive any pending checkpoint."""

        try:
            self.checkpoint_archived_path = self._archive_checkpoint_file_locked()
        except Exception as exc:
            self.checkpoint_error = f"Failed to archive checkpoint: {exc}"
            self.checkpoint_archived_path = ""
        self.checkpoint_pending_payload = None
        self.checkpoint_choice_required = False
        self.checkpoint_loaded = False
        self.checkpoint_loaded_at = ""
        self.checkpoint_available = False
        self.checkpoint_mode = "new" if self.checkpoint_enabled else "off"
        self.checkpoint_runtime_cleared = False
        self.restored_checkpoint_active = False
        self.run_memory_restored = False
        self.run_memory_restore_summary = {}
        self.run_memory_restore_error = None
        self.turn_history = []
        self.turn_counter = 0
        self.processing_turn_id = None
        self.controller_arrival_turn_id = None
        self.latest_error = None
        live_state = self.agent.get_thread_state(self.thread_id) or {}
        self.state_cache = live_state if _state_has_thread_content(live_state) else {}
        self.updated_at = _now_display()

    def choose_session_mode(
        self,
        mode: str,
        *,
        run_memory_snapshot_path: str | None = None,
    ) -> dict[str, Any]:
        """Apply the browser's startup session choice."""

        clean_mode = str(mode or "").strip().lower()
        with self.lock:
            if self.processing_turn_id is not None:
                self.latest_error = "A turn is already in progress. Wait before changing session mode."
                return self._build_payload_locked()

            if clean_mode in {"resume", "restore"}:
                selected_snapshot = str(run_memory_snapshot_path or "").strip()
                payload = None
                if selected_snapshot:
                    try:
                        payload = _checkpoint_payload_from_run_memory_snapshot(
                            Path(selected_snapshot).expanduser(),
                            thread_id=self.thread_id,
                        )
                    except Exception as exc:
                        self.latest_error = f"Failed to load selected session: {exc}"
                        return self._build_payload_locked()
                if payload is None:
                    payload = self.checkpoint_pending_payload or self._load_checkpoint_payload()
                if payload is None:
                    self.latest_error = "No saved session checkpoint is available."
                    self.checkpoint_choice_required = False
                    self.checkpoint_mode = "new" if self.checkpoint_enabled else "off"
                    return self._build_payload_locked()
                live_state = self.agent.get_thread_state(self.thread_id) or {}
                self._apply_checkpoint_payload(payload, live_state=live_state)
                self.latest_error = None
                self.updated_at = _now_display()
                return self._build_payload_locked()

            if clean_mode in {"new", "fresh", "reset"}:
                self._start_new_session_locked()
                return self._build_payload_locked()

            self.latest_error = f"Unknown session mode: {mode}"
            return self._build_payload_locked()

    def _build_task_view(self, state: dict[str, Any]) -> list[dict[str, Any]]:
        """Convert raw task records into ordered task cards for the UI."""

        tasks = state.get("tasks", {}) or {}
        current_plan_id = state.get("current_plan_id")
        current_task_id = state.get("current_task_id")
        ordered_task_ids = collect_ordered_task_ids_for_plan(
            tasks,
            plan_id=current_plan_id,
        )
        if not ordered_task_ids:
            ordered_task_ids = collect_ordered_task_ids_for_plan(tasks, plan_id=None)

        remaining_task_ids = [task_id for task_id in sorted(tasks) if task_id not in ordered_task_ids]
        visible_task_ids = ordered_task_ids + remaining_task_ids
        ordered_index = {
            task_id: index + 1 for index, task_id in enumerate(ordered_task_ids)
        }
        ordered_total = len(ordered_task_ids)
        task_cards: list[dict[str, Any]] = []

        for task_id in visible_task_ids:
            task = tasks[task_id]
            latest_result = (task.get("result") or [])[-1] if task.get("result") else None
            sequence_label = None
            if task_id in ordered_index and ordered_total > 0:
                sequence_label = f"step {ordered_index[task_id]}/{ordered_total}"

            task_view = {
                "task_id": task_id,
                "title": f"Task #{task_id}: {task.get('description', '')}",
                "type": task.get("type", "action"),
                "status": task.get("status", "pending"),
                "is_current": task_id == current_task_id,
                "is_inserted": bool(task.get("inserted")),
                "wait_for_event": task.get("wait_for_event"),
                "plan_id": task.get("plan_id"),
                "next_task_id": task.get("next_task_id"),
                "depends_on": task.get("depends_on"),
                "latest_result_summary": latest_result.get("summary") if latest_result else None,
                "object_approach_debug": _extract_object_approach_debug(latest_result),
                "terminal_reason": task.get("terminal_reason"),
                "sequence_label": sequence_label,
            }
            task_view["detail"] = _normalize_task_detail(task_view)
            task_cards.append(task_view)

        return task_cards

    def _build_plan_graph_view(self, state: dict[str, Any]) -> dict[str, Any]:
        """Build a read-only PlanGraph payload for the UI graph tab."""

        tasks = state.get("tasks", {}) or {}
        current_plan_id = state.get("current_plan_id")
        current_task_id = state.get("current_task_id")
        ordered_task_ids = collect_ordered_task_ids_for_plan(
            tasks,
            plan_id=current_plan_id,
        )
        plan_id = current_plan_id if ordered_task_ids else None
        if not ordered_task_ids:
            ordered_task_ids = collect_ordered_task_ids_for_plan(tasks, plan_id=None)

        try:
            summary = summarize_plan_graph(tasks, plan_id=plan_id)
            edges = iter_plan_edges(tasks, plan_id=plan_id)
            issues = validate_plan_graph(tasks, plan_id=plan_id)
        except Exception as exc:
            return {
                "summary": {
                    "plan_id": plan_id,
                    "node_count": 0,
                    "edge_count": 0,
                    "root_task_ids": [],
                    "leaf_task_ids": [],
                    "is_dag": False,
                },
                "nodes": [],
                "edges": [],
                "issues": [
                    {
                        "severity": "error",
                        "code": "plan_graph_build_failed",
                        "task_id": -1,
                        "message": f"Failed to build PlanGraph view: {exc}",
                        "details": {},
                    }
                ],
            }

        root_ids = set(summary.get("root_task_ids") or [])
        leaf_ids = set(summary.get("leaf_task_ids") or [])
        scoped_task_ids = [
            task_id
            for task_id in ordered_task_ids
            if task_id in tasks and (plan_id is None or tasks[task_id].get("plan_id") == plan_id)
        ]
        nodes = [
            {
                "task_id": task_id,
                "description": tasks[task_id].get("description", ""),
                "type": tasks[task_id].get("type", "action"),
                "status": tasks[task_id].get("status", "pending"),
                "is_current": task_id == current_task_id,
                "is_root": task_id in root_ids,
                "is_leaf": task_id in leaf_ids,
                "plan_id": tasks[task_id].get("plan_id"),
                "next_task_id": tasks[task_id].get("next_task_id"),
                "depends_on": tasks[task_id].get("depends_on") or [],
                "branches": tasks[task_id].get("branches") or {},
                "wait_for_event": tasks[task_id].get("wait_for_event"),
                "latest_result_summary": (
                    (tasks[task_id].get("result") or [])[-1].get("summary")
                    if tasks[task_id].get("result")
                    else None
                ),
            }
            for task_id in scoped_task_ids
        ]

        return {
            "summary": summary,
            "nodes": nodes,
            "edges": edges,
            "issues": issues,
        }

    def _build_event_view(self, state: dict[str, Any]) -> list[dict[str, Any]]:
        """Convert recent structured events into short UI-friendly event cards."""

        event_views: list[dict[str, Any]] = []
        for event in list(state.get("events", []))[-3:]:
            payload = event.get("payload", {}) or {}
            event_views.append(
                {
                    "event_id": event.get("event_id"),
                    "type": event.get("type"),
                    "source": event.get("source"),
                    "task_id": event.get("task_id"),
                    "created_at": event.get("created_at"),
                    "summary": payload.get("summary") or payload.get("content") or "",
                }
            )
        return event_views

    def _read_log_tail(self, file_path: Path, max_lines: int) -> str:
        """Read the latest lines from one log file without loading excessive history."""

        if not file_path.exists() or not file_path.is_file():
            return ""

        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as handle:
                lines = deque(handle, maxlen=max_lines)
        except Exception:
            return ""

        return "".join(lines).strip()

    def _read_log_tail_entries(self, file_path: Path, max_lines: int) -> list[str]:
        """Read the latest grouped log entries from one log file."""

        text = self._read_log_tail(file_path, max_lines)
        if not text:
            return []

        entries: list[str] = []
        current_lines: list[str] = []
        for raw_line in text.splitlines():
            line = raw_line.rstrip()
            if LOG_ENTRY_START_RE.match(line):
                if current_lines:
                    entry = "\n".join(current_lines).strip()
                    if entry:
                        entries.append(entry)
                current_lines = [line]
                continue

            if not current_lines:
                continue

            if not line.strip():
                current_lines.append("")
            else:
                current_lines.append(line)

        if current_lines:
            entry = "\n".join(current_lines).strip()
            if entry:
                entries.append(entry)

        return entries

    def _build_console_entries(self) -> list[str]:
        """Build merged console entries while preserving multi-line log blocks."""

        logger = getattr(self.agent, "logger", None)
        if logger is None or not hasattr(logger, "get_session_dir"):
            return []

        session_dir_raw = logger.get_session_dir()
        if not session_dir_raw:
            return []

        session_dir = Path(session_dir_raw)
        log_specs = [
            ("foreground", session_dir / "foreground_workflow_agents.log"),
            ("background", session_dir / "background_agents.log"),
            ("physical", session_dir / "physical_layer.log"),
        ]

        merged_entries: list[tuple[str, int, str]] = []
        sequence = 0
        for _, file_path in log_specs:
            for entry in self._read_log_tail_entries(file_path, self.max_log_lines * 6):
                first_line = entry.splitlines()[0] if entry.splitlines() else ""
                timestamp_key = first_line[:32]
                merged_entries.append((timestamp_key, sequence, entry))
                sequence += 1

        if not merged_entries:
            return []

        merged_entries.sort(key=lambda item: (item[0], item[1]))
        return [entry for _, _, entry in merged_entries[-120:]]

    def _build_console_output(self) -> str:
        """Build one merged console-style stream from the current logger session files."""

        return "\n\n".join(self._build_console_entries())

    def _build_input_window_view(self, state: dict[str, Any]) -> dict[str, Any]:
        """Describe whether the composer should be open, locked, or waiting-focused."""

        next_action = state.get("next_action", {}) or {}
        next_action_type = next_action.get("type", "idle")
        current_task_id = state.get("current_task_id")
        tasks = state.get("tasks", {}) or {}
        current_task = tasks.get(current_task_id) if current_task_id in tasks else None

        if self.processing_turn_id is not None:
            return {
                "locked": True,
                "mode": "busy",
                "title": "正在处理",
                "detail": "我正在理解并执行你的指令，完成当前步骤后会继续接收输入。",
            }

        if current_task and current_task.get("status") == "waiting":
            wait_for_event = current_task.get("wait_for_event") or "external_event"
            if wait_for_event == "navigation_arrived":
                detail = "正在导航，等待小车到达。"
            else:
                detail = "正在等待外部事件。"
            return {
                "locked": False,
                "mode": "waiting",
                "title": "正在等待",
                "detail": detail,
            }

        if next_action_type in {"plan", "execute"}:
            return {
                "locked": True,
                "mode": "settling",
                "title": "正在切换步骤",
                "detail": "我正在进入下一步，很快可以继续输入。",
            }

        if state.get("current_plan_id"):
            return {
                "locked": False,
                "mode": "active_plan",
                "title": "任务进行中",
                "detail": "可以继续补充指令，我会根据当前任务调整后续安排。",
            }

        return {
            "locked": False,
            "mode": "ready",
            "title": "可以输入",
            "detail": "可以发送新的指令，控制器到达事件会自动处理。",
        }

    def _get_state_snapshot_locked(
        self,
        state_override: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return the safest currently-available state snapshot for payload rendering."""

        if state_override is not None:
            self.state_cache = state_override
            return state_override

        if self.restored_checkpoint_active:
            return dict(self.state_cache)

        if self.processing_turn_id is not None:
            return dict(self.state_cache)

        state = self.agent.get_thread_state(self.thread_id) or {}
        self.state_cache = state
        return state

    def _find_turn_locked(self, turn_id: int) -> dict[str, Any] | None:
        """Locate one stored turn entry by id while the caller already holds the app lock."""

        for turn in reversed(self.turn_history):
            if turn.get("turn_id") == turn_id:
                return turn
        return None

    def _merge_response_items(
        self,
        existing_items: Any,
        new_items: Any,
    ) -> list[dict[str, Any]]:
        """Append only previously unseen response items while preserving order."""

        merged_items = normalize_turn_response_items(existing_items)
        incoming_items = normalize_turn_response_items(new_items)
        seen_response_ids = {
            str(item.get("response_id") or "").strip()
            for item in merged_items
            if str(item.get("response_id") or "").strip()
        }
        seen_response_keys = {
            (
                str(item.get("response_type") or "").strip(),
                str(item.get("response_text") or "").strip(),
            )
            for item in merged_items
        }
        for item in incoming_items:
            response_id = str(item.get("response_id") or "").strip()
            response_key = (
                str(item.get("response_type") or "").strip(),
                str(item.get("response_text") or "").strip(),
            )
            if response_id and response_id in seen_response_ids:
                continue
            if response_key in seen_response_keys:
                continue
            merged_items.append(item)
            if response_id:
                seen_response_ids.add(response_id)
            seen_response_keys.add(response_key)
        return merged_items

    def _apply_stream_update_locked(self, turn_id: int, update: dict[str, Any]) -> None:
        """Apply one streamed node update to cached UI state for live refresh."""

        turn = self._find_turn_locked(turn_id)
        if turn is None:
            return

        node_state = update.get("node_state")
        persisted_state = update.get("state")
        merged_state = dict(self.state_cache)
        if isinstance(node_state, dict):
            merged_state.update(node_state)
        if isinstance(persisted_state, dict) and persisted_state:
            merged_state.update(persisted_state)
        if merged_state:
            self.state_cache = merged_state

        step_summary = update.get("step_summary")
        if isinstance(step_summary, dict):
            step_trace = list(turn.get("step_trace", []))
            step_trace.append(step_summary)
            turn["step_trace"] = step_trace[-24:]
            response_items = normalize_turn_response_items(
                step_summary.get("turn_response_items")
            )
            if response_items:
                existing_items = self._merge_response_items(
                    turn.get("response_items"),
                    response_items,
                )
                turn["response_items"] = existing_items
                turn["response"] = str(
                    existing_items[-1].get("response_text") or ""
                )

        turn["visited_nodes"] = list(update.get("visited_nodes", []) or [])
        turn["status"] = "running"
        turn["live_node"] = str(update.get("node_name") or "")
        turn["updated_at"] = _now_display()
        self.updated_at = _now_display()
        self._save_checkpoint_locked(self.state_cache)

    def _finalize_turn_locked(
        self,
        turn_id: int,
        turn_result: dict[str, Any],
    ) -> None:
        """Store one completed turn result and release the composer lock."""

        turn = self._find_turn_locked(turn_id)
        if turn is None:
            self.processing_turn_id = None
            self.state_cache = turn_result.get("state", {}) or self.state_cache
            self.updated_at = _now_display()
            self._save_checkpoint_locked(self.state_cache)
            return

        response_text = str(turn_result.get("turn_response_text") or "").strip()
        final_response_items = normalize_turn_response_items(
            turn_result.get("response_items", [])
        )
        if turn_result.get("language_adapted"):
            response_items = final_response_items
        else:
            response_items = self._merge_response_items(
                turn.get("response_items"),
                final_response_items,
            )
        if not response_text:
            response_text = str(
                (turn_result.get("state", {}) or {}).get("user_facing_response") or ""
            ).strip()
        if not response_text and response_items:
            response_text = str(response_items[-1].get("response_text") or "").strip()

        turn["response"] = response_text
        turn["response_items"] = response_items
        turn["turn_response_type"] = str(turn_result.get("turn_response_type") or "")
        turn["output_language"] = str(turn_result.get("output_language") or "")
        if turn_result.get("agent_message") and turn_result.get("agent_message") != turn.get("content"):
            turn["agent_message"] = str(turn_result.get("agent_message") or "")
        turn["status"] = "completed"
        turn["visited_nodes"] = list(turn_result.get("visited_nodes", []) or [])
        turn["step_trace"] = list(turn_result.get("step_trace", []) or [])[-24:]
        turn["updated_at"] = _now_display()
        turn["finished_at"] = _now_display()
        turn["saw_plan_node"] = bool(turn_result.get("saw_plan_node"))
        turn["saw_navigation_activity"] = bool(
            turn_result.get("saw_navigation_activity")
        )
        self.state_cache = turn_result.get("state", {}) or self.state_cache
        self.processing_turn_id = None
        self.latest_error = None
        self.updated_at = _now_display()
        self._save_checkpoint_locked(self.state_cache)

    def _adapt_controller_turn_result_language(
        self,
        turn_result: dict[str, Any],
    ) -> dict[str, Any]:
        """Use the latest UI language preference for background arrival turns."""

        output_language = self.last_output_language or str(
            self.io_cfg.get("output_language", "zh")
        )
        output_language = normalize_language(output_language, fallback="zh")
        input_language = normalize_language(self.last_input_language or "zh", fallback="zh")
        return adapt_turn_result_language(
            turn_result,
            output_language=output_language,
            original_input_language=input_language,
        )

    def _append_completed_controller_arrival_turn_locked(
        self,
        turn_result: dict[str, Any],
    ) -> None:
        """Append a completed background controller arrival turn to web history."""

        if turn_result.get("thread_id") != self.thread_id:
            return
        turn_result = self._adapt_controller_turn_result_language(turn_result)

        turn_id = self.controller_arrival_turn_id
        turn = self._find_turn_locked(turn_id) if turn_id is not None else None
        if turn is None:
            self.turn_counter += 1
            turn_id = self.turn_counter
            turn = {
                "turn_id": turn_id,
                "role": "system",
                "role_label": "Controller Arrival",
                "source": "controller-watchdog",
                "content": str(turn_result.get("message") or ""),
                "response": "",
                "response_items": [],
                "created_at": _now_display(),
                "updated_at": _now_display(),
                "visited_nodes": [],
                "step_trace": [],
                "status": "running",
                "live_node": "",
            }
            self.turn_history.append(turn)

        final_response_items = normalize_turn_response_items(
            turn_result.get("response_items", [])
        )
        if turn_result.get("language_adapted"):
            response_items = final_response_items
        else:
            response_items = self._merge_response_items(
                turn.get("response_items"),
                final_response_items,
            )
        response_text = str(turn_result.get("turn_response_text") or "").strip()
        if not response_text and response_items:
            response_text = str(response_items[-1].get("response_text") or "").strip()
        turn["response"] = response_text
        turn["response_items"] = response_items
        turn["content"] = str(turn_result.get("message") or turn.get("content") or "")
        turn["updated_at"] = _now_display()
        turn["finished_at"] = _now_display()
        turn["visited_nodes"] = list(turn_result.get("visited_nodes", []) or [])
        turn["step_trace"] = list(turn_result.get("step_trace", []) or [])[-24:]
        turn["status"] = "completed"
        turn["live_node"] = ""
        turn["turn_response_type"] = str(turn_result.get("turn_response_type") or "")
        turn["output_language"] = str(turn_result.get("output_language") or "")
        turn["saw_plan_node"] = bool(turn_result.get("saw_plan_node"))
        turn["saw_navigation_activity"] = bool(
            turn_result.get("saw_navigation_activity")
        )
        self.turn_history = self.turn_history[-80:]
        self.state_cache = turn_result.get("state", {}) or self.state_cache
        self.controller_arrival_turn_id = None
        self.latest_error = None
        self.updated_at = _now_display()
        self._save_checkpoint_locked(self.state_cache)

    def _apply_controller_arrival_update_locked(self, update: dict[str, Any]) -> None:
        """Create/update the visible background controller turn while it is running."""

        if update.get("thread_id") != self.thread_id:
            return
        update = self._adapt_controller_turn_result_language(update)

        turn_id = self.controller_arrival_turn_id
        turn = self._find_turn_locked(turn_id) if turn_id is not None else None
        if turn is None:
            self.turn_counter += 1
            turn_id = self.turn_counter
            self.controller_arrival_turn_id = turn_id
            self.turn_history.append(
                {
                    "turn_id": turn_id,
                    "role": "system",
                    "role_label": "Controller Arrival",
                    "source": "controller-watchdog",
                    "content": str(update.get("message") or ""),
                    "response": "",
                    "response_items": [],
                    "created_at": _now_display(),
                    "updated_at": _now_display(),
                    "visited_nodes": [],
                    "step_trace": [],
                    "status": "running",
                    "live_node": "queued",
                }
            )
            self.turn_history = self.turn_history[-80:]

        self._apply_stream_update_locked(turn_id, update)

    def _handle_background_controller_arrival_turn(
        self,
        turn_result: dict[str, Any],
    ) -> None:
        """Receive physical arrivals dispatched outside a web-submitted turn."""

        if not turn_result.get("dispatched"):
            return
        with self.lock:
            if turn_result.get("controller_arrival_update"):
                self._apply_controller_arrival_update_locked(turn_result)
            else:
                self._append_completed_controller_arrival_turn_locked(turn_result)

    def _fail_turn_locked(self, turn_id: int, error_text: str) -> None:
        """Mark one pending turn as failed and surface the error to the UI."""

        turn = self._find_turn_locked(turn_id)
        if turn is not None:
            turn["status"] = "failed"
            turn["error"] = error_text
            turn["updated_at"] = _now_display()
            turn["finished_at"] = _now_display()
        self.processing_turn_id = None
        self.latest_error = error_text
        self.updated_at = _now_display()
        self._save_checkpoint_locked(self.state_cache)

    def _run_turn_worker(
        self,
        turn_id: int,
        message: str,
        role: str,
        source: str,
        input_language: str | None = None,
        output_language: str | None = None,
        image_data_url: str | None = None,
    ) -> None:
        """Process one message turn off-thread while publishing live state updates."""

        try:
            configured_input_language = normalize_language(
                input_language,
                fallback=self.last_input_language or self.io_cfg.get("input_language", "zh"),
            )
            requested_output_language = normalize_language(
                output_language or self.io_cfg.get("output_language", "zh"),
                fallback="zh",
            )
            agent_message = prepare_user_message_for_agent(
                message,
                input_language=configured_input_language,
                output_language=requested_output_language,
                translate_boundary=bool(self.io_cfg.get("translate_boundary", True)),
            )
            if image_data_url:
                image_ref = self._save_attached_image(
                    turn_id=turn_id,
                    image_data_url=image_data_url,
                    source=source,
                )
                agent_message = self._compose_agent_message_with_image_reference(
                    agent_message,
                    image_ref=image_ref,
                )
            turn_result = self.agent.run_message_turn(
                agent_message,
                self.thread_id,
                role,
                on_update=lambda update: self._handle_turn_update(turn_id, update),
                original_message=message,
            )
            turn_result["original_message"] = message
            turn_result["agent_message"] = agent_message
            if image_data_url:
                turn_result["image_attached"] = True
            turn_result = adapt_turn_result_language(
                turn_result,
                output_language=requested_output_language,
                original_input_language=configured_input_language,
            )
            with self.lock:
                self._finalize_turn_locked(turn_id, turn_result)
        except Exception as exc:
            with self.lock:
                self._fail_turn_locked(turn_id, str(exc))

    def _save_attached_image(
        self,
        *,
        turn_id: int,
        image_data_url: str,
        source: str,
    ) -> dict[str, Any]:
        """Persist a browser image payload and return non-visual metadata."""

        image = image_from_data_url(image_data_url)
        self.user_image_dir.mkdir(parents=True, exist_ok=True)
        image_ref_id = f"turn_{turn_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        image_path = self.user_image_dir / f"{image_ref_id}.jpg"
        image.convert("RGB").save(image_path, format="JPEG", quality=92)
        return {
            "image_ref_id": image_ref_id,
            "path": str(image_path),
            "source": source,
            "created_at": _now_display(),
        }

    def _compose_agent_message_with_image_reference(
        self,
        message: str,
        *,
        image_ref: dict[str, Any],
    ) -> str:
        """Append structured attached-image metadata to the agent message."""

        clean_message = str(message or "").strip()
        parts = []
        if clean_message:
            parts.append(f"User text request:\n{clean_message}")
        else:
            parts.append("User text request:\nUse the attached target image as the navigation/query reference.")
        parts.append("[ATTACHED_IMAGES_JSON]")
        parts.append(json.dumps([image_ref], ensure_ascii=False, indent=2))
        parts.append("[/ATTACHED_IMAGES_JSON]")
        parts.append(
            "Instruction: the attached image metadata above is available to the planner and executor. "
            "Only tasks that must inspect the image should set image_refs:[\"latest\"]. "
            "For image-only navigation, resolve the closest matching keyframe or image-contained object before navigating."
        )
        return "\n\n".join(parts)

    def _handle_turn_update(self, turn_id: int, update: dict[str, Any]) -> None:
        """Receive one streamed node update from AsyncAgent and cache it for polling."""

        with self.lock:
            self._apply_stream_update_locked(turn_id, update)

    def _build_state_view(self, state: dict[str, Any]) -> dict[str, Any]:
        """Build the aggregated state view consumed by the browser dashboard."""

        tasks = self._build_task_view(state)
        progress = get_task_progress_context(
            state.get("tasks", {}) or {},
            current_task_id=state.get("current_task_id"),
            current_plan_id=state.get("current_plan_id"),
        )
        current_task_label = None
        if progress is not None:
            current_task_label = (
                f"Task #{progress['task_id']} · step {progress['position']}/{progress['total']}"
            )
        elif state.get("current_task_id") is not None:
            current_task_label = f"Task #{state['current_task_id']}"

        if progress is not None:
            current_task_label = (
                f"Task #{progress['task_id']} | step {progress['position']}/{progress['total']}"
            )

        next_action = state.get("next_action", {}) or {}
        input_window = self._build_input_window_view(state)
        plan_graph = self._build_plan_graph_view(state)
        return {
            "current_plan_id": state.get("current_plan_id"),
            "current_task_id": state.get("current_task_id"),
            "current_task_label": current_task_label,
            "next_action": next_action,
            "next_action_type": next_action.get("type", "idle"),
            "processing": self.processing_turn_id is not None,
            "processing_turn_id": self.processing_turn_id,
            "input_window": input_window,
            "agent_status": input_window["title"],
            "agent_status_detail": input_window["detail"],
            "user_facing_response": state.get("user_facing_response"),
            "turn_response_items": state.get("turn_response_items", []),
            "guidance_events": list(state.get("guidance_events") or [])[-80:],
            "interaction_profile": get_interaction_profile(),
            "visible_task_count": len(tasks),
            "tasks": tasks,
            "plan_graph": plan_graph,
            "events": self._build_event_view(state),
        }

    def _filter_user_turn_response_items(
        self,
        turn: dict[str, Any],
        *,
        lite: bool = False,
    ) -> list[dict[str, Any]]:
        """Keep user-facing answers focused on final replies, not every progress tick."""

        if str(turn.get("status") or "") != "completed":
            return []

        response_items = normalize_turn_response_items(turn.get("response_items"))
        if not response_items:
            return []

        if lite:
            original_items = list(response_items)
            guidance_backed_events = {
                "plan_created",
                "plan_edited",
                "plan_updated",
                "task_waiting",
                "navigation_arrived",
                "task_failed",
                "task_cancelled",
            }
            response_items = [
                item
                for item in response_items
                if str(item.get("source_event_type") or "").strip()
                not in guidance_backed_events
            ]
            if not response_items:
                fallback_text = str(turn.get("response") or "").strip()
                if fallback_text:
                    if not any(
                        str(item.get("response_text") or "").strip() == fallback_text
                        for item in original_items
                    ):
                        return [
                            {
                                "response_id": f"{turn.get('turn_id')}:final",
                                "response_type": str(
                                    turn.get("turn_response_type") or "result"
                                ),
                                "response_text": fallback_text,
                            }
                        ]

        return list(response_items)

    def _build_route_completion_notice(
        self,
        turn: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Surface controller-arrival responses as a visible turn when the plan is done."""

        state = turn.get("state")
        if not isinstance(state, dict):
            return None
        if state.get("current_plan_id") or state.get("current_task_id"):
            return None

        response_items = normalize_turn_response_items(turn.get("response_items"))
        if not response_items:
            return None

        headline = str(
            response_items[-1].get("response_text") or ""
        ).strip()
        if not headline:
            return None

        return {
            "turn_id": turn.get("turn_id"),
            "role": "system",
            "role_label": "System",
            "source": "controller-watchdog",
            "content": str(turn.get("content") or ""),
            "response": headline,
            "response_items": [dict(item) for item in response_items],
            "created_at": turn.get("created_at"),
            "updated_at": turn.get("updated_at"),
            "finished_at": turn.get("finished_at"),
            "visited_nodes": list(turn.get("visited_nodes", []) or []),
            "step_trace": list(turn.get("step_trace", []) or [])[-24:],
            "status": "completed",
            "live_node": "",
            "turn_response_type": str(turn.get("turn_response_type") or "result"),
        }

    def _build_lite_controller_result_notice(
        self,
        turn: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Expose real controller-arrival answers in lite while hiding progress chatter."""

        response_type = str(turn.get("turn_response_type") or "").strip()
        if response_type not in {"result", "error"}:
            return None

        response_items = normalize_turn_response_items(turn.get("response_items"))
        result_items = [
            item
            for item in response_items
            if str(item.get("response_type") or "").strip() in {"result", "error"}
            and str(item.get("response_text") or "").strip()
        ]
        response_text = str(turn.get("response") or "").strip()
        if result_items:
            response_text = str(result_items[-1].get("response_text") or "").strip()
        if not response_text:
            return None

        if not result_items:
            result_items = [
                {
                    "response_id": f"{turn.get('turn_id')}:controller-result",
                    "response_type": response_type or "result",
                    "response_text": response_text,
                }
            ]

        return {
            "turn_id": turn.get("turn_id"),
            "role": "system",
            "role_label": "CarAgent 回复",
            "source": str(turn.get("source") or "controller-watchdog"),
            "content": str(turn.get("content") or ""),
            "response": response_text,
            "response_items": [dict(item) for item in result_items],
            "created_at": turn.get("created_at"),
            "updated_at": turn.get("updated_at"),
            "finished_at": turn.get("finished_at"),
            "visited_nodes": list(turn.get("visited_nodes", []) or []),
            "step_trace": list(turn.get("step_trace", []) or [])[-24:],
            "status": "completed",
            "live_node": "",
            "turn_response_type": response_type or "result",
            "output_language": str(turn.get("output_language") or ""),
        }

    def _build_conversation_history(self, *, lite: bool = False) -> list[dict[str, Any]]:
        """Build the clean chat timeline shown in the browser conversation panel."""

        conversation: list[dict[str, Any]] = []

        for turn in self.turn_history[-80:]:
            source = str(turn.get("source") or "")
            role = str(turn.get("role") or "")

            if source in {"controller", "controller-watchdog"}:
                if lite:
                    notice = self._build_lite_controller_result_notice(turn)
                    if notice is not None:
                        conversation.append(notice)
                    continue
                notice = self._build_route_completion_notice(turn)
                if notice is not None:
                    conversation.append(notice)
                elif normalize_turn_response_items(turn.get("response_items")):
                    visible_turn = dict(turn)
                    visible_turn["role_label"] = "System"
                    conversation.append(visible_turn)
                continue

            visible_turn = dict(turn)
            if role == "user":
                response_items = self._filter_user_turn_response_items(turn, lite=lite)
                visible_turn["response_items"] = response_items
                if str(turn.get("status") or "") != "completed":
                    visible_turn["response"] = ""
                elif response_items:
                    visible_turn["response"] = str(
                        response_items[-1].get("response_text") or ""
                    )
                elif lite:
                    visible_turn["response"] = ""
                elif turn.get("response"):
                    visible_turn["response"] = str(turn.get("response") or "")
            conversation.append(visible_turn)

        return conversation[-80:]

    def _build_payload_locked(self, state_override: dict[str, Any] | None = None) -> dict[str, Any]:
        """Build one complete browser payload while the caller already holds the app lock."""

        state = self._get_state_snapshot_locked(state_override=state_override)
        console_entries = self._build_console_entries()
        self.updated_at = _now_display()
        self._save_checkpoint_locked(state)
        return {
            "thread_id": self.thread_id,
            "updated_at": self.updated_at,
            "latest_error": self.latest_error,
            "turn_history": self.turn_history[-80:],
            "conversation_history": self._build_conversation_history(),
            "console_entries": console_entries,
            "console_output": "\n\n".join(console_entries),
            "input_language": self.last_input_language,
            "output_language": self.last_output_language,
            "state": self._build_state_view(state),
            "latest_agent_capture": self._latest_agent_capture_payload(),
            "checkpoint": self._checkpoint_view(),
        }

    def _latest_agent_capture_payload(self) -> dict[str, Any] | None:
        """Return the newest image saved by the agent capture tool."""

        if not self.agent_capture_dir.exists():
            return None
        candidates: list[Path] = []
        for pattern in ("*.jpg", "*.jpeg", "*.png"):
            candidates.extend(self.agent_capture_dir.glob(pattern))
        candidates = [path for path in candidates if path.is_file()]
        if not candidates:
            return None
        latest = max(candidates, key=lambda path: path.stat().st_mtime)
        try:
            return {
                "status": "ok",
                "name": latest.name,
                "path": str(latest),
                "mtime": latest.stat().st_mtime,
                "image_data_url": file_to_data_url(latest),
            }
        except Exception as exc:
            return {
                "status": "error",
                "name": latest.name,
                "path": str(latest),
                "error": str(exc),
                "image_data_url": None,
            }

    def get_payload(self) -> dict[str, Any]:
        """Return the latest UI payload for polling requests."""

        with self.lock:
            return self._build_payload_locked()

    def get_lite_payload(self) -> dict[str, Any]:
        """Return a compact polling payload for the phone-friendly lite UI."""

        with self.lock:
            state = self._get_state_snapshot_locked()
            input_window = self._build_input_window_view(state)
            state_view = {
                "processing": self.processing_turn_id is not None,
                "processing_turn_id": self.processing_turn_id,
                "input_window": input_window,
                "agent_status": input_window["title"],
                "agent_status_detail": input_window["detail"],
                "user_facing_response": state.get("user_facing_response"),
                "guidance_events": list(state.get("guidance_events") or [])[-40:],
                "interaction_profile": get_interaction_profile(),
            }
            self.updated_at = _now_display()
            return {
                "thread_id": self.thread_id,
                "updated_at": self.updated_at,
                "latest_error": self.latest_error,
                "conversation_history": self._build_conversation_history(lite=True),
                "input_language": self.last_input_language,
                "output_language": self.last_output_language,
                "state": state_view,
                "checkpoint": self._checkpoint_view(),
            }

    def get_sim_payload(self) -> dict[str, Any]:
        """Return a compact, read-only snapshot for the simulation map view."""

        with self.lock:
            state = self._get_state_snapshot_locked()
            input_window = self._build_input_window_view(state)
            controller = getattr(self.agent, "controller", None)
            robot_state: dict[str, Any] = {}
            if controller is not None:
                get_current_state = getattr(controller, "get_current_state", None)
                if callable(get_current_state):
                    try:
                        current = get_current_state()
                        if isinstance(current, dict):
                            robot_state = dict(current)
                    except Exception as exc:
                        robot_state = {"status": "controller_state_error", "error": str(exc)}
            scene_memory = getattr(self.agent, "scene_memory", None)
            keyframes: list[dict[str, Any]] = []
            nodes = getattr(scene_memory, "keyframe_nodes", {}) if scene_memory is not None else {}
            if isinstance(nodes, dict):
                for node_id, node in sorted(nodes.items(), key=lambda item: int(item[0])):
                    try:
                        position = getattr(node, "position", None)
                        pos_list = position.astype(float).tolist() if hasattr(position, "astype") else list(position or [])
                        keyframes.append(
                            {
                                "kf_id": int(getattr(node, "kf_id", node_id)),
                                "name": str(getattr(node, "name", node_id)),
                                "position": pos_list[:3],
                                "semantic_excerpt": str(getattr(node, "semantic", "") or "")[:160],
                            }
                        )
                    except Exception:
                        continue
            active_navigation = state.get("active_navigation")
            pending_navigation = state.get("pending_navigation")
            navigation = active_navigation if isinstance(active_navigation, dict) else pending_navigation
            navigation_view: dict[str, Any] = {"active": False}
            if isinstance(navigation, dict):
                target = navigation.get("target") if isinstance(navigation.get("target"), dict) else {}
                label = (
                    target.get("display_label")
                    or target.get("user_query")
                    or target.get("query")
                    or target.get("object_description")
                    or navigation.get("description")
                    or "目标"
                )
                target_type = str(target.get("type") or "")
                navigation_view = {
                    "active": True,
                    "task_id": navigation.get("task_id"),
                    "plan_id": navigation.get("plan_id"),
                    "token": navigation.get("navigation_token"),
                    "description": navigation.get("description"),
                    "label": str(label),
                    "target_type": target_type,
                    "kind": "目标物体" if target_type == "semantic_object" else "目标关键帧",
                    "destination_position": navigation.get("destination_position"),
                    "destination_keyframe_id": navigation.get("destination_keyframe_id"),
                    "created_at": navigation.get("created_at"),
                    "waiting_summary": navigation.get("waiting_summary"),
                }
            guidance_events = [
                {
                    "event_type": item.get("event_type"),
                    "text": item.get("text"),
                    "created_at": item.get("created_at"),
                    "dedupe_key": item.get("dedupe_key"),
                    "task_id": item.get("task_id"),
                }
                for item in list(state.get("guidance_events") or [])[-40:]
                if isinstance(item, dict)
            ]
            self.updated_at = _now_display()
            sim_meta = robot_state.get("simulation") if isinstance(robot_state.get("simulation"), dict) else {}
            return {
                "thread_id": self.thread_id,
                "updated_at": self.updated_at,
                "agent_status": input_window["title"],
                "agent_status_detail": input_window["detail"],
                "processing": self.processing_turn_id is not None,
                "simulation_mode": bool(sim_meta) or str(robot_state.get("source") or "") == "simulation",
                "robot": robot_state,
                "navigation": navigation_view,
                "keyframes": keyframes,
                "guidance_events": guidance_events,
                "latest_error": self.latest_error,
            }

    def submit_message(
        self,
        message: str,
        role: str,
        *,
        source: str = "manual",
        input_language: str | None = None,
        output_language: str | None = None,
        image_data_url: str | None = None,
    ) -> dict[str, Any]:
        """Queue one user message for async processing and return immediately."""

        clean_message = str(message or "").strip()
        clean_role = "user"
        if not clean_message and not image_data_url:
            return self.get_payload()
        if not clean_message and image_data_url:
            clean_message = "Use the attached image as the task reference."

        with self.lock:
            if self.processing_turn_id is not None:
                self.latest_error = "A turn is already in progress. Please wait for the live workflow to settle."
                return self._build_payload_locked()
            if self.checkpoint_choice_required:
                self.latest_error = "Choose Resume or New session before sending commands."
                return self._build_payload_locked()

            self._drop_restored_runtime_snapshot_locked()
            self.last_input_language = normalize_language(
                input_language or self.io_cfg.get("input_language", "zh"),
                fallback="zh",
            )
            self.last_output_language = normalize_language(
                output_language or self.io_cfg.get("output_language", "zh"),
                fallback="zh",
            )
            self.turn_counter += 1
            turn_id = self.turn_counter
            self.processing_turn_id = turn_id
            self.turn_history.append(
                {
                    "turn_id": turn_id,
                    "role": clean_role,
                    "role_label": "User Instruction",
                    "source": source,
                    "content": clean_message,
                    "image_attached": bool(image_data_url),
                    "response": "",
                    "response_items": [],
                    "created_at": _now_display(),
                    "updated_at": _now_display(),
                    "visited_nodes": [],
                    "step_trace": [],
                    "status": "running",
                    "live_node": "queued",
                }
            )
            self.turn_history = self.turn_history[-80:]
            self.latest_error = None
            payload = self._build_payload_locked()

        worker = threading.Thread(
            target=self._run_turn_worker,
            args=(turn_id, clean_message, clean_role, source, input_language, output_language, image_data_url),
            name=f"async-agent-web-turn-{turn_id}",
            daemon=True,
        )
        worker.start()
        return payload

    def current_image_payload(self) -> dict[str, Any]:
        """Return the latest controller image as a data URL for the browser."""

        image = current_controller_image(getattr(self.agent, "controller", None))
        if image is None:
            return {
                "status": "blocked",
                "error": "Current image is unavailable.",
                "image_data_url": None,
            }
        path = self._save_robot_capture_image(image)
        return {
            "status": "ok",
            "image_data_url": image_to_data_url(image),
            "path": str(path),
            "source": "robot-camera",
        }

    def teleop(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Apply one manual teleop command through the attached controller."""

        controller = getattr(self.agent, "controller", None)
        if controller is None:
            return {"status": "blocked", "error": "当前控制器不可用，暂时不能遥控小车。"}
        command = str(payload.get("command") or "").strip().lower()
        if command in {"stop", "emergency_stop", "zero"}:
            stop = getattr(controller, "manual_stop", None)
            if callable(stop):
                result = stop(cancel_navigation=True)
            else:
                cancel = getattr(controller, "cancel_navigation", None)
                if callable(cancel):
                    cancel()
                result = {"status": "ok", "summary": "已请求停车。"}
        else:
            teleop = getattr(controller, "manual_teleop", None)
            if not callable(teleop):
                return {"status": "blocked", "error": "当前控制器不支持网页遥控。"}
            result = teleop(
                linear=float(payload.get("linear") or 0.0),
                angular=float(payload.get("angular") or 0.0),
                mode=str(payload.get("mode") or "normal"),
                cancel_navigation=True,
            )
        with self.lock:
            self.latest_error = None
            self.updated_at = _now_display()
        response = {"status": "ok", "teleop": result}
        if bool(payload.get("include_ui")):
            with self.lock:
                response["ui"] = self._build_payload_locked()
        return response

    def _save_robot_capture_image(self, image) -> Path:
        """Persist a robot-camera snapshot so remote browser captures are auditable."""

        self.user_image_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        image_path = self.user_image_dir / f"capture_{stamp}.jpg"
        image.convert("RGB").save(image_path, format="JPEG", quality=92)
        return image_path

    def describe_uploaded_image(
        self,
        image_data_url: str,
        *,
        submit: bool = False,
        input_language: str | None = None,
        output_language: str | None = None,
    ) -> dict[str, Any]:
        """Describe an uploaded/captured image and optionally send it to the agent."""

        image = image_from_data_url(image_data_url)
        description = describe_image_for_navigation(image)
        payload: dict[str, Any] = {
            "status": "ok",
            "description": description,
        }
        if submit:
            message = (
                "This is a compact search description generated from a target image. "
                "Quickly find the closest matching candidate keyframe in scene memory "
                "and navigate there. Do not require a perfect detail-by-detail match; "
                "if several candidates are close, choose the best one.\n"
                f"{description}"
            )
            payload["agent_payload"] = self.submit_message(
                message,
                "user",
                source="image-upload",
                input_language=input_language,
                output_language=output_language,
            )
        return payload

    def clear_history(self) -> dict[str, Any]:
        """Clear visible session history while keeping backend thread state intact."""

        with self.lock:
            preserved_turns: list[dict[str, Any]] = []
            if self.processing_turn_id is not None:
                active_turn = self._find_turn_locked(self.processing_turn_id)
                if active_turn is not None:
                    preserved_turns.append(active_turn)

            self.turn_history = preserved_turns
            self.latest_error = None
            self.updated_at = _now_display()
            return self._build_payload_locked()



class AsyncAgentWebHandler(BaseHTTPRequestHandler):
    """HTTP handler bound to one AsyncAgentWebApp instance."""

    app: AsyncAgentWebApp

    def log_message(self, format: str, *args: Any) -> None:
        """Silence default HTTP access logs because the UI renders its own activity view."""

        return

    def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        """Write one JSON response with UTF-8 encoding."""

        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)
        except Exception as exc:
            if _is_client_disconnect_error(exc):
                return
            raise

    def _send_html(self, html: str) -> None:
        """Write the embedded application shell as the root HTML response."""

        encoded = html.encode("utf-8")
        try:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)
        except Exception as exc:
            if _is_client_disconnect_error(exc):
                return
            raise

    def _read_json_body(self) -> dict[str, Any]:
        """Decode the request body as JSON when the client sends one."""

        length = int(self.headers.get("Content-Length", "0") or "0")
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        if not raw:
            return {}
        return json.loads(raw.decode("utf-8"))

    def do_GET(self) -> None:
        """Serve either the app shell or the latest state snapshot."""

        try:
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self._send_html(APP_HTML)
                return
            if parsed.path in {"/lite", "/chat"}:
                self._send_html(LITE_APP_HTML)
                return
            if parsed.path in {"/sim", "/simulation"}:
                self._send_html(SIM_VIEW_HTML)
                return
            if parsed.path == "/api/state":
                self._send_json(self.app.get_payload())
                return
            if parsed.path == "/api/lite-state":
                self._send_json(self.app.get_lite_payload())
                return
            if parsed.path == "/api/sim-state":
                self._send_json(self.app.get_sim_payload())
                return
            if parsed.path == "/api/current-image":
                self._send_json(self.app.current_image_payload())
                return
            self._send_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)
        except Exception as exc:
            if _is_client_disconnect_error(exc):
                return
            raise

    def do_POST(self) -> None:
        """Handle message submission and history-clear requests."""

        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/message":
                payload = self._read_json_body()
                message = payload.get("message", "")
                role = payload.get("role", "user")
                self._send_json(
                    self.app.submit_message(
                        message,
                        role,
                        input_language=payload.get("input_language"),
                        output_language=payload.get("output_language"),
                        image_data_url=payload.get("image_data_url"),
                    )
                )
                return
            if parsed.path == "/api/upload-image":
                payload = self._read_json_body()
                self._send_json(
                    self.app.describe_uploaded_image(
                        payload.get("image_data_url", ""),
                        submit=bool(payload.get("submit", False)),
                        input_language=payload.get("input_language"),
                        output_language=payload.get("output_language"),
                    )
                )
                return
            if parsed.path == "/api/teleop":
                payload = self._read_json_body()
                self._send_json(self.app.teleop(payload))
                return
            if parsed.path == "/api/clear":
                self._send_json(self.app.clear_history())
                return
            if parsed.path == "/api/session":
                payload = self._read_json_body()
                self._send_json(
                    self.app.choose_session_mode(
                        payload.get("mode", ""),
                        run_memory_snapshot_path=payload.get("run_memory_snapshot_path"),
                    )
                )
                return
            self._send_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)
        except Exception as exc:
            if _is_client_disconnect_error(exc):
                return
            self.app.latest_error = str(exc)
            try:
                self._send_json({"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
            except Exception as write_exc:
                if _is_client_disconnect_error(write_exc):
                    return
                raise


def _load_scene_memory(dataset_dir: Path) -> SceneMemory:
    """Load the scene-memory dataset required by the local demo."""

    return SceneMemory(dataset_dir)


class WebDemoController:
    """Small controller for the web UI when ROS2/Nav2 is not embedded in-process."""

    def __init__(
        self,
        *,
        dry_run: bool = True,
        scene_memory: SceneMemory | None = None,
        current_image_path: Path | None = None,
    ):
        self._dry_run = bool(dry_run)
        self._status = "idle"
        self._latest_msg = ""
        self._path: list[list[float]] = []
        self._scene_memory = scene_memory
        self._current_image_path = current_image_path

    def update_path(self, new_path: list[list[float]]) -> None:
        self._path = list(new_path or [])
        self._status = "dry_run_dispatched" if self._dry_run else "dispatched"
        if self._path:
            goal = self._path[-1]
            self._latest_msg = f"Arrived at demo goal x={goal[0]:.2f}, y={goal[1]:.2f}"
        else:
            self._latest_msg = "Demo navigation received an empty path."

    def update_status(self, status: str) -> None:
        self._status = str(status or "idle")

    def update_latest_msg(self, msg: str) -> None:
        self._latest_msg = str(msg or "")

    def get_current_state(self) -> dict[str, Any]:
        return {
            "position": [0.0, 0.0, 0.0],
            "orientation": [0.0, 0.0, 0.0, 1.0],
            "status": self._status,
        }

    def get_current_image(self) -> Any:
        try:
            from PIL import Image

            image_path = self._current_image_path or self._nearest_keyframe_image_path()
            if image_path is None:
                return None
            return Image.open(image_path).convert("RGB")
        except Exception:
            return None

    def check_for_new_messages(self) -> str:
        message = self._latest_msg
        self._latest_msg = ""
        return message

    def get_status(self) -> str:
        return self._status

    def cancel_navigation(self) -> None:
        self._status = "cancelled"
        self._latest_msg = "Navigation cancelled."

    def manual_teleop(
        self,
        *,
        linear: float = 0.0,
        angular: float = 0.0,
        mode: str = "normal",
        cancel_navigation: bool = True,
    ) -> dict[str, Any]:
        if cancel_navigation:
            self.cancel_navigation()
        nav_cfg = config.get("navigation") if isinstance(config.get("navigation"), dict) else {}
        slow = str(mode or "normal").strip().lower() in {"slow", "extra_slow", "safer"}
        linear_limit = float(
            nav_cfg.get(
                "teleop_slow_linear_limit_mps" if slow else "teleop_linear_limit_mps",
                0.06 if slow else 0.12,
            )
            or (0.06 if slow else 0.12)
        )
        angular_limit = float(
            nav_cfg.get(
                "teleop_slow_angular_limit_radps" if slow else "teleop_angular_limit_radps",
                0.20 if slow else 0.35,
            )
            or (0.20 if slow else 0.35)
        )
        linear_cmd = max(-abs(linear_limit), min(abs(linear_limit), float(linear or 0.0)))
        angular_cmd = max(-abs(angular_limit), min(abs(angular_limit), float(angular or 0.0)))
        self._status = "manual_teleop"
        self._latest_msg = (
            f"遥控模拟：线速度 {linear_cmd:.2f}，角速度 {angular_cmd:.2f}，"
            f"模式 {'更低速' if slow else '低速'}。"
        )
        return {
            "status": "ok",
            "linear": linear_cmd,
            "angular": angular_cmd,
            "mode": "slow" if slow else "normal",
            "dry_run": True,
        }

    def manual_stop(self, *, cancel_navigation: bool = True) -> dict[str, Any]:
        if cancel_navigation:
            self.cancel_navigation()
        self._status = "manual_stop"
        self._latest_msg = "遥控模拟：已停车。"
        return {"status": "ok", "summary": "遥控模拟：已停车。", "dry_run": True}

    def _nearest_keyframe_image_path(self) -> Path | None:
        if self._scene_memory is None:
            return None
        try:
            node_id = self._scene_memory.find_nearest_node([0.0, 0.0, 0.0])
            node = self._scene_memory.keyframe_nodes[node_id]
            return node.rgb_path or node.left_path
        except Exception:
            return None


def _build_arg_parser() -> argparse.ArgumentParser:
    """Create the command-line parser for the AsyncAgent web demo."""

    parser = argparse.ArgumentParser(
        description="Run a minimal local web UI for AsyncAgent.",
    )
    parser.add_argument(
        "--dataset",
        default=str(DEFAULT_DATASET_DIR),
        help="Path to the scene-memory dataset directory.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind the local web server.")
    parser.add_argument("--port", type=int, default=8123, help="Port to bind the local web server.")
    parser.add_argument(
        "--controller-type",
        default="demo",
        choices=["demo"],
        help="Controller backend used by navigation tools. The web demo uses a dry-run controller.",
    )
    parser.add_argument(
        "--dry-run-navigation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use dry-run navigation in the local web demo.",
    )
    parser.add_argument(
        "--current-image",
        type=Path,
        default=None,
        help="Optional image path used by analyse_on_current_image in the local web demo.",
    )
    parser.add_argument(
        "--background-workers",
        type=int,
        default=None,
        help="Number of background workers for async planning. Defaults to runtime profile.",
    )
    parser.add_argument(
        "--thread-id",
        default="web_console",
        help="LangGraph thread id used by the web session.",
    )
    parser.add_argument(
        "--session-checkpoint",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Persist and restore the visible web session plus previous run memory.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=None,
        help="Optional explicit JSON checkpoint path for this web session.",
    )
    parser.add_argument(
        "--clear-resumed-plan",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Restore historical run memory but keep the active runtime plan idle.",
    )
    parser.add_argument(
        "--disable-navigation",
        action="store_true",
        help="Disable navigation-related tools and controller polling.",
    )
    parser.add_argument(
        "--disable-logging",
        action="store_true",
        help="Disable conversation logging for the web session.",
    )
    return parser


def main() -> None:
    """Start the local web server and keep it running until shutdown."""

    args = _build_arg_parser().parse_args()
    dataset_dir = normalize_runtime_path(args.dataset)
    scene_memory = _load_scene_memory(dataset_dir)
    controller = None if args.disable_navigation else WebDemoController(
        dry_run=args.dry_run_navigation,
        scene_memory=scene_memory,
        current_image_path=args.current_image,
    )
    agent = AsyncAgent(
        scene_memory,
        is_navigation_mode=not args.disable_navigation,
        controller_type=args.controller_type,
        controller=controller,
        enable_logging=not args.disable_logging,
        num_background_workers=args.background_workers,
    )

    app = AsyncAgentWebApp(
        agent,
        thread_id=args.thread_id,
        resume_checkpoint=args.session_checkpoint,
        checkpoint_path=args.checkpoint_path,
        clear_resumed_plan=args.clear_resumed_plan,
    )

    handler = type("BoundAsyncAgentWebHandler", (AsyncAgentWebHandler,), {"app": app})
    server = ThreadingHTTPServer((args.host, args.port), handler)

    print(f"CarAgent web UI is ready at http://{args.host}:{args.port}")
    print(f"Dataset: {dataset_dir}")
    print(f"Thread ID: {args.thread_id}")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
