"""Session-local evidence anchors for target resolution."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from typing import Any, Optional

from .types import NavigableAnchor, TargetRef


ANCHOR_STORE_KEY = "session_anchors"
MAX_SESSION_ANCHORS = 80


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _stable_json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    except Exception:
        return str(value)


def _anchor_id(*parts: Any) -> str:
    digest = hashlib.sha1("|".join(_stable_json(part) for part in parts).encode("utf-8")).hexdigest()
    return f"anchor_{digest[:16]}"


def _clean_dict(value: dict[str, Any]) -> dict[str, Any]:
    return {
        key: item
        for key, item in value.items()
        if item not in (None, "", [], {})
    }


def _task_id(current_task: Optional[dict[str, Any]]) -> Optional[int]:
    try:
        return int((current_task or {}).get("task_id"))
    except Exception:
        return None


def _description_from_ref(target_ref: dict[str, Any]) -> str:
    return str(target_ref.get("description") or target_ref.get("query") or "").strip()


def infer_mobility(description: Any) -> str:
    """Conservative mobility hint.

    This is a safety guard, not a semantic classifier.  Only obvious moving
    subjects are marked dynamic; everything else stays unknown unless a future
    perception/LLM evidence step promotes it.
    """

    text = str(description or "").lower()
    if not text:
        return "unknown"
    dynamic_tokens = (
        "person",
        "people",
        "human",
        "man",
        "woman",
        "child",
        "visitor",
        "pedestrian",
        "walking",
        "moving",
        "人",
        "行人",
        "学生",
        "老师",
        "访客",
    )
    if any(token in text for token in dynamic_tokens):
        return "dynamic"
    return "unknown"


def _anchor_store(runtime_control: Optional[dict[str, Any]]) -> list[dict[str, Any]]:
    if runtime_control is None:
        return []
    store = runtime_control.setdefault(ANCHOR_STORE_KEY, [])
    if not isinstance(store, list):
        store = []
        runtime_control[ANCHOR_STORE_KEY] = store
    return store


def _append_anchor(runtime_control: Optional[dict[str, Any]], anchor: dict[str, Any]) -> dict[str, Any]:
    store = _anchor_store(runtime_control)
    existing_id = anchor.get("anchor_id")
    if existing_id:
        store[:] = [item for item in store if not isinstance(item, dict) or item.get("anchor_id") != existing_id]
    store.append(anchor)
    if len(store) > MAX_SESSION_ANCHORS:
        del store[: len(store) - MAX_SESSION_ANCHORS]
    return anchor


def get_session_anchors(runtime_control: Optional[dict[str, Any]]) -> list[dict[str, Any]]:
    return [dict(item) for item in _anchor_store(runtime_control) if isinstance(item, dict)]


def _position_from_anchor(anchor: dict[str, Any]) -> Optional[list[float]]:
    position = anchor.get("position")
    if isinstance(position, (list, tuple)) and len(position) >= 2:
        try:
            return [
                float(position[0]),
                float(position[1]),
                float(position[2]) if len(position) >= 3 else 0.0,
            ]
        except Exception:
            return None
    navigable = anchor.get("navigable_anchor") if isinstance(anchor.get("navigable_anchor"), dict) else {}
    position = navigable.get("position")
    if isinstance(position, (list, tuple)) and len(position) >= 2:
        try:
            return [
                float(position[0]),
                float(position[1]),
                float(position[2]) if len(position) >= 3 else 0.0,
            ]
        except Exception:
            return None
    return None


def _keyframe_from_anchor(anchor: dict[str, Any]) -> Optional[int]:
    for value in (
        anchor.get("keyframe_id"),
        (anchor.get("navigable_anchor") or {}).get("keyframe_id")
        if isinstance(anchor.get("navigable_anchor"), dict)
        else None,
    ):
        if value is None:
            continue
        try:
            return int(value)
        except Exception:
            continue
    return None


def _text_score(query: str, candidate: str) -> float:
    query_tokens = {
        token
        for token in re.split(r"[^0-9a-zA-Z\u4e00-\u9fff]+", query.lower())
        if len(token) >= 2
    }
    if not query_tokens:
        return 0.0
    candidate_text = candidate.lower()
    hits = sum(1 for token in query_tokens if token in candidate_text)
    return hits / max(1, len(query_tokens))


def _anchor_matches_source(target_ref: TargetRef, anchor: dict[str, Any]) -> bool:
    source = str(target_ref.get("source") or "").strip()
    anchor_source = str(anchor.get("source") or "").strip()
    if source == "session_memory":
        return anchor_source in {"session_memory", "arrived_scene", "scene_memory", "attached_image", "current_view"}
    if source in {"arrived_scene", "upstream_result"}:
        return anchor_source in {"arrived_scene", "session_memory", "scene_memory", "attached_image"}
    if source in {"scene_memory", "explicit"}:
        # A fresh scene-memory target is a new map query, not a reference to
        # this session's previous arrival.  Reusing session anchors here can
        # silently turn "go to elevator entrance" into "go to the last entrance".
        return False
    if source == "attached_image":
        return str(anchor.get("source") or "") == "attached_image"
    return False


def find_reusable_anchor(
    target_ref: TargetRef,
    current_task: Optional[dict[str, Any]],
    tasks: Optional[dict[int, dict[str, Any]]],
    runtime_control: Optional[dict[str, Any]],
) -> Optional[dict[str, Any]]:
    anchors = get_session_anchors(runtime_control)
    if not anchors:
        return None
    kind = str(target_ref.get("kind") or "").strip()
    source = str(target_ref.get("source") or "").strip()
    query = _description_from_ref(target_ref)
    inputs_from = target_ref.get("inputs_from")
    upstream_ids: set[int] = set()

    def add_task_id(value: Any) -> None:
        try:
            upstream_ids.add(int(value))
        except Exception:
            pass

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            if value.get("task_id") is not None:
                add_task_id(value.get("task_id"))
            for item in value.values():
                visit(item)
            return
        if isinstance(value, (list, tuple, set)):
            for item in value:
                visit(item)

    visit(inputs_from)
    # arrived_scene/upstream_result mean "use evidence from a specific upstream
    # arrival/result".  Without that structural link, text/freshness must not
    # make the latest arrival look eligible.
    if source in {"arrived_scene", "upstream_result"} and not upstream_ids:
        return None

    scored: list[tuple[float, dict[str, Any]]] = []
    for anchor in anchors:
        if not isinstance(anchor, dict):
            continue
        if not _anchor_matches_source(target_ref, anchor):
            continue
        anchor_kind = str(anchor.get("kind") or "").strip()
        if kind == "keyframe" and anchor_kind not in {"keyframe", "view", "attached_image"}:
            continue
        if kind == "object":
            object_anchor_ok = anchor_kind == "object"
            staging_anchor_ok = (
                source in {"arrived_scene", "upstream_result", "session_memory"}
                and anchor_kind in {"keyframe", "view", "attached_image", "object_staging"}
            )
            if not object_anchor_ok and not staging_anchor_ok:
                continue
        matched_upstream = False
        matched_upstream_task_ids: list[int] = []
        if upstream_ids:
            anchor_task_id = anchor.get("task_id")
            try:
                if int(anchor_task_id) in upstream_ids:
                    matched_upstream = True
                    matched_upstream_task_ids.append(int(anchor_task_id))
                else:
                    evidence = anchor.get("evidence") if isinstance(anchor.get("evidence"), list) else []
                    evidence_text = _stable_json(evidence)
                    matched_in_evidence = [
                        task_id
                        for task_id in upstream_ids
                        if f'"task_id": {task_id}' in evidence_text
                    ]
                    if not matched_in_evidence:
                        continue
                    matched_upstream = True
                    matched_upstream_task_ids.extend(matched_in_evidence)
            except Exception:
                continue
        score = 0.2
        matched_by = ["source"]
        match_reason_parts = [f"source={source or 'unknown'} accepts anchor_source={anchor.get('source')}"]
        if kind == "keyframe" and _keyframe_from_anchor(anchor) is not None:
            score += 0.5
            matched_by.append("keyframe_anchor")
            match_reason_parts.append(f"keyframe_id={_keyframe_from_anchor(anchor)}")
        if kind == "object" and anchor_kind == "object" and _position_from_anchor(anchor) is not None:
            score += 0.5
            matched_by.append("position_anchor")
            match_reason_parts.append("has reusable object position")
        if kind == "object" and anchor_kind in {"keyframe", "view", "attached_image", "object_staging"}:
            staging_keyframe_id = _keyframe_from_anchor(anchor)
            if staging_keyframe_id is not None:
                score += 0.35
                matched_by.append("staging_anchor")
                match_reason_parts.append(f"staging_keyframe_id={staging_keyframe_id}")
        if matched_upstream:
            score += 0.4
            matched_by.append("upstream_task")
            match_reason_parts.append(f"matched upstream task(s) {matched_upstream_task_ids}")
        text_score = _text_score(query, str(anchor.get("description") or ""))
        score += 0.3 * text_score
        if text_score > 0:
            matched_by.append("text")
            match_reason_parts.append(f"text_score={text_score:.2f}")
        if str(anchor.get("freshness") or "") == "fresh":
            score += 0.1
            matched_by.append("freshness")
            match_reason_parts.append("fresh anchor")
        annotated_anchor = dict(anchor)
        annotated_anchor["_match_score"] = round(score, 3)
        annotated_anchor["_matched_by"] = matched_by
        annotated_anchor["_match_reason"] = "; ".join(match_reason_parts)
        annotated_anchor["_text_score"] = round(text_score, 3)
        if matched_upstream_task_ids:
            annotated_anchor["_matched_upstream_task_ids"] = matched_upstream_task_ids
        scored.append((score, annotated_anchor))
    if not scored:
        return None
    scored.sort(key=lambda item: item[0], reverse=True)
    best_score, best_anchor = scored[0]
    if best_score < 0.45:
        return None
    return dict(best_anchor)


def anchor_to_navigable(anchor: dict[str, Any]) -> Optional[NavigableAnchor]:
    navigable = anchor.get("navigable_anchor") if isinstance(anchor.get("navigable_anchor"), dict) else {}
    anchor_type = str(navigable.get("anchor_type") or "").strip()
    if anchor_type == "keyframe" and navigable.get("keyframe_id") is not None:
        return {
            "anchor_type": "keyframe",
            "keyframe_id": int(navigable["keyframe_id"]),
            "source": "session_anchor",
        }
    if anchor_type == "position":
        position = _position_from_anchor(anchor)
        if position is None:
            return None
        return {
            "anchor_type": "position",
            "position": position,
            "yaw_deg": float(navigable.get("yaw_deg") or anchor.get("yaw_deg") or 0.0),
            "source": "session_anchor",
        }
    keyframe_id = _keyframe_from_anchor(anchor)
    if keyframe_id is not None:
        return {
            "anchor_type": "keyframe",
            "keyframe_id": int(keyframe_id),
            "source": "session_anchor",
        }
    position = _position_from_anchor(anchor)
    if position is not None:
        return {
            "anchor_type": "position",
            "position": position,
            "yaw_deg": float(anchor.get("yaw_deg") or 0.0),
            "source": "session_anchor",
        }
    return None


def record_anchor_from_resolution_result(
    runtime_control: Optional[dict[str, Any]],
    result: dict[str, Any],
    *,
    current_task: Optional[dict[str, Any]] = None,
) -> Optional[dict[str, Any]]:
    anchor = result.get("anchor") if isinstance(result.get("anchor"), dict) else {}
    target_ref = result.get("target_ref") if isinstance(result.get("target_ref"), dict) else {}
    if not anchor:
        return None
    kind = str(target_ref.get("kind") or "").strip()
    source = str(target_ref.get("source") or anchor.get("source") or "session_memory").strip()
    description = _description_from_ref(target_ref) or str((current_task or {}).get("description") or "")
    keyframe_id = anchor.get("keyframe_id")
    position = anchor.get("position")
    evidence = list(result.get("evidence") or [])[:4]
    first_data = {}
    for item in evidence:
        if isinstance(item, dict) and isinstance(item.get("data"), dict):
            first_data = item.get("data") or {}
            break
    target_kind = str(target_ref.get("target_kind") or "").strip()
    image_focus = str(target_ref.get("image_focus") or "").strip()
    is_attached_object_staging = (
        source == "attached_image"
        and str(anchor.get("anchor_type") or "") == "keyframe"
        and (kind == "object" or target_kind == "object" or image_focus == "object")
    )
    mobility = infer_mobility(description) if kind == "object" or is_attached_object_staging else "static"
    anchor_kind = "object_staging" if is_attached_object_staging else kind
    if source == "attached_image" and kind == "keyframe" and not is_attached_object_staging:
        anchor_kind = "attached_image"
    session_anchor = _clean_dict(
        {
            "anchor_id": _anchor_id("resolution", _task_id(current_task), kind, source, description, keyframe_id, position),
            "kind": anchor_kind,
            "source": source,
            "task_id": _task_id(current_task),
            "created_at": _now_iso(),
            "freshness": "fresh",
            "description": description,
            "object_description": description if is_attached_object_staging else None,
            "keyframe_id": int(keyframe_id) if keyframe_id is not None else None,
            "recommended_keyframe_id": int(keyframe_id) if is_attached_object_staging and keyframe_id is not None else None,
            "candidate_keyframe_ids": first_data.get("candidate_keyframe_ids") if is_attached_object_staging else None,
            "recommendation_reason": first_data.get("recommendation_reason") if is_attached_object_staging else None,
            "retrieval_mode": first_data.get("retrieval_mode") if is_attached_object_staging else None,
            "position": list(position) if isinstance(position, (list, tuple)) else None,
            "yaw_deg": anchor.get("yaw_deg"),
            "image_ref": _first_image_ref(result),
            "evidence": evidence,
            "navigable_anchor": dict(anchor),
            "mobility": mobility,
        }
    )
    return _append_anchor(runtime_control, session_anchor)


def _first_image_ref(result: dict[str, Any]) -> Optional[str]:
    for evidence in list(result.get("evidence") or []):
        if not isinstance(evidence, dict):
            continue
        data = evidence.get("data") if isinstance(evidence.get("data"), dict) else {}
        if data.get("image_ref"):
            return str(data.get("image_ref"))
    return None


def record_anchor_from_object_destination(
    runtime_control: Optional[dict[str, Any]],
    *,
    current_task: Optional[dict[str, Any]],
    destination: Optional[dict[str, Any]],
    selected_object: Optional[dict[str, Any]],
    evidence: Optional[list[dict[str, Any]]] = None,
) -> Optional[dict[str, Any]]:
    if not isinstance(destination, dict):
        return None
    position = destination.get("position")
    if not isinstance(position, (list, tuple)) or len(position) < 2:
        return None
    description = str(
        (selected_object or {}).get("description")
        or destination.get("target_description")
        or (current_task or {}).get("description")
        or ""
    ).strip()
    yaw_deg = destination.get("yaw_deg")
    navigable_anchor = {
        "anchor_type": "position",
        "position": [float(position[0]), float(position[1]), float(position[2]) if len(position) >= 3 else 0.0],
        "yaw_deg": float(yaw_deg or 0.0),
        "source": "session_memory",
    }
    session_anchor = _clean_dict(
        {
            "anchor_id": _anchor_id("object", _task_id(current_task), description, navigable_anchor),
            "kind": "object",
            "source": str((selected_object or {}).get("target_source") or "session_memory"),
            "task_id": _task_id(current_task),
            "created_at": _now_iso(),
            "freshness": "fresh",
            "description": description,
            "position": navigable_anchor["position"],
            "yaw_deg": navigable_anchor["yaw_deg"],
            "evidence": list(evidence or [])[:4],
            "navigable_anchor": navigable_anchor,
            "mobility": infer_mobility(description),
        }
    )
    return _append_anchor(runtime_control, session_anchor)


def record_anchor_from_navigation_arrival(
    runtime_control: Optional[dict[str, Any]],
    *,
    task: Optional[dict[str, Any]],
    event: Optional[dict[str, Any]],
    image_ref: Optional[dict[str, Any]] = None,
) -> Optional[dict[str, Any]]:
    payload = (event or {}).get("payload") if isinstance((event or {}).get("payload"), dict) else {}
    if image_ref is None and isinstance(payload.get("arrival_image_ref"), dict):
        image_ref = dict(payload.get("arrival_image_ref") or {})
    latest_result = (task or {}).get("result")
    latest = latest_result[-1] if isinstance(latest_result, list) and latest_result else {}
    destination = latest.get("destination") if isinstance(latest, dict) else None
    keyframe_id = None
    position = None
    yaw_deg = None
    if isinstance(destination, dict):
        if destination.get("type") == "keyframe" and destination.get("keyframe_id") is not None:
            try:
                keyframe_id = int(destination.get("keyframe_id"))
            except Exception:
                keyframe_id = None
        if destination.get("type") == "position":
            position = destination.get("position")
            yaw_deg = destination.get("yaw_deg")
    reported = payload.get("reported_position") or payload.get("position") or payload.get("destination_position")
    if position is None and isinstance(reported, (list, tuple)) and len(reported) >= 2:
        position = reported
    description = str(payload.get("destination_description") or (task or {}).get("description") or "").strip()
    navigable_anchor: dict[str, Any] = {}
    kind = "view"
    if keyframe_id is not None:
        kind = "keyframe"
        navigable_anchor = {"anchor_type": "keyframe", "keyframe_id": keyframe_id, "source": "arrival"}
    elif isinstance(position, (list, tuple)) and len(position) >= 2:
        navigable_anchor = {
            "anchor_type": "position",
            "position": [float(position[0]), float(position[1]), float(position[2]) if len(position) >= 3 else 0.0],
            "yaw_deg": float(yaw_deg or 0.0),
            "source": "arrival",
        }
    session_anchor = _clean_dict(
        {
            "anchor_id": _anchor_id("arrival", _task_id(task), description, keyframe_id, position),
            "kind": kind,
            "source": "arrived_scene",
            "task_id": _task_id(task),
            "created_at": _now_iso(),
            "freshness": "fresh",
            "description": description,
            "keyframe_id": keyframe_id,
            "position": list(position) if isinstance(position, (list, tuple)) else None,
            "yaw_deg": yaw_deg,
            "image_ref": image_ref,
            "evidence": [{"tool_name": "navigation_arrived", "status": "ok", "data": payload}],
            "navigable_anchor": navigable_anchor or None,
            "mobility": "static",
        }
    )
    return _append_anchor(runtime_control, session_anchor)
