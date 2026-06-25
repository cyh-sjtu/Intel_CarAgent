"""Deterministic semantic target policy helpers."""

from __future__ import annotations

from typing import Any

from .types import TargetDraft, TargetRef, TargetSource


def normalize_source(value: Any, *, default: TargetSource) -> TargetSource:
    normalized = str(value or "").strip().lower()
    if normalized in {
        "scene_memory",
        "current_view",
        "attached_image",
        "arrived_scene",
        "upstream_result",
        "session_memory",
        "explicit",
    }:
        return normalized  # type: ignore[return-value]
    return default


def draft_from_navigation_target(
    current_task: dict[str, Any],
    target: dict[str, Any],
) -> TargetDraft:
    target_type = str(target.get("type") or "").strip()
    if target_type == "semantic_object":
        description = str(target.get("object_description") or "").strip()
        source_hint = normalize_source(target.get("target_source"), default="unknown")
    elif target_type == "semantic_keyframe":
        description = str(target.get("query") or "").strip()
        source_hint = normalize_source(target.get("target_source"), default="scene_memory")
    else:
        description = ""
        source_hint = "unknown"
    if not description:
        description = str(current_task.get("description") or "").strip()
    return {
        "target_type": target_type,
        "description": description,
        "source_hint": source_hint,
        "raw_target": dict(target),
    }


def target_ref_from_draft(
    draft: TargetDraft,
    current_task: dict[str, Any],
) -> TargetRef:
    target = draft.get("raw_target") or {}
    target_type = draft.get("target_type")
    if target_type == "semantic_object":
        ref: TargetRef = {
            "kind": "object",
            "source": draft["source_hint"],
            "description": draft["description"],
        }
        if target.get("image_refs"):
            ref["image_refs"] = target.get("image_refs")
        image_focus = str(target.get("image_focus") or "").strip().lower()
        if image_focus in {"scene", "object"}:
            ref["image_focus"] = image_focus
        if target.get("target_kind"):
            ref["target_kind"] = str(target.get("target_kind") or "").strip()
        if target.get("selection_policy") is not None:
            ref["selection_policy"] = target.get("selection_policy")
        inputs_from = target.get("inputs_from") or current_task.get("inputs_from")
        if inputs_from:
            ref["inputs_from"] = inputs_from
        if target.get("stop_distance_m") is not None:
            try:
                ref["stop_distance_m"] = float(target.get("stop_distance_m"))
            except Exception:
                pass
        return ref
    ref: TargetRef = {
        "kind": "keyframe",
        "source": draft["source_hint"],
        "description": draft["description"],
        "query": draft["description"],
    }
    if target.get("image_refs"):
        ref["image_refs"] = target.get("image_refs")
    image_focus = str(target.get("image_focus") or "").strip().lower()
    if image_focus in {"scene", "object"}:
        ref["image_focus"] = image_focus
    if target.get("target_kind"):
        ref["target_kind"] = str(target.get("target_kind") or "").strip()
    if target.get("selection_policy") is not None:
        ref["selection_policy"] = target.get("selection_policy")
    return ref


def next_step_for_unsupported_object(source: str, *, has_inputs: bool) -> tuple[str, str]:
    if source in {"arrived_scene", "upstream_result"} and not has_inputs:
        return (
            "needs_upstream_evidence",
            f"semantic_object source={source} needs inputs_from before live localization.",
        )
    if source in {"arrived_scene", "upstream_result"}:
        return (
            "needs_live_localization_after_arrival",
            f"semantic_object source={source} is allowed only after upstream arrival evidence.",
        )
    return (
        "needs_staging_keyframe",
        f"semantic_object source={source or 'unknown'} needs a staging keyframe before localization.",
    )
