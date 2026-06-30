"""Compatibility helpers for task metadata used by old saved sessions.

New planner/task contracts should use target.type, outputs, inputs_from, and
submit_task_result. The helpers in this module are only for reading historical
plans or run-memory snapshots that still contain legacy fields.
"""

from __future__ import annotations

from typing import Any, Optional


LEGACY_OBJECT_KIND_VALUES = {"object_level", "object-level", "object_destination"}
LEGACY_OBJECT_PREANALYSIS_POLICY = "historical_keyframe_then_live"
LEGACY_STAGING_KIND_VALUES = {"staging_keyframe", "place_matching"}


def legacy_object_kind(task: Optional[dict[str, Any]]) -> bool:
    if not isinstance(task, dict):
        return False
    legacy_kind = str(task.get("resolver_kind") or "").strip().lower()
    legacy_policy = str(task.get("preanalysis_policy") or "").strip().lower()
    return (
        legacy_kind in LEGACY_OBJECT_KIND_VALUES
        or legacy_policy == LEGACY_OBJECT_PREANALYSIS_POLICY
    )


def legacy_staging_kind(task: Optional[dict[str, Any]]) -> bool:
    if not isinstance(task, dict):
        return False
    legacy_kind = str(task.get("resolver_kind") or "").strip().lower()
    return legacy_kind in LEGACY_STAGING_KIND_VALUES


def legacy_upstream_task_id(task: Optional[dict[str, Any]]) -> Optional[int]:
    if not isinstance(task, dict):
        return None
    raw_value = task.get("staging_task_id")
    try:
        return int(raw_value)
    except Exception:
        return None


def has_legacy_grounding_metadata(task: Optional[dict[str, Any]]) -> bool:
    if not isinstance(task, dict):
        return False
    return any(
        task.get(key) not in (None, "", [], {})
        for key in ("resolver_kind", "preanalysis_policy", "staging_task_id", "primary_target")
    )


__all__ = [
    "has_legacy_grounding_metadata",
    "legacy_object_kind",
    "legacy_staging_kind",
    "legacy_upstream_task_id",
]
