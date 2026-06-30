"""Runtime resource profile helpers for async foreground/background work."""

from __future__ import annotations

from typing import Any

from caragent_agent.config.runtime_profiles import (
    RUNTIME_PROFILE_PRESETS,
    resolve_runtime_profile,
)


def should_enable_background_immediately(profile: dict[str, Any]) -> bool:
    """Return True when background workers may run as soon as a plan is created."""

    return (
        str(profile.get("background_start_policy") or "").strip()
        == "immediate_after_plan"
    )


def speculative_branch_preanalysis_enabled(profile: dict[str, Any]) -> bool:
    """Return True when background workers may precompute unresolved branch options."""

    return bool(profile.get("speculative_branch_preanalysis", False))


def llm_background_yield_to_foreground_enabled(config: dict[str, Any]) -> bool:
    """Return True when background LLM calls should wait for foreground demand."""

    profile = resolve_runtime_profile(config)
    return bool(profile.get("llm_background_yield_to_foreground", True))


def clip_search_lock_enabled(config: dict[str, Any]) -> bool:
    """Return True when CLIP search should serialize access to shared model state."""

    profile = resolve_runtime_profile(config)
    return bool(profile.get("clip_search_lock_enabled", True))


__all__ = [
    "RUNTIME_PROFILE_PRESETS",
    "clip_search_lock_enabled",
    "llm_background_yield_to_foreground_enabled",
    "resolve_runtime_profile",
    "should_enable_background_immediately",
    "speculative_branch_preanalysis_enabled",
]
