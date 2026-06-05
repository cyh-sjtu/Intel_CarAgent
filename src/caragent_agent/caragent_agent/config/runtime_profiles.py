"""Shared runtime profile presets for async-agent resource configuration."""

from __future__ import annotations

from typing import Any


INTERACTIVE_SAFE_PROFILE: dict[str, Any] = {
    "background_start_policy": "after_first_navigation_dispatch",
    "background_workers_default": 1,
    "speculative_branch_preanalysis": False,
    "llm_background_yield_to_foreground": True,
    "clip_search_lock_enabled": True,
    "llm_max_concurrency_per_provider": {
        "default": 2,
        "qwen": 1,
        "deepseek": 2,
        "doubao": 1,
    },
    "llm_max_concurrency_per_model": {
        "default": 2,
        "qwen3.6-flash": 1,
        "qwen3.6-plus": 1,
        "qwen3-vl-flash": 1,
    },
}

STANDARD_PROFILE: dict[str, Any] = {
    "background_start_policy": "immediate_after_plan",
    "background_workers_default": 3,
    "speculative_branch_preanalysis": False,
    "llm_background_yield_to_foreground": False,
    "clip_search_lock_enabled": False,
    "llm_max_concurrency_per_provider": {
        "default": 4,
        "qwen": 2,
        "deepseek": 3,
        "doubao": 2,
    },
    "llm_max_concurrency_per_model": {
        "default": 4,
        "qwen3.6-flash": 2,
        "qwen3.6-plus": 2,
        "qwen3-vl-flash": 2,
        "deepseek-chat": 3,
    },
}

RATE_LIMIT_SAFE_PROFILE: dict[str, Any] = {
    "background_start_policy": "after_first_navigation_dispatch",
    "background_workers_default": 0,
    "speculative_branch_preanalysis": False,
    "llm_background_yield_to_foreground": True,
    "clip_search_lock_enabled": True,
    "llm_max_concurrency_per_provider": {
        "default": 1,
        "qwen": 1,
        "deepseek": 1,
        "doubao": 1,
    },
    "llm_max_concurrency_per_model": {
        "default": 1,
        "qwen3.6-flash": 1,
        "qwen3.6-plus": 1,
        "qwen3-vl-flash": 1,
        "deepseek-chat": 1,
    },
}


RUNTIME_PROFILE_PRESETS: dict[str, dict[str, Any]] = {
    "standard": dict(STANDARD_PROFILE),
    "interactive_safe": dict(INTERACTIVE_SAFE_PROFILE),
    "rate_limit_safe": dict(RATE_LIMIT_SAFE_PROFILE),
    "cloud_high_resource": dict(STANDARD_PROFILE),
    "local_low_resource": dict(INTERACTIVE_SAFE_PROFILE),
    "single_api_safe": dict(RATE_LIMIT_SAFE_PROFILE),
}


def default_runtime_profile_name() -> str:
    """Return the repository default runtime profile name."""

    return "standard"


def resolve_runtime_profile(config: dict[str, Any]) -> dict[str, Any]:
    """Return the configured runtime profile plus explicit overrides."""

    profile_name = str(
        config.get("runtime_profile") or default_runtime_profile_name()
    ).strip()
    custom_profiles = config.get("runtime_profiles")
    custom_profile = None
    if isinstance(custom_profiles, dict):
        raw_custom_profile = custom_profiles.get(profile_name)
        if isinstance(raw_custom_profile, dict):
            custom_profile = raw_custom_profile

    if custom_profile is not None:
        base_name = str(
            custom_profile.get("base") or default_runtime_profile_name()
        ).strip()
        preset = dict(
            RUNTIME_PROFILE_PRESETS.get(
                base_name,
                RUNTIME_PROFILE_PRESETS[default_runtime_profile_name()],
            )
        )
        preset.update(
            {
                key: value
                for key, value in custom_profile.items()
                if key != "base"
            }
        )
    else:
        preset = dict(
            RUNTIME_PROFILE_PRESETS.get(
                profile_name,
                RUNTIME_PROFILE_PRESETS[default_runtime_profile_name()],
            )
        )
    preset["name"] = profile_name

    raw_overrides = config.get("runtime_profile_overrides")
    if isinstance(raw_overrides, dict):
        preset.update(raw_overrides)

    return preset


__all__ = [
    "RUNTIME_PROFILE_PRESETS",
    "default_runtime_profile_name",
    "resolve_runtime_profile",
]
