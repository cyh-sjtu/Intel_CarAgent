"""Project configuration loader with optional local profile overrides."""

from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from caragent_agent.config.runtime_profiles import resolve_runtime_profile


CONFIG_DIR = Path(__file__).resolve().parent
BASE_CONFIG_NAME = "config.yaml"
LOCAL_CONFIG_NAME = "local_config.yaml"
LLM_API_CONFIG_NAME = "llm_api.yaml"

PROFILE_ENV_VAR = "CARAGENT_PROFILE"
LOCAL_CONFIG_ENV_VAR = "CARAGENT_CONFIG_FILE"
BASE_CONFIG_ENV_VAR = "CARAGENT_BASE_CONFIG_FILE"
EXTRA_CONFIG_ENV_VAR = "CARAGENT_EXTRA_CONFIG_FILE"

API_ENV_BY_PROVIDER = {
    "qwen": "DASHSCOPE_API_KEY",
    "dashscope": "DASHSCOPE_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "doubao": "DOUBAO_API_KEY",
}

API_CONFIG_KEYS_BY_PROVIDER = {
    "qwen": ("qwen-api", "dashscope"),
    "dashscope": ("qwen-api", "dashscope"),
    "deepseek": ("deepseek-api", "deepseek"),
    "doubao": ("doubao-api", "doubao"),
}


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {path}")
    return loaded


def _package_share_config_path() -> Path | None:
    """Return the installed share config path when running from an install tree."""

    for prefix in os.environ.get("AMENT_PREFIX_PATH", "").split(os.pathsep):
        if not prefix:
            continue
        candidate = Path(prefix) / "share" / "caragent_agent" / "config" / BASE_CONFIG_NAME
        if candidate.exists():
            return candidate
    return None


def _source_tree_config_path() -> Path | None:
    """Return the source-tree ROS config path."""

    for parent in CONFIG_DIR.parents:
        candidate = parent / "src" / "caragent_agent" / "config" / BASE_CONFIG_NAME
        if candidate.exists():
            return candidate
    return None


def _base_config_path() -> Path:
    base_override = os.environ.get(BASE_CONFIG_ENV_VAR, "").strip()
    if base_override:
        return Path(base_override).expanduser().resolve()

    for candidate in (
        _package_share_config_path(),
        _source_tree_config_path(),
    ):
        if candidate and candidate.exists():
            return candidate

    return CONFIG_DIR / BASE_CONFIG_NAME


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge two config mappings without mutating either input."""

    merged = deepcopy(base)
    for key, value in override.items():
        if (
            isinstance(value, dict)
            and isinstance(merged.get(key), dict)
        ):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _local_config_path() -> Path:
    override = os.environ.get(LOCAL_CONFIG_ENV_VAR, "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return CONFIG_DIR / LOCAL_CONFIG_NAME


def _extra_config_path() -> Path | None:
    override = os.environ.get(EXTRA_CONFIG_ENV_VAR, "").strip()
    if not override:
        return None
    return Path(override).expanduser().resolve()


def _selected_profile_name(data: dict[str, Any]) -> str:
    env_profile = os.environ.get(PROFILE_ENV_VAR, "").strip()
    if env_profile:
        return env_profile
    return str(
        data.get("active_profile")
        or data.get("profile")
        or ""
    ).strip()


def _apply_selected_profile(data: dict[str, Any]) -> dict[str, Any]:
    profile_name = _selected_profile_name(data)
    if not profile_name:
        return data

    profiles = data.get("profiles")
    if not isinstance(profiles, dict):
        raise KeyError(
            f"Profile '{profile_name}' was requested, but config['profiles'] is missing."
        )

    profile_data = profiles.get(profile_name)
    if not isinstance(profile_data, dict):
        available = ", ".join(sorted(str(key) for key in profiles)) or "none"
        raise KeyError(
            f"Profile '{profile_name}' was not found. Available profiles: {available}."
        )

    merged = deep_merge(data, profile_data)
    merged["active_profile"] = profile_name
    return merged


def _runtime_profile_defaults(data: dict[str, Any]) -> dict[str, Any]:
    """Return selected runtime-profile defaults as config-level values."""

    profile = resolve_runtime_profile(data)
    return {
        key: value
        for key, value in profile.items()
        if key not in {"name", "base"}
    }


def load_config() -> dict[str, Any]:
    """Load tracked defaults, optional local config, selected profile, and secrets."""

    base_path = _base_config_path()
    if not base_path.exists():
        raise FileNotFoundError(f"Base config file not found: {base_path}")

    base_data = _read_yaml(base_path)

    local_path = _local_config_path()
    local_data = _read_yaml(local_path) if local_path.exists() else {}
    extra_path = _extra_config_path()
    extra_data = _read_yaml(extra_path) if extra_path and extra_path.exists() else {}

    provisional_data = deep_merge(deep_merge(base_data, local_data), extra_data)
    profile_name = _selected_profile_name(provisional_data)
    profile_data: dict[str, Any] = {}
    if profile_name:
        profiles = provisional_data.get("profiles")
        if not isinstance(profiles, dict):
            raise KeyError(
                f"Profile '{profile_name}' was requested, but config['profiles'] is missing."
            )

        raw_profile_data = profiles.get(profile_name)
        if not isinstance(raw_profile_data, dict):
            available = ", ".join(sorted(str(key) for key in profiles)) or "none"
            raise KeyError(
                f"Profile '{profile_name}' was not found. Available profiles: {available}."
            )
        profile_data = raw_profile_data

    runtime_selection = deep_merge(provisional_data, profile_data)
    if profile_name:
        runtime_selection["active_profile"] = profile_name

    data = deep_merge(base_data, _runtime_profile_defaults(runtime_selection))
    data = deep_merge(data, local_data)
    if profile_data:
        data = deep_merge(data, profile_data)
        data["active_profile"] = profile_name
    data = deep_merge(data, extra_data)

    # Keep the local secret-only file supported and ignored by version control.
    llm_api_path = CONFIG_DIR / LLM_API_CONFIG_NAME
    if llm_api_path.exists():
        data = deep_merge(data, _read_yaml(llm_api_path))

    return data

class Config:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.data = load_config()
                    
        return cls._instance


def get_api_key(provider: str) -> str:
    """Return an API key from env vars first, then local ignored config."""

    normalized = str(provider or "").strip().lower()
    env_var = API_ENV_BY_PROVIDER.get(normalized)
    if env_var:
        env_value = os.environ.get(env_var, "").strip()
        if env_value:
            return env_value

    api_keys = config.get("api_keys")
    if isinstance(api_keys, dict):
        for key in API_CONFIG_KEYS_BY_PROVIDER.get(normalized, ()):
            value = api_keys.get(key)
            if value and str(value).strip():
                return str(value).strip()

    for key in API_CONFIG_KEYS_BY_PROVIDER.get(normalized, ()):
        value = config.get(key)
        if value and str(value).strip():
            return str(value).strip()

    return ""


def ensure_api_key_env(provider: str) -> str:
    """Populate the provider env var from config when a local key is available."""

    normalized = str(provider or "").strip().lower()
    api_key = get_api_key(normalized)
    env_var = API_ENV_BY_PROVIDER.get(normalized)
    if api_key and env_var and not os.environ.get(env_var):
        os.environ[env_var] = api_key
    return api_key


config = Config().data

if __name__ == "__main__":
    print(config)
