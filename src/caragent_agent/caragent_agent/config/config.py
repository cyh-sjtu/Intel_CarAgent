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

API_POOL_ENV_BY_PROVIDER = {
    "qwen": "DASHSCOPE_API_KEYS",
    "dashscope": "DASHSCOPE_API_KEYS",
}

API_CONFIG_KEYS_BY_PROVIDER = {
    "qwen": ("qwen-api", "qwen", "dashscope", "dashscope_keys"),
    "dashscope": ("qwen-api", "qwen", "dashscope", "dashscope_keys"),
    "deepseek": ("deepseek-api", "deepseek", "deepseek_keys"),
    "doubao": ("doubao-api", "doubao", "doubao_keys"),
}

TOP_LEVEL_API_KEYS_BY_PROVIDER = {
    "qwen": (
        "qwen-api",
        "qwen_api_key",
        "qwen_api_keys",
        "dashscope",
        "dashscope_api_key",
        "dashscope_api_keys",
    ),
    "dashscope": (
        "qwen-api",
        "qwen_api_key",
        "qwen_api_keys",
        "dashscope",
        "dashscope_api_key",
        "dashscope_api_keys",
    ),
    "deepseek": ("deepseek-api", "deepseek_api_key", "deepseek_api_keys"),
    "doubao": ("doubao-api", "doubao_api_key", "doubao_api_keys"),
}


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {path}")
    return loaded


def _looks_like_mojibake(value: str) -> bool:
    """Return True for typical UTF-8 text decoded as a single-byte encoding."""

    if not value:
        return False
    markers = ("Ã", "Â", "Ð", "Ñ", "æ", "ç", "è", "é", "å", "ï¿½", "�")
    return any(marker in value for marker in markers)


def _repair_mojibake_text(value: str) -> str:
    """Repair common mojibake without changing ordinary valid strings."""

    if not _looks_like_mojibake(value):
        return value
    candidates = []
    for encoding in ("latin1", "cp1252"):
        try:
            repaired = value.encode(encoding).decode("utf-8")
        except Exception:
            continue
        candidates.append(repaired)
    if not candidates:
        return value

    def score(text: str) -> int:
        bad_markers = sum(text.count(marker) for marker in ("Ã", "Â", "æ", "ç", "è", "é", "�"))
        cjk_chars = sum(1 for ch in text if "\u3400" <= ch <= "\u9fff")
        return cjk_chars * 4 - bad_markers * 8

    best = max(candidates, key=score)
    return best if score(best) > score(value) else value


def _repair_config_text(value: Any) -> Any:
    """Recursively repair text values loaded from config files."""

    if isinstance(value, str):
        return _repair_mojibake_text(value)
    if isinstance(value, dict):
        return {key: _repair_config_text(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_repair_config_text(item) for item in value]
    return value


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

    return _repair_config_text(data)

class Config:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.data = load_config()
                    
        return cls._instance


def _coerce_api_key_list(raw_value: Any) -> list[str]:
    """Normalize API key config values into an ordered de-duplicated list."""

    values: list[str] = []

    if raw_value is None:
        return values
    if isinstance(raw_value, str):
        values.extend(
            item.strip()
            for item in raw_value.replace("\n", ",").split(",")
            if item.strip()
        )
    elif isinstance(raw_value, (list, tuple, set)):
        for item in raw_value:
            values.extend(_coerce_api_key_list(item))
    elif isinstance(raw_value, dict):
        matched_named_field = False
        for key in ("keys", "api_keys", "key", "api_key", "value"):
            if key in raw_value:
                matched_named_field = True
                values.extend(_coerce_api_key_list(raw_value.get(key)))
        if not matched_named_field:
            for item in raw_value.values():
                values.extend(_coerce_api_key_list(item))
    else:
        text = str(raw_value).strip()
        if text:
            values.append(text)

    deduped: list[str] = []
    for value in values:
        if value and value not in deduped:
            deduped.append(value)
    return deduped


def get_api_keys(provider: str) -> list[str]:
    """Return configured API keys from env vars first, then ignored local config."""

    normalized = str(provider or "").strip().lower()
    keys: list[str] = []

    pool_env_var = API_POOL_ENV_BY_PROVIDER.get(normalized)
    if pool_env_var:
        keys.extend(_coerce_api_key_list(os.environ.get(pool_env_var, "")))

    env_var = API_ENV_BY_PROVIDER.get(normalized)
    if env_var:
        keys.extend(_coerce_api_key_list(os.environ.get(env_var, "")))

    api_keys = config.get("api_keys")
    if isinstance(api_keys, dict):
        for key in API_CONFIG_KEYS_BY_PROVIDER.get(normalized, ()):
            keys.extend(_coerce_api_key_list(api_keys.get(key)))

    for key in TOP_LEVEL_API_KEYS_BY_PROVIDER.get(normalized, ()):
        keys.extend(_coerce_api_key_list(config.get(key)))

    deduped: list[str] = []
    for key in keys:
        if key and key not in deduped:
            deduped.append(key)
    return deduped


def get_api_key(provider: str) -> str:
    """Return the first configured API key for provider compatibility paths."""

    keys = get_api_keys(provider)
    return keys[0] if keys else ""


def ensure_api_key_env(provider: str) -> str:
    """Populate the provider env var from config when a local key is available."""

    normalized = str(provider or "").strip().lower()
    api_keys = get_api_keys(normalized)
    api_key = api_keys[0] if api_keys else ""
    env_var = API_ENV_BY_PROVIDER.get(normalized)
    if api_key and env_var and not os.environ.get(env_var):
        os.environ[env_var] = api_key
    pool_env_var = API_POOL_ENV_BY_PROVIDER.get(normalized)
    if api_keys and pool_env_var and not os.environ.get(pool_env_var):
        os.environ[pool_env_var] = ",".join(api_keys)
    return api_key


config = Config().data

if __name__ == "__main__":
    print(config)
