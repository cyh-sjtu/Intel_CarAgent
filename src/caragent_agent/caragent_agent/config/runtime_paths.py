"""Runtime path helpers for scene memory paths."""

from __future__ import annotations

import os
from pathlib import Path

from caragent_agent.config.config import config


def _config_paths() -> dict:
    paths = config.get("paths")
    return paths if isinstance(paths, dict) else {}


def get_repo_root() -> Path:
    """Return the CarAgent agent package root, with optional env override."""

    override = os.environ.get("CARAGENT_REPO_ROOT", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return Path(__file__).resolve().parents[2]


def normalize_runtime_path(path_value: str | Path, *, base_dir: Path | None = None) -> Path:
    """Normalize one runtime path into an absolute Path.

    Relative config paths are resolved from the repository root by default.
    """

    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return ((base_dir or get_repo_root()) / path).resolve()


def _path_from_config(key: str) -> Path | None:
    value = _config_paths().get(key)
    if value is None or not str(value).strip():
        return None
    return normalize_runtime_path(value)


def get_scene_dataset_dir(scene_name: str = "default") -> Path:
    """Return the dataset directory for a named scene."""

    override = os.environ.get("CARAGENT_DATASET_DIR", "").strip()
    if override:
        return normalize_runtime_path(override)

    paths = _config_paths()

    scene_key = f"{scene_name}_dataset_dir"
    scene_dataset_dir = _path_from_config(scene_key)
    if scene_dataset_dir is not None:
        return scene_dataset_dir

    default_dataset_dir = _path_from_config("default_dataset_dir")
    if default_dataset_dir is not None:
        return default_dataset_dir

    scene_memory_root = (
        paths.get("scene_memory_root")
        or "Impression_graph/caragent_agent/pre-built_impression_graph"
    )
    return normalize_runtime_path(scene_memory_root) / scene_name


def get_scene_tasks_path(
    scene_name: str = "default",
    tasks_filename: str = "tasks_test.json",
) -> Path:
    """Return the tasks file path for a named scene."""

    override = os.environ.get("CARAGENT_TASKS_FILE", "").strip()
    if override:
        return normalize_runtime_path(override)

    task_key = f"{scene_name}_tasks_file"
    tasks_path = _path_from_config(task_key)
    if tasks_path is not None:
        return tasks_path

    configured_tasks = _path_from_config("default_tasks_path")
    if configured_tasks is not None:
        return configured_tasks

    tasks_filename = str(
        _config_paths().get("default_tasks_file") or tasks_filename
    )
    return get_scene_dataset_dir(scene_name) / tasks_filename


def get_default_scene_dataset_dir() -> Path:
    """Return the default scene-memory dataset directory."""

    override = os.environ.get("CARAGENT_DATASET_DIR", "").strip()
    if override:
        return normalize_runtime_path(override)

    configured = _path_from_config("default_dataset_dir")
    if configured is not None:
        return configured

    return get_scene_dataset_dir("default")


def get_default_scene_tasks_path(tasks_filename: str = "tasks_test.json") -> Path:
    """Return the default scene tasks file."""

    return get_scene_tasks_path("default", tasks_filename)


__all__ = [
    "get_default_scene_dataset_dir",
    "get_default_scene_tasks_path",
    "get_repo_root",
    "get_scene_dataset_dir",
    "get_scene_tasks_path",
    "normalize_runtime_path",
]
