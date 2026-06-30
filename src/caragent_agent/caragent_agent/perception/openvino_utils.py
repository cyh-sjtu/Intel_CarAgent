"""Small OpenVINO runtime helpers used by perception wrappers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


DEFAULT_OPENVINO_CACHE_DIR = Path(
    os.environ.get("CARAGENT_OPENVINO_CACHE_DIR", "/home/car/caragent_ws/.openvino_cache")
)


def create_openvino_core(*, cache_dir: str | Path | None = None) -> Any:
    """Create an OpenVINO Core with compile cache enabled when possible."""

    import openvino as ov

    core = ov.Core()
    resolved_cache = Path(cache_dir) if cache_dir is not None else DEFAULT_OPENVINO_CACHE_DIR
    try:
        resolved_cache.mkdir(parents=True, exist_ok=True)
        core.set_property({"CACHE_DIR": str(resolved_cache)})
    except Exception:
        # Cache is an optimization only; perception must still run if the
        # runtime or filesystem does not allow setting it.
        pass
    return core
