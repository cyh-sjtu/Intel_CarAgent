"""File exports for simplified async-agent memory tables."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


MEMORY_TABLE_FILENAMES = {
    "conversation": "memory_conversation.json",
    "plan": "memory_plan.json",
    "task": "memory_task.json",
    "navigation": "memory_navigation.json",
    "observation": "memory_observation.json",
}


def write_memory_tables(
    session_dir: Path,
    tables: dict[str, list[dict[str, Any]]],
) -> None:
    """Write each simplified memory table as one JSON file."""

    session_dir.mkdir(parents=True, exist_ok=True)
    for scope, filename in MEMORY_TABLE_FILENAMES.items():
        table_path = session_dir / filename
        temp_path = table_path.with_name(f"{table_path.stem}_{os.getpid()}.tmp")
        temp_path.write_text(
            json.dumps(tables.get(scope, []), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temp_path.replace(table_path)


__all__ = ["MEMORY_TABLE_FILENAMES", "write_memory_tables"]
