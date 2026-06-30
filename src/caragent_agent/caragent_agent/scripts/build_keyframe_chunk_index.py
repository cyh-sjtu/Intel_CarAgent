"""Build a persistent semantic chunk index for keyframe scene memory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from caragent_agent.config.config import config
from caragent_agent.impression_graph.scene_memory import SceneMemory
from caragent_agent.agents.tools.search.requirement_search import (
    build_persistent_semantic_chunk_index,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Precompute semantic chunk embeddings for keyframe retrieval. "
            "Run this after keyframe selection/annotation so the agent does not "
            "build the chunk index during the first live navigation request."
        )
    )
    parser.add_argument(
        "--dataset",
        default="",
        help="Selected keyframe dataset directory. Defaults to config scene_memory.dataset_dir.",
    )
    parser.add_argument(
        "--device",
        default="",
        help="OpenVINO text encoder device. Defaults to config scene_memory.device.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild even if a matching persisted index already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scene_cfg = config.get("scene_memory", {}) or {}
    paths_cfg = config.get("paths", {}) or {}
    dataset = (
        args.dataset
        or scene_cfg.get("dataset_dir")
        or paths_cfg.get("default_dataset_dir")
    )
    if not dataset:
        raise SystemExit("No dataset provided and config has no scene_memory.dataset_dir.")
    dataset_path = Path(dataset).expanduser().resolve()
    device = args.device or scene_cfg.get("device") or "CPU"

    scene_memory = SceneMemory(dataset_dir=dataset_path, device=device)
    result = build_persistent_semantic_chunk_index(
        scene_memory,
        force_rebuild=bool(args.force),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
