#!/usr/bin/env python3
"""Build a complete keyframe scene-memory dataset from one recording session."""

from __future__ import annotations

import argparse
import json
import os
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from caragent_memory.dataset import write_json
from caragent_memory.select_keyframes import select_keyframes


ANNOTATE_MODES = ("auto", "always", "never")
CHUNK_INDEX_MODES = ("auto", "always", "never")


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _workspace() -> Path:
    return Path(os.environ.get("CARAGENT_WORKSPACE", "~/caragent_ws")).expanduser()


def _candidate_dataset(path: Path) -> Path:
    dataset = path.expanduser().resolve()
    if dataset.name == "selected":
        return dataset.parent.resolve()
    return dataset


def _selected_output(dataset: Path, output: Path | None) -> Path:
    if output is not None:
        return output.expanduser().resolve()
    return dataset / "selected"


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def _count_keyframe_nodes(selected: Path) -> dict[str, Any]:
    node_dir = selected / "constructed_memory" / "keyframe_nodes"
    nodes = []
    semantic_nodes = 0
    if node_dir.exists():
        for path in sorted(node_dir.glob("kf_*.json")):
            nodes.append(path)
            data = _load_json(path)
            if str(data.get("semantic") or "").strip():
                semantic_nodes += 1
    return {
        "keyframe_nodes": len(nodes),
        "semantic_nodes": semantic_nodes,
    }


def _chunk_index_stats(selected: Path) -> dict[str, Any]:
    root = selected / "constructed_memory"
    records_path = root / "semantic_chunk_index_records.json"
    matrix_path = root / "semantic_chunk_index_matrix.npy"
    stats: dict[str, Any] = {
        "records_path": str(records_path) if records_path.exists() else "",
        "matrix_path": str(matrix_path) if matrix_path.exists() else "",
        "record_count": 0,
        "matrix_shape": [],
        "backend": "",
    }
    if records_path.exists():
        payload = _load_json(records_path)
        records = payload.get("records") if isinstance(payload, dict) else []
        metadata = payload.get("metadata") if isinstance(payload, dict) else {}
        stats["record_count"] = len(records) if isinstance(records, list) else 0
        if isinstance(metadata, dict):
            stats["backend"] = str(metadata.get("backend") or "")
    if matrix_path.exists():
        try:
            matrix = np.load(matrix_path, mmap_mode="r")
            stats["matrix_shape"] = [int(value) for value in matrix.shape]
        except Exception:
            stats["matrix_shape"] = []
    return stats


def _existing_selection_summary(dataset: Path, selected: Path) -> dict[str, Any]:
    summary = _load_json(selected / "selection_summary.json")
    selection = dict(summary)
    selection.setdefault("dataset", str(dataset.resolve()))
    selection.setdefault("output", str(selected.resolve()))
    if selection.get("candidate_count") is None:
        selection["candidate_count"] = _count_jsonl(dataset / "manifest.jsonl")
    if selection.get("selected_count") is None:
        selection["selected_count"] = _count_jsonl(selected / "selected_manifest.jsonl")
    if selection.get("rejected_count") is None:
        selection["rejected_count"] = _count_jsonl(selected / "rejected_manifest.jsonl")
    selection["adopted_existing"] = True
    return selection


def _validate_existing_selected(selected: Path) -> None:
    node_dir = selected / "constructed_memory" / "keyframe_nodes"
    graph_path = selected / "constructed_memory" / "keyframe_graph.json"
    if not selected.exists():
        raise FileNotFoundError(f"selected dataset not found: {selected}")
    if not node_dir.exists() or not any(node_dir.glob("kf_*.json")):
        raise FileNotFoundError(f"keyframe nodes not found: {node_dir}")
    if not graph_path.exists():
        raise FileNotFoundError(f"keyframe graph not found: {graph_path}")


def _has_qwen_api_key() -> bool:
    if os.environ.get("DASHSCOPE_API_KEY", "").strip():
        return True
    if os.environ.get("DASHSCOPE_API_KEYS", "").strip():
        return True
    try:
        from caragent_agent.config.config import get_api_keys

        return bool(get_api_keys("qwen"))
    except Exception:
        return False


def _run_annotation(
    selected: Path,
    *,
    mode: str,
    model: str,
    batch_size: int,
    force: bool,
    skip_clip: bool,
) -> dict[str, Any]:
    if mode == "never":
        return {"status": "skipped", "reason": "annotation_disabled"}
    if mode == "auto" and not _has_qwen_api_key():
        return {"status": "skipped", "reason": "dashscope_api_key_unavailable"}

    try:
        from caragent_agent.config.config import config
        from caragent_agent.scripts.annotate_keyframes import annotate

        resolved_model = model or str(config.get("vlm_model_get_semantic") or "qwen3-vl-flash")
        annotated = annotate(
            selected,
            model=resolved_model,
            batch_size=max(1, int(batch_size)),
            force=bool(force),
            compute_clip=not bool(skip_clip),
        )
        node_counts = _count_keyframe_nodes(selected)
        return {
            "status": "ok",
            "model": resolved_model,
            "annotated_count": int(annotated),
            **node_counts,
        }
    except Exception as exc:
        if mode == "always":
            raise
        return {
            "status": "failed",
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }


def _run_chunk_index(
    selected: Path,
    *,
    mode: str,
    device: str,
    force: bool,
) -> dict[str, Any]:
    if mode == "never":
        return {"status": "skipped", "reason": "chunk_index_disabled"}

    node_counts = _count_keyframe_nodes(selected)
    if mode == "auto" and int(node_counts.get("semantic_nodes") or 0) <= 0:
        return {"status": "skipped", "reason": "no_semantic_nodes", **node_counts}

    try:
        from caragent_agent.agents.tools.search.requirement_search import (
            build_persistent_semantic_chunk_index,
        )
        from caragent_agent.impression_graph.scene_memory import SceneMemory

        scene_memory = SceneMemory(dataset_dir=selected, device=device)
        result = build_persistent_semantic_chunk_index(
            scene_memory,
            force_rebuild=bool(force),
        )
        if mode == "always" and result.get("status") != "ok":
            raise RuntimeError(f"semantic chunk index was not built: {result}")
        return result
    except Exception as exc:
        if mode == "always":
            raise
        return {
            "status": "failed",
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path)


def _write_scene_memory_manifest(selected: Path, summary: dict[str, Any]) -> dict[str, Any]:
    constructed = selected / "constructed_memory"
    chunk_stats = _chunk_index_stats(selected)
    node_counts = _count_keyframe_nodes(selected)
    manifest = {
        "format": "caragent_scene_memory",
        "version": 1,
        "generated_at": _now_iso(),
        "dataset_dir": str(selected.resolve()),
        "source_dataset": str(Path(summary.get("source_dataset") or "").expanduser())
        if summary.get("source_dataset")
        else "",
        "status": summary.get("status"),
        "counts": {
            "candidate_frames": summary.get("selection", {}).get("candidate_count"),
            "selected_frames": summary.get("selection", {}).get("selected_count"),
            "rejected_frames": summary.get("selection", {}).get("rejected_count"),
            **node_counts,
            "semantic_chunks": chunk_stats.get("record_count"),
        },
        "artifacts": {
            "selected_manifest": _relative(selected / "selected_manifest.jsonl", selected),
            "rejected_manifest": _relative(selected / "rejected_manifest.jsonl", selected),
            "review_html": _relative(selected / "review.html", selected),
            "keyframe_nodes_dir": _relative(constructed / "keyframe_nodes", selected),
            "keyframe_graph": _relative(constructed / "keyframe_graph.json", selected),
            "semantic_chunk_index_records": (
                _relative(Path(chunk_stats["records_path"]), selected)
                if chunk_stats.get("records_path")
                else ""
            ),
            "semantic_chunk_index_matrix": (
                _relative(Path(chunk_stats["matrix_path"]), selected)
                if chunk_stats.get("matrix_path")
                else ""
            ),
        },
        "build": {
            "selection": summary.get("selection", {}),
            "annotation": summary.get("annotation", {}),
            "chunk_index": summary.get("chunk_index", {}),
        },
    }
    write_json(constructed / "scene_memory_manifest.json", manifest)
    return manifest


def build_scene_memory(
    *,
    dataset: Path,
    output: Path | None,
    clip_model: Path,
    device: str,
    clip_device: str | None,
    dinov2_model: Path,
    dinov2_backend: str,
    dinov2_openvino_model: Path,
    dinov2_device: str,
    dinov2_allow_download: bool,
    dedupe_backend: str,
    search_radius_m: float,
    near_duplicate_distance_m: float,
    yaw_keep_deg: float,
    dedupe_keep_similarity: float,
    dedupe_duplicate_similarity: float,
    annotate_mode: str,
    annotation_model: str,
    annotation_batch_size: int,
    annotation_force: bool,
    annotation_skip_clip: bool,
    chunk_index_mode: str,
    chunk_index_device: str,
    chunk_index_force: bool,
    adopt_existing: bool = False,
) -> dict[str, Any]:
    dataset = _candidate_dataset(dataset)
    selected = _selected_output(dataset, output)
    summary: dict[str, Any] = {
        "status": "running",
        "started_at": _now_iso(),
        "finished_at": "",
        "source_dataset": str(dataset),
        "selected_dataset": str(selected),
        "selection": {},
        "annotation": {},
        "chunk_index": {},
        "warnings": [],
    }

    try:
        if adopt_existing:
            _validate_existing_selected(selected)
            selection = _existing_selection_summary(dataset, selected)
        else:
            selection = select_keyframes(
                dataset=dataset,
                output=selected,
                clip_model=clip_model.expanduser().resolve(),
                device=device,
                clip_device=clip_device,
                dinov2_model=dinov2_model.expanduser(),
                dinov2_backend=dinov2_backend,
                dinov2_openvino_model=dinov2_openvino_model.expanduser(),
                dinov2_device=dinov2_device,
                dinov2_local_files_only=not bool(dinov2_allow_download),
                dedupe_backend=dedupe_backend,
                search_radius_m=search_radius_m,
                near_duplicate_distance_m=near_duplicate_distance_m,
                yaw_keep_deg=yaw_keep_deg,
                dedupe_keep_similarity=dedupe_keep_similarity,
                dedupe_duplicate_similarity=dedupe_duplicate_similarity,
            )
        summary["selection"] = selection

        annotation = _run_annotation(
            selected,
            mode=annotate_mode,
            model=annotation_model,
            batch_size=annotation_batch_size,
            force=annotation_force,
            skip_clip=annotation_skip_clip,
        )
        summary["annotation"] = annotation
        if annotation.get("status") == "failed":
            summary["warnings"].append("annotation_failed")

        chunk_index = _run_chunk_index(
            selected,
            mode=chunk_index_mode,
            device=chunk_index_device or device or "CPU",
            force=chunk_index_force,
        )
        summary["chunk_index"] = chunk_index
        if chunk_index.get("status") == "failed":
            summary["warnings"].append("chunk_index_failed")

        summary["status"] = "partial" if summary["warnings"] else "ok"
    except Exception as exc:
        summary["status"] = "failed"
        summary["error"] = str(exc)
        summary["traceback"] = traceback.format_exc()
        selected.mkdir(parents=True, exist_ok=True)
        write_json(selected / "scene_memory_summary.json", summary)
        raise
    finally:
        summary["finished_at"] = _now_iso()

    manifest = _write_scene_memory_manifest(selected, summary)
    summary["manifest"] = str(selected / "constructed_memory" / "scene_memory_manifest.json")
    summary["counts"] = manifest.get("counts", {})
    summary["chunk_index_stats"] = _chunk_index_stats(selected)
    write_json(selected / "scene_memory_summary.json", summary)
    return summary


def _parse_args() -> argparse.Namespace:
    workspace = _workspace()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path, help="Recording session or selected dataset directory.")
    parser.add_argument("--output", type=Path, default=None, help="Selected dataset output. Defaults to <dataset>/selected.")
    parser.add_argument(
        "--adopt-existing",
        action="store_true",
        help=(
            "Do not rerun keyframe selection. Instead, wrap an existing selected/ "
            "dataset with the unified scene-memory summary and manifest."
        ),
    )
    parser.add_argument(
        "--clip-model",
        type=Path,
        default=workspace / "models" / "clip-vit-base-patch32" / "image_encoder.xml",
    )
    parser.add_argument("--device", default="GPU", help="OpenVINO CLIP image device.")
    parser.add_argument("--clip-device", default=None)
    parser.add_argument(
        "--dinov2-model",
        type=Path,
        default=workspace / "models" / "dinov2",
    )
    parser.add_argument("--dinov2-backend", choices=("openvino", "torch"), default="openvino")
    parser.add_argument(
        "--dinov2-openvino-model",
        type=Path,
        default=workspace / "models" / "dinov2-small-openvino" / "openvino_model.xml",
    )
    parser.add_argument("--dinov2-device", default="NPU")
    parser.add_argument("--dinov2-allow-download", action="store_true")
    parser.add_argument("--dedupe-backend", choices=("dinov2", "clip"), default="dinov2")
    parser.add_argument("--search-radius-m", type=float, default=2.0)
    parser.add_argument("--near-duplicate-distance-m", type=float, default=0.35)
    parser.add_argument("--yaw-keep-deg", type=float, default=35.0)
    parser.add_argument("--dedupe-keep-similarity", type=float, default=0.90)
    parser.add_argument("--dedupe-duplicate-similarity", type=float, default=0.85)
    parser.add_argument("--annotate", choices=ANNOTATE_MODES, default="auto")
    parser.add_argument("--annotation-model", default="")
    parser.add_argument("--annotation-batch-size", type=int, default=5)
    parser.add_argument("--annotation-force", action="store_true")
    parser.add_argument("--annotation-skip-clip", action="store_true")
    parser.add_argument("--chunk-index", choices=CHUNK_INDEX_MODES, default="auto")
    parser.add_argument("--chunk-index-device", default="")
    parser.add_argument("--chunk-index-force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    try:
        summary = build_scene_memory(
            dataset=args.dataset,
            output=args.output,
            clip_model=args.clip_model,
            device=args.device,
            clip_device=args.clip_device,
            dinov2_model=args.dinov2_model,
            dinov2_backend=args.dinov2_backend,
            dinov2_openvino_model=args.dinov2_openvino_model,
            dinov2_device=args.dinov2_device,
            dinov2_allow_download=args.dinov2_allow_download,
            dedupe_backend=args.dedupe_backend,
            search_radius_m=args.search_radius_m,
            near_duplicate_distance_m=args.near_duplicate_distance_m,
            yaw_keep_deg=args.yaw_keep_deg,
            dedupe_keep_similarity=args.dedupe_keep_similarity,
            dedupe_duplicate_similarity=args.dedupe_duplicate_similarity,
            annotate_mode=args.annotate,
            annotation_model=args.annotation_model,
            annotation_batch_size=args.annotation_batch_size,
            annotation_force=args.annotation_force,
            annotation_skip_clip=args.annotation_skip_clip,
            chunk_index_mode=args.chunk_index,
            chunk_index_device=args.chunk_index_device,
            chunk_index_force=args.chunk_index_force,
            adopt_existing=args.adopt_existing,
        )
    except Exception as exc:
        raise SystemExit(f"build_scene_memory failed: {exc}") from exc
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
