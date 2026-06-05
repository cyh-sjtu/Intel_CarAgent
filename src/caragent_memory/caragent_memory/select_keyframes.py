#!/usr/bin/env python3
"""Offline keyframe selection with CLIP semantics and DINOv2 visual deduplication."""

from __future__ import annotations

import argparse
import html
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from caragent_memory.dataset import (
    FrameRecord,
    append_jsonl,
    copy_record_assets,
    iter_frame_records,
    write_json,
)
from caragent_memory.dinov2_encoder import DINOv2ImageEncoder
from caragent_memory.geometry import planar_distance, yaw_difference_deg
from caragent_memory.openvino_clip import OpenVINOClipImageEncoder, cosine_similarity


@dataclass
class SelectedRecord:
    record: FrameRecord
    clip_embedding: np.ndarray
    dinov2_embedding: np.ndarray
    manifest: dict


@dataclass
class FrameEmbeddings:
    clip: np.ndarray
    dinov2: np.ndarray


def _make_selected_record(
    record: FrameRecord,
    embeddings: FrameEmbeddings,
    manifest: dict,
) -> SelectedRecord:
    return SelectedRecord(
        record=record,
        clip_embedding=embeddings.clip,
        dinov2_embedding=embeddings.dinov2,
        manifest=manifest,
    )


def _dedupe_embedding(embeddings: FrameEmbeddings | SelectedRecord, backend: str) -> np.ndarray:
    if backend == "clip":
        return embeddings.clip_embedding if isinstance(embeddings, SelectedRecord) else embeddings.clip
    if backend == "dinov2":
        return embeddings.dinov2_embedding if isinstance(embeddings, SelectedRecord) else embeddings.dinov2
    raise ValueError(f"unsupported dedupe backend: {backend}")


def _quality_ok(record: FrameRecord) -> bool:
    return bool(record.meta.get("quality_ok", False))


def _is_manual(record: FrameRecord) -> bool:
    return bool(record.meta.get("manual", False))


def _pose_for_node(record: FrameRecord) -> tuple[list[float], list[float]]:
    pose = record.pose
    return (
        [float(pose.get("x", 0.0)), float(pose.get("y", 0.0)), float(pose.get("z", 0.0))],
        list(pose.get("orientation_xyzw", [0.0, 0.0, 0.0, 1.0])),
    )


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _copy_selected_record(
    *,
    record: FrameRecord,
    embeddings: FrameEmbeddings,
    output_root: Path,
    source_dataset: Path,
    reason: str,
    max_similarity: Optional[float],
    nearest_distance_m: Optional[float],
    dedupe_backend: str,
) -> dict:
    copied = copy_record_assets(record, output_root)

    embedding_dir = output_root / "embeddings"
    clip_embedding_dir = embedding_dir / "clip"
    dinov2_embedding_dir = embedding_dir / "dinov2"
    clip_embedding_dir.mkdir(parents=True, exist_ok=True)
    dinov2_embedding_dir.mkdir(parents=True, exist_ok=True)
    clip_embedding_path = clip_embedding_dir / f"{record.frame_id}.npy"
    dinov2_embedding_path = dinov2_embedding_dir / f"{record.frame_id}.npy"
    np.save(clip_embedding_path, embeddings.clip.astype(np.float32))
    np.save(dinov2_embedding_path, embeddings.dinov2.astype(np.float32))
    dedupe_embedding_path = dinov2_embedding_path if dedupe_backend == "dinov2" else clip_embedding_path

    node_dir = output_root / "constructed_memory" / "keyframe_nodes"
    node_dir.mkdir(parents=True, exist_ok=True)
    position, orientation = _pose_for_node(record)
    node_payload = {
        "kf_id": int(record.frame_id),
        "name": record.frame_id,
        "dataset_dir": str(output_root.resolve()),
        "position": position,
        "orientation": orientation,
        "intrinsic": [],
        "timestamp": record.pose.get("timestamp"),
        "semantic": "",
        "clip_encoding": embeddings.clip.astype(float).tolist(),
        "dinov2_encoding": embeddings.dinov2.astype(float).tolist(),
        "semantic_clip_encoding": None,
        "visual_similarity_backend": dedupe_backend,
        "rgb_path": copied["left_path"],
        "raw_path": copied["raw_path"],
        "right_path": copied["right_path"],
        "pose_path": copied["pose_path"],
        "scan_path": copied["scan_path"],
        "source_dataset": str(source_dataset.resolve()),
        "source_frame_id": record.frame_id,
    }
    write_json(node_dir / f"kf_{record.frame_id}.json", node_payload)

    return {
        **copied,
        "embedding_path": _relative(dedupe_embedding_path, output_root),
        "clip_embedding_path": _relative(clip_embedding_path, output_root),
        "dinov2_embedding_path": _relative(dinov2_embedding_path, output_root),
        "visual_similarity_backend": dedupe_backend,
        "clip_embedding_dim": int(embeddings.clip.reshape(-1).shape[0]),
        "dinov2_embedding_dim": int(embeddings.dinov2.reshape(-1).shape[0]),
        "selected_reason": reason,
        "max_similarity": max_similarity,
        "nearest_distance_m": nearest_distance_m,
        "quality_ok": _quality_ok(record),
        "manual": _is_manual(record),
        "timestamp": record.pose.get("timestamp"),
        "x": float(record.pose.get("x", 0.0)),
        "y": float(record.pose.get("y", 0.0)),
        "yaw": float(record.pose.get("yaw", 0.0)),
    }


def _nearest_distance(record: FrameRecord, selected: list[SelectedRecord]) -> Optional[float]:
    if not selected:
        return None
    return min(planar_distance(record.pose, item.record.pose) for item in selected)


def _nearby_selected(
    record: FrameRecord,
    selected: list[SelectedRecord],
    radius_m: float,
) -> list[SelectedRecord]:
    return [
        item
        for item in selected
        if planar_distance(record.pose, item.record.pose) <= radius_m
    ]


def select_keyframes(
    *,
    dataset: Path,
    output: Path,
    clip_model: Path,
    device: Optional[str] = None,
    clip_device: Optional[str] = None,
    dinov2_model: str | Path = Path("~/caragent_ws/models/dinov2"),
    dinov2_device: str = "auto",
    dinov2_local_files_only: bool = True,
    dedupe_backend: str = "dinov2",
    search_radius_m: float = 2.0,
    near_duplicate_distance_m: float = 0.35,
    yaw_keep_deg: float = 35.0,
    dedupe_keep_similarity: float = 0.90,
    dedupe_duplicate_similarity: float = 0.85,
) -> dict:
    if output.resolve() == dataset.resolve():
        raise ValueError("--output must not be the same directory as --dataset")
    dedupe_backend = str(dedupe_backend).lower()
    if dedupe_backend not in {"dinov2", "clip"}:
        raise ValueError("--dedupe-backend must be either 'dinov2' or 'clip'")
    resolved_clip_device = str(clip_device or device or "GPU")

    records = list(iter_frame_records(dataset))
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "constructed_memory").mkdir(parents=True, exist_ok=True)
    session_path = dataset / "session.json"
    if session_path.exists():
        shutil.copy2(session_path, output / "source_session.json")

    clip_encoder = OpenVINOClipImageEncoder(clip_model, device=resolved_clip_device)
    dinov2_encoder = DINOv2ImageEncoder(
        dinov2_model,
        device=dinov2_device,
        local_files_only=dinov2_local_files_only,
    )
    selected: list[SelectedRecord] = []
    rejected = []

    for record in records:
        quality_ok = _quality_ok(record)
        manual = _is_manual(record)
        if not quality_ok and not manual:
            rejected.append(_reject_record(record, "quality", None, None))
            continue

        embeddings = FrameEmbeddings(
            clip=clip_encoder.encode_path(record.left_path),
            dinov2=dinov2_encoder.encode_path(record.left_path),
        )
        dedupe_embedding = _dedupe_embedding(embeddings, dedupe_backend)
        nearest_distance = _nearest_distance(record, selected)
        if not selected:
            manifest = _copy_selected_record(
                record=record,
                embeddings=embeddings,
                output_root=output,
                source_dataset=dataset,
                reason="first" if quality_ok else "manual_low_quality",
                max_similarity=None,
                nearest_distance_m=nearest_distance,
                dedupe_backend=dedupe_backend,
            )
            selected.append(_make_selected_record(record, embeddings, manifest))
            append_jsonl(output / "selected_manifest.jsonl", manifest)
            continue

        nearby = _nearby_selected(record, selected, search_radius_m)
        if not nearby:
            manifest = _copy_selected_record(
                record=record,
                embeddings=embeddings,
                output_root=output,
                source_dataset=dataset,
                reason="spatial_coverage",
                max_similarity=None,
                nearest_distance_m=nearest_distance,
                dedupe_backend=dedupe_backend,
            )
            selected.append(_make_selected_record(record, embeddings, manifest))
            append_jsonl(output / "selected_manifest.jsonl", manifest)
            continue

        similarities = [
            cosine_similarity(dedupe_embedding, _dedupe_embedding(item, dedupe_backend))
            for item in nearby
        ]
        max_similarity = max(similarities) if similarities else None
        nearest_item = min(nearby, key=lambda item: planar_distance(record.pose, item.record.pose))
        nearest_yaw_delta = yaw_difference_deg(record.pose["yaw"], nearest_item.record.pose["yaw"])

        if (
            nearest_distance is not None
            and nearest_distance < near_duplicate_distance_m
            and max_similarity is not None
            and max_similarity >= dedupe_duplicate_similarity
        ):
            rejected.append(_reject_record(record, "near_duplicate", max_similarity, nearest_distance))
            continue

        if manual:
            keep = True
            reason = "manual"
        elif nearest_yaw_delta >= yaw_keep_deg:
            keep = True
            reason = "yaw"
        elif max_similarity is not None and max_similarity < dedupe_keep_similarity:
            keep = True
            reason = f"{dedupe_backend}_novelty"
        else:
            keep = False
            reason = "similar"

        if keep:
            manifest = _copy_selected_record(
                record=record,
                embeddings=embeddings,
                output_root=output,
                source_dataset=dataset,
                reason=reason,
                max_similarity=max_similarity,
                nearest_distance_m=nearest_distance,
                dedupe_backend=dedupe_backend,
            )
            selected.append(_make_selected_record(record, embeddings, manifest))
            append_jsonl(output / "selected_manifest.jsonl", manifest)
        else:
            rejected.append(_reject_record(record, reason, max_similarity, nearest_distance))

    for item in rejected:
        append_jsonl(output / "rejected_manifest.jsonl", item)

    _write_keyframe_graph(output, selected)
    _write_review_html(output, selected, rejected)
    summary = {
        "dataset": str(dataset.resolve()),
        "output": str(output.resolve()),
        "clip_model": str(clip_model.resolve()),
        "clip_device": resolved_clip_device,
        "dinov2_model": (
            str(Path(dinov2_model).expanduser().resolve())
            if Path(str(dinov2_model)).expanduser().exists()
            else str(dinov2_model)
        ),
        "dinov2_device": str(dinov2_encoder.device),
        "visual_similarity_backend": dedupe_backend,
        "candidate_count": len(records),
        "selected_count": len(selected),
        "rejected_count": len(rejected),
        "parameters": {
            "search_radius_m": search_radius_m,
            "near_duplicate_distance_m": near_duplicate_distance_m,
            "yaw_keep_deg": yaw_keep_deg,
            "dedupe_keep_similarity": dedupe_keep_similarity,
            "dedupe_duplicate_similarity": dedupe_duplicate_similarity,
        },
    }
    write_json(output / "selection_summary.json", summary)
    return summary


def _reject_record(
    record: FrameRecord,
    reason: str,
    max_similarity: Optional[float],
    nearest_distance_m: Optional[float],
) -> dict:
    return {
        "frame_id": record.frame_id,
        "reject_reason": reason,
        "quality_ok": _quality_ok(record),
        "manual": _is_manual(record),
        "max_similarity": max_similarity,
        "nearest_distance_m": nearest_distance_m,
        "timestamp": record.pose.get("timestamp"),
        "x": float(record.pose.get("x", 0.0)),
        "y": float(record.pose.get("y", 0.0)),
        "yaw": float(record.pose.get("yaw", 0.0)),
    }


def _write_keyframe_graph(output: Path, selected: list[SelectedRecord]) -> None:
    edges = []
    for index in range(len(selected) - 1):
        current = selected[index].record
        nxt = selected[index + 1].record
        distance = planar_distance(current.pose, nxt.pose)
        edges.append([int(current.frame_id), int(nxt.frame_id), {"weight": float(distance), "type": "sequential"}])

    graph = {
        "nodes": [int(item.record.frame_id) for item in selected],
        "edges": edges,
    }
    write_json(output / "constructed_memory" / "keyframe_graph.json", graph)


def _write_review_html(output: Path, selected: list[SelectedRecord], rejected: list[dict]) -> None:
    cards = []
    for item in selected:
        manifest = item.manifest
        # Use absolute path via /api/file so images work regardless of how the HTML is served
        left_abs = str((output / manifest["left_path"]).resolve())
        img_src = html.escape(f"/api/file?path={left_abs}")
        quality = "ok" if manifest["quality_ok"] else "manual-low-quality"
        sim = manifest["max_similarity"]
        sim_text = "" if sim is None else f" sim={sim:.3f}"
        cards.append(
            "<article>"
            f"<img src='{img_src}' alt='kf {manifest['frame_id']}'>"
            f"<h2>#{html.escape(manifest['frame_id'])} {html.escape(manifest['selected_reason'])}</h2>"
            f"<p>x={manifest['x']:.2f} y={manifest['y']:.2f} yaw={math.degrees(manifest['yaw']):.1f} deg</p>"
            f"<p>{quality}{html.escape(sim_text)}</p>"
            "</article>"
        )

    html_text = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>CarAgent Keyframe Review</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 24px; background: #f5f5f5; color: #1f2933; }
    .summary { margin-bottom: 16px; }
    .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 12px; }
    article { background: white; border: 1px solid #d7dde4; border-radius: 6px; padding: 10px; }
    img { width: 100%; aspect-ratio: 4 / 3; object-fit: cover; background: #111; }
    h2 { font-size: 14px; margin: 8px 0 4px; }
    p { font-size: 12px; margin: 3px 0; }
  </style>
</head>
<body>
  <section class="summary">
    <h1>CarAgent Keyframe Review</h1>
    <p>Selected: SELECTED_COUNT | Rejected: REJECTED_COUNT</p>
  </section>
  <section class="grid">
    CARDS
  </section>
</body>
</html>
"""
    html_text = html_text.replace("SELECTED_COUNT", str(len(selected)))
    html_text = html_text.replace("REJECTED_COUNT", str(len(rejected)))
    html_text = html_text.replace("CARDS", "\n".join(cards))
    (output / "review.html").write_text(html_text, encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path, help="Candidate keyframe dataset directory.")
    parser.add_argument("--output", type=Path, default=None, help="Output directory. Defaults to <dataset>/selected.")
    parser.add_argument("--clip-model", required=True, type=Path, help="OpenVINO CLIP image encoder .xml file.")
    parser.add_argument("--device", default="AUTO", help="OpenVINO CLIP device, e.g. AUTO, CPU, GPU, NPU.")
    parser.add_argument("--clip-device", default=None, help="Alias for --device when naming devices explicitly.")
    parser.add_argument(
        "--dinov2-model",
        type=Path,
        default=Path("~/caragent_ws/models/dinov2"),
        help="Local Hugging Face DINOv2-small model directory or model id.",
    )
    parser.add_argument("--dinov2-device", default="auto", help="DINOv2 torch device: auto, cpu, cuda, cuda:0.")
    parser.add_argument(
        "--dinov2-allow-download",
        action="store_true",
        help="Allow transformers to download DINOv2 files if --dinov2-model is a remote model id.",
    )
    parser.add_argument(
        "--dedupe-backend",
        choices=("dinov2", "clip"),
        default="dinov2",
        help="Embedding backend used for frame-to-frame deduplication.",
    )
    parser.add_argument("--search-radius-m", type=float, default=2.0,
                        help="Spatial radius for nearby-frame comparison. Frames outside this radius "
                             "are kept for spatial coverage without similarity check.")
    parser.add_argument("--near-duplicate-distance-m", type=float, default=0.35)
    parser.add_argument("--yaw-keep-deg", type=float, default=35.0)
    parser.add_argument("--dedupe-keep-similarity", type=float, default=None)
    parser.add_argument("--dedupe-duplicate-similarity", type=float, default=None)
    parser.add_argument(
        "--clip-keep-similarity",
        type=float,
        default=None,
        help="Deprecated compatibility alias for --dedupe-keep-similarity.",
    )
    parser.add_argument(
        "--clip-duplicate-similarity",
        type=float,
        default=None,
        help="Deprecated compatibility alias for --dedupe-duplicate-similarity.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    dataset = args.dataset.expanduser().resolve()
    output = (args.output.expanduser().resolve() if args.output else dataset / "selected")
    dedupe_keep_similarity = (
        args.dedupe_keep_similarity
        if args.dedupe_keep_similarity is not None
        else (args.clip_keep_similarity if args.clip_keep_similarity is not None else 0.90)
    )
    dedupe_duplicate_similarity = (
        args.dedupe_duplicate_similarity
        if args.dedupe_duplicate_similarity is not None
        else (args.clip_duplicate_similarity if args.clip_duplicate_similarity is not None else 0.85)
    )
    try:
        summary = select_keyframes(
            dataset=dataset,
            output=output,
            clip_model=args.clip_model.expanduser().resolve(),
            device=args.device,
            clip_device=args.clip_device,
            dinov2_model=args.dinov2_model.expanduser(),
            dinov2_device=args.dinov2_device,
            dinov2_local_files_only=not args.dinov2_allow_download,
            dedupe_backend=args.dedupe_backend,
            search_radius_m=args.search_radius_m,
            near_duplicate_distance_m=args.near_duplicate_distance_m,
            yaw_keep_deg=args.yaw_keep_deg,
            dedupe_keep_similarity=dedupe_keep_similarity,
            dedupe_duplicate_similarity=dedupe_duplicate_similarity,
        )
    except Exception as exc:
        raise SystemExit(f"select_keyframes failed: {exc}") from exc
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
