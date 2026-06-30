import json

import numpy as np

from caragent_memory.build_scene_memory import (
    _existing_selection_summary,
    _chunk_index_stats,
    _validate_existing_selected,
    _write_scene_memory_manifest,
)


def test_scene_memory_manifest_collects_keyframe_and_chunk_artifacts(tmp_path):
    selected = tmp_path / "session_001" / "selected"
    node_dir = selected / "constructed_memory" / "keyframe_nodes"
    node_dir.mkdir(parents=True)
    (selected / "selected_manifest.jsonl").write_text("{}\n", encoding="utf-8")
    (selected / "rejected_manifest.jsonl").write_text("", encoding="utf-8")
    (selected / "review.html").write_text("<html></html>", encoding="utf-8")
    (selected / "constructed_memory" / "keyframe_graph.json").write_text(
        '{"nodes":[1],"edges":[]}',
        encoding="utf-8",
    )
    (node_dir / "kf_000001.json").write_text(
        json.dumps({"kf_id": 1, "semantic": "front desk. exit sign."}),
        encoding="utf-8",
    )
    chunk_payload = {
        "metadata": {"backend": "openvino_text"},
        "records": [{"keyframe_id": 1, "text": "front desk"}],
    }
    (selected / "constructed_memory" / "semantic_chunk_index_records.json").write_text(
        json.dumps(chunk_payload),
        encoding="utf-8",
    )
    np.save(
        selected / "constructed_memory" / "semantic_chunk_index_matrix.npy",
        np.ones((1, 512), dtype=np.float32),
    )

    summary = {
        "status": "ok",
        "source_dataset": str(selected.parent),
        "selection": {
            "candidate_count": 3,
            "selected_count": 1,
            "rejected_count": 2,
        },
        "annotation": {"status": "ok"},
        "chunk_index": {"status": "ok"},
    }

    manifest = _write_scene_memory_manifest(selected, summary)
    stats = _chunk_index_stats(selected)

    assert manifest["format"] == "caragent_scene_memory"
    assert manifest["counts"]["keyframe_nodes"] == 1
    assert manifest["counts"]["semantic_nodes"] == 1
    assert manifest["counts"]["semantic_chunks"] == 1
    assert manifest["artifacts"]["keyframe_nodes_dir"] == "constructed_memory/keyframe_nodes"
    assert manifest["artifacts"]["semantic_chunk_index_records"] == (
        "constructed_memory/semantic_chunk_index_records.json"
    )
    assert stats["record_count"] == 1
    assert stats["matrix_shape"] == [1, 512]
    assert (selected / "constructed_memory" / "scene_memory_manifest.json").exists()


def test_existing_selected_summary_can_be_adopted_without_rerun(tmp_path):
    dataset = tmp_path / "session_legacy"
    selected = dataset / "selected"
    node_dir = selected / "constructed_memory" / "keyframe_nodes"
    node_dir.mkdir(parents=True)
    (dataset / "manifest.jsonl").write_text("{}\n{}\n", encoding="utf-8")
    (selected / "selected_manifest.jsonl").write_text("{}\n", encoding="utf-8")
    (selected / "rejected_manifest.jsonl").write_text("{}\n", encoding="utf-8")
    (selected / "selection_summary.json").write_text(
        json.dumps({"candidate_count": 2, "selected_count": 1, "rejected_count": 1}),
        encoding="utf-8",
    )
    (selected / "constructed_memory" / "keyframe_graph.json").write_text(
        '{"nodes":[1],"edges":[]}',
        encoding="utf-8",
    )
    (node_dir / "kf_000001.json").write_text(
        json.dumps({"kf_id": 1, "semantic": ""}),
        encoding="utf-8",
    )

    _validate_existing_selected(selected)
    summary = _existing_selection_summary(dataset, selected)

    assert summary["adopted_existing"] is True
    assert summary["candidate_count"] == 2
    assert summary["selected_count"] == 1
    assert summary["rejected_count"] == 1
