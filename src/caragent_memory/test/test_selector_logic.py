import numpy as np
import pytest

from caragent_memory.openvino_clip import cosine_similarity, extract_clip_embedding
from caragent_memory.select_keyframes import FrameEmbeddings, _copy_selected_record


class DummyRecord:
    frame_id = "000001"
    raw_path = None
    left_path = None
    right_path = None
    pose_path = None
    meta_path = None
    scan_path = None
    pose = {
        "timestamp": 1.0,
        "x": 1.0,
        "y": 2.0,
        "z": 0.0,
        "yaw": 0.25,
        "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
    }
    meta = {
        "quality_ok": True,
        "manual": False,
    }


def test_cosine_similarity_handles_normalized_vectors():
    assert np.isclose(cosine_similarity(np.array([1, 0]), np.array([1, 0])), 1.0)
    assert np.isclose(cosine_similarity(np.array([1, 0]), np.array([0, 1])), 0.0)


def test_cosine_similarity_handles_zero_vector():
    assert cosine_similarity(np.array([0, 0]), np.array([1, 0])) == 0.0


def test_extract_clip_embedding_accepts_projected_512_dim_output():
    output = np.ones((1, 512), dtype=np.float32)
    embedding = extract_clip_embedding([output])
    assert embedding.shape == (512,)


def test_extract_clip_embedding_rejects_token_hidden_states():
    output = np.ones((1, 50, 768), dtype=np.float32)
    with pytest.raises(ValueError, match="token hidden states"):
        extract_clip_embedding([output])


def test_extract_clip_embedding_rejects_flattened_token_hidden_states():
    output = np.ones((38400,), dtype=np.float32)
    with pytest.raises(ValueError, match="token hidden states"):
        extract_clip_embedding([output])


def test_copy_selected_record_writes_clip_and_dinov2_embeddings(monkeypatch, tmp_path):
    def fake_copy_record_assets(record, output_root):
        return {
            "frame_id": record.frame_id,
            "raw_path": "raw/000001.png",
            "left_path": "left/000001.png",
            "right_path": "right/000001.png",
            "pose_path": "pose/000001_pose.json",
            "meta_path": "meta/000001_meta.json",
            "scan_path": "scan/000001_scan.npz",
        }

    monkeypatch.setattr("caragent_memory.select_keyframes.copy_record_assets", fake_copy_record_assets)
    embeddings = FrameEmbeddings(
        clip=np.ones(512, dtype=np.float32),
        dinov2=np.ones(384, dtype=np.float32),
    )

    manifest = _copy_selected_record(
        record=DummyRecord(),
        embeddings=embeddings,
        output_root=tmp_path,
        source_dataset=tmp_path / "source",
        reason="first",
        max_similarity=None,
        nearest_distance_m=None,
        dedupe_backend="dinov2",
    )

    assert manifest["clip_embedding_path"] == "embeddings/clip/000001.npy"
    assert manifest["dinov2_embedding_path"] == "embeddings/dinov2/000001.npy"
    assert manifest["embedding_path"] == "embeddings/dinov2/000001.npy"
    assert np.load(tmp_path / manifest["clip_embedding_path"]).shape == (512,)
    assert np.load(tmp_path / manifest["dinov2_embedding_path"]).shape == (384,)

    node_payload = (tmp_path / "constructed_memory" / "keyframe_nodes" / "kf_000001.json").read_text()
    assert '"clip_encoding"' in node_payload
    assert '"dinov2_encoding"' in node_payload


def test_output_node_encodings_are_separate_vectors(monkeypatch, tmp_path):
    """clip_encoding (512-d) and dinov2_encoding (384-d) are distinct fields in output JSON."""
    import json

    def fake_copy_record_assets(record, output_root):
        return {
            "frame_id": record.frame_id,
            "raw_path": "raw/000042.png",
            "left_path": "left/000042.png",
            "right_path": "right/000042.png",
            "pose_path": "pose/000042_pose.json",
            "meta_path": "meta/000042_meta.json",
            "scan_path": "scan/000042_scan.npz",
        }

    monkeypatch.setattr("caragent_memory.select_keyframes.copy_record_assets", fake_copy_record_assets)

    rng = np.random.default_rng(42)
    embeddings = FrameEmbeddings(
        clip=rng.normal(size=512).astype(np.float32),
        dinov2=rng.normal(size=384).astype(np.float32),
    )

    _copy_selected_record(
        record=DummyRecord(),
        embeddings=embeddings,
        output_root=tmp_path,
        source_dataset=tmp_path / "source",
        reason="dinov2_novelty",
        max_similarity=0.42,
        nearest_distance_m=0.28,
        dedupe_backend="dinov2",
    )

    node = json.loads(
        (tmp_path / "constructed_memory" / "keyframe_nodes" / "kf_000001.json").read_text()
    )

    clip_arr = np.asarray(node["clip_encoding"], dtype=np.float32)
    dinov2_arr = np.asarray(node["dinov2_encoding"], dtype=np.float32)

    assert clip_arr.shape == (512,), f"clip_encoding shape mismatch: {clip_arr.shape}"
    assert dinov2_arr.shape == (384,), f"dinov2_encoding shape mismatch: {dinov2_arr.shape}"
    assert node["visual_similarity_backend"] == "dinov2"

    assert not np.allclose(clip_arr[:384], dinov2_arr, atol=1e-6), (
        "clip_encoding and dinov2_encoding must be different vectors"
    )
