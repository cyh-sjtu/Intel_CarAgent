import asyncio
import concurrent.futures
import json
import re
import time
from pathlib import Path
from typing import List, Dict
import torch
import numpy as np

from caragent_agent.agents.tools.base.tool_base import ToolBase
from caragent_agent.utils.llm_handler import UnifiedLLMClient
from caragent_agent.utils.llm_request_generator import (
    llm_search_requirement_on_kf_request_message,
    extract_and_convert_ids,
    extract_answer_tags,
    divide_nodes_into_subsets,
    vlm_multi_images_request_message_co_analysis,
)
from caragent_agent.config.config import config
from caragent_agent.agents.async_agent.runtime.resource_scheduler import (
    clip_search_lock_enabled,
)
from caragent_agent.agents.async_agent.execution.runtime_tool_context import (
    get_runtime_tool_context,
)

NODES_NUMBER_IN_A_REQUEST = int(config.get("nodes_number_in_a_request", 8))
CLIP_FILTER_TOP_K = int(config.get("keyframe_search_clip_top_k", 16))
HYBRID_TOP_K = int(config.get("keyframe_search_hybrid_top_k", 12))
CHUNK_MATCH_ENABLED = bool(config.get("keyframe_search_chunk_match_enabled", True))
CHUNK_CLIP_TOP_K = int(config.get("keyframe_search_chunk_clip_top_k", 24))
CHUNK_MAX_CHARS = int(config.get("keyframe_search_chunk_max_chars", 260))
CHUNK_LIMIT_PER_KEYFRAME = int(config.get("keyframe_search_chunk_limit_per_keyframe", 10))
VLM_RERANK_ENABLED = bool(config.get("keyframe_search_vlm_rerank_enabled", False))
VLM_RERANK_TOP_K = int(config.get("keyframe_search_vlm_rerank_top_k", 6))
VLM_RERANK_TIMEOUT_SEC = float(config.get("keyframe_search_vlm_rerank_timeout_sec", 45))
SEARCH_BACKEND = str(config.get("keyframe_search_backend", "local_hybrid")).strip().lower()
SEARCH_TOOL_TIMEOUT_SEC = float(config.get("search_tool_timeout_sec", 45))
LEXICAL_FALLBACK_TOP_K = int(config.get("search_lexical_fallback_top_k", 12))
SEMANTIC_EXCERPT_CHARS = int(config.get("keyframe_semantic_excerpt_chars", 360))
VISIBILITY_HINT_CHARS = 220
VISIBILITY_HINT_LIMIT = 4
HYBRID_WEIGHTS = config.get("keyframe_search_hybrid_weights", {}) or {}
HYBRID_CHUNK_CLIP_WEIGHT = float(HYBRID_WEIGHTS.get("chunk_clip", 0.30))
HYBRID_FULL_CLIP_WEIGHT = float(HYBRID_WEIGHTS.get("full_clip", HYBRID_WEIGHTS.get("clip", 0.15)))
HYBRID_LEXICAL_WEIGHT = float(HYBRID_WEIGHTS.get("lexical", 0.20))
HYBRID_COVERAGE_WEIGHT = float(HYBRID_WEIGHTS.get("coverage", 0.25))
HYBRID_VISIBILITY_WEIGHT = float(HYBRID_WEIGHTS.get("visibility", 0.10))
RETRIEVAL_SCORE_NOTE = (
    "retrieval_score is a local candidate-ranking signal, not answer confidence; "
    "choose the final keyframe from semantic evidence, best_semantic_chunk, "
    "target_visibility_hints, and task constraints."
)
_CHUNK_INDEX_CACHE: dict[tuple, dict] = {}


def _elapsed_since(start_time: float) -> float:
    return round(max(0.0, time.perf_counter() - start_time), 3)


def _runtime_logger():
    logger = get_runtime_tool_context().get("logger")
    if callable(logger):
        return logger
    log_foreground = getattr(logger, "log_foreground", None)
    return log_foreground if callable(log_foreground) else None


def _log_search_progress(message: str) -> None:
    logger = _runtime_logger()
    if logger:
        logger(message)

VISIBILITY_HINT_TERMS = (
    "visible",
    "visibility",
    "clear",
    "clearly",
    "complete",
    "complete view",
    "partially",
    "partial",
    "occluded",
    "blocked",
    "cut off",
    "edge",
    "center",
    "centered",
    "close",
    "closer",
    "near",
    "nearby",
    "large",
    "larger",
    "small",
    "distant",
    "front",
    "behind",
    "beside",
    "next to",
    "可见",
    "清晰",
    "完整",
    "部分",
    "遮挡",
    "挡住",
    "露出",
    "边缘",
    "中央",
    "中间",
    "靠近",
    "近处",
    "远处",
    "前方",
    "后方",
    "旁边",
)

POSITIVE_VISIBILITY_TERMS = (
    "visible",
    "clearly visible",
    "clear",
    "clearly",
    "complete",
    "complete view",
    "center",
    "centered",
    "large",
    "larger",
    "close",
    "closer",
    "near",
    "nearby",
    "front",
    "可见",
    "清晰",
    "完整",
    "中央",
    "中间",
    "靠近",
    "近处",
    "前方",
)

NEGATIVE_VISIBILITY_TERMS = (
    "partially",
    "partial",
    "occluded",
    "blocked",
    "cut off",
    "edge",
    "small",
    "distant",
    "behind",
    "部分",
    "遮挡",
    "挡住",
    "露出",
    "边缘",
    "远处",
    "后方",
)


def _truncate_text(value, limit: int) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def semantic_excerpt_for_requirement(semantic: str, requirement: str = "", *, limit: int = SEMANTIC_EXCERPT_CHARS) -> str:
    text = str(semantic or "").strip()
    if not text:
        return ""
    requirement_terms = _extract_query_terms(requirement)
    lowered = text.lower()
    best_index = -1
    for term in requirement_terms:
        idx = lowered.find(term.lower())
        if idx >= 0 and (best_index < 0 or idx < best_index):
            best_index = idx
    if best_index < 0 or len(text) <= limit:
        return _truncate_text(text, limit)
    half = limit // 2
    start = max(0, best_index - half)
    end = min(len(text), start + limit)
    excerpt = text[start:end].strip()
    if start > 0:
        excerpt = "..." + excerpt
    if end < len(text):
        excerpt = excerpt + "..."
    return excerpt


def extract_visibility_hints(semantic: str, requirement: str = "") -> list[str]:
    """Return compact target-view evidence snippets from semantic text.

    This does not decide the winner. It only exposes potentially useful clues
    such as visibility, completeness, occlusion, and viewpoint quality.
    """

    text = str(semantic or "").strip()
    if not text:
        return []
    requirement_terms = [term.lower() for term in _extract_query_terms(requirement)]
    hint_terms = [term.lower() for term in VISIBILITY_HINT_TERMS]
    chunks = [
        chunk.strip()
        for chunk in re.split(r"(?<=[.!?。！？；;])\s+|\n+", text)
        if chunk.strip()
    ]
    if not chunks:
        chunks = [text]

    hints: list[str] = []
    for chunk in chunks:
        lowered = chunk.lower()
        has_hint = any(term in lowered for term in hint_terms)
        has_target = any(term in lowered for term in requirement_terms) if requirement_terms else False
        if has_hint and (has_target or not requirement_terms):
            hints.append(_truncate_text(chunk, VISIBILITY_HINT_CHARS))
        elif has_hint and len(hints) < 2:
            hints.append(_truncate_text(chunk, VISIBILITY_HINT_CHARS))
        if len(hints) >= VISIBILITY_HINT_LIMIT:
            break
    return hints


def split_semantic_chunks(
    semantic: str,
    *,
    max_chars: int = CHUNK_MAX_CHARS,
    limit: int = CHUNK_LIMIT_PER_KEYFRAME,
) -> list[str]:
    """Split a long keyframe semantic description into retrievable evidence chunks."""

    text = str(semantic or "").strip()
    if not text:
        return []
    rough_parts = [
        part.strip(" -\t\r\n")
        for part in re.split(r"(?<=[.!?。！？；;])\s+|\n+", text)
        if part.strip(" -\t\r\n")
    ]
    if not rough_parts:
        rough_parts = [text]

    chunks: list[str] = []
    buffer = ""
    for part in rough_parts:
        if len(part) > max_chars:
            sub_parts = [
                sub.strip(" -\t\r\n")
                for sub in re.split(r"(?<=[,，、:：])\s*", part)
                if sub.strip(" -\t\r\n")
            ]
            if len(sub_parts) <= 1:
                sub_parts = [
                    part[index : index + max_chars].strip()
                    for index in range(0, len(part), max_chars)
                ]
        else:
            sub_parts = [part]

        for sub in sub_parts:
            if not sub:
                continue
            candidate = f"{buffer} {sub}".strip() if buffer else sub
            if len(candidate) <= max_chars:
                buffer = candidate
            else:
                if buffer:
                    chunks.append(buffer)
                buffer = sub[:max_chars].strip()
            if len(chunks) >= limit:
                break
        if len(chunks) >= limit:
            break

    if buffer and len(chunks) < limit:
        chunks.append(buffer)

    deduped: list[str] = []
    for chunk in chunks:
        compact = re.sub(r"\s+", " ", chunk).strip()
        if compact and compact not in deduped:
            deduped.append(compact)
        if len(deduped) >= limit:
            break
    return deduped


def _best_lexical_chunk(semantic: str, requirement: str) -> str:
    """Return the strongest local text evidence chunk without running CLIP."""

    chunks = split_semantic_chunks(semantic)
    if not chunks:
        return ""
    query_terms = _extract_query_terms(requirement)
    scored = [
        (_score_semantic_text(chunk.lower(), query_terms), index, chunk)
        for index, chunk in enumerate(chunks)
    ]
    scored.sort(key=lambda item: (-item[0], item[1]))
    best_score, _, best_chunk = scored[0]
    if best_score <= 0:
        return chunks[0]
    return best_chunk


def _best_evidence_chunk(
    semantic: str,
    requirement: str,
    *,
    clip_chunk: str | None = None,
) -> str:
    """Pick the most useful human-readable evidence chunk for this query."""

    chunks = split_semantic_chunks(semantic)
    if not chunks:
        return str(clip_chunk or "").strip()
    query_terms = _extract_query_terms(requirement)
    if not query_terms:
        return str(clip_chunk or "").strip() or chunks[0]

    scored: list[tuple[float, int, str]] = []
    for index, chunk in enumerate(chunks):
        lowered = chunk.lower()
        matched_terms = sum(1 for term in query_terms if term.lower() in lowered)
        lexical_score = _score_semantic_text(lowered, query_terms)
        visibility_bonus = 1 if any(term in lowered for term in VISIBILITY_HINT_TERMS) else 0
        score = matched_terms * 10.0 + lexical_score + visibility_bonus
        scored.append((score, index, chunk))
    scored.sort(key=lambda item: (-item[0], item[1]))
    best_score, _, best_chunk = scored[0]
    if best_score > 0:
        return best_chunk
    return str(clip_chunk or "").strip() or chunks[0]


def _target_coverage_score(semantic: str, requirement: str) -> float:
    """Score whether the keyframe semantics cover the concrete target terms."""

    query_terms = _extract_query_terms(requirement)
    if not query_terms:
        return 0.5
    text = str(semantic or "").lower()
    if not text:
        return 0.0
    covered = sum(1 for term in query_terms if term.lower() in text)
    return float(covered) / float(max(1, len(query_terms)))


def _scene_cache_key(scene) -> tuple:
    dataset_dir = str(getattr(scene, "dataset_dir", "") or "")
    node_ids = tuple(sorted(int(kf_id) for kf_id in scene.keyframe_nodes.keys()))
    encoder_kind = "openvino" if getattr(scene, "clip_text_encoder", None) is not None else "torch"
    return (
        dataset_dir,
        node_ids,
        encoder_kind,
        id(getattr(scene, "clip_text_encoder", None)),
        id(getattr(scene, "clip_model", None)),
    )


def _chunk_index_paths(scene) -> tuple[Path, Path]:
    root = Path(getattr(scene, "dataset_dir", "") or ".") / "constructed_memory"
    return root / "semantic_chunk_index_records.json", root / "semantic_chunk_index_matrix.npy"


def _chunk_index_metadata(scene, *, backend: str, record_count: int, matrix_shape: tuple[int, ...]) -> dict:
    return {
        "version": 1,
        "node_ids": [int(kf_id) for kf_id in sorted(scene.keyframe_nodes.keys())],
        "chunk_max_chars": int(CHUNK_MAX_CHARS),
        "chunk_limit_per_keyframe": int(CHUNK_LIMIT_PER_KEYFRAME),
        "backend": str(backend or ""),
        "record_count": int(record_count),
        "matrix_shape": [int(value) for value in matrix_shape],
    }


def _metadata_matches_scene(scene, metadata: dict) -> bool:
    try:
        node_ids = [int(kf_id) for kf_id in sorted(scene.keyframe_nodes.keys())]
        return (
            int(metadata.get("version") or 0) == 1
            and list(metadata.get("node_ids") or []) == node_ids
            and int(metadata.get("chunk_max_chars") or 0) == int(CHUNK_MAX_CHARS)
            and int(metadata.get("chunk_limit_per_keyframe") or 0) == int(CHUNK_LIMIT_PER_KEYFRAME)
        )
    except Exception:
        return False


def _load_persisted_chunk_index(scene) -> dict | None:
    records_path, matrix_path = _chunk_index_paths(scene)
    if not records_path.exists() or not matrix_path.exists():
        return None
    try:
        payload = json.loads(records_path.read_text(encoding="utf-8"))
        metadata = payload.get("metadata") if isinstance(payload, dict) else {}
        records = payload.get("records") if isinstance(payload, dict) else None
        if not isinstance(metadata, dict) or not isinstance(records, list):
            return None
        if not _metadata_matches_scene(scene, metadata):
            return None
        matrix = np.load(matrix_path).astype(np.float32)
        if matrix.ndim != 2 or matrix.shape[0] != len(records):
            return None
        backend = str(metadata.get("backend") or "")
        return {"records": records, "matrix": matrix, "backend": backend, "persisted": True}
    except Exception:
        return None


def _load_compatible_persisted_chunk_index(scene) -> dict | None:
    """Load an older chunk index for row-level reuse when scene nodes changed."""

    records_path, matrix_path = _chunk_index_paths(scene)
    if not records_path.exists() or not matrix_path.exists():
        return None
    try:
        payload = json.loads(records_path.read_text(encoding="utf-8"))
        metadata = payload.get("metadata") if isinstance(payload, dict) else {}
        records = payload.get("records") if isinstance(payload, dict) else None
        if not isinstance(metadata, dict) or not isinstance(records, list):
            return None
        if int(metadata.get("version") or 0) != 1:
            return None
        if int(metadata.get("chunk_max_chars") or 0) != int(CHUNK_MAX_CHARS):
            return None
        if int(metadata.get("chunk_limit_per_keyframe") or 0) != int(CHUNK_LIMIT_PER_KEYFRAME):
            return None
        matrix = np.load(matrix_path).astype(np.float32)
        if matrix.ndim != 2 or matrix.shape[0] != len(records):
            return None
        backend = str(metadata.get("backend") or "")
        return {"records": records, "matrix": matrix, "backend": backend, "persisted": True}
    except Exception:
        return None


def _save_persisted_chunk_index(scene, index: dict) -> None:
    records = list(index.get("records") or [])
    matrix = np.asarray(index.get("matrix"), dtype=np.float32)
    backend = str(index.get("backend") or "")
    if not records or matrix.ndim != 2 or matrix.shape[0] != len(records):
        raise ValueError("Cannot persist invalid semantic chunk index.")
    records_path, matrix_path = _chunk_index_paths(scene)
    records_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = _chunk_index_metadata(
        scene,
        backend=backend,
        record_count=len(records),
        matrix_shape=tuple(matrix.shape),
    )
    records_path.write_text(
        json.dumps({"metadata": metadata, "records": records}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    np.save(matrix_path, matrix)


def _current_chunk_records(scene) -> tuple[list[dict], list[str]]:
    chunk_records: list[dict] = []
    chunk_texts: list[str] = []
    for kf_id, node in scene.keyframe_nodes.items():
        semantic = str(getattr(node, "semantic", "") or "")
        for chunk in split_semantic_chunks(semantic):
            chunk_records.append({"keyframe_id": int(kf_id), "text": chunk})
            chunk_texts.append(chunk)
    return chunk_records, chunk_texts


def _encode_chunks_with_openvino(scene, chunks: list[str]) -> np.ndarray | None:
    encoder = getattr(scene, "clip_text_encoder", None)
    if encoder is None:
        return None
    embeddings = []
    for chunk in chunks:
        embeddings.append(encoder.encode_text(chunk))
    if not embeddings:
        return None
    return np.stack(embeddings).astype(np.float32)


def _encode_chunks_for_backend(scene, chunks: list[str], backend: str) -> np.ndarray | None:
    if not chunks:
        return np.zeros((0, 0), dtype=np.float32)
    if backend == "openvino_text":
        return _encode_chunks_with_openvino(scene, chunks)
    if backend == "torch_text":
        return _encode_chunks_with_torch(scene, chunks)
    return None


def _build_incremental_chunk_index(
    scene,
    chunk_records: list[dict],
    chunk_texts: list[str],
) -> dict | None:
    persisted = _load_compatible_persisted_chunk_index(scene)
    if persisted is None:
        return None
    backend = str(persisted.get("backend") or "")
    if backend not in {"openvino_text", "torch_text"}:
        return None
    if backend == "openvino_text" and getattr(scene, "clip_text_encoder", None) is None:
        return None
    if backend == "torch_text" and getattr(scene, "clip_model", None) is None:
        return None

    old_records = list(persisted.get("records") or [])
    old_matrix = np.asarray(persisted.get("matrix"), dtype=np.float32)
    if old_matrix.ndim != 2 or old_matrix.shape[0] != len(old_records):
        return None

    reusable: dict[tuple[int, str], np.ndarray] = {}
    for idx, record in enumerate(old_records):
        try:
            key = (int(record.get("keyframe_id")), str(record.get("text") or ""))
        except Exception:
            continue
        reusable[key] = old_matrix[idx].astype(np.float32)

    rows: list[np.ndarray | None] = []
    missing_indices: list[int] = []
    missing_texts: list[str] = []
    for idx, record in enumerate(chunk_records):
        key = (int(record["keyframe_id"]), str(record["text"]))
        existing = reusable.get(key)
        if existing is None:
            rows.append(None)
            missing_indices.append(idx)
            missing_texts.append(chunk_texts[idx])
        else:
            rows.append(existing)

    if not missing_indices:
        matrix = np.stack([row for row in rows if row is not None]).astype(np.float32)
        return {"records": chunk_records, "matrix": matrix, "backend": backend, "persisted": False}

    new_matrix = _encode_chunks_for_backend(scene, missing_texts, backend)
    if new_matrix is None or new_matrix.ndim != 2 or new_matrix.shape[0] != len(missing_indices):
        return None
    if old_matrix.shape[1] != new_matrix.shape[1]:
        return None
    for idx, row in zip(missing_indices, new_matrix):
        rows[idx] = row.astype(np.float32)
    if any(row is None for row in rows):
        return None
    matrix = np.stack([row for row in rows if row is not None]).astype(np.float32)
    _log_search_progress(
        f"semantic chunk index reused {len(chunk_records) - len(missing_indices)} rows; "
        f"encoded {len(missing_indices)} new rows"
    )
    return {"records": chunk_records, "matrix": matrix, "backend": backend, "persisted": False}


def _encode_chunks_with_torch(scene, chunks: list[str]) -> np.ndarray | None:
    if getattr(scene, "clip_model", None) is None or getattr(scene, "device", None) is None:
        return None
    clip = _load_clip_module()
    device = scene.device
    model = scene.clip_model
    encoded_batches: list[np.ndarray] = []
    batch_size = 64
    lock = (
        getattr(scene, "clip_lock", None)
        if clip_search_lock_enabled(config)
        else None
    )
    context = lock if lock is not None else _NullContext()
    with context:
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            text_tokens = clip.tokenize(batch, truncate=True).to(device)
            with torch.no_grad():
                text_features = model.encode_text(text_tokens)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            encoded_batches.append(text_features.detach().cpu().numpy().astype(np.float32))
    if not encoded_batches:
        return None
    return np.concatenate(encoded_batches, axis=0).astype(np.float32)


class _NullContext:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


def _build_chunk_index(
    scene,
    *,
    prefer_persisted: bool = True,
    persist: bool = False,
    force_rebuild: bool = False,
) -> dict | None:
    """Build or reuse a per-scene sentence/chunk CLIP index."""

    if not CHUNK_MATCH_ENABLED:
        return None
    cache_key = _scene_cache_key(scene)
    if not force_rebuild:
        cached = _CHUNK_INDEX_CACHE.get(cache_key)
        if cached is not None:
            return cached
        if prefer_persisted:
            persisted = _load_persisted_chunk_index(scene)
            if persisted is not None:
                _CHUNK_INDEX_CACHE[cache_key] = persisted
                return persisted

    chunk_records, chunk_texts = _current_chunk_records(scene)
    if not chunk_texts:
        return None

    if prefer_persisted and not force_rebuild:
        incremental = _build_incremental_chunk_index(scene, chunk_records, chunk_texts)
        if incremental is not None:
            matrix = np.asarray(incremental["matrix"], dtype=np.float32)
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            incremental["matrix"] = matrix / np.maximum(norms, 1e-12)
            if persist:
                _save_persisted_chunk_index(scene, incremental)
                incremental["persisted"] = True
            _CHUNK_INDEX_CACHE[cache_key] = incremental
            return incremental

    matrix = None
    backend = ""
    if getattr(scene, "clip_text_encoder", None) is not None:
        matrix = _encode_chunks_with_openvino(scene, chunk_texts)
        backend = "openvino_text" if matrix is not None else ""
    if matrix is None:
        matrix = _encode_chunks_with_torch(scene, chunk_texts)
        backend = "torch_text" if matrix is not None else ""
    if matrix is None:
        return None

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = matrix / np.maximum(norms, 1e-12)
    index = {"records": chunk_records, "matrix": matrix, "backend": backend, "persisted": False}
    if persist:
        _save_persisted_chunk_index(scene, index)
        index["persisted"] = True
    _CHUNK_INDEX_CACHE[cache_key] = index
    return index


def build_persistent_semantic_chunk_index(scene, *, force_rebuild: bool = False) -> dict:
    """Build and persist the semantic chunk retrieval index for one scene memory."""

    index = _build_chunk_index(
        scene,
        prefer_persisted=not force_rebuild,
        persist=True,
        force_rebuild=force_rebuild,
    )
    if index is None:
        return {
            "status": "skipped",
            "reason": "chunk_match_disabled_or_no_embeddings",
        }
    records = list(index.get("records") or [])
    matrix = np.asarray(index.get("matrix"), dtype=np.float32)
    records_path, matrix_path = _chunk_index_paths(scene)
    return {
        "status": "ok",
        "records_path": str(records_path),
        "matrix_path": str(matrix_path),
        "record_count": len(records),
        "matrix_shape": [int(value) for value in matrix.shape],
        "backend": str(index.get("backend") or ""),
        "persisted": bool(index.get("persisted")),
    }


def _encode_query_for_chunk_index(scene, query_text: str, backend: str) -> np.ndarray | None:
    if backend == "openvino_text" and getattr(scene, "clip_text_encoder", None) is not None:
        return scene.clip_text_encoder.encode_text(query_text).reshape(-1).astype(np.float32)
    if backend == "torch_text" and getattr(scene, "clip_model", None) is not None:
        matrix = _encode_chunks_with_torch(scene, [query_text])
        if matrix is None or matrix.size == 0:
            return None
        return matrix[0].reshape(-1).astype(np.float32)
    return None


def _score_semantic_chunks_by_clip(scene, requirement: str, *, top_k: int = CHUNK_CLIP_TOP_K) -> tuple[dict[int, float], dict[int, str], str]:
    """Score keyframes by their best matching semantic chunk."""

    index = _build_chunk_index(scene)
    if not index:
        return {}, {}, ""
    query_feature = _encode_query_for_chunk_index(
        scene,
        requirement,
        str(index.get("backend") or ""),
    )
    if query_feature is None or query_feature.size == 0:
        return {}, {}, str(index.get("backend") or "")
    display_backend = str(index.get("backend") or "")
    if index.get("persisted"):
        display_backend = f"{display_backend}:persisted"
    query_norm = float(np.linalg.norm(query_feature))
    if query_norm > 0.0:
        query_feature = query_feature / query_norm
    matrix = np.asarray(index["matrix"], dtype=np.float32)
    similarity = (matrix @ query_feature.reshape(-1, 1)).reshape(-1)
    k = min(max(1, int(top_k)), int(similarity.size))
    top_indices = np.argsort(-similarity)[:k]

    scores: dict[int, float] = {}
    chunks: dict[int, str] = {}
    records = index["records"]
    for idx in top_indices:
        record = records[int(idx)]
        kf_id = int(record["keyframe_id"])
        score = float(similarity[int(idx)])
        if kf_id not in scores or score > scores[kf_id]:
            scores[kf_id] = score
            chunks[kf_id] = str(record["text"])
    return scores, chunks, display_backend


def _load_clip_module():
    try:
        import clip

        return clip
    except Exception as exc:
        raise RuntimeError(
            "OpenAI CLIP package is unavailable for torch fallback. "
            "Use scene_memory.use_openvino_clip_text=true with text_encoder.xml, "
            "or install OpenAI CLIP if torch fallback is required."
        ) from exc


def _as_feature_tensor(feature) -> torch.Tensor | None:
    """Normalize stored numpy/torch CLIP embeddings to a 2D torch tensor."""

    if feature is None:
        return None
    if isinstance(feature, torch.Tensor):
        tensor = feature.detach()
    else:
        try:
            tensor = torch.as_tensor(feature, dtype=torch.float32)
        except Exception:
            return None
    if tensor.numel() == 0:
        return None
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)
    elif tensor.dim() > 2:
        tensor = tensor.reshape(1, -1)
    return tensor.float()


def _as_feature_array(feature) -> np.ndarray | None:
    if feature is None:
        return None
    try:
        array = np.asarray(feature, dtype=np.float32).reshape(-1)
    except Exception:
        return None
    if array.size == 0:
        return None
    return array


def _minmax_normalize(scores: dict[int, float]) -> dict[int, float]:
    """Normalize a score dictionary to 0..1 while preserving all keys."""

    if not scores:
        return {}
    values = list(scores.values())
    low = min(values)
    high = max(values)
    if high <= low:
        return {key: 0.5 for key in scores}
    span = high - low
    return {key: (value - low) / span for key, value in scores.items()}


def _visibility_score(semantic_text: str, requirement: str) -> float:
    """Return a lightweight 0..1 visibility/view-quality score."""

    text = str(semantic_text or "").lower()
    if not text:
        return 0.5
    requirement_terms = [term.lower() for term in _extract_query_terms(requirement)]
    target_context = any(term in text for term in requirement_terms) if requirement_terms else True
    positive = sum(text.count(term.lower()) for term in POSITIVE_VISIBILITY_TERMS)
    negative = sum(text.count(term.lower()) for term in NEGATIVE_VISIBILITY_TERMS)
    score = 0.5 + 0.08 * positive - 0.10 * negative
    if target_context and positive:
        score += 0.08
    return float(max(0.0, min(1.0, score)))

class RequirementSearchTool(ToolBase):
    def __init__(self):
        super().__init__(
            name="search_requirement_on_keyframe_nodes",
            description="""
                Retrieve candidate keyframe nodes matching a natural-language
                navigation/search requirement using semantic scene memory.

                Implements batch processing of keyframe nodes through multiple LLM requests to handle large datasets
                efficiently. Matches nodes based on conceptual understanding rather than exact string matching.
                
                Use this as the primary keyframe retrieval step. Make one
                focused request that includes the destination and any important
                target constraints, then compare the returned candidate
                summaries. Do not repeatedly search for a perfect visual match.

                Args:
                    requirement (str): 
                        Natural language description of the search criteria. Should describe desired node characteristics.
                        Example: "areas with high security clearance" or "locations near emergency exits"
                
                Returns:
                    Structured tool result containing matched_keyframe_ids and
                    candidate_keyframes. Each candidate includes keyframe_id,
                    position, semantic_excerpt, best_semantic_chunk,
                    match_reason, evidence_terms, target_visibility_hints, and
                    retrieval_score. retrieval_score is only a local candidate
                    ranking signal; choose the final destination by inspecting
                    semantic evidence and the active task constraints. The full
                    semantic payload is not returned by default.
                
                Example:
                    >>> toolkit.search_requirement_on_keyframe_nodes("Consist book")
                    [32, 4, 38, 20, 21, 40, 8, 10, 11, 12, 13, 14, 26, 27, 28, 29]
                """,
            capability_tags=("scene_memory_search", "background_safe"),
        )
        self.llm_client = UnifiedLLMClient()

    def _semantic_excerpt(self, kf_id: int, requirement: str = "") -> str:
        node = self.scene_memory.keyframe_nodes.get(kf_id)
        semantic = str(getattr(node, "semantic", "") or "").strip() if node is not None else ""
        return semantic_excerpt_for_requirement(semantic, requirement)

    def _candidate_summary(
        self,
        kf_id: int,
        requirement: str,
        *,
        retrieval_score: float | None = None,
        score_breakdown: dict | None = None,
        best_semantic_chunk: str | None = None,
        vlm_rank: int | None = None,
    ) -> dict:
        node = self.scene_memory.keyframe_nodes.get(kf_id)
        position = None
        if node is not None and getattr(node, "position", None) is not None:
            try:
                position = [float(value) for value in list(node.position)[:3]]
            except Exception:
                position = self.to_jsonable(node.position)
        semantic = str(getattr(node, "semantic", "") or "") if node is not None else ""
        evidence_terms = [
            term
            for term in _extract_query_terms(requirement)
            if term.lower() in semantic.lower()
        ][:10]
        excerpt = self._semantic_excerpt(kf_id, requirement)
        chunk = str(best_semantic_chunk or "").strip() or _best_lexical_chunk(
            semantic,
            requirement,
        )
        payload = {
            "keyframe_id": int(kf_id),
            "position": position,
            "semantic_excerpt": excerpt,
            "short_semantics_excerpt": excerpt,
            "best_semantic_chunk": _truncate_text(chunk, SEMANTIC_EXCERPT_CHARS),
            "evidence_terms": evidence_terms,
            "target_visibility_hints": extract_visibility_hints(semantic, requirement),
            "match_reason": (
                "Retrieved by local hybrid keyframe search; compare candidates "
                "using best_semantic_chunk, target_visibility_hints, the active "
                "task target, and selection policy. Do not treat retrieval_score "
                "as answer confidence."
            ),
        }
        if retrieval_score is not None:
            payload["retrieval_score"] = round(float(retrieval_score), 4)
            payload["retrieval_score_note"] = RETRIEVAL_SCORE_NOTE
        if score_breakdown:
            payload["score_breakdown"] = {
                key: round(float(value), 4)
                for key, value in score_breakdown.items()
            }
        if vlm_rank is not None:
            payload["vlm_rerank_rank"] = int(vlm_rank)
        return payload

    def _lexical_fallback_search(self, query_text: str, *, top_k: int = LEXICAL_FALLBACK_TOP_K) -> List[int]:
        """Return local semantic-text matches when the LLM search path times out."""

        query_terms = _extract_query_terms(query_text)
        if not query_terms:
            return []

        scored_nodes: list[tuple[int, int]] = []
        for kf_id, node in self.scene_memory.keyframe_nodes.items():
            semantic_text = str(getattr(node, "semantic", "") or "").lower()
            if not semantic_text:
                continue
            score = _score_semantic_text(semantic_text, query_terms)
            if score > 0:
                try:
                    scored_nodes.append((score, int(kf_id)))
                except Exception:
                    continue

        scored_nodes.sort(key=lambda item: (-item[0], item[1]))
        return [kf_id for _, kf_id in scored_nodes[:top_k]]

    def _rerank_with_vlm(
        self,
        requirement: str,
        scored: list[tuple[float, int, dict[str, float]]],
    ) -> tuple[list[tuple[float, int, dict[str, float]]], dict]:
        """Use one stitched-image VLM call to reorder the strongest candidates."""

        if not VLM_RERANK_ENABLED or len(scored) < 2:
            return scored, {"enabled": bool(VLM_RERANK_ENABLED), "used": False}
        candidate_ids = [kf_id for _, kf_id, _ in scored[: max(1, VLM_RERANK_TOP_K)]]
        kf_set = []
        valid_ids = []
        for kf_id in candidate_ids:
            node = self.scene_memory.keyframe_nodes.get(kf_id)
            if node is None or getattr(node, "rgb_path", None) is None:
                continue
            try:
                if not node.rgb_path.exists():
                    continue
            except Exception:
                continue
            kf_set.append(node)
            valid_ids.append(int(kf_id))
        if len(kf_set) < 2:
            return scored, {
                "enabled": True,
                "used": False,
                "reason": "not_enough_keyframe_images",
                "candidate_ids": candidate_ids,
            }

        question = (
            "You are selecting a robot navigation staging keyframe from labeled keyframe images. "
            f"Target requirement: {requirement}\n"
            "Look at the visible scene evidence, not retrieval scores. Prefer keyframes where the requested target/place/object is actually visible, close, clear, complete, centered, and useful for later robot localization or object approach. "
            "Reject candidates that only match the broad place but do not show the target object or show it too far/tiny/occluded when better candidates exist. "
            "The ranked_keyframe_ids list must contain only candidates that satisfy the target requirement; put rejected or weak candidates only in rejected_keyframe_ids. "
            "Return only this JSON format inside <answer>: "
            "{\"ranked_keyframe_ids\":[id1,id2,...],\"best_keyframe_id\":id,\"rejected_keyframe_ids\":[id3,...],\"reason\":\"short evidence-based reason\"}"
        )
        rerank_start = time.perf_counter()
        _log_search_progress(
            "Keyframe search: VLM rerank started "
            f"for top {len(valid_ids)} candidates; timeout={VLM_RERANK_TIMEOUT_SEC:.1f}s."
        )
        try:
            messages = vlm_multi_images_request_message_co_analysis(kf_set, question)
            request = {
                "request_id": 0,
                "model": config.get("vlm_model_analyse_images", config.get("llm_model", "deepseek-chat")),
                "messages": messages,
            }
            client = UnifiedLLMClient()
            results = asyncio.run(
                asyncio.wait_for(
                    client.batch_chat_completion([request]),
                    timeout=VLM_RERANK_TIMEOUT_SEC,
                )
            )
            raw_response = results.get(0, {}).get(0)
            answer = extract_answer_tags(str(raw_response or ""))
            ranked_ids = _parse_ranked_keyframe_ids(answer)
            ranked_ids = [kf_id for kf_id in ranked_ids if kf_id in valid_ids]
            if not ranked_ids:
                elapsed_sec = _elapsed_since(rerank_start)
                _log_search_progress(
                    "Keyframe search: VLM rerank returned no valid ids "
                    f"in {elapsed_sec:.3f}s."
                )
                return scored, {
                    "enabled": True,
                    "used": False,
                    "reason": "vlm_returned_no_valid_ids",
                    "candidate_ids": valid_ids,
                    "answer": _truncate_text(answer, 600),
                    "elapsed_sec": elapsed_sec,
                }
            order = {kf_id: index for index, kf_id in enumerate(ranked_ids)}
            local_order = {kf_id: index for index, (_, kf_id, _) in enumerate(scored)}
            reranked = sorted(
                scored,
                key=lambda item: (
                    order.get(item[1], len(order) + local_order.get(item[1], 9999)),
                    local_order.get(item[1], 9999),
                ),
            )
            elapsed_sec = _elapsed_since(rerank_start)
            _log_search_progress(
                "Keyframe search: VLM rerank finished "
                f"in {elapsed_sec:.3f}s; ranked={ranked_ids[:6]}."
            )
            return reranked, {
                "enabled": True,
                "used": True,
                "candidate_ids": valid_ids,
                "ranked_keyframe_ids": ranked_ids,
                "answer": _truncate_text(answer, 800),
                "recommendation_reason": _truncate_text(
                    _extract_vlm_recommendation_reason(answer),
                    420,
                ),
                "elapsed_sec": elapsed_sec,
            }
        except Exception as exc:
            elapsed_sec = _elapsed_since(rerank_start)
            _log_search_progress(
                "Keyframe search: VLM rerank failed or timed out "
                f"after {elapsed_sec:.3f}s; using local hybrid ranking."
            )
            return scored, {
                "enabled": True,
                "used": False,
                "reason": "vlm_rerank_failed",
                "error": str(exc),
                "candidate_ids": valid_ids,
                "elapsed_sec": elapsed_sec,
            }

    def _filter_nodes_by_clip(
        self,
        query_text: str,
        top_k: int = CLIP_FILTER_TOP_K,
        *,
        return_scores: bool = False,
    ):
        """Use CLIP model to pre-filter nodes based on similarity."""
        try:
            scene = self.scene_memory
            _use_ov = bool(config.get("scene_memory", {}).get("use_openvino_clip_text", False))
            if _use_ov and getattr(scene, "clip_text_encoder", None) is not None:
                text_features = scene.clip_text_encoder.encode_text(query_text)
                candidates = []
                features_list = []
                for kf_id, node in scene.keyframe_nodes.items():
                    feature = None
                    if hasattr(node, 'semantic_clip_encoding') and node.semantic_clip_encoding is not None:
                        feature = node.semantic_clip_encoding
                    elif hasattr(node, 'clip_encoding') and node.clip_encoding is not None:
                        feature = node.clip_encoding
                    feature = _as_feature_array(feature)
                    if feature is not None:
                        candidates.append(kf_id)
                        features_list.append(feature)
                filtered_nodes = {}
                clip_scores: dict[int, float] = {}
                if candidates:
                    image_features = np.stack(features_list).astype(np.float32)
                    image_norms = np.linalg.norm(image_features, axis=1, keepdims=True)
                    image_features = image_features / np.maximum(image_norms, 1e-12)
                    similarity = (image_features @ text_features.reshape(-1, 1)).reshape(-1)
                    k = min(top_k, len(candidates))
                    indices = np.argsort(-similarity)[:k]
                    for idx in indices:
                        kf_id = candidates[int(idx)]
                        filtered_nodes[kf_id] = scene.keyframe_nodes[kf_id]
                        clip_scores[int(kf_id)] = float(similarity[int(idx)])

                # Additional Frequency-based Filter (same as torch path below)
                freq_scores = []
                query_words = query_text.lower().split()
                stop_words = {'a', 'an', 'the', 'in', 'on', 'at', 'with', 'and', 'or', 'of', 'to', 'for', 'is', 'are'}
                query_words = [w for w in query_words if w not in stop_words]
                if query_words:
                    for kf_id, node in scene.keyframe_nodes.items():
                        if kf_id in filtered_nodes:
                            continue
                        score = 0
                        if hasattr(node, 'semantic') and node.semantic:
                            node_text = node.semantic.lower()
                            for word in query_words:
                                score += node_text.count(word)
                        if score > 0:
                            freq_scores.append((score, kf_id))
                    freq_scores.sort(key=lambda x: x[0], reverse=True)
                    freq_k = min(top_k // 2, len(freq_scores))
                    for i in range(freq_k):
                        kf_id = freq_scores[i][1]
                        filtered_nodes[kf_id] = scene.keyframe_nodes[kf_id]
                        clip_scores.setdefault(int(kf_id), 0.0)

                if return_scores:
                    return filtered_nodes, clip_scores
                return filtered_nodes

            if getattr(scene, 'clip_model', None) is None or getattr(scene, 'device', None) is None:
                if return_scores:
                    return scene.keyframe_nodes, {}
                return scene.keyframe_nodes

            clip = _load_clip_module()
            device = scene.device
            model = scene.clip_model
            clip_lock = (
                getattr(scene, "clip_lock", None)
                if clip_search_lock_enabled(config)
                else None
            )

            if clip_lock is None:
                text_tokens = clip.tokenize([query_text], truncate=True).to(device)
                with torch.no_grad():
                    text_features = model.encode_text(text_tokens)
                    text_features /= text_features.norm(dim=-1, keepdim=True)

                candidates = []
                features_list = []
                for kf_id, node in scene.keyframe_nodes.items():
                    feature = None
                    if hasattr(node, 'semantic_clip_encoding') and node.semantic_clip_encoding is not None:
                        feature = node.semantic_clip_encoding
                    elif hasattr(node, 'clip_encoding') and node.clip_encoding is not None:
                        feature = node.clip_encoding

                    feature = _as_feature_tensor(feature)
                    if feature is not None:
                        candidates.append(kf_id)
                        features_list.append(feature)
            else:
                with clip_lock:
                    text_tokens = clip.tokenize([query_text], truncate=True).to(device)
                    with torch.no_grad():
                        text_features = model.encode_text(text_tokens)
                        text_features /= text_features.norm(dim=-1, keepdim=True)

                    candidates = []
                    features_list = []
                    for kf_id, node in scene.keyframe_nodes.items():
                        feature = None
                        if hasattr(node, 'semantic_clip_encoding') and node.semantic_clip_encoding is not None:
                            feature = node.semantic_clip_encoding
                        elif hasattr(node, 'clip_encoding') and node.clip_encoding is not None:
                            feature = node.clip_encoding

                        feature = _as_feature_tensor(feature)
                        if feature is not None:
                            candidates.append(kf_id)
                            features_list.append(feature)

            if not candidates:
                return scene.keyframe_nodes

            image_features = torch.cat([f.to(device) for f in features_list])
            image_features = image_features.to(dtype=text_features.dtype)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            image_features = image_features.to(dtype=text_features.dtype)

            similarity = (image_features @ text_features.T).squeeze()

            k = min(top_k, len(candidates))
            _, indices = similarity.topk(k)

            filtered_nodes = {}
            clip_scores: dict[int, float] = {}
            for idx in indices:
                kf_id = candidates[idx.item()]
                filtered_nodes[kf_id] = scene.keyframe_nodes[kf_id]
                try:
                    clip_scores[int(kf_id)] = float(similarity[idx.item()].item())
                except Exception:
                    clip_scores[int(kf_id)] = 0.0
            
            # Additional Frequency-based Filter
            # Count occurrences of query words in node semantics (naive exact match)
            # This helps rescue nodes that might be semantically relevant (high word overlap) 
            # but scored low by CLIP for some reason.
            freq_scores = []
            query_words = query_text.lower().split()
            # Remove common stop words to avoid noise
            stop_words = {'a', 'an', 'the', 'in', 'on', 'at', 'with', 'and', 'or', 'of', 'to', 'for', 'is', 'are'}
            query_words = [w for w in query_words if w not in stop_words]

            if query_words:
                for kf_id, node in scene.keyframe_nodes.items():
                    if kf_id in filtered_nodes: 
                        continue # Already selected by CLIP
                    
                    score = 0
                    if hasattr(node, 'semantic') and node.semantic:
                         node_text = node.semantic.lower()
                         for word in query_words:
                             score += node_text.count(word)
                    
                    if score > 0:
                        freq_scores.append((score, kf_id))
                
                # Sort by score descending and take top K/2
                # We mix in some exact-match candidates to improve robustness
                freq_scores.sort(key=lambda x: x[0], reverse=True)
                freq_k = min(top_k // 2, len(freq_scores)) # Add up to half of top_k size
                
                for i in range(freq_k):
                    kf_id = freq_scores[i][1]
                    filtered_nodes[kf_id] = scene.keyframe_nodes[kf_id]
                    clip_scores.setdefault(int(kf_id), 0.0)
                    # print(f"DEBUG: Added node {kf_id} by freq score {freq_scores[i][0]}")

            # print(f"CLIP filtered from {len(scene.keyframe_nodes)} to {len(filtered_nodes)} nodes using query: '{query_text}'")
            if return_scores:
                return filtered_nodes, clip_scores
            return filtered_nodes

        except Exception as e:
            print(f"Error during CLIP filtering: {e}. Fallback to all nodes.")
            if return_scores:
                return self.scene_memory.keyframe_nodes, {}
            return self.scene_memory.keyframe_nodes

    def _local_hybrid_search(self, requirement: str, *, top_k: int = HYBRID_TOP_K) -> dict:
        """Return locally ranked keyframe candidates without remote LLM filtering."""

        timings: dict[str, float] = {}
        total_start = time.perf_counter()
        _log_search_progress(
            "Keyframe search: local hybrid retrieval started "
            f"for '{requirement}'."
        )
        clip_start = time.perf_counter()
        filtered_nodes, raw_clip_scores = self._filter_nodes_by_clip(
            requirement,
            top_k=CLIP_FILTER_TOP_K,
            return_scores=True,
        )
        timings["clip_filter_sec"] = _elapsed_since(clip_start)
        _log_search_progress(
            "Keyframe search: CLIP/full-text filtering finished "
            f"in {timings['clip_filter_sec']:.3f}s; "
            f"filtered_count={len(filtered_nodes)}."
        )
        chunk_start = time.perf_counter()
        _log_search_progress(
            "Keyframe search: semantic chunk matching started "
            "(first run after restart may build the chunk index)."
        )
        raw_chunk_scores, best_chunks, chunk_backend = _score_semantic_chunks_by_clip(
            self.scene_memory,
            requirement,
            top_k=CHUNK_CLIP_TOP_K,
        )
        timings["chunk_match_sec"] = _elapsed_since(chunk_start)
        _log_search_progress(
            "Keyframe search: semantic chunk matching finished "
            f"in {timings['chunk_match_sec']:.3f}s; "
            f"backend={chunk_backend or 'none'}; "
            f"persisted={':persisted' in str(chunk_backend or '')}; "
            f"chunk_candidates={len(raw_chunk_scores)}."
        )
        for kf_id in raw_chunk_scores:
            if kf_id not in filtered_nodes and kf_id in self.scene_memory.keyframe_nodes:
                filtered_nodes[kf_id] = self.scene_memory.keyframe_nodes[kf_id]
        local_score_start = time.perf_counter()
        query_terms = _extract_query_terms(requirement)
        raw_lexical_scores: dict[int, float] = {}
        raw_visibility_scores: dict[int, float] = {}
        raw_coverage_scores: dict[int, float] = {}
        for kf_id, node in filtered_nodes.items():
            try:
                normalized_id = int(kf_id)
            except Exception:
                continue
            semantic_text = str(getattr(node, "semantic", "") or "").lower()
            raw_lexical_scores[normalized_id] = float(
                _score_semantic_text(semantic_text, query_terms)
            )
            raw_coverage_scores[normalized_id] = _target_coverage_score(
                semantic_text,
                requirement,
            )
            raw_visibility_scores[normalized_id] = _visibility_score(
                semantic_text,
                requirement,
            )

        normalized_clip = _minmax_normalize({
            int(kf_id): float(raw_clip_scores.get(int(kf_id), 0.0))
            for kf_id in filtered_nodes
        })
        normalized_chunk_clip = _minmax_normalize({
            int(kf_id): float(raw_chunk_scores.get(int(kf_id), 0.0))
            for kf_id in filtered_nodes
        })
        normalized_lexical = _minmax_normalize(raw_lexical_scores)

        scored: list[tuple[float, int, dict[str, float]]] = []
        for kf_id in filtered_nodes:
            try:
                normalized_id = int(kf_id)
            except Exception:
                continue
            breakdown = {
                "chunk_clip_score": normalized_chunk_clip.get(normalized_id, 0.0),
                "full_clip_score": normalized_clip.get(normalized_id, 0.0),
                "lexical_score": normalized_lexical.get(normalized_id, 0.0),
                "target_coverage_score": raw_coverage_scores.get(normalized_id, 0.0),
                "visibility_score": raw_visibility_scores.get(normalized_id, 0.5),
            }
            final_score = (
                HYBRID_CHUNK_CLIP_WEIGHT * breakdown["chunk_clip_score"]
                + HYBRID_FULL_CLIP_WEIGHT * breakdown["full_clip_score"]
                + HYBRID_LEXICAL_WEIGHT * breakdown["lexical_score"]
                + HYBRID_COVERAGE_WEIGHT * breakdown["target_coverage_score"]
                + HYBRID_VISIBILITY_WEIGHT * breakdown["visibility_score"]
            )
            scored.append((final_score, normalized_id, breakdown))

        scored.sort(key=lambda item: (-item[0], item[1]))
        timings["local_scoring_sec"] = _elapsed_since(local_score_start)
        _log_search_progress(
            "Keyframe search: local hybrid scoring finished "
            f"in {timings['local_scoring_sec']:.3f}s; "
            f"candidate_count={len(scored)}."
        )
        scored, vlm_rerank = self._rerank_with_vlm(requirement, scored)
        if isinstance(vlm_rerank, dict):
            timings["vlm_rerank_sec"] = float(vlm_rerank.get("elapsed_sec") or 0.0)
        timings["total_sec"] = _elapsed_since(total_start)
        vlm_rank_by_id = {
            int(kf_id): index + 1
            for index, kf_id in enumerate(vlm_rerank.get("ranked_keyframe_ids") or [])
            if str(kf_id).strip().lstrip("-").isdigit()
        } if isinstance(vlm_rerank, dict) and vlm_rerank.get("used") else {}
        top_scored = scored[: max(1, int(top_k))]
        candidate_keyframes = [
            self._candidate_summary(
                kf_id,
                requirement,
                retrieval_score=score,
                score_breakdown=breakdown,
                best_semantic_chunk=_best_evidence_chunk(
                    str(getattr(self.scene_memory.keyframe_nodes.get(kf_id), "semantic", "") or ""),
                    requirement,
                    clip_chunk=best_chunks.get(kf_id),
                ),
                vlm_rank=vlm_rank_by_id.get(kf_id),
            )
            for score, kf_id, breakdown in top_scored
        ]
        matched_ids = [kf_id for _, kf_id, _ in top_scored]
        recommended_keyframe_id = matched_ids[0] if matched_ids else None
        recommended_destination = (
            {"type": "keyframe", "keyframe_id": int(recommended_keyframe_id)}
            if recommended_keyframe_id is not None
            else None
        )
        resolution_status = "resolved" if recommended_keyframe_id is not None else "failed"
        recommendation_reason = ""
        if isinstance(vlm_rerank, dict) and vlm_rerank.get("used"):
            recommendation_reason = str(vlm_rerank.get("recommendation_reason") or "").strip()
        if not recommendation_reason and candidate_keyframes:
            top_candidate = candidate_keyframes[0]
            recommendation_reason = str(
                top_candidate.get("best_semantic_chunk")
                or top_candidate.get("semantic_excerpt")
                or "Top-ranked candidate from local hybrid keyframe retrieval."
            ).strip()
        return self.ok(
            (
                f"Resolved keyframe {recommended_keyframe_id} from local hybrid keyframe search."
                if recommended_keyframe_id is not None
                else "No keyframe candidate matched the requirement."
            ),
            data={
                "requirement": requirement,
                "retrieval_mode": "local_hybrid",
                "resolution_status": resolution_status,
                "recommended_keyframe_id": recommended_keyframe_id,
                "recommended_destination": recommended_destination,
                "recommendation_reason": _truncate_text(recommendation_reason, 500),
                "matched_keyframe_ids": matched_ids,
                "candidate_count": len(matched_ids),
                "candidate_keyframes": candidate_keyframes,
                "scoring": {
                    "score_note": RETRIEVAL_SCORE_NOTE,
                    "weights": {
                        "chunk_clip": HYBRID_CHUNK_CLIP_WEIGHT,
                        "full_clip": HYBRID_FULL_CLIP_WEIGHT,
                        "lexical": HYBRID_LEXICAL_WEIGHT,
                        "coverage": HYBRID_COVERAGE_WEIGHT,
                        "visibility": HYBRID_VISIBILITY_WEIGHT,
                    },
                    "clip_top_k": CLIP_FILTER_TOP_K,
                    "chunk_clip_top_k": CHUNK_CLIP_TOP_K,
                    "hybrid_top_k": top_k,
                    "filtered_count": len(filtered_nodes),
                    "chunk_match_enabled": bool(CHUNK_MATCH_ENABLED),
                    "chunk_backend": chunk_backend,
                    "vlm_rerank": vlm_rerank,
                    "timings_sec": timings,
                },
            },
            provenance={"source_type": "scene_memory"},
        )
        
    def execute(self, 
                requirement: str) -> List[int]:
        try:
            if SEARCH_BACKEND in {"local_hybrid", "hybrid", "local"}:
                return self._local_hybrid_search(requirement, top_k=HYBRID_TOP_K)

            requests_list = []
            
            filtered_nodes = self._filter_nodes_by_clip(requirement)
            
            divided_keyframe_nodes = divide_nodes_into_subsets(filtered_nodes, NODES_NUMBER_IN_A_REQUEST)
            for index, subset in enumerate(divided_keyframe_nodes):
                request_metadata = {}
                request_metadata["request_id"] = index
                request_metadata['model'] = config['llm_model_search_on_keyframe_nodes']
                request_metadata['messages'] = llm_search_requirement_on_kf_request_message(subset, requirement)
                requests_list.append(request_metadata)

            client = UnifiedLLMClient()
            results = asyncio.run(
                asyncio.wait_for(
                    client.batch_chat_completion(requests_list),
                    timeout=SEARCH_TOOL_TIMEOUT_SEC,
                )
            )
            target_keyframe_nodes_id_list = []
            for req_id, response in results.items():
                target_keyframe_nodes_id_list += extract_and_convert_ids(response[req_id])

            unique_ids = []
            for matched_id in target_keyframe_nodes_id_list:
                try:
                    normalized_id = int(matched_id)
                except Exception:
                    continue
                if normalized_id not in unique_ids:
                    unique_ids.append(normalized_id)

            return self.ok(
                "Searched keyframe nodes by natural-language requirement.",
                data={
                    "requirement": requirement,
                    "retrieval_mode": "llm_batch",
                    "matched_keyframe_ids": unique_ids,
                    "candidate_count": len(unique_ids),
                    "candidate_keyframes": [
                        self._candidate_summary(kf_id, requirement)
                        for kf_id in unique_ids[:12]
                    ],
                },
                provenance={"source_type": "scene_memory"},
            )
        except Exception as exc:
            error_message = str(exc)
            if isinstance(exc, (asyncio.TimeoutError, TimeoutError, concurrent.futures.TimeoutError)):
                error_message = (
                    f"Requirement search timed out after {SEARCH_TOOL_TIMEOUT_SEC:.0f}s."
                )
                fallback_ids = self._lexical_fallback_search(requirement)
                if fallback_ids:
                    return self.partial(
                        "Requirement search timed out; returned local semantic-text fallback matches.",
                        data={
                            "requirement": requirement,
                            "matched_keyframe_ids": fallback_ids,
                            "candidate_count": len(fallback_ids),
                            "candidate_keyframes": [
                                self._candidate_summary(kf_id, requirement)
                                for kf_id in fallback_ids[:12]
                            ],
                            "fallback_mode": "local_semantic_text",
                        },
                        error={
                            "code": "requirement_search_timeout_fallback",
                            "message": error_message,
                        },
                        provenance={"source_type": "scene_memory"},
                    )
            return self.error_result(
                "Requirement search failed.",
                data={"requirement": requirement},
                error={
                    "code": "requirement_search_failed",
                    "message": error_message,
                },
                provenance={"source_type": "scene_memory"},
            )
    
def _extract_query_terms(query_text: str) -> list[str]:
    """Extract stable local-search terms from a natural language query."""

    normalized = str(query_text or "").lower()
    tokens = re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]{2,}", normalized)
    stop_words = {
        "a", "an", "and", "are", "at", "be", "by", "can", "for", "from", "go",
        "in", "is", "it", "of", "on", "or", "photo", "picture", "place", "see",
        "showing", "the", "there", "to", "top", "view", "where", "with",
        "choose", "clear", "close", "keyframe", "localization", "localize",
        "object", "visible", "visibility", "complete", "candidate", "target",
        "navigation", "navigate", "robot", "best", "good",
    }
    terms = [token for token in tokens if len(token) > 2 and token not in stop_words]
    deduped_terms: list[str] = []
    for term in terms:
        if term not in deduped_terms:
            deduped_terms.append(term)
    return deduped_terms


def _parse_ranked_keyframe_ids(text: str) -> list[int]:
    """Parse VLM ranked keyframe ids from JSON-ish or free-form answers."""

    raw = str(text or "").strip()
    if not raw:
        return []
    json_start = raw.find("{")
    json_end = raw.rfind("}")
    if 0 <= json_start < json_end:
        try:
            import json

            payload = json.loads(raw[json_start : json_end + 1])
            ranked = payload.get("ranked_keyframe_ids") or payload.get("keyframe_ids") or []
            if not isinstance(ranked, list):
                ranked = [ranked]
            ids = []
            best = payload.get("best_keyframe_id")
            if best is not None:
                try:
                    ids.append(int(best))
                except Exception:
                    pass
            for value in ranked:
                try:
                    normalized = int(value)
                except Exception:
                    continue
                if normalized not in ids:
                    ids.append(normalized)
            if ids:
                return ids
        except Exception:
            pass

    ids: list[int] = []
    for match in re.findall(r"(?:keyframe|kf|frame)?\s*#?\s*(\d+)", raw, flags=re.IGNORECASE):
        try:
            normalized = int(match)
        except Exception:
            continue
        if normalized not in ids:
            ids.append(normalized)
    return ids


def _extract_vlm_recommendation_reason(text: str) -> str:
    """Extract the compact VLM rerank reason from JSON-ish answers."""

    raw = str(text or "").strip()
    if not raw:
        return ""
    json_start = raw.find("{")
    json_end = raw.rfind("}")
    if 0 <= json_start < json_end:
        try:
            import json

            payload = json.loads(raw[json_start : json_end + 1])
            reason = str(payload.get("reason") or "").strip()
            if reason:
                return reason
        except Exception:
            pass
    return raw


def _score_semantic_text(semantic_text: str, query_terms: list[str]) -> int:
    """Score semantic text by exact phrase and token overlap."""

    score = 0
    for term in query_terms:
        count = semantic_text.count(term)
        if count:
            score += count
            if len(term) >= 5:
                score += 1
    if query_terms and all(term in semantic_text for term in query_terms):
        score += 10
    return score


if __name__ == "__main__":
    from caragent_agent.config.runtime_paths import get_default_scene_dataset_dir
    from caragent_agent.impression_graph.scene_memory import SceneMemory
    dataset_dir = get_default_scene_dataset_dir()
    scene_memory = SceneMemory(dataset_dir)
    scene_memory.load_keyframe_nodes()
    scene_memory.load_keyframe_graph()
    tool = RequirementSearchTool()
    tool.scene_memory = scene_memory
    print(tool.execute("the scene with red paintings and door"))
