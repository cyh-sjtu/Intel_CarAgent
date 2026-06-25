"""Tools for user-attached image understanding and image-grounded navigation."""

from __future__ import annotations

import asyncio
import json
import math
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from caragent_agent.agents.tools.base.tool_base import ToolBase
from caragent_agent.agents.tools.search.requirement_search import (
    RequirementSearchTool,
    _build_chunk_index,
    extract_visibility_hints,
    semantic_excerpt_for_requirement,
)
from caragent_agent.config.config import config
from caragent_agent.io_adapters import describe_image_for_navigation
from caragent_agent.utils.llm_handler import UnifiedLLMClient
from caragent_agent.utils.llm_request_generator import (
    extract_answer_tags,
    vlm_analyse_on_each_kf_images_request_message,
)


def _workspace_root() -> Path:
    return Path((config.get("paths") or {}).get("workspace_root", "/home/car/caragent_ws"))


def _optional_path(value: Any) -> Path | None:
    text = str(value or "").strip()
    return Path(text).expanduser() if text else None


def _elapsed_since(start_time: float) -> float:
    return round(max(0.0, time.perf_counter() - start_time), 3)


def _norm_vector(vec: Any) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    denom = float(np.linalg.norm(arr))
    if denom > 0.0:
        arr = arr / denom
    return arr.astype(np.float32)


def _top_score_items(scores: dict[int, float], *, limit: int) -> list[dict[str, Any]]:
    return [
        {"keyframe_id": int(kf_id), "score": round(float(score), 6)}
        for kf_id, score in sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:limit]
    ]


def _minmax_scores(scores: dict[int, float]) -> dict[int, float]:
    if not scores:
        return {}
    values = list(scores.values())
    low = min(values)
    high = max(values)
    if high <= low:
        return {key: 0.5 for key in scores}
    return {key: (value - low) / (high - low) for key, value in scores.items()}


def _load_keyframe_vector_matrix(scene_memory: Any, attr: str) -> tuple[list[int], np.ndarray]:
    ids: list[int] = []
    vectors: list[np.ndarray] = []
    for raw_id, node in sorted(
        getattr(scene_memory, "keyframe_nodes", {}).items(),
        key=lambda item: int(item[0]),
    ):
        vec = getattr(node, attr, None)
        arr = np.asarray(vec, dtype=np.float32).reshape(-1)
        if arr.size <= 0:
            continue
        ids.append(int(raw_id))
        vectors.append(_norm_vector(arr))
    if not vectors:
        return [], np.empty((0, 0), dtype=np.float32)
    return ids, np.stack(vectors).astype(np.float32)


def _score_vector_matrix(ids: list[int], matrix: np.ndarray, query: np.ndarray) -> dict[int, float]:
    if not ids or matrix.size <= 0 or query.size <= 0:
        return {}
    sim = (matrix @ query.reshape(-1, 1)).reshape(-1)
    return {int(kf_id): float(sim[index]) for index, kf_id in enumerate(ids)}


def _clip_image_vector(image_path: Path) -> np.ndarray:
    from caragent_memory.openvino_clip import OpenVINOClipImageEncoder

    model_path = _workspace_root() / "models" / "clip-vit-base-patch32" / "image_encoder.xml"
    device = str(
        config.get("attached_image_match_clip_device")
        or config.get("scene_memory", {}).get("device")
        or "AUTO"
    )
    encoder = OpenVINOClipImageEncoder(model_path, device=device)
    return _norm_vector(encoder.encode_path(image_path))


def _dinov2_image_vector(image_path: Path) -> np.ndarray:
    from caragent_memory.dinov2_encoder import DINOv2ImageEncoder

    device = str(config.get("attached_image_match_dinov2_device") or "cpu")
    encoder = DINOv2ImageEncoder(
        _workspace_root() / "models" / "dinov2",
        device=device,
        local_files_only=True,
    )
    return _norm_vector(encoder.encode_path(image_path))


def _score_clip_image_to_semantic_chunks(
    scene_memory: Any,
    query_clip: np.ndarray,
) -> tuple[dict[int, float], dict[int, str], dict[str, Any]]:
    index = _build_chunk_index(scene_memory, prefer_persisted=True, persist=False)
    if not index:
        return {}, {}, {"status": "skipped", "reason": "chunk_index_unavailable"}
    matrix = np.asarray(index.get("matrix"), dtype=np.float32)
    records = list(index.get("records") or [])
    if matrix.ndim != 2 or matrix.shape[0] != len(records) or query_clip.size != matrix.shape[1]:
        return {}, {}, {
            "status": "skipped",
            "reason": "dimension_mismatch",
            "query_dim": int(query_clip.size),
            "matrix_shape": [int(value) for value in matrix.shape],
        }
    sim = (matrix @ query_clip.reshape(-1, 1)).reshape(-1)
    per_kf_max: dict[int, float] = {}
    best_chunk: dict[int, str] = {}
    for idx, record in enumerate(records):
        kf_id = int(record["keyframe_id"])
        score = float(sim[idx])
        if kf_id not in per_kf_max or score > per_kf_max[kf_id]:
            per_kf_max[kf_id] = score
            best_chunk[kf_id] = str(record.get("text") or "")
    diagnostics = {
        "status": "ok",
        "backend": str(index.get("backend") or ""),
        "persisted": bool(index.get("persisted")),
        "top_by_max": [
            {**item, "best_chunk": best_chunk.get(int(item["keyframe_id"]), "")[:220]}
            for item in _top_score_items(per_kf_max, limit=12)
        ],
    }
    return per_kf_max, best_chunk, diagnostics


def _direct_hybrid_scores(
    *,
    dino_scores: dict[int, float],
    chunk_scores: dict[int, float],
) -> tuple[dict[int, float], dict[str, float]]:
    weights_cfg = config.get("attached_image_direct_hybrid_weights") or {}
    use_clip_chunks = bool(config.get("attached_image_direct_hybrid_use_clip_chunks", False))
    weights = {
        "dinov2_image_to_image": float(weights_cfg.get("dinov2_image_to_image", 1.0)),
        "clip_image_to_semantic_chunk": (
            float(weights_cfg.get("clip_image_to_semantic_chunk", 0.0))
            if use_clip_chunks
            else 0.0
        ),
    }
    dino_norm = _minmax_scores(dino_scores)
    chunk_norm = _minmax_scores(chunk_scores) if use_clip_chunks else {}
    ids = set(dino_norm) | set(chunk_norm)
    return {
        kf_id: (
            weights["dinov2_image_to_image"] * dino_norm.get(kf_id, 0.0)
            + weights["clip_image_to_semantic_chunk"] * chunk_norm.get(kf_id, 0.0)
        )
        for kf_id in ids
    }, weights


def _configured_object_approach_depth_backend() -> str:
    agent_cfg = config.get("agent") if isinstance(config.get("agent"), dict) else {}
    backend = str(agent_cfg.get("object_approach_depth_backend") or "stereo").strip().lower()
    if backend == "auto":
        backend = "stereo"
    if backend not in {"stereo", "stereo_primary_mono_guard", "mono_relative_lidar"}:
        backend = "stereo"
    return backend


def _image_path_from_ref(image_ref: Any) -> tuple[Path | None, dict[str, Any]]:
    """Resolve a path from a JSON ref, dict, direct path, or latest-image alias."""

    meta: dict[str, Any] = {}
    if isinstance(image_ref, dict):
        meta = dict(image_ref)
    else:
        text = str(image_ref or "").strip()
        if not text:
            return None, meta
        if text.lower() == "latest":
            root = _workspace_root() / "perception_outputs" / "agent_user_images"
            candidates = sorted(root.glob("*.jpg"), key=lambda item: item.stat().st_mtime, reverse=True)
            return (candidates[0], {"image_ref_id": "latest"}) if candidates else (None, {"image_ref_id": "latest"})
        if text.startswith("{"):
            try:
                parsed = json.loads(text)
            except Exception:
                parsed = None
            if isinstance(parsed, dict):
                meta = parsed
            else:
                meta = {"path": text}
        else:
            meta = {"path": text}

    path_text = str(meta.get("path") or "").strip()
    if not path_text:
        return None, meta
    path = Path(path_text).expanduser()
    return path, meta


def _load_image_from_ref(image_ref: Any) -> tuple[Image.Image | None, Path | None, dict[str, Any], str | None]:
    path, meta = _image_path_from_ref(image_ref)
    if path is None:
        return None, None, meta, "image_ref_did_not_resolve_to_path"
    if not path.exists():
        return None, path, meta, "image_path_not_found"
    try:
        return Image.open(path).convert("RGB"), path, meta, None
    except Exception as exc:
        return None, path, meta, f"image_open_failed: {exc}"


def _node_records(scene_memory: Any, keyframe_ids: list[int], requirement: str = "") -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for kf_id in keyframe_ids:
        node = getattr(scene_memory, "keyframe_nodes", {}).get(int(kf_id))
        if node is None:
            continue
        semantic = str(getattr(node, "semantic", "") or "")
        excerpt = semantic_excerpt_for_requirement(semantic, requirement)
        records.append(
            {
                "keyframe_id": int(kf_id),
                "name": getattr(node, "name", ""),
                "semantic_excerpt": excerpt,
                "short_semantics_excerpt": excerpt,
                "target_visibility_hints": extract_visibility_hints(semantic, requirement),
                "match_reason": (
                    "Candidate retrieved from attached-image requirement search; "
                    "executor must compare it against the active task target."
                ),
                "position": np.asarray(getattr(node, "position", []), dtype=float).reshape(-1).tolist(),
                "image": str(getattr(node, "rgb_path", "") or getattr(node, "left_path", "") or ""),
            }
        )
    return records


def _score_text(text: str, terms: list[str]) -> float:
    lowered = str(text or "").lower()
    score = 0.0
    for term in terms:
        if not term:
            continue
        count = lowered.count(term)
        if count:
            score += 1.0 + min(3, count) * 0.25
    return score


def _terms_from_text(text: str) -> list[str]:
    stop = {
        "the", "and", "with", "near", "this", "that", "from", "into", "image",
        "photo", "picture", "scene", "place", "location", "there", "here",
    }
    terms = []
    for token in re.findall(r"[a-zA-Z0-9\u4e00-\u9fff]+", str(text or "").lower()):
        if len(token) <= 1 or token in stop:
            continue
        if token not in terms:
            terms.append(token)
    return terms[:16]


def _rank_keyframes_by_text(scene_memory: Any, query: str, *, limit: int = 8) -> list[int]:
    terms = _terms_from_text(query)
    scored: list[tuple[float, int]] = []
    for raw_id, node in getattr(scene_memory, "keyframe_nodes", {}).items():
        try:
            kf_id = int(raw_id)
        except Exception:
            continue
        text = " ".join(
            str(value or "")
            for value in (
                getattr(node, "name", ""),
                getattr(node, "semantic", ""),
            )
        )
        score = _score_text(text, terms)
        if score > 0:
            scored.append((score, kf_id))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [kf_id for _, kf_id in scored[:limit]]


def _matched_ids_from_tool_result(result: Any) -> list[int]:
    if not isinstance(result, dict):
        return []
    data = result.get("data")
    if not isinstance(data, dict):
        return []
    ids = data.get("matched_keyframe_ids")
    if not isinstance(ids, list):
        return []
    normalized: list[int] = []
    for raw_id in ids:
        try:
            kf_id = int(raw_id)
        except Exception:
            continue
        if kf_id not in normalized:
            normalized.append(kf_id)
    return normalized


def _structured_data_from_tool_result(result: Any) -> dict[str, Any]:
    if not isinstance(result, dict):
        return {}
    data = result.get("data")
    return data if isinstance(data, dict) else result


def _build_attached_keyframe_requirement(description: str, query: str) -> str:
    clean_description = str(description or "").strip()
    clean_query = str(query or "").strip()
    if clean_query:
        return (
            "Find keyframes that are useful staging viewpoints for navigating to "
            f"this target object: {clean_query}. The attached image is approximate "
            "visual evidence, not a detail-by-detail checklist. The keyframe map is "
            "a finite set of robot viewpoints, so do not require the candidate "
            "keyframe to match every object, crop, viewpoint, or transient detail "
            "from the attached image. The main question is whether the same target "
            "object appears in the same general place and can be localized after "
            "navigation. Prefer keyframes where the target object is closer, larger, "
            "clearer, more complete, more centered, and less occluded; avoid "
            "viewpoints where the target is only barely visible or cut off when "
            "better target views exist. Use the surrounding scene context only to "
            "disambiguate the correct place. "
            f"Attached image description: {clean_description}"
        )
    return (
        "Find keyframes that best match the attached target/place image for navigation. "
        "The attached image is approximate visual evidence from one viewpoint, not a "
        "perfect-match template. Use stable scene layout, landmarks, and distinctive "
        "objects; do not reject candidates because of minor crop, viewpoint, lighting, "
        "or object-detail differences. "
        f"Attached image description: {clean_description}"
    )


def _keyframe_image_path(node: Any) -> Path | None:
    return _optional_path(getattr(node, "rgb_path", None) or getattr(node, "left_path", None))


def _fit_image(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    fitted = image.convert("RGB").copy()
    fitted.thumbnail(size, Image.LANCZOS)
    canvas = Image.new("RGB", size, (245, 245, 245))
    canvas.paste(fitted, ((size[0] - fitted.width) // 2, (size[1] - fitted.height) // 2))
    return canvas


def _make_query_candidate_grid(
    query_image: Image.Image,
    scene_memory: Any,
    candidate_ids: list[int],
) -> Image.Image:
    cell_w, cell_h = 420, 280
    label_h = 38
    pad = 14
    cols = 3
    items: list[tuple[str, Image.Image, tuple[int, int, int]]] = [
        ("QUERY", query_image, (0, 90, 170)),
    ]
    for index, kf_id in enumerate(candidate_ids, start=1):
        node = getattr(scene_memory, "keyframe_nodes", {}).get(int(kf_id))
        image_path = _keyframe_image_path(node) if node is not None else None
        if image_path is None or not image_path.exists():
            continue
        items.append((f"#{index} KF{int(kf_id)}", Image.open(image_path).convert("RGB"), (45, 45, 45)))

    rows = int(math.ceil(len(items) / float(cols)))
    grid = Image.new(
        "RGB",
        (pad * 2 + cols * cell_w, pad * 2 + rows * (cell_h + label_h)),
        (255, 255, 255),
    )
    draw = ImageDraw.Draw(grid)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 24)
    except Exception:
        font = ImageFont.load_default()
    for index, (label, item_image, color) in enumerate(items):
        row = index // cols
        col = index % cols
        x = pad + col * cell_w
        y = pad + row * (cell_h + label_h)
        grid.paste(_fit_image(item_image, (cell_w - 12, cell_h)), (x + 6, y))
        draw.text((x + 10, y + cell_h + 6), label, font=font, fill=color)
    return grid


def _parse_vlm_keyframe_rerank(answer: str, candidate_ids: list[int]) -> dict[str, Any]:
    text = str(answer or "").strip()
    parsed: dict[str, Any] = {}
    try:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        parsed_value = json.loads(match.group(0) if match else text)
        if isinstance(parsed_value, dict):
            parsed = parsed_value
    except Exception:
        parsed = {}

    allowed = [int(kf_id) for kf_id in candidate_ids]
    allowed_set = set(allowed)

    def _coerce_keyframe_id(raw_value: Any) -> int | None:
        try:
            kf_id = int(raw_value)
            return kf_id if kf_id in allowed_set else None
        except Exception:
            pass
        raw_text = str(raw_value or "")
        kf_match = re.search(r"\bKF\s*#?\s*(\d+)\b", raw_text, flags=re.IGNORECASE)
        if kf_match:
            kf_id = int(kf_match.group(1))
            return kf_id if kf_id in allowed_set else None
        # Labels often look like "#3 KF34"; if no KF marker exists, the actual
        # keyframe id is usually the last number, not the display rank.
        numbers = [int(value) for value in re.findall(r"\d+", raw_text)]
        for kf_id in reversed(numbers):
            if kf_id in allowed_set:
                return kf_id
        return None

    ranked_raw = parsed.get("ranked_keyframe_ids") or parsed.get("ranked") or []
    ranked: list[int] = []
    if isinstance(ranked_raw, list):
        for raw_id in ranked_raw:
            kf_id = _coerce_keyframe_id(raw_id)
            if kf_id is not None and kf_id not in ranked:
                ranked.append(kf_id)
    raw_best = parsed.get("best_keyframe_id")
    best = _coerce_keyframe_id(raw_best)
    if best is None:
        best = ranked[0] if ranked else allowed[0]
    if best not in allowed_set:
        best = ranked[0] if ranked else allowed[0]
    if best not in ranked:
        ranked.insert(0, best)
    for kf_id in allowed:
        if kf_id not in ranked:
            ranked.append(kf_id)
    return {
        "best_keyframe_id": int(best),
        "ranked_keyframe_ids": ranked,
        "reason": str(parsed.get("reason") or parsed.get("rationale") or text[:800]).strip(),
        "raw_answer": text[:1600],
    }

class AttachedImageAnalyzerTool(ToolBase):
    """Answer questions about one user-attached image."""

    def __init__(self) -> None:
        super().__init__(
            name="analyse_attached_image",
            description=(
                "Analyze a user-attached image with a VLM. Use only when the current "
                "task has image_refs and the question requires looking at the attached image. "
                "Pass image_ref as the attached image path or JSON object from execution context."
            ),
            capability_tags=("attached_image", "vlm", "background_unsafe"),
        )

    def execute(self, image_ref: str, question: str) -> dict[str, Any]:
        image, path, meta, error = _load_image_from_ref(image_ref)
        if error or image is None:
            return self.blocked(
                "Attached image is unavailable.",
                data={"image_ref": image_ref, "resolved_path": str(path) if path else None},
                error={"code": "attached_image_unavailable", "message": error or ""},
                provenance={"source_type": "attached_image"},
            )
        try:
            request = {
                "request_id": 0,
                "model": config.get("vlm_model_analyse_images", config.get("llm_model", "deepseek-chat")),
                "messages": vlm_analyse_on_each_kf_images_request_message(image, question),
            }
            results = asyncio.run(UnifiedLLMClient().batch_chat_completion([request]))
            response = results.get(0, {}).get(0)
            if response is None:
                raise RuntimeError("VLM returned no response.")
            answer = extract_answer_tags(str(response)).strip()
            return self.ok(
                "Analyzed the attached image.",
                data={
                    "image_ref": meta,
                    "image_path": str(path),
                    "question": question,
                    "answer": answer,
                },
                provenance={"source_type": "attached_image"},
            )
        except Exception as exc:
            return self.error_result(
                "Attached image analysis failed.",
                data={"image_ref": meta, "image_path": str(path), "question": question},
                error={"code": "attached_image_analysis_failed", "message": str(exc)},
                provenance={"source_type": "attached_image"},
            )


class AttachedImageKeyframeMatcherTool(ToolBase):
    """Retrieve candidate keyframes for one user-attached place/target image."""

    def __init__(self) -> None:
        super().__init__(
            name="match_attached_image_to_keyframes",
            description=(
                "Match a user-attached target/place image to scene-memory keyframes and "
                "return a recommended keyframe plus comparable candidate evidence. "
                "The attached image is approximate viewpoint evidence, not a perfect "
                "whole-image template. For object targets, prioritize whether the target "
                "object is visible and useful for later live localization. Use focus='scene' "
                "for place matching and focus='object' for object-centric staging."
            ),
            capability_tags=("attached_image", "scene_memory_search", "semantic_grounding", "background_unsafe"),
        )
        self._requirement_search = RequirementSearchTool()

    def execute(self, image_ref: str, query: str = "", focus: str = "scene") -> dict[str, Any]:
        match = self._match(image_ref, query=query, focus=focus)
        if match.get("status") == "ok":
            recommended = match.get("recommended_keyframe_id")
            return self.ok(
                (
                    f"Matched attached image to keyframe {recommended}."
                    if recommended is not None
                    else "Returned candidate keyframes for the attached image."
                ),
                data=match,
                provenance={"source_type": "attached_image"},
            )
        return self.partial(
            "Could not confidently match the attached image to a keyframe.",
            data=match,
            error={"code": "attached_image_keyframe_match_failed", "message": match.get("reason") or ""},
            provenance={"source_type": "attached_image"},
        )

    def _match(
        self,
        image_ref: str,
        *,
        query: str = "",
        focus: str = "scene",
        top_k: int = 8,
    ) -> dict[str, Any]:
        image, path, meta, error = _load_image_from_ref(image_ref)
        if error or image is None:
            return {"status": "blocked", "reason": error, "image_ref": meta}
        direct_error = ""
        try:
            focus = str(focus or "scene").strip().lower()
            if focus not in {"scene", "object"}:
                focus = "scene"
            direct = self._match_direct_hybrid(
                image=image,
                image_path=path,
                image_ref=meta,
                query=query,
                focus=focus,
                top_k=top_k,
            )
            if direct.get("status") == "ok":
                return direct
            direct_error = str(direct.get("reason") or "")
        except Exception as exc:
            direct_error = f"{type(exc).__name__}: {exc}"

        try:
            description = str(meta.get("description") or "").strip()
            if not description:
                description = describe_image_for_navigation(image)
            requirement = _build_attached_keyframe_requirement(description, query)
            if focus == "object":
                requirement = (
                    requirement
                    + "\nObject-focus staging requirement: prefer a keyframe where the "
                    "target object from the attached image is visible, close, clear, "
                    "complete, and suitable for later robot-scene object localization."
                )
            self._requirement_search.scene_memory = self.scene_memory
            self._requirement_search.run_memory = self.run_memory
            requirement_result = self._requirement_search.execute(requirement)
            candidate_ids = _matched_ids_from_tool_result(requirement_result)
            requirement_payload = _structured_data_from_tool_result(requirement_result)
            retrieval_mode = "requirement_search"
            if not candidate_ids:
                search_text = " ".join(part for part in [query, description] if str(part or "").strip())
                candidate_ids = _rank_keyframes_by_text(self.scene_memory, search_text, limit=top_k)
                retrieval_mode = "local_text_fallback"
            if not candidate_ids:
                candidate_ids = sorted(list(getattr(self.scene_memory, "keyframe_nodes", {}).keys()))[:top_k]
                retrieval_mode = "id_order_fallback"
            candidate_ids = [int(kf_id) for kf_id in candidate_ids[:top_k]]
            candidate_records = _node_records(self.scene_memory, candidate_ids, requirement)
            if not candidate_ids:
                return {"status": "unreliable", "reason": "no_candidate_keyframes", "image_ref": meta}
            recommended_keyframe_id = None
            recommended_destination = None
            recommendation_reason = ""
            resolution_status = "resolved"
            if isinstance(requirement_payload, dict):
                for key in ("recommended_keyframe_id", "keyframe_id", "target_keyframe_id"):
                    if requirement_payload.get(key) is not None:
                        try:
                            recommended_keyframe_id = int(requirement_payload.get(key))
                            break
                        except Exception:
                            pass
                recommended_destination = (
                    requirement_payload.get("recommended_destination")
                    or requirement_payload.get("destination")
                )
                recommendation_reason = str(requirement_payload.get("recommendation_reason") or "").strip()
                resolution_status = str(requirement_payload.get("resolution_status") or resolution_status).strip() or resolution_status
            if recommended_keyframe_id is None:
                recommended_keyframe_id = int(candidate_ids[0])
            if not isinstance(recommended_destination, dict):
                recommended_destination = {
                    "type": "keyframe",
                    "keyframe_id": int(recommended_keyframe_id),
                }
            if not recommendation_reason:
                recommendation_reason = (
                    "Best attached-image keyframe candidate from scene-memory matching."
                )
            if direct_error:
                recommendation_reason = (
                    f"Direct image hybrid matching fell back to text retrieval: {direct_error}. "
                    + recommendation_reason
                )
            ranked_keyframes = []
            for index, kf_id in enumerate(candidate_ids, start=1):
                ranked_keyframes.append(
                    {
                        "keyframe_id": int(kf_id),
                        "rank": int(index),
                    }
                )
            return {
                "status": "ok",
                "image_ref": meta,
                "image_path": str(path),
                "description": description,
                "query": query,
                "focus": focus,
                "requirement": requirement,
                "retrieval_mode": f"{retrieval_mode}_fallback_after_direct_hybrid",
                "direct_hybrid_error": direct_error,
                "resolution_status": resolution_status,
                "recommended_keyframe_id": int(recommended_keyframe_id),
                "recommended_destination": recommended_destination,
                "best_keyframe_id": int(recommended_keyframe_id),
                "recommendation_reason": recommendation_reason,
                "ranked_keyframes": ranked_keyframes,
                "candidate_keyframe_ids": [int(kf_id) for kf_id in candidate_ids],
                "candidate_keyframes": candidate_records,
                "selection_required": False,
                "selection_instruction": (
                    "Use recommended_keyframe_id when resolution_status is resolved. "
                    "Candidate evidence is included for explanation and debugging."
                ),
            }
        except Exception as exc:
            return {"status": "error", "reason": str(exc), "image_ref": meta}

    def _match_direct_hybrid(
        self,
        *,
        image: Image.Image,
        image_path: Path,
        image_ref: dict[str, Any],
        query: str,
        focus: str,
        top_k: int,
    ) -> dict[str, Any]:
        timings: dict[str, float] = {}
        total_start = time.perf_counter()
        use_clip_chunks = bool(config.get("attached_image_direct_hybrid_use_clip_chunks", False))

        chunk_scores: dict[int, float] = {}
        best_chunks: dict[int, str] = {}
        chunk_diagnostics: dict[str, Any] = {
            "status": "disabled",
            "reason": "attached_image_direct_hybrid_use_clip_chunks=false",
        }
        if use_clip_chunks:
            clip_start = time.perf_counter()
            clip_query = _clip_image_vector(image_path)
            timings["encode_clip_image_sec"] = _elapsed_since(clip_start)

            chunk_start = time.perf_counter()
            chunk_scores, best_chunks, chunk_diagnostics = _score_clip_image_to_semantic_chunks(
                self.scene_memory,
                clip_query,
            )
            timings["clip_image_to_semantic_chunk_sec"] = _elapsed_since(chunk_start)

        dino_start = time.perf_counter()
        dino_query = _dinov2_image_vector(image_path)
        dino_ids, dino_matrix = _load_keyframe_vector_matrix(self.scene_memory, "dinov2_encoding")
        dino_scores = _score_vector_matrix(dino_ids, dino_matrix, dino_query)
        timings["dinov2_image_to_keyframe_sec"] = _elapsed_since(dino_start)

        if not dino_scores and not chunk_scores:
            return {
                "status": "unreliable",
                "reason": "direct_image_embeddings_unavailable",
                "image_ref": image_ref,
                "image_path": str(image_path),
            }

        hybrid_scores, weights = _direct_hybrid_scores(
            dino_scores=dino_scores,
            chunk_scores=chunk_scores,
        )
        candidate_limit = int(config.get("attached_image_direct_hybrid_top_k", 5) or 5)
        candidate_limit = max(1, min(candidate_limit, max(1, top_k)))
        hybrid_top = _top_score_items(hybrid_scores, limit=candidate_limit)
        candidate_ids = [int(item["keyframe_id"]) for item in hybrid_top]
        if not candidate_ids:
            return {
                "status": "unreliable",
                "reason": "direct_hybrid_no_candidate_keyframes",
                "image_ref": image_ref,
                "image_path": str(image_path),
            }

        rerank_start = time.perf_counter()
        rerank = self._rerank_direct_hybrid_candidates(
            image=image,
            candidate_ids=candidate_ids,
            focus=focus,
            query=query,
        )
        timings["vlm_rerank_sec"] = _elapsed_since(rerank_start)

        ranked_ids = [
            int(kf_id)
            for kf_id in rerank.get("ranked_keyframe_ids", [])
            if int(kf_id) in set(candidate_ids)
        ]
        for kf_id in candidate_ids:
            if kf_id not in ranked_ids:
                ranked_ids.append(kf_id)
        recommended_keyframe_id = int(rerank.get("best_keyframe_id") or ranked_ids[0])
        recommended_destination = {
            "type": "keyframe",
            "keyframe_id": int(recommended_keyframe_id),
        }
        requirement = (
            "Direct attached-image keyframe matching. "
            f"focus={focus}; query={str(query or '').strip() or '(none)'}"
        )
        candidate_records = _node_records(self.scene_memory, ranked_ids, requirement)
        ranked_keyframes = [
            {"keyframe_id": int(kf_id), "rank": index}
            for index, kf_id in enumerate(ranked_ids, start=1)
        ]
        timings["total_sec"] = _elapsed_since(total_start)
        diagnostic_by_id: dict[int, dict[str, Any]] = {}
        dino_top_ids = {int(item["keyframe_id"]) for item in _top_score_items(dino_scores, limit=12)}
        chunk_top_ids = {int(item["keyframe_id"]) for item in _top_score_items(chunk_scores, limit=12)}
        for item in hybrid_top:
            kf_id = int(item["keyframe_id"])
            diagnostic_by_id[kf_id] = {
                "hybrid_score": item["score"],
                "dino_score": round(float(dino_scores.get(kf_id, 0.0)), 6),
                "chunk_score": round(float(chunk_scores.get(kf_id, 0.0)), 6),
                "sources": [
                    name
                    for name, included in (
                        ("dinov2_image_to_image", kf_id in dino_top_ids),
                        ("clip_image_to_semantic_chunk", kf_id in chunk_top_ids),
                    )
                    if included
                ],
                "best_semantic_chunk": best_chunks.get(kf_id, "")[:260],
            }
        for record in candidate_records:
            kf_id = int(record.get("keyframe_id"))
            retrieval_label = (
                "DINO image similarity"
                if not use_clip_chunks
                else "DINO image similarity + CLIP image-to-semantic-chunk similarity"
            )
            record["match_reason"] = (
                f"Candidate retrieved by direct attached-image matching ({retrieval_label}), "
                "then ordered by VLM rerank over the query image and candidate keyframes."
            )
            if kf_id in diagnostic_by_id:
                record["direct_hybrid_scores"] = diagnostic_by_id[kf_id]

        return {
            "status": "ok",
            "image_ref": image_ref,
            "image_path": str(image_path),
            "description": "",
            "query": query,
            "focus": focus,
            "requirement": requirement,
            "retrieval_mode": "attached_image_direct_hybrid",
            "resolution_status": "resolved",
            "recommended_keyframe_id": int(recommended_keyframe_id),
            "recommended_destination": recommended_destination,
            "best_keyframe_id": int(recommended_keyframe_id),
            "recommendation_reason": str(rerank.get("reason") or "").strip()
            or "Best keyframe from direct image hybrid retrieval and VLM rerank.",
            "ranked_keyframes": ranked_keyframes,
            "candidate_keyframe_ids": [int(kf_id) for kf_id in ranked_ids],
            "candidate_keyframes": candidate_records,
            "selection_required": False,
            "selection_instruction": (
                "Use recommended_keyframe_id when resolution_status is resolved. "
                "Candidate evidence includes direct_hybrid_scores for debugging."
            ),
            "direct_hybrid": {
                "weights": weights,
                "clip_chunks_used": use_clip_chunks,
                "timings_sec": timings,
                "hybrid_top": [
                    {
                        **item,
                        "best_semantic_chunk": best_chunks.get(int(item["keyframe_id"]), "")[:180],
                    }
                    for item in hybrid_top
                ],
                "dinov2_top": _top_score_items(dino_scores, limit=12),
                "clip_image_to_semantic_chunk": chunk_diagnostics,
                "vlm_rerank": rerank,
            },
        }

    def _rerank_direct_hybrid_candidates(
        self,
        *,
        image: Image.Image,
        candidate_ids: list[int],
        focus: str,
        query: str,
    ) -> dict[str, Any]:
        if not bool(config.get("attached_image_direct_hybrid_vlm_rerank_enabled", True)):
            return {
                "best_keyframe_id": int(candidate_ids[0]),
                "ranked_keyframe_ids": [int(kf_id) for kf_id in candidate_ids],
                "reason": "VLM rerank disabled; using top direct-hybrid candidate.",
                "used": False,
            }
        grid = _make_query_candidate_grid(image, self.scene_memory, candidate_ids)
        question = (
            "You are evaluating visual place recognition for a mobile robot. "
            "The stitched image contains one QUERY photo and candidate keyframes labeled by KF id. "
            "Select the candidate that looks most like it was photographed from the same physical place "
            "as the QUERY image. Prioritize spatial consistency over object-category matches. "
            "Compare the geometry and relative layout of stable structures: walls, floors, doors, elevators, "
            "columns, cabinets, display walls, corners, openings, large furniture, and floor lines. "
            "Objects may have moved, been added, removed, wrapped, or partially occluded since the keyframe "
            "map was collected, so treat transient objects as weak evidence unless their position agrees with "
            "the stable scene layout. Do not choose a keyframe merely because it contains similar objects "
            "such as an elevator, fire box, chair, or sign; their spatial positions and surrounding structure "
            "must also match. Camera viewpoint, crop, tilt, and lighting can differ, but the best match should "
            "still preserve the same place layout and nearby landmark arrangement. "
            "If a candidate has the same object types but a different layout, rank it lower than a candidate "
            "with fewer matching objects but stronger spatial/structural consistency. "
            f"Focus: {focus}. User query if any: {str(query or '').strip() or '(none)'}\n"
            "Return only JSON inside <answer>: "
            "{\"ranked_keyframe_ids\":[id1,id2,...],\"best_keyframe_id\":id1,"
            "\"reason\":\"short evidence focused on same-place spatial and structural consistency\"}"
        )
        request = {
            "request_id": 0,
            "model": config.get("vlm_model_analyse_images", config.get("llm_model", "deepseek-chat")),
            "messages": vlm_analyse_on_each_kf_images_request_message(grid, question),
        }
        try:
            results = asyncio.run(
                asyncio.wait_for(
                    UnifiedLLMClient().batch_chat_completion([request]),
                    timeout=float(config.get("attached_image_direct_hybrid_vlm_timeout_sec", 45)),
                )
            )
            response = results.get(0, {}).get(0)
            answer = extract_answer_tags(str(response or ""))
            parsed = _parse_vlm_keyframe_rerank(answer, candidate_ids)
            parsed["used"] = True
            return parsed
        except Exception as exc:
            return {
                "best_keyframe_id": int(candidate_ids[0]),
                "ranked_keyframe_ids": [int(kf_id) for kf_id in candidate_ids],
                "reason": f"VLM rerank failed; using top direct-hybrid candidate. {type(exc).__name__}: {exc}",
                "used": False,
            }


class HistoricalKeyframeObjectPreanalysisTool(ToolBase):
    """Run semantic object approach on a historical stereo keyframe image."""

    def __init__(self, *, preload: bool = True) -> None:
        super().__init__(
            name="preanalyze_object_on_keyframe",
            description=(
                "Pre-analyze a static object on a historical stereo keyframe and return "
                "a map-frame position destination if reliable. This is background-safe "
                "because it never reads live camera images."
            ),
            capability_tags=("object_preanalysis", "semantic_grounding", "background_safe"),
        )
        self._pipeline = None
        self._preload_error = ""
        if preload:
            self._preload_default_pipeline()

    def execute(
        self,
        keyframe_id: int,
        object_description: str,
        stop_distance_m: float = 0.8,
    ) -> dict[str, Any]:
        try:
            node = self.scene_memory.keyframe_nodes.get(int(keyframe_id))
        except Exception:
            node = None
        if node is None:
            return self.blocked(
                f"Keyframe {keyframe_id} is unavailable for object preanalysis.",
                data={"keyframe_id": keyframe_id, "object_description": object_description},
                error={"code": "keyframe_not_found"},
                provenance={"source_type": "scene_memory"},
            )
        left_path = _optional_path(getattr(node, "left_path", None) or getattr(node, "rgb_path", None))
        right_path = _optional_path(getattr(node, "right_path", None))
        if left_path is None or not left_path.exists():
            return self.blocked(
                "Historical keyframe left image is unavailable.",
                data={"keyframe_id": int(keyframe_id), "left_path": str(left_path) if left_path else None},
                error={"code": "keyframe_left_image_missing"},
                provenance={"source_type": "scene_memory"},
            )
        if right_path is None or not right_path.exists():
            return self.partial(
                "Historical keyframe right image is unavailable; stereo object preanalysis was skipped.",
                data={"keyframe_id": int(keyframe_id), "left_path": str(left_path), "right_path": str(right_path) if right_path else None},
                error={"code": "keyframe_right_image_missing"},
                provenance={"source_type": "scene_memory"},
            )
        current_state = {
            "position": np.asarray(getattr(node, "position", [0.0, 0.0, 0.0]), dtype=float).reshape(-1).tolist(),
            "orientation": np.asarray(getattr(node, "orientation", [0.0, 0.0, 0.0, 1.0]), dtype=float).reshape(-1).tolist(),
            "source": "historical_keyframe_preanalysis",
            "keyframe_id": int(keyframe_id),
        }
        try:
            pipeline = self._get_pipeline()
            result = pipeline.run(
                image=Image.open(left_path).convert("RGB"),
                right_image=Image.open(right_path).convert("RGB"),
                scan_msg=None,
                target_description=object_description,
                current_state=current_state,
                stop_distance_m=float(stop_distance_m or 0.8),
                depth_backend=_configured_object_approach_depth_backend(),
                dispatch=False,
            )
        except Exception as exc:
            return self.partial(
                "Historical object preanalysis failed.",
                data={
                    "keyframe_id": int(keyframe_id),
                    "object_description": object_description,
                    "left_path": str(left_path),
                    "right_path": str(right_path),
                    "output_root": str(self._output_root()),
                },
                error={"code": "historical_object_preanalysis_failed", "message": str(exc)},
                provenance={"source_type": "scene_memory"},
            )
        approach = result.get("approach") if isinstance(result, dict) else None
        goal = (approach or {}).get("map_goal") if isinstance(approach, dict) else None
        position = (goal or {}).get("position") if isinstance(goal, dict) else None
        yaw_deg = (goal or {}).get("yaw_deg") if isinstance(goal, dict) else None
        if isinstance(position, list) and len(position) >= 2 and yaw_deg is not None:
            destination = {
                "type": "position",
                "position": [
                    float(position[0]),
                    float(position[1]),
                    float(position[2] if len(position) > 2 else 0.0),
                ],
                "yaw_deg": float(yaw_deg),
                "source": "historical_keyframe_object_preanalysis",
                "keyframe_id": int(keyframe_id),
                "target_description": object_description,
            }
            result["destination"] = destination
            return self.ok(
                f"Preanalyzed static object '{object_description}' on keyframe {keyframe_id}.",
                data=result,
                provenance={"source_type": "scene_memory"},
            )
        reason = ""
        if isinstance(approach, dict):
            reason = str(approach.get("reason") or approach.get("status") or "")
        return self.partial(
            "Historical object preanalysis did not produce a reliable destination.",
            data=result,
            error={"code": "historical_object_destination_unavailable", "message": reason},
            provenance={"source_type": "scene_memory"},
        )

    def _output_root(self) -> Path:
        return _workspace_root() / "perception_outputs" / "object_preanalysis"

    def _get_pipeline(self):
        output_root = self._output_root()
        from caragent_agent.perception.fusion.object_approach_pipeline import ObjectApproachPipeline

        if self._pipeline is None:
            self._pipeline = ObjectApproachPipeline(output_root=output_root)
            self._pipeline.preload_models()
        else:
            self._pipeline.output_root = output_root
        return self._pipeline

    def _preload_default_pipeline(self) -> None:
        try:
            self._get_pipeline()
        except Exception as exc:
            self._preload_error = f"{type(exc).__name__}: {exc}"


class AttachedImageObjectResolverTool(ToolBase):
    """Try static historical preanalysis for an object named in an attached image."""

    def __init__(self) -> None:
        super().__init__(
            name="resolve_object_from_attached_image",
            description=(
                "Optional shortcut for static historical preanalysis of an object named "
                "in a user-attached image. It first matches the image to a historical "
                "keyframe, then attempts stereo semantic object preanalysis on that saved "
                "keyframe image. Do not use as the normal main path for going to an "
                "object shown in an attached image; normally match the attached image "
                "to a keyframe first, navigate there, then use live current-view object "
                "localization."
            ),
            capability_tags=("attached_image", "semantic_grounding", "object_preanalysis", "background_unsafe"),
        )
        self._matcher = AttachedImageKeyframeMatcherTool()
        self._preanalyzer = HistoricalKeyframeObjectPreanalysisTool(preload=False)

    @property
    def scene_memory(self):
        return self._scene_memory

    @scene_memory.setter
    def scene_memory(self, value):
        self._scene_memory = value
        self._matcher.scene_memory = value
        self._preanalyzer.scene_memory = value

    @property
    def run_memory(self):
        return self._run_memory

    @run_memory.setter
    def run_memory(self, value):
        self._run_memory = value
        self._matcher.run_memory = value
        self._preanalyzer.run_memory = value

    def execute(self, image_ref: str, object_description: str, stop_distance_m: float = 0.8) -> dict[str, Any]:
        match_result = self._matcher._match(image_ref, query=object_description)
        candidate_ids = match_result.get("candidate_keyframe_ids")
        keyframe_id = candidate_ids[0] if isinstance(candidate_ids, list) and candidate_ids else None
        if keyframe_id is None:
            return self.partial(
                "Could not match the attached image to a keyframe for object localization.",
                data={"match": match_result, "object_description": object_description},
                error={"code": "attached_image_keyframe_unmatched", "message": match_result.get("reason") or ""},
                provenance={"source_type": "attached_image"},
            )
        preanalysis = self._preanalyzer.execute(
            keyframe_id=int(keyframe_id),
            object_description=object_description,
            stop_distance_m=stop_distance_m,
        )
        status = str(preanalysis.get("status") or "").lower() if isinstance(preanalysis, dict) else ""
        data = preanalysis.get("data") if isinstance(preanalysis, dict) else None
        payload = {
            "match": match_result,
            "static_preanalysis_keyframe_id": int(keyframe_id),
            "object_preanalysis": data,
        }
        if status == "ok" and isinstance(data, dict) and data.get("destination"):
            payload["destination"] = data.get("destination")
            return self.ok(
                "Resolved object from attached image to a map position.",
                data=payload,
                provenance={"source_type": "attached_image"},
            )
        return self.partial(
            "Attached image object localization did not produce a reliable destination.",
            data=payload,
            error=(preanalysis.get("error") if isinstance(preanalysis, dict) else {"code": "object_preanalysis_failed"}),
            provenance={"source_type": "attached_image"},
        )


__all__ = [
    "AttachedImageAnalyzerTool",
    "AttachedImageKeyframeMatcherTool",
    "AttachedImageObjectResolverTool",
    "HistoricalKeyframeObjectPreanalysisTool",
]
