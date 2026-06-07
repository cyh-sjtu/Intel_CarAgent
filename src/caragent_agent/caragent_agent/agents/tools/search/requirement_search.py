import asyncio
import concurrent.futures
import re
from typing import List, Dict
import torch
import numpy as np

from caragent_agent.agents.tools.base.tool_base import ToolBase
from caragent_agent.utils.llm_handler import UnifiedLLMClient
from caragent_agent.utils.llm_request_generator import llm_search_requirement_on_kf_request_message, extract_and_convert_ids, divide_nodes_into_subsets
from caragent_agent.config.config import config
from caragent_agent.agents.async_agent.runtime.resource_scheduler import (
    clip_search_lock_enabled,
)

NODES_NUMBER_IN_A_REQUEST = int(config.get("nodes_number_in_a_request", 8))
CLIP_FILTER_TOP_K = 20
SEARCH_TOOL_TIMEOUT_SEC = float(config.get("search_tool_timeout_sec", 45))
LEXICAL_FALLBACK_TOP_K = int(config.get("search_lexical_fallback_top_k", 12))


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

class RequirementSearchTool(ToolBase):
    def __init__(self):
        super().__init__(
            name="search_requirement_on_keyframe_nodes",
            description="""
                Retrieve keyframe nodes matching natural language requirements using LLM-powered search.

                Implements batch processing of keyframe nodes through multiple LLM requests to handle large datasets
                efficiently. Matches nodes based on conceptual understanding rather than exact string matching.
                
                Args:
                    requirement (str): 
                        Natural language description of the search criteria. Should describe desired node characteristics.
                        Example: "areas with high security clearance" or "locations near emergency exits"
                
                Returns:
                    List[int]: 
                        Unique identifiers of keyframe nodes satisfying the requirement, aggregated from all batch responses.
                
                Example:
                    >>> toolkit.search_requirement_on_keyframe_nodes("Consist book")
                    [32, 4, 38, 20, 21, 40, 8, 10, 11, 12, 13, 14, 26, 27, 28, 29]
                """,
            capability_tags=("scene_memory_search", "background_safe"),
        )
        self.llm_client = UnifiedLLMClient()

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

    def _filter_nodes_by_clip(self, query_text: str, top_k: int = CLIP_FILTER_TOP_K) -> Dict:
        """Use CLIP model to pre-filter nodes based on similarity (Duplicated logic for stability)"""
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
                if candidates:
                    image_features = np.stack(features_list).astype(np.float32)
                    image_norms = np.linalg.norm(image_features, axis=1, keepdims=True)
                    image_features = image_features / np.maximum(image_norms, 1e-12)
                    similarity = image_features @ text_features.reshape(-1, 1)
                    k = min(top_k, len(candidates))
                    indices = np.argsort(-similarity.reshape(-1))[:k]
                    for idx in indices:
                        kf_id = candidates[int(idx)]
                        filtered_nodes[kf_id] = scene.keyframe_nodes[kf_id]

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

                return filtered_nodes

            if getattr(scene, 'clip_model', None) is None or getattr(scene, 'device', None) is None:
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
            for idx in indices:
                kf_id = candidates[idx.item()]
                filtered_nodes[kf_id] = scene.keyframe_nodes[kf_id]
            
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
                    # print(f"DEBUG: Added node {kf_id} by freq score {freq_scores[i][0]}")

            # print(f"CLIP filtered from {len(scene.keyframe_nodes)} to {len(filtered_nodes)} nodes using query: '{query_text}'")
            return filtered_nodes

        except Exception as e:
            print(f"Error during CLIP filtering: {e}. Fallback to all nodes.")
            return self.scene_memory.keyframe_nodes
        
    def execute(self, 
                requirement: str) -> List[int]:
        try:
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
                    "matched_keyframe_ids": unique_ids,
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
    tokens = re.findall(r"[a-z0-9]+", normalized)
    stop_words = {
        "a", "an", "and", "are", "at", "be", "by", "can", "for", "from", "go",
        "in", "is", "it", "of", "on", "or", "photo", "picture", "place", "see",
        "showing", "the", "there", "to", "top", "view", "where", "with",
    }
    terms = [token for token in tokens if len(token) > 2 and token not in stop_words]
    deduped_terms: list[str] = []
    for term in terms:
        if term not in deduped_terms:
            deduped_terms.append(term)
    return deduped_terms


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
