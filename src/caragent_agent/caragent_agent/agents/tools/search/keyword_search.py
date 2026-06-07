import asyncio
import concurrent.futures
import re
from typing import List, Dict
import torch
import numpy as np

from caragent_agent.agents.tools.base.tool_base import ToolBase
from caragent_agent.utils.llm_handler import UnifiedLLMClient
from caragent_agent.utils.llm_request_generator import llm_search_keywords_on_kf_request_message, extract_and_convert_ids, divide_nodes_into_subsets
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

class KeywordSearchTool(ToolBase):
    def __init__(self):
        super().__init__(
            name="search_keywords_on_keyframe_nodes",
            description="""
                Search keyframe nodes using keyword matching with configurable search strategy.

                Implements two distinct search modes:
                1. Blur search (LLM-based): Uses language model understanding for conceptual matches
                2. Exact search: Performs direct string matching with boolean logic. Mismatch may happen

                Args:
                    keywords (List[str]): 
                        Search terms to match against keyframe node descriptions
                    logic (str, optional): 
                        Boolean logic for combining keywords:
                        - 'or': Match any keyword (default)
                        - 'and': Require all keywords
                        Default: 'or'
                    isBlur (bool, optional): 
                        Search mode selector:
                        - True: Use LLM for semantic similarity search
                        - False: Use exact string matching
                        Default: True

                Returns:
                    List[int]: IDs of keyframe nodes matching the search criteria

                Raises:
                    ValueError: If invalid logic operator is provided

                Example:
                    >>> toolkit.search_keywords_on_keyframe_nodes(["human"], logic='and')
                    [22, 26, 27, 28, 29, 30]
                """,
            capability_tags=("scene_memory_search", "background_safe"),
        )
        self.llm_client = UnifiedLLMClient()

    def _lexical_fallback_search(
        self,
        keywords: List[str],
        logic: str,
        *,
        top_k: int = LEXICAL_FALLBACK_TOP_K,
    ) -> List[int]:
        """Return local semantic-text matches when the LLM keyword path times out."""

        query_terms = _extract_keyword_terms(keywords)
        if not query_terms:
            return []

        scored_nodes: list[tuple[int, int]] = []
        for kf_id, node in self.scene_memory.keyframe_nodes.items():
            semantic_text = str(getattr(node, "semantic", "") or "").lower()
            if not semantic_text:
                continue
            if logic == "and" and not all(term in semantic_text for term in query_terms):
                continue
            score = _score_semantic_text(semantic_text, query_terms)
            if score > 0:
                try:
                    scored_nodes.append((score, int(kf_id)))
                except Exception:
                    continue

        scored_nodes.sort(key=lambda item: (-item[0], item[1]))
        return [kf_id for _, kf_id in scored_nodes[:top_k]]
        
    def execute(self, 
                keywords: List[str],
                logic: str = 'or',
                is_blur: bool = True) -> List[int]:
        try:
            if logic not in {"or", "and"}:
                return self.blocked(
                    "Keyword search received an unsupported boolean logic mode.",
                    data={
                        "keywords": list(keywords),
                        "logic": logic,
                        "is_blur": bool(is_blur),
                    },
                    error={
                        "code": "invalid_logic",
                        "message": "logic must be 'or' or 'and'.",
                    },
                    provenance={"source_type": "scene_memory"},
                )

            if is_blur:
                matched_ids = self._llm_based_search(keywords, logic)
                search_mode = "semantic"
            else:
                matched_ids = self._exact_search(keywords, logic)
                search_mode = "exact"

            unique_ids = []
            for matched_id in matched_ids:
                try:
                    normalized_id = int(matched_id)
                except Exception:
                    continue
                if normalized_id not in unique_ids:
                    unique_ids.append(normalized_id)

            return self.ok(
                "Searched keyframe nodes by keywords.",
                data={
                    "keywords": list(keywords),
                    "logic": logic,
                    "is_blur": bool(is_blur),
                    "search_mode": search_mode,
                    "matched_keyframe_ids": unique_ids,
                },
                provenance={"source_type": "scene_memory"},
            )
        except Exception as exc:
            error_message = str(exc)
            if isinstance(exc, (asyncio.TimeoutError, TimeoutError, concurrent.futures.TimeoutError)):
                error_message = (
                    f"Keyword search timed out after {SEARCH_TOOL_TIMEOUT_SEC:.0f}s."
                )
                fallback_ids = self._lexical_fallback_search(keywords, logic)
                if fallback_ids:
                    return self.partial(
                        "Keyword search timed out; returned local semantic-text fallback matches.",
                        data={
                            "keywords": list(keywords),
                            "logic": logic,
                            "is_blur": bool(is_blur),
                            "search_mode": "local_semantic_text_fallback",
                            "matched_keyframe_ids": fallback_ids,
                        },
                        error={
                            "code": "keyword_search_timeout_fallback",
                            "message": error_message,
                        },
                        provenance={"source_type": "scene_memory"},
                    )
            return self.error_result(
                "Keyword search failed.",
                data={
                    "keywords": list(keywords),
                    "logic": logic,
                    "is_blur": bool(is_blur),
                },
                error={
                    "code": "keyword_search_failed",
                    "message": error_message,
                },
                provenance={"source_type": "scene_memory"},
            )

    def _filter_nodes_by_clip(self, query_text: str, top_k: int = CLIP_FILTER_TOP_K) -> Dict:
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
                stop_words = {'a', 'an', 'the', 'in', 'on', 'at', 'with', 'and', 'or', 'of', 'to', 'for', 'is', 'are', 'photo', 'containing'}
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
                
            # Stack features
            # Note: clip_encoding from SceneMemory should already be a tensor on correct device or CPU
            # We ensure they are on the target device for computation
            image_features = torch.cat([f.to(device) for f in features_list])
            
            # Ensure dtype matches text_features (Float32 vs Float16 on GPU)
            image_features = image_features.to(dtype=text_features.dtype)

            # Normalize if not already (CLIP encodings usually need normalization for dot product similarity)
            # Use out-of-place division and re-cast to ensure dtype stability (FP16/FP32 mixture avoidance)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            image_features = image_features.to(dtype=text_features.dtype)
            
            # Calculate similarity
            similarity = (image_features @ text_features.T).squeeze()
            
            # Get top K
            k = min(top_k, len(candidates))
            values, indices = similarity.topk(k)
            
            filtered_nodes = {}
            for idx in indices:
                kf_id = candidates[idx.item()]
                filtered_nodes[kf_id] = scene.keyframe_nodes[kf_id]
            
            # Additional Frequency-based Filter
            # Count occurrences of query words in node semantics (naive exact match)
            freq_scores = []
            query_words = query_text.lower().split()
            # Remove common stop words
            stop_words = {'a', 'an', 'the', 'in', 'on', 'at', 'with', 'and', 'or', 'of', 'to', 'for', 'is', 'are', 'photo', 'containing'}
            query_words = [w for w in query_words if w not in stop_words]

            if query_words:
                for kf_id, node in scene.keyframe_nodes.items():
                    if kf_id in filtered_nodes: continue
                    
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

            # print(f"CLIP filtered from {len(scene.keyframe_nodes)} to {len(filtered_nodes)} nodes using query: '{query_text}'")
            return filtered_nodes

        except Exception as e:
            print(f"Error during CLIP filtering: {e}. Fallback to all nodes.")
            return self.scene_memory.keyframe_nodes
        
    def _llm_based_search(self, keywords: List[str], logic: str) -> List[int]:
        """基于LLM的模糊搜索"""
        requests_list = []
        
        # 使用 CLIP 初筛
        # 构造一个简单的 query prompt
        query_text = f"a photo containing {', '.join(keywords)}"
        filtered_nodes = self._filter_nodes_by_clip(query_text)

        # 将节点分组以适应LLM请求限制
        divided_keyframe_nodes = divide_nodes_into_subsets(filtered_nodes, NODES_NUMBER_IN_A_REQUEST)
        
        for index, subset in enumerate(divided_keyframe_nodes):
            request_metadata = {}
            request_metadata["request_id"] = index
            request_metadata['model'] = config['llm_model_search_on_keyframe_nodes']
            request_metadata['messages'] = llm_search_keywords_on_kf_request_message(subset, keywords, logic)
            requests_list.append(request_metadata)
        client = UnifiedLLMClient()
        import time
        begin_time = time.time()
        results = asyncio.run(
            asyncio.wait_for(
                client.batch_chat_completion(requests_list),
                timeout=SEARCH_TOOL_TIMEOUT_SEC,
            )
        )
        end_time = time.time()
        # print(f"Use {end_time-begin_time} seconds using model {config['llm_model_search_on_semantic_nodes']}")
        target_keyframe_nodes_id_list = []
        for req_id, response in results.items():
                target_keyframe_nodes_id_list += extract_and_convert_ids(response[req_id])
        return target_keyframe_nodes_id_list
        
    def _exact_search(self, keywords: List[str], logic: str) -> List[int]:
        """精确字符串匹配搜索"""
        matching_nodes = []
        for kf_id_str, keyframe_node in self.scene_memory.keyframe_nodes.items():
            if logic == 'or':
                if any(keyword in keyframe_node.semantic for keyword in keywords):
                    matching_nodes.append(keyframe_node.kf_id)
            elif logic == 'and':
                if all(keyword in keyframe_node.semantic for keyword in keywords):
                    matching_nodes.append(keyframe_node.kf_id)
        return matching_nodes
    

def _extract_keyword_terms(keywords: List[str]) -> list[str]:
    """Extract stable local-search terms from keyword inputs."""

    raw_text = " ".join(str(keyword or "") for keyword in keywords).lower()
    tokens = re.findall(r"[a-z0-9]+", raw_text)
    stop_words = {
        "a", "an", "and", "are", "at", "by", "for", "from", "in", "is",
        "of", "on", "or", "photo", "picture", "the", "to", "with",
    }
    terms = [token for token in tokens if len(token) > 2 and token not in stop_words]
    deduped_terms: list[str] = []
    for term in terms:
        if term not in deduped_terms:
            deduped_terms.append(term)
    return deduped_terms


def _score_semantic_text(semantic_text: str, query_terms: list[str]) -> int:
    """Score semantic text by exact token overlap."""

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
    from pathlib import Path
    from caragent_agent.config.runtime_paths import get_default_scene_dataset_dir
    from caragent_agent.impression_graph.scene_memory import SceneMemory
    dataset_dir = get_default_scene_dataset_dir()
    scene_memory = SceneMemory(dataset_dir)
    scene_memory.load_keyframe_nodes()
    scene_memory.load_keyframe_graph()
    tool = KeywordSearchTool()
    tool.scene_memory = scene_memory
    print(tool.execute(["gold, silver, and green sculptures"], logic='or', is_blur=True))
    # print(tool.execute(["door", "painting"], logic='and', is_blur=False))
