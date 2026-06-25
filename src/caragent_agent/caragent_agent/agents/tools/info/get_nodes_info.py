import ast
import json
from typing import Any, Dict, List, Optional

from caragent_agent.agents.tools.base.tool_base import ToolBase
from caragent_agent.utils.llm_handler import UnifiedLLMClient

DEFAULT_SEMANTIC_EXCERPT_CHARS = 500


class GetKeyFrameNodesInfoTool(ToolBase):
    def __init__(self):
        super().__init__(
            name="get_keyframe_nodes_info",
            description="""
                Retrieve metadata of keyframe nodes from scene memory.
                
                Allows selective retrieval of node attributes through boolean flags to optimize
                data transfer. Returns a structured dictionary with requested metadata fields.

                Args:
                    keyframe_nodes_id_list (Optional[List[int]]): 
                        Specific keyframe node IDs to query. If None, returns data for all nodes.
                        Default: None.
                    get_name (bool): 
                        Include human-readable name in metadata. Default: False.
                    get_timestamp (bool): 
                        Include the timestamp. Default: False.
                    get_position (bool): 
                        Include 3D position coordinates. Default: False.
                    get_orientation (bool): 
                        Include 3D orientation data. Default: False.
                    get_semantics (bool):
                        Include semantic description. Default: False. By default this returns
                        a compact excerpt, not the full semantic payload.
                    semantic_mode (str):
                        "excerpt" by default; use "full" only when complete debug-level
                        semantics are explicitly needed.

                Returns:
                    Dict[int, Dict]: Nested dictionary containing:
                        - Keys: Keyframe node IDs (integer)
                        - Values: Metadata dictionaries with these optional fields:
                            * kf_id (int): Always included node identifier
                            * name (str): Present when get_name=True
                            * timestamp (str): Present when get_timestamp=True
                            * position (Tuple[float, float, float]): 3D coordinates when get_position=True
                            * orientation (Tuple[float, float, float]): 3D orientation when get_orientation=True
                            * semantics (str): Semantic description when get_semantics=True

                Example Usage:
                    >>> toolkit.get_keyframe_nodes_info([0, 1], get_timestamp=False, get_name=True, get_position=True, get_orientation=True)
                    {0: {'kf_id': 0, 'name': '1305031910.765238', 'position': array([          0,           0,           0]), 'orientation': array([          0,           0,           0,           1])}, 1: {'kf_id': 1, 'name': '1305031911.497291', 'position': array([   -0.18962,   -0.056156,    0.056369]), 'orientation': array([   0.034504,    -0.19873,     -0.1482,     0.96817])}}
                """,
            capability_tags=("scene_memory_search", "background_safe"),
        )
        self.llm_client = UnifiedLLMClient()

    def _normalize_vector(self, value):
        """Convert a coordinate-like value into a plain float list when possible."""

        normalized = self.to_jsonable(value)
        if isinstance(normalized, list):
            normalized_list = []
            for item in normalized:
                try:
                    normalized_list.append(float(item))
                except Exception:
                    normalized_list.append(item)
            return normalized_list
        return normalized

    def _normalize_keyframe_nodes_id_list(self, raw_value: Any) -> List[int]:
        """Accept both real lists and stringified list inputs emitted by LLM tool calls."""

        candidate_value = raw_value
        if candidate_value is None:
            return []

        if isinstance(candidate_value, str):
            stripped = candidate_value.strip()
            if not stripped:
                return []
            for parser in (json.loads, ast.literal_eval):
                try:
                    parsed_value = parser(stripped)
                except Exception:
                    continue
                if isinstance(parsed_value, (list, tuple)):
                    candidate_value = parsed_value
                    break

        if isinstance(candidate_value, tuple):
            candidate_value = list(candidate_value)

        if not isinstance(candidate_value, list):
            raise ValueError("keyframe_nodes_id_list must be a list of keyframe ids")

        normalized_ids: List[int] = []
        for item in candidate_value:
            try:
                normalized_ids.append(int(item))
            except Exception as exc:
                raise ValueError(
                    f"Invalid keyframe node id value: {item!r}"
                ) from exc
        return normalized_ids

    def _semantic_excerpt(self, semantic: str, *, limit: int = DEFAULT_SEMANTIC_EXCERPT_CHARS) -> str:
        text = str(semantic or "").strip()
        if len(text) <= limit:
            return text
        return text[: max(0, limit - 3)].rstrip() + "..."
        
    def execute(self, 
            keyframe_nodes_id_list: Optional[List[int]] = None,
            get_name: bool = False,
            get_timestamp: bool = False,
            get_position: bool = False,
            get_orientation: bool = False,
            get_semantics: bool = False,
            semantic_mode: str = "excerpt") -> Dict[int, Dict]:
        normalized_keyframe_nodes_id_list = self._normalize_keyframe_nodes_id_list(
            keyframe_nodes_id_list
        )

        if not normalized_keyframe_nodes_id_list:
            keyframe_nodes_id_list = []
            for keyframe_node in self.scene_memory.keyframe_nodes.values():
                keyframe_nodes_id_list.append(keyframe_node.kf_id)
        else:
            keyframe_nodes_id_list = normalized_keyframe_nodes_id_list

        info_dict = {}
        for keyframe_nodes_id in keyframe_nodes_id_list:
            keyframe_node = self.scene_memory.keyframe_nodes[keyframe_nodes_id]
            keyframe_node_meta = {}

            keyframe_node_meta['kf_id'] = keyframe_nodes_id
            
            if get_name:
                keyframe_node_meta['name'] = keyframe_node.name
            
            if get_timestamp:
                keyframe_node_meta['timestamp'] = keyframe_node.timestamp
            
            if get_position:
                keyframe_node_meta['position'] = self._normalize_vector(
                    keyframe_node.position
                )

            if get_orientation:
                keyframe_node_meta['orientation'] = self._normalize_vector(
                    keyframe_node.orientation
                )

            if get_semantics:
                mode = str(semantic_mode or "excerpt").strip().lower()
                if mode == "full":
                    keyframe_node_meta['semantics'] = keyframe_node.semantic
                    keyframe_node_meta['semantics_mode'] = "full"
                else:
                    keyframe_node_meta['semantics_excerpt'] = self._semantic_excerpt(
                        keyframe_node.semantic
                    )
                    keyframe_node_meta['semantics_mode'] = "excerpt"
                    keyframe_node_meta['semantics_full_available'] = True
            info_dict[str(keyframe_nodes_id)] = keyframe_node_meta

        normalized_info = self.to_jsonable(info_dict)
        mode = str(semantic_mode or "excerpt").strip().lower()
        return self.ok(
            "Retrieved keyframe-node metadata from scene memory.",
            data={
                "nodes": normalized_info,
                "requested_ids": list(keyframe_nodes_id_list),
                "field_flags": {
                    "get_name": bool(get_name),
                    "get_timestamp": bool(get_timestamp),
                    "get_position": bool(get_position),
                    "get_orientation": bool(get_orientation),
                    "get_semantics": bool(get_semantics),
                    "semantic_mode": "full" if mode == "full" else "excerpt",
                },
            },
            provenance={"source_type": "scene_memory"},
        )

# Smallest test unit
if __name__ == "__main__":
    from pathlib import Path
    from caragent_agent.config.runtime_paths import get_default_scene_dataset_dir
    from caragent_agent.impression_graph.scene_memory import SceneMemory
    dataset_dir = get_default_scene_dataset_dir()
    scene_memory = SceneMemory(dataset_dir)
    scene_memory.load_keyframe_nodes()
    scene_memory.load_keyframe_graph()
    tool = GetKeyFrameNodesInfoTool()
    tool.scene_memory = scene_memory
    print(tool.execute([0, 1], get_timestamp=False, get_name=True, get_position=True, get_orientation=True, get_semantics=True))
