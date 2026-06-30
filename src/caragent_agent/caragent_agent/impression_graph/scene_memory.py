"""Scene memory loader for pre-selected CarAgent keyframes."""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path

import networkx as nx
import numpy as np

from caragent_agent.impression_graph.node import KeyFrameNode


class SceneMemory:
    """Load an already constructed CarAgent scene memory from disk.

    Optionally loads a CLIP ViT-B/32 model for text-to-image similarity search.
    When available, search tools use CLIP to pre-filter keyframe nodes before LLM
    ranking.  Without CLIP the tools fall back to full-set LLM matching.
    """

    def __init__(
        self,
        dataset_dir: str | Path,
        from_stream: bool = True,
        get_semantic: bool = True,
        construct: bool = False,
        coordinate_type: str = "map",
        device: str | None = None,
    ) -> None:
        del from_stream, get_semantic, construct, coordinate_type
        self.dataset_dir = Path(dataset_dir).expanduser().resolve()
        self.region_nodes: dict[int, object] = {}
        self.keyframe_nodes: dict[int, KeyFrameNode] = {}
        self.keyframe_graph = nx.Graph()

        self.device: str | None = None
        self.clip_model = None
        self.clip_preprocess = None
        self.clip_text_encoder = None
        self.clip_lock = threading.RLock()

        load_device = (device or "cpu").strip()
        if load_device.upper() in {"AUTO", "GPU", "NPU", "CPU"}:
            text_model_path = (
                Path(os.environ.get("CARAGENT_CLIP_TEXT_MODEL", ""))
                if os.environ.get("CARAGENT_CLIP_TEXT_MODEL")
                else Path(os.environ.get("CARAGENT_WORKSPACE", "/home/car/caragent_ws"))
                / "models"
                / "clip-vit-base-patch32"
                / "text_encoder.xml"
            )
            try:
                from caragent_memory.openvino_clip import OpenVINOClipTextEncoder

                self.clip_text_encoder = OpenVINOClipTextEncoder(
                    text_model_path,
                    device=load_device,
                )
                self.device = load_device
            except Exception:
                self.clip_text_encoder = None
        # Always load torch CLIP as a reliable fallback (OpenVINO text encoder may
        # produce degenerate embeddings on some hardware / export configurations).
        torch_device = load_device
        if torch_device.upper() in {"AUTO", "GPU", "NPU"}:
            torch_device = "cpu"  # torch CLIP on DK-2500 runs on CPU
        if torch_device == "cuda":
            try:
                import torch
                if not torch.cuda.is_available():
                    torch_device = "cpu"
            except Exception:
                torch_device = "cpu"

        try:
            import clip

            self.clip_model, self.clip_preprocess = clip.load(
                "ViT-B/32", device=torch_device
            )
            self.device = torch_device  # torch path always uses this device
        except Exception:
            self.clip_model = None
            self.clip_preprocess = None
            if self.clip_text_encoder is None:
                self.device = None

        self.load_keyframe_nodes()
        self.load_keyframe_graph()

    def load_keyframe_nodes(self, load_dir: str | Path | None = None) -> None:
        node_dir = Path(load_dir) if load_dir is not None else self.dataset_dir / "constructed_memory" / "keyframe_nodes"
        if not node_dir.exists():
            raise FileNotFoundError(f"Keyframe node directory not found: {node_dir}")

        self.keyframe_nodes = {}
        for json_file in sorted(node_dir.glob("kf_*.json")):
            node = KeyFrameNode.from_json(json_file, dataset_root=self.dataset_dir)
            self.keyframe_nodes[node.kf_id] = node

        if not self.keyframe_nodes:
            raise ValueError(f"No keyframe nodes found in {node_dir}")

    def load_keyframe_graph(self, load_path: str | Path | None = None) -> None:
        graph_path = Path(load_path) if load_path is not None else self.dataset_dir / "constructed_memory" / "keyframe_graph.json"
        if not graph_path.exists():
            raise FileNotFoundError(f"Keyframe graph file not found: {graph_path}")

        graph_data = json.loads(graph_path.read_text(encoding="utf-8"))
        self.keyframe_graph = nx.Graph()
        self.keyframe_graph.add_nodes_from(int(node_id) for node_id in graph_data.get("nodes", []))
        for edge in graph_data.get("edges", []):
            if len(edge) < 2:
                continue
            u = int(edge[0])
            v = int(edge[1])
            data = edge[2] if len(edge) >= 3 and isinstance(edge[2], dict) else {}
            self.keyframe_graph.add_edge(u, v, **data)

        for node_id, node in self.keyframe_nodes.items():
            self.keyframe_graph.add_node(node_id, keyframe_node=node)

    def find_nearest_node(self, position: list[float] | tuple[float, ...] | np.ndarray) -> int:
        query = np.asarray(position, dtype=np.float32)
        if query.size < 2:
            raise ValueError("Position must contain at least x and y.")

        nearest_id = -1
        nearest_distance = float("inf")
        for node_id, node in self.keyframe_nodes.items():
            distance = float(np.linalg.norm(query[:2] - node.position[:2]))
            if distance < nearest_distance:
                nearest_id = node_id
                nearest_distance = distance

        if nearest_id < 0:
            raise ValueError("Scene memory contains no keyframe nodes.")
        return nearest_id

    def get_clip_embedding_matrix(self) -> np.ndarray:
        vectors = [node.clip_encoding for node in self.keyframe_nodes.values() if node.clip_encoding.size > 0]
        return np.stack(vectors) if vectors else np.empty((0, 0), dtype=np.float32)

    def get_dinov2_embedding_matrix(self) -> np.ndarray:
        vectors = [node.dinov2_encoding for node in self.keyframe_nodes.values() if node.dinov2_encoding.size > 0]
        return np.stack(vectors) if vectors else np.empty((0, 0), dtype=np.float32)

    def _get_pyramid_connections(self, *args, **kwargs) -> list:
        del args, kwargs
        return []
