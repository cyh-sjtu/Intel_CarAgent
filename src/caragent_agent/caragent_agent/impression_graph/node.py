"""CarAgent keyframe node representation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


class KeyFrameNode:
    """In-memory view of one selected CarAgent keyframe JSON file."""

    def __init__(
        self,
        *,
        kf_id: int,
        name: str,
        dataset_dir: str | Path,
        position: np.ndarray,
        orientation: np.ndarray,
        intrinsic: np.ndarray | None = None,
        timestamp: float | None = None,
        clip_encoding: np.ndarray | None = None,
        dinov2_encoding: np.ndarray | None = None,
        semantic: str = "",
        semantic_clip_encoding: np.ndarray | None = None,
        visual_similarity_backend: str = "dinov2",
        rgb_path: str | Path | None = None,
        left_path: str | Path | None = None,
        right_path: str | Path | None = None,
        raw_path: str | Path | None = None,
        scan_path: str | Path | None = None,
    ) -> None:
        self.kf_id = int(kf_id)
        self.name = str(name)
        self.position = np.asarray(position, dtype=np.float32)
        self.orientation = np.asarray(orientation, dtype=np.float32)
        self.intrinsic = np.asarray(intrinsic if intrinsic is not None else [], dtype=np.float32)
        self.timestamp = timestamp
        self.clip_encoding = np.asarray(clip_encoding if clip_encoding is not None else [], dtype=np.float32)
        self.dinov2_encoding = np.asarray(dinov2_encoding if dinov2_encoding is not None else [], dtype=np.float32)
        self.semantic = semantic or ""
        self.semantic_clip_encoding = (
            None
            if semantic_clip_encoding is None
            else np.asarray(semantic_clip_encoding, dtype=np.float32)
        )
        self.visual_similarity_backend = visual_similarity_backend or "dinov2"
        self._asset_paths = {
            "rgb_path": rgb_path,
            "left_path": left_path,
            "right_path": right_path,
            "raw_path": raw_path,
            "scan_path": scan_path,
        }
        self.bind_dataset_dir(dataset_dir)

    @classmethod
    def from_json(cls, path: str | Path, dataset_root: str | Path | None = None) -> "KeyFrameNode":
        json_path = Path(path)
        data = json.loads(json_path.read_text(encoding="utf-8"))
        root = Path(dataset_root or data.get("dataset_dir") or json_path.parents[2]).expanduser()
        rgb_path = data.get("rgb_path") or data.get("left_path")
        return cls(
            kf_id=data["kf_id"],
            name=data.get("name", data.get("frame_id", data["kf_id"])),
            dataset_dir=root,
            position=np.asarray(data.get("position", [0.0, 0.0, 0.0]), dtype=np.float32),
            orientation=np.asarray(data.get("orientation", [0.0, 0.0, 0.0, 1.0]), dtype=np.float32),
            intrinsic=np.asarray(data.get("intrinsic", []), dtype=np.float32),
            timestamp=data.get("timestamp"),
            clip_encoding=np.asarray(data.get("clip_encoding", []), dtype=np.float32),
            dinov2_encoding=np.asarray(data.get("dinov2_encoding", []), dtype=np.float32),
            semantic=data.get("semantic", ""),
            semantic_clip_encoding=data.get("semantic_clip_encoding"),
            visual_similarity_backend=data.get("visual_similarity_backend", "dinov2"),
            rgb_path=rgb_path,
            left_path=data.get("left_path") or rgb_path,
            right_path=data.get("right_path"),
            raw_path=data.get("raw_path"),
            scan_path=data.get("scan_path"),
        )

    @staticmethod
    def load_from_disk(json_path: str | Path) -> "KeyFrameNode":
        return KeyFrameNode.from_json(json_path)

    def bind_dataset_dir(self, dataset_dir: str | Path) -> None:
        self.dataset_dir = Path(dataset_dir).expanduser().resolve()
        for attr, path_value in self._asset_paths.items():
            setattr(self, attr, self._resolve_asset(path_value))
        self.rgb_path = self.rgb_path or self.left_path
        self.depth_path = None

    def _resolve_asset(self, path_value: Any) -> Path | None:
        if path_value is None or str(path_value).strip() == "":
            return None
        path = Path(path_value)
        if path.is_absolute():
            return path
        return self.dataset_dir / path

    def save_to_disk(self, save_dir: str | Path) -> None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "kf_id": self.kf_id,
            "name": self.name,
            "dataset_dir": str(self.dataset_dir),
            "position": self.position.astype(float).tolist(),
            "orientation": self.orientation.astype(float).tolist(),
            "intrinsic": self.intrinsic.astype(float).tolist(),
            "timestamp": self.timestamp,
            "semantic": self.semantic,
            "clip_encoding": self.clip_encoding.astype(float).tolist(),
            "dinov2_encoding": self.dinov2_encoding.astype(float).tolist(),
            "semantic_clip_encoding": (
                None
                if self.semantic_clip_encoding is None
                else self.semantic_clip_encoding.astype(float).tolist()
            ),
            "visual_similarity_backend": self.visual_similarity_backend,
            "rgb_path": self._relative_asset(self.rgb_path),
            "left_path": self._relative_asset(self.left_path),
            "right_path": self._relative_asset(self.right_path),
            "raw_path": self._relative_asset(self.raw_path),
            "scan_path": self._relative_asset(self.scan_path),
        }
        (save_dir / f"kf_{self.kf_id:06d}.json").write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )

    def _relative_asset(self, path_value: Path | None) -> str | None:
        if path_value is None:
            return None
        try:
            return path_value.resolve().relative_to(self.dataset_dir).as_posix()
        except ValueError:
            return str(path_value)
