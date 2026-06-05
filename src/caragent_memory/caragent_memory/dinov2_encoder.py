"""DINOv2 image encoder for visual keyframe deduplication."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _resolve_model_ref(model_ref: str | Path) -> str:
    raw = str(model_ref)
    path = Path(raw).expanduser()
    if path.exists():
        return str(path.resolve())
    return raw


def _normalize(embedding: np.ndarray) -> np.ndarray:
    embedding = np.asarray(embedding, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(embedding))
    if norm > 0.0:
        embedding = embedding / norm
    return embedding.astype(np.float32)


class DINOv2ImageEncoder:
    """Hugging Face DINOv2 encoder using the CLS token as image embedding."""

    def __init__(
        self,
        model_ref: str | Path,
        *,
        device: str = "auto",
        local_files_only: bool = True,
    ) -> None:
        try:
            import torch
            from transformers import AutoImageProcessor, Dinov2Model
        except Exception as exc:
            raise RuntimeError(
                "DINOv2 encoding requires torch, transformers, and pillow. "
                "Install them before running select_keyframes "
                "(both CLIP and DINOv2 embeddings are always computed regardless of --dedupe-backend)."
            ) from exc

        self.torch = torch
        self.model_ref = _resolve_model_ref(model_ref)
        self.device = self._select_device(device)
        self.processor = AutoImageProcessor.from_pretrained(
            self.model_ref,
            local_files_only=local_files_only,
        )
        self.model = Dinov2Model.from_pretrained(
            self.model_ref,
            local_files_only=local_files_only,
        ).eval()
        self.model.to(self.device)
        self.embedding_dim = int(getattr(self.model.config, "hidden_size", 0) or 0)

    def _select_device(self, device: str):
        normalized = str(device).strip().lower()
        if normalized in {"", "auto"}:
            normalized = "cuda" if self.torch.cuda.is_available() else "cpu"
        elif normalized == "gpu":
            normalized = "cuda"
        return self.torch.device(normalized)

    def encode_path(self, image_path: str | Path) -> np.ndarray:
        from PIL import Image

        with Image.open(image_path) as raw_image:
            image = raw_image.convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {name: value.to(self.device) for name, value in inputs.items()}
        with self.torch.no_grad():
            output = self.model(**inputs)
            embedding = output.last_hidden_state[:, 0, :]
        return _normalize(embedding.detach().cpu().numpy())
