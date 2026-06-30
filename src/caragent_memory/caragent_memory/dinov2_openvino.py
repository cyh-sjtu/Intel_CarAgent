"""OpenVINO DINOv2 image encoder for keyframe visual deduplication."""

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


class DINOv2OpenVINOImageEncoder:
    """DINOv2 CLS-token image embedding through OpenVINO Runtime."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        processor_ref: str | Path,
        device: str = "NPU",
        input_size: int = 224,
        local_files_only: bool = True,
    ) -> None:
        try:
            import openvino as ov
            from transformers import AutoImageProcessor
        except Exception as exc:
            raise RuntimeError("DINOv2 OpenVINO encoding requires openvino and transformers.") from exc

        self.model_path = Path(model_path).expanduser()
        if not self.model_path.exists():
            raise FileNotFoundError(f"DINOv2 OpenVINO IR not found: {self.model_path}")

        self.processor_ref = _resolve_model_ref(processor_ref)
        self.device = str(device)
        self.input_size = int(input_size)
        self.processor = AutoImageProcessor.from_pretrained(
            self.processor_ref,
            local_files_only=local_files_only,
        )

        self.core = ov.Core()
        self.model = self.core.read_model(str(self.model_path))
        if "NPU" in self.device.upper():
            try:
                self.model.reshape({self.model.input(0).get_any_name(): [1, 3, self.input_size, self.input_size]})
            except Exception as exc:
                raise RuntimeError(f"Failed to reshape DINOv2 IR to static shape for NPU: {exc}") from exc
        self.compiled_model = self.core.compile_model(self.model, self.device)
        self.input_name = self.compiled_model.input(0).get_any_name()
        self.output_layer = self.compiled_model.output(0)
        self.embedding_dim = int(np.prod([int(dim) for dim in self.output_layer.shape[1:]]))

    def encode_path(self, image_path: str | Path) -> np.ndarray:
        from PIL import Image

        with Image.open(image_path) as raw_image:
            image = raw_image.convert("RGB")
        inputs = self.processor(images=image, return_tensors="np")
        pixel_values = inputs["pixel_values"].astype(np.float32)
        result = self.compiled_model({self.input_name: pixel_values})
        embedding = np.asarray(result[self.output_layer], dtype=np.float32)
        if not np.isfinite(embedding).all():
            raise RuntimeError(f"DINOv2 OpenVINO encoder produced non-finite values on {self.device}.")
        return _normalize(embedding)
