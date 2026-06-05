"""OpenVINO CLIP encoder wrappers."""

from __future__ import annotations

from pathlib import Path

import numpy as np


CLIP_IMAGE_MEAN = np.asarray([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
CLIP_IMAGE_STD = np.asarray([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


def _load_openvino_core():
    """Load OpenVINO Core across old and new Python APIs."""

    try:
        from openvino import Core

        return Core
    except Exception as new_api_exc:
        try:
            from openvino.runtime import Core

            return Core
        except Exception as old_api_exc:
            raise RuntimeError(
                "OpenVINO runtime is not installed or cannot be imported. "
                "Install openvino before running CLIP selection."
            ) from old_api_exc


def _shape_text(array: np.ndarray) -> str:
    return "x".join(str(dim) for dim in array.shape)


def extract_clip_embedding(outputs: list[np.ndarray], expected_dim: int = 512) -> np.ndarray:
    """Extract a single CLIP image embedding from OpenVINO outputs.

    The correct ViT-B/32 image embedding is the projected vector returned by
    CLIPModel.get_image_features(), usually shaped [1, 512]. Hidden-state
    exports such as [1, 50, 768] are rejected because flattening them would
    corrupt cosine similarity.
    """

    expected_dim = int(expected_dim)
    candidates = [np.asarray(output, dtype=np.float32) for output in outputs]
    exact = [output.reshape(-1) for output in candidates if output.reshape(-1).size == expected_dim]
    if exact:
        return exact[0]

    shapes = ", ".join(_shape_text(output) for output in candidates)
    token_like = []
    for output in candidates:
        flat_size = int(output.reshape(-1).size)
        hidden_width = int(output.shape[-1]) if output.ndim >= 2 else None
        sequence_like = output.ndim >= 2 and hidden_width in {512, 768, 1024} and flat_size > expected_dim
        flattened_sequence = flat_size > expected_dim and any(flat_size % width == 0 for width in (512, 768, 1024))
        if sequence_like or flattened_sequence:
            token_like.append(output)
    if token_like:
        raise ValueError(
            "OpenVINO CLIP model returned token hidden states "
            f"with shape(s) [{shapes}], not a {expected_dim}-D projected image embedding. "
            "Re-export the model with CLIPModel.get_image_features() so visual_projection is included."
        )

    raise ValueError(
        f"OpenVINO CLIP model output shape(s) [{shapes}] do not contain a {expected_dim}-D image embedding."
    )


class OpenVINOClipImageEncoder:
    """Small OpenVINO wrapper for CLIP image encoder IR models."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        device: str = "AUTO",
        input_size: int = 224,
        expected_dim: int = 512,
    ) -> None:
        model_path = Path(model_path).expanduser()
        if not model_path.exists():
            raise FileNotFoundError(f"OpenVINO CLIP model not found: {model_path}")

        self.model_path = model_path
        self.device = str(device)
        self.input_size = int(input_size)
        self.expected_dim = int(expected_dim)
        Core = _load_openvino_core()
        self.core = Core()
        self.model = self.core.read_model(str(model_path))
        self.compiled_model = self.core.compile_model(self.model, self.device)
        self.input_layer = self.compiled_model.input(0)
        self.output_layers = [self.compiled_model.output(index) for index in range(len(self.compiled_model.outputs))]

    def encode_path(self, image_path: str | Path) -> np.ndarray:
        import cv2

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"failed to read image for CLIP encoding: {image_path}")
        return self.encode_bgr(image)

    def encode_bgr(self, image_bgr: np.ndarray) -> np.ndarray:
        tensor = self._preprocess(image_bgr)
        result = self.compiled_model([tensor])
        embedding = extract_clip_embedding(
            [result[output_layer] for output_layer in self.output_layers],
            expected_dim=self.expected_dim,
        )
        norm = float(np.linalg.norm(embedding))
        if norm > 0.0:
            embedding = embedding / norm
        return embedding.astype(np.float32)

    def _preprocess(self, image_bgr: np.ndarray) -> np.ndarray:
        import cv2

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        height, width = image_rgb.shape[:2]
        short_side = min(height, width)
        if short_side <= 0:
            raise ValueError("empty image")

        scale = float(self.input_size) / float(short_side)
        new_width = int(round(width * scale))
        new_height = int(round(height * scale))
        resized = cv2.resize(image_rgb, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

        y0 = max(0, (new_height - self.input_size) // 2)
        x0 = max(0, (new_width - self.input_size) // 2)
        cropped = resized[y0 : y0 + self.input_size, x0 : x0 + self.input_size]
        if cropped.shape[0] != self.input_size or cropped.shape[1] != self.input_size:
            cropped = cv2.resize(cropped, (self.input_size, self.input_size), interpolation=cv2.INTER_CUBIC)

        image = cropped.astype(np.float32) / 255.0
        image = (image - CLIP_IMAGE_MEAN) / CLIP_IMAGE_STD
        image = np.transpose(image, (2, 0, 1))[None, ...]
        return image.astype(np.float32)


class OpenVINOClipTextEncoder:
    """OpenVINO wrapper for CLIP text encoder IR models."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        model_id: str = "openai/clip-vit-base-patch32",
        device: str = "AUTO",
        expected_dim: int = 512,
    ) -> None:
        model_path = Path(model_path).expanduser()
        if not model_path.exists():
            raise FileNotFoundError(f"OpenVINO CLIP text model not found: {model_path}")

        from transformers import CLIPTokenizer

        self.model_path = model_path
        self.model_id = str(model_id)
        self.device = str(device)
        self.expected_dim = int(expected_dim)
        self.tokenizer = CLIPTokenizer.from_pretrained(
            self.model_id,
            local_files_only=True,
            use_fast=False,
        )
        Core = _load_openvino_core()
        self.core = Core()
        self.model = self.core.read_model(str(model_path))
        self.compiled_model = self.core.compile_model(self.model, self.device)
        self.output_layers = [self.compiled_model.output(index) for index in range(len(self.compiled_model.outputs))]

    def encode_text(self, text: str) -> np.ndarray:
        tokens = self.tokenizer(
            [str(text)],
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="np",
        )
        input_ids = tokens["input_ids"].astype(np.int64)
        attention_mask = tokens["attention_mask"].astype(np.int64)
        inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if len(self.compiled_model.inputs) == 1:
            inputs = [input_ids]
        result = self.compiled_model(inputs)
        embedding = extract_clip_embedding(
            [result[output_layer] for output_layer in self.output_layers],
            expected_dim=self.expected_dim,
        )
        norm = float(np.linalg.norm(embedding))
        if norm > 0.0:
            embedding = embedding / norm
        return embedding.astype(np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Return cosine similarity for normalized or unnormalized vectors."""

    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)
