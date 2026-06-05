"""GroundingDINO inference through OpenVINO.

The model forward pass runs with OpenVINO IR. Hugging Face's processor is still
used for image/text preprocessing and postprocessing so the output schema stays
compatible with the PyTorch prototype.
"""

from __future__ import annotations

import inspect
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


DEFAULT_MODEL_ID = "IDEA-Research/grounding-dino-tiny"
DEFAULT_IMAGE = Path("/home/car/caragent_ws/keyframes/session_20260524_005910/selected/left/000123.png")
DEFAULT_MODELS_DIR = Path("/home/car/caragent_ws/models/grounding_dino_openvino")
DEFAULT_OUTPUT_DIR = Path("/home/car/caragent_ws/perception_outputs/grounding_dino")


def build_text_attention_inputs(tokenizer: Any, inputs: dict[str, Any], max_text_len: int = 256) -> dict[str, np.ndarray]:
    """Build the extra text tensors required by the OpenVINO GroundingDINO fork."""

    from groundingdino.models.GroundingDINO.bertwarper import (
        generate_masks_with_special_tokens_and_transfer_map,
    )

    import torch

    torch_inputs = {
        key: torch.from_numpy(np.asarray(value)) for key, value in inputs.items() if key in {"input_ids", "attention_mask", "token_type_ids"}
    }
    special_tokens = tokenizer.convert_tokens_to_ids(["[CLS]", "[SEP]", ".", "?"])
    text_self_attention_masks, position_ids, _ = generate_masks_with_special_tokens_and_transfer_map(
        torch_inputs,
        special_tokens,
        tokenizer,
    )
    if text_self_attention_masks.shape[1] > max_text_len:
        text_self_attention_masks = text_self_attention_masks[:, :max_text_len, :max_text_len]
        position_ids = position_ids[:, :max_text_len]
    return {
        "position_ids": tensor_to_numpy(position_ids),
        "text_self_attention_masks": tensor_to_numpy(text_self_attention_masks),
    }


def tensor_to_numpy(value: Any) -> np.ndarray:
    """Convert torch/openvino/numpy tensor-like values to numpy arrays."""

    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    if hasattr(value, "cpu") and hasattr(value.cpu(), "numpy"):
        return value.cpu().numpy()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def resize_pixel_values_to_model_input(pixel_values: np.ndarray, target_size: tuple[int, int] | None) -> np.ndarray:
    """Resize processor output to the static image shape captured in the IR."""

    if target_size is None:
        return pixel_values
    expected_h, expected_w = target_size
    if list(pixel_values.shape[-2:]) == [expected_h, expected_w]:
        return pixel_values

    import torch
    import torch.nn.functional as F

    resized = F.interpolate(
        torch.from_numpy(np.asarray(pixel_values)),
        size=(expected_h, expected_w),
        mode="bilinear",
        align_corners=False,
    )
    return resized.numpy()


def normalize_detection_result(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize HF GroundingDINO postprocessor output to CarAgent schema."""

    boxes = tensor_to_numpy(result.get("boxes", []))
    scores = tensor_to_numpy(result.get("scores", []))
    labels = result.get("text_labels", result.get("labels", []))
    if hasattr(labels, "detach") or hasattr(labels, "numpy"):
        labels = tensor_to_numpy(labels).tolist()

    detections: list[dict[str, Any]] = []
    for idx, box in enumerate(boxes):
        label = str(labels[idx]) if idx < len(labels) else "object"
        score = float(scores[idx]) if idx < len(scores) else 0.0
        box_list = [float(v) for v in np.asarray(box).reshape(-1)[:4]]
        detections.append(
            {
                "label": label,
                "score": score,
                "box": box_list,
                "box_int": [round(v) for v in box_list],
            }
        )
    return detections


def postprocess_grounding_dino(
    processor: Any,
    outputs: Any,
    input_ids: Any,
    image_size: tuple[int, int],
    box_threshold: float,
    text_threshold: float,
) -> list[dict[str, Any]]:
    """Run the version-compatible HF GroundingDINO postprocessor."""

    import torch
    from transformers.models.grounding_dino.modeling_grounding_dino import (
        GroundingDinoObjectDetectionOutput,
    )

    if isinstance(outputs, dict):
        logits = outputs.get("logits")
        pred_boxes = outputs.get("pred_boxes")
    else:
        logits = getattr(outputs, "logits", None)
        pred_boxes = getattr(outputs, "pred_boxes", None)
    if logits is None or pred_boxes is None:
        raise RuntimeError("OpenVINO output must contain logits and pred_boxes.")

    logits_t = torch.from_numpy(tensor_to_numpy(logits))
    boxes_t = torch.from_numpy(tensor_to_numpy(pred_boxes))
    model_output = GroundingDinoObjectDetectionOutput(logits=logits_t, pred_boxes=boxes_t)
    target_sizes = torch.tensor([image_size[::-1]], dtype=torch.float32)

    postprocess = processor.post_process_grounded_object_detection
    params = inspect.signature(postprocess).parameters
    kwargs: dict[str, Any] = {
        "input_ids": input_ids,
        "text_threshold": text_threshold,
        "target_sizes": target_sizes,
    }
    if "box_threshold" in params:
        kwargs["box_threshold"] = box_threshold
    else:
        kwargs["threshold"] = box_threshold
    processed = postprocess(model_output, **kwargs)
    return normalize_detection_result(processed[0])


class GroundingDINOOpenVINO:
    """Minimal OpenVINO backend for GroundingDINO object grounding."""

    def __init__(
        self,
        model_dir: str | Path = DEFAULT_MODELS_DIR,
        model_id: str = DEFAULT_MODEL_ID,
        device: str = "CPU",
    ) -> None:
        import openvino as ov
        from transformers import AutoProcessor, AutoTokenizer

        self.model_dir = Path(model_dir)
        self.model_id = model_id
        self.device = device
        self.model_xml = self.model_dir / "openvino_model.xml"
        if not self.model_xml.exists():
            raise FileNotFoundError(
                f"OpenVINO model not found: {self.model_xml}. "
                "Run convert_grounding_dino_openvino.py first."
            )

        self.processor = AutoProcessor.from_pretrained(model_id)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        except Exception:
            self.tokenizer = getattr(self.processor, "tokenizer", None)
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        core = ov.Core()
        self.compiled_model = core.compile_model(str(self.model_xml), device)
        self.input_names = [inp.get_any_name() for inp in self.compiled_model.inputs]
        self.image_input_size = self._load_image_input_size()
        self.output_names = [out.get_any_name() for out in self.compiled_model.outputs]
        source_repo_path = self.model_dir / "source_repo.txt"
        if source_repo_path.exists():
            source_repo = source_repo_path.read_text(encoding="utf-8").strip()
            if source_repo:
                import sys

                sys.path.insert(0, source_repo)

    def _load_image_input_size(self) -> tuple[int, int] | None:
        size_path = self.model_dir / "image_input_size.txt"
        if size_path.exists():
            parts = size_path.read_text(encoding="utf-8").split()
            if len(parts) >= 2:
                return int(parts[0]), int(parts[1])
        return 1024, 1280

    def detect(
        self,
        image_path: str | Path,
        text_prompt: str,
        box_threshold: float = 0.25,
        text_threshold: float = 0.20,
    ) -> dict[str, Any]:
        image_path = Path(image_path)
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, text=text_prompt, return_tensors="np")
        if "pixel_values" not in inputs:
            raise RuntimeError(
                "GroundingDINO processor did not produce pixel_values. "
                f"Loaded processor from {DEFAULT_MODEL_ID}; check that Transformers "
                "can access the model processor files."
            )
        inputs["pixel_values"] = resize_pixel_values_to_model_input(
            inputs["pixel_values"],
            self.image_input_size,
        )
        ov_inputs = {
            "pixel_values": inputs["pixel_values"],
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "token_type_ids": inputs["token_type_ids"],
        }
        ov_inputs.update(build_text_attention_inputs(self.tokenizer, ov_inputs))
        if any(name not in ov_inputs for name in self.input_names):
            ordered_keys = [
                "pixel_values",
                "input_ids",
                "attention_mask",
                "position_ids",
                "token_type_ids",
                "text_self_attention_masks",
            ]
            ov_inputs = {
                model_input: ov_inputs[source_key]
                for model_input, source_key in zip(self.input_names, ordered_keys)
                if source_key in ov_inputs
            }

        start = time.perf_counter()
        raw_outputs = self.compiled_model(ov_inputs)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        outputs_by_name: dict[str, Any] = {}
        for ov_output, value in raw_outputs.items():
            name = ov_output.get_any_name()
            outputs_by_name[name] = value
        if "logits" not in outputs_by_name or "pred_boxes" not in outputs_by_name:
            # Some OpenVINO versions expose generic names. GroundingDINO returns
            # exactly two outputs, logits then pred_boxes, after conversion below.
            values = list(raw_outputs.values())
            if len(values) >= 2:
                outputs_by_name = {"logits": values[0], "pred_boxes": values[1]}

        detections = postprocess_grounding_dino(
            processor=self.processor,
            outputs=outputs_by_name,
            input_ids=inputs["input_ids"],
            image_size=image.size,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )

        return {
            "detections": detections,
            "metadata": {
                "backend": "openvino",
                "device": self.device,
                "model_id": self.model_id,
                "model_dir": str(self.model_dir),
                "image": str(image_path),
                "image_size": [image.width, image.height],
                "text": text_prompt,
                "box_threshold": box_threshold,
                "text_threshold": text_threshold,
                "elapsed_ms": elapsed_ms,
            },
        }
