"""EfficientSAM inference through OpenVINO IR.

Encoder-decoder split: encoder runs once per image, decoder runs per detection box.

API mirrors run_efficientsam_from_box.py: takes image + box → returns (mask, iou).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


# Default IR paths (board)
DEFAULT_ENCODER_XML = Path("/home/car/caragent_ws/models/efficient_sam_openvino/efficient_sam_vitt_encoder.xml")
DEFAULT_DECODER_XML = Path("/home/car/caragent_ws/models/efficient_sam_openvino/efficient_sam_vitt_decoder.xml")


def _safe_print(message: str) -> None:
    try:
        print(message, flush=True)
    except BrokenPipeError:
        pass


class EfficientSAMOpenVINO:
    """EfficientSAM inference wrapper using OpenVINO IR."""

    def __init__(
        self,
        encoder_xml: Path | str,
        decoder_xml: Path | str,
        device: str = "CPU",
        encoder_device: str | None = None,
        decoder_device: str | None = None,
    ):
        from caragent_agent.perception.openvino_utils import create_openvino_core

        self.core = create_openvino_core()
        self.device = device
        self.encoder_device = encoder_device or device
        self.decoder_device = decoder_device or device

        encoder_xml = Path(encoder_xml)
        decoder_xml = Path(decoder_xml)

        if not encoder_xml.exists():
            raise FileNotFoundError(f"Encoder IR not found: {encoder_xml}")
        if not decoder_xml.exists():
            raise FileNotFoundError(f"Decoder IR not found: {decoder_xml}")

        _safe_print(f"Loading encoder: {encoder_xml} on {self.encoder_device}")
        self.encoder = self.core.compile_model(str(encoder_xml), self.encoder_device)

        _safe_print(f"Loading decoder: {decoder_xml} on {self.decoder_device}")
        self.decoder = self.core.compile_model(str(decoder_xml), self.decoder_device)

        # Cache encoder input shape for image preprocessing
        enc_input = self.encoder.input(0)
        self._enc_input_name = enc_input.get_any_name()
        _, _, self.enc_h, self.enc_w = enc_input.shape

        dec_output = self.decoder.output(0)
        self._dec_output_mask_name = dec_output.get_any_name()
        self._dec_output_iou_name = self.decoder.output(1).get_any_name()
        self._dec_input_emb = self.decoder.input(0).get_any_name()
        self._dec_input_coords = self.decoder.input(1).get_any_name()
        self._dec_input_labels = self.decoder.input(2).get_any_name()
        self._dec_input_size = self.decoder.input(3).get_any_name()

    def get_embedding(self, image: np.ndarray) -> np.ndarray:
        """Run encoder: image (H, W, 3) np.uint8 → embeddings (1, 256, 64, 64)."""
        # Resize to encoder's expected size and normalize
        pil_img = Image.fromarray(image).resize((self.enc_w, self.enc_h), Image.BILINEAR)
        img_np = np.array(pil_img, dtype=np.float32) / 255.0
        img_np = img_np.transpose(2, 0, 1)[np.newaxis, ...]  # (1, 3, H, W)

        result = self.encoder({self._enc_input_name: img_np})
        embeddings = result[self.encoder.output(0).get_any_name()]  # (1, 256, 64, 64)
        if not np.isfinite(embeddings).all():
            raise RuntimeError(
                f"EfficientSAM encoder produced non-finite values on {self.encoder_device}."
            )
        return embeddings

    def predict_mask(
        self,
        image_embeddings: np.ndarray,
        box_xyxy: tuple[float, float, float, float],
        orig_h: int,
        orig_w: int,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Run decoder with a box prompt.

        Args:
            image_embeddings: from get_embedding(), shape (1, 256, 64, 64)
            box_xyxy: (x1, y1, x2, y2) in original image coordinates
            orig_h, orig_w: original image dimensions

        Returns:
            mask: binary mask at orig_h×orig_w
            all_masks: all 3 candidate masks (for debug)
            iou: predicted IoU of best mask
        """
        x1, y1, x2, y2 = box_xyxy

        # Keep prompt coordinates in original image pixel space. The exported
        # EfficientSAM decoder rescales them internally using orig_im_size.
        point_coords = np.array([[[[x1, y1], [x2, y2]]]], dtype=np.float32)
        point_labels = np.array([[[2, 3]]], dtype=np.float32)  # 2=tl, 3=br
        orig_size = np.array([orig_h, orig_w], dtype=np.int64)

        inputs = {
            self._dec_input_emb: image_embeddings,
            self._dec_input_coords: point_coords,
            self._dec_input_labels: point_labels,
            self._dec_input_size: orig_size,
        }

        result = self.decoder(inputs)
        output_masks = result[self._dec_output_mask_name]  # (1, 1, 3, H, W)
        iou_predictions = result[self._dec_output_iou_name]  # (1, 1, 3)
        if not np.isfinite(output_masks).all() or not np.isfinite(iou_predictions).all():
            raise RuntimeError(
                "EfficientSAM decoder produced non-finite values "
                f"on {self.decoder_device}. Try --decoder-device CPU."
            )

        # Pick best mask by IoU
        best_idx = np.argmax(iou_predictions[0, 0])
        best_mask_logits = output_masks[0, 0, best_idx]
        best_iou = float(iou_predictions[0, 0, best_idx])
        best_mask = (best_mask_logits >= 0.0).astype(np.uint8)

        return best_mask, output_masks[0, 0], best_iou


def create_sam_detector(
    encoder_xml: Path | str = DEFAULT_ENCODER_XML,
    decoder_xml: Path | str = DEFAULT_DECODER_XML,
    device: str = "CPU",
) -> EfficientSAMOpenVINO:
    """Factory function matching the pattern in grounding_dino_openvino.py."""
    return EfficientSAMOpenVINO(
        encoder_xml=Path(encoder_xml),
        decoder_xml=Path(decoder_xml),
        device=device,
    )
