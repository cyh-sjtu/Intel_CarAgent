"""Image splitting and quality metrics for keyframe recording."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class ImageQuality:
    blur_score: float
    brightness_mean: float
    brightness_std: float
    quality_ok: bool
    reject_reason: str

    def to_dict(self) -> dict:
        return {
            "blur_score": float(self.blur_score),
            "brightness_mean": float(self.brightness_mean),
            "brightness_std": float(self.brightness_std),
            "quality_ok": bool(self.quality_ok),
            "reject_reason": self.reject_reason,
        }


def split_side_by_side(
    frame: np.ndarray,
    *,
    left_width: int = 640,
    right_width: int = 640,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Split a side-by-side stereo frame into left and right images."""

    if frame is None or frame.ndim < 2:
        raise ValueError("frame must be a non-empty image array")
    expected_width = int(left_width) + int(right_width)
    width = int(frame.shape[1])
    if width < expected_width:
        return frame.copy(), None
    left = frame[:, : int(left_width)].copy()
    right = frame[:, int(left_width) : expected_width].copy()
    return left, right


def compute_image_quality(
    image_bgr: np.ndarray,
    *,
    blur_min: float = 80.0,
    brightness_min: float = 35.0,
    brightness_max: float = 235.0,
    contrast_min: float = 15.0,
    metric_width: int = 320,
) -> ImageQuality:
    """Compute deterministic quality metrics for one BGR image."""

    if image_bgr is None or image_bgr.size == 0:
        return ImageQuality(0.0, 0.0, 0.0, False, "empty_image")

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if image_bgr.ndim == 3 else image_bgr
    if metric_width > 0 and gray.shape[1] > metric_width:
        scale = float(metric_width) / float(gray.shape[1])
        gray = cv2.resize(
            gray,
            (metric_width, max(1, int(round(gray.shape[0] * scale)))),
            interpolation=cv2.INTER_AREA,
        )

    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness_mean = float(np.mean(gray))
    brightness_std = float(np.std(gray))

    reasons = []
    if blur_score < blur_min:
        reasons.append("blur")
    if brightness_mean < brightness_min:
        reasons.append("dark")
    if brightness_mean > brightness_max:
        reasons.append("bright")
    if brightness_std < contrast_min:
        reasons.append("low_contrast")

    return ImageQuality(
        blur_score=blur_score,
        brightness_mean=brightness_mean,
        brightness_std=brightness_std,
        quality_ok=not reasons,
        reject_reason=",".join(reasons),
    )
