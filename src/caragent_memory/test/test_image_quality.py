import numpy as np
import pytest

pytest.importorskip("cv2")

from caragent_memory.image_quality import compute_image_quality, split_side_by_side


def test_split_side_by_side_returns_left_and_right():
    frame = np.zeros((4, 8, 3), dtype=np.uint8)
    frame[:, :4] = 10
    frame[:, 4:] = 200

    left, right = split_side_by_side(frame, left_width=4, right_width=4)

    assert left.shape == (4, 4, 3)
    assert right.shape == (4, 4, 3)
    assert int(left.mean()) == 10
    assert int(right.mean()) == 200


def test_split_side_by_side_short_frame_returns_left_only():
    frame = np.zeros((4, 6, 3), dtype=np.uint8)

    left, right = split_side_by_side(frame, left_width=4, right_width=4)

    assert left.shape == frame.shape
    assert right is None


def test_quality_rejects_low_contrast_dark_image():
    image = np.zeros((80, 80, 3), dtype=np.uint8)

    quality = compute_image_quality(image)

    assert not quality.quality_ok
    assert "dark" in quality.reject_reason
