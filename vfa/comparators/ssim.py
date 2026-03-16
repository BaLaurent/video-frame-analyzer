"""SSIM-based frame comparator using scikit-image."""

import cv2
import numpy as np
from skimage.metrics import structural_similarity

from vfa.comparators.base import FrameComparator


class SSIMComparator(FrameComparator):
    """Compare frames using Structural Similarity Index (SSIM).

    Frames are converted to grayscale and resized to a fixed resolution
    before comparison to keep computation fast and consistent regardless
    of source resolution.

    The SSIM score (1.0 = identical) is inverted to match the base-class
    contract (0.0 = identical, 1.0 = completely different).

    Args:
        resize_to: Side length for the square comparison resolution.
            Both frames are resized to (resize_to, resize_to) before
            comparison. Default: 256.
    """

    def __init__(self, resize_to: int = 256) -> None:
        self._resize_to = resize_to

    def compare(self, frame_a: np.ndarray, frame_b: np.ndarray) -> float:
        """Return a difference score between two frames using SSIM.

        Args:
            frame_a: First frame as a BGR numpy array (OpenCV format).
            frame_b: Second frame as a BGR numpy array (OpenCV format).

        Returns:
            Float in [0.0, 1.0] where 0.0 means identical and 1.0 means
            completely different.
        """
        size = (self._resize_to, self._resize_to)

        gray_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
        gray_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)

        gray_a = cv2.resize(gray_a, size, interpolation=cv2.INTER_AREA)
        gray_b = cv2.resize(gray_b, size, interpolation=cv2.INTER_AREA)

        ssim_score: float = structural_similarity(gray_a, gray_b, data_range=255)

        return float(np.clip(1.0 - ssim_score, 0.0, 1.0))

    @property
    def name(self) -> str:
        """Algorithm name used in logs and metadata output."""
        return "ssim"
