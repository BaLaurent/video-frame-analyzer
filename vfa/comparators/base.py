"""Abstract base class for frame comparison algorithms."""

from abc import ABC, abstractmethod

import numpy as np


class FrameComparator(ABC):
    """Abstract base for frame comparison algorithms.

    All comparators return a difference score normalized to [0.0, 1.0]:
        - 0.0 = frames are identical
        - 1.0 = frames are completely different

    This is the inverse of a similarity score. Each concrete comparator
    handles the inversion internally (e.g., SSIM natively returns 1.0 for
    identical frames, so the SSIM comparator returns ``1.0 - ssim_score``).
    """

    @abstractmethod
    def compare(self, frame_a: np.ndarray, frame_b: np.ndarray) -> float:
        """Return a difference score between two frames.

        Args:
            frame_a: First frame as a BGR numpy array (OpenCV format).
            frame_b: Second frame as a BGR numpy array (OpenCV format).

        Returns:
            Float in [0.0, 1.0] where 0.0 means identical and 1.0 means
            completely different.
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Algorithm name used in logs and metadata output."""
        ...
