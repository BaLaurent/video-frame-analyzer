"""Frame comparator using OpenCV HSV histogram correlation."""

import cv2
import numpy as np

from vfa.comparators.base import FrameComparator


class HistogramComparator(FrameComparator):
    """Compare frames via 3D HSV histogram correlation.

    Converts each frame from BGR to HSV, computes a 3D histogram over
    the H, S, and V channels, normalizes it, then uses
    ``cv2.compareHist`` with the correlation method.  The correlation
    score (1.0 = identical, -1.0 = inverse) is inverted and clamped to
    produce a difference score in [0.0, 1.0].
    """

    _H_BINS = 50
    _S_BINS = 60
    _V_BINS = 64
    _CHANNELS = [0, 1, 2]
    _RANGES = [0, 180, 0, 256, 0, 256]

    def compare(self, frame_a: np.ndarray, frame_b: np.ndarray) -> float:
        """Return histogram-based difference between two BGR frames.

        Args:
            frame_a: First frame as a BGR numpy array (OpenCV format).
            frame_b: Second frame as a BGR numpy array (OpenCV format).

        Returns:
            Float in [0.0, 1.0] where 0.0 means identical and 1.0 means
            completely different.
        """
        hist_a = self._compute_histogram(frame_a)
        hist_b = self._compute_histogram(frame_b)

        correlation = cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_CORREL)
        difference = 1.0 - correlation
        return float(np.clip(difference, 0.0, 1.0))

    @property
    def name(self) -> str:
        """Algorithm name used in logs and metadata output."""
        return "histogram"

    def _compute_histogram(self, frame: np.ndarray) -> np.ndarray:
        """Convert a BGR frame to HSV and return its normalized 3D histogram."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist(
            [hsv],
            self._CHANNELS,
            None,
            [self._H_BINS, self._S_BINS, self._V_BINS],
            self._RANGES,
        )
        cv2.normalize(hist, hist)
        return hist
