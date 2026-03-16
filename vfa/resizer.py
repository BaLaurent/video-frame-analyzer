"""Optional frame resizing with aspect-ratio preservation."""

from __future__ import annotations

import cv2
import numpy as np


class FrameResizer:
    """Resize frames so the largest dimension does not exceed a limit.

    Downscaling uses LANCZOS interpolation for high-quality results.
    Frames already within the limit are returned unchanged.
    Upscaling is never performed.

    Parameters
    ----------
    max_dimension:
        Maximum allowed size (in pixels) for width or height.
    enabled:
        When ``False``, :meth:`resize` returns the frame unchanged.
    """

    def __init__(self, max_dimension: int = 1024, enabled: bool = True) -> None:
        self._max_dimension = max_dimension
        self._enabled = enabled

    def resize(self, frame: np.ndarray) -> np.ndarray:
        """Return a resized copy of *frame*, or the original if no resize is needed.

        Parameters
        ----------
        frame:
            A BGR image as a ``numpy.ndarray`` (height, width, channels).

        Returns
        -------
        numpy.ndarray
            The (possibly resized) frame.
        """
        if not self._enabled:
            return frame

        height, width = frame.shape[:2]

        if width <= self._max_dimension and height <= self._max_dimension:
            return frame

        if width >= height:
            new_width = self._max_dimension
            new_height = int(height * (self._max_dimension / width))
        else:
            new_height = self._max_dimension
            new_width = int(width * (self._max_dimension / height))

        return cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
