"""Streaming video frame extraction via OpenCV."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

from vfa.models import VideoInfo

logger = logging.getLogger(__name__)


def _fourcc_to_str(fourcc_int: int) -> str:
    """Decode a FourCC integer into a 4-character codec string."""
    return "".join(chr((fourcc_int >> (8 * i)) & 0xFF) for i in range(4))


class FrameExtractor:
    """Stream frames from a video file one at a time.

    Yields ``(frame_number, frame)`` tuples where *frame* is a BGR
    ``numpy.ndarray``.  At most two frames are held in memory at any
    point (the current yield and the internal read buffer).

    Parameters
    ----------
    video_path:
        Path to the video file on disk.
    sample_rate:
        Keep one frame every *sample_rate* frames.  ``1`` means every
        frame, ``2`` means every other frame, etc.
    """

    def __init__(self, video_path: str, sample_rate: int = 1) -> None:
        path = Path(video_path)
        if not path.exists():
            raise FileNotFoundError(f"Fichier vidéo introuvable : {video_path}")

        self._path = str(path)
        self._sample_rate = max(1, sample_rate)
        self._cap: cv2.VideoCapture | None = None

        # Eagerly open to validate the file and extract metadata.
        self._open()
        self._video_info = self._build_video_info()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def video_info(self) -> VideoInfo:
        """Return metadata about the opened video."""
        return self._video_info

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> FrameExtractor:
        return self

    def __exit__(self, *args: object) -> None:
        self.release()

    # ------------------------------------------------------------------
    # Iterator protocol
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[tuple[int, np.ndarray]]:
        """Yield ``(frame_number, bgr_frame)`` for every sampled frame."""
        self._reset()

        frame_number = 0
        while True:
            ok, frame = self._cap.read()  # type: ignore[union-attr]
            if not ok:
                if frame_number < self._video_info.total_frames - 1:
                    logger.warning(
                        "Frame %d corrompue ou illisible, passage à la suivante",
                        frame_number,
                    )
                    frame_number += 1
                    continue
                # Genuine end of stream.
                break

            if frame_number % self._sample_rate == 0:
                yield frame_number, frame

            frame_number += 1

    # ------------------------------------------------------------------
    # Resource management
    # ------------------------------------------------------------------

    def release(self) -> None:
        """Release the underlying ``VideoCapture`` resource."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _open(self) -> None:
        """Open the video file, raising on failure."""
        cap = cv2.VideoCapture(self._path)
        if not cap.isOpened():
            cap.release()
            raise ValueError(f"Impossible d'ouvrir le fichier vidéo : {self._path}")
        self._cap = cap

    def _reset(self) -> None:
        """Reset capture to the first frame for re-iteration."""
        if self._cap is None or not self._cap.isOpened():
            self._open()
        else:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def _build_video_info(self) -> VideoInfo:
        """Extract metadata from the opened capture."""
        cap = self._cap
        assert cap is not None  # noqa: S101 — guaranteed by _open()

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec = _fourcc_to_str(fourcc_int)

        if fourcc_int == 0 and total_frames == 0:
            raise RuntimeError(f"Codec non supporté : {codec}")

        duration = total_frames / fps if fps > 0 else 0.0

        return VideoInfo(
            duration_seconds=duration,
            total_frames=total_frames,
            fps=fps,
            width=width,
            height=height,
            codec=codec,
        )
