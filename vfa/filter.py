"""Frame filtering logic based on visual difference and temporal intervals."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import numpy as np

from vfa.comparators.base import FrameComparator


class FrameFilter:
    """Filters video frames by visual difference and temporal constraints.

    Frames are retained when they exceed a visual difference threshold compared
    to the last retained reference frame, subject to minimum and maximum time
    interval constraints.

    Yields tuples of ``(frame_number, timestamp, frame, difference_score, reason)``
    where *reason* is one of:

    - ``"first_frame"`` -- the very first frame (if ``always_keep_first`` is True)
    - ``"last_frame"`` -- the very last frame (if ``always_keep_last`` is True)
    - ``"threshold_exceeded"`` -- visual difference exceeded the threshold
    - ``"max_interval_reached"`` -- maximum time gap since last retained frame
    """

    def __init__(
        self,
        comparator: FrameComparator,
        threshold: float = 0.3,
        min_interval: float = 0.5,
        max_interval: float = 30.0,
        always_keep_first: bool = True,
        always_keep_last: bool = True,
    ) -> None:
        self.comparator = comparator
        self.threshold = threshold
        self.min_interval = min_interval
        self.max_interval = max_interval
        self.always_keep_first = always_keep_first
        self.always_keep_last = always_keep_last

    def process(
        self, frames: Iterable[tuple[int, np.ndarray]], fps: float
    ) -> Iterator[tuple[int, float, np.ndarray, float, str]]:
        """Filter frames based on visual difference and temporal constraints.

        Args:
            frames: Iterable of ``(frame_number, frame_ndarray)`` tuples,
                typically produced by a ``FrameExtractor``.
            fps: Frames per second of the source video, used to convert frame
                numbers into timestamps.

        Yields:
            Tuples of ``(frame_number, timestamp, frame, difference_score, reason)``.
        """
        reference_frame: np.ndarray | None = None
        last_retained_timestamp: float = 0.0
        last_retained_frame_number: int | None = None

        # Buffer for always_keep_last: tracks the most recent frame so we can
        # emit it at the end if it wasn't already retained.
        last_frame_number: int | None = None
        last_frame: np.ndarray | None = None
        is_first = True

        for frame_number, frame in frames:
            timestamp = frame_number / fps

            # Always track the latest frame for possible last-frame emission.
            last_frame_number = frame_number
            last_frame = frame

            if is_first:
                is_first = False
                reference_frame = frame
                last_retained_timestamp = timestamp
                last_retained_frame_number = frame_number
                if self.always_keep_first:
                    yield (frame_number, timestamp, frame, 0.0, "first_frame")
                continue

            time_since_last = timestamp - last_retained_timestamp

            # Max-interval temporal guard: force retention if too much time passed.
            if time_since_last >= self.max_interval:
                score = self.comparator.compare(reference_frame, frame)
                reference_frame = frame
                last_retained_timestamp = timestamp
                last_retained_frame_number = frame_number
                yield (frame_number, timestamp, frame, score, "max_interval_reached")
                continue

            # Min-interval gate: skip frames that are too close in time.
            if time_since_last < self.min_interval:
                continue

            # Visual difference check.
            score = self.comparator.compare(reference_frame, frame)
            if score >= self.threshold:
                reference_frame = frame
                last_retained_timestamp = timestamp
                last_retained_frame_number = frame_number
                yield (frame_number, timestamp, frame, score, "threshold_exceeded")

        # Emit the last frame if requested and it wasn't already retained.
        if (
            self.always_keep_last
            and last_frame_number is not None
            and last_frame_number != last_retained_frame_number
        ):
            timestamp = last_frame_number / fps
            score = (
                self.comparator.compare(reference_frame, last_frame)
                if reference_frame is not None
                else 0.0
            )
            yield (last_frame_number, timestamp, last_frame, score, "last_frame")
