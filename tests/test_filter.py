"""Unit tests for FrameFilter."""

from __future__ import annotations

import numpy as np
import pytest

from vfa.comparators.base import FrameComparator
from vfa.filter import FrameFilter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class MockComparator(FrameComparator):
    """Comparator that returns a fixed score or a sequence of scores."""

    def __init__(self, scores=None, fixed_score=0.5):
        self._scores = iter(scores) if scores else None
        self._fixed = fixed_score

    def compare(self, frame_a: np.ndarray, frame_b: np.ndarray) -> float:
        if self._scores:
            return next(self._scores)
        return self._fixed

    @property
    def name(self) -> str:
        return "mock"


def _make_frames(count: int, fps: float = 10.0) -> list[tuple[int, np.ndarray]]:
    """Return list of (frame_number, frame) tuples with small dummy frames."""
    return [(i, np.zeros((10, 10, 3), dtype=np.uint8)) for i in range(count)]


def _collect(filt: FrameFilter, frames, fps: float = 10.0):
    """Run filter.process and return list of (frame_number, timestamp, score, reason)."""
    return [
        (fn, ts, score, reason)
        for fn, ts, _frame, score, reason in filt.process(frames, fps)
    ]


# ---------------------------------------------------------------------------
# 1. First frame kept
# ---------------------------------------------------------------------------


class TestFirstFrameKept:
    def test_first_frame_emitted_with_reason(self):
        filt = FrameFilter(MockComparator(), always_keep_first=True)
        results = _collect(filt, _make_frames(5))
        assert results[0][0] == 0
        assert results[0][3] == "first_frame"

    def test_first_frame_has_zero_score(self):
        filt = FrameFilter(MockComparator(), always_keep_first=True)
        results = _collect(filt, _make_frames(3))
        assert results[0][2] == 0.0


# ---------------------------------------------------------------------------
# 2. First frame NOT kept
# ---------------------------------------------------------------------------


class TestFirstFrameNotKept:
    def test_first_frame_not_emitted(self):
        filt = FrameFilter(
            MockComparator(fixed_score=0.0),
            always_keep_first=False,
            always_keep_last=False,
        )
        results = _collect(filt, _make_frames(5))
        frame_numbers = [r[0] for r in results]
        assert 0 not in frame_numbers

    def test_first_frame_still_used_as_reference(self):
        """Even when not emitted, the first frame is the reference for comparisons."""
        scores = [0.8]
        filt = FrameFilter(
            MockComparator(scores=scores),
            threshold=0.3,
            min_interval=0.0,
            always_keep_first=False,
            always_keep_last=False,
        )
        # Frame 0 not emitted; frame 1 compared against frame 0.
        frames = _make_frames(2, fps=10.0)
        results = _collect(filt, frames, fps=10.0)
        assert len(results) == 1
        assert results[0][0] == 1
        assert results[0][3] == "threshold_exceeded"


# ---------------------------------------------------------------------------
# 3. Last frame kept
# ---------------------------------------------------------------------------


class TestLastFrameKept:
    def test_last_frame_emitted_when_not_already_retained(self):
        # All scores below threshold so only first + last are emitted.
        filt = FrameFilter(
            MockComparator(fixed_score=0.0),
            threshold=0.3,
            always_keep_first=True,
            always_keep_last=True,
        )
        results = _collect(filt, _make_frames(10))
        assert results[-1][3] == "last_frame"
        assert results[-1][0] == 9

    def test_last_frame_not_duplicated_if_already_retained(self):
        """If the last frame was already retained by another rule, no duplicate."""
        # 2 frames: frame 0 = first, frame 1 already retained by threshold.
        filt = FrameFilter(
            MockComparator(fixed_score=1.0),
            threshold=0.3,
            min_interval=0.0,
            always_keep_first=True,
            always_keep_last=True,
        )
        results = _collect(filt, _make_frames(2))
        reasons = [r[3] for r in results]
        assert reasons.count("last_frame") == 0
        # Frame 1 retained as threshold_exceeded, not duplicated as last_frame.
        assert reasons == ["first_frame", "threshold_exceeded"]


# ---------------------------------------------------------------------------
# 4. Last frame NOT kept
# ---------------------------------------------------------------------------


class TestLastFrameNotKept:
    def test_last_frame_not_emitted(self):
        filt = FrameFilter(
            MockComparator(fixed_score=0.0),
            threshold=0.3,
            always_keep_first=True,
            always_keep_last=False,
        )
        results = _collect(filt, _make_frames(10))
        reasons = [r[3] for r in results]
        assert "last_frame" not in reasons


# ---------------------------------------------------------------------------
# 5. Threshold exceeded
# ---------------------------------------------------------------------------


class TestThresholdExceeded:
    def test_frame_retained_when_score_meets_threshold(self):
        filt = FrameFilter(
            MockComparator(fixed_score=0.5),
            threshold=0.5,
            min_interval=0.0,
            always_keep_first=True,
            always_keep_last=False,
        )
        results = _collect(filt, _make_frames(3))
        threshold_results = [r for r in results if r[3] == "threshold_exceeded"]
        assert len(threshold_results) > 0

    def test_frame_retained_when_score_exceeds_threshold(self):
        filt = FrameFilter(
            MockComparator(fixed_score=0.8),
            threshold=0.3,
            min_interval=0.0,
            always_keep_first=True,
            always_keep_last=False,
        )
        results = _collect(filt, _make_frames(3))
        threshold_results = [r for r in results if r[3] == "threshold_exceeded"]
        assert len(threshold_results) == 2  # frames 1 and 2


# ---------------------------------------------------------------------------
# 6. Below threshold
# ---------------------------------------------------------------------------


class TestBelowThreshold:
    def test_frame_skipped_when_below_threshold(self):
        filt = FrameFilter(
            MockComparator(fixed_score=0.1),
            threshold=0.3,
            min_interval=0.0,
            always_keep_first=True,
            always_keep_last=False,
        )
        results = _collect(filt, _make_frames(5))
        # Only first frame kept.
        assert len(results) == 1
        assert results[0][3] == "first_frame"


# ---------------------------------------------------------------------------
# 7. Min interval gate
# ---------------------------------------------------------------------------


class TestMinIntervalGate:
    def test_frames_within_min_interval_skipped_even_if_above_threshold(self):
        # fps=10 => each frame is 0.1s apart.  min_interval=0.5 => frames 1-4 skipped.
        filt = FrameFilter(
            MockComparator(fixed_score=1.0),
            threshold=0.3,
            min_interval=0.5,
            max_interval=999.0,
            always_keep_first=True,
            always_keep_last=False,
        )
        frames = _make_frames(10, fps=10.0)
        results = _collect(filt, frames, fps=10.0)
        retained_numbers = [r[0] for r in results]
        # Frame 0 at t=0.0 (first), frame 5 at t=0.5 (first to pass min_interval).
        assert 0 in retained_numbers
        assert 5 in retained_numbers
        # Frames 1-4 must not appear.
        for fn in range(1, 5):
            assert fn not in retained_numbers

    def test_min_interval_zero_allows_all_frames(self):
        filt = FrameFilter(
            MockComparator(fixed_score=1.0),
            threshold=0.3,
            min_interval=0.0,
            max_interval=999.0,
            always_keep_first=True,
            always_keep_last=False,
        )
        frames = _make_frames(5, fps=10.0)
        results = _collect(filt, frames, fps=10.0)
        # All frames retained: first_frame + 4 threshold_exceeded.
        assert len(results) == 5


# ---------------------------------------------------------------------------
# 8. Max interval forced
# ---------------------------------------------------------------------------


class TestMaxIntervalForced:
    def test_frame_retained_at_max_interval(self):
        # fps=1 => 1 frame/sec.  max_interval=3.0 => frame 3 forced.
        filt = FrameFilter(
            MockComparator(fixed_score=0.0),  # No visual change.
            threshold=0.9,
            min_interval=0.0,
            max_interval=3.0,
            always_keep_first=True,
            always_keep_last=False,
        )
        frames = _make_frames(10, fps=1.0)
        results = _collect(filt, frames, fps=1.0)
        reasons = {r[0]: r[3] for r in results}
        assert reasons[0] == "first_frame"
        assert reasons[3] == "max_interval_reached"
        assert reasons[6] == "max_interval_reached"
        assert reasons[9] == "max_interval_reached"

    def test_max_interval_reason_string(self):
        filt = FrameFilter(
            MockComparator(fixed_score=0.0),
            threshold=0.9,
            min_interval=0.0,
            max_interval=2.0,
            always_keep_first=True,
            always_keep_last=False,
        )
        frames = _make_frames(5, fps=1.0)
        results = _collect(filt, frames, fps=1.0)
        max_results = [r for r in results if r[3] == "max_interval_reached"]
        assert len(max_results) >= 1
        assert max_results[0][0] == 2  # frame 2 at t=2.0, 2.0 >= max_interval=2.0


# ---------------------------------------------------------------------------
# 9. Empty input
# ---------------------------------------------------------------------------


class TestEmptyInput:
    def test_no_frames_yields_nothing(self):
        filt = FrameFilter(MockComparator())
        results = _collect(filt, [])
        assert results == []

    def test_empty_iterator_yields_nothing(self):
        filt = FrameFilter(MockComparator())
        results = _collect(filt, iter([]))
        assert results == []


# ---------------------------------------------------------------------------
# 10. Single frame
# ---------------------------------------------------------------------------


class TestSingleFrame:
    def test_single_frame_keep_first_and_last(self):
        filt = FrameFilter(
            MockComparator(),
            always_keep_first=True,
            always_keep_last=True,
        )
        results = _collect(filt, _make_frames(1))
        # Single frame emitted as first_frame; not duplicated as last_frame
        # because last_retained_frame_number == last_frame_number.
        assert len(results) == 1
        assert results[0][3] == "first_frame"

    def test_single_frame_keep_first_only(self):
        filt = FrameFilter(
            MockComparator(),
            always_keep_first=True,
            always_keep_last=False,
        )
        results = _collect(filt, _make_frames(1))
        assert len(results) == 1
        assert results[0][3] == "first_frame"

    def test_single_frame_no_keep_first_but_keep_last(self):
        filt = FrameFilter(
            MockComparator(),
            always_keep_first=False,
            always_keep_last=True,
        )
        results = _collect(filt, _make_frames(1))
        # First frame not emitted, but it IS the last retained frame_number
        # (set on line 79 regardless of always_keep_first). So
        # last_frame_number == last_retained_frame_number => no last_frame emit.
        assert len(results) == 0

    def test_single_frame_no_keep(self):
        filt = FrameFilter(
            MockComparator(),
            always_keep_first=False,
            always_keep_last=False,
        )
        results = _collect(filt, _make_frames(1))
        assert len(results) == 0


# ---------------------------------------------------------------------------
# 11. Timestamp calculation
# ---------------------------------------------------------------------------


class TestTimestampCalculation:
    @pytest.mark.parametrize(
        "frame_number, fps, expected_ts",
        [
            (0, 10.0, 0.0),
            (5, 10.0, 0.5),
            (10, 10.0, 1.0),
            (15, 30.0, 0.5),
            (100, 25.0, 4.0),
        ],
    )
    def test_timestamp_equals_frame_number_over_fps(self, frame_number, fps, expected_ts):
        filt = FrameFilter(
            MockComparator(),
            always_keep_first=True,
            always_keep_last=False,
        )
        single_frame = [(frame_number, np.zeros((10, 10, 3), dtype=np.uint8))]
        results = _collect(filt, single_frame, fps=fps)
        assert len(results) == 1
        assert results[0][1] == pytest.approx(expected_ts)

    def test_last_frame_timestamp_correct(self):
        filt = FrameFilter(
            MockComparator(fixed_score=0.0),
            threshold=0.9,
            always_keep_first=True,
            always_keep_last=True,
        )
        frames = _make_frames(20, fps=10.0)
        results = _collect(filt, frames, fps=10.0)
        last = results[-1]
        assert last[0] == 19
        assert last[1] == pytest.approx(19 / 10.0)


# ---------------------------------------------------------------------------
# 12. All identical frames
# ---------------------------------------------------------------------------


class TestAllIdenticalFrames:
    def test_only_first_and_last_kept(self):
        filt = FrameFilter(
            MockComparator(fixed_score=0.0),
            threshold=0.3,
            min_interval=0.0,
            max_interval=999.0,
            always_keep_first=True,
            always_keep_last=True,
        )
        results = _collect(filt, _make_frames(20))
        assert len(results) == 2
        assert results[0][3] == "first_frame"
        assert results[1][3] == "last_frame"

    def test_identical_with_max_interval(self):
        """Identical frames still trigger max_interval retention."""
        filt = FrameFilter(
            MockComparator(fixed_score=0.0),
            threshold=0.3,
            min_interval=0.0,
            max_interval=5.0,
            always_keep_first=True,
            always_keep_last=True,
        )
        # 100 frames at fps=10 => 10 seconds total.
        frames = _make_frames(100, fps=10.0)
        results = _collect(filt, frames, fps=10.0)
        reasons = [r[3] for r in results]
        assert "first_frame" in reasons
        assert "max_interval_reached" in reasons
        assert "last_frame" in reasons


# ---------------------------------------------------------------------------
# 13. All different frames
# ---------------------------------------------------------------------------


class TestAllDifferentFrames:
    def test_all_frames_after_min_interval_kept(self):
        filt = FrameFilter(
            MockComparator(fixed_score=1.0),
            threshold=0.3,
            min_interval=0.0,
            max_interval=999.0,
            always_keep_first=True,
            always_keep_last=False,
        )
        frames = _make_frames(10, fps=10.0)
        results = _collect(filt, frames, fps=10.0)
        assert len(results) == 10
        assert results[0][3] == "first_frame"
        for r in results[1:]:
            assert r[3] == "threshold_exceeded"

    def test_all_different_with_min_interval(self):
        """High-difference frames still gated by min_interval."""
        filt = FrameFilter(
            MockComparator(fixed_score=1.0),
            threshold=0.3,
            min_interval=0.5,
            max_interval=999.0,
            always_keep_first=True,
            always_keep_last=False,
        )
        # fps=10 => 0.1s/frame. Only every 5th frame passes min_interval.
        frames = _make_frames(20, fps=10.0)
        results = _collect(filt, frames, fps=10.0)
        retained_numbers = [r[0] for r in results]
        assert retained_numbers == [0, 5, 10, 15]


# ---------------------------------------------------------------------------
# Additional edge cases
# ---------------------------------------------------------------------------


class TestSequenceOfScores:
    def test_mixed_scores_filter_correctly(self):
        """Verify filter uses per-frame scores from the comparator."""
        # Frame indices:         0(first)  1     2     3     4
        # Scores (vs reference):           0.1   0.8   0.1   0.9
        scores = [0.1, 0.8, 0.1, 0.9]
        filt = FrameFilter(
            MockComparator(scores=scores),
            threshold=0.3,
            min_interval=0.0,
            max_interval=999.0,
            always_keep_first=True,
            always_keep_last=False,
        )
        results = _collect(filt, _make_frames(5))
        retained = {r[0]: (r[2], r[3]) for r in results}
        assert 0 in retained  # first_frame
        assert 1 not in retained  # score 0.1 < 0.3
        assert 2 in retained  # score 0.8 >= 0.3
        assert retained[2][1] == "threshold_exceeded"
        # After frame 2 becomes the new reference:
        assert 3 not in retained  # score 0.1 < 0.3
        assert 4 in retained  # score 0.9 >= 0.3


class TestProcessReturnsFrameData:
    def test_yielded_frame_is_original_array(self):
        """The yielded frame ndarray should be the same object passed in."""
        original_frames = _make_frames(3)
        filt = FrameFilter(
            MockComparator(fixed_score=1.0),
            threshold=0.3,
            min_interval=0.0,
            always_keep_first=True,
            always_keep_last=False,
        )
        results = list(filt.process(original_frames, fps=10.0))
        for fn, _ts, frame, _score, _reason in results:
            assert frame is original_frames[fn][1]


class TestIterableInput:
    def test_generator_input(self):
        """process() accepts any iterable, not just lists."""
        def gen():
            for i in range(5):
                yield (i, np.zeros((10, 10, 3), dtype=np.uint8))

        filt = FrameFilter(
            MockComparator(fixed_score=0.0),
            always_keep_first=True,
            always_keep_last=True,
        )
        results = _collect(filt, gen())
        assert len(results) == 2
        assert results[0][3] == "first_frame"
        assert results[1][3] == "last_frame"
