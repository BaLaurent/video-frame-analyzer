"""Unit tests for vfa.extractor.FrameExtractor.

Covers constructor validation, metadata extraction, iteration with various
sample rates, context-manager semantics, re-iteration, release idempotency,
and the single-frame edge case.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from vfa.extractor import FrameExtractor


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _create_video(
    path,
    num_frames: int = 30,
    fps: float = 10.0,
    width: int = 100,
    height: int = 100,
    color: tuple[int, int, int] = (0, 0, 255),
) -> None:
    """Write a synthetic MJPG .avi video filled with solid-colour frames."""
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    out = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    for _ in range(num_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:] = color
        out.write(frame)
    out.release()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFrameExtractorInit:
    """Constructor validation."""

    def test_file_not_found_raises(self, tmp_path):
        """Non-existent path raises FileNotFoundError."""
        bogus = tmp_path / "does_not_exist.avi"
        with pytest.raises(FileNotFoundError):
            FrameExtractor(str(bogus))


class TestVideoInfo:
    """Metadata exposed by the video_info property."""

    def test_metadata_matches_written_video(self, tmp_path):
        """fps, total_frames, width, and height match the values used to create the video."""
        video_path = tmp_path / "meta.avi"
        _create_video(video_path, num_frames=30, fps=10.0, width=100, height=100)

        ext = FrameExtractor(str(video_path))
        try:
            info = ext.video_info
            assert info.total_frames == 30
            assert info.fps == pytest.approx(10.0, abs=0.5)
            assert info.width == 100
            assert info.height == 100
        finally:
            ext.release()


class TestIteration:
    """Iteration behaviour with different sample rates."""

    def test_yields_all_frames_sample_rate_1(self, tmp_path):
        """A 30-frame video with sample_rate=1 yields exactly 30 frames."""
        video_path = tmp_path / "sr1.avi"
        _create_video(video_path, num_frames=30)

        ext = FrameExtractor(str(video_path), sample_rate=1)
        try:
            frames = list(ext)
            assert len(frames) == 30
        finally:
            ext.release()

    def test_sample_rate_2(self, tmp_path):
        """A 30-frame video with sample_rate=2 yields 15 frames."""
        video_path = tmp_path / "sr2.avi"
        _create_video(video_path, num_frames=30)

        ext = FrameExtractor(str(video_path), sample_rate=2)
        try:
            frames = list(ext)
            assert len(frames) == 15
        finally:
            ext.release()

    def test_sample_rate_3(self, tmp_path):
        """A 30-frame video with sample_rate=3 yields 10 frames."""
        video_path = tmp_path / "sr3.avi"
        _create_video(video_path, num_frames=30)

        ext = FrameExtractor(str(video_path), sample_rate=3)
        try:
            frames = list(ext)
            assert len(frames) == 10
        finally:
            ext.release()

    def test_frame_shape(self, tmp_path):
        """Each yielded frame has shape (height, width, 3)."""
        width, height = 120, 80
        video_path = tmp_path / "shape.avi"
        _create_video(video_path, num_frames=5, width=width, height=height)

        ext = FrameExtractor(str(video_path))
        try:
            for _frame_num, frame in ext:
                assert frame.shape == (height, width, 3)
        finally:
            ext.release()

    def test_frame_numbers_with_sample_rate_2(self, tmp_path):
        """With sample_rate=2 the yielded frame numbers are 0, 2, 4, ..."""
        video_path = tmp_path / "fnums.avi"
        _create_video(video_path, num_frames=30)

        ext = FrameExtractor(str(video_path), sample_rate=2)
        try:
            frame_numbers = [num for num, _ in ext]
            expected = list(range(0, 30, 2))
            assert frame_numbers == expected
        finally:
            ext.release()


class TestContextManager:
    """Context-manager protocol."""

    def test_with_statement_works(self, tmp_path):
        """FrameExtractor can be used as a context manager and yields frames."""
        video_path = tmp_path / "ctx.avi"
        _create_video(video_path, num_frames=10)

        with FrameExtractor(str(video_path)) as ext:
            frames = list(ext)
            assert len(frames) == 10

    def test_release_called_on_exit(self, tmp_path):
        """After exiting the with block the internal capture is released."""
        video_path = tmp_path / "ctx_rel.avi"
        _create_video(video_path, num_frames=5)

        with FrameExtractor(str(video_path)) as ext:
            pass  # just enter and exit

        # After __exit__, the internal _cap should be None (released).
        assert ext._cap is None


class TestReIteration:
    """Re-iteration over the same extractor instance."""

    def test_iterate_twice(self, tmp_path):
        """Iterating twice over the same extractor yields the same frames both times."""
        video_path = tmp_path / "reiter.avi"
        _create_video(video_path, num_frames=20)

        ext = FrameExtractor(str(video_path))
        try:
            first_pass = [(n, f.copy()) for n, f in ext]
            second_pass = [(n, f.copy()) for n, f in ext]

            assert len(first_pass) == len(second_pass) == 20

            for (n1, _), (n2, _) in zip(first_pass, second_pass):
                assert n1 == n2
        finally:
            ext.release()


class TestRelease:
    """Resource release semantics."""

    def test_release_multiple_times(self, tmp_path):
        """Calling release() more than once does not raise."""
        video_path = tmp_path / "multi_rel.avi"
        _create_video(video_path, num_frames=5)

        ext = FrameExtractor(str(video_path))
        ext.release()
        ext.release()  # should be a no-op, not an error


class TestSingleFrameVideo:
    """Edge case: video with exactly one frame."""

    def test_single_frame(self, tmp_path):
        """A 1-frame video yields exactly one frame with frame_number 0."""
        video_path = tmp_path / "one.avi"
        _create_video(video_path, num_frames=1)

        ext = FrameExtractor(str(video_path))
        try:
            frames = list(ext)
            assert len(frames) == 1
            frame_number, frame = frames[0]
            assert frame_number == 0
            assert frame.shape == (100, 100, 3)
        finally:
            ext.release()
