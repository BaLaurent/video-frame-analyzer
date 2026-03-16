"""Unit tests for FrameResizer."""

import numpy as np
import pytest

from vfa.resizer import FrameResizer


# ---------------------------------------------------------------------------
# Frame helper
# ---------------------------------------------------------------------------


def _make_frame(width: int, height: int) -> np.ndarray:
    """Create a BGR frame of given dimensions."""
    return np.random.RandomState(42).randint(0, 256, (height, width, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Disabled / no-op paths
# ---------------------------------------------------------------------------


class TestResizerDisabledAndNoOp:
    """Tests where the resizer should return the frame unchanged."""

    def test_disabled_returns_unchanged(self):
        """FrameResizer(enabled=False) returns the exact same object."""
        frame = _make_frame(2000, 1500)
        resizer = FrameResizer(enabled=False)
        result = resizer.resize(frame)
        assert result is frame

    def test_small_frame_unchanged(self):
        """A 50x50 frame with max_dimension=1024 is returned unchanged."""
        frame = _make_frame(50, 50)
        resizer = FrameResizer(max_dimension=1024)
        result = resizer.resize(frame)
        assert result is frame

    def test_exact_size_unchanged(self):
        """A 1024x1024 frame at max_dimension=1024 is returned unchanged."""
        frame = _make_frame(1024, 1024)
        resizer = FrameResizer(max_dimension=1024)
        result = resizer.resize(frame)
        assert result is frame

    def test_no_upscaling(self):
        """A 500x300 frame with max_dimension=1024 is not upscaled."""
        frame = _make_frame(500, 300)
        resizer = FrameResizer(max_dimension=1024)
        result = resizer.resize(frame)
        assert result is frame


# ---------------------------------------------------------------------------
# Downscale paths
# ---------------------------------------------------------------------------


class TestResizerDownscale:
    """Tests where the resizer must shrink the frame."""

    def test_landscape_downscale(self):
        """2000x1000 with max_dimension=1024 produces 1024x512."""
        frame = _make_frame(2000, 1000)
        resizer = FrameResizer(max_dimension=1024)
        result = resizer.resize(frame)
        h, w = result.shape[:2]
        assert w == 1024
        assert h == 512

    def test_portrait_downscale(self):
        """1000x2000 with max_dimension=1024 produces 512x1024."""
        frame = _make_frame(1000, 2000)
        resizer = FrameResizer(max_dimension=1024)
        result = resizer.resize(frame)
        h, w = result.shape[:2]
        assert w == 512
        assert h == 1024

    def test_square_downscale(self):
        """2000x2000 with max_dimension=1024 produces 1024x1024."""
        frame = _make_frame(2000, 2000)
        resizer = FrameResizer(max_dimension=1024)
        result = resizer.resize(frame)
        h, w = result.shape[:2]
        assert w == 1024
        assert h == 1024

    def test_aspect_ratio_preserved(self):
        """After downscaling, the aspect ratio stays within a small tolerance."""
        width, height = 1920, 1080
        frame = _make_frame(width, height)
        resizer = FrameResizer(max_dimension=1024)
        result = resizer.resize(frame)

        original_ratio = width / height
        rh, rw = result.shape[:2]
        result_ratio = rw / rh

        assert result_ratio == pytest.approx(original_ratio, abs=0.01)


# ---------------------------------------------------------------------------
# Custom max_dimension
# ---------------------------------------------------------------------------


class TestResizerCustomMaxDimension:
    """Tests for non-default max_dimension values."""

    def test_custom_max_512(self):
        """max_dimension=512 limits the larger side to 512."""
        frame = _make_frame(2000, 1000)
        resizer = FrameResizer(max_dimension=512)
        result = resizer.resize(frame)
        h, w = result.shape[:2]
        assert w == 512
        assert h == 256

    @pytest.mark.parametrize("max_dim", [256, 512, 1024], ids=["256", "512", "1024"])
    def test_different_max_dimensions(self, max_dim):
        """Parametrize: each max_dimension caps the larger side correctly."""
        frame = _make_frame(4000, 2000)
        resizer = FrameResizer(max_dimension=max_dim)
        result = resizer.resize(frame)
        h, w = result.shape[:2]
        assert w == max_dim
        assert h == max_dim // 2


# ---------------------------------------------------------------------------
# Output type
# ---------------------------------------------------------------------------


class TestResizerOutputType:
    """Validate the output array properties."""

    def test_output_is_numpy_array_with_three_channels(self):
        """The resized result is an np.ndarray with shape (H, W, 3)."""
        frame = _make_frame(2000, 1000)
        resizer = FrameResizer(max_dimension=1024)
        result = resizer.resize(frame)

        assert isinstance(result, np.ndarray)
        assert result.ndim == 3
        assert result.shape[2] == 3

    def test_output_dtype_uint8(self):
        """The resized result preserves uint8 dtype."""
        frame = _make_frame(2000, 1000)
        resizer = FrameResizer(max_dimension=1024)
        result = resizer.resize(frame)
        assert result.dtype == np.uint8
