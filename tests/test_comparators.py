"""Unit tests for frame comparators."""

import numpy as np
import pytest

from vfa.comparators import get_comparator
from vfa.comparators.histogram import HistogramComparator
from vfa.comparators.phash import PHashComparator
from vfa.comparators.ssim import SSIMComparator


# ---------------------------------------------------------------------------
# Frame helpers
# ---------------------------------------------------------------------------


def _make_solid_frame(b: int, g: int, r: int, size: int = 100) -> np.ndarray:
    """Create a solid-colour BGR frame."""
    frame = np.zeros((size, size, 3), dtype=np.uint8)
    frame[:, :] = (b, g, r)
    return frame


def _make_noisy_frame(base_frame: np.ndarray, seed: int = 42) -> np.ndarray:
    """Add low-amplitude noise to *base_frame*."""
    rng = np.random.RandomState(seed)
    noise = rng.randint(0, 10, base_frame.shape, dtype=np.uint8)
    return np.clip(base_frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def _make_gradient_frame(size: int = 100) -> np.ndarray:
    """Create a BGR frame with a vertical blue-to-red gradient."""
    frame = np.zeros((size, size, 3), dtype=np.uint8)
    for i in range(size):
        frame[i, :] = (int(i * 255 / (size - 1)), 0, int(255 - i * 255 / (size - 1)))
    return frame


def _make_checkerboard_frame(size: int = 100, block: int = 10) -> np.ndarray:
    """Create a black-and-white checkerboard pattern."""
    frame = np.zeros((size, size, 3), dtype=np.uint8)
    for y in range(size):
        for x in range(size):
            if (y // block + x // block) % 2 == 0:
                frame[y, x] = (255, 255, 255)
    return frame


def _make_random_frame(seed: int, size: int = 100) -> np.ndarray:
    """Create a pseudo-random BGR frame (deterministic via seed)."""
    rng = np.random.RandomState(seed)
    return rng.randint(0, 256, (size, size, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Shared test frames
# ---------------------------------------------------------------------------

RED_FRAME = _make_solid_frame(0, 0, 255)
BLUE_FRAME = _make_solid_frame(255, 0, 0)
BLACK_FRAME = _make_solid_frame(0, 0, 0)
WHITE_FRAME = _make_solid_frame(255, 255, 255)
NOISY_RED_FRAME = _make_noisy_frame(RED_FRAME)
GRADIENT_FRAME = _make_gradient_frame()
CHECKERBOARD_FRAME = _make_checkerboard_frame()
RANDOM_FRAME_A = _make_random_frame(seed=0)
RANDOM_FRAME_B = _make_random_frame(seed=18)


# ---------------------------------------------------------------------------
# Per-comparator tests — identical frames
# ---------------------------------------------------------------------------

ALL_COMPARATORS = [
    pytest.param(SSIMComparator(), id="ssim"),
    pytest.param(PHashComparator(), id="phash"),
    pytest.param(HistogramComparator(), id="histogram"),
]


@pytest.mark.parametrize("comparator", ALL_COMPARATORS)
class TestComparatorContract:
    """Tests that every comparator fulfils the base-class contract."""

    def test_identical_frames_score_near_zero(self, comparator):
        """Comparing a frame to itself must return ~0.0."""
        score = comparator.compare(RED_FRAME, RED_FRAME)
        assert score == pytest.approx(0.0, abs=0.01)

    def test_score_always_in_valid_range(self, comparator):
        """Score must always be in [0.0, 1.0] for every frame pair."""
        pairs = [
            (RED_FRAME, RED_FRAME),
            (RED_FRAME, BLUE_FRAME),
            (RED_FRAME, NOISY_RED_FRAME),
            (BLACK_FRAME, WHITE_FRAME),
            (GRADIENT_FRAME, CHECKERBOARD_FRAME),
        ]
        for frame_a, frame_b in pairs:
            score = comparator.compare(frame_a, frame_b)
            assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# SSIM-specific tests
#
# SSIM operates on grayscale intensity and spatial structure.
# Solid red vs solid blue map to the same gray value, so they look alike.
# Black vs white maximises the intensity difference.
# ---------------------------------------------------------------------------


class TestSSIMComparator:
    """SSIM-specific behavioural tests."""

    def test_completely_different_frames(self):
        """Black vs white must produce a high difference score (> 0.5)."""
        comp = SSIMComparator()
        score = comp.compare(BLACK_FRAME, WHITE_FRAME)
        assert score > 0.5

    def test_similar_frames(self):
        """Red vs noisy-red must produce a low difference score (< 0.3)."""
        comp = SSIMComparator()
        score = comp.compare(RED_FRAME, NOISY_RED_FRAME)
        assert score < 0.3

    def test_name(self):
        assert SSIMComparator().name == "ssim"


# ---------------------------------------------------------------------------
# PHash-specific tests
#
# Perceptual hashing captures spatial frequency. Solid frames of any colour
# hash identically (no structure). Frames with distinct spatial patterns
# (e.g. checkerboard vs solid) produce meaningful distance.
# ---------------------------------------------------------------------------


class TestPHashComparator:
    """PHash-specific behavioural tests."""

    def test_completely_different_frames(self):
        """Two random-noise frames with distinct structure must score > 0.5."""
        comp = PHashComparator()
        score = comp.compare(RANDOM_FRAME_A, RANDOM_FRAME_B)
        assert score > 0.5

    def test_similar_frames(self):
        """Red vs red (identical) is 0.0 — trivially similar."""
        comp = PHashComparator()
        # Two structurally identical solid frames
        red2 = _make_solid_frame(0, 0, 200)
        score = comp.compare(RED_FRAME, red2)
        assert score < 0.3

    def test_name(self):
        assert PHashComparator().name == "phash"


# ---------------------------------------------------------------------------
# Histogram-specific tests
#
# Histogram comparison operates on HSV colour distributions. Solid frames
# with different hues produce completely uncorrelated histograms.
# ---------------------------------------------------------------------------


class TestHistogramComparator:
    """Histogram-specific behavioural tests."""

    def test_completely_different_frames(self):
        """Red vs blue must produce a high difference score (> 0.5)."""
        comp = HistogramComparator()
        score = comp.compare(RED_FRAME, BLUE_FRAME)
        assert score > 0.5

    def test_similar_frames(self):
        """Red vs noisy-red must produce a low difference score (< 0.3)."""
        comp = HistogramComparator()
        score = comp.compare(RED_FRAME, NOISY_RED_FRAME)
        assert score < 0.3

    def test_name(self):
        assert HistogramComparator().name == "histogram"


# ---------------------------------------------------------------------------
# Name property — parametrized across all comparators
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "comparator, expected_name",
    [
        pytest.param(SSIMComparator(), "ssim", id="ssim"),
        pytest.param(PHashComparator(), "phash", id="phash"),
        pytest.param(HistogramComparator(), "histogram", id="histogram"),
    ],
)
def test_name_property(comparator, expected_name):
    """Each comparator exposes the correct algorithm name."""
    assert comparator.name == expected_name


# ---------------------------------------------------------------------------
# Factory function tests
# ---------------------------------------------------------------------------


class TestGetComparator:
    """Tests for the ``get_comparator`` factory function."""

    def test_returns_ssim_by_default(self):
        assert isinstance(get_comparator(), SSIMComparator)

    def test_returns_ssim(self):
        assert isinstance(get_comparator("ssim"), SSIMComparator)

    def test_returns_phash(self):
        assert isinstance(get_comparator("phash"), PHashComparator)

    def test_returns_histogram(self):
        assert isinstance(get_comparator("histogram"), HistogramComparator)

    def test_unknown_method_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown comparison method"):
            get_comparator("unknown")

    def test_kwargs_forwarded_to_ssim(self):
        comp = get_comparator("ssim", resize_to=128)
        assert isinstance(comp, SSIMComparator)
        assert comp._resize_to == 128

    def test_kwargs_forwarded_to_phash(self):
        comp = get_comparator("phash", hash_size=16)
        assert isinstance(comp, PHashComparator)
        assert comp._hash_size == 16
        assert comp._hash_bits == 16 ** 2
