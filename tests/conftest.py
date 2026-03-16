"""Shared pytest fixtures for video-frame-analyzer tests.

Provides:
- In-memory BGR frame fixtures (numpy arrays) for comparator/resizer/filter tests.
- Synthetic video file fixtures (written to tmp_path) for extractor/pipeline tests.
- A clean output directory fixture for exporter tests.
"""

import cv2
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Frame fixtures (pure numpy, no disk I/O)
# ---------------------------------------------------------------------------


@pytest.fixture
def solid_red_frame():
    """100x100 BGR frame, solid red."""
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[:, :, 2] = 255  # BGR: red channel
    return frame


@pytest.fixture
def solid_blue_frame():
    """100x100 BGR frame, solid blue."""
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[:, :, 0] = 255  # BGR: blue channel
    return frame


@pytest.fixture
def solid_green_frame():
    """100x100 BGR frame, solid green."""
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[:, :, 1] = 255  # BGR: green channel
    return frame


@pytest.fixture
def gradient_frame():
    """100x100 BGR frame with horizontal gradient (black to white)."""
    gray = np.tile(np.linspace(0, 255, 100, dtype=np.uint8), (100, 1))
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


@pytest.fixture
def noisy_red_frame(solid_red_frame):
    """Red frame with slight random noise (should be 'similar' to solid_red)."""
    rng = np.random.RandomState(42)
    noise = rng.randint(0, 10, solid_red_frame.shape, dtype=np.uint8)
    return cv2.add(solid_red_frame, noise)


@pytest.fixture
def large_frame():
    """2000x1500 BGR frame for resizer tests."""
    rng = np.random.RandomState(42)
    return rng.randint(0, 256, (1500, 2000, 3), dtype=np.uint8)


@pytest.fixture
def small_frame():
    """50x50 BGR frame for resizer tests (should NOT be resized)."""
    rng = np.random.RandomState(42)
    return rng.randint(0, 256, (50, 50, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Video file fixtures (written to pytest tmp_path)
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_video(tmp_path):
    """Create a small .avi video file with 30 frames at 10 fps.

    Frame pattern:
    - Frames  0-9:  solid red   (BGR: 0, 0, 255)
    - Frames 10-19: solid green (BGR: 0, 255, 0)  -- scene change
    - Frames 20-29: solid blue  (BGR: 255, 0, 0)  -- scene change

    Returns the path (str) to the video file.
    """
    video_path = str(tmp_path / "test_video.avi")
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    out = cv2.VideoWriter(video_path, fourcc, 10.0, (100, 100))

    colors = [
        (0, 0, 255),  # red   (BGR) for frames 0-9
        (0, 255, 0),  # green (BGR) for frames 10-19
        (255, 0, 0),  # blue  (BGR) for frames 20-29
    ]

    for i in range(30):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[:] = colors[i // 10]
        out.write(frame)

    out.release()
    return video_path


@pytest.fixture
def gradual_change_video(tmp_path):
    """Create a video where brightness gradually increases across 20 frames at 10 fps.

    Useful for testing threshold sensitivity — no single pair of adjacent frames
    has a dramatic difference, but first vs. last are very different.

    Returns the path (str) to the video file.
    """
    video_path = str(tmp_path / "gradual_video.avi")
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    out = cv2.VideoWriter(video_path, fourcc, 10.0, (100, 100))

    for i in range(20):
        brightness = int(255 * i / 19)
        frame = np.full((100, 100, 3), brightness, dtype=np.uint8)
        out.write(frame)

    out.release()
    return video_path


@pytest.fixture
def single_frame_video(tmp_path):
    """Create a 1-frame video. Edge case testing.

    Returns the path (str) to the video file.
    """
    video_path = str(tmp_path / "single_frame.avi")
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    out = cv2.VideoWriter(video_path, fourcc, 10.0, (100, 100))

    frame = np.full((100, 100, 3), 128, dtype=np.uint8)
    out.write(frame)

    out.release()
    return video_path


# ---------------------------------------------------------------------------
# Output directory fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def output_dir(tmp_path):
    """Provide a clean temporary output directory for exporter tests."""
    d = tmp_path / "output"
    d.mkdir()
    return str(d)
