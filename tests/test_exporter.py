"""Unit tests for vfa.exporter.FrameExporter.

Covers directory setup (creation, nesting, conflict handling, force mode),
frame export (PNG/JPEG, filename format, disk presence), and metadata
serialization (structure, keys, frame entries).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from vfa.exporter import FrameExporter
from vfa.models import AnalysisResult, FrameInfo, ProcessingInfo, VideoInfo


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _make_result(output_dir: str, num_frames: int = 2) -> AnalysisResult:
    """Build a minimal but complete AnalysisResult for metadata tests."""
    video_info = VideoInfo(
        duration_seconds=3.0,
        total_frames=30,
        fps=10.0,
        width=100,
        height=100,
        codec="MJPG",
    )
    proc_info = ProcessingInfo(
        method="ssim",
        threshold=0.3,
        min_interval_seconds=0.5,
        max_interval_seconds=30.0,
        resize_max_dimension=1024,
        output_format="png",
        sample_rate=1,
        processing_time_seconds=1.5,
    )
    frames = [
        FrameInfo(
            file=f"frame_{i:06d}.png",
            frame_number=i,
            timestamp_seconds=i / 10.0,
            timestamp_human=f"00:00:0{i}.000",
            difference_score=0.0 if i == 0 else 0.5,
            reason="first_frame" if i == 0 else "threshold_exceeded",
            width=100,
            height=100,
        )
        for i in range(num_frames)
    ]
    return AnalysisResult(
        source_path="/tmp/video.mp4",
        output_dir=output_dir,
        video_info=video_info,
        processing_info=proc_info,
        frames_analyzed=30,
        frames_retained=num_frames,
        reduction_ratio=0.93,
        reduction_percentage="93.33%",
        frames=frames,
    )


def _solid_frame(b: int = 0, g: int = 0, r: int = 255) -> np.ndarray:
    """Return a 100x100 solid-colour BGR frame."""
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[:, :] = (b, g, r)
    return frame


# ---------------------------------------------------------------------------
# setup() tests
# ---------------------------------------------------------------------------


class TestSetup:
    """Directory creation and conflict detection."""

    def test_creates_nonexistent_directory(self, tmp_path):
        """setup() creates the output directory when it does not exist."""
        target = tmp_path / "new_dir"
        exporter = FrameExporter(str(target))

        exporter.setup()

        assert target.is_dir()

    def test_creates_nested_path(self, tmp_path):
        """setup() creates intermediate parent directories."""
        target = tmp_path / "a" / "b" / "c"
        exporter = FrameExporter(str(target))

        exporter.setup()

        assert target.is_dir()

    def test_raises_file_exists_error_when_dir_has_files(self, tmp_path):
        """setup() raises FileExistsError when the directory contains files and force=False."""
        target = tmp_path / "occupied"
        target.mkdir()
        (target / "existing.txt").write_text("data")

        exporter = FrameExporter(str(target), force=False)

        with pytest.raises(FileExistsError, match="already contains"):
            exporter.setup()

    def test_force_allows_existing_files(self, tmp_path):
        """setup() with force=True does not raise even when directory has files."""
        target = tmp_path / "occupied"
        target.mkdir()
        (target / "existing.txt").write_text("data")

        exporter = FrameExporter(str(target), force=True)

        exporter.setup()  # should not raise

    def test_empty_existing_dir_does_not_raise(self, tmp_path):
        """setup() succeeds on an existing empty directory without force."""
        target = tmp_path / "empty"
        target.mkdir()

        exporter = FrameExporter(str(target), force=False)

        exporter.setup()  # should not raise


# ---------------------------------------------------------------------------
# export_frame() tests
# ---------------------------------------------------------------------------


class TestExportFrame:
    """Frame writing and filename generation."""

    def test_export_png_returns_correct_filename(self, tmp_path):
        """export_frame() returns 'frame_000042.png' for frame_number=42."""
        exporter = FrameExporter(str(tmp_path), output_format="png")
        frame = _solid_frame()

        filename = exporter.export_frame(frame, 42)

        assert filename == "frame_000042.png"

    def test_export_jpeg_returns_correct_filename(self, tmp_path):
        """export_frame() returns a .jpeg filename when output_format is 'jpeg'."""
        exporter = FrameExporter(str(tmp_path), output_format="jpeg")
        frame = _solid_frame()

        filename = exporter.export_frame(frame, 7)

        assert filename == "frame_000007.jpeg"

    def test_filename_zero_padded_to_six_digits(self, tmp_path):
        """Frame numbers are zero-padded to exactly 6 digits."""
        exporter = FrameExporter(str(tmp_path), output_format="png")
        frame = _solid_frame()

        for frame_number, expected in [
            (0, "frame_000000.png"),
            (1, "frame_000001.png"),
            (999, "frame_000999.png"),
            (123456, "frame_123456.png"),
        ]:
            filename = exporter.export_frame(frame, frame_number)
            assert filename == expected

    def test_exported_file_exists_on_disk(self, tmp_path):
        """The exported file is actually written to disk."""
        exporter = FrameExporter(str(tmp_path), output_format="png")
        frame = _solid_frame()

        filename = exporter.export_frame(frame, 1)
        path = tmp_path / filename

        assert path.exists()
        assert path.stat().st_size > 0


# ---------------------------------------------------------------------------
# write_metadata() tests
# ---------------------------------------------------------------------------


class TestWriteMetadata:
    """Metadata JSON serialization."""

    def test_creates_metadata_json(self, tmp_path):
        """write_metadata() creates a valid metadata.json file."""
        exporter = FrameExporter(str(tmp_path))
        result = _make_result(str(tmp_path))

        exporter.write_metadata(result)

        meta_path = tmp_path / "metadata.json"
        assert meta_path.exists()
        # Verify it is valid JSON by parsing it
        data = json.loads(meta_path.read_text(encoding="utf-8"))
        assert isinstance(data, dict)

    def test_metadata_top_level_keys(self, tmp_path):
        """metadata.json contains the expected top-level keys."""
        exporter = FrameExporter(str(tmp_path))
        result = _make_result(str(tmp_path))

        exporter.write_metadata(result)

        data = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
        expected_keys = {"version", "source", "processing", "results", "frames"}
        assert set(data.keys()) == expected_keys

    def test_metadata_version(self, tmp_path):
        """metadata.json includes a version field set to '1.0'."""
        exporter = FrameExporter(str(tmp_path))
        result = _make_result(str(tmp_path))

        exporter.write_metadata(result)

        data = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
        assert data["version"] == "1.0"

    def test_metadata_source_section(self, tmp_path):
        """The 'source' section contains video file and resolution info."""
        exporter = FrameExporter(str(tmp_path))
        result = _make_result(str(tmp_path))

        exporter.write_metadata(result)

        data = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
        source = data["source"]
        assert source["file"] == "/tmp/video.mp4"
        assert source["fps"] == 10.0
        assert source["total_frames"] == 30
        assert source["resolution"] == {"width": 100, "height": 100}

    def test_metadata_results_section(self, tmp_path):
        """The 'results' section reflects the AnalysisResult fields."""
        exporter = FrameExporter(str(tmp_path))
        result = _make_result(str(tmp_path), num_frames=3)

        exporter.write_metadata(result)

        data = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
        results = data["results"]
        assert results["frames_analyzed"] == 30
        assert results["frames_retained"] == 3
        assert results["reduction_ratio"] == pytest.approx(0.93)
        assert results["reduction_percentage"] == "93.33%"

    def test_metadata_frame_entries(self, tmp_path):
        """Each frame entry in metadata.json has the expected fields."""
        exporter = FrameExporter(str(tmp_path))
        result = _make_result(str(tmp_path), num_frames=2)

        exporter.write_metadata(result)

        data = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
        frames = data["frames"]

        assert len(frames) == 2

        expected_frame_keys = {
            "file",
            "frame_number",
            "timestamp_seconds",
            "timestamp_human",
            "difference_score",
            "reason",
            "resolution",
        }
        for entry in frames:
            assert set(entry.keys()) == expected_frame_keys

        # Verify first frame values
        first = frames[0]
        assert first["file"] == "frame_000000.png"
        assert first["frame_number"] == 0
        assert first["timestamp_seconds"] == 0.0
        assert first["difference_score"] == 0.0
        assert first["reason"] == "first_frame"
        assert first["resolution"] == {"width": 100, "height": 100}

        # Verify second frame values
        second = frames[1]
        assert second["file"] == "frame_000001.png"
        assert second["frame_number"] == 1
        assert second["difference_score"] == 0.5
        assert second["reason"] == "threshold_exceeded"
