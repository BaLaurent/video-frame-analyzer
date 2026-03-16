"""Integration tests for vfa.pipeline and vfa.cli.

Covers:
- Full pipeline execution with AnalysisPipeline and analyze_video().
- Output file creation, metadata.json structure, dry-run, no-metadata flags.
- Scene-change detection across comparator methods.
- CLI argument handling, exit codes, and edge cases.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from vfa.cli import main
from vfa.models import AnalysisResult
from vfa.pipeline import AnalysisPipeline, analyze_video


# ---------------------------------------------------------------------------
# Video helper
# ---------------------------------------------------------------------------


def _create_scene_change_video(path: str | Path, fps: float = 10.0) -> None:
    """30 frames: red(0-9), green(10-19), blue(20-29) at 10fps."""
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    out = cv2.VideoWriter(str(path), fourcc, fps, (100, 100))
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
    for i in range(30):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[:] = colors[i // 10]
        out.write(frame)
    out.release()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def scene_video(tmp_path: Path) -> str:
    """Create a scene-change video and return its path as a string."""
    video_path = tmp_path / "scene_change.avi"
    _create_scene_change_video(video_path)
    return str(video_path)


@pytest.fixture
def out_dir(tmp_path: Path) -> str:
    """Return a path to a non-existent output directory (pipeline will create it)."""
    return str(tmp_path / "pipeline_output")


# ---------------------------------------------------------------------------
# Pipeline integration tests
# ---------------------------------------------------------------------------


class TestPipelineRun:
    """Full pipeline execution tests."""

    def test_full_pipeline_returns_analysis_result(self, scene_video: str, out_dir: str) -> None:
        """Pipeline run returns AnalysisResult with frames_retained > 0."""
        pipeline = AnalysisPipeline(
            video_path=scene_video,
            output_dir=out_dir,
            quiet=True,
            force=True,
        )
        result = pipeline.run()

        assert isinstance(result, AnalysisResult)
        assert result.frames_retained > 0
        assert result.frames_analyzed > 0

    def test_output_files_created(self, scene_video: str, out_dir: str) -> None:
        """Pipeline creates frame_*.png files and metadata.json in the output directory."""
        pipeline = AnalysisPipeline(
            video_path=scene_video,
            output_dir=out_dir,
            quiet=True,
            force=True,
        )
        result = pipeline.run()

        output_path = Path(out_dir)
        frame_files = sorted(output_path.glob("frame_*.png"))
        assert len(frame_files) == result.frames_retained
        assert len(frame_files) > 0

        metadata_path = output_path / "metadata.json"
        assert metadata_path.exists()

    def test_metadata_json_valid_structure(self, scene_video: str, out_dir: str) -> None:
        """metadata.json has the expected top-level keys and nested structure."""
        pipeline = AnalysisPipeline(
            video_path=scene_video,
            output_dir=out_dir,
            quiet=True,
            force=True,
        )
        pipeline.run()

        metadata_path = Path(out_dir) / "metadata.json"
        with open(metadata_path, encoding="utf-8") as fh:
            metadata = json.load(fh)

        # Top-level keys
        assert set(metadata.keys()) == {"version", "source", "processing", "results", "frames"}
        assert metadata["version"] == "1.0"

        # Source structure
        source = metadata["source"]
        assert "file" in source
        assert "duration_seconds" in source
        assert "total_frames" in source
        assert "fps" in source
        assert "resolution" in source
        assert "codec" in source

        # Processing structure
        processing = metadata["processing"]
        assert "method" in processing
        assert "threshold" in processing
        assert "sample_rate" in processing

        # Results structure
        results = metadata["results"]
        assert "frames_analyzed" in results
        assert "frames_retained" in results
        assert "reduction_ratio" in results
        assert "reduction_percentage" in results

        # Frames list
        assert isinstance(metadata["frames"], list)
        assert len(metadata["frames"]) > 0
        frame_entry = metadata["frames"][0]
        assert "file" in frame_entry
        assert "frame_number" in frame_entry
        assert "timestamp_seconds" in frame_entry
        assert "timestamp_human" in frame_entry
        assert "difference_score" in frame_entry
        assert "reason" in frame_entry
        assert "resolution" in frame_entry

    def test_dry_run_no_files(self, scene_video: str, out_dir: str) -> None:
        """dry_run=True returns a result but writes no files to disk."""
        pipeline = AnalysisPipeline(
            video_path=scene_video,
            output_dir=out_dir,
            quiet=True,
            force=True,
            dry_run=True,
        )
        result = pipeline.run()

        assert isinstance(result, AnalysisResult)
        assert result.frames_retained > 0
        assert not Path(out_dir).exists()

    def test_no_metadata_flag(self, scene_video: str, out_dir: str) -> None:
        """no_metadata=True exports frames but skips metadata.json."""
        pipeline = AnalysisPipeline(
            video_path=scene_video,
            output_dir=out_dir,
            quiet=True,
            force=True,
            no_metadata=True,
        )
        pipeline.run()

        output_path = Path(out_dir)
        frame_files = list(output_path.glob("frame_*.png"))
        assert len(frame_files) > 0

        metadata_path = output_path / "metadata.json"
        assert not metadata_path.exists()

    def test_scene_changes_detected(self, scene_video: str, out_dir: str) -> None:
        """Low threshold on a scene-change video retains at least 3 frames (one per scene)."""
        pipeline = AnalysisPipeline(
            video_path=scene_video,
            output_dir=out_dir,
            threshold=0.1,
            quiet=True,
            force=True,
        )
        result = pipeline.run()

        assert result.frames_retained >= 3


class TestAnalyzeVideoConvenience:
    """Tests for the analyze_video() shortcut function."""

    def test_returns_valid_result(self, scene_video: str, out_dir: str) -> None:
        """analyze_video returns an AnalysisResult with frames retained."""
        result = analyze_video(
            scene_video,
            output_dir=out_dir,
            quiet=True,
            force=True,
        )
        assert isinstance(result, AnalysisResult)
        assert result.frames_retained > 0


class TestPipelineMethods:
    """Verify all comparator methods work through the pipeline."""

    @pytest.mark.parametrize("method", ["ssim", "phash", "histogram"])
    def test_method_runs_without_error(
        self, method: str, scene_video: str, tmp_path: Path
    ) -> None:
        """Each method completes and returns a valid AnalysisResult."""
        out = str(tmp_path / f"output_{method}")
        pipeline = AnalysisPipeline(
            video_path=scene_video,
            output_dir=out,
            method=method,
            quiet=True,
            force=True,
        )
        result = pipeline.run()

        assert isinstance(result, AnalysisResult)
        assert result.frames_retained > 0


class TestPipelineParameters:
    """Verify non-default parameter combinations."""

    def test_sample_rate_parameter(self, scene_video: str, out_dir: str) -> None:
        """Explicit sample_rate=2 runs without error."""
        pipeline = AnalysisPipeline(
            video_path=scene_video,
            output_dir=out_dir,
            sample_rate=2,
            quiet=True,
            force=True,
        )
        result = pipeline.run()

        assert isinstance(result, AnalysisResult)
        assert result.processing_info.sample_rate == 2

    def test_no_resize_flag(self, scene_video: str, out_dir: str) -> None:
        """no_resize=True preserves original frame dimensions (100x100)."""
        pipeline = AnalysisPipeline(
            video_path=scene_video,
            output_dir=out_dir,
            no_resize=True,
            quiet=True,
            force=True,
        )
        result = pipeline.run()

        assert result.processing_info.resize_max_dimension is None
        # Original video is 100x100, well under the default 1024px limit,
        # so frames should remain 100x100 regardless. Verify via frame metadata.
        for frame_info in result.frames:
            assert frame_info.width == 100
            assert frame_info.height == 100


# ---------------------------------------------------------------------------
# CLI tests
# ---------------------------------------------------------------------------


class TestCLI:
    """Tests for the vfa CLI entry point."""

    def test_cli_valid_args(self, scene_video: str, tmp_path: Path) -> None:
        """CLI with valid arguments succeeds without calling sys.exit."""
        out = str(tmp_path / "cli_output")
        main([scene_video, "-o", out, "-q", "--force"])

    def test_cli_invalid_threshold(self, scene_video: str) -> None:
        """CLI exits with code 1 for threshold outside 0.0-1.0."""
        with pytest.raises(SystemExit) as exc_info:
            main([scene_video, "-t", "2.0"])
        assert exc_info.value.code == 1

    def test_cli_file_not_found(self) -> None:
        """CLI exits with code 2 for a non-existent video path."""
        with pytest.raises(SystemExit) as exc_info:
            main(["nonexistent.mp4", "-q"])
        assert exc_info.value.code == 2

    def test_cli_version(self) -> None:
        """CLI --version exits with code 0."""
        with pytest.raises(SystemExit) as exc_info:
            main(["--version"])
        assert exc_info.value.code == 0

    def test_cli_dry_run(self, scene_video: str) -> None:
        """CLI --dry-run succeeds without calling sys.exit."""
        main([scene_video, "--dry-run", "-q"])
