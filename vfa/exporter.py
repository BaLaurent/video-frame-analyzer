"""Frame exporter for writing extracted frames to disk and generating metadata."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from vfa.models import AnalysisResult, FrameInfo


class FrameExporter:
    """Writes extracted video frames to disk and produces metadata.json."""

    def __init__(
        self,
        output_dir: str,
        output_format: str = "png",
        jpeg_quality: int = 85,
        force: bool = False,
    ) -> None:
        self._output_dir = Path(output_dir)
        self._output_format = output_format
        self._jpeg_quality = jpeg_quality
        self._force = force

    def setup(self) -> None:
        """Create the output directory and check for conflicts.

        Raises:
            FileExistsError: If the output directory already contains files
                and ``force`` is False.
        """
        if self._output_dir.exists():
            has_files = any(self._output_dir.iterdir())
            if has_files and not self._force:
                raise FileExistsError(
                    f"Output directory '{self._output_dir}' already contains "
                    "files. Use --force to overwrite."
                )
        else:
            self._output_dir.mkdir(parents=True)

    def export_frame(self, frame: np.ndarray, frame_number: int) -> str:
        """Write a single frame image to the output directory.

        Args:
            frame: BGR image array as returned by OpenCV.
            frame_number: Sequential frame number used in the filename.

        Returns:
            The basename of the written file (e.g. ``frame_000042.png``).
        """
        filename = f"frame_{frame_number:06d}.{self._output_format}"
        path = str(self._output_dir / filename)

        if self._output_format == "jpeg" or self._output_format == "jpg":
            cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality])
        else:
            cv2.imwrite(path, frame)

        return filename

    def write_metadata(self, result: AnalysisResult) -> None:
        """Write ``metadata.json`` into the output directory.

        Args:
            result: The complete analysis result to serialize.
        """
        metadata = {
            "version": "1.0",
            "source": _serialize_source(result),
            "processing": _serialize_processing(result),
            "results": {
                "frames_analyzed": result.frames_analyzed,
                "frames_retained": result.frames_retained,
                "reduction_ratio": result.reduction_ratio,
                "reduction_percentage": result.reduction_percentage,
            },
            "frames": [_serialize_frame(f) for f in result.frames],
        }

        path = self._output_dir / "metadata.json"
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(metadata, fh, indent=2)


# --- private serialization helpers -------------------------------------------


def _serialize_source(result: AnalysisResult) -> dict:
    vi = result.video_info
    return {
        "file": result.source_path,
        "duration_seconds": vi.duration_seconds,
        "total_frames": vi.total_frames,
        "fps": vi.fps,
        "resolution": {"width": vi.width, "height": vi.height},
        "codec": vi.codec,
    }


def _serialize_processing(result: AnalysisResult) -> dict:
    pi = result.processing_info
    return {
        "method": pi.method,
        "threshold": pi.threshold,
        "min_interval_seconds": pi.min_interval_seconds,
        "max_interval_seconds": pi.max_interval_seconds,
        "resize_max_dimension": pi.resize_max_dimension,
        "output_format": pi.output_format,
        "sample_rate": pi.sample_rate,
        "processing_time_seconds": pi.processing_time_seconds,
    }


def _serialize_frame(frame: FrameInfo) -> dict:
    return {
        "file": frame.file,
        "frame_number": frame.frame_number,
        "timestamp_seconds": frame.timestamp_seconds,
        "timestamp_human": frame.timestamp_human,
        "difference_score": frame.difference_score,
        "reason": frame.reason,
        "resolution": {"width": frame.width, "height": frame.height},
    }
