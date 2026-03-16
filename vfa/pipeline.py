"""Analysis pipeline orchestrating extract, filter, resize, and export stages."""

from __future__ import annotations

import time
from pathlib import Path

from tqdm import tqdm

from vfa.comparators import get_comparator
from vfa.exporter import FrameExporter
from vfa.extractor import FrameExtractor
from vfa.filter import FrameFilter
from vfa.models import AnalysisResult, FrameInfo, ProcessingInfo
from vfa.resizer import FrameResizer

_VERSION = "1.0"


def _format_duration(seconds: float) -> str:
    """Format a duration in seconds as ``MM:SS``."""
    minutes = int(seconds) // 60
    secs = int(seconds) % 60
    return f"{minutes}:{secs:02d}"


def _format_timestamp(seconds: float) -> str:
    """Format a timestamp in seconds as ``HH:MM:SS.mmm``."""
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    whole_secs = int(secs)
    millis = int(round((secs - whole_secs) * 1000))
    return f"{int(hours):02d}:{int(minutes):02d}:{whole_secs:02d}.{millis:03d}"


def _determine_sample_rate(duration_seconds: float) -> int:
    """Return an adaptive sample rate based on video duration.

    Rules:
        - Video < 10 min  -> 1
        - Video 10-30 min -> 2
        - Video 30-60 min -> 3
        - Video > 60 min  -> 5
    """
    duration_minutes = duration_seconds / 60.0
    if duration_minutes < 10:
        return 1
    if duration_minutes <= 30:
        return 2
    if duration_minutes <= 60:
        return 3
    return 5


class AnalysisPipeline:
    """Orchestrates the full extract -> filter -> resize -> export pipeline.

    Parameters
    ----------
    video_path:
        Path to the source video file.
    output_dir:
        Directory where extracted frames and metadata are written.
    method:
        Comparison algorithm name (``"ssim"``, ``"phash"``, ``"histogram"``).
    threshold:
        Visual difference threshold for retaining a frame.
    min_interval:
        Minimum seconds between retained frames.
    max_interval:
        Maximum seconds before a frame is forcibly retained.
    sample_rate:
        Keep one in every *sample_rate* frames.  ``None`` uses an adaptive
        heuristic based on video duration.
    max_dimension:
        Resize frames so the largest side does not exceed this value.
    no_resize:
        Disable resizing entirely.
    output_format:
        Image format for exported frames (``"png"`` or ``"jpeg"``).
    jpeg_quality:
        JPEG quality when *output_format* is ``"jpeg"``.
    force:
        Overwrite existing files in *output_dir*.
    quiet:
        Suppress all console output including the progress bar.
    verbose:
        Print per-frame details to the console.
    dry_run:
        Run the pipeline without writing any files to disk.
    no_metadata:
        Skip writing ``metadata.json`` after export.
    keep_first:
        Always retain the first frame.
    keep_last:
        Always retain the last frame.
    """

    def __init__(
        self,
        video_path: str,
        output_dir: str = "./vfa_output/",
        method: str = "ssim",
        threshold: float = 0.3,
        min_interval: float = 0.5,
        max_interval: float = 30.0,
        sample_rate: int | None = None,
        max_dimension: int = 1024,
        no_resize: bool = False,
        output_format: str = "png",
        jpeg_quality: int = 85,
        force: bool = False,
        quiet: bool = False,
        verbose: bool = False,
        dry_run: bool = False,
        no_metadata: bool = False,
        keep_first: bool = True,
        keep_last: bool = True,
    ) -> None:
        self.video_path = video_path
        self.output_dir = output_dir
        self.method = method
        self.threshold = threshold
        self.min_interval = min_interval
        self.max_interval = max_interval
        self.sample_rate = sample_rate
        self.max_dimension = max_dimension
        self.no_resize = no_resize
        self.output_format = output_format
        self.jpeg_quality = jpeg_quality
        self.force = force
        self.quiet = quiet
        self.verbose = verbose
        self.dry_run = dry_run
        self.no_metadata = no_metadata
        self.keep_first = keep_first
        self.keep_last = keep_last

    def run(self) -> AnalysisResult:
        """Execute the full analysis pipeline and return the result."""
        extractor = FrameExtractor(self.video_path, sample_rate=1)
        video_info = extractor.video_info

        # Determine effective sample rate.
        if self.sample_rate is not None:
            effective_sample_rate = self.sample_rate
            sample_rate_label = str(effective_sample_rate)
        else:
            effective_sample_rate = _determine_sample_rate(video_info.duration_seconds)
            sample_rate_label = f"{effective_sample_rate} (auto)"

        # Re-create extractor with the resolved sample rate.
        extractor.release()
        extractor = FrameExtractor(self.video_path, sample_rate=effective_sample_rate)

        comparator = get_comparator(self.method)
        frame_filter = FrameFilter(
            comparator,
            threshold=self.threshold,
            min_interval=self.min_interval,
            max_interval=self.max_interval,
            always_keep_first=self.keep_first,
            always_keep_last=self.keep_last,
        )
        resizer = FrameResizer(max_dimension=self.max_dimension, enabled=not self.no_resize)
        exporter = FrameExporter(
            self.output_dir,
            output_format=self.output_format,
            jpeg_quality=self.jpeg_quality,
            force=self.force,
        )

        if not self.dry_run:
            exporter.setup()

        # Print header.
        if not self.quiet:
            source_name = Path(self.video_path).name
            duration_str = _format_duration(video_info.duration_seconds)
            resize_label = f"{self.max_dimension}px max" if not self.no_resize else "disabled"
            print(f"Video Frame Analyzer v{_VERSION}")
            print("\u2500" * 25)
            print(
                f"Source      : {source_name} "
                f"({duration_str}, {video_info.width}x{video_info.height}, "
                f"{video_info.fps:.0f}fps, {video_info.codec})"
            )
            print(f"Method      : {self.method} (threshold: {self.threshold:.2f})")
            print(f"Sample rate : {sample_rate_label}")
            print(f"Resize      : {resize_label}")
            print()
            print("Extraction and filtering...")

        total_sampled = video_info.total_frames // effective_sample_rate
        frames: list[FrameInfo] = []

        start_time = time.time()

        with tqdm(total=total_sampled, disable=self.quiet) as progress:
            last_progress = 0
            for frame_number, timestamp, frame, difference_score, reason in frame_filter.process(
                extractor, video_info.fps
            ):
                resized = resizer.resize(frame)

                if not self.dry_run:
                    filename = exporter.export_frame(resized, frame_number)
                else:
                    filename = f"frame_{frame_number:06d}.{self.output_format}"

                height, width = resized.shape[:2]
                frame_info = FrameInfo(
                    file=filename,
                    frame_number=frame_number,
                    timestamp_seconds=timestamp,
                    timestamp_human=_format_timestamp(timestamp),
                    difference_score=difference_score,
                    reason=reason,
                    width=width,
                    height=height,
                )

                if self.verbose:
                    print(
                        f"  [{frame_info.timestamp_human}] frame {frame_number} "
                        f"({reason}, score={difference_score:.4f}) -> {filename}"
                    )

                frames.append(frame_info)

                # Advance progress bar to match the current frame position.
                current_progress = frame_number // effective_sample_rate + 1
                advance = current_progress - last_progress
                if advance > 0:
                    progress.update(advance)
                    last_progress = current_progress

            # Flush remaining progress.
            remaining = total_sampled - last_progress
            if remaining > 0:
                progress.update(remaining)

        processing_time = time.time() - start_time

        frames_analyzed = total_sampled
        frames_retained = len(frames)
        if frames_analyzed > 0:
            reduction_ratio = (frames_analyzed - frames_retained) / frames_analyzed
        else:
            reduction_ratio = 0.0
        reduction_percentage = f"{reduction_ratio * 100:.2f}%"

        processing_info = ProcessingInfo(
            method=self.method,
            threshold=self.threshold,
            min_interval_seconds=self.min_interval,
            max_interval_seconds=self.max_interval,
            resize_max_dimension=self.max_dimension if not self.no_resize else None,
            output_format=self.output_format,
            sample_rate=effective_sample_rate,
            processing_time_seconds=processing_time,
        )

        result = AnalysisResult(
            source_path=self.video_path,
            output_dir=self.output_dir,
            video_info=video_info,
            processing_info=processing_info,
            frames_analyzed=frames_analyzed,
            frames_retained=frames_retained,
            reduction_ratio=reduction_ratio,
            reduction_percentage=reduction_percentage,
            frames=frames,
        )

        if not self.dry_run and not self.no_metadata:
            exporter.write_metadata(result)

        extractor.release()

        # Print summary.
        if not self.quiet:
            print()
            print("Results")
            print("\u2500" * 9)
            print(f"Frames analyzed  : {frames_analyzed:,}")
            print(f"Frames retained  : {frames_retained:,} ({frames_retained / max(frames_analyzed, 1) * 100:.2f}%)")
            print(f"Reduction ratio  : {reduction_percentage}")
            print(f"Processing time  : {processing_time:.1f}s")
            output_label = self.output_dir if not self.dry_run else "(dry run)"
            file_count = f"{frames_retained} files" if not self.dry_run else "no files written"
            print(f"Output           : {output_label} ({file_count})")

        return result


def analyze_video(video_path: str, **kwargs: object) -> AnalysisResult:
    """Convenience function to run a full video analysis pipeline.

    Parameters
    ----------
    video_path:
        Path to the source video file.
    **kwargs:
        All other parameters accepted by :class:`AnalysisPipeline`.

    Returns
    -------
    AnalysisResult
        The complete analysis result.
    """
    pipeline = AnalysisPipeline(video_path, **kwargs)  # type: ignore[arg-type]
    return pipeline.run()
