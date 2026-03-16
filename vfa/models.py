"""Data models for Video Frame Analyzer results and metadata."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class VideoInfo:
    """Metadata about the source video file."""

    duration_seconds: float
    total_frames: int
    fps: float
    width: int
    height: int
    codec: str


@dataclass
class ProcessingInfo:
    """Parameters used during the analysis run."""

    method: str
    threshold: float
    min_interval_seconds: float
    max_interval_seconds: float
    resize_max_dimension: int | None
    output_format: str
    sample_rate: int
    processing_time_seconds: float


@dataclass
class FrameInfo:
    """Metadata about a single retained frame."""

    file: str
    frame_number: int
    timestamp_seconds: float
    timestamp_human: str
    difference_score: float
    reason: str  # "first_frame" | "last_frame" | "threshold_exceeded" | "max_interval_reached"
    width: int
    height: int


@dataclass
class AnalysisResult:
    """Complete result of a video analysis run."""

    source_path: str
    output_dir: str
    video_info: VideoInfo
    processing_info: ProcessingInfo
    frames_analyzed: int
    frames_retained: int
    reduction_ratio: float
    reduction_percentage: str
    frames: list[FrameInfo] = field(default_factory=list)
