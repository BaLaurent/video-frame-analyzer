"""Video Frame Analyzer — extract and filter video frames for LLM context windows.

Usage::

    from vfa import analyze_video

    result = analyze_video("video.mp4")
    print(result.frames_retained)
    print(result.reduction_percentage)
"""

from __future__ import annotations

__version__ = "1.0.0"

from vfa.models import AnalysisResult, FrameInfo, ProcessingInfo, VideoInfo
from vfa.pipeline import AnalysisPipeline, analyze_video

__all__ = [
    "__version__",
    "analyze_video",
    "AnalysisPipeline",
    "AnalysisResult",
    "FrameInfo",
    "ProcessingInfo",
    "VideoInfo",
]
