<p align="center">
  <h1 align="center">🎬 Video Frame Analyzer (VFA)</h1>
  <p align="center">
    <strong>Extract representative frames from videos — built for LLM context windows.</strong>
  </p>
  <p align="center">
    <a href="https://pypi.org/project/video-frame-analyzer/"><img alt="PyPI" src="https://img.shields.io/pypi/v/video-frame-analyzer"></a>
    <a href="https://pypi.org/project/video-frame-analyzer/"><img alt="Python" src="https://img.shields.io/pypi/pyversions/video-frame-analyzer"></a>
    <a href="https://github.com/BaLaurent/video-frame-analyzer/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/github/license/BaLaurent/video-frame-analyzer"></a>
  </p>
</p>

---

VFA intelligently reduces a video to its **key frames** by comparing consecutive frames and keeping only those that are visually distinct. Feed the output to an LLM, build visual summaries, or extract screenshots — all from a single command.

## Features

- **3 comparison algorithms** — SSIM (structural), Histogram (color), PHash (perceptual)
- **Adaptive sampling** — automatically adjusts to video length
- **Smart filtering** — min/max interval, threshold tuning, first/last frame control
- **Metadata export** — JSON file with timestamps, scores, and video info
- **Python API** — use as a library in your own scripts
- **Zero GPU required** — runs on CPU with OpenCV

## Quick Start

### Installation

```bash
pip install video-frame-analyzer
```

Or with [pipx](https://pipx.pypa.io/) (recommended for CLI usage):

```bash
pipx install video-frame-analyzer
```

### Basic Usage

```bash
# Extract key frames with default settings (SSIM, threshold 0.3)
vfa video.mp4

# Save to a specific directory
vfa video.mp4 -o ./frames

# Use a stricter threshold for less frames
vfa video.mp4 -t 0.5

# Use histogram comparison (best for slides/presentations)
vfa video.mp4 -m histogram -t 0.15
```

### Output

```
Video Frame Analyzer v1.0
─────────────────────────
Source      : video.mp4 (5:30, 1920x1080, 30fps, h264)
Method      : ssim (threshold: 0.30)
Sample rate : 1 (auto)
Resize      : 1024px max

Extraction and filtering...
100%|████████████████████| 9900/9900 [00:12<00:00, 825.00it/s]

Results
─────────
Frames analyzed  : 9,900
Frames retained  : 12 (0.12%)
Reduction ratio  : 99.88%
Processing time  : 12.3s
Output           : ./vfa_output (12 files)
```

The output directory contains:
```
vfa_output/
├── frame_000000.png
├── frame_000547.png
├── frame_001203.png
├── ...
└── metadata.json
```

## Algorithm Guide

| Algorithm | Best For | Speed | How It Works |
|-----------|----------|-------|--------------|
| **SSIM** (default) | General purpose, screencasts, UI changes | ⚡ Fast | Structural similarity on grayscale. Detects layout/text changes. |
| **Histogram** | Slides, presentations, color transitions | ⚡ Fast | HSV color distribution comparison. Detects palette changes. |
| **PHash** | Gameplay, scene changes, major transitions | 🐢 ~3x slower | Perceptual hash via DCT. Captures broad spatial structure. |

### Choosing the Right Parameters

| Video Type | Method | Threshold | Min Interval | Example |
|------------|--------|-----------|--------------|---------|
| Screencast / scrolling | ssim | 0.35 | 5.0s | `vfa video.mp4 -t 0.35 --min-interval 5.0 --no-keep-last` |
| Slides / presentation | histogram | 0.15 | 1.0s | `vfa slides.mp4 -m histogram -t 0.15` |
| Gameplay / action | phash | 0.30 | 2.0s | `vfa gameplay.mp4 -m phash --min-interval 2.0` |
| Long video (>1h) | ssim | 0.30 | 1.0s | `vfa long.mp4 -t 0.30 --sample-rate 5` |
| Quick overview | ssim | 0.50 | 10.0s | `vfa video.mp4 -t 0.50 --min-interval 10` |

## CLI Reference

```
usage: vfa [-h] [-o OUTPUT] [-m {ssim,phash,histogram}] [-t THRESHOLD]
           [--min-interval MIN_INTERVAL] [--max-interval MAX_INTERVAL]
           [--sample-rate SAMPLE_RATE] [--max-dimension MAX_DIMENSION]
           [--no-resize] [-f {png,jpeg}] [--jpeg-quality JPEG_QUALITY]
           [--no-metadata] [--keep-first | --no-keep-first]
           [--keep-last | --no-keep-last] [-q] [-v] [--version]
           [--dry-run] [--force]
           video_path
```

| Flag | Default | Description |
|------|---------|-------------|
| `-o, --output` | `./vfa_output/` | Output directory |
| `-m, --method` | `ssim` | Comparison algorithm: `ssim`, `phash`, `histogram` |
| `-t, --threshold` | `0.3` | Difference threshold (0.0–1.0). Higher = fewer frames |
| `--min-interval` | `0.5` | Min seconds between retained frames |
| `--max-interval` | `30.0` | Max seconds before forcing a frame capture |
| `--sample-rate` | auto | Sample 1 frame every N. Auto adjusts by duration |
| `--max-dimension` | `1024` | Max pixel size for output frames |
| `--no-resize` | off | Keep original resolution |
| `-f, --format` | `png` | Output format: `png` or `jpeg` |
| `--jpeg-quality` | `85` | JPEG quality (1–100) |
| `--no-metadata` | off | Skip `metadata.json` generation |
| `--keep-first` / `--no-keep-first` | on | Always keep the first frame |
| `--keep-last` / `--no-keep-last` | on | Always keep the last frame |
| `-q, --quiet` | off | Suppress progress bar |
| `-v, --verbose` | off | Print per-frame scores |
| `--dry-run` | off | Simulate without writing files |
| `--force` | off | Overwrite existing output directory |

## Python API

```python
from vfa import analyze_video

result = analyze_video(
    "video.mp4",
    method="ssim",
    threshold=0.35,
    min_interval=5.0,
    output_dir="./frames",
    force=True,
)

print(f"Kept {result.frames_retained} / {result.frames_analyzed} frames")
print(f"Reduction: {result.reduction_percentage}")

for frame in result.frames:
    print(f"  {frame.timestamp_human} — {frame.file} (score: {frame.difference_score:.4f})")
```

### Pipeline API (advanced)

```python
from vfa.pipeline import AnalysisPipeline

pipeline = AnalysisPipeline(
    video_path="video.mp4",
    method="phash",
    threshold=0.30,
    min_interval=2.0,
    max_interval=30.0,
    output_dir="./output",
    output_format="jpeg",
    jpeg_quality=90,
    force=True,
)

result = pipeline.run()
```

## Metadata Output

Every run generates a `metadata.json` with full traceability:

```json
{
  "vfa_version": "1.0",
  "source": "video.mp4",
  "video_info": {
    "duration_seconds": 330.0,
    "total_frames": 9900,
    "fps": 30.0,
    "width": 1920,
    "height": 1080,
    "codec": "h264"
  },
  "processing": {
    "method": "ssim",
    "threshold": 0.30,
    "sample_rate": 1,
    "processing_time_seconds": 12.3
  },
  "summary": {
    "frames_analyzed": 9900,
    "frames_retained": 12,
    "reduction_percentage": "99.88%"
  },
  "frames": [
    {
      "file": "frame_000000.png",
      "frame_number": 0,
      "timestamp": "00:00:00.000",
      "difference_score": 0.0,
      "reason": "first_frame"
    }
  ]
}
```

## How It Works

1. **Extract** — OpenCV reads frames from the video at the configured sample rate
2. **Compare** — Each sampled frame is compared to the **last retained frame** (not the previous frame)
3. **Filter** — Frames exceeding the difference threshold are retained, respecting min/max interval constraints
4. **Resize** — Retained frames are downscaled to fit within `max_dimension`
5. **Export** — Frames are saved as PNG/JPEG with a metadata JSON sidecar

### Adaptive Sampling

VFA automatically adjusts the sample rate based on video duration:

| Duration | Sample Rate | Frames Checked |
|----------|-------------|----------------|
| < 10 min | Every frame | 100% |
| 10–30 min | 1 in 2 | 50% |
| 30–60 min | 1 in 3 | 33% |
| > 60 min | 1 in 5 | 20% |

Override with `--sample-rate N` for full control.

## Development

```bash
git clone https://github.com/BaLaurent/video-frame-analyzer.git
cd video-frame-analyzer
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

### Project Structure

```
video-frame-analyzer/
├── vfa/
│   ├── __init__.py          # Public API exports
│   ├── cli.py               # Command-line interface
│   ├── pipeline.py          # Orchestration pipeline
│   ├── extractor.py         # Frame extraction (OpenCV)
│   ├── filter.py            # Frame filtering logic
│   ├── resizer.py           # Frame resizing
│   ├── exporter.py          # File export + metadata
│   ├── models.py            # Data models (dataclasses)
│   └── comparators/
│       ├── base.py          # Abstract comparator
│       ├── ssim.py          # SSIM algorithm
│       ├── histogram.py     # Histogram algorithm
│       └── phash.py         # PHash algorithm
├── tests/                   # 116 tests
├── pyproject.toml
├── LICENSE
└── README.md
```

## License

MIT — see [LICENSE](LICENSE) for details.

## Credits

Built with [OpenCV](https://opencv.org/), [scikit-image](https://scikit-image.org/), [ImageHash](https://github.com/JohannesBuchner/imagehash), and [Pillow](https://python-pillow.org/).
