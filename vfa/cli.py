"""Command-line interface for the Video Frame Analyzer.

Entry point registered as ``vfa = "vfa.cli:main"`` in pyproject.toml.
Parses arguments, validates inputs, delegates to :class:`vfa.pipeline.AnalysisPipeline`.
"""

from __future__ import annotations

import argparse
import sys

__version__ = "1.0.0"


def _build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for the ``vfa`` command."""

    parser = argparse.ArgumentParser(
        prog="vfa",
        description="Video Frame Analyzer — extract representative frames from a video.",
    )
    parser.add_argument("video_path", help="Path to the video file")
    parser.add_argument(
        "-o",
        "--output",
        default="./vfa_output/",
        help="Output directory (default: ./vfa_output/)",
    )
    parser.add_argument(
        "-m",
        "--method",
        choices=["ssim", "phash", "histogram"],
        default="ssim",
        help="Frame-difference algorithm (default: ssim)",
    )
    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=0.3,
        help="Difference threshold 0.0-1.0 (default: 0.3)",
    )
    parser.add_argument(
        "--min-interval",
        type=float,
        default=0.5,
        help="Minimum seconds between retained frames (default: 0.5)",
    )
    parser.add_argument(
        "--max-interval",
        type=float,
        default=30.0,
        help="Maximum seconds without a retained frame (default: 30.0)",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=None,
        help="Sample 1 frame every N (default: auto/adaptive)",
    )
    parser.add_argument(
        "--max-dimension",
        type=int,
        default=1024,
        help="Max pixel dimension for output frames (default: 1024)",
    )
    parser.add_argument(
        "--no-resize",
        action="store_true",
        default=False,
        help="Disable frame resizing",
    )
    parser.add_argument(
        "-f",
        "--format",
        choices=["png", "jpeg"],
        default="png",
        help="Output image format (default: png)",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=85,
        help="JPEG quality 1-100 (default: 85)",
    )
    parser.add_argument(
        "--no-metadata",
        action="store_true",
        default=False,
        help="Skip generating metadata.json",
    )
    parser.add_argument(
        "--keep-first",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep the first frame (default: True)",
    )
    parser.add_argument(
        "--keep-last",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep the last frame (default: True)",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        default=False,
        help="Silent mode — suppress the progress bar",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=False,
        help="Verbose mode — print per-frame scores",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Simulate analysis without writing files",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Overwrite existing output directory",
    )

    return parser


def main(argv: list[str] | None = None) -> None:
    """Parse CLI arguments and run the analysis pipeline.

    Parameters
    ----------
    argv:
        Argument list to parse.  ``None`` (the default) reads from
        ``sys.argv[1:]``, which is the standard behaviour when invoked
        as a console script.  Passing an explicit list is useful for
        testing.
    """

    parser = _build_parser()
    args = parser.parse_args(argv)

    # --- Validate threshold --------------------------------------------------
    if not 0.0 <= args.threshold <= 1.0:
        print(
            f"error: threshold must be between 0.0 and 1.0, got {args.threshold}",
            file=sys.stderr,
        )
        sys.exit(1)

    # --- Build pipeline and run ----------------------------------------------
    try:
        from vfa.pipeline import AnalysisPipeline  # noqa: PLC0415

        pipeline = AnalysisPipeline(
            video_path=args.video_path,
            output_dir=args.output,
            method=args.method,
            threshold=args.threshold,
            min_interval=args.min_interval,
            max_interval=args.max_interval,
            sample_rate=args.sample_rate,
            max_dimension=args.max_dimension,
            no_resize=args.no_resize,
            output_format=args.format,
            jpeg_quality=args.jpeg_quality,
            no_metadata=args.no_metadata,
            keep_first=args.keep_first,
            keep_last=args.keep_last,
            quiet=args.quiet,
            verbose=args.verbose,
            dry_run=args.dry_run,
            force=args.force,
        )
        pipeline.run()

    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(2)
    except FileExistsError as exc:
        print(f"error: {exc} (use --force to overwrite)", file=sys.stderr)
        sys.exit(2)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(3)
    except Exception as exc:  # noqa: BLE001
        print(f"error: unexpected failure: {exc}", file=sys.stderr)
        sys.exit(3)
