"""Frame comparator algorithms and factory function.

Usage::

    from vfa.comparators import get_comparator

    comparator = get_comparator("ssim", resize_to=128)
    diff = comparator.compare(frame_a, frame_b)
"""

from vfa.comparators.base import FrameComparator
from vfa.comparators.histogram import HistogramComparator
from vfa.comparators.phash import PHashComparator
from vfa.comparators.ssim import SSIMComparator

__all__ = [
    "FrameComparator",
    "SSIMComparator",
    "PHashComparator",
    "HistogramComparator",
    "get_comparator",
]

_COMPARATORS: dict[str, type[FrameComparator]] = {
    "ssim": SSIMComparator,
    "phash": PHashComparator,
    "histogram": HistogramComparator,
}


def get_comparator(method: str = "ssim", **kwargs: object) -> FrameComparator:
    """Create a frame comparator by algorithm name.

    Args:
        method: Algorithm name. One of ``"ssim"``, ``"phash"``, or
            ``"histogram"``.
        **kwargs: Forwarded to the comparator constructor (e.g.
            ``resize_to`` for SSIM, ``hash_size`` for pHash).

    Returns:
        A configured :class:`FrameComparator` instance.

    Raises:
        ValueError: If *method* is not a recognised algorithm name.
    """
    cls = _COMPARATORS.get(method)
    if cls is None:
        available = ", ".join(sorted(_COMPARATORS))
        raise ValueError(
            f"Unknown comparison method: {method}. Available: {available}"
        )
    return cls(**kwargs)  # type: ignore[arg-type]
