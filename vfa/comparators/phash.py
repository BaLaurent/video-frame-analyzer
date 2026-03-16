"""Perceptual hash (pHash) frame comparator using imagehash."""

import imagehash
import numpy as np
from PIL import Image

from vfa.comparators.base import FrameComparator


class PHashComparator(FrameComparator):
    """Compare frames using perceptual hashing (pHash).

    Perceptual hashing reduces each frame to a compact binary fingerprint
    by applying a DCT to a downscaled grayscale version of the image and
    keeping only the low-frequency coefficients. Two frames are compared
    by computing the Hamming distance between their hashes, normalized to
    the [0.0, 1.0] range.

    Args:
        hash_size: Side length of the hash grid. The resulting hash has
            ``hash_size ** 2`` bits. Default is 8 (64-bit hash).
    """

    def __init__(self, hash_size: int = 8) -> None:
        self._hash_size = hash_size
        self._hash_bits = hash_size ** 2

    def compare(self, frame_a: np.ndarray, frame_b: np.ndarray) -> float:
        """Return the normalized Hamming distance between perceptual hashes.

        Args:
            frame_a: First frame as a BGR numpy array (OpenCV format).
            frame_b: Second frame as a BGR numpy array (OpenCV format).

        Returns:
            Float in [0.0, 1.0] where 0.0 means identical perceptual hashes
            and 1.0 means every hash bit differs.
        """
        hash_a = imagehash.phash(self._to_pil(frame_a), hash_size=self._hash_size)
        hash_b = imagehash.phash(self._to_pil(frame_b), hash_size=self._hash_size)

        hamming_distance = hash_a - hash_b
        return float(hamming_distance / self._hash_bits)

    @property
    def name(self) -> str:
        """Algorithm name used in logs and metadata output."""
        return "phash"

    @staticmethod
    def _to_pil(frame: np.ndarray) -> Image.Image:
        """Convert a BGR numpy array (OpenCV) to an RGB PIL Image."""
        rgb = frame[:, :, ::-1]
        return Image.fromarray(rgb)
