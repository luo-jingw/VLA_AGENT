"""Action normalization statistics per point of a chunk.

openpi pools the statistics of an action dimension over the H points of a chunk, so a span much longer than
H frames sets every point's range from its farthest points. Computed over flattened (H * D) chunks and
reshaped to (H, D), each point gets its own range; openpi's Normalize and Unnormalize broadcast (H, D)
statistics over (H, D) actions, and Unnormalize leaves model padding past D unchanged.
"""

from __future__ import annotations

import numpy as np
from openpi.shared import normalize


def per_point(flat: normalize.NormStats, chunk_shape: tuple[int, ...]) -> normalize.NormStats:
    """Statistics of flattened (H * D) chunks -> the same statistics shaped (H, D)."""

    def shaped(values: np.ndarray | None) -> np.ndarray | None:
        return None if values is None else np.asarray(values).reshape(chunk_shape)

    return normalize.NormStats(mean=shaped(flat.mean), std=shaped(flat.std), q01=shaped(flat.q01), q99=shaped(flat.q99))
