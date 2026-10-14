"""Geometric curvature of a sampled path. Pure function, no IO.

kappa at sample i is the turning angle between two chords that each span at least `window` of arc length,
divided by half the arc length they cover:

    S          cumulative Euclidean arc length along the samples
    j          latest index with S[i] - S[j] >= window (0 if none)
    k          earliest index with S[k] - S[i] >= window (N - 1 if none)
    u, v       chords q[i] - q[j] and q[k] - q[i]
    kappa(i)   angle(u, v) / ((S[k] - S[j]) / 2), or 0 when either chord is shorter than 1e-12

Arc length, not the sample index, sets the chord ends, so the estimate depends on the path's shape and not
on how fast it was traversed: repeated (still) samples add no arc length and produce no spurious curvature.
On a circle of radius R the estimate is 1/R up to the polyline-versus-arc length difference.
"""

from __future__ import annotations

import numpy as np

_MIN_CHORD = 1e-12


def curvature(q: np.ndarray, window: float) -> np.ndarray:
    """kappa at every sample of an (N, D) path; float64 array of length N."""
    if q.ndim != 2:
        raise ValueError(f"path must be (N, D), got shape {q.shape}")
    if window <= 0.0:
        raise ValueError(f"window must be positive arc length, got {window}")
    path = q.astype(np.float64)
    n = len(path)
    if n < 3:
        return np.zeros(n)
    arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))])
    behind = np.maximum(np.searchsorted(arc, arc - window, side="right") - 1, 0)
    ahead = np.minimum(np.searchsorted(arc, arc + window, side="left"), n - 1)
    u = path - path[behind]
    v = path[ahead] - path
    u_len = np.linalg.norm(u, axis=1)
    v_len = np.linalg.norm(v, axis=1)
    valid = (u_len >= _MIN_CHORD) & (v_len >= _MIN_CHORD)
    kappa = np.zeros(n)
    u_unit = u[valid] / u_len[valid, None]
    v_unit = v[valid] / v_len[valid, None]
    # Angle between unit vectors, accurate near 0 and pi: 2 * atan2(|u - v|, |u + v|).
    angle = 2.0 * np.arctan2(np.linalg.norm(u_unit - v_unit, axis=1), np.linalg.norm(u_unit + v_unit, axis=1))
    kappa[valid] = angle / ((arc[ahead[valid]] - arc[behind[valid]]) / 2.0)
    return kappa
