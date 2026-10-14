"""The continuous joint curve of one episode, rebuilt from its recorded rows. Pure functions, no IO.

The recorded rows sample the joint path uniformly in time. The curve is a not-a-knot cubic spline over the
row index through every recorded row, and stays at the last recorded row beyond it, so end-padding rows do
not shape it. Selected by measurement on 50 LIBERO demonstrations against PCHIP and against chord-length and
centripetal parameterizations: rebuilt from every 4th row it is closest to the held-out rows (p95 0.105
path steps, next 0.134) and leaves the recorded path least (p95 0.081); see docs/validation.md.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import scipy.interpolate


@dataclasses.dataclass(frozen=True)
class TrajectoryCurve:
    spline: scipy.interpolate.CubicSpline | None  # None when a single row is recorded
    last: np.ndarray  # (J,) joints of the last recorded row
    recorded: int

    def joints(self, frames: np.ndarray) -> np.ndarray:
        """(N,) fractional row indices -> (N, J) joints; before row 0 the first row, past the last recorded row
        the last one."""
        frames = np.clip(np.asarray(frames, dtype=np.float64), 0.0, self.recorded - 1.0)
        if self.spline is None:
            return np.repeat(self.last[None, :], len(frames), axis=0)
        return self.spline(frames)


def build(joints: np.ndarray, recorded: int) -> TrajectoryCurve:
    """`joints` (T, J) of every row; the first `recorded` rows are the recording."""
    if not 1 <= recorded <= len(joints):
        raise ValueError(f"recorded must be in 1..{len(joints)}, got {recorded}")
    rows = np.asarray(joints[:recorded], dtype=np.float64)
    if recorded < 2:
        return TrajectoryCurve(spline=None, last=rows[-1].copy(), recorded=recorded)
    spline = scipy.interpolate.CubicSpline(np.arange(recorded, dtype=np.float64), rows, axis=0, bc_type="not-a-knot")
    return TrajectoryCurve(spline=spline, last=rows[-1].copy(), recorded=recorded)
