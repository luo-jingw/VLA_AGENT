"""Resampled action labels for one episode. Pure functions, no IO.

A label is built in three stages:

1. Curve. The recorded joints are rebuilt as a continuous curve (`trajectory_curve`: a not-a-knot cubic
   spline over the row index through every recorded row).
2. Window. Frame t's window is the `span_frames` segments that follow its state. Segment i joins action
   rows i and i+1; the state of frame t is action row t-1 under anchor "state" (a virtual segment repeating
   segment 0 precedes row 0) and row t under "action".
3. Sparsify. The H points are placed inside the window by the metric's progress:

    free(i)    progress of segment i outside keyframe windows (everywhere when event_density is 0):
               time               1
               curvature          max(c(i), curvature_floor * c_mean) * l(i), l(i) = ||q[i+1] - q[i]||,
                                  c(i) = ((kappa[i] + kappa[i+1]) / 2)^(1/3), kappa the chord-angle curvature
                                  of the recorded rows (path_curvature) with chords of curvature_length on
                                  either side, c_mean the l-weighted mean of c. Progress per unit path length
                                  depends on the path's shape only; still and end-padding segments add nothing
               second_difference  ||q[i+1] - 2 q[i] + q[i-1]||^(1/3), the acceleration per frame; depends on
                                  teleop speed
    step(t)    A * (span/H) / N, with A the free progress over the window's free segments that lie between
               recorded rows and N their count: one point per span/H recorded frames on average over the
               free part of the window
    keyframes  a respaced segment (inside a keyframe window, event_density > 0) carries
               step(t) * event_density / (span/H): event_density points per span/H frames, evenly in time,
               whatever the rest of the window holds
    still      a window without free progress (A = 0) is spaced in time: 1 per free recorded segment, the
               segments' own progress on end padding, event_density per respaced segment, step = span/H

Point k (k = 1..H) sits at the frame's cumulative progress s_t(anchor) + k * step(t), mapped back to a
fractional action index (linear inside a segment; a target on a plateau, i.e. still segments, resolves to
the plateau's first row) and read from the curve (joints) and the nearest row (commands). A point past the episode's last progress is padded and lands on the last
row. A window without keyframe segments inside the recorded rows ends exactly at its H-th point.
Index convention: actions[i] is the target for frame i+1, i.e. state[i+1].
"""

from __future__ import annotations

import dataclasses
import typing

import numpy as np

from vla_tools import path_curvature
from vla_tools import trajectory_curve
from vla_tools.resample_config import Metric
from vla_tools.resample_config import ResampleConfig
from vla_tools.robot_spec import RobotSpec


@dataclasses.dataclass(frozen=True)
class ProgressCurve:
    free: np.ndarray  # (T-1,) metric progress of each segment, 0 on respaced keyframe segments
    respaced: np.ndarray  # (T-1,) True on keyframe segments carried at event_density
    window_mask: np.ndarray  # (T-1,) True inside a keyframe window
    recorded: int


@dataclasses.dataclass(frozen=True)
class Chunks:
    points: np.ndarray  # (T, H, action_dim)
    frames: np.ndarray  # (T, H) fractional action index of each point
    padded: np.ndarray  # (T, H) target ran past the episode end
    in_window: np.ndarray  # (T, H) point drawn from a keyframe window
    weights: np.ndarray  # (T, H)
    step: np.ndarray  # (T,) progress between consecutive points of each frame's label


def command_edges(actions: np.ndarray, dim: int) -> list[int]:
    on = actions[:, dim] > 0.5
    return (np.flatnonzero(np.diff(on.astype(np.int8)) != 0) + 1).tolist()


def _curvature_metric(q: np.ndarray, recorded: int, config: ResampleConfig) -> np.ndarray:
    out = np.zeros(max(len(q) - 1, 0))
    rows = q[:recorded]
    length = np.linalg.norm(np.diff(rows, axis=0), axis=1)
    if not (length > 0.0).any():
        return out  # no motion: every window is still and spaced in time
    assert config.curvature_length is not None  # ResampleConfig requires it for this metric
    kappa = path_curvature.curvature(rows, config.curvature_length)
    c = np.cbrt((kappa[:-1] + kappa[1:]) / 2.0)
    c_mean = float((c * length).sum() / length.sum())
    out[: recorded - 1] = np.maximum(c, config.curvature_floor * c_mean) * length
    return out


def _raw_metric(actions: np.ndarray, spec: RobotSpec, config: ResampleConfig, recorded: int) -> np.ndarray:
    q = actions[:, list(spec.joint_dims)].astype(np.float64)
    if config.metric == "time":
        return np.ones(len(q) - 1)
    if config.metric == "curvature":
        return _curvature_metric(q, recorded, config)
    if config.metric == "second_difference":
        second = np.linalg.norm(np.diff(q, n=2, axis=0), axis=1)
        return np.power(np.concatenate([second[:1], second]) + 1e-12, 1.0 / 3.0)
    raise ValueError(f"unknown metric {config.metric!r}; allowed: {', '.join(typing.get_args(Metric))}")


def progress_curve(
    actions: np.ndarray, spec: RobotSpec, config: ResampleConfig, windows: list[tuple[int, int]], recorded: int
) -> ProgressCurve:
    """`recorded` is the number of recorded rows of `actions`; the rows after them are end padding."""
    if not 1 <= recorded <= len(actions):
        raise ValueError(f"recorded must be in 1..{len(actions)}, got {recorded}")
    raw = _raw_metric(actions, spec, config, recorded)
    n = len(raw)
    mask = np.zeros(n, dtype=bool)
    for start, end in windows:
        mask[max(start, 0) : min(end, n)] = True
    respaced = mask.copy() if config.event_density > 0 else np.zeros(n, dtype=bool)
    free = np.maximum(raw - config.deadzone, 0.0)
    free[respaced] = 0.0
    return ProgressCurve(free=free, respaced=respaced, window_mask=mask, recorded=recorded)


def invert(s: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Non-decreasing cumulative progress (one value per row) -> fractional row index of each target: the first
    row whose progress reaches it, interpolated linearly inside the segment before that row. A target within
    1e-12 of the total above a plateau's value resolves to the plateau's first row."""
    s = np.asarray(s, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    if len(s) < 2:
        return np.zeros_like(targets)
    tolerance = max(float(s[-1]), 1.0) * 1e-12
    row = np.clip(np.searchsorted(s, targets - tolerance, side="left"), 1, len(s) - 1)
    low, high = s[row - 1], s[row]
    span = high - low
    fraction = np.divide(targets - low, span, out=np.zeros_like(targets), where=span > 0.0)
    return row - 1 + np.clip(fraction, 0.0, 1.0)


def _cumulative(values: np.ndarray) -> np.ndarray:
    return np.concatenate([[0.0], np.cumsum(values)])


def chunks(
    actions: np.ndarray, spec: RobotSpec, config: ResampleConfig, windows: list[tuple[int, int]], recorded: int
) -> tuple[ProgressCurve, Chunks]:
    curve = progress_curve(actions, spec, config, windows, recorded)
    joints, commands = list(spec.joint_dims), list(spec.command_dims)
    path = trajectory_curve.build(actions[:, joints], recorded)
    n, horizon, per_step = len(actions), config.action_horizon, config.frames_per_step
    k = np.arange(1, horizon + 1, dtype=np.float64)

    # Extended segments: index 0 is the virtual segment before row 0, index i + 1 is segment i; extended row
    # r + 1 is action row r. Recorded segments are the extended ones below `recorded`.
    first = slice(0, 1) if len(curve.free) else slice(0, 0)
    free = np.concatenate([curve.free[first], curve.free]) if len(curve.free) else np.zeros(1)
    respaced = np.concatenate([curve.respaced[first], curve.respaced]) if len(curve.free) else np.zeros(1, dtype=bool)
    between_recorded = np.arange(len(free)) < recorded
    open_segment = ~respaced
    window_free = _cumulative(np.where(between_recorded & open_segment, free, 0.0))
    window_count = _cumulative((between_recorded & open_segment).astype(np.float64))
    metric_progress = _cumulative(np.where(open_segment, free, 0.0))
    time_progress = _cumulative(np.where(open_segment, np.where(between_recorded, 1.0, free), 0.0))
    keyframe_count = _cumulative(respaced.astype(np.float64))

    frames = np.zeros((n, horizon))
    padded = np.zeros((n, horizon), dtype=bool)
    step = np.zeros(n)
    for t in range(n):
        a = t if config.anchor == "state" else t + 1  # extended row of the frame's state
        b = max(min(a + config.span_frames, recorded), a)
        progress = window_free[b] - window_free[a]
        if progress > 0.0:
            step[t] = progress * per_step / (window_count[b] - window_count[a])
            s = metric_progress + step[t] * config.event_density / per_step * keyframe_count
        else:
            step[t] = per_step
            s = time_progress + config.event_density * keyframe_count
        targets = s[a] + k * step[t]
        padded[t] = targets > s[-1]
        frames[t] = np.maximum(invert(s, np.minimum(targets, s[-1])) - 1.0, 0.0)

    points = np.zeros((n, horizon, actions.shape[1]), dtype=np.float64)
    flat = frames.ravel()
    points.reshape(-1, actions.shape[1])[:, joints] = path.joints(flat)
    nearest = np.clip(np.round(flat).astype(int), 0, n - 1)
    points.reshape(-1, actions.shape[1])[:, commands] = np.round(actions[nearest][:, commands])
    index = np.clip(np.rint(frames).astype(int), 0, max(len(curve.window_mask) - 1, 0))
    in_window = curve.window_mask[index] if len(curve.window_mask) else np.zeros_like(padded)
    # Padding wins over emphasis: a padded target is clamped onto the last frame.
    weights = np.where(padded, config.pad_weight, np.where(in_window, config.key_weight, 1.0))
    chunk = Chunks(points=points, frames=frames, padded=padded, in_window=in_window, weights=weights, step=step)
    return curve, chunk
