"""Resampled action labels for one episode. Pure functions, no IO.

Progress is a monotone integral with two regimes:

    ds(i) = | step * event_density / frames_per_step     i inside a keyframe window and event_density > 0
            | max(raw(i) - deadzone, 0)                   everywhere else

raw(i) is the progress of the segment from frame i to i+1, measured on the joint dims q:

    time               1
    curvature          max(c(i), curvature_floor * c_mean) * l(i), with l(i) = ||q[i+1] - q[i]||,
                       c(i) = ((kappa[i] + kappa[i+1]) / 2)^(1/3), kappa the geometric curvature of the joint
                       path (path_curvature, chords of curvature_window median per-frame steps), and c_mean
                       the l-weighted mean of c over the episode. Progress per unit path length depends on the
                       path's shape only, not on the teleop speed; still frames add nothing
    second_difference  ||q[i+1] - 2 q[i] + q[i-1]||^(1/3), the acceleration per frame; depends on teleop speed

`step` makes one point per `span_frames / action_horizon` frames on average outside the respaced windows.

Frame t's label is H points at progress a(t) + k*step, k = 1..H, mapped back to fractional action
indices and sampled from the recording: PCHIP on joints, nearest frame on commands. a(t) is the anchor:
the current state's progress s(t-1) (state[0] extrapolated one frame back), or s(t) under "action".
Index convention: actions[i] is the target for frame i+1, i.e. state[i+1].
"""

from __future__ import annotations

import dataclasses
import typing

import numpy as np
import scipy.interpolate

from vla_tools import path_curvature
from vla_tools.resample_config import Metric
from vla_tools.resample_config import ResampleConfig
from vla_tools.robot_spec import RobotSpec


@dataclasses.dataclass(frozen=True)
class ProgressCurve:
    s: np.ndarray  # (T,) monotone, s[0] = 0
    ds: np.ndarray  # (T-1,)
    window_mask: np.ndarray  # (T-1,) True inside a keyframe window
    step: float

    @property
    def total(self) -> float:
        return float(self.s[-1])


@dataclasses.dataclass(frozen=True)
class Chunks:
    points: np.ndarray  # (T, H, action_dim)
    frames: np.ndarray  # (T, H) fractional action index of each point
    padded: np.ndarray  # (T, H) target ran past the episode end
    in_window: np.ndarray  # (T, H) point drawn from a keyframe window
    weights: np.ndarray  # (T, H)


def command_edges(actions: np.ndarray, dim: int) -> list[int]:
    on = actions[:, dim] > 0.5
    return (np.flatnonzero(np.diff(on.astype(np.int8)) != 0) + 1).tolist()


def _curvature_metric(q: np.ndarray, config: ResampleConfig) -> np.ndarray:
    length = np.linalg.norm(np.diff(q, axis=0), axis=1)
    moving = length > 0.0
    if not moving.any():
        return np.zeros(len(length))  # no motion: progress_curve falls back to time
    step = float(np.median(length[moving]))
    kappa = path_curvature.curvature(q, config.curvature_window * step)
    c = np.cbrt((kappa[:-1] + kappa[1:]) / 2.0)
    c_mean = float((c * length).sum() / length.sum())
    return np.maximum(c, config.curvature_floor * c_mean) * length


def _raw_metric(actions: np.ndarray, spec: RobotSpec, config: ResampleConfig) -> np.ndarray:
    q = actions[:, list(spec.joint_dims)].astype(np.float64)
    if config.metric == "time":
        return np.ones(len(q) - 1)
    if config.metric == "curvature":
        return _curvature_metric(q, config)
    if config.metric == "second_difference":
        second = np.linalg.norm(np.diff(q, n=2, axis=0), axis=1)
        return np.power(np.concatenate([second[:1], second]) + 1e-12, 1.0 / 3.0)
    raise ValueError(f"unknown metric {config.metric!r}; allowed: {', '.join(typing.get_args(Metric))}")


def progress_curve(
    actions: np.ndarray, spec: RobotSpec, config: ResampleConfig, windows: list[tuple[int, int]]
) -> ProgressCurve:
    raw = _raw_metric(actions, spec, config)
    n = len(raw)
    mask = np.zeros(n, dtype=bool)
    for start, end in windows:
        mask[max(start, 0) : min(end, n)] = True
    respaced = mask if config.event_density > 0 else np.zeros(n, dtype=bool)

    free = np.maximum(raw - config.deadzone, 0.0)
    free[respaced] = 0.0
    n_free = n - int(respaced.sum())
    total_free = float(free.sum())
    if total_free <= 0.0 and n_free > 0:
        free[~respaced] = 1.0  # a path with no measurable progress falls back to time
        total_free = float(n_free)
    step = 1.0 / max(n, 1) if n_free <= 0 else total_free * config.frames_per_step / n_free
    ds = free.copy()
    ds[respaced] = step * config.event_density / config.frames_per_step
    return ProgressCurve(s=np.concatenate([[0.0], np.cumsum(ds)]), ds=ds, window_mask=mask, step=float(step))


def invert(curve: ProgressCurve, targets: np.ndarray) -> np.ndarray:
    """Progress values -> fractional action indices. A negligible drift makes s strictly increasing."""
    index = np.arange(len(curve.s), dtype=np.float64)
    strict = curve.s + index * (max(curve.total, 1.0) * 1e-12)
    return np.interp(targets, strict, index)


def sample(actions: np.ndarray, spec: RobotSpec, frames: np.ndarray) -> np.ndarray:
    joints, commands = list(spec.joint_dims), list(spec.command_dims)
    grid = np.arange(len(actions), dtype=np.float64)
    out = np.zeros((len(frames), actions.shape[1]), dtype=np.float64)
    pchip = scipy.interpolate.PchipInterpolator(grid, actions[:, joints].astype(np.float64), axis=0)
    out[:, joints] = pchip(np.clip(frames, 0.0, len(actions) - 1.0))
    nearest = np.clip(np.round(frames).astype(int), 0, len(actions) - 1)
    out[:, commands] = np.round(actions[nearest][:, commands])
    return out


def chunks(
    actions: np.ndarray, spec: RobotSpec, config: ResampleConfig, windows: list[tuple[int, int]]
) -> tuple[ProgressCurve, Chunks]:
    curve = progress_curve(actions, spec, config, windows)
    n, horizon = len(actions), config.action_horizon
    first = curve.s[0] - (curve.ds[0] if len(curve.ds) else 0.0)
    anchor = curve.s if config.anchor == "action" else np.concatenate([[first], curve.s[:-1]])
    targets = anchor[:, None] + np.arange(1, horizon + 1, dtype=np.float64)[None, :] * curve.step
    padded = targets > curve.total
    frames = invert(curve, np.clip(targets, 0.0, curve.total).ravel()).reshape(n, horizon)
    points = sample(actions, spec, frames.ravel()).reshape(n, horizon, actions.shape[1])
    index = np.clip(np.rint(frames).astype(int), 0, max(len(curve.ds) - 1, 0))
    in_window = curve.window_mask[index] if len(curve.ds) else np.zeros_like(padded)
    # Padding wins over emphasis: a padded target is clamped onto the last frame.
    weights = np.where(padded, config.pad_weight, np.where(in_window, config.key_weight, 1.0))
    return curve, Chunks(points=points, frames=frames, padded=padded, in_window=in_window, weights=weights)
