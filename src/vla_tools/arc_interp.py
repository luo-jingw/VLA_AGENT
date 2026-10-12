"""Server side: a model's resampled points -> one target per control frame. numpy + scipy only.

Standalone on purpose: it imports nothing from the rest of vla_tools, so this one file can be copied to a
serving machine. It runs the label builder's sampling backwards:

    joints    PCHIP through the current pose at frame 0 and the points
    commands  the nearest point in time, re-binarised. The current state is not a knot: its command
              value is the previous command echoed back, and letting it decide the frames before the
              first point re-opens a gripper the model already closed whenever the echo lags

Point k is placed at offset + k * span / H frames after the observation, offset 0 for anchor "state"
and 1 for "action". Under metric "time" that is exact; under a progress metric it is an assumption
whose cost the label replay measures.

Output: out[j] is the target for frame t + j + 1, the meaning a per-frame policy's actions have.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import scipy.interpolate


def point_frames(span_frames: int, action_horizon: int, anchor: str) -> np.ndarray:
    offsets = {"state": 0.0, "action": 1.0}
    if anchor not in offsets:
        raise ValueError(f"anchor must be one of {sorted(offsets)}, got {anchor!r}")
    return offsets[anchor] + np.arange(1, action_horizon + 1, dtype=np.float64) * (span_frames / action_horizon)


def to_frames(points: np.ndarray, current: np.ndarray, times: np.ndarray, command_dims: tuple[int, ...]) -> np.ndarray:
    """(H, D) points at `times`, starting from the (D,) current pose -> (floor(times[-1]), D) targets."""
    dim = points.shape[1]
    joints = [d for d in range(dim) if d not in command_dims]
    commands = list(command_dims)
    frames = np.arange(1, int(np.floor(times[-1])) + 1, dtype=np.float64)
    out = np.zeros((len(frames), dim), dtype=np.float64)
    knots = np.concatenate([[0.0], times])
    values = np.concatenate([current[None, joints], points[:, joints]], axis=0)
    out[:, joints] = scipy.interpolate.PchipInterpolator(knots, values, axis=0)(frames)
    nearest = np.abs(frames[:, None] - times[None, :]).argmin(axis=1)
    out[:, commands] = np.round(points[nearest][:, commands])
    return out


@dataclasses.dataclass(frozen=True)
class ArcInterpolator:
    """Construct once per model from the label report; call once per inference."""

    span_frames: int
    action_horizon: int
    anchor: str
    action_dim: int
    command_dims: tuple[int, ...]
    state_dims: tuple[int, ...]  # state index of each action dim, -1 when none

    @staticmethod
    def from_report(path: Path) -> ArcInterpolator:
        """Everything from the arc_resample.json the label builder wrote, nothing restated by hand."""
        report = json.loads(Path(path).read_text(encoding="utf-8"))
        config, robot = report["config"], report["robot"]
        if "anchor" not in config:
            raise ValueError(f"{path} records no anchor; pass it explicitly (datasets without it used 'action')")
        return ArcInterpolator(
            span_frames=int(config["span_frames"]),
            action_horizon=int(config["action_horizon"]),
            anchor=str(config["anchor"]),
            action_dim=int(robot["action_dim"]),
            command_dims=tuple(int(d) for d in robot["command_dims"]),
            state_dims=tuple(int(d) for d in robot["state_dims"]),
        )

    def frames(self) -> np.ndarray:
        return point_frames(self.span_frames, self.action_horizon, self.anchor)

    def current_pose(self, state: np.ndarray) -> np.ndarray:
        """The state expressed in the action layout; command dims without a state counterpart are 0."""
        state = np.asarray(state, dtype=np.float64)
        return np.array([state[s] if s >= 0 else 0.0 for s in self.state_dims])

    def __call__(self, points: np.ndarray, state: np.ndarray) -> np.ndarray:
        """Absolute (H, >=D) points and the state they were predicted from -> (n, D) per-frame targets.

        Model padding past action_dim is dropped. `points` must be absolute targets, i.e. after the
        policy has undone normalization and delta encoding.
        """
        points = np.asarray(points, dtype=np.float64)[:, : self.action_dim]
        if points.shape != (self.action_horizon, self.action_dim):
            raise ValueError(f"points {points.shape}, expected ({self.action_horizon}, {self.action_dim})")
        return to_frames(points, self.current_pose(state), self.frames(), self.command_dims)
