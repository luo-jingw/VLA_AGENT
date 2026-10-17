"""Server side: a model's resampled points -> one target per control frame. numpy + scipy only.

Standalone on purpose: it imports nothing from the rest of vla_tools, so this one file can be copied to a
serving machine. It runs the label builder's sampling backwards:

    joints    not-a-knot cubic spline through the current pose at frame 0 and the points (two knots: the
              chord; three: the parabola through them)
    commands  the nearest point in time, re-binarised. The current state is not a knot: its command
              value is the previous command echoed back, and letting it decide the frames before the
              first point re-opens a gripper the model already closed whenever the echo lags

Point k is placed at (offset + k * span / H) / rate frames after the observation, offset 0 for anchor
"state" and 1 for "action". The execution rate is a deployment choice: at rate 1 a chunk that covers span
recorded frames is executed over span frames, the recording's average pace; at rate r the same spline
path is traversed r times faster. Under metric "time" at rate 1 the placement is exact; otherwise the
timing differs from the recording, which for static tasks costs nothing but speed, measured by the label
replay.

Output: out[j] is the target for frame t + j + 1, the meaning a per-frame policy's actions have.

Inference latency: a chunk that arrives d frames after its observation, while the robot keeps executing the
previous chunk, is rebuilt from the pose commanded at arrival (after_latency): knots = that pose at 0 and the
points whose time minus d exceeds half a point spacing, at their times minus d. The switch between chunks
is continuous in position; out[j] is then the target for frame arrival + j + 1.

Velocity timing (VelocityTiming, a deployment setting like the rate): the same joint spline, traversed with
chunk time T (frames of the uniform timing above) mapped to control frames
s(T) = integral of max(1, fps * max_j |dq_j/dT| / (fraction * v_j)) dT, the integrand taken as the larger
endpoint value on intervals of 1/32 control frame; one target at each s = 1, 2, ..., floor(s(T_end)). A
chunk is never faster than the rate (the cap) and never moves a joint faster than the fraction of its
velocity limit along the spline; the rate then caps only the slow and still stretches. At the first target
whose command differs from the previous target, or from the pose the chunk starts at, the arm stays
round(hold * fps) further frames at that target (rise_hold_s for a command rising above 0.5, fall_hold_s for
one falling below it): a fixed still segment for the gripper to act, since velocity timing passes the nearly
still stretch around a switch at the cap. The start pose's command must then be the commanded value, not a
lagging echo, or the hold repeats. A chunk that replaces one before its hold is over starts with the rest of
the hold (VelocityTiming.owed_hold, from the commanded poses so far). Arriving after a latency, the chunk time already passed is that of target
d of the chunk timed from the pose commanded at the observation (arrival_time), instead of d.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import scipy.interpolate


def point_frames(span_frames: int, action_horizon: int, anchor: str, rate: float) -> np.ndarray:
    offsets = {"state": 0.0, "action": 1.0}
    if rate <= 0.0:
        raise ValueError(f"rate must be positive, got {rate}")
    if anchor not in offsets:
        raise ValueError(f"anchor must be one of {sorted(offsets)}, got {anchor!r}")
    return (
        offsets[anchor] + np.arange(1, action_horizon + 1, dtype=np.float64) * (span_frames / action_horizon)
    ) / rate


@dataclasses.dataclass(frozen=True)
class VelocityTiming:
    """Chunk timing at a fraction of the joint velocity limits, capped by the rate, with a still segment at
    each command switch."""

    velocity_limits: tuple[float, ...]  # rad/s, one per non-command dim in dim order
    fraction: float  # of each limit
    fps: float  # control rate
    rise_hold_s: float  # still after a command rises above 0.5
    fall_hold_s: float  # still after a command falls below 0.5

    def __post_init__(self) -> None:
        if not self.velocity_limits or min(self.velocity_limits) <= 0.0:
            raise ValueError(f"velocity_limits must be positive, got {self.velocity_limits}")
        if self.fraction <= 0.0 or self.fps <= 0.0 or min(self.rise_hold_s, self.fall_hold_s) < 0.0:
            raise ValueError(f"fraction and fps must be positive and holds >= 0, got {self}")

    def frame_times(self, spline: scipy.interpolate.CubicSpline, end: float) -> np.ndarray:
        """Chunk time T of control frames 1, 2, ..., floor(s(end)) along the joint spline. The integrand is
        sampled every 1/32 control frame (a first pass at 1/32 of a chunk-time unit sets the step)."""
        limits = self.fraction * np.asarray(self.velocity_limits) / self.fps  # rad per control frame

        def cost(grid: np.ndarray) -> np.ndarray:
            return np.maximum(1.0, (np.abs(spline(grid, 1)) / limits).max(axis=1))

        grid = np.linspace(0.0, end, int(np.ceil(end * 32.0)) + 1)
        grid = np.linspace(0.0, end, int(np.ceil(end * 32.0 * float(cost(grid).max()))) + 1)
        c = cost(grid)
        s = np.concatenate([[0.0], np.cumsum(np.maximum(c[1:], c[:-1]) * np.diff(grid))])
        return np.interp(np.arange(1, int(np.floor(s[-1] + 1e-9)) + 1, dtype=np.float64), s, grid)

    def hold_frames(self, before: np.ndarray, after: np.ndarray) -> int:
        """Further frames at a target whose command values `after` differ from `before`."""
        rising = bool(((after > 0.5) & (before <= 0.5)).any())
        falling = bool(((after <= 0.5) & (before > 0.5)).any())
        return max(
            round(self.rise_hold_s * self.fps) if rising else 0, round(self.fall_hold_s * self.fps) if falling else 0
        )

    def owed_hold(self, history: Sequence[np.ndarray], command_dims: tuple[int, ...]) -> int:
        """Further frames the arm must still stay at the last of the commanded (D,) poses `history` (oldest
        first): what remains of the hold that the last command switch started, when every pose since the
        switch repeats the switching target; 0 otherwise. A chunk that replaces one still holding starts with
        that many copies of the commanded pose."""
        commands = list(command_dims)
        last, k = history[-1], len(history) - 1
        while k > 0 and np.array_equal(history[k - 1], last):
            k -= 1
        if k == 0 or not commands or ((history[k - 1][commands] > 0.5) == (last[commands] > 0.5)).all():
            return 0
        return max(0, self.hold_frames(history[k - 1][commands], last[commands]) - (len(history) - 1 - k))


def _timed(
    points: np.ndarray, current: np.ndarray, times: np.ndarray, command_dims: tuple[int, ...],
    timing: VelocityTiming | None,
) -> tuple[np.ndarray, np.ndarray]:  # fmt: skip
    """-> (n, D) targets and the (n,) chunk time of each."""
    dim = points.shape[1]
    joints = [d for d in range(dim) if d not in command_dims]
    commands = list(command_dims)
    knots = np.concatenate([[0.0], times])
    values = np.concatenate([current[None, joints], points[:, joints]], axis=0)
    spline = scipy.interpolate.CubicSpline(knots, values, axis=0, bc_type="not-a-knot")
    if timing is None:
        at = np.arange(1, int(np.floor(times[-1])) + 1, dtype=np.float64)
    else:
        at = timing.frame_times(spline, float(times[-1]))
    out = np.zeros((len(at), dim), dtype=np.float64)
    out[:, joints] = spline(at)
    nearest = np.abs(at[:, None] - times[None, :]).argmin(axis=1)
    out[:, commands] = np.round(points[nearest][:, commands])
    if timing is None or not commands:
        return out, at
    repeats = np.ones(len(out), dtype=np.int64)
    previous = current[commands]
    for k in range(len(out)):
        if ((out[k, commands] > 0.5) != (previous > 0.5)).any():
            repeats[k] += timing.hold_frames(previous, out[k, commands])
        previous = out[k, commands]
    return np.repeat(out, repeats, axis=0), np.repeat(at, repeats)


def to_frames(
    points: np.ndarray, current: np.ndarray, times: np.ndarray, command_dims: tuple[int, ...],
    timing: VelocityTiming | None = None,
) -> np.ndarray:  # fmt: skip
    """(H, D) points at `times`, starting from the (D,) current pose -> (n, D) targets: n = floor(times[-1])
    with uniform timing, at least that with velocity timing (holds included)."""
    return _timed(points, current, times, command_dims, timing)[0]


def target_times(
    points: np.ndarray,
    current: np.ndarray,
    times: np.ndarray,
    command_dims: tuple[int, ...],
    timing: VelocityTiming | None = None,
) -> np.ndarray:
    """(n,) chunk time of each target of to_frames; a held target repeats its time."""
    return _timed(points, current, times, command_dims, timing)[1]


def arrival_time(
    points: np.ndarray, observed: np.ndarray, times: np.ndarray, command_dims: tuple[int, ...],
    timing: VelocityTiming | None, elapsed_frames: int,
) -> float:  # fmt: skip
    """Chunk time passed `elapsed_frames` after the observation: that of target `elapsed_frames` of the chunk
    timed from the (D,) pose commanded at the observation (the elapsed frames with uniform timing; the end
    of the chunk once it is spent)."""
    if elapsed_frames <= 0:
        return 0.0
    at = target_times(points, observed, times, command_dims, timing)
    return float(at[elapsed_frames - 1]) if elapsed_frames <= len(at) else float(times[-1])


def after_latency(
    points: np.ndarray, times: np.ndarray, commanded: np.ndarray, elapsed: float, command_dims: tuple[int, ...],
    timing: VelocityTiming | None = None,
) -> np.ndarray:  # fmt: skip
    """(H, D) points at `times` after the observation, arriving when `elapsed` of the chunk time has passed
    (the elapsed frames with uniform timing) and the robot is commanded to the (D,) pose `commanded` -> (n, D)
    targets for the frames after the arrival. Points within half a point spacing of the arrival or before it
    are dropped; with none left n is 0 and the commanded pose is held."""
    spacing = float(np.diff(np.concatenate([[0.0], times])).min())
    keep = times - elapsed > 0.5 * spacing
    if not keep.any():
        return np.zeros((0, points.shape[1]), dtype=np.float64)
    return to_frames(points[keep], commanded, times[keep] - elapsed, command_dims, timing)


@dataclasses.dataclass(frozen=True)
class ArcInterpolator:
    """Construct once per model from the label report; call once per inference."""

    span_frames: int
    action_horizon: int
    anchor: str
    action_dim: int
    command_dims: tuple[int, ...]
    state_dims: tuple[int, ...]  # state index of each action dim, -1 when none
    rate: float = 1.0  # execution rate, chosen at deployment; the label report does not record it
    timing: VelocityTiming | None = None  # velocity timing capped by the rate; chosen at deployment

    @staticmethod
    def from_report(path: Path, rate: float = 1.0, timing: VelocityTiming | None = None) -> ArcInterpolator:
        """Everything from the arc_resample.json the label builder wrote, nothing restated by hand, the rate
        and the timing."""
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
            rate=float(rate),
            timing=timing,
        )

    def frames(self) -> np.ndarray:
        return point_frames(self.span_frames, self.action_horizon, self.anchor, self.rate)

    def current_pose(self, state: np.ndarray) -> np.ndarray:
        """The state expressed in the action layout; command dims without a state counterpart are 0."""
        state = np.asarray(state, dtype=np.float64)
        return np.array([state[s] if s >= 0 else 0.0 for s in self.state_dims])

    def _checked(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64)[:, : self.action_dim]
        if points.shape != (self.action_horizon, self.action_dim):
            raise ValueError(f"points {points.shape}, expected ({self.action_horizon}, {self.action_dim})")
        return points

    def __call__(self, points: np.ndarray, state: np.ndarray) -> np.ndarray:
        """Absolute (H, >=D) points and the state they were predicted from -> (n, D) per-frame targets.

        Model padding past action_dim is dropped. `points` must be absolute targets, i.e. after the
        policy has undone normalization and delta encoding.
        """
        return to_frames(self._checked(points), self.current_pose(state), self.frames(), self.command_dims, self.timing)

    def after_latency(
        self, points: np.ndarray, observed: np.ndarray, commanded: np.ndarray, elapsed_frames: int
    ) -> np.ndarray:
        """The same points arriving `elapsed_frames` after the observation, rebuilt from the (D,) pose the
        robot is commanded to at arrival -> (n, D) targets for the frames after the arrival; n = 0: hold the
        commanded pose. `observed` is the (D,) pose commanded at the observation: the chunk timed from it
        tells how much chunk time has passed (the elapsed frames with uniform timing). Poses in action
        layout."""
        points = self._checked(points)
        commanded = np.asarray(commanded, dtype=np.float64)[: self.action_dim]
        observed = np.asarray(observed, dtype=np.float64)[: self.action_dim]
        times = self.frames()
        elapsed = arrival_time(points, observed, times, self.command_dims, self.timing, elapsed_frames)
        return after_latency(points, times, commanded, elapsed, self.command_dims, self.timing)
