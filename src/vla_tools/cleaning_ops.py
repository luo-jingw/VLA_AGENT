"""Detection and repair of recording faults in one episode. Pure functions on arrays, no IO.

Index convention: actions[t] is the target for frame t+1, so state[t+1] echoes actions[t]. A repair to
actions[t] is mirrored into state[t+1] for every dim that has a state counterpart.
"""

from __future__ import annotations

import dataclasses
from typing import Literal

import numpy as np

from vla_tools.robot_spec import RobotSpec

JumpKind = Literal["isolated", "continuing", "discontinuity"]


@dataclasses.dataclass(frozen=True)
class Jump:
    """A one-frame joint change far above the dataset's typical step.

    isolated       the value comes straight back on the next frame: a recording glitch, repairable. The
                   change back on the next frame belongs to the same glitch and is not reported again
    continuing     the next frame keeps moving the same way: a real fast motion
    discontinuity  the value moves and then stops: a teleport-like level shift, needs a decision
    """

    frame: int  # action index t; the change is actions[t] - actions[t-1]
    dim: int  # action dim with the largest change
    steps: float  # size in median steps
    kind: JumpKind


def find_jumps(actions: np.ndarray, spec: RobotSpec, step: float, factor: float) -> list[Jump]:
    joints = list(spec.joint_dims)
    delta = np.diff(actions[:, joints], axis=0)
    size = np.abs(delta).max(axis=1)
    out: list[Jump] = []
    for i in np.flatnonzero(size > factor * step):
        t = int(i) + 1
        if out and out[-1].kind == "isolated" and out[-1].frame == t - 1:
            continue
        j = int(np.abs(delta[i]).argmax())
        d_in = delta[i, j]
        if t + 1 >= len(actions):
            kind: JumpKind = "discontinuity"
        else:
            d_out = actions[t + 1, joints[j]] - actions[t, joints[j]]
            back = abs(actions[t + 1, joints[j]] - actions[t - 1, joints[j]])
            if back < 0.25 * abs(d_in):
                kind = "isolated"
            elif np.sign(d_out) == np.sign(d_in) and abs(d_out) > 0.1 * abs(d_in):
                kind = "continuing"
            else:
                kind = "discontinuity"
        out.append(Jump(frame=t, dim=joints[j], steps=float(size[i] / step), kind=kind))
    return out


def repair_isolated(actions: np.ndarray, states: np.ndarray, spec: RobotSpec, frame: int) -> None:
    """Replace actions[frame] on the joint dims by the midpoint of its neighbours, in place."""
    for dim in spec.joint_dims:
        value = 0.5 * (actions[frame - 1, dim] + actions[frame + 1, dim])
        actions[frame, dim] = value
        if frame + 1 < len(states):
            states[frame + 1, spec.state_dims[dim]] = value


def command_pulses(actions: np.ndarray, dim: int, min_frames: int) -> list[tuple[int, int]]:
    """Interior runs of a constant command shorter than `min_frames`, as [start, end) action indices."""
    values = actions[:, dim] > 0.5
    edges = np.flatnonzero(np.diff(values.astype(np.int8)) != 0) + 1
    bounds = [0, *edges.tolist(), len(values)]
    runs = list(zip(bounds[:-1], bounds[1:], strict=True))
    return [(s, e) for k, (s, e) in enumerate(runs) if 0 < k < len(runs) - 1 and e - s < min_frames]


def remove_pulse(actions: np.ndarray, states: np.ndarray, spec: RobotSpec, dim: int, pulse: tuple[int, int]) -> None:
    """Hold the command at its value before the pulse, in place; the state echo one frame later too."""
    start, end = pulse
    before = actions[start - 1, dim]
    actions[start:end, dim] = before
    state_dim = spec.state_dims[dim]
    if state_dim >= 0:
        states[start + 1 : min(end + 1, len(states)), state_dim] = before
