"""Idle frames and episode ends: find still stretches, classify them, and build the new row order. Pure functions.

A still stretch is a run of frames whose largest joint change stays below a threshold. Where it sits
decides what it means:

  leading    before the first motion: the operator had not started; teaches nothing beyond "start"
  trailing   after the last motion: the task is done and the arm holds
  keyframe   overlaps a keyframe window: part of the task (settling before a grasp, waiting for a seal)
  pause      anywhere else: hesitation, or a task wait the keyframe spec does not name yet

End padding appends rows that repeat the last recorded row: the arm holds its final target, the state
equals that target, the cameras repeat the last image. They exist so the policy sees "task done" and is
taught to hold; they are marked in an `is_padding` column so evaluation can score them separately.
"""

from __future__ import annotations

import dataclasses
from typing import Literal

import numpy as np

from vla_tools.robot_spec import RobotSpec

StillKind = Literal["leading", "trailing", "keyframe", "pause"]


@dataclasses.dataclass(frozen=True)
class StillRun:
    start: int  # first row, inclusive
    end: int  # first row after, exclusive
    kind: StillKind


def still_runs(
    actions: np.ndarray, spec: RobotSpec, threshold: float, min_frames: int, windows: list[tuple[int, int]]
) -> list[StillRun]:
    speed = np.concatenate([[0.0], np.abs(np.diff(actions[:, list(spec.joint_dims)], axis=0)).max(axis=1)])
    still = speed < threshold
    still[0] = still[1] if len(still) > 1 else True
    runs, start = [], None
    for t in range(len(still) + 1):
        if t < len(still) and still[t]:
            start = t if start is None else start
        elif start is not None:
            if t - start >= min_frames:
                runs.append((start, t))
            start = None
    out = []
    n = len(actions)
    for s, e in runs:
        if s == 0:
            kind: StillKind = "leading"
        elif e == n:
            kind = "trailing"
        elif any(s < we and ws < e for ws, we in windows):
            kind = "keyframe"
        else:
            kind = "pause"
        out.append(StillRun(s, e, kind))
    return out


@dataclasses.dataclass(frozen=True)
class TimingPlan:
    threshold: float
    min_frames: int
    leading_keep: int | None  # keep this many still frames before the first motion; None keeps all
    pause_keep: int | None  # keep this many frames of each pause; None keeps all
    trailing_keep: int | None
    end_padding: int  # rows appended after the last row


def row_order(n: int, runs: list[StillRun], plan: TimingPlan) -> list[int]:
    """Source rows of the new episode, in order. Keyframe stretches are always kept whole."""
    drop: set[int] = set()
    for run in runs:
        keep = {"leading": plan.leading_keep, "pause": plan.pause_keep, "trailing": plan.trailing_keep}.get(run.kind)
        if keep is None:
            continue
        length = run.end - run.start
        if length <= keep:
            continue
        if run.kind == "leading":
            drop.update(range(run.start, run.end - keep))  # keep the frames just before motion starts
        else:
            drop.update(range(run.start + keep, run.end))
    rows = [t for t in range(n) if t not in drop]
    return rows + [n - 1] * plan.end_padding


def padded_arrays(
    actions: np.ndarray, states: np.ndarray, spec: RobotSpec, rows: list[int], padding: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Actions, states and the is_padding flag for the new row order.

    On padding rows the action holds the last target and the state equals it: the arm has arrived.
    """
    new_actions, new_states = actions[rows].copy(), states[rows].copy()
    flag = np.zeros(len(rows), dtype=np.float32)
    if padding:
        last = actions[-1]
        new_actions[-padding:] = last
        for dim, state_dim in enumerate(spec.state_dims):
            if state_dim >= 0:
                new_states[-padding:, state_dim] = last[dim]
        flag[-padding:] = 1.0
    return new_actions, new_states, flag


def moving_bounds(actions: np.ndarray, spec: RobotSpec, threshold: float) -> tuple[int, int]:
    """Still frames before the first and after the last joint motion above `threshold`."""
    speed = np.abs(np.diff(actions[:, list(spec.joint_dims)], axis=0)).max(axis=1)
    moving = np.flatnonzero(speed > threshold)
    if len(moving) == 0:
        return len(actions), 0
    return int(moving[0]), int(len(actions) - 2 - moving[-1])
