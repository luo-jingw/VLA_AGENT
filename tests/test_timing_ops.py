"""timing_ops on hand-built arrays: still-run classes, row order, end padding."""

from __future__ import annotations

import numpy as np

from vla_tools import timing_ops
from vla_tools.robot_spec import RobotSpec

SPEC = RobotSpec(action_dim=2, joint_dims=(0,), command_dims=(1,), state_dims=(0, 1), fps=10.0)


def _episode() -> np.ndarray:
    """Still 0-5, move 6-15, still 16-23 (a pause), move 24-33, still 34-39 at the grasp, move 40-49, still 50-59."""
    speed = np.zeros(60)
    for start, end in ((6, 16), (24, 34), (40, 50)):
        speed[start:end] = 0.05
    actions = np.zeros((60, 2))
    actions[:, 0] = np.cumsum(speed)
    return actions


def _plan(**keep: int | None) -> timing_ops.TimingPlan:
    values = {"leading_keep": None, "pause_keep": None, "trailing_keep": None, "end_padding": 0} | keep
    return timing_ops.TimingPlan(threshold=0.01, min_frames=4, **values)


def test_still_runs_are_classified_by_position() -> None:
    runs = timing_ops.still_runs(_episode(), SPEC, threshold=0.01, min_frames=4, windows=[(32, 38)])
    print(runs)
    assert [(r.start, r.end, r.kind) for r in runs] == [
        (0, 6, "leading"),
        (16, 24, "pause"),
        (34, 40, "keyframe"),
        (50, 60, "trailing"),
    ]


def test_row_order_keeps_frames_next_to_motion_and_whole_keyframe_runs() -> None:
    actions = _episode()
    runs = timing_ops.still_runs(actions, SPEC, threshold=0.01, min_frames=4, windows=[(32, 38)])
    rows = timing_ops.row_order(len(actions), runs, _plan(leading_keep=2, pause_keep=3, trailing_keep=1))
    assert rows[:3] == [4, 5, 6]  # the two still frames just before the first motion
    assert [t for t in range(16, 24) if t in rows] == [16, 17, 18]
    assert all(t in rows for t in range(34, 40))
    assert rows[-1] == 50


def test_row_order_appends_padding_rows_repeating_the_last_row() -> None:
    rows = timing_ops.row_order(10, [], _plan(end_padding=3))
    assert rows == [*range(10), 9, 9, 9]


def test_padded_arrays_hold_the_last_target_and_set_the_state_to_it() -> None:
    actions = np.stack([np.arange(5.0), np.array([0, 0, 1, 1, 1.0])], axis=1)
    states = np.concatenate([actions[:1], actions[:-1]])
    rows = timing_ops.row_order(5, [], _plan(end_padding=2))
    new_actions, new_states, flag = timing_ops.padded_arrays(actions, states, SPEC, rows, 2)
    np.testing.assert_array_equal(new_actions[-2:], [actions[-1], actions[-1]])
    np.testing.assert_array_equal(new_states[-2:], [actions[-1], actions[-1]])
    np.testing.assert_array_equal(flag, [0, 0, 0, 0, 0, 1, 1])


def test_moving_bounds_counts_still_frames_at_both_ends() -> None:
    assert timing_ops.moving_bounds(_episode(), SPEC, threshold=0.01) == (5, 10)
