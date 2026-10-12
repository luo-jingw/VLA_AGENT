"""cleaning_ops on hand-built arrays: jump classes, isolated repair, command pulses."""

from __future__ import annotations

import numpy as np

from vla_tools import cleaning_ops
from vla_tools.robot_spec import RobotSpec

SPEC = RobotSpec(action_dim=3, joint_dims=(0, 1), command_dims=(2,), state_dims=(0, 1, 2), fps=10.0)
STEP = 0.01


def _ramp(frames: int) -> np.ndarray:
    """Joint 0 moves one step per frame, joint 1 is still, the command is open."""
    actions = np.zeros((frames, 3))
    actions[:, 0] = STEP * np.arange(frames)
    return actions


def test_isolated_glitch_is_one_jump_and_its_return_is_not_reported() -> None:
    actions = _ramp(20)
    actions[8, 1] += 0.5
    jumps = cleaning_ops.find_jumps(actions, SPEC, STEP, factor=8.0)
    print(jumps)
    assert [(j.frame, j.dim, j.kind) for j in jumps] == [(8, 1, "isolated")]


def test_fast_motion_is_continuing_and_its_end_a_discontinuity() -> None:
    actions = _ramp(20)
    actions[8:, 0] += 0.2
    actions[9:, 0] += 0.2
    jumps = cleaning_ops.find_jumps(actions, SPEC, STEP, factor=8.0)
    print(jumps)
    assert [(j.frame, j.kind) for j in jumps] == [(8, "continuing"), (9, "discontinuity")]


def test_level_shift_is_a_discontinuity() -> None:
    actions = np.zeros((20, 3))
    actions[10:, 1] = 0.3
    jumps = cleaning_ops.find_jumps(actions, SPEC, STEP, factor=8.0)
    assert [(j.frame, j.kind) for j in jumps] == [(10, "discontinuity")]


def test_jump_on_the_last_frame_is_a_discontinuity() -> None:
    actions = _ramp(10)
    actions[9, 0] += 0.5
    assert [j.kind for j in cleaning_ops.find_jumps(actions, SPEC, STEP, factor=8.0)] == ["discontinuity"]


def test_repair_isolated_takes_the_midpoint_and_mirrors_into_the_next_state() -> None:
    actions = _ramp(10)
    actions[4, 1] = 0.5
    states = np.concatenate([actions[:1], actions[:-1]])
    cleaning_ops.repair_isolated(actions, states, SPEC, 4)
    np.testing.assert_allclose(actions[4, :2], [4 * STEP, 0.0])
    np.testing.assert_allclose(states[5, :2], actions[4, :2])


def test_command_pulses_are_interior_short_runs_only() -> None:
    actions = np.zeros((30, 3))
    actions[0:2, 2] = 1.0  # leading run: not interior
    actions[10:12, 2] = 1.0  # pulse
    actions[20:26, 2] = 1.0  # long run
    actions[29:, 2] = 1.0  # trailing run: not interior
    assert cleaning_ops.command_pulses(actions, 2, min_frames=3) == [(10, 12)]


def test_remove_pulse_holds_the_previous_value_in_actions_and_state_echo() -> None:
    actions = np.zeros((20, 3))
    actions[10:12, 2] = 1.0
    states = np.concatenate([actions[:1], actions[:-1]])
    cleaning_ops.remove_pulse(actions, states, SPEC, 2, (10, 12))
    assert not actions[:, 2].any()
    assert not states[:, 2].any()
