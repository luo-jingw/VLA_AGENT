"""Keyframe detectors on a hand-built episode, and the spec's reference rules."""

from __future__ import annotations

import numpy as np
import pytest

from vla_tools.keyframe_detectors import detect
from vla_tools.keyframe_detectors import detect_episode
from vla_tools.keyframe_spec import Detector
from vla_tools.keyframe_spec import KeyframeSpec
from vla_tools.keyframe_spec import KeyframeType
from vla_tools.keyframe_spec import Signal
from vla_tools.keyframe_spec import Window
from vla_tools.robot_spec import RobotSpec

SPEC = RobotSpec(action_dim=2, joint_dims=(0,), command_dims=(1,), state_dims=(0, 1), fps=10.0)
WINDOW = Window(before=2, after=2)


def _episode() -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Joint 0 moves on 5-14 (slowest at 12 within it), stops, closes at 20, moves again from 26, opens at 40.

    The `force` column steps up at 22 and settles from 25.
    """
    speed = np.zeros(50)
    speed[5:15] = [0.05, 0.05, 0.05, 0.05, 0.05, 0.04, 0.03, 0.01, 0.02, 0.02]
    speed[26:36] = 0.05
    actions = np.zeros((50, 2))
    actions[:, 0] = np.cumsum(speed)
    actions[20:40, 1] = 1.0
    states = np.concatenate([actions[:1], actions[:-1]])
    force = np.zeros(50)
    force[22:25] = [2.0, 4.0, 5.0]
    force[25:] = 5.0
    return actions, states, {"force": force}


def _detect(detector: Detector, found: dict[str, list[int]] | None = None) -> list[int]:
    actions, states, columns = _episode()
    return detect(detector, actions, states, SPEC, found or {}, columns)


def test_command_edges_rise_and_fall() -> None:
    gripper = Signal("action", dim=1)
    assert _detect(Detector("command_edge", signal=gripper, direction="rise")) == [20]
    assert _detect(Detector("command_edge", signal=gripper, direction="fall")) == [40]


def test_column_signal_on_the_action_column_matches_the_action_signal() -> None:
    actions, states, columns = _episode()
    as_column = Signal("column", dim=1, name="action")
    found = detect(Detector("command_edge", signal=as_column, direction="rise"), actions, states, SPEC, {},
                   columns | {"action": actions})  # fmt: skip
    assert found == _detect(Detector("command_edge", signal=Signal("action", dim=1), direction="rise"))


def test_motion_onset_after_a_reference() -> None:
    onset = Detector("motion_onset", threshold=0.01, hold_frames=2, reference="grasp", search_frames=15)
    assert _detect(onset, {"grasp": [20]}) == [26]


def test_speed_minimum_before_a_reference() -> None:
    slowest = Detector("speed_minimum", reference="stop", search_frames=5)
    assert _detect(slowest, {"stop": [15]}) == [12]


def test_settle_after_a_reference() -> None:
    settle = Detector("settle", signal=Signal("column", name="force"), threshold=0.1, hold_frames=3,
                      reference="contact", search_frames=10)  # fmt: skip
    assert _detect(settle, {"contact": [22]}) == [25]


def test_threshold_cross_without_reference_finds_every_crossing() -> None:
    cross = Detector("threshold_cross", signal=Signal("joint_speed"), threshold=0.03, direction="rise", hold_frames=2)
    assert _detect(cross) == [5, 26]


def test_manual_detects_nothing() -> None:
    assert _detect(Detector("manual")) == []


def test_detect_episode_runs_types_in_order() -> None:
    actions, states, columns = _episode()
    kspec = KeyframeSpec(
        task="",
        effector="",
        types=(
            KeyframeType("grasp", "", Detector("command_edge", Signal("action", 1), "rise"), WINDOW, 1),
            KeyframeType(
                "lift",
                "",
                Detector("motion_onset", threshold=0.01, hold_frames=2, reference="grasp", search_frames=15),
                WINDOW,
                1,
            ),
        ),
        order=("grasp", "lift"),
    )
    assert detect_episode(kspec, actions, states, SPEC, columns) == {"grasp": [20], "lift": [26]}


def test_spec_rejects_unknown_or_later_references() -> None:
    grasp = KeyframeType("grasp", "", Detector("command_edge", Signal("action", 1), "rise"), WINDOW, 1)
    lift = KeyframeType("lift", "", Detector("motion_onset", threshold=0.01, reference="grasp"), WINDOW, 1)
    with pytest.raises(ValueError, match="must be listed before"):
        KeyframeSpec(task="", effector="", types=(lift, grasp), order=())
    orphan = KeyframeType("lift", "", Detector("motion_onset", threshold=0.01, reference="seal"), WINDOW, 1)
    with pytest.raises(ValueError, match="unknown type"):
        KeyframeSpec(task="", effector="", types=(grasp, orphan), order=())
