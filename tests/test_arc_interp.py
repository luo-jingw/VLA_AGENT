"""arc_interp: point placement, per-frame targets through the points, and construction from a label report."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from vla_tools import arc_core
from vla_tools.arc_interp import ArcInterpolator
from vla_tools.arc_interp import after_latency
from vla_tools.arc_interp import point_frames
from vla_tools.arc_interp import to_frames
from vla_tools.resample_config import ResampleConfig
from vla_tools.robot_spec import RobotSpec

SPEC = RobotSpec(action_dim=3, joint_dims=(0, 1), command_dims=(2,), state_dims=(0, 1, -1), fps=10.0)


def test_point_frames_by_anchor() -> None:
    np.testing.assert_allclose(point_frames(20, 10, "state", 1.0), np.arange(2, 21, 2))
    np.testing.assert_allclose(point_frames(20, 10, "action", 1.0), np.arange(3, 22, 2))
    np.testing.assert_allclose(point_frames(20, 10, "state", 2.0), np.arange(1, 11, 1))
    with pytest.raises(ValueError, match="anchor"):
        point_frames(20, 10, "observation", 1.0)


def test_to_frames_passes_through_the_points_and_takes_the_nearest_command() -> None:
    times = point_frames(8, 4, "state", 1.0)  # 2, 4, 6, 8
    points = np.array([[0.2, -0.2, 0.0], [0.4, -0.4, 1.0], [0.6, -0.6, 1.0], [0.8, -0.8, 0.0]])
    out = to_frames(points, current=np.array([0.0, 0.0, 1.0]), times=times, command_dims=(2,))
    assert out.shape == (8, 3)
    np.testing.assert_allclose(out[[1, 3, 5, 7], :2], points[:, :2], atol=1e-12)
    np.testing.assert_allclose(out[:, 0], 0.1 * np.arange(1, 9), atol=1e-12)  # a straight line stays straight
    np.testing.assert_array_equal(
        out[:, 2], [0, 0, 0, 1, 1, 1, 1, 0]
    )  # nearest point; the current command is not a knot


def test_to_frames_is_the_not_a_knot_cubic_spline() -> None:
    """A cubic in time is rebuilt exactly from four or more knots; two knots give the chord, three the parabola."""

    def cubic(t: np.ndarray) -> np.ndarray:
        return np.stack([0.3 * t - 0.02 * t**2 + 0.001 * t**3, -0.1 * t + 0.004 * t**3], axis=1)

    worst = {}
    for horizon in (1, 2, 3, 10):
        times = point_frames(2 * horizon, horizon, "state", 1.0)
        frames = np.arange(1, int(times[-1]) + 1, dtype=np.float64)
        points = np.concatenate([cubic(times), np.zeros((horizon, 1))], axis=1)
        out = to_frames(points, current=np.zeros(3), times=times, command_dims=(2,))
        knots = np.concatenate([[0.0], times])
        values = np.concatenate([np.zeros((1, 2)), cubic(times)])
        if horizon == 1:
            expected = values[0] + (values[1] - values[0]) * (frames / knots[1])[:, None]
        elif horizon == 2:
            expected = np.stack([np.polyval(np.polyfit(knots, values[:, j], 2), frames) for j in range(2)], axis=1)
        else:
            expected = cubic(frames)
        worst[horizon] = float(np.abs(out[:, :2] - expected).max())
    print("largest difference to the chord / parabola / cubic by horizon:", {h: f"{w:.1e}" for h, w in worst.items()})
    assert max(worst.values()) < 1e-12


def _report(path: Path, config: ResampleConfig) -> Path:
    payload = {"robot": {"action_dim": 3, "command_dims": [2], "state_dims": [0, 1, -1]}, "config": config.to_dict()}
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_from_report_reproduces_the_recording_at_the_point_frames(tmp_path: Path) -> None:
    rng = np.random.default_rng(5)
    actions = np.zeros((60, 3))
    actions[:, :2] = np.cumsum(rng.normal(0.0, 0.02, (60, 2)), axis=0)
    actions[25:45, 2] = 1.0
    states = np.concatenate([actions[:1], actions[:-1]])
    config = ResampleConfig(metric="time", span_frames=20, action_horizon=10, anchor="state")
    _, chunks = arc_core.chunks(actions, SPEC, config, windows=[], recorded=len(actions))
    interp = ArcInterpolator.from_report(_report(tmp_path / "arc_resample.json", config))
    at_points = (interp.frames() - 1).astype(int)
    worst = 0.0
    for t in range(len(actions) - 20):
        out = interp(chunks.points[t], states[t])
        worst = max(worst, float(np.abs(out[at_points, :2] - actions[t + at_points, :2]).max()))
    print(f"max difference at the point frames {worst:.1e}")
    assert worst < 1e-9  # invert() adds a 1e-12-per-frame drift to keep progress strictly increasing


def test_points_are_trimmed_to_the_robot_and_checked(tmp_path: Path) -> None:
    config = ResampleConfig(metric="time", span_frames=20, action_horizon=10, anchor="state")
    interp = ArcInterpolator.from_report(_report(tmp_path / "arc_resample.json", config))
    padded = np.zeros((10, 32))
    assert interp(padded, np.zeros(2)).shape == (20, 3)
    with pytest.raises(ValueError, match="expected"):
        interp(np.zeros((8, 3)), np.zeros(2))


def test_report_without_anchor_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "arc_resample.json"
    path.write_text(json.dumps({"robot": {}, "config": {"span_frames": 20, "action_horizon": 10}}), encoding="utf-8")
    with pytest.raises(ValueError, match="anchor"):
        ArcInterpolator.from_report(path)


def test_rate_traverses_the_same_path_faster(tmp_path: Path) -> None:
    rng = np.random.default_rng(7)
    points = np.zeros((10, 3))
    points[:, :2] = np.cumsum(rng.normal(0.0, 0.05, (10, 2)), axis=0)
    points[6:, 2] = 1.0
    state = np.array([0.01, -0.02, 0.0])
    config = ResampleConfig(metric="curvature", span_frames=20, action_horizon=10, anchor="state", curvature_length=0.1)
    report = _report(tmp_path / "arc_resample.json", config)
    slow = ArcInterpolator.from_report(report, 1.0)(points, state)
    fast = ArcInterpolator.from_report(report, 2.0)(points, state)
    difference = float(np.abs(fast - slow[1::2]).max())
    print(
        f"rate 1: {len(slow)} targets, rate 2: {len(fast)} targets; rate-2 target j vs rate-1 target 2j+1: {difference:.1e}"
    )
    assert len(fast) == len(slow) // 2
    assert difference <= 1e-12


def test_after_latency_starts_at_the_commanded_pose(tmp_path: Path) -> None:
    """Per-frame points of a curve, arriving d frames late: the targets after the arrival are the points not
    yet passed, the spline starting at the commanded pose; zero latency is the plain rebuild."""
    t = np.arange(1, 11, dtype=np.float64)
    curve = np.stack([np.sin(0.3 * t), np.cos(0.2 * t), (t > 4).astype(np.float64)], axis=1)
    config = ResampleConfig(metric="time", span_frames=10, action_horizon=10, anchor="state")
    interp = ArcInterpolator.from_report(_report(tmp_path / "arc_resample.json", config))
    start = np.array([0.0, 1.0, 0.0])
    np.testing.assert_array_equal(interp.after_latency(curve, start, 0), interp(curve, start))
    for d in (1, 3, 9):
        out = interp.after_latency(curve, curve[d - 1], d)
        assert out.shape == (10 - d, 3)
        np.testing.assert_allclose(out, curve[d:], atol=1e-12)
    assert interp.after_latency(curve, curve[-1], 10).shape == (0, 3)


def test_after_latency_drops_points_within_half_a_spacing() -> None:
    """Points on a line except the one at frame 4; the commanded pose on the line. Arriving at frame 3 the point
    at 4 lies half a spacing ahead and is dropped: the rebuild is the line. Arriving at 2 it is kept."""
    times = point_frames(20, 10, "state", 1.0)  # 2, 4, ..., 20
    points = np.stack([0.1 * times, -0.1 * times, np.zeros(10)], axis=1)
    points[1, :2] += 0.5
    out = after_latency(points, times, np.array([0.3, -0.3, 0.0]), 3, command_dims=(2,))
    assert out.shape == (17, 3)  # frames 4..20 after the observation
    np.testing.assert_allclose(out[:, 0], 0.1 * np.arange(4, 21), atol=1e-12)
    out = after_latency(points, times, np.array([0.2, -0.2, 0.0]), 2, command_dims=(2,))
    np.testing.assert_allclose(out[1, :2], points[1, :2], atol=1e-12)
