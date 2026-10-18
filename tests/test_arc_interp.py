"""arc_interp: point placement, per-frame targets through the points, and construction from a label report."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from vla_tools import arc_core
from vla_tools.arc_interp import ArcInterpolator
from vla_tools.arc_interp import VelocityTiming
from vla_tools.arc_interp import after_latency
from vla_tools.arc_interp import arrival_time
from vla_tools.arc_interp import chunk_targets
from vla_tools.arc_interp import point_frames
from vla_tools.arc_interp import target_times
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
    np.testing.assert_array_equal(interp.after_latency(curve, start, start, 0), interp(curve, start))
    for d in (1, 3, 9):
        out = interp.after_latency(curve, start, curve[d - 1], d)
        assert out.shape == (10 - d, 3)
        np.testing.assert_allclose(out, curve[d:], atol=1e-12)
    assert interp.after_latency(curve, start, curve[-1], 10).shape == (0, 3)


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


def _timing(limit: float, rise: float = 0.0, fall: float = 0.0, fraction: float = 1.0) -> VelocityTiming:
    return VelocityTiming(
        velocity_limits=(limit, limit), fraction=fraction, fps=10.0, rise_hold_s=rise, fall_hold_s=fall
    )


def _wiggle(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    points = np.zeros((10, 3))
    points[:, :2] = np.cumsum(rng.normal(0.0, 0.08, (10, 2)), axis=0)
    return points


def test_velocity_timing_without_a_binding_limit_is_the_uniform_timing() -> None:
    times = point_frames(20, 10, "state", 2.0)
    points, current = _wiggle(3), np.zeros(3)
    uniform = to_frames(points, current, times, (2,))
    timed = to_frames(points, current, times, (2,), _timing(1e6))
    print(f"{len(uniform)} targets; largest difference {np.abs(timed - uniform).max():.1e}")
    assert timed.shape == uniform.shape
    np.testing.assert_allclose(timed, uniform, atol=1e-12)


def test_velocity_timing_slows_a_fast_straight_to_the_fraction_of_the_limit() -> None:
    """A straight line of 2 rad in 10 frames at rate 1 needs 0.2 rad per frame; at 80% of 0.5 rad/s and 10
    fps a frame may move 0.04 rad: 50 frames, each of 0.04 rad."""
    times = point_frames(10, 10, "state", 1.0)
    points = np.stack([0.2 * times, np.zeros(10), np.zeros(10)], axis=1)
    out = to_frames(points, np.zeros(3), times, (2,), _timing(0.5, fraction=0.8))
    steps = np.abs(np.diff(np.concatenate([np.zeros((1, 3)), out])[:, 0]))
    print(f"{len(out)} targets, step {steps.min():.6f}..{steps.max():.6f} rad")
    assert len(out) == 50
    np.testing.assert_allclose(steps, 0.04, atol=1e-9)


def test_still_chunk_runs_at_the_rate() -> None:
    times = point_frames(32, 10, "state", 4.0)
    current = np.array([0.3, -0.2, 0.0])
    out = to_frames(np.tile(current, (10, 1)), current, times, (2,), _timing(0.5))
    assert len(out) == 8


def test_velocity_timing_keeps_every_joint_within_the_fraction() -> None:
    worst, lengths = 0.0, []
    for seed in range(20):
        for rate in (1.0, 4.0, 16.0):
            times = point_frames(32, 10, "state", rate)
            points, current = _wiggle(seed), np.array([0.05, -0.05, 0.0])
            out = to_frames(points, current, times, (2,), _timing(0.6, fraction=0.9))
            steps = np.abs(np.diff(np.concatenate([current[None, :2], out[:, :2]]), axis=0))
            worst = max(worst, float(steps.max() * 10.0 / (0.9 * 0.6)))
            lengths.append(len(out))
    print(f"largest joint speed / (fraction x limit) {worst:.4f}; targets per chunk {min(lengths)}..{max(lengths)}")
    assert worst <= 1.0


def test_command_switches_hold_the_arm() -> None:
    """Rising at the 4th point and falling at the 8th: 0.5 s and 0.2 s at 10 fps hold 5 and 2 further frames at
    the switching targets; the times repeat with them."""
    times = point_frames(20, 10, "state", 1.0)
    points = _wiggle(4)
    points[3:7, 2] = 1.0
    plain = to_frames(points, np.zeros(3), times, (2,), _timing(1e6))
    held = to_frames(points, np.zeros(3), times, (2,), _timing(1e6, rise=0.5, fall=0.2))
    at = target_times(points, np.zeros(3), times, (2,), _timing(1e6, rise=0.5, fall=0.2))
    rise, fall = int(np.flatnonzero(plain[:, 2] > 0.5)[0]), int(np.flatnonzero(plain[:, 2] > 0.5)[-1]) + 1
    print(f"{len(plain)} -> {len(held)} targets; rise at {rise}, fall at {fall}")
    assert len(held) == len(plain) + 7
    np.testing.assert_array_equal(held[rise : rise + 6], np.repeat(plain[rise : rise + 1], 6, axis=0))
    np.testing.assert_array_equal(held[fall + 5 : fall + 8], np.repeat(plain[fall : fall + 1], 3, axis=0))
    assert len(at) == len(held) and np.all(np.diff(at) >= 0.0)
    started = to_frames(points[3:], np.array([0.0, 0.0, 0.0]), times[:7], (2,), _timing(1e6, rise=0.5))
    assert np.array_equal(started[0], started[5]) and not np.array_equal(
        started[5], started[6]
    )  # against the start pose


def test_velocity_timed_arrival_rebuild_starts_at_the_commanded_pose() -> None:
    """A chunk timed from the observed pose is d = 3 frames under way at arrival: its chunk time there is that
    of target 3, the points past it are kept, and the rebuild leaves the commanded pose within the limit."""
    times = point_frames(32, 10, "state", 4.0)
    points, observed = _wiggle(6), np.zeros(3)
    timing = _timing(0.4)
    first = to_frames(points, observed, times, (2,), timing)
    elapsed = arrival_time(points, observed, times, (2,), timing, 3)
    assert elapsed == pytest.approx(target_times(points, observed, times, (2,), timing)[2])
    out = after_latency(points, times, first[2], elapsed, (2,), timing)
    step = float(np.abs(out[0, :2] - first[2, :2]).max()) * 10.0 / 0.4
    print(f"chunk {len(first)} targets, chunk time at arrival {elapsed:.3f} of {times[-1]:g}; {len(out)} after it; "
          f"first step {step:.3f} of the limit")  # fmt: skip
    assert elapsed < 3.0 and step <= 1.0
    assert len(out) >= len(first) - 3 - 1


def test_owed_hold_is_what_remains_of_the_last_switch() -> None:
    timing = _timing(1.0, rise=0.5, fall=0.2)  # 5 and 2 further frames at 10 fps
    a, b, c = np.array([0.0, 0.0, 0.0]), np.array([0.1, 0.0, 1.0]), np.array([0.2, 0.0, 1.0])
    assert timing.owed_hold([a, b], (2,)) == 5
    assert timing.owed_hold([a, b, b, b], (2,)) == 3
    assert timing.owed_hold([a, b, *[b] * 7], (2,)) == 0
    assert timing.owed_hold([a, b, b, c], (2,)) == 0  # moved on
    assert timing.owed_hold([b, c, c], (2,)) == 0  # no switch
    assert timing.owed_hold([b], (2,)) == 0
    assert timing.owed_hold([c, a], (2,)) == 2  # falling


def test_first_and_sync_chunks_start_at_the_observed_pose() -> None:
    """The first chunk (async or sync) and a sync chunk are the rebuild from the observed pose, cut to e."""
    times = point_frames(20, 10, "state", 2.0)
    points, start = _wiggle(8), np.zeros(3)
    plain = to_frames(points, start, times, (2,))
    for mode in ("sync", "async"):
        out = chunk_targets(points, times, (2,), None, [start], 1, 3, mode, 4)
        np.testing.assert_array_equal(out.targets, plain[:4])
        assert out.stalls == 0
    later = [start, plain[0], plain[1]]
    out = chunk_targets(points, times, (2,), None, later, 3, 3, "sync", 4)
    np.testing.assert_array_equal(out.targets, to_frames(points, plain[1], times, (2,))[:4])


def test_async_chunk_is_rebuilt_after_the_latency_and_padded() -> None:
    """Observed at history[1], arriving 2 frames later: the rebuild from history[-1] at chunk time 2; a chunk
    shorter than e repeats its last target, counted as stalls."""
    times = point_frames(20, 10, "state", 2.0)
    points, start = _wiggle(9), np.zeros(3)
    history = [start, np.full(3, 0.01), np.full(3, 0.02), np.full(3, 0.03)]
    out = chunk_targets(points, times, (2,), None, history, 2, 2, "async", 4)
    np.testing.assert_array_equal(out.targets, after_latency(points, times, history[-1], 2.0, (2,))[:4])
    short = chunk_targets(points, times, (2,), None, history, 2, 2, "async", 12)
    print(f"chunk of {len(after_latency(points, times, history[-1], 2.0, (2,)))} targets after the arrival, "
          f"e 12: {short.stalls} stalls")  # fmt: skip
    assert len(short.targets) == 12 and short.stalls == 12 - 8
    np.testing.assert_array_equal(short.targets[8:], np.repeat(short.targets[7:8], 4, axis=0))


def test_owed_hold_leads_the_next_chunk_less_the_wait() -> None:
    """A switch commanded at the last frame before the observation still owes all 5 further hold frames: an
    async chunk arriving at once starts with them; a sync chunk, having waited 3 frames, with 2."""
    times = point_frames(20, 10, "state", 2.0)
    timing = _timing(1e6, rise=0.5)
    points = _wiggle(10)
    points[:, 2] = 1.0
    off, on = np.zeros(3), np.array([0.0, 0.0, 1.0])
    out = chunk_targets(points, times, (2,), timing, [off, on], 2, 0, "async", 8)
    np.testing.assert_array_equal(out.targets[:5], np.repeat(on[None, :], 5, axis=0))
    assert not np.array_equal(out.targets[5], on)
    out = chunk_targets(points, times, (2,), timing, [off, on], 2, 3, "sync", 8)
    np.testing.assert_array_equal(out.targets[:2], np.repeat(on[None, :], 2, axis=0))
    assert not np.array_equal(out.targets[2], on)


def test_interpolator_chunk_targets_matches_the_function(tmp_path: Path) -> None:
    config = ResampleConfig(metric="time", span_frames=20, action_horizon=10, anchor="state")
    interp = ArcInterpolator.from_report(_report(tmp_path / "arc_resample.json", config), 2.0, _timing(0.5, 0.3))
    points, history = _wiggle(11), [np.zeros(3), np.full(3, 0.01)]
    padded = np.concatenate([points, np.zeros((10, 29))], axis=1)
    a = interp.chunk_targets(padded, history, 1, 1, "async", 5)
    b = chunk_targets(points, interp.frames(), (2,), interp.timing, history, 1, 1, "async", 5)
    np.testing.assert_array_equal(a.targets, b.targets)
    assert a.stalls == b.stalls
