from pathlib import Path

import numpy as np
import pytest

from vla_tools.arc_interp import VelocityTiming
from vla_tools.episodes import Episode
from vla_tools.keyframe_record import KeyframeRecord
from vla_tools.label_replay import closed_loop
from vla_tools.label_replay import completion_frames
from vla_tools.label_replay import edge_check
from vla_tools.resample_config import ResampleConfig
from vla_tools.robot_spec import RobotSpec

SPEC = RobotSpec(action_dim=3, joint_dims=(0, 1), command_dims=(2,), state_dims=(0, 1, 2), fps=10.0)
RECORD = KeyframeRecord(dataset=Path("."), episodes={}, open_issues=0)
PER_FRAME = ResampleConfig(metric="time", span_frames=10, action_horizon=10)
TOLERANCE = 0.005  # rad, final tolerance; the recorded steps are about 0.025


def _episode(frames: int = 80, close: int = 30, release: int = 55) -> Episode:
    """A planar arc with a varying speed; the gripper command closes at `close` and opens at `release`.
    state[t] is action[t - 1] (state[0] continues the motion backwards), as anchor "state" assumes."""
    u = np.linspace(0.0, 1.0, frames + 1)
    phase = u + 0.15 * np.sin(2 * np.pi * u)  # speed changes along the path
    path = np.stack([np.cos(2.0 * phase), np.sin(2.0 * phase)], axis=1)
    command = ((np.arange(frames + 1) >= close) & (np.arange(frames + 1) < release)).astype(np.float64)
    rows = np.concatenate([path, command[:, None]], axis=1)
    actions, states = rows[1:], rows[:-1]
    return Episode(index=0, path=Path("."), states=states, actions=actions, padding=np.zeros(frames, dtype=bool))


def test_zero_latency_async_is_sync() -> None:
    e = _episode()
    config = ResampleConfig(metric="time", span_frames=40, action_horizon=10, event_density=2.0)
    for execute in (1, 4, 10):
        a = closed_loop(e, SPEC, RECORD, config, execute, 1.0, 0, "sync", final_tolerance=TOLERANCE)
        b = closed_loop(e, SPEC, RECORD, config, execute, 1.0, 0, "async", final_tolerance=TOLERANCE)
        assert np.array_equal(a.executed, b.executed)
        assert a.observations == b.observations
        assert a.stall_frames == b.stall_frames == 0


def test_per_frame_labels_execute_every_recorded_action() -> None:
    """At rate 1, e = 1 and no latency the observation follows the recording one frame at a time; the chunk of
    the last recorded frame is executed, so the run is the recorded actions and completes when they do."""
    e = _episode()
    run = closed_loop(e, SPEC, RECORD, PER_FRAME, 1, 1.0, final_tolerance=TOLERANCE)
    recorded = completion_frames(e.actions, e.actions[-1], SPEC, TOLERANCE)
    print(f"{len(run.executed)} executed, completion {run.completion_frames}, recorded completion {recorded}")
    assert np.allclose(run.executed, e.actions, atol=1e-12)
    assert run.observations == list(range(e.real))
    assert run.completion_frames == recorded


def test_sparse_chunks_keep_observing_the_last_frame_until_at_the_final_row() -> None:
    """Span 40 at e = 1: a chunk executes one target of a spline whose first point lies 4 frames ahead, so at the
    last recorded frame the robot is still short of the final row and observes that frame again until it is
    within the final tolerance. Completion stops counting there; nothing is held after it."""
    e = _episode()
    config = ResampleConfig(metric="time", span_frames=40, action_horizon=10)
    for latency, mode in ((0, "sync"), (2, "async"), (2, "sync")):
        run = closed_loop(e, SPEC, RECORD, config, max(latency, 1), 1.0, latency, mode, final_tolerance=TOLERANCE)
        repeats = run.observations.count(e.real - 1)
        print(f"d {latency} {mode}: {len(run.observations)} inferences, last frame observed {repeats} times, "
              f"{len(run.executed)} executed, completion {run.completion_frames}")  # fmt: skip
        assert run.observations[-repeats:] == [e.real - 1] * repeats and repeats > 1
        assert np.linalg.norm(run.executed[-1, :2] - e.actions[-1, :2]) <= TOLERANCE
        assert run.completion_frames <= len(run.executed)


def test_async_latency_skips_the_passed_targets_of_each_chunk() -> None:
    """Per-frame labels are the recording: with latency d each arriving chunk is rebuilt from the commanded pose
    through its points not yet passed, so after the d frames held for the first inference the executed targets
    are the recorded rows without a seam."""
    e = _episode()
    for latency, execute in ((2, 2), (2, 5), (3, 7)):
        run = closed_loop(e, SPEC, RECORD, PER_FRAME, execute, 1.0, latency, "async", final_tolerance=TOLERANCE)
        held = run.executed[:latency]
        assert np.array_equal(held, np.repeat(e.states[:1], latency, axis=0))
        moving = run.executed[latency:]
        n = min(len(moving), len(e.actions))
        assert np.allclose(moving[:n], e.actions[:n], atol=1e-12)
        assert np.allclose(moving[n:], e.actions[-1], atol=1e-12)  # labels past the end hold the last row
        assert run.stall_frames == latency


def test_sync_latency_holds_at_every_observation() -> None:
    e = _episode()
    latency, execute = 2, 5
    run = closed_loop(e, SPEC, RECORD, PER_FRAME, execute, 1.0, latency, "sync", final_tolerance=TOLERANCE)
    assert run.stall_frames == latency * len(run.observations)
    assert len(run.executed) == (latency + execute) * len(run.observations)
    moving = np.concatenate([run.executed[k * 7 + 2 : k * 7 + 7] for k in range(len(run.observations))])
    n = min(len(moving), len(e.actions))
    assert np.allclose(moving[:n], e.actions[:n], atol=1e-12)


def test_async_sparse_chunks_switch_continuously() -> None:
    """Sparse labels arriving d frames late: each chunk is rebuilt from the commanded pose, so a switch is one
    step of the spline (here up to 1.2 of the recording's largest step), not a jump to a target planned from an
    older pose."""
    e = _episode()
    config = ResampleConfig(metric="time", span_frames=40, action_horizon=10, event_density=2.0)
    largest = float(np.linalg.norm(np.diff(e.actions[:, :2], axis=0), axis=1).max())
    for latency, execute in ((2, 4), (3, 6), (4, 12)):
        run = closed_loop(e, SPEC, RECORD, config, execute, 1.0, latency, "async", final_tolerance=TOLERANCE)
        print(f"d {latency} e {execute}: largest seam {max(run.seams) / largest:.2f} recorded steps")
        assert max(run.seams) <= 1.5 * largest


def test_async_chunk_that_runs_out_holds_its_last_target() -> None:
    """Rate 2.5: points at 0.4, 0.8, ..., 4.0 frames; arriving 2 frames late a chunk keeps the points at 2.4 to 4.0,
    i.e. targets for 2 frames; with e = 3 the third is its last target, held."""
    e = _episode()
    run = closed_loop(e, SPEC, RECORD, PER_FRAME, 3, 2.5, 2, "async", final_tolerance=TOLERANCE)
    assert run.stall_frames == 2 + (len(run.observations) - 1)
    later = run.executed[5:].reshape(-1, 3, 3)
    assert np.array_equal(later[:, 1], later[:, 2])


def test_chunk_spent_by_its_arrival_is_refused() -> None:
    e = _episode()
    with pytest.raises(ValueError, match="keeps no target"):
        closed_loop(e, SPEC, RECORD, PER_FRAME, 2, 4.0, 2, "async", final_tolerance=TOLERANCE)  # points up to 2.5
        # frames, latency 2


def test_completion_counts_frames_until_the_final_row_is_reached_for_good() -> None:
    rows = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 1.0], [1.0, 0.0, 1.0]])
    final = np.array([1.0, 0.0, 1.0])
    assert completion_frames(rows, final, SPEC, 0.1) == 4  # row 1 is at the joints with the other command
    assert completion_frames(rows[:2], rows[1], SPEC, 0.1) == 2
    with pytest.raises(ValueError, match="not at the final row"):
        completion_frames(rows[:3], final, SPEC, 0.1)


def test_velocity_timing_with_latency_stays_within_the_limit_and_holds_at_switches() -> None:
    """Span 40 at cap 4 would move up to 1.6 rad/s; timed at a 0.5 rad/s limit, arriving 2 frames late, the
    executed targets (from the observed pose) stay within it, and the arm stays at least 1 + 5 frames at the
    close and 1 + 2 at the open (0.5 s and 0.2 s at 10 fps)."""
    e = _episode()
    config = ResampleConfig(metric="time", span_frames=40, action_horizon=10)
    timing = VelocityTiming(velocity_limits=(0.5, 0.5), fraction=1.0, fps=10.0, rise_hold_s=0.5, fall_hold_s=0.2)
    for execute in (2, 4, 8):
        run = closed_loop(e, SPEC, RECORD, config, execute, 4.0, 2, "async", final_tolerance=TOLERANCE, timing=timing)
        q = np.concatenate([e.states[:1, :2], run.executed[:, :2]])
        ratio = float(np.abs(np.diff(q, axis=0)).max() * 10.0 / 0.5)
        holds = edge_check(e, SPEC, run.executed, tolerance=1e-9).holds
        print(f"e {execute}: {len(run.executed)} executed, completion {run.completion_frames}, velocity ratio "
              f"{ratio:.4f}, holds at the edges {holds}")  # fmt: skip
        assert ratio <= 1.0
        assert holds[0] >= 6 and holds[1] >= 3


def test_edge_check_counts_missed_edges_and_holds() -> None:
    e = _episode()
    run = closed_loop(e, SPEC, RECORD, PER_FRAME, 10, 1.0, final_tolerance=TOLERANCE)
    edges = edge_check(e, SPEC, run.executed, tolerance=1e-9)
    assert edges.missed == 0
    assert edges.positions == [0.0, 0.0]
    assert edges.holds == edges.recorded_holds == [1, 1]
    never = run.executed.copy()
    never[:, 2] = 0.0
    assert edge_check(e, SPEC, never, tolerance=1e-9).missed == 2
