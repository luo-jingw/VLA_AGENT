import dataclasses
from pathlib import Path

import numpy as np
import pytest

from vla_tools.episodes import Episode
from vla_tools.keyframe_record import KeyframeRecord
from vla_tools.rate_ceiling import Case
from vla_tools.rate_ceiling import EpisodeJob
from vla_tools.rate_ceiling import check
from vla_tools.rate_ceiling import episode_checks
from vla_tools.rate_ceiling import execution_lengths
from vla_tools.rate_ceiling import summarize
from vla_tools.resample_config import ResampleConfig
from vla_tools.robot_spec import RobotSpec

FPS = 10.0
STEP = 0.01  # rad per recorded frame
RECORD = KeyframeRecord(dataset=Path("."), episodes={}, open_issues=0)
PER_FRAME = ResampleConfig(metric="time", span_frames=10, action_horizon=10)


def _spec(velocity: float, acceleration: float) -> RobotSpec:
    return RobotSpec(action_dim=2, joint_dims=(0, 1), command_dims=(), state_dims=(0, 1), fps=FPS,
                     joint_velocity_limits=(velocity, velocity), joint_acceleration_limits=(acceleration, acceleration))  # fmt: skip


def _profile(frames: int, distance: float) -> Episode:
    """Joint 0 follows a minimum-jerk profile from rest to rest over `distance`, joint 1 stands still;
    state[t] = action[t - 1]."""
    u = np.linspace(0.0, 1.0, frames + 1)
    s = 10 * u**3 - 15 * u**4 + 6 * u**5
    q = np.stack([distance * s, np.zeros(frames + 1)], axis=1)
    return Episode(index=0, path=Path("."), states=q[:-1], actions=q[1:], padding=np.zeros(frames, dtype=bool))


def test_robot_spec_rejects_limits_of_the_wrong_length() -> None:
    with pytest.raises(ValueError, match="one positive entry per joint dim"):
        RobotSpec(
            action_dim=2, joint_dims=(0, 1), command_dims=(), state_dims=(0, 1), fps=FPS, joint_velocity_limits=(1.0,)
        )


def test_velocity_and_acceleration_scale_with_the_rate() -> None:
    e = _profile(200, 2.0)
    step = np.abs(np.diff(e.actions[:, 0])).max() * FPS
    accel = np.abs(np.diff(e.actions[:, 0], n=2)).max() * FPS**2
    spec = _spec(velocity=1.0, acceleration=1.0)
    for rate in (1.0, 2.0, 3.0):
        c = check(e, spec, RECORD, PER_FRAME, rate, 3, 0, "async", STEP)
        assert c.velocity_ratio == pytest.approx(rate * step, rel=1e-2)
        assert c.acceleration_ratio == pytest.approx(rate**2 * accel, rel=5e-2)
        assert c.wall_frames == pytest.approx(e.real / rate, abs=3)


def test_largest_rate_within_the_velocity_limit() -> None:
    e = _profile(200, 2.0)
    step = np.abs(np.diff(e.actions[:, 0])).max() * FPS
    spec = _spec(velocity=3.5 * step, acceleration=1e9)
    case = Case(config=PER_FRAME, latency_s=0.0, mode="async")
    rates = (1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
    checks = episode_checks(EpisodeJob(e, spec, RECORD, case, rates, STEP))
    result = summarize(case, checks, [0], FPS, rates)
    assert result.rate == 3.0
    assert result.speedup == pytest.approx(3.0, rel=0.05)
    assert result.next_rate_failures == {"velocity": 1}


def test_latency_caps_the_rate_of_short_chunks() -> None:
    """2 frames of latency, no binding velocity limit: the largest rate is the last one whose chunk keeps a
    target after the latency, span 10: rate 3 (floor(10 / 3) = 3 targets, 1 left; rate 4 leaves none), span 40:
    rate 12. There a chunk runs out every other frame and the robot holds: the speed-up stays below the rate."""
    e = _profile(400, 4.0)
    spec = _spec(velocity=1e9, acceleration=2.0)
    rates = (1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0)
    found = {}
    for span in (10, 40):
        config = dataclasses.replace(PER_FRAME, span_frames=span)
        case = Case(config=config, latency_s=0.2, mode="async")
        result = summarize(case, episode_checks(EpisodeJob(e, spec, RECORD, case, rates, STEP)), [0], FPS, rates)
        speedups = {row["rate"]: round(float(row["speedup"]), 2) for row in result.by_rate}
        print(f"span {span}: speed-up by rate {speedups}")
        found[span] = (result.rate, result.next_rate_failures)
        assert result.stall_frames > 2  # holds beyond the first inference
    assert found == {10: (3.0, {"latency": 1}), 40: (12.0, {"latency": 1})}


def test_execution_lengths() -> None:
    config = dataclasses.replace(PER_FRAME, span_frames=40)
    assert execution_lengths(config, 2.0, 3, "async") == [3, 8, 17]
    assert execution_lengths(config, 2.0, 3, "sync") == [10, 20]
    assert execution_lengths(PER_FRAME, 3.0, 2, "async") == [2]  # 1 target left after the latency: runs out
    assert execution_lengths(PER_FRAME, 4.0, 2, "async") == []  # none left
    assert execution_lengths(PER_FRAME, 12.0, 0, "sync") == []
