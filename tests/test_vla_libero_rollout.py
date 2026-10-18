"""vla_libero.rollout with a fake environment (perfect tracking: the state is the last target) and a fake policy."""

from __future__ import annotations

import numpy as np
import pytest

from vla_libero.interfaces import Observation
from vla_libero.rollout import rollout
from vla_tools.arc_interp import ArcInterpolator
from vla_tools.arc_interp import VelocityTiming

IMAGE = np.zeros((4, 4, 3), dtype=np.uint8)


class FakeEnv:
    def __init__(
        self, success_at: int | None = None, lift: list[tuple[int, float]] | None = None, settle: int = 0
    ) -> None:
        self.success_at = success_at  # frames after settling
        self.lift = lift or []  # (from frame, height) steps
        self.settle = settle
        self.targets: list[np.ndarray] = []
        self.state = np.zeros(8)

    def reset(self, sim_state: np.ndarray) -> Observation:
        self.state = np.concatenate([sim_state[:7], [0.0]])
        self.targets = []
        return Observation(IMAGE, IMAGE, self.state.copy())

    def step(self, target: np.ndarray) -> Observation:
        self.targets.append(np.asarray(target, dtype=np.float64).copy())
        self.state = self.targets[-1].copy()
        return Observation(IMAGE, IMAGE, self.state.copy())

    def frames(self) -> int:
        return len(self.targets) - self.settle

    def success(self) -> bool:
        return self.success_at is not None and self.frames() >= self.success_at

    def object_height(self) -> float:
        heights = [h for start, h in self.lift if self.frames() >= start]
        return heights[-1] if heights else 0.0


class FakePolicy:
    """Joint 0 rises 0.01 per point from the observed pose; the command closes from inference `close_at` on."""

    def __init__(self, close_at: int | None = None) -> None:
        self.close_at = close_at
        self.observed: list[np.ndarray] = []

    def infer(self, observation: Observation, prompt: str) -> np.ndarray:
        self.observed.append(observation.state.copy())
        points = np.repeat(observation.state[None, :], 10, axis=0)
        points[:, 0] += 0.01 * np.arange(1, 11)
        closed = self.close_at is not None and len(self.observed) > self.close_at
        points[:, 7] = 1.0 if closed else 0.0
        return points


def _interp(timing: VelocityTiming | None = None) -> ArcInterpolator:
    return ArcInterpolator(span_frames=10, action_horizon=10, anchor="state", action_dim=8, command_dims=(7,),
                           state_dims=tuple(range(8)), rate=1.0, timing=timing)  # fmt: skip


def test_async_waits_for_the_first_inference_then_observes_every_e_frames() -> None:
    env, policy = FakeEnv(), FakePolicy()
    trial = rollout(env, policy, _interp(), "task", np.zeros(8), execute=4, latency_frames=2, mode="async",
                    settle_frames=0, budget=24)  # fmt: skip
    first_moves = int(np.flatnonzero(np.stack(env.targets)[:, 0] > 0)[0])
    print(f"{trial.inferences} inferences in {trial.frames} frames, first motion at frame {first_moves + 1}, "
          f"stalls {trial.stall_frames}")  # fmt: skip
    assert first_moves == 2  # two frames held at the start pose while the first inference is in flight
    assert trial.stall_frames == 2 and trial.frames == 24 and trial.phase == "not lifted"
    # Observations after 4, 8, 12, ... frames: the robot keeps moving through every later inference.
    observed_x = [o[0] for o in policy.observed[1:]]
    commanded_x = [env.targets[f - 1][0] for f in range(4, 24, 4)]
    np.testing.assert_allclose(observed_x[: len(commanded_x)], commanded_x)


def test_sync_holds_the_pose_for_the_latency_before_every_chunk() -> None:
    env, policy = FakeEnv(), FakePolicy()
    trial = rollout(env, policy, _interp(), "task", np.zeros(8), execute=4, latency_frames=2, mode="sync",
                    settle_frames=0, budget=24)  # fmt: skip
    x = np.stack(env.targets)[:, 0]
    print(f"joint 0 commanded: {np.round(x, 3).tolist()}")
    for k in range(4):  # each block: 2 frames at the previous pose, then 4 moving frames
        block = x[6 * k : 6 * k + 6]
        previous = x[6 * k - 1] if k else 0.0
        np.testing.assert_allclose(block[:2], previous)
        assert np.all(np.diff(np.concatenate([[previous], block[2:]])) > 0)
    assert trial.stall_frames == 8 and trial.inferences == 4  # the budget ends the 4th block before an inference


def test_episode_stops_at_success_and_classifies_failures() -> None:
    done = rollout(FakeEnv(success_at=7, settle=3), FakePolicy(), _interp(), "task", np.zeros(8), execute=4, latency_frames=0,
                   mode="async", settle_frames=3, budget=50)  # fmt: skip
    assert done.success and done.completion_frames == 7 and done.frames == 7 and done.phase == "success"
    up = rollout(FakeEnv(lift=[(5, 0.10)], settle=3), FakePolicy(), _interp(), "task", np.zeros(8), execute=4, latency_frames=0, mode="async",
                 settle_frames=3, budget=20)  # fmt: skip
    dropped = rollout(FakeEnv(lift=[(5, 0.10), (12, 0.0)], settle=3), FakePolicy(), _interp(), "task", np.zeros(8), execute=4, latency_frames=0, mode="async",
                      settle_frames=3, budget=20)  # fmt: skip
    print(f"lifted {up.phase} (rise {up.lift_max:.2f} m), dropped {dropped.phase}")
    assert (up.phase, up.completion_frames, up.frames) == ("not placed", None, 20)
    assert dropped.phase == "dropped"


def test_switch_hold_outlasts_short_chunks() -> None:
    """A close command in the second chunk holds round(0.5 s * 10 fps) = 5 further frames although each chunk
    executes only 2 targets."""
    timing = VelocityTiming(velocity_limits=(100.0,) * 7, fraction=1.0, fps=10.0, rise_hold_s=0.5, fall_hold_s=0.0)
    env = FakeEnv()
    rollout(env, FakePolicy(close_at=1), _interp(timing), "task", np.zeros(8), execute=2, latency_frames=0,
            mode="async", settle_frames=0, budget=20)  # fmt: skip
    targets = np.stack(env.targets)
    switch = int(np.flatnonzero(targets[:, 7] > 0.5)[0])
    run = 1
    while switch + run < len(targets) and np.array_equal(targets[switch + run], targets[switch]):
        run += 1
    print(f"close commanded at frame {switch + 1}, the switching target repeated over {run} frames")
    assert run == 6


def test_async_needs_e_at_least_the_latency() -> None:
    with pytest.raises(ValueError, match="async"):
        rollout(FakeEnv(), FakePolicy(), _interp(), "task", np.zeros(8), execute=2, latency_frames=3, mode="async",
                settle_frames=0, budget=10)  # fmt: skip
