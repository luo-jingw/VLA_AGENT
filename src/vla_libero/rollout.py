"""One closed-loop episode in a simulator: observe, infer, execute the chunk's targets one per control frame, with
inference latency emulated in simulated time.

After `settle_frames` holding the initial pose (objects come to rest), the policy is queried on the latest
observation. Each chunk's targets come from `ArcInterpolator.chunk_targets` on the commanded history, as in
the label replay: in sync mode, and for the first chunk, the robot holds the observed pose for
`latency_frames` before the chunk's targets; in async mode it executes the chunk up to `latency_frames`
before its end, is observed, and executes the rest while the next inference is in flight, whose chunk is
rebuilt after the latency. The simulator does not advance during an inference, so the latency is exact
whatever the inference takes. The episode ends at the first success or after `budget` control frames
(settling excluded).

Failure phase from the object's height above its height after settling: not lifted (never 5 cm), dropped
(lifted, below 2 cm at the end), not placed (lifted, still up).
"""

from __future__ import annotations

import dataclasses
import time

import numpy as np

from vla_libero.interfaces import ChunkPolicy
from vla_libero.interfaces import Observation
from vla_libero.interfaces import SimEnv
from vla_tools.arc_interp import ArcInterpolator
from vla_tools.arc_interp import Mode

LIFTED = 0.05  # m
DOWN = 0.02  # m


@dataclasses.dataclass(frozen=True)
class Trial:
    trial: int  # index of the initial state
    success: bool
    completion_frames: int | None  # control frames after settling until the first success
    phase: str  # success, not lifted, dropped, not placed
    frames: int  # control frames stepped after settling
    inferences: int
    stall_frames: int  # frames held for latency or repeating a chunk's last target
    lift_max: float  # m
    infer_seconds: float  # median wall-clock time of one inference


class _Episode:
    """The commanded history and the frame count of one episode."""

    def __init__(self, env: SimEnv, observation: Observation, budget: int) -> None:
        self.env = env
        self.observation = observation
        self.history = [observation.state.astype(np.float64)]
        self.frames = 0
        self.budget = budget
        self.base = env.object_height()
        self.lift_max = 0.0
        self.lift = 0.0
        self.success = False

    def step(self, target: np.ndarray) -> bool:
        """Command one target; True when the episode is over."""
        target = np.asarray(target, dtype=np.float64)
        self.observation = self.env.step(target)
        self.history.append(target)
        self.frames += 1
        self.lift = self.env.object_height() - self.base
        self.lift_max = max(self.lift_max, self.lift)
        self.success = self.env.success()
        return self.success or self.frames >= self.budget


def rollout(
    env: SimEnv, policy: ChunkPolicy, interp: ArcInterpolator, prompt: str, initial_state: np.ndarray, *,
    execute: int, latency_frames: int, mode: Mode, settle_frames: int, budget: int, trial: int = 0,
) -> Trial:  # fmt: skip
    if execute < 1 or latency_frames < 0 or (mode == "async" and execute < latency_frames):
        raise ValueError(
            f"need execute >= 1, latency >= 0 and in async execute >= latency; got {execute}, {latency_frames}"
        )
    observation = env.reset(initial_state)
    start = observation.state.astype(np.float64)
    for _ in range(settle_frames):
        observation = env.step(start)
    run = _Episode(env, observation, budget)
    inferences, stalls, seconds = 0, 0, []

    def infer() -> np.ndarray:
        nonlocal inferences
        began = time.perf_counter()
        points = policy.infer(run.observation, prompt)
        seconds.append(time.perf_counter() - began)
        inferences += 1
        return points

    points, observed, over = infer(), 1, False
    while not over:
        planned = interp.chunk_targets(points, run.history, observed, latency_frames, mode, execute)
        stalls += planned.stalls
        before = run.frames
        if mode == "sync" or len(run.history) == 1:
            for _ in range(latency_frames):
                stalls += 1
                if over := run.step(run.history[-1]):
                    break
        targets = planned.targets
        split = len(targets) - latency_frames if mode == "async" else len(targets)
        for target in [] if over else targets[:split]:
            if over := run.step(target):
                break
        if over:
            break
        observed = len(run.history)
        points = infer()
        for target in targets[split:]:
            if over := run.step(target):
                break
        if run.frames == before:
            raise RuntimeError("a chunk produced no target and no latency frame; the loop would not advance")
    if run.success:
        phase = "success"
    elif run.lift_max < LIFTED:
        phase = "not lifted"
    elif run.lift < DOWN:
        phase = "dropped"
    else:
        phase = "not placed"
    return Trial(trial=trial, success=run.success, completion_frames=run.frames if run.success else None, phase=phase,
                 frames=run.frames, inferences=inferences, stall_frames=stalls, lift_max=run.lift_max,
                 infer_seconds=float(np.median(seconds)))  # fmt: skip
