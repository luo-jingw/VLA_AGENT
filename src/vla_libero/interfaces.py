"""The boundaries of a simulated episode: what an environment reports and accepts, and what a policy answers."""

from __future__ import annotations

import dataclasses
from typing import Protocol

import numpy as np


@dataclasses.dataclass(frozen=True)
class Observation:
    agentview: np.ndarray  # (H, W, 3) uint8, rotated 180 degrees as in the converted dataset
    wrist: np.ndarray  # (H, W, 3) uint8, rotated 180 degrees
    state: np.ndarray  # (8,) measured joint positions + the gripper command in effect (0 open, 1 close)


class SimEnv(Protocol):
    def reset(self, sim_state: np.ndarray) -> Observation:
        """Load a flattened simulator state; the gripper command in effect becomes 0 (open)."""
        ...

    def step(self, target: np.ndarray) -> Observation:
        """Advance one control frame towards a (8,) target: 7 absolute joint positions and the command."""
        ...

    def success(self) -> bool: ...

    def object_height(self) -> float:
        """Height (m) of the object the task moves."""
        ...


class ChunkPolicy(Protocol):
    def infer(self, observation: Observation, prompt: str) -> np.ndarray:
        """(H, D) absolute points: 7 joint positions and the command per point."""
        ...
