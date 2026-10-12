"""What one robot's vectors mean in a LeRobot dataset. Configuration only."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np


@dataclasses.dataclass(frozen=True)
class RobotSpec:
    """Layout of a robot's action and state vectors.

    joint_dims     action dims that are continuous positions: measured, interpolated, cleaned
    command_dims   action dims that are binary commands (gripper close, suction on); never
                   interpolated. Continuous effectors (gripper width, finger joints) are joint dims
    state_dims     for every action dim, the state dim holding the same quantity, or -1 when the
                   state has none. The current pose a chunk starts from is read through it
    image_keys     camera columns, used by visual review and image descriptors
    """

    action_dim: int
    joint_dims: tuple[int, ...]
    command_dims: tuple[int, ...]
    state_dims: tuple[int, ...]
    fps: float
    action_key: str = "action"
    state_key: str = "observation.state"
    image_keys: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        dims = sorted((*self.joint_dims, *self.command_dims))
        if dims != list(range(self.action_dim)):
            raise ValueError(f"joint_dims + command_dims must cover 0..{self.action_dim - 1} exactly once")
        if len(self.state_dims) != self.action_dim:
            raise ValueError(f"state_dims needs one entry per action dim ({self.action_dim})")
        if any(self.state_dims[d] < 0 for d in self.joint_dims):
            raise ValueError("every joint dim needs a state counterpart")

    @property
    def joint_state_dims(self) -> tuple[int, ...]:
        return tuple(self.state_dims[d] for d in self.joint_dims)

    def current_pose(self, state: np.ndarray) -> np.ndarray:
        """A state vector in the action layout; command dims without a state counterpart are 0."""
        return np.array([state[s] if s >= 0 else 0.0 for s in self.state_dims], dtype=np.float64)

    @staticmethod
    def load(path: Path) -> RobotSpec:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return RobotSpec(
            action_dim=int(raw["action_dim"]),
            joint_dims=tuple(int(d) for d in raw["joint_dims"]),
            command_dims=tuple(int(d) for d in raw["command_dims"]),
            state_dims=tuple(int(d) for d in raw["state_dims"]),
            fps=float(raw["fps"]),
            action_key=str(raw.get("action_key", "action")),
            state_key=str(raw.get("state_key", "observation.state")),
            image_keys=tuple(str(k) for k in raw.get("image_keys", [])),
        )
