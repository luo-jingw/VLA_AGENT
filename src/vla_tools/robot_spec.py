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
    joint_velocity_limits      per joint dim, the robot's joint velocity limit (rad/s); empty when unknown
    joint_acceleration_limits  per joint dim, the robot's joint acceleration limit (rad/s^2); empty when unknown
    switch_hold_s  seconds the effector takes to act after a command rises / falls (rise, fall), the hold
                   velocity timing keeps the arm still for at each switch; empty when not measured
    """

    action_dim: int
    joint_dims: tuple[int, ...]
    command_dims: tuple[int, ...]
    state_dims: tuple[int, ...]
    fps: float
    action_key: str = "action"
    state_key: str = "observation.state"
    image_keys: tuple[str, ...] = ()
    joint_velocity_limits: tuple[float, ...] = ()
    joint_acceleration_limits: tuple[float, ...] = ()
    switch_hold_s: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        dims = sorted((*self.joint_dims, *self.command_dims))
        if dims != list(range(self.action_dim)):
            raise ValueError(f"joint_dims + command_dims must cover 0..{self.action_dim - 1} exactly once")
        if len(self.state_dims) != self.action_dim:
            raise ValueError(f"state_dims needs one entry per action dim ({self.action_dim})")
        if any(self.state_dims[d] < 0 for d in self.joint_dims):
            raise ValueError("every joint dim needs a state counterpart")
        for name in ("joint_velocity_limits", "joint_acceleration_limits"):
            limits = getattr(self, name)
            if limits and (len(limits) != len(self.joint_dims) or min(limits) <= 0.0):
                raise ValueError(
                    f"{name} needs one positive entry per joint dim ({len(self.joint_dims)}), got {limits}"
                )
        if self.switch_hold_s and (len(self.switch_hold_s) != 2 or min(self.switch_hold_s) < 0.0):
            raise ValueError(f"switch_hold_s needs (rise, fall) seconds >= 0, got {self.switch_hold_s}")

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
            joint_velocity_limits=tuple(float(v) for v in raw.get("joint_velocity_limits", [])),
            joint_acceleration_limits=tuple(float(v) for v in raw.get("joint_acceleration_limits", [])),
            switch_hold_s=tuple(float(v) for v in raw.get("switch_hold_s", [])),
        )
