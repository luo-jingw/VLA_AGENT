"""A task's keyframe definition: which moments matter, how each is found, and how far its window reaches.

Configuration only. Keyframes differ by task and by end effector, so nothing here assumes a gripper:
a keyframe type names a detector over a signal, optionally relative to another keyframe type.

keyframes_spec.json:
    {
      "task": "pick the red cube, place it on the plate",
      "effector": "parallel gripper, binary close command, no feedback",
      "order": ["grasp", "lift", "release"],
      "types": [
        {"name": "grasp", "description": "close command issued",
         "detector": {"type": "command_edge", "signal": {"source": "action", "dim": 6}, "direction": "rise"},
         "window": {"before": 8, "after": 5}, "expected": 1},
        {"name": "lift", "description": "arm starts moving after the grasp",
         "detector": {"type": "motion_onset", "signal": {"source": "joint_speed"}, "threshold": 0.02,
                      "hold_frames": 2, "reference": "grasp", "search_frames": 30},
         "window": {"before": 2, "after": 6}, "expected": 1}
      ]
    }

Frames are action indices: a keyframe at f is the change carried by actions[f], reached at frame f+1.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Literal

SignalSource = Literal["action", "state", "column", "joint_speed"]
DetectorType = Literal["command_edge", "threshold_cross", "motion_onset", "speed_minimum", "settle", "manual"]
Direction = Literal["rise", "fall"]


@dataclasses.dataclass(frozen=True)
class Signal:
    """A 1-D series per episode.

    action / state  one dim of the action or state vector
    column          one dim of any other dataset column, named by `name` (force/torque, gripper width
                    feedback, vacuum, a derived signal such as end-effector height written by a separate step)
    joint_speed     the largest joint change per frame, 0 at frame 0
    """

    source: SignalSource
    dim: int = -1
    name: str = ""


@dataclasses.dataclass(frozen=True)
class Detector:
    """How one keyframe type is found.

    command_edge     a binary action dim changes in `direction`
    threshold_cross  `signal` crosses `threshold` in `direction` and stays for `hold_frames`
    motion_onset     joint_speed exceeds `threshold` for `hold_frames`; needs `reference`
    speed_minimum    the slowest frame in the `search_frames` before `reference`
    settle           |change of `signal`| stays below `threshold` for `hold_frames`; needs `reference`
    manual           nothing is detected; every occurrence comes from the overrides file

    With `reference`, one occurrence is searched per occurrence of the reference type, within
    `search_frames` after it (before it, for speed_minimum).
    """

    type: DetectorType
    signal: Signal | None = None
    direction: Direction | None = None
    threshold: float | None = None
    hold_frames: int = 1
    reference: str | None = None
    search_frames: int | None = None


@dataclasses.dataclass(frozen=True)
class Window:
    """The frames around a keyframe that belong to it: [frame - before, frame + after), or up to the next
    occurrence of `until` when given (a measured window, e.g. command until vacuum seal)."""

    before: int
    after: int
    until: str | None = None


@dataclasses.dataclass(frozen=True)
class KeyframeType:
    name: str
    description: str
    detector: Detector
    window: Window
    expected: int  # occurrences per episode; -1 when it varies


@dataclasses.dataclass(frozen=True)
class KeyframeSpec:
    task: str
    effector: str
    types: tuple[KeyframeType, ...]
    order: tuple[str, ...]

    def columns(self) -> list[str]:
        """Dataset columns, beyond state and action, that some detector reads."""
        return sorted(
            {t.detector.signal.name for t in self.types if t.detector.signal and t.detector.signal.source == "column"}
        )

    def __post_init__(self) -> None:
        names = [t.name for t in self.types]
        if len(set(names)) != len(names):
            raise ValueError("keyframe type names must be unique")
        known: set[str] = set()
        for t in self.types:
            for ref in (t.detector.reference, t.window.until):
                if ref is not None and ref not in names:
                    raise ValueError(f"{t.name} refers to unknown type {ref!r}")
            if t.detector.reference is not None and t.detector.reference not in known:
                raise ValueError(f"{t.name} refers to {t.detector.reference!r}, which must be listed before it")
            known.add(t.name)
        if any(name not in names for name in self.order):
            raise ValueError("order names an unknown type")

    @staticmethod
    def load(path: Path) -> KeyframeSpec:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        types = []
        for t in raw["types"]:
            d = t["detector"]
            signal = Signal(**d["signal"]) if "signal" in d else None
            detector = Detector(**{k: v for k, v in d.items() if k != "signal"}, signal=signal)
            types.append(
                KeyframeType(
                    name=t["name"],
                    description=t.get("description", ""),
                    detector=detector,
                    window=Window(**t["window"]),
                    expected=int(t.get("expected", -1)),
                )
            )
        return KeyframeSpec(
            task=raw.get("task", ""),
            effector=raw.get("effector", ""),
            types=tuple(types),
            order=tuple(raw.get("order", [])),
        )
