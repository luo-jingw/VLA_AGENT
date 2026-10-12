"""Settings of a resampled action label. Configuration only; recorded verbatim with every built dataset."""

from __future__ import annotations

import dataclasses
from typing import Literal

Metric = Literal["time", "arc", "curvature"]
Anchor = Literal["state", "action"]


@dataclasses.dataclass(frozen=True)
class ResampleConfig:
    """
    metric          progress per frame outside keyframe windows: time = 1 (uniform stride, exact timing at
                    deployment), arc = ||dq||, curvature = |d2q|^(1/3)
    span_frames     recorded frames one chunk of `action_horizon` points reaches across
    action_horizon  points per chunk; must equal the model's action_horizon
    event_density   point density inside keyframe windows relative to outside, spread evenly in time;
                    0 leaves the windows to the metric, 1 matches the surroundings, 2 doubles them.
                    Below 1 the window is sparser than its surroundings and the server compresses it
    key_weight      loss multiplier on points drawn from keyframe windows; independent of density
    pad_weight      loss multiplier on points past the end of the episode
    deadzone        metric below this counts as standing still
    anchor          state: a chunk starts at the current state's progress; point k sits k*span/H frames
                    after the observation, and time with span = H is exactly actions[t : t+H].
                    action: one frame later (1 + k*span/H); datasets built before the field existed
    """

    metric: Metric
    span_frames: int
    action_horizon: int
    event_density: float = 0.0
    key_weight: float = 1.0
    pad_weight: float = 1.0
    deadzone: float = 0.0
    anchor: Anchor = "state"

    @property
    def frames_per_step(self) -> float:
        return self.span_frames / self.action_horizon

    def to_dict(self) -> dict[str, float | int | str]:
        return dataclasses.asdict(self)

    @staticmethod
    def from_dict(raw: dict[str, float | int | str]) -> ResampleConfig:
        """Read a recorded config. `anchor` is required: a default would silently shift every point."""
        if "anchor" not in raw:
            raise ValueError("config has no 'anchor'; datasets built before the field existed used 'action'")
        names = {field.name for field in dataclasses.fields(ResampleConfig)}
        return ResampleConfig(**{k: v for k, v in raw.items() if k in names})  # type: ignore[arg-type]
