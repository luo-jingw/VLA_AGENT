"""Settings of a resampled action label. Configuration only; recorded verbatim with every built dataset."""

from __future__ import annotations

import dataclasses
import typing
from typing import Literal

Metric = Literal["time", "curvature", "second_difference"]
Anchor = Literal["state", "action"]


@dataclasses.dataclass(frozen=True)
class ResampleConfig:
    """
    metric          progress per frame outside keyframe windows: time = 1 (uniform stride, exact timing at
                    deployment); curvature = kappa^(1/3) * ||dq||, kappa the joint path's geometric
                    curvature, independent of teleop speed; second_difference = |d2q|^(1/3), the
                    acceleration per frame, kept to reproduce datasets built with it
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
    curvature_window  curvature: arc length each chord spans on either side of a sample, in multiples of
                    the episode's median per-frame joint step (still frames excluded)
    curvature_floor   curvature: minimum kappa^(1/3) as a fraction of its length-weighted episode mean;
                    0 gives straight segments no progress
    """

    metric: Metric
    span_frames: int
    action_horizon: int
    event_density: float = 0.0
    key_weight: float = 1.0
    pad_weight: float = 1.0
    deadzone: float = 0.0
    anchor: Anchor = "state"
    curvature_window: float = 4.0
    curvature_floor: float = 0.0

    @property
    def frames_per_step(self) -> float:
        return self.span_frames / self.action_horizon

    def to_dict(self) -> dict[str, float | int | str]:
        return dataclasses.asdict(self)

    @staticmethod
    def from_dict(raw: dict[str, float | int | str]) -> ResampleConfig:
        """Read a recorded config. `anchor` is required: a default would silently shift every point.
        A `curvature` config without `curvature_window` was recorded with the second-difference metric."""
        if "anchor" not in raw:
            raise ValueError("config has no 'anchor'; datasets built before the field existed used 'action'")
        allowed = typing.get_args(Metric)
        if raw.get("metric") not in allowed:
            raise ValueError(f"unknown metric {raw.get('metric')!r}; allowed: {', '.join(allowed)}")
        if raw["metric"] == "curvature" and "curvature_window" not in raw:
            raise ValueError(
                "config has metric 'curvature' but no 'curvature_window': it was recorded with the previous "
                "second-difference metric, now named 'second_difference'"
            )
        names = {field.name for field in dataclasses.fields(ResampleConfig)}
        return ResampleConfig(**{k: v for k, v in raw.items() if k in names})  # type: ignore[arg-type]
