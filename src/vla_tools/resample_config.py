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
    span_frames     recorded frames each frame's window covers: the `action_horizon` points are placed by
                    the metric inside the span frames that follow the frame's state
    action_horizon  points per chunk; must equal the model's action_horizon
    event_density   point density inside keyframe windows, in points per span/H recorded frames, spread
                    evenly in time whatever the rest of the window holds; 0 leaves the windows to the
                    metric. The free segments of a window keep one point per span/H frames on average, so
                    above 1 a window holding keyframe frames covers fewer than span frames (the server
                    slows down there), below 1 more
    key_weight      loss multiplier on points drawn from keyframe windows; independent of density
    pad_weight      loss multiplier on points past the end of the episode
    deadzone        metric below this counts as standing still
    anchor          state: a frame's window starts at its current state (action row t-1), and time with
                    span = H is exactly actions[t : t+H]. action: one row later; datasets built before the
                    field existed
    curvature_length  curvature: joint-space arc length (rad) each chord spans on either side of a recorded
                    row; a fixed length, so the curvature does not depend on the teleop speed. Required
                    for `curvature`
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
    curvature_length: float | None = None
    curvature_floor: float = 0.0

    def __post_init__(self) -> None:
        if self.metric == "curvature" and (self.curvature_length is None or self.curvature_length <= 0.0):
            raise ValueError(f"metric 'curvature' needs a positive curvature_length (rad), got {self.curvature_length}")

    @property
    def frames_per_step(self) -> float:
        return self.span_frames / self.action_horizon

    def to_dict(self) -> dict[str, float | int | str | None]:
        return dataclasses.asdict(self)

    @staticmethod
    def from_dict(raw: dict[str, float | int | str | None]) -> ResampleConfig:
        """Read a recorded config. `anchor` is required: a default would silently shift every point.
        A `curvature` config without `curvature_length` was recorded either with a chord window relative to
        each episode's median step (`curvature_window`) or with the second-difference metric."""
        if "anchor" not in raw:
            raise ValueError("config has no 'anchor'; datasets built before the field existed used 'action'")
        allowed = typing.get_args(Metric)
        if raw.get("metric") not in allowed:
            raise ValueError(f"unknown metric {raw.get('metric')!r}; allowed: {', '.join(allowed)}")
        if raw["metric"] == "curvature" and raw.get("curvature_length") is None:
            if "curvature_window" in raw:
                raise ValueError(
                    "config has metric 'curvature' with 'curvature_window', a chord window relative to each "
                    "episode's median step, and an episode-wide step; rebuild the labels with curvature_length"
                )
            raise ValueError(
                "config has metric 'curvature' but no 'curvature_length': it was recorded with the previous "
                "second-difference metric, now named 'second_difference'"
            )
        names = {field.name for field in dataclasses.fields(ResampleConfig)}
        return ResampleConfig(**{k: v for k, v in raw.items() if k in names})  # type: ignore[arg-type]
