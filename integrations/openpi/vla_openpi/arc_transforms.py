"""Robot-agnostic openpi transforms for datasets carrying a resampled label (`action_chunk`, `action_loss_weight`).

ArcInputs   robot observation -> the model's Observation schema; in training also reshapes the flat
            label to (H, D) and passes the per-point loss weight through
RawOutputs  the model's points, trimmed to the robot's action dims (training, evaluation of points)
ArcOutputs  the model's points -> one target per control frame, via vla_tools.arc_interp (serving)

The transforms sit where a robot-specific `*_policy.py` Inputs/Outputs pair would: after the repack
transform, before DeltaActions / after AbsoluteActions.
"""

from __future__ import annotations

import dataclasses

import einops
import numpy as np
from openpi import transforms
from vla_tools.arc_interp import ArcInterpolator

MODEL_CAMERAS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")


def parse_image(image: np.ndarray) -> np.ndarray:
    """uint8 HWC either way: LeRobot yields float32 CHW in [0, 1], a robot client sends uint8 HWC."""
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.ndim == 3 and image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class ArcInputs(transforms.DataTransformFn):
    """Map repacked keys onto the model's inputs.

    cameras         (repacked key, model camera) pairs; model cameras are base_0_rgb, left_wrist_0_rgb,
                    right_wrist_0_rgb. A model camera with no source gets a black image and a False mask
    action_dim      D, the robot's action dims
    action_horizon  H; must equal the model's and the dataset's (checked against the label length)
    """

    cameras: tuple[tuple[str, str], ...]
    action_dim: int
    action_horizon: int
    state_key: str = "observation/state"

    def __call__(self, data: dict) -> dict:
        images = {camera: parse_image(data[key]) for key, camera in self.cameras}
        if not images:
            raise ValueError("ArcInputs needs at least one camera")
        blank = np.zeros_like(next(iter(images.values())))
        inputs = {
            "state": np.asarray(data[self.state_key], dtype=np.float32),
            "image": {camera: images.get(camera, blank) for camera in MODEL_CAMERAS},
            "image_mask": {camera: np.bool_(camera in images) for camera in MODEL_CAMERAS},
        }
        if "actions" in data:
            chunk = np.asarray(data["actions"], dtype=np.float32)
            expected = self.action_horizon * self.action_dim
            if chunk.shape[-1] != expected:
                raise ValueError(
                    f"action_chunk has {chunk.shape[-1]} values, expected {self.action_horizon} x {self.action_dim}; "
                    "the dataset was built with another action_horizon"
                )
            inputs["actions"] = chunk.reshape(*chunk.shape[:-1], self.action_horizon, self.action_dim)
        if "action_loss_weight" in data:
            inputs["action_loss_weight"] = data["action_loss_weight"]
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        return inputs


@dataclasses.dataclass(frozen=True)
class RawOutputs(transforms.DataTransformFn):
    """The model's (H, D) absolute points."""

    action_dim: int

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, : self.action_dim])}


@dataclasses.dataclass(frozen=True)
class ArcOutputs(transforms.DataTransformFn):
    """Per-frame targets for the robot, plus the points and the frames they were placed at.

    actions[j] is the target for frame t + j + 1 after the observation -- the meaning a per-frame
    policy's actions have -- so an existing client works unchanged.
    """

    interp: ArcInterpolator

    def __call__(self, data: dict) -> dict:
        points = np.asarray(data["actions"][:, : self.interp.action_dim])
        return {
            "actions": self.interp(points, data["state"]).astype(np.float32),
            "arc_points": points.astype(np.float32),
            "arc_point_frames": self.interp.frames().astype(np.float32),
        }
