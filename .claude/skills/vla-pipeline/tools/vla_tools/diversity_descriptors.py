"""Per-episode descriptors for diversity and coverage analysis. Pure extraction from arrays and images.

Each descriptor group answers one question about how episodes differ:

  start_pose          where the arm starts (joint state at frame 0)
  pose@<keyframe>     where the arm is at a keyframe that occurs once per episode -- at a grasp this is
                      a proxy for where the object was, at a release for where it was placed
  static_pose         the mean pose of joints that never move within an episode; differences here are
                      setup or session differences (a parked arm, a mounting change)
  motion              length in frames and joint-space path length
  image:<camera>      the first frame as a small grayscale thumbnail: scene layout and lighting
"""

from __future__ import annotations

import dataclasses

import numpy as np
from PIL import Image

from vla_tools.episodes import Episode
from vla_tools.robot_spec import RobotSpec

THUMBNAIL = (24, 18)


@dataclasses.dataclass(frozen=True)
class Descriptors:
    episode: int
    groups: dict[str, np.ndarray]  # group name -> 1-D vector


def static_dims(episodes: list[Episode], spec: RobotSpec, threshold: float) -> list[int]:
    return [d for d in spec.joint_dims if max(np.ptp(e.actions[:, d]) for e in episodes) < threshold]


def thumbnail(image: np.ndarray) -> np.ndarray:
    gray = Image.fromarray(image).convert("L").resize(THUMBNAIL, Image.BILINEAR)
    values = np.asarray(gray, dtype=np.float64).ravel()
    return (values - values.mean()) / (values.std() + 1e-9)


def describe(
    e: Episode,
    spec: RobotSpec,
    keyframe_frames: dict[str, int],
    static: list[int],
    first_images: dict[str, np.ndarray],
) -> Descriptors:
    joint_states = list(spec.joint_state_dims)
    moving = [s for d, s in zip(spec.joint_dims, joint_states, strict=True) if d not in static]
    groups: dict[str, np.ndarray] = {"start_pose": e.states[0, moving]}
    for name, frame in keyframe_frames.items():
        groups[f"pose@{name}"] = e.states[min(frame + 1, len(e.states) - 1), moving]
    if static:
        groups["static_pose"] = e.states[:, [spec.state_dims[d] for d in static]].mean(axis=0)
    path = float(np.linalg.norm(np.diff(e.actions[: e.real, list(spec.joint_dims)], axis=0), axis=1).sum())
    groups["motion"] = np.array([float(e.real), path])
    for camera, image in first_images.items():
        groups[f"image:{camera}"] = thumbnail(image)
    return Descriptors(episode=e.index, groups=groups)
