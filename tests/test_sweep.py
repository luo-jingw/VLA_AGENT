from pathlib import Path

import numpy as np

from vla_tools.episodes import Episode
from vla_tools.keyframe_record import KeyframeRecord
from vla_tools.resample_config import ResampleConfig
from vla_tools.robot_spec import RobotSpec
from vla_tools.sweep import PathScale
from vla_tools.sweep import common_frames
from vla_tools.sweep import evaluate
from vla_tools.sweep import unpadded_frames

SPEC = RobotSpec(action_dim=3, joint_dims=(0, 1), command_dims=(2,), state_dims=(0, 1, 2), fps=10.0)
RECORD = KeyframeRecord(dataset=Path("."), episodes={}, open_issues=0)


def _episode(index: int, frames: int, padding: int) -> Episode:
    """A planar arc with a varying speed, the gripper closed in the middle third, and `padding` end-padding rows
    that repeat the last row."""
    u = np.linspace(0.0, 1.0, frames + 1)
    phase = u + 0.15 * np.sin(2 * np.pi * u)
    path = np.stack([np.cos(2.0 * phase), np.sin(2.0 * phase)], axis=1)
    k = np.arange(frames + 1)
    command = ((k >= frames // 3) & (k < 2 * frames // 3)).astype(np.float64)
    rows = np.concatenate([path, command[:, None]], axis=1)
    rows = np.concatenate([rows, np.repeat(rows[-1:], padding, axis=0)])
    pad = np.zeros(frames + padding, dtype=bool)
    pad[frames:] = True
    return Episode(index=index, path=Path("."), states=rows[:-1], actions=rows[1:], padding=pad)


def test_cells_of_one_span_are_scored_on_the_same_frames() -> None:
    episodes = [_episode(0, 90, 40), _episode(1, 130, 40)]
    configs = [
        ResampleConfig(metric=metric, span_frames=40, action_horizon=10, event_density=density,  # type: ignore[arg-type]
                       curvature_length=0.05 if metric == "curvature" else None)
        for metric in ("time", "curvature")
        for density in (0.0, 2.0)
    ]  # fmt: skip
    per_cell = [unpadded_frames(episodes, SPEC, RECORD, c, 2) for c in configs]
    common = common_frames(per_cell)
    own = [sum(len(f) for f in cell.values()) for cell in per_cell]
    shared = sum(len(f) for f in common.values())
    print(f"frames each cell leaves unpadded: {own}; common: {shared}")
    for cell in per_cell:
        assert all(set(common[e.index]) <= cell[e.index] for e in episodes)
    scale = PathScale.from_episodes(episodes, SPEC)
    cells = [evaluate(episodes, SPEC, RECORD, c, 5, common, scale) for c in configs]
    assert [c.frames for c in cells] == [shared] * len(configs)
