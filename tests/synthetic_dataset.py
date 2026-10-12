"""A small synthetic LeRobot v2.1 dataset: a 3-joint arm with a binary gripper picking a block and placing it.

Every episode: still at the start pose, move to the block, settle, close, move to the bin, settle, open,
and stop three frames later (the recording ends right after the release). Blocks lie in one of two
regions, alternating by episode, so the data has two clusters. The index convention is the one the tools
assume: actions[t] = targets[t+1] = state[t+1].

With faults, three recording faults are injected, each at a known action index:
    isolated jump   one frame of joint 1 jumps and comes straight back
    command pulse   the gripper closes for two frames during the approach
    fast motion     joint 2 moves ten times its usual step for two frames and keeps the new level
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import datasets
import numpy as np

from vla_tools.robot_spec import RobotSpec

CAMERA = "observation.images.top"
TASK = "pick the block and place it in the bin"
SPEC = RobotSpec(
    action_dim=4,
    joint_dims=(0, 1, 2),
    command_dims=(3,),
    state_dims=(0, 1, 2, 3),
    fps=10.0,
    image_keys=(CAMERA,),
)
IMAGE_SHAPE = (24, 32, 3)
LEAD_FRAMES = 6
TRAIL_FRAMES = 3
BLOCK_REGIONS = (np.array([0.6, -0.4, 0.3]), np.array([-0.5, -0.6, 0.5]))
BIN_POSE = np.array([0.1, 0.5, -0.4])


@dataclasses.dataclass(frozen=True)
class EpisodeTruth:
    """What the generator put into one episode, in action indices."""

    length: int
    grasp: int  # action index of the close edge
    lift: int  # first action index of the move to the bin
    release: int  # action index of the open edge
    region: int  # 0 or 1: which block region


@dataclasses.dataclass(frozen=True)
class InjectedFaults:
    isolated_jump: tuple[int, int]  # (episode, action index of the glitch)
    command_pulse: tuple[int, int, int]  # (episode, start, end) action indices of the pulse
    fast_motion: tuple[int, int]  # (episode, action index of the first fast frame)


@dataclasses.dataclass(frozen=True)
class SyntheticDataset:
    root: Path
    truth: dict[int, EpisodeTruth]
    faults: InjectedFaults | None


def _move(start: np.ndarray, end: np.ndarray, frames: int) -> np.ndarray:
    """`frames` rows after `start`, ending at `end`, with a cosine speed profile."""
    s = (1.0 - np.cos(np.pi * np.arange(1, frames + 1) / frames)) / 2.0
    return start + s[:, None] * (end - start)


def _targets(rng: np.random.Generator, region: int) -> tuple[np.ndarray, EpisodeTruth]:
    start = rng.normal(0.0, 0.03, 3)
    block = BLOCK_REGIONS[region] + rng.normal(0.0, 0.02, 3)
    place = BIN_POSE + rng.normal(0.0, 0.02, 3)
    approach, carry = int(rng.integers(18, 23)), int(rng.integers(18, 23))
    joints = [
        np.repeat(start[None], LEAD_FRAMES, axis=0),
        _move(start, block, approach),
        np.repeat(block[None], 3 + 4, axis=0),  # settle 3, then 4 with the gripper closed
        _move(block, place, carry),
        np.repeat(place[None], 2 + TRAIL_FRAMES + 1, axis=0),  # settle 2, then open and hold
    ]
    q = np.concatenate(joints)
    gripper = np.zeros(len(q))
    close_row = LEAD_FRAMES + approach + 3
    open_row = len(q) - TRAIL_FRAMES - 1
    gripper[close_row:open_row] = 1.0
    targets = np.concatenate([q, gripper[:, None]], axis=1)
    truth = EpisodeTruth(
        length=len(targets) - 1,
        grasp=close_row - 1,
        lift=close_row + 4 - 1,
        release=open_row - 1,
        region=region,
    )
    return targets, truth


def _image(rng: np.random.Generator, region: int, joint0: float) -> np.ndarray:
    image = np.full(IMAGE_SHAPE, 100, dtype=np.int16) + rng.integers(-3, 4, IMAGE_SHAPE)
    col = 6 if region == 0 else 22
    image[8:12, col : col + 4] = (200, 40, 40)
    arm = int(np.clip(16 + 10 * joint0, 1, 30))
    image[18:22, arm - 1 : arm + 1] = (40, 40, 200)
    return np.clip(image, 0, 255).astype(np.uint8)


def _inject(targets: dict[int, np.ndarray], truth: dict[int, EpisodeTruth]) -> InjectedFaults:
    e = 2
    row = LEAD_FRAMES + 9
    targets[e][row, 1] += 0.5
    isolated = (e, row - 1)

    e = 5
    row = LEAD_FRAMES + 8
    targets[e][row : row + 2, 3] = 1.0
    pulse = (e, row - 1, row + 1)

    e = 7
    row = truth[e].lift + 1 + 9
    targets[e][row, 2] += 0.4
    targets[e][row + 1 :, 2] += 0.8
    fast = (e, row - 1)
    return InjectedFaults(isolated_jump=isolated, command_pulse=pulse, fast_motion=fast)


def _features() -> datasets.Features:
    vector = datasets.Sequence(datasets.Value("float32"), length=SPEC.action_dim)
    return datasets.Features(
        {
            CAMERA: datasets.Image(),
            "observation.state": vector,
            "action": vector,
            "timestamp": datasets.Value("float32"),
            "frame_index": datasets.Value("int64"),
            "episode_index": datasets.Value("int64"),
            "index": datasets.Value("int64"),
            "task_index": datasets.Value("int64"),
        }
    )


def _info(episodes: int, frames: int) -> dict:
    vector = {"dtype": "float32", "shape": [SPEC.action_dim], "names": ["j0", "j1", "j2", "gripper"]}
    scalar = {"dtype": "int64", "shape": [1], "names": None}
    return {
        "codebase_version": "v2.1",
        "robot_type": "synthetic_arm",
        "total_episodes": episodes,
        "total_frames": frames,
        "total_tasks": 1,
        "total_videos": 0,
        "total_chunks": 1,
        "chunks_size": 1000,
        "fps": int(SPEC.fps),
        "splits": {"train": f"0:{episodes}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": None,
        "features": {
            CAMERA: {"dtype": "image", "shape": list(IMAGE_SHAPE), "names": ["height", "width", "channel"]},
            "observation.state": vector,
            "action": vector,
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": scalar,
            "episode_index": scalar,
            "index": scalar,
            "task_index": scalar,
        },
    }


def _vector_stats(values: np.ndarray) -> dict[str, list[float] | list[int]]:
    return {"min": values.min(axis=0).tolist(), "max": values.max(axis=0).tolist(),
            "mean": values.mean(axis=0).tolist(), "std": values.std(axis=0).tolist(), "count": [len(values)]}  # fmt: skip


def write_pick_place(root: Path, episodes: int, seed: int, faults: bool) -> SyntheticDataset:
    """Write the dataset under `root` (which must not exist) and return what was put into it."""
    rng = np.random.default_rng(seed)
    targets, truth = {}, {}
    for e in range(episodes):
        targets[e], truth[e] = _targets(rng, e % 2)
    injected = _inject(targets, truth) if faults else None

    (root / "meta").mkdir(parents=True)
    running = 0
    stats = []
    for e in range(episodes):
        q = targets[e]
        n = len(q) - 1
        table = datasets.Dataset.from_dict(
            {
                CAMERA: [_image(rng, truth[e].region, float(q[t, 0])) for t in range(n)],
                "observation.state": q[:-1].astype(np.float32).tolist(),
                "action": q[1:].astype(np.float32).tolist(),
                "timestamp": [t / SPEC.fps for t in range(n)],
                "frame_index": list(range(n)),
                "episode_index": [e] * n,
                "index": list(range(running, running + n)),
                "task_index": [0] * n,
            },
            features=_features(),
        )
        path = root / "data" / "chunk-000" / f"episode_{e:06d}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        table.to_parquet(str(path))
        running += n
        vectors = {"observation.state": q[:-1], "action": q[1:]}
        stats.append({"episode_index": e, "stats": {k: _vector_stats(v) for k, v in vectors.items()}})

    lengths = [truth[e].length for e in range(episodes)]
    meta = root / "meta"
    (meta / "info.json").write_text(json.dumps(_info(episodes, sum(lengths)), indent=2), encoding="utf-8")
    (meta / "tasks.jsonl").write_text(json.dumps({"task_index": 0, "task": TASK}) + "\n", encoding="utf-8")
    rows = [{"episode_index": e, "tasks": [TASK], "length": lengths[e]} for e in range(episodes)]
    (meta / "episodes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    (meta / "episodes_stats.jsonl").write_text("".join(json.dumps(r) + "\n" for r in stats), encoding="utf-8")
    return SyntheticDataset(root=root, truth=truth, faults=injected)


def write_spec(path: Path) -> Path:
    path.write_text(json.dumps(dataclasses.asdict(SPEC), indent=2), encoding="utf-8")
    return path
