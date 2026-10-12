"""Read a LeRobot v2.1 dataset: episode arrays, metadata, and single camera frames. Read-only."""

from __future__ import annotations

import dataclasses
import io
import json
from pathlib import Path

import av
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from PIL import Image

from vla_tools.robot_spec import RobotSpec


@dataclasses.dataclass(frozen=True)
class Episode:
    index: int  # episode_index as stored
    path: Path  # its parquet file
    states: np.ndarray  # (T, state_dim)
    actions: np.ndarray  # (T, action_dim)
    padding: np.ndarray  # (T,) True on rows appended by end padding; all False when the column is absent

    @property
    def real(self) -> int:
        """Number of recorded rows; padding rows only ever follow them."""
        return int((~self.padding).sum())


def info(root: Path) -> dict:
    return json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))


def tasks(root: Path) -> dict[int, str]:
    rows = [json.loads(line) for line in (root / "meta" / "tasks.jsonl").read_text().splitlines() if line.strip()]
    return {int(row["task_index"]): str(row["task"]) for row in rows}


def episode_paths(root: Path) -> list[tuple[int, Path]]:
    """(episode_index, parquet path), sorted by index."""
    return sorted((int(p.stem.split("_")[-1]), p) for p in root.glob("data/chunk-*/episode_*.parquet"))


def load_episode(index: int, path: Path, spec: RobotSpec) -> Episode:
    columns = [spec.state_key, spec.action_key]
    if "is_padding" in pq.read_schema(path).names:
        columns.append("is_padding")
    frame = pd.read_parquet(path, columns=columns)
    padding = (
        np.array([float(np.ravel(v)[0]) for v in frame["is_padding"]]) > 0.5
        if "is_padding" in frame
        else np.zeros(len(frame), dtype=bool)
    )
    return Episode(
        index=index,
        path=path,
        states=np.stack(frame[spec.state_key].to_numpy()).astype(np.float64),
        actions=np.stack(frame[spec.action_key].to_numpy()).astype(np.float64),
        padding=padding,
    )


def load_episodes(root: Path, spec: RobotSpec, only: list[int] | None = None) -> list[Episode]:
    wanted = None if only is None else set(only)
    return [load_episode(i, p, spec) for i, p in episode_paths(root) if wanted is None or i in wanted]


def median_step(episodes: list[Episode], spec: RobotSpec) -> float:
    """One "frame of motion": the median over recorded frames of the largest joint change in one frame."""
    joints = list(spec.joint_dims)
    steps = np.concatenate([np.abs(np.diff(e.actions[: e.real, joints], axis=0)).max(axis=1) for e in episodes])
    return float(np.median(steps))


class FrameReader:
    """Camera frames by (episode, frame), from parquet-embedded images or from the episode's video."""

    def __init__(self, root: Path) -> None:
        self._root = root
        self._info = info(root)
        self._tables: dict[tuple[int, str], pd.Series] = {}
        self._videos: dict[tuple[int, str], list[np.ndarray]] = {}

    def frame(self, episode: int, column: str, index: int) -> np.ndarray:
        feature = self._info["features"][column]
        if feature["dtype"] == "video":
            return self._video_frames(episode, column)[index]
        key = (episode, column)
        if key not in self._tables:
            path = dict(episode_paths(self._root))[episode]
            self._tables[key] = pd.read_parquet(path, columns=[column])[column]
        cell = self._tables[key].iloc[index]
        if isinstance(cell, dict):
            return np.asarray(Image.open(io.BytesIO(cell["bytes"])).convert("RGB"))
        return np.asarray(cell)

    def _video_frames(self, episode: int, column: str) -> list[np.ndarray]:
        key = (episode, column)
        if key not in self._videos:
            chunk = episode // int(self._info.get("chunks_size", 1000))
            path = self._root / self._info["video_path"].format(
                episode_chunk=chunk, video_key=column, episode_index=episode
            )
            with av.open(str(path)) as container:
                self._videos[key] = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
        return self._videos[key]
