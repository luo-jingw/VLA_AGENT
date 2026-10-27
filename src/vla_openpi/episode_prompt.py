"""Training prompts drawn per sample from an episode's instructions (meta/episodes.jsonl `tasks`).

EpisodePrompt replaces a sample's `episode_index` with one of that episode's instructions, drawn uniformly
at random on every call, so the same frames are seen under every wording of their task. It belongs in the
repack stage, which openpi applies to training data only; a served request carries its own prompt. The
instructions are read on the first sample, so building the config on a serving machine without the
dataset does not touch the file.
"""

from __future__ import annotations

import dataclasses
import functools
import json
from pathlib import Path

import numpy as np
from openpi import transforms


def load_episode_prompts(dataset_root: Path) -> dict[int, tuple[str, ...]]:
    """episode_index -> the episode's instructions, from a LeRobot v2.1 dataset's meta/episodes.jsonl."""
    lines = (dataset_root / "meta" / "episodes.jsonl").read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines if line.strip()]
    return {int(row["episode_index"]): tuple(row["tasks"]) for row in rows}


@dataclasses.dataclass(frozen=True)
class EpisodePrompt(transforms.DataTransformFn):
    dataset_root: Path

    @functools.cached_property
    def prompts(self) -> dict[int, tuple[str, ...]]:
        return load_episode_prompts(self.dataset_root)

    def __call__(self, data: dict) -> dict:
        data = dict(data)
        choices = self.prompts[int(data.pop("episode_index"))]
        data["prompt"] = choices[int(np.random.default_rng().integers(len(choices)))]
        return data
