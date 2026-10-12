"""Read-only view of a keyframes.json written by keyframe_detect: frames and windows per episode."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path


@dataclasses.dataclass(frozen=True)
class KeyframeRecord:
    dataset: Path
    episodes: dict[int, list[dict]]  # episode -> [{"type", "frame", "window": [start, end), "source"}]
    open_issues: int

    @staticmethod
    def load(path: Path) -> KeyframeRecord:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return KeyframeRecord(
            dataset=Path(raw["dataset"]).resolve(),
            episodes={int(k): v["keyframes"] for k, v in raw["episodes"].items()},
            open_issues=sum(1 for i in raw["issues"] if not i["accepted"]),
        )

    def check_dataset(self, dataset: Path) -> None:
        """Keyframes index frames of one dataset version; using them on another is refused."""
        if Path(dataset).resolve() != self.dataset:
            raise ValueError(f"keyframes were detected on {self.dataset}, not {Path(dataset).resolve()}")

    def windows(self, episode: int, types: tuple[str, ...] | None = None) -> list[tuple[int, int]]:
        """[start, end) windows of the episode's keyframes, all types or the given ones."""
        return [
            (int(k["window"][0]), int(k["window"][1]))
            for k in self.episodes.get(episode, [])
            if types is None or k["type"] in types
        ]
