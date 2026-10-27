"""Write each episode's instructions into a new version of a LeRobot dataset. Rows and columns are unchanged.

prompts.json, one entry per episode, keyed by episode (see --source-map):
    {"<episode>": {"scene": "<what the episode's video shows>", "prompts": ["<instruction>", ...]}, ...}

The first instruction becomes the episode's task (`task_index` on every row); all of them go to the
episode's `tasks` in meta/episodes.jsonl and meta/tasks.jsonl holds every distinct instruction. A training
config draws one of an episode's instructions per sample with `LeRobotArcDataConfig(rotate_prompts=True)`.
Every episode of the dataset needs an entry with at least one instruction. meta/prompt_annotation.json
records the file and each episode's scene.

--source-map names a JSON file whose `episode_map` maps the keys of prompts.json to this dataset's episode
indices (a converter's report or a derived version's report carries one): annotations written against
source episode numbers then land on the right episodes after exclusions.

Usage:
    python -m vla_tools.prompt_annotation --dataset <dir> --prompts prompts.json [--source-map <report.json>] \
        --out <new_dir>
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

from vla_tools import dataset_writer
from vla_tools.episodes import episode_paths


@dataclasses.dataclass(frozen=True)
class EpisodeAnnotation:
    scene: str
    prompts: tuple[str, ...]


def load_annotations(path: Path, source_map: Path | None) -> dict[int, EpisodeAnnotation]:
    """Episode index of the dataset -> its annotation."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    mapping = None
    if source_map is not None:
        mapping = {str(k): int(v) for k, v in json.loads(source_map.read_text(encoding="utf-8"))["episode_map"].items()}
    out: dict[int, EpisodeAnnotation] = {}
    for key, entry in raw.items():
        if mapping is not None and key not in mapping:
            continue  # an annotated source episode that this version excludes
        episode = mapping[key] if mapping is not None else int(key)
        out[episode] = EpisodeAnnotation(scene=str(entry["scene"]), prompts=tuple(entry["prompts"]))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--source-map", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    annotations = load_annotations(args.prompts, args.source_map)
    episodes = [index for index, _ in episode_paths(args.dataset)]
    missing = [e for e in episodes if e not in annotations or not annotations[e].prompts]
    if missing:
        raise SystemExit(f"episodes without instructions in {args.prompts}: {missing}")
    plans = [dataset_writer.EpisodePlan(source=e, tasks=annotations[e].prompts) for e in episodes]
    report = {
        "prompts": str(args.prompts),
        "source_map": str(args.source_map) if args.source_map else None,
        "scenes": {str(e): annotations[e].scene for e in episodes},
    }
    dataset_writer.write(args.dataset, args.out, plans, {"prompt_annotation.json": report})
    distinct = len({p for e in episodes for p in annotations[e].prompts})
    total = sum(len(annotations[e].prompts) for e in episodes)
    print(f"wrote {len(episodes)} episodes to {args.out}: {total} instructions, {distinct} distinct")


if __name__ == "__main__":
    main()
