"""Write a training dataset with the resampled label appended. Rows, observations and existing columns are unchanged.

    action_chunk        (action_horizon * action_dim,)  this frame's label, flattened
    action_loss_weight  (action_horizon,)               one loss multiplier per point

The label is materialised because no `delta_timestamps` slice can express it. meta/arc_resample.json
records the robot spec, the config, the keyframe file and per-episode statistics; copy it next to the
norm stats in the checkpoint's assets, where the server reads it (arc_interp.ArcInterpolator.from_report).

`--exclude` holds episodes out of this copy (renumbering the rest); build the evaluation copy without it,
so held-out episodes keep their source numbers and carry labels for comparison.

Usage:
    python -m vla_tools.build_labels --dataset <cleaned_dir> --spec robot_spec.json --keyframes keyframes.json \
        --config config.json --out <new_dir> [--exclude 3 9 17]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np

from vla_tools import arc_core
from vla_tools import dataset_writer
from vla_tools.episodes import load_episodes
from vla_tools.keyframe_record import KeyframeRecord
from vla_tools.resample_config import ResampleConfig
from vla_tools.robot_spec import RobotSpec


@dataclasses.dataclass(frozen=True)
class EpisodeStats:
    frames: int
    reach_median_frames: float
    pad_fraction: float
    window_fraction: float
    weight_mean: float


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--keyframes", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--exclude", nargs="*", type=int, default=[])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    spec = RobotSpec.load(args.spec)
    record = KeyframeRecord.load(args.keyframes)
    record.check_dataset(args.dataset)
    if record.open_issues:
        raise SystemExit(f"{args.keyframes} has {record.open_issues} open issues; resolve or accept them first")
    config = ResampleConfig.from_dict(json.loads(args.config.read_text(encoding="utf-8")))
    horizon, dim = config.action_horizon, spec.action_dim
    excluded = set(args.exclude)
    missing = excluded - set(record.episodes)
    if missing:
        raise SystemExit(f"--exclude names episodes not in the keyframe record: {sorted(missing)}")

    plans, stats = [], {}
    for e in load_episodes(args.dataset, spec, only=sorted(record.episodes)):
        if e.index in excluded:
            continue
        _, ch = arc_core.chunks(e.actions, spec, config, record.windows(e.index), e.real)
        reach = ch.frames[:, -1] - np.arange(len(e.actions)) + 1.0
        unpadded = ~ch.padded.any(axis=1)
        stats[e.index] = EpisodeStats(
            frames=len(e.actions),
            reach_median_frames=float(np.median(reach[unpadded])) if unpadded.any() else 0.0,
            pad_fraction=float(ch.padded.mean()),
            window_fraction=float(ch.in_window.mean()),
            weight_mean=float(ch.weights.mean()),
        )
        add = {
            "action_chunk": dataset_writer.NewColumn(
                [row.reshape(horizon * dim).astype(np.float32).tolist() for row in ch.points],
                {"dtype": "float32", "shape": [horizon * dim], "names": ["action_chunk"]},
            ),
            "action_loss_weight": dataset_writer.NewColumn(
                [row.astype(np.float32).tolist() for row in ch.weights],
                {"dtype": "float32", "shape": [horizon], "names": ["loss_weight"]},
            ),
        }
        plans.append(dataset_writer.EpisodePlan(source=e.index, add=add))

    report = {
        "robot": dataclasses.asdict(spec),
        "config": config.to_dict(),
        "keyframes": str(Path(args.keyframes).resolve()),
        "excluded_source_episodes": sorted(excluded),
        "episodes": {str(k): dataclasses.asdict(v) for k, v in stats.items()},
    }
    dataset_writer.write(args.dataset, args.out, plans, {"arc_resample.json": report})
    values = list(stats.values())
    print(f"wrote {len(plans)} episodes to {args.out}: reach median {np.median([s.reach_median_frames for s in values]):.1f} "
          f"frames, pad {np.mean([s.pad_fraction for s in values]):.1%}, window {np.mean([s.window_fraction for s in values]):.1%}, "
          f"mean weight {np.mean([s.weight_mean for s in values]):.2f}")  # fmt: skip


if __name__ == "__main__":
    main()
