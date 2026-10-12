"""Decide which still frames stay and pad episode ends so the policy learns to hold after the task. Writes a new version.

Run after cleaning, with keyframes detected on the cleaned version (they mark which still stretches are
part of the task). Keyframe frame numbers change with the rows, so detect keyframes again on the output
before reviewing or using them: keyframe tools refuse a keyframes.json from another dataset version.

Without --plan it only reports every still stretch and what it would be classified as. With a plan it
writes the new version, an `is_padding` column (1 on appended rows) and meta/timing_report.json.

timing_plan.json:
    {"threshold": 0.002, "min_frames": 8,
     "leading_keep": 5,      # still frames kept before the first motion; null keeps all
     "pause_keep": null,     # frames kept of each pause outside keyframe windows; null keeps all
     "trailing_keep": null,  # still frames kept after the last motion; null keeps all
     "end_padding": 20,      # rows appended holding the final target; 0 disables
     "reasons": {"end_padding": "recordings stop 1-2 frames after the release", "pause_keep": "..."}}

Every non-null entry needs a reason; the report records them.

Usage:
    python -m vla_tools.episode_timing --dataset <cleaned> --spec robot_spec.json --keyframes keyframes.json \
        [--plan timing_plan.json --out <new_dir>]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

from vla_tools import dataset_writer
from vla_tools import timing_ops
from vla_tools.episodes import load_episodes
from vla_tools.keyframe_record import KeyframeRecord
from vla_tools.robot_spec import RobotSpec


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--keyframes", type=Path, required=True)
    parser.add_argument("--plan", type=Path, default=None)
    parser.add_argument("--threshold", type=float, default=0.002, help="report mode only")
    parser.add_argument("--min-frames", type=int, default=8, help="report mode only")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    spec = RobotSpec.load(args.spec)
    record = KeyframeRecord.load(args.keyframes)
    record.check_dataset(args.dataset)
    raw = json.loads(args.plan.read_text(encoding="utf-8")) if args.plan else {}
    plan = timing_ops.TimingPlan(
        threshold=float(raw.get("threshold", args.threshold)),
        min_frames=int(raw.get("min_frames", args.min_frames)),
        leading_keep=raw.get("leading_keep"),
        pause_keep=raw.get("pause_keep"),
        trailing_keep=raw.get("trailing_keep"),
        end_padding=int(raw.get("end_padding", 0)),
    )
    if args.plan:
        reasons = raw.get("reasons", {})
        missing = [
            k
            for k in ("leading_keep", "pause_keep", "trailing_keep", "end_padding")
            if raw.get(k) not in (None, 0) and k not in reasons
        ]
        if missing:
            raise SystemExit(f"timing plan entries without a reason: {missing}")
        if args.out is None:
            raise SystemExit("--out is required with --plan")

    totals: dict[str, list[int]] = {"leading": [], "trailing": [], "keyframe": [], "pause": []}
    plans, logs = [], {}
    for e in load_episodes(args.dataset, spec, only=sorted(record.episodes)):
        runs = timing_ops.still_runs(e.actions, spec, plan.threshold, plan.min_frames, record.windows(e.index))
        for run in runs:
            totals[run.kind].append(run.end - run.start)
        rows = timing_ops.row_order(len(e.actions), runs, plan)
        actions, states, flag = timing_ops.padded_arrays(e.actions, e.states, spec, rows, plan.end_padding)
        logs[e.index] = {
            "frames_in": len(e.actions),
            "frames_out": len(rows),
            "still_runs": [dataclasses.asdict(r) for r in runs],
            "dropped": len(e.actions) - (len(rows) - plan.end_padding),
            "padded": plan.end_padding,
        }
        plans.append(
            dataset_writer.EpisodePlan(
                source=e.index,
                rows=rows,
                replace={spec.action_key: actions, spec.state_key: states},
                add={
                    "is_padding": dataset_writer.NewColumn(
                        flag.tolist(), {"dtype": "float32", "shape": [1], "names": ["is_padding"]}
                    )
                },
            )
        )

    print(f"still stretches (< {plan.threshold:g} per frame for >= {plan.min_frames} frames):")
    for kind, lengths in totals.items():
        if lengths:
            print(f"  {kind:<9} {len(lengths):>3} runs, {sum(lengths):>5} frames, longest {max(lengths)}")
        else:
            print(f"  {kind:<9}   0 runs")
    if not args.plan:
        return
    dropped = sum(log["dropped"] for log in logs.values())
    print(f"dropped {dropped} still frames, appended {plan.end_padding} padding rows per episode")
    report = {"plan": raw, "episodes": {str(k): v for k, v in logs.items()}}
    dataset_writer.write(args.dataset, args.out, plans, {"timing_report.json": report})
    print(f"wrote {len(plans)} episodes to {args.out}; detect keyframes again on it before using them")


if __name__ == "__main__":
    main()
