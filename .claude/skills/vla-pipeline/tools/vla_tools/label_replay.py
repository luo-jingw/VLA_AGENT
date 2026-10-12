"""Execute the LABELS the way the deployment executes model output, for several replan intervals.

At frames 0, e, 2e, ... take that frame's label, place and interpolate it as the server does, keep the
first e targets, stitch. The result is what a perfect model would make the robot do; its error against
the recording is the floor any model evaluation is read against, and it chooses e.

  err     per-frame worst-joint error, median / p90, frames of motion
  seam    jump between consecutive chunks, recorded steps, median / max (the max usually marks a fast
          recorded motion, see the audit's jumps)
  edge    signed shift of each recorded command edge in the stitched trajectory, median [min, max]

Usage:
    python -m vla_tools.label_replay --dataset <dir> --spec robot_spec.json --keyframes keyframes.json \
        --config config.json --execute 2 4 6 8 10
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np

from vla_tools import arc_core
from vla_tools.arc_interp import point_frames
from vla_tools.arc_interp import to_frames
from vla_tools.episodes import Episode
from vla_tools.episodes import load_episodes
from vla_tools.episodes import median_step
from vla_tools.keyframe_record import KeyframeRecord
from vla_tools.resample_config import ResampleConfig
from vla_tools.robot_spec import RobotSpec


@dataclasses.dataclass(frozen=True)
class ReplayRow:
    execute: int
    err: tuple[float, float]
    seam: tuple[float, float]
    edge_shift: tuple[float, float, float]
    edges_matched: int
    edges_missed: int


def stitch(
    e: Episode, spec: RobotSpec, record: KeyframeRecord, config: ResampleConfig, execute: int
) -> tuple[np.ndarray, list[int]]:
    _, ch = arc_core.chunks(e.actions, spec, config, record.windows(e.index))
    times = point_frames(config.span_frames, config.action_horizon, config.anchor)
    n = len(e.actions)
    pieces, starts = [], []
    for t in range(0, n, execute):
        starts.append(t)
        pieces.append(to_frames(ch.points[t], spec.current_pose(e.states[t]), times, spec.command_dims)[:execute])
    return np.concatenate(pieces)[:n], starts[1:]


def matched_shifts(executed: np.ndarray, recorded: np.ndarray, dims: tuple[int, ...]) -> tuple[list[float], int]:
    """Signed shift of every recorded command edge to the nearest stitched edge of the same kind."""
    shifts, missed = [], 0
    for dim in dims:
        predicted = arc_core.command_edges(executed, dim)
        for edge in arc_core.command_edges(recorded, dim):
            same = [p for p in predicted if (executed[p, dim] > 0.5) == (recorded[edge, dim] > 0.5)]
            if not same:
                missed += 1
                continue
            shifts.append(float(min(same, key=lambda p: abs(p - edge)) - edge))
    return shifts, missed


def replay(
    episodes: list[Episode], spec: RobotSpec, record: KeyframeRecord, config: ResampleConfig, execute: int, step: float
) -> ReplayRow:
    joints = list(spec.joint_dims)
    errors, seams, shifts, missed = [], [], [], 0
    for e in episodes:
        executed, starts = stitch(e, spec, record, config, execute)
        recorded = e.actions[: len(executed)]
        errors.append(np.abs(executed[: e.real, joints] - recorded[: e.real, joints]).max(axis=1))
        seams += [
            float(np.abs(executed[t, joints] - executed[t - 1, joints]).max()) / step
            for t in starts
            if t < len(executed)
        ]
        s, m = matched_shifts(executed, recorded, spec.command_dims)
        shifts += s
        missed += m
    err = np.concatenate(errors)
    return ReplayRow(
        execute=execute,
        err=(float(np.median(err)), float(np.percentile(err, 90))),
        seam=(float(np.median(seams)), float(np.max(seams))) if seams else (0.0, 0.0),
        edge_shift=(float(np.median(shifts)), float(np.min(shifts)), float(np.max(shifts)))
        if shifts
        else (float("nan"),) * 3,
        edges_matched=len(shifts),
        edges_missed=missed,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--keyframes", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--execute", nargs="+", type=int, default=[2, 4, 6, 8, 10])
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    spec = RobotSpec.load(args.spec)
    record = KeyframeRecord.load(args.keyframes)
    record.check_dataset(args.dataset)
    config = ResampleConfig.from_dict(json.loads(args.config.read_text(encoding="utf-8")))
    episodes = load_episodes(args.dataset, spec, only=sorted(record.episodes))
    step = median_step(episodes, spec)
    print(f"{len(episodes)} episodes, {config.to_dict()}, 1 frame of motion = {step:.4f}")
    print(f"{'execute':>7} {'err med/p90':>12} {'seam med/max':>13} {'edge med [min, max]':>20} {'missed':>6}")
    rows = []
    for execute in args.execute:
        r = replay(episodes, spec, record, config, execute, step)
        rows.append(r)
        print(
            f"{r.execute:>7} {r.err[0] / step:>5.2f}/{r.err[1] / step:<6.2f} {r.seam[0]:>5.2f}/{r.seam[1]:<7.2f} "
            f"{r.edge_shift[0]:>+6.1f} [{r.edge_shift[1]:+3.0f}, {r.edge_shift[2]:+3.0f}]     {r.edges_missed:>6}"
        )
    if args.out is not None:
        payload = {
            "config": config.to_dict(),
            "frame_of_motion_rad": step,
            "rows": [dataclasses.asdict(r) for r in rows],
        }
        args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
