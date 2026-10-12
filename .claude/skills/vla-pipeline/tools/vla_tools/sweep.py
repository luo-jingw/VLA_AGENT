"""Offline grid search over resampled-label settings. Data only: no model, no GPU.

Every cell is scored on what the deployed system executes: the label points placed at the server's
assumed times and interpolated onto frames (arc_interp), against the recorded per-frame targets.

  reach        median frames from the current state to the last point
  true         error with the points' true times: how well H points describe the path
  head / full  error with the assumed times over the first --head frames / the whole chunk: what a
               perfect model would still be off by after deployment
  edge         signed shift (frames) of a command edge inside the executed head; negative = early
  jump         first point's distance from the current state, in recorded steps
  pad          fraction of points past the episode end
  kept         command edges still visible in some label

Errors: worst joint over the compared frames, median / p90 over inference frames, in frames of motion.

Admissible: head median <= --max-head, |edge median| <= --max-edge, all edges kept, and when given
jump <= --max-jump (the controller's limit) and pad <= --max-pad. Selection: the largest span with an
admissible cell, then the smallest full error within it. The shortlist (best admissible cell per span)
is what goes to a training comparison; span is only the offline proxy for what reach buys.

Usage:
    python -m vla_tools.sweep --dataset <dir> --spec robot_spec.json --keyframes keyframes.json --horizon 10 \
        --metrics time curvature arc --spans 10 15 20 30 --event-density 0 1 2 --out sweep.json
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
class CellResult:
    config: dict[str, float | int | str]
    reach_frames: float
    true_err: tuple[float, float]  # median, p90 (rad)
    head_err: tuple[float, float]
    full_err: tuple[float, float]
    edge_shift: tuple[float, float, float]  # median, p10, p90 (frames)
    jump_steps: float
    pad_fraction: float
    edges_kept: int
    edges_total: int


@dataclasses.dataclass(frozen=True)
class Bounds:
    max_head: float
    max_edge: float
    max_jump: float | None
    max_pad: float | None


def _edge_shift(rec: np.ndarray, actions: np.ndarray, t: int, dim: int, head: int) -> float | None:
    n = min(head, len(rec), len(actions) - t)
    recorded = actions[t : t + n, dim] > 0.5
    predicted = rec[:n, dim] > 0.5
    current = bool(actions[t - 1, dim] > 0.5) if t > 0 else bool(recorded[0])
    rec_change = np.flatnonzero(recorded != current)
    if len(rec_change) == 0:
        return None
    pred_change = np.flatnonzero(predicted != current)
    if len(pred_change) == 0:
        return float(n - rec_change[0])  # missed inside the head: at least this late
    return float(pred_change[0] - rec_change[0])


def evaluate(
    episodes: list[Episode], spec: RobotSpec, record: KeyframeRecord, config: ResampleConfig, head: int, stride: int
) -> CellResult:
    joints = list(spec.joint_dims)
    assumed = point_frames(config.span_frames, config.action_horizon, config.anchor)
    reach, true_err, head_err, full_err, shifts, jumps, pad = [], [], [], [], [], [], []
    kept = total = 0
    step_norm = float(np.median(np.concatenate(
        [np.linalg.norm(np.diff(e.actions[:, joints], axis=0), axis=1) for e in episodes]
    )))  # fmt: skip
    for e in episodes:
        actions = e.actions
        poses = np.stack([spec.current_pose(s) for s in e.states])
        _, ch = arc_core.chunks(actions, spec, config, record.windows(e.index))
        pad.append(float(ch.padded.mean()))
        jumps.append(float(np.median(np.linalg.norm(ch.points[:, 0, joints] - poses[:, joints], axis=1))))
        for dim in spec.command_dims:
            for edge in arc_core.command_edges(actions, dim):
                total += 1
                after = bool(actions[edge, dim] > 0.5)
                window = ch.points[max(edge - config.action_horizon, 0) : edge + 1, :, dim] > 0.5
                kept += bool(np.any(window == after))
        for t in range(0, e.real, stride):  # inference on recorded frames only; padding is scored by replay
            if ch.padded[t].any():
                continue
            true_times = ch.frames[t] - t + 1.0  # action index -> frames after state[t]
            reach.append(float(true_times[-1]))
            for times, sink, limit in (
                (true_times, true_err, int(np.floor(true_times[-1]))),
                (assumed, head_err, head),
                (assumed, full_err, int(np.floor(assumed[-1]))),
            ):
                rec = to_frames(ch.points[t], poses[t], times, spec.command_dims)
                n = min(limit, len(rec), len(actions) - t)
                if n < 1:
                    continue  # the whole chunk falls inside one recorded frame
                sink.append(float(np.abs(rec[:n, joints] - actions[t : t + n, joints]).max()))
                if sink is head_err:
                    for dim in spec.command_dims:
                        shift = _edge_shift(rec, actions, t, dim, head)
                        if shift is not None:
                            shifts.append(shift)

    def stats(values: list[float]) -> tuple[float, float]:
        return (float(np.median(values)), float(np.percentile(values, 90))) if values else (float("nan"),) * 2

    return CellResult(
        config=config.to_dict(),
        reach_frames=float(np.median(reach)) if reach else 0.0,
        true_err=stats(true_err),
        head_err=stats(head_err),
        full_err=stats(full_err),
        edge_shift=(float(np.median(shifts)), float(np.percentile(shifts, 10)), float(np.percentile(shifts, 90)))
        if shifts
        else (float("nan"),) * 3,
        jump_steps=float(np.median(jumps)) / max(step_norm, 1e-12),
        pad_fraction=float(np.mean(pad)),
        edges_kept=kept,
        edges_total=total,
    )


def admissible(cell: CellResult, bounds: Bounds, step: float) -> bool:
    edge_ok = np.isnan(cell.edge_shift[0]) or abs(cell.edge_shift[0]) <= bounds.max_edge
    jump_ok = bounds.max_jump is None or cell.jump_steps <= bounds.max_jump
    pad_ok = bounds.max_pad is None or cell.pad_fraction <= bounds.max_pad
    return (
        cell.head_err[0] / step <= bounds.max_head
        and edge_ok
        and jump_ok
        and pad_ok
        and cell.edges_kept == cell.edges_total
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--keyframes", type=Path, required=True)
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--metrics", nargs="+", default=["time", "curvature", "arc"])
    parser.add_argument("--spans", nargs="+", type=int, required=True)
    parser.add_argument("--event-density", nargs="+", type=float, default=[0.0])
    parser.add_argument("--anchor", choices=["state", "action"], default="state")
    parser.add_argument("--head", type=int, default=6, help="frames executed before replanning")
    parser.add_argument("--stride", type=int, default=2, help="inference-frame subsampling")
    parser.add_argument("--max-head", type=float, default=1.0, help="frames of motion")
    parser.add_argument("--max-edge", type=float, default=1.0, help="frames")
    parser.add_argument("--max-jump", type=float, default=None, help="recorded steps; the controller's limit")
    parser.add_argument("--max-pad", type=float, default=None, help="fraction")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    spec = RobotSpec.load(args.spec)
    record = KeyframeRecord.load(args.keyframes)
    record.check_dataset(args.dataset)
    episodes = load_episodes(args.dataset, spec, only=sorted(record.episodes))
    step = median_step(episodes, spec)
    bounds = Bounds(args.max_head, args.max_edge, args.max_jump, args.max_pad)
    print(f"{len(episodes)} episodes, horizon {args.horizon}, head {args.head}, 1 frame of motion = {step:.4f}")
    header = (
        f"{'metric':>9} {'span':>4} {'dens':>4} {'reach':>5} {'true':>11} {'head':>11} {'full':>11} "
        f"{'edge med[p10,p90]':>18} {'jump':>5} {'pad':>5} {'kept':>6} ok"
    )
    print(header)
    print("-" * len(header))
    cells = []
    for metric in args.metrics:
        for span in args.spans:
            for density in args.event_density:
                config = ResampleConfig(
                    metric=metric,
                    span_frames=span,
                    action_horizon=args.horizon,
                    event_density=density,
                    anchor=args.anchor,
                )
                cell = evaluate(episodes, spec, record, config, args.head, args.stride)
                cells.append(cell)
                fr = lambda v: f"{v[0] / step:4.2f}/{v[1] / step:5.2f}"  # noqa: E731
                edge = "-" if np.isnan(cell.edge_shift[0]) else (
                    f"{cell.edge_shift[0]:+5.1f}[{cell.edge_shift[1]:+3.0f},{cell.edge_shift[2]:+3.0f}]"
                )  # fmt: skip
                print(
                    f"{metric:>9} {span:>4} {density:>4g} {cell.reach_frames:>5.1f} {fr(cell.true_err):>11} "
                    f"{fr(cell.head_err):>11} {fr(cell.full_err):>11} {edge:>18} {cell.jump_steps:>5.1f} "
                    f"{cell.pad_fraction:>5.1%} {cell.edges_kept:>3}/{cell.edges_total:<3}"
                    f"{'  Y' if admissible(cell, bounds, step) else '  -'}"
                )

    chosen = [c for c in cells if admissible(c, bounds, step)]
    shortlist: dict[int, CellResult] = {}
    for cell in chosen:
        span = int(cell.config["span_frames"])
        if span not in shortlist or (cell.full_err[0], cell.head_err[0]) < (
            shortlist[span].full_err[0],
            shortlist[span].head_err[0],
        ):
            shortlist[span] = cell
    best = shortlist[max(shortlist)] if shortlist else None
    if shortlist:
        print("\nshortlist (best admissible cell per span):")
        for span in sorted(shortlist):
            c = shortlist[span]
            print(f"  span {span:>3}: {c.config['metric']:>9} density {c.config['event_density']:g}  "
                  f"head {c.head_err[0] / step:.2f}  full {c.full_err[0] / step:.2f}/{c.full_err[1] / step:.2f}  "
                  f"edge {c.edge_shift[0]:+.1f}  jump {c.jump_steps:.1f}  pad {c.pad_fraction:.1%}")  # fmt: skip
    if best is None:
        print("\nno admissible cell: lower the span, change the metric, or relax a bound and say why")
    else:
        print(f"\nselected: {best.config}")
    payload = {
        "dataset": str(args.dataset),
        "keyframes": str(args.keyframes),
        "frame_of_motion_rad": step,
        "head": args.head,
        "bounds": dataclasses.asdict(bounds),
        "cells": [dataclasses.asdict(c) for c in cells],
        "shortlist": {str(span): cell.config for span, cell in sorted(shortlist.items())},
        "selected": None if best is None else best.config,
    }
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
