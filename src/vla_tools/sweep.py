"""Offline grid search over resampled-label settings. Data only: no model, no GPU.

Every cell is scored on what the deployed system executes: the label points placed at the server's
assumed times and interpolated onto frames (arc_interp.to_frames), compared with the recording by
geometry. For static tasks the timing along the demonstrated path does not matter; where the deployed
targets go and where they switch a command does. Scored at every --stride-th recorded frame t whose
label has no padded point:

  reach        median frames from the current state to the last point, in recorded time
  path head    largest distance of the first --head deployed targets to the recorded path segment: the
               current state's joints followed by actions[t] up to 3 frames past the last label point
  path full    the same for every deployed target of the chunk
  kf position  for each recorded command edge inside the first --head frames: distance between the
               deployed target at which that command first switches and the recorded joints at the edge
               (searched from the midpoint after the previous edge of the same dim); an edge whose command
               never switches inside the deployed chunk is counted in kf_missed instead
  speed        largest per-frame step along [current state, first --head targets] / the reference step
  jump         first point's distance from the current state
  pad          fraction of points past the episode end
  kept         command edges still visible in some label

Distances are joint-space Euclidean (rad), printed in path steps: the median non-zero per-frame Euclidean
joint step of the recorded frames. The reference step is the 95th percentile of the same steps. Columns
show median / p90 over inference frames; speed shows median / p95.

The JSON also keeps the time-indexed figures for information: true / head / full, the worst-joint error
against actions[t + j] frame by frame with the points' true or assumed times, and edge_shift, the signed
shift in frames of the first command edge inside the head.

Admissible: path head median <= --max-head, kf position median <= --max-kf (no edge passes), all edges
kept, and when given speed p95 <= --max-speed-ratio, jump <= --max-jump (the controller's limit) and
pad <= --max-pad. Selection: the largest span with an admissible cell, then the smallest path full median
within it (tie: path head median). The shortlist (best admissible cell per span) is what goes to a
training comparison; span is only the offline proxy for what reach buys.

Usage:
    python -m vla_tools.sweep --dataset <dir> --spec robot_spec.json --keyframes keyframes.json --horizon 10 \
        --metrics time curvature --spans 10 15 20 30 --event-density 0 1 2 --out sweep.json
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
from vla_tools.path_metrics import keyframe_position_error
from vla_tools.path_metrics import path_deviation
from vla_tools.path_metrics import speed_ratio
from vla_tools.path_metrics import step_statistics
from vla_tools.resample_config import ResampleConfig
from vla_tools.robot_spec import RobotSpec

PATH_MARGIN_FRAMES = 3  # recorded frames past the last label point that still belong to the compared path


@dataclasses.dataclass(frozen=True)
class CellResult:
    config: dict[str, float | int | str]
    reach_frames: float
    path_head: tuple[float, float]  # median, p90 (rad)
    path_full: tuple[float, float]  # median, p90 (rad)
    kf_position: tuple[float, float]  # median, p90 (rad); nan when no edge falls inside a head
    kf_missed: int  # (inference frame, edge) pairs whose command never switches in the deployed chunk
    speed_ratio: tuple[float, float]  # median, p95 (ratio to the reference step)
    jump_steps: float  # path steps
    pad_fraction: float
    edges_kept: int
    edges_total: int
    true_err: tuple[float, float]  # time-indexed, informational: median, p90 (rad)
    head_err: tuple[float, float]
    full_err: tuple[float, float]
    edge_shift: tuple[float, float, float]  # time-indexed, informational: median, p10, p90 (frames)


@dataclasses.dataclass(frozen=True)
class Bounds:
    max_head: float  # path steps
    max_kf: float  # path steps
    max_speed_ratio: float | None
    max_jump: float | None  # path steps
    max_pad: float | None


@dataclasses.dataclass(frozen=True)
class PathScale:
    path_step: float  # median non-zero per-frame Euclidean joint step of the recorded frames (rad)
    reference_step: float  # 95th percentile of the same steps (rad)

    @staticmethod
    def from_episodes(episodes: list[Episode], spec: RobotSpec) -> PathScale:
        joints = list(spec.joint_dims)
        median, p95 = step_statistics([e.actions[: e.real, joints] for e in episodes])
        return PathScale(path_step=median, reference_step=p95)


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


def _keyframe_errors(
    rec: np.ndarray, actions: np.ndarray, t: int, rows: int, joints: list[int], edges: dict[int, list[int]]
) -> tuple[list[float], int]:
    """Keyframe position error of every recorded edge in actions[t : t + rows] against the deployed `rec`.

    Each edge is searched for from the midpoint after the previous recorded edge of the same dim, so a
    second edge inside one chunk is not matched to the first one's switch. Returns (errors, missed).
    """
    errors: list[float] = []
    missed = 0
    for dim, dim_edges in edges.items():
        previous: int | None = None
        for edge in dim_edges:
            if t <= edge < t + rows:
                start = 0 if previous is None else max(int(np.ceil((previous + edge) / 2)) - t, 0)
                error = keyframe_position_error(
                    rec[start:, joints], rec[start:, dim], bool(actions[edge - 1, dim] > 0.5), actions[:, joints], edge
                )
                if error is None:
                    missed += 1
                else:
                    errors.append(error)
            previous = edge
    return errors, missed


def _stats(values: list[float], percentile: float) -> tuple[float, float]:
    """(median, given percentile), nan when there are no values."""
    if not values:
        return float("nan"), float("nan")
    return float(np.median(values)), float(np.percentile(values, percentile))


def evaluate(
    episodes: list[Episode],
    spec: RobotSpec,
    record: KeyframeRecord,
    config: ResampleConfig,
    head: int,
    stride: int,
    scale: PathScale,
) -> CellResult:
    joints = list(spec.joint_dims)
    assumed = point_frames(config.span_frames, config.action_horizon, config.anchor)
    reach: list[float] = []
    path_head: list[float] = []
    path_full: list[float] = []
    kf_position: list[float] = []
    speeds: list[float] = []
    true_err: list[float] = []
    head_err: list[float] = []
    full_err: list[float] = []
    shifts: list[float] = []
    jumps: list[float] = []
    pad: list[float] = []
    kept = total = missed = 0
    for e in episodes:
        actions = e.actions
        poses = np.stack([spec.current_pose(s) for s in e.states])
        _, ch = arc_core.chunks(actions, spec, config, record.windows(e.index))
        edges = {dim: arc_core.command_edges(actions, dim) for dim in spec.command_dims}
        pad.append(float(ch.padded.mean()))
        jumps.append(float(np.median(np.linalg.norm(ch.points[:, 0, joints] - poses[:, joints], axis=1))))
        for dim, dim_edges in edges.items():
            for edge in dim_edges:
                total += 1
                after = bool(actions[edge, dim] > 0.5)
                window = ch.points[max(edge - config.action_horizon, 0) : edge + 1, :, dim] > 0.5
                kept += bool(np.any(window == after))
        for t in range(0, e.real, stride):  # inference on recorded frames only; padding is scored by replay
            if ch.padded[t].any():
                continue
            true_times = ch.frames[t] - t + 1.0  # action index -> frames after state[t]
            reach.append(float(true_times[-1]))
            exact = to_frames(ch.points[t], poses[t], true_times, spec.command_dims)
            n_true = min(int(np.floor(true_times[-1])), len(exact), len(actions) - t)
            if n_true >= 1:  # otherwise the whole chunk falls inside one recorded frame
                true_err.append(float(np.abs(exact[:n_true, joints] - actions[t : t + n_true, joints]).max()))

            rec = to_frames(ch.points[t], poses[t], assumed, spec.command_dims)  # what the server executes
            n_head = min(head, len(rec), len(actions) - t)
            n_full = min(len(rec), len(actions) - t)
            head_err.append(float(np.abs(rec[:n_head, joints] - actions[t : t + n_head, joints]).max()))
            full_err.append(float(np.abs(rec[:n_full, joints] - actions[t : t + n_full, joints]).max()))
            for dim in spec.command_dims:
                shift = _edge_shift(rec, actions, t, dim, head)
                if shift is not None:
                    shifts.append(shift)

            last = min(len(actions), int(np.ceil(ch.frames[t, -1])) + PATH_MARGIN_FRAMES)
            segment = np.concatenate([poses[t, joints][None, :], actions[t:last, joints]])
            path_head.append(path_deviation(rec[:head, joints], segment))
            path_full.append(path_deviation(rec[:, joints], segment))
            speeds.append(speed_ratio(rec[:head, joints], poses[t, joints], scale.reference_step))
            errors, misses = _keyframe_errors(rec[:n_full], actions, t, n_head, joints, edges)
            kf_position += errors
            missed += misses

    return CellResult(
        config=config.to_dict(),
        reach_frames=float(np.median(reach)) if reach else 0.0,
        path_head=_stats(path_head, 90),
        path_full=_stats(path_full, 90),
        kf_position=_stats(kf_position, 90),
        kf_missed=missed,
        speed_ratio=_stats(speeds, 95),
        jump_steps=float(np.median(jumps)) / scale.path_step,
        pad_fraction=float(np.mean(pad)),
        edges_kept=kept,
        edges_total=total,
        true_err=_stats(true_err, 90),
        head_err=_stats(head_err, 90),
        full_err=_stats(full_err, 90),
        edge_shift=(float(np.median(shifts)), float(np.percentile(shifts, 10)), float(np.percentile(shifts, 90)))
        if shifts
        else (float("nan"),) * 3,
    )


def admissible(cell: CellResult, bounds: Bounds, path_step: float) -> bool:
    kf_ok = bool(np.isnan(cell.kf_position[0])) or cell.kf_position[0] <= bounds.max_kf * path_step
    speed_ok = bounds.max_speed_ratio is None or cell.speed_ratio[1] <= bounds.max_speed_ratio
    jump_ok = bounds.max_jump is None or cell.jump_steps <= bounds.max_jump
    pad_ok = bounds.max_pad is None or cell.pad_fraction <= bounds.max_pad
    return (
        cell.path_head[0] <= bounds.max_head * path_step
        and kf_ok
        and speed_ok
        and jump_ok
        and pad_ok
        and cell.edges_kept == cell.edges_total
    )


def _pair(values: tuple[float, float], unit: float) -> str:
    """median / upper figure in `unit`s, '-' when there is no value."""
    if np.isnan(values[0]):
        return "-"
    return f"{values[0] / unit:4.2f}/{values[1] / unit:5.2f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--keyframes", type=Path, required=True)
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--metrics", nargs="+", default=["time", "curvature"])
    parser.add_argument("--spans", nargs="+", type=int, required=True)
    parser.add_argument("--event-density", nargs="+", type=float, default=[0.0])
    parser.add_argument("--anchor", choices=["state", "action"], default="state")
    parser.add_argument("--head", type=int, default=6, help="frames executed before replanning")
    parser.add_argument("--stride", type=int, default=2, help="inference-frame subsampling")
    parser.add_argument("--max-head", type=float, default=1.0, help="path steps")
    parser.add_argument("--max-kf", type=float, default=1.0, help="path steps")
    parser.add_argument("--max-speed-ratio", type=float, default=None, help="ratio to the recorded p95 step")
    parser.add_argument("--max-jump", type=float, default=None, help="path steps; the controller's limit")
    parser.add_argument("--max-pad", type=float, default=None, help="fraction")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    spec = RobotSpec.load(args.spec)
    record = KeyframeRecord.load(args.keyframes)
    record.check_dataset(args.dataset)
    episodes = load_episodes(args.dataset, spec, only=sorted(record.episodes))
    step = median_step(episodes, spec)
    scale = PathScale.from_episodes(episodes, spec)
    bounds = Bounds(args.max_head, args.max_kf, args.max_speed_ratio, args.max_jump, args.max_pad)
    print(
        f"{len(episodes)} episodes, horizon {args.horizon}, head {args.head}, path step {scale.path_step:.4f} rad, "
        f"reference step (p95) {scale.reference_step:.4f} rad"
    )
    print("path head / path full / kf pos: median/p90 in path steps; speed: median/p95 of the reference step")
    header = (
        f"{'metric':>9} {'span':>4} {'dens':>4} {'reach':>5} {'path head':>10} {'path full':>10} {'kf pos':>10} "
        f"{'speed':>10} {'jump':>5} {'pad':>5} {'kept':>7} ok"
    )
    print(header)
    print("-" * len(header))
    cells: list[CellResult] = []
    oks: list[bool] = []
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
                cell = evaluate(episodes, spec, record, config, args.head, args.stride, scale)
                ok = admissible(cell, bounds, scale.path_step)
                cells.append(cell)
                oks.append(ok)
                print(
                    f"{metric:>9} {span:>4} {density:>4g} {cell.reach_frames:>5.1f} "
                    f"{_pair(cell.path_head, scale.path_step):>10} {_pair(cell.path_full, scale.path_step):>10} "
                    f"{_pair(cell.kf_position, scale.path_step):>10} {_pair(cell.speed_ratio, 1.0):>10} "
                    f"{cell.jump_steps:>5.1f} {cell.pad_fraction:>5.1%} {cell.edges_kept:>3}/{cell.edges_total:<3}"
                    f"{'  Y' if ok else '  -'}"
                )

    shortlist: dict[int, CellResult] = {}
    for cell, ok in zip(cells, oks, strict=True):
        if not ok:
            continue
        span = int(cell.config["span_frames"])
        if span not in shortlist or (cell.path_full[0], cell.path_head[0]) < (
            shortlist[span].path_full[0],
            shortlist[span].path_head[0],
        ):
            shortlist[span] = cell
    best = shortlist[max(shortlist)] if shortlist else None
    if shortlist:
        print("\nshortlist (best admissible cell per span; path steps):")
        for span in sorted(shortlist):
            c = shortlist[span]
            kf = "-" if np.isnan(c.kf_position[0]) else f"{c.kf_position[0] / scale.path_step:.2f}"
            print(f"  span {span:>3}: {c.config['metric']:>9} density {c.config['event_density']:g}  "
                  f"path head {c.path_head[0] / scale.path_step:.2f}  "
                  f"path full {c.path_full[0] / scale.path_step:.2f}/{c.path_full[1] / scale.path_step:.2f}  "
                  f"kf {kf}  speed p95 {c.speed_ratio[1]:.2f}  jump {c.jump_steps:.1f}  pad {c.pad_fraction:.1%}")  # fmt: skip
    if best is None:
        print("\nno admissible cell: lower the span, change the metric, or relax a bound and say why")
    else:
        print(f"\nselected: {best.config}")
    payload = {
        "dataset": str(args.dataset),
        "keyframes": str(args.keyframes),
        "frame_of_motion_rad": step,
        "path_step_rad": scale.path_step,
        "reference_step_rad": scale.reference_step,
        "head": args.head,
        "bounds": dataclasses.asdict(bounds),
        "cells": [{**dataclasses.asdict(c), "admissible": ok} for c, ok in zip(cells, oks, strict=True)],
        "shortlist": {str(span): cell.config for span, cell in sorted(shortlist.items())},
        "selected": None if best is None else best.config,
    }
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
