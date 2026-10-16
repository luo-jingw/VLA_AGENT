"""Highest task speed-up a robot reaches with the labels as policy output, within its joint velocity limits
and under an inference latency. Data only: no model, no GPU.

For every label configuration, inference latency L (s) and execution mode, each episode is replayed closed
loop (label_replay.closed_loop, latency d = ceil(L * fps) control frames, modes sync and async) at every
rate of the grid and execution length e: for async e in {max(d, 1), usable // 2, usable}, usable = the
chunk's floor(span / rate) targets minus d; for sync e in {usable // 2, usable}, usable = all of them. A rate
whose chunk keeps no target (async: none after the latency) has no run. A run is feasible when

  velocity      max over control frames and joints of |dq * fps| / the joint's velocity limit <= 1
                (q: the first observed pose followed by the executed targets, holds included)
  edges         every recorded command edge is switched

Reported, not constrained: acceleration max |d2q * fps^2| / the joint's nominal acceleration limit (nan
without limits; a deployed controller's acceleration is a setting), path deviation (executed chunk targets
to the recorded segment each chunk covers) and seams (the step into each chunk's first target), in path
steps.

Episodes whose recording itself exceeds a velocity limit (the recorded actions from the first state, the same
differences) are left out of every case and listed: no label can be held to a limit the demonstration breaks.
Per configuration, latency and mode: the (rate, e) with the largest speed-up among those feasible on every
remaining episode. Speed-up = recorded duration / wall-clock duration (holds included), summed over episodes.
The largest rate feasible per episode (best e) is reported as median / p10; for the next rate of the grid the
constraints that failed are counted.

Finite differences are taken at the dataset's control rate: the targets are what a serving loop at that rate
sends; a robot controller running faster interpolates between them.

Usage:
    python -m vla_tools.rate_ceiling --dataset <dir> --spec robot_spec.json --keyframes keyframes.json \
        --horizon 10 --per-frame --metrics time curvature --spans 32 64 128 256 --event-density 0 2 \
        --latency 0 0.05 0.1 0.2 0.4 --modes async sync --out rate_ceiling.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import multiprocessing
from pathlib import Path

import numpy as np

from vla_tools import label_replay
from vla_tools.episodes import Episode
from vla_tools.episodes import load_episodes
from vla_tools.keyframe_record import KeyframeRecord
from vla_tools.path_metrics import step_statistics
from vla_tools.resample_config import ResampleConfig
from vla_tools.robot_spec import RobotSpec

RATES = (1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0)
CURVATURE_LENGTH_STEPS = 4.0  # as sweep: the curvature chord length in path steps


@dataclasses.dataclass(frozen=True)
class RateCheck:
    episode: int
    rate: float
    execute: int
    velocity_ratio: float  # max |dq * fps| / limit
    acceleration_ratio: float  # max |d2q * fps^2| / nominal limit; reported only
    edges_missed: int
    path_max: float  # path steps
    seam_max: float  # path steps
    inferences: int
    wall_frames: int
    stall_frames: int
    recorded_frames: int

    @property
    def failures(self) -> tuple[str, ...]:
        names = []
        if self.velocity_ratio > 1.0:
            names.append("velocity")
        if self.edges_missed:
            names.append("edge")
        return tuple(names)

    @property
    def feasible(self) -> bool:
        return not self.failures


@dataclasses.dataclass(frozen=True)
class Case:
    config: ResampleConfig
    latency_s: float
    mode: label_replay.Mode


@dataclasses.dataclass(frozen=True)
class CaseResult:
    config: dict[str, float | int | str | None]
    latency_s: float
    latency_frames: int
    mode: str
    rate: float  # largest-speed-up feasible rate on every episode; 0 when none
    execute: int
    speedup: float  # recorded / wall-clock duration at (rate, execute); 0 when none
    inferences: int
    stall_frames: int
    velocity_ratio: float  # max over episodes at (rate, execute)
    acceleration_ratio: float
    seam_max: float  # path steps, max over episodes
    path: tuple[float, float]  # path steps, median and max over episodes of the episode's max
    next_rate_failures: dict[str, int]  # episodes failing each constraint at the next grid rate (best e per
    # episode); "latency": the chunk keeps no target after the latency
    episode_rate: tuple[float, float]  # largest feasible rate per episode: median, p10
    by_rate: list[dict[str, float | int]]  # per rate: best e, episodes feasible, speed-up


def latency_frames(latency_s: float, fps: float) -> int:
    return int(math.ceil(latency_s * fps - 1e-9))


def execution_lengths(config: ResampleConfig, rate: float, latency: int, mode: str) -> list[int]:
    """Async: back to back, half and all of the targets left after the latency (one length, max(d, 1), when
    fewer are left: the chunk runs out and holds; none when no target is left). Sync: half and all of the
    chunk."""
    chunk = int(math.floor(config.span_frames / rate + 1e-9))
    if mode == "async":
        low, usable = max(latency, 1), chunk - latency
        if usable < 1:
            return []
        return [low] if usable < low else sorted({low, max(usable // 2, low), usable})
    return sorted({max(chunk // 2, 1), chunk}) if chunk >= 1 else []


def limit_ratios(spec: RobotSpec, q: np.ndarray) -> tuple[float, float]:
    """(T, J) joint targets at the control rate -> max velocity and acceleration as fractions of the limits;
    acceleration nan when the spec has no acceleration limits."""
    velocity = np.abs(np.diff(q, axis=0)) * spec.fps
    acceleration = np.abs(np.diff(q, n=2, axis=0)) * spec.fps**2
    v = float((velocity / np.asarray(spec.joint_velocity_limits)).max()) if len(velocity) else 0.0
    if not spec.joint_acceleration_limits:
        return v, float("nan")
    a = float((acceleration / np.asarray(spec.joint_acceleration_limits)).max()) if len(acceleration) else 0.0
    return v, a


def recording_ratios(e: Episode, spec: RobotSpec) -> tuple[float, float]:
    joints = list(spec.joint_dims)
    first = spec.current_pose(e.states[0])[joints]
    return limit_ratios(spec, np.concatenate([first[None, :], e.actions[: e.real, joints].astype(np.float64)]))


def check(
    e: Episode, spec: RobotSpec, record: KeyframeRecord, config: ResampleConfig, rate: float, execute: int,
    latency: int, mode: label_replay.Mode, path_step: float,
) -> RateCheck:  # fmt: skip
    joints = list(spec.joint_dims)
    run = label_replay.closed_loop(e, spec, record, config, execute, rate, latency, mode)
    first = spec.current_pose(e.states[0])[joints]
    q = np.concatenate([first[None, :], run.executed[:, joints]]) if len(run.executed) else first[None, :]
    v_ratio, a_ratio = limit_ratios(spec, q)
    edges = label_replay.edge_check(e, spec, run.executed, label_replay.HOLD_TOLERANCE * path_step)
    return RateCheck(
        episode=e.index,
        rate=rate,
        execute=execute,
        velocity_ratio=v_ratio,
        acceleration_ratio=a_ratio,
        edges_missed=edges.missed,
        path_max=float(run.distances.max()) / path_step if len(run.distances) else 0.0,
        seam_max=max(run.seams) / path_step if run.seams else 0.0,
        inferences=len(run.observations),
        wall_frames=len(run.executed),
        stall_frames=run.stall_frames,
        recorded_frames=e.real,
    )


@dataclasses.dataclass(frozen=True)
class EpisodeJob:
    episode: Episode
    spec: RobotSpec
    record: KeyframeRecord
    case: Case
    rates: tuple[float, ...]
    path_step: float


def episode_checks(job: EpisodeJob) -> list[RateCheck]:
    """Every (rate, e) of one episode and case, rates ascending, until two consecutive rates have no feasible
    run."""
    latency = latency_frames(job.case.latency_s, job.spec.fps)
    out: list[RateCheck] = []
    misses = 0
    for rate in job.rates:
        lengths = execution_lengths(job.case.config, rate, latency, job.case.mode)
        checks = [
            check(job.episode, job.spec, job.record, job.case.config, rate, n, latency, job.case.mode,
                  job.path_step)
            for n in lengths
        ]  # fmt: skip
        out += checks
        misses = 0 if any(c.feasible for c in checks) else misses + 1
        if misses == 2:
            break
    return out


def summarize(
    case: Case, checks: list[RateCheck], episodes: list[int], fps: float, rates: tuple[float, ...]
) -> CaseResult:
    """`rates`: the grid searched; a grid rate without runs keeps no target after the latency."""
    latency = latency_frames(case.latency_s, fps)
    table: dict[tuple[float, int], dict[int, RateCheck]] = {}
    for c in checks:
        table.setdefault((c.rate, c.execute), {})[c.episode] = c
    by_rate: list[dict[str, float | int]] = []
    best: tuple[float, float, int] | None = None  # speed-up, rate, execute
    for rate in sorted({k[0] for k in table}):
        rows = []
        for (r, n), runs in table.items():
            if r != rate:
                continue
            ok = sum(runs[i].feasible for i in episodes if i in runs)
            wall = sum(runs[i].wall_frames for i in episodes if i in runs)
            recorded = sum(runs[i].recorded_frames for i in episodes if i in runs)
            rows.append((ok, recorded / wall if wall else 0.0, n, len(runs)))
        if not rows:
            continue
        ok, speedup, n, total = max(rows)
        by_rate.append({"rate": rate, "execute": n, "feasible_episodes": ok, "speedup": speedup})
        if ok == len(episodes) and total == len(episodes) and (best is None or speedup > best[0]):
            best = (speedup, rate, n)
    per_episode = []
    for i in episodes:
        feasible = [c.rate for c in checks if c.episode == i and c.feasible]
        per_episode.append(max(feasible) if feasible else 0.0)
    if best is None:
        return CaseResult(
            config=case.config.to_dict(), latency_s=case.latency_s, latency_frames=latency, mode=case.mode,
            rate=0.0, execute=0, speedup=0.0, inferences=0, stall_frames=0, velocity_ratio=float("nan"),
            acceleration_ratio=float("nan"), seam_max=float("nan"), path=(float("nan"), float("nan")),
            next_rate_failures={},
            episode_rate=(float(np.median(per_episode)), float(np.percentile(per_episode, 10))), by_rate=by_rate,
        )  # fmt: skip
    speedup, rate, n = best
    runs = table[(rate, n)]
    paths = [runs[i].path_max for i in episodes]
    higher = [r for r in sorted(rates) if r > rate]
    failures: dict[str, int] = {}
    if higher and higher[0] not in {k[0] for k in table}:
        failures["latency"] = len(episodes)
    elif higher:
        nxt = higher[0]
        for i in episodes:
            options = [c for c in checks if c.episode == i and c.rate == nxt]
            if not options or any(c.feasible for c in options):
                continue
            least = min(options, key=lambda c: (len(c.failures), c.velocity_ratio))
            for name in least.failures:
                failures[name] = failures.get(name, 0) + 1
    return CaseResult(
        config=case.config.to_dict(),
        latency_s=case.latency_s,
        latency_frames=latency,
        mode=case.mode,
        rate=rate,
        execute=n,
        speedup=speedup,
        inferences=sum(runs[i].inferences for i in episodes),
        stall_frames=sum(runs[i].stall_frames for i in episodes),
        velocity_ratio=max(runs[i].velocity_ratio for i in episodes),
        acceleration_ratio=max(runs[i].acceleration_ratio for i in episodes),
        seam_max=max(runs[i].seam_max for i in episodes),
        path=(float(np.median(paths)), float(np.max(paths))),
        next_rate_failures=failures,
        episode_rate=(float(np.median(per_episode)), float(np.percentile(per_episode, 10))),
        by_rate=by_rate,
    )


def configs_from_grid(
    horizon: int, per_frame: bool, metrics: list[str], spans: list[int], densities: list[float],
    anchor: str, curvature_length: float,
) -> list[ResampleConfig]:  # fmt: skip
    out = []
    if per_frame:
        out.append(ResampleConfig(metric="time", span_frames=horizon, action_horizon=horizon, anchor=anchor))  # type: ignore[arg-type]
    for metric in metrics:
        for span in spans:
            for density in densities:
                out.append(ResampleConfig(
                    metric=metric, span_frames=span, action_horizon=horizon, event_density=density,  # type: ignore[arg-type]
                    anchor=anchor, curvature_length=curvature_length if metric == "curvature" else None,  # type: ignore[arg-type]
                ))  # fmt: skip
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--spec", type=Path, required=True, help="robot spec with joint velocity limits (acceleration: reported)"
    )
    parser.add_argument("--keyframes", type=Path, required=True)
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--per-frame", action="store_true", help="add the per-frame labels: time, span = horizon")
    parser.add_argument("--metrics", nargs="+", default=["time", "curvature"])
    parser.add_argument("--spans", nargs="+", type=int, default=[])
    parser.add_argument("--event-density", nargs="+", type=float, default=[0.0])
    parser.add_argument("--anchor", choices=["state", "action"], default="state")
    parser.add_argument("--curvature-length", type=float, default=None,
                        help=f"curvature chord length (rad); default {CURVATURE_LENGTH_STEPS:g} path steps")  # fmt: skip
    parser.add_argument("--rates", nargs="+", type=float, default=list(RATES))
    parser.add_argument("--latency", nargs="+", type=float, default=[0.0], help="inference latency (s)")
    parser.add_argument("--modes", nargs="+", choices=["async", "sync"], default=["async", "sync"])
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    spec = RobotSpec.load(args.spec)
    if not spec.joint_velocity_limits:
        raise SystemExit(f"{args.spec} has no joint_velocity_limits")
    record = KeyframeRecord.load(args.keyframes)
    record.check_dataset(args.dataset)
    loaded = load_episodes(args.dataset, spec, only=sorted(record.episodes))
    ratios = {e.index: recording_ratios(e, spec) for e in loaded}
    excluded = {i: r for i, r in ratios.items() if r[0] > 1.0}
    episodes = [e for e in loaded if e.index not in excluded]
    joints = list(spec.joint_dims)
    path_step, _ = step_statistics([e.actions[: e.real, joints] for e in episodes])
    curvature_length = (
        args.curvature_length if args.curvature_length is not None else CURVATURE_LENGTH_STEPS * path_step
    )
    configs = configs_from_grid(args.horizon, args.per_frame, args.metrics, args.spans, args.event_density,
                                args.anchor, curvature_length)  # fmt: skip
    cases = [
        Case(config=c, latency_s=lat, mode=m)
        for c in configs
        for lat in args.latency
        for m in args.modes
        if not (lat == 0.0 and m == "sync" and "async" in args.modes)  # identical at zero latency
    ]
    rec_v = [r[0] for r in ratios.values()]
    rec_a = [r[1] for r in ratios.values()]
    print(
        f"recordings, max over the episode as a fraction of the limits: velocity median {np.median(rec_v):.2f} "
        f"max {max(rec_v):.2f}, acceleration median {np.median(rec_a):.2f} max {max(rec_a):.2f}"
    )
    print(
        f"left out (recording beyond the velocity limit): {len(excluded)} of {len(loaded)} "
        + (", ".join(f"{i} (v {v:.2f})" for i, (v, _) in sorted(excluded.items())) or "none")
    )
    print(f"{len(episodes)} episodes, fps {spec.fps:g}, path step {path_step:.4f} rad, "
          f"{len(configs)} configurations, {len(cases)} cases")  # fmt: skip
    print(f"{'metric':>9} {'span':>4} {'dens':>4} {'L(s)':>5} {'mode':>5} {'rate':>5} {'e':>4} {'speed-up':>8} "
          f"{'vel':>5} {'acc':>5} {'seam':>5} {'path med/max':>12} {'ep rate med/p10':>15}  "
          "next rate fails")  # fmt: skip
    ids = [e.index for e in episodes]
    results = []
    rates = tuple(sorted(args.rates))
    with multiprocessing.get_context("fork").Pool(args.workers) as pool:
        for case in cases:
            jobs = [EpisodeJob(e, spec, record, case, rates, path_step) for e in episodes]
            checks = [c for part in pool.map(episode_checks, jobs) for c in part]
            r = summarize(case, checks, ids, spec.fps, rates)
            results.append(r)
            c = case.config
            fails = ", ".join(f"{k} {v}" for k, v in sorted(r.next_rate_failures.items())) or "-"
            print(f"{c.metric:>9} {c.span_frames:>4} {c.event_density:>4g} {case.latency_s:>5g} {case.mode:>5} "
                  f"{r.rate:>5g} {r.execute:>4} {r.speedup:>8.2f} {r.velocity_ratio:>5.2f} {r.acceleration_ratio:>5.2f} "
                  f"{r.seam_max:>5.2f} {r.path[0]:>5.2f}/{r.path[1]:<6.2f} {r.episode_rate[0]:>7g}/{r.episode_rate[1]:<7g}  {fails}", flush=True)  # fmt: skip
    payload = {
        "dataset": str(args.dataset),
        "keyframes": str(args.keyframes),
        "episodes": len(episodes),
        "recording_ratios": {str(i): list(r) for i, r in sorted(ratios.items())},
        "excluded": sorted(excluded),
        "fps": spec.fps,
        "path_step_rad": path_step,
        "joint_velocity_limits": list(spec.joint_velocity_limits),
        "joint_acceleration_limits": list(spec.joint_acceleration_limits),
        "rates": sorted(args.rates),
        "results": [dataclasses.asdict(r) for r in results],
    }
    args.out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
