"""Highest task speed-up a robot reaches with the labels as policy output, within its joint velocity limits
and under an inference latency. Data only: no model, no GPU.

For every label configuration, inference latency L (s) and execution mode, each episode is replayed closed
loop (label_replay.closed_loop, latency d = ceil(L * fps) control frames, modes sync and async) at every
rate of the grid and execution length e: for async e in {max(d, 1), usable // 2, usable}, usable = the
chunk's floor(span / rate) targets minus d; for sync e in {usable // 2, usable}, usable = all of them. A rate
whose chunk keeps no target (async: none after the latency) has no run. Per episode the rates are searched
upwards until two consecutive rates have no run within the velocity limit. A run is feasible when

  velocity      max over control frames and joints of |dq * fps| / the joint's velocity limit <= 1
                (q: the first observed pose followed by the executed targets, holds included)
  edges         every recorded command edge is switched
  path          the largest distance of an executed chunk target to the recorded segment its chunk covers
                <= the path bound (path steps)

Reported, not constrained: acceleration max |d2q * fps^2| / the joint's nominal acceleration limit (nan
without limits; a deployed controller's acceleration is a setting) and seams (the step into each chunk's
first target, path steps).

Completion (label_replay.completion_frames): frames until a sequence reaches the final recorded row and stays
within the final tolerance (label_replay.HOLD_TOLERANCE path steps) of it, for the executed targets (holds
included) and the recorded actions alike. Speed-up = recorded / executed completion frames.

Episodes whose recording itself exceeds a velocity limit (the recorded actions from the first state, the same
differences) are left out of every case and listed: no label can be held to a limit the demonstration breaks.
Per configuration, latency and mode: the (rate, e) with the largest speed-up among those feasible on at least
the feasible fraction of the remaining episodes; its speed-up is summed over the episodes feasible there and
the others are listed. Per episode the largest feasible rate and the largest feasible speed-up (any e) are
reported as median / p10; for the next rate of the grid the constraints that failed are counted. Path step
and curvature chord length come from all episodes, as in sweep.

Server timing: uniform at each rate of the grid, or with --velocity-fraction velocity timing
(arc_interp.VelocityTiming: every chunk timed so that no joint exceeds that fraction of its velocity limit,
the rate of the grid as the cap) with one case set per --switch-hold pair (seconds the arm stays still after
a command rises / falls). The executed frames held at each command switch (within the hold tolerance) are
reported against the recording's, not constrained.

Finite differences are taken at the dataset's control rate: the targets are what a serving loop at that rate
sends; a robot controller running faster interpolates between them.

Usage:
    python -m vla_tools.rate_ceiling --dataset <dir> --spec robot_spec.json --keyframes keyframes.json \
        --horizon 10 --per-frame --metrics time curvature --spans 32 64 128 256 --event-density 0 2 \
        --latency 0 0.05 0.1 0.2 0.4 --modes async sync --path-bound 2 --feasible-fraction 0.9 \
        [--velocity-fraction 1 --switch-hold 0 0 --switch-hold 0.53 0.67] --out rate_ceiling.json
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
from vla_tools.arc_interp import VelocityTiming
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
    path_bound: float  # path steps
    seam_max: float  # path steps
    inferences: int
    completion_frames: int
    stall_frames: int
    recorded_frames: int  # completion frames of the recorded actions
    holds: tuple[float, float]  # executed frames held at the command switches: median, min; reported only
    recorded_holds: tuple[float, float]  # the recording's at the same edges

    @property
    def failures(self) -> tuple[str, ...]:
        names = []
        if self.velocity_ratio > 1.0:
            names.append("velocity")
        if self.edges_missed:
            names.append("edge")
        if self.path_max > self.path_bound:
            names.append("path")
        return tuple(names)

    @property
    def speedup(self) -> float:
        return self.recorded_frames / self.completion_frames

    @property
    def feasible(self) -> bool:
        return not self.failures


@dataclasses.dataclass(frozen=True)
class Case:
    config: ResampleConfig
    latency_s: float
    mode: label_replay.Mode
    timing: VelocityTiming | None  # None: uniform timing at each rate


@dataclasses.dataclass(frozen=True)
class CaseResult:
    config: dict[str, float | int | str | None]
    latency_s: float
    latency_frames: int
    mode: str
    velocity_fraction: float | None  # velocity timing; None: uniform
    switch_hold_s: tuple[float, float] | None  # rise, fall
    rate: float  # largest-speed-up rate feasible on at least the feasible fraction of episodes; 0 when none
    execute: int
    speedup: float  # recorded / executed completion frames at (rate, execute) over its feasible episodes
    feasible_episodes: int
    left_out: list[int]  # episodes not feasible at (rate, execute)
    inferences: int  # the rest: over the feasible episodes at (rate, execute)
    stall_frames: int
    velocity_ratio: float  # max
    acceleration_ratio: float  # max
    seam_max: float  # path steps, max
    path: tuple[float, float]  # path steps, median and max over episodes of the episode's max
    holds: tuple[float, float]  # frames held at command switches: median of the episodes' medians, min
    recorded_holds: tuple[float, float]
    next_rate_failures: dict[str, int]  # episodes failing each constraint at the next grid rate (best e per
    # episode); "latency": the chunk keeps no target after the latency; an episode whose search stopped
    # below that rate counts as velocity
    episode_rate: tuple[float, float]  # largest feasible rate per episode: median, p10
    episode_speedup: tuple[float, float]  # largest feasible speed-up per episode: median, p10
    by_rate: list[dict[str, float | int]]  # per rate: best e, episodes feasible, speed-up over them


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
    latency: int, mode: label_replay.Mode, path_step: float, path_bound: float, timing: VelocityTiming | None = None,
) -> RateCheck:  # fmt: skip
    joints = list(spec.joint_dims)
    tolerance = label_replay.HOLD_TOLERANCE * path_step
    run = label_replay.closed_loop(
        e, spec, record, config, execute, rate, latency, mode, final_tolerance=tolerance, timing=timing
    )
    first = spec.current_pose(e.states[0])[joints]
    v_ratio, a_ratio = limit_ratios(spec, np.concatenate([first[None, :], run.executed[:, joints]]))
    edges = label_replay.edge_check(e, spec, run.executed, tolerance)
    return RateCheck(
        episode=e.index,
        rate=rate,
        execute=execute,
        velocity_ratio=v_ratio,
        acceleration_ratio=a_ratio,
        edges_missed=edges.missed,
        path_max=float(run.distances.max()) / path_step,
        path_bound=path_bound,
        seam_max=max(run.seams) / path_step if run.seams else 0.0,
        inferences=len(run.observations),
        completion_frames=run.completion_frames,
        stall_frames=run.stall_frames,
        recorded_frames=label_replay.completion_frames(e.actions[: e.real], e.actions[e.real - 1], spec, tolerance),
        holds=_median_min(edges.holds),
        recorded_holds=_median_min(edges.recorded_holds),
    )


def _median_min(values: list[int]) -> tuple[float, float]:
    return (float(np.median(values)), float(min(values))) if values else (float("nan"), float("nan"))


@dataclasses.dataclass(frozen=True)
class EpisodeJob:
    episode: Episode
    spec: RobotSpec
    record: KeyframeRecord
    case: Case
    rates: tuple[float, ...]
    path_step: float
    path_bound: float


def episode_checks(job: EpisodeJob) -> list[RateCheck]:
    """Every (rate, e) of one episode and case, rates ascending, until two consecutive rates have no run within
    the velocity limit."""
    latency = latency_frames(job.case.latency_s, job.spec.fps)
    out: list[RateCheck] = []
    misses = 0
    for rate in job.rates:
        lengths = execution_lengths(job.case.config, rate, latency, job.case.mode)
        checks = [
            check(job.episode, job.spec, job.record, job.case.config, rate, n, latency, job.case.mode,
                  job.path_step, job.path_bound, job.case.timing)
            for n in lengths
        ]  # fmt: skip
        out += checks
        misses = 0 if any(c.velocity_ratio <= 1.0 for c in checks) else misses + 1
        if misses == 2:
            break
    return out


def summarize(
    case: Case, checks: list[RateCheck], episodes: list[int], fps: float, rates: tuple[float, ...],
    feasible_fraction: float,
) -> CaseResult:  # fmt: skip
    """`rates`: the grid searched; a grid rate without runs keeps no target after the latency."""
    latency = latency_frames(case.latency_s, fps)
    needed = math.ceil(feasible_fraction * len(episodes) - 1e-9)
    timing = case.timing
    fraction = timing.fraction if timing is not None else None
    holds_s = (timing.rise_hold_s, timing.fall_hold_s) if timing is not None else None
    table: dict[tuple[float, int], dict[int, RateCheck]] = {}
    for c in checks:
        table.setdefault((c.rate, c.execute), {})[c.episode] = c

    def feasible_runs(key: tuple[float, int]) -> list[RateCheck]:
        return [table[key][i] for i in episodes if i in table[key] and table[key][i].feasible]

    def speedup(runs: list[RateCheck]) -> float:
        return sum(c.recorded_frames for c in runs) / sum(c.completion_frames for c in runs) if runs else 0.0

    by_rate: list[dict[str, float | int]] = []
    for rate in sorted({k[0] for k in table}):
        ok, best, n = max((len(feasible_runs(k)), speedup(feasible_runs(k)), k[1]) for k in table if k[0] == rate)
        by_rate.append({"rate": rate, "execute": n, "feasible_episodes": ok, "speedup": best})
    admissible = [k for k in table if len(feasible_runs(k)) >= needed]
    per_rate, per_speedup = [], []
    for i in episodes:
        mine = [c for c in checks if c.episode == i and c.feasible]
        per_rate.append(max((c.rate for c in mine), default=0.0))
        per_speedup.append(max((c.speedup for c in mine), default=0.0))
    episode_rate = (float(np.median(per_rate)), float(np.percentile(per_rate, 10)))
    episode_speedup = (float(np.median(per_speedup)), float(np.percentile(per_speedup, 10)))
    if not admissible:
        nan, nan2 = float("nan"), (float("nan"), float("nan"))
        return CaseResult(
            config=case.config.to_dict(), latency_s=case.latency_s, latency_frames=latency, mode=case.mode,
            velocity_fraction=fraction, switch_hold_s=holds_s, rate=0.0, execute=0, speedup=0.0,
            feasible_episodes=0, left_out=[], inferences=0, stall_frames=0, velocity_ratio=nan,
            acceleration_ratio=nan, seam_max=nan, path=nan2, holds=nan2, recorded_holds=nan2,
            next_rate_failures={}, episode_rate=episode_rate, episode_speedup=episode_speedup, by_rate=by_rate,
        )  # fmt: skip
    rate, n = max(admissible, key=lambda k: speedup(feasible_runs(k)))
    runs = feasible_runs((rate, n))
    done = {c.episode for c in runs}
    paths = [c.path_max for c in runs]
    higher = [r for r in sorted(rates) if r > rate]
    failures: dict[str, int] = {}
    if higher and not execution_lengths(case.config, higher[0], latency, case.mode):
        failures["latency"] = len(episodes)
    elif higher:
        nxt = higher[0]
        for i in episodes:
            options = [c for c in checks if c.episode == i and c.rate == nxt]
            if any(c.feasible for c in options):
                continue
            names = (
                min(options, key=lambda c: (len(c.failures), c.velocity_ratio)).failures if options else ("velocity",)
            )
            for name in names:
                failures[name] = failures.get(name, 0) + 1
    return CaseResult(
        config=case.config.to_dict(),
        latency_s=case.latency_s,
        latency_frames=latency,
        mode=case.mode,
        velocity_fraction=fraction,
        switch_hold_s=holds_s,
        rate=rate,
        execute=n,
        speedup=speedup(runs),
        feasible_episodes=len(runs),
        left_out=[i for i in episodes if i not in done],
        inferences=sum(c.inferences for c in runs),
        stall_frames=sum(c.stall_frames for c in runs),
        velocity_ratio=max(c.velocity_ratio for c in runs),
        acceleration_ratio=max(c.acceleration_ratio for c in runs),
        seam_max=max(c.seam_max for c in runs),
        path=(float(np.median(paths)), float(np.max(paths))),
        holds=_summary_holds([c.holds for c in runs]),
        recorded_holds=_summary_holds([c.recorded_holds for c in runs]),
        next_rate_failures=failures,
        episode_rate=episode_rate,
        episode_speedup=episode_speedup,
        by_rate=by_rate,
    )


def _summary_holds(per_episode: list[tuple[float, float]]) -> tuple[float, float]:
    """Median of the episodes' median holds and the smallest hold; nan without command switches."""
    medians = [m for m, _ in per_episode if not math.isnan(m)]
    smallest = [s for _, s in per_episode if not math.isnan(s)]
    return (float(np.median(medians)), float(min(smallest))) if medians else (float("nan"), float("nan"))


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
    parser.add_argument("--path-bound", type=float, default=2.0,
                        help="largest closed-loop path deviation of a feasible run (path steps)")  # fmt: skip
    parser.add_argument("--feasible-fraction", type=float, default=0.9,
                        help="fraction of the episodes a selected (rate, e) must be feasible on")  # fmt: skip
    parser.add_argument("--velocity-fraction", type=float, default=None,
                        help="velocity timing at this fraction of the joint velocity limits, the rate as the cap")  # fmt: skip
    parser.add_argument("--switch-hold", nargs=2, type=float, action="append", metavar=("RISE_S", "FALL_S"),
                        help="with --velocity-fraction: seconds held still after a command rises / falls; "
                             "repeat for several case sets (default 0 0)")  # fmt: skip
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    spec = RobotSpec.load(args.spec)
    if not spec.joint_velocity_limits:
        raise SystemExit(f"{args.spec} has no joint_velocity_limits")
    if args.switch_hold and args.velocity_fraction is None:
        raise SystemExit("--switch-hold needs --velocity-fraction")
    if args.velocity_fraction is None:
        timings: list[VelocityTiming | None] = [None]
    else:
        timings = [
            VelocityTiming(velocity_limits=tuple(spec.joint_velocity_limits), fraction=args.velocity_fraction,
                           fps=spec.fps, rise_hold_s=rise, fall_hold_s=fall)
            for rise, fall in (args.switch_hold or [[0.0, 0.0]])
        ]  # fmt: skip
    record = KeyframeRecord.load(args.keyframes)
    record.check_dataset(args.dataset)
    loaded = load_episodes(args.dataset, spec, only=sorted(record.episodes))
    ratios = {e.index: recording_ratios(e, spec) for e in loaded}
    excluded = {i: r for i, r in ratios.items() if r[0] > 1.0}
    episodes = [e for e in loaded if e.index not in excluded]
    joints = list(spec.joint_dims)
    path_step, _ = step_statistics([e.actions[: e.real, joints] for e in loaded])
    curvature_length = (
        args.curvature_length if args.curvature_length is not None else CURVATURE_LENGTH_STEPS * path_step
    )
    configs = configs_from_grid(args.horizon, args.per_frame, args.metrics, args.spans, args.event_density,
                                args.anchor, curvature_length)  # fmt: skip
    cases = [
        Case(config=c, latency_s=lat, mode=m, timing=timing)
        for timing in timings
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
    print(f"{len(episodes)} episodes, fps {spec.fps:g}, path step {path_step:.4f} rad (all {len(loaded)} episodes), "
          f"path bound {args.path_bound:g} path steps, feasible fraction {args.feasible_fraction:g}, "
          f"{len(configs)} configurations, {len(cases)} cases")  # fmt: skip
    print("timing: " + ("uniform" if args.velocity_fraction is None else
          f"velocity at {args.velocity_fraction:g} of the limits, holds (rise, fall) {[list(h) for h in (args.switch_hold or [[0.0, 0.0]])]} s"))  # fmt: skip
    print(f"{'hold':>9} {'metric':>9} {'span':>4} {'dens':>4} {'L(s)':>5} {'mode':>5} {'rate':>5} {'e':>4} {'ok':>3} "
          f"{'speed-up':>8} {'vel':>5} {'acc':>5} {'seam':>5} {'path med/max':>12} {'ep rate med/p10':>15} "
          f"{'ep speed-up':>11} {'held/rec':>9}  next rate fails")  # fmt: skip
    ids = [e.index for e in episodes]
    results = []
    rates = tuple(sorted(args.rates))
    jobs = [EpisodeJob(e, spec, record, case, rates, path_step, args.path_bound) for case in cases for e in episodes]
    with multiprocessing.get_context("fork").Pool(args.workers) as pool:
        parts = pool.imap(episode_checks, jobs)  # in order: every case's episodes in turn, all cases in flight
        for case in cases:
            checks = [c for _ in episodes for c in next(parts)]
            r = summarize(case, checks, ids, spec.fps, rates, args.feasible_fraction)
            results.append(r)
            c = case.config
            fails = ", ".join(f"{k} {v}" for k, v in sorted(r.next_rate_failures.items())) or "-"
            hold = "-" if r.switch_hold_s is None else f"{r.switch_hold_s[0]:g}/{r.switch_hold_s[1]:g}"
            print(f"{hold:>9} {c.metric:>9} {c.span_frames:>4} {c.event_density:>4g} {case.latency_s:>5g} {case.mode:>5} "
                  f"{r.rate:>5g} {r.execute:>4} {r.feasible_episodes:>3} {r.speedup:>8.2f} {r.velocity_ratio:>5.2f} "
                  f"{r.acceleration_ratio:>5.2f} {r.seam_max:>5.2f} {r.path[0]:>5.2f}/{r.path[1]:<6.2f} "
                  f"{r.episode_rate[0]:>7g}/{r.episode_rate[1]:<7g} {r.episode_speedup[0]:>5.2f}/{r.episode_speedup[1]:<5.2f} "
                  f"{r.holds[0]:>4g}/{r.recorded_holds[0]:<4g}  {fails}", flush=True)  # fmt: skip
    payload = {
        "dataset": str(args.dataset),
        "keyframes": str(args.keyframes),
        "episodes": len(episodes),
        "recording_ratios": {str(i): list(r) for i, r in sorted(ratios.items())},
        "excluded": sorted(excluded),
        "fps": spec.fps,
        "path_step_rad": path_step,
        "path_bound": args.path_bound,
        "feasible_fraction": args.feasible_fraction,
        "velocity_fraction": args.velocity_fraction,
        "switch_holds": [list(h) for h in (args.switch_hold or [])],
        "joint_velocity_limits": list(spec.joint_velocity_limits),
        "joint_acceleration_limits": list(spec.joint_acceleration_limits),
        "rates": sorted(args.rates),
        "results": [dataclasses.asdict(r) for r in results],
    }
    args.out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
