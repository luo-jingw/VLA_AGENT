"""Closed-loop trials of one condition in a LIBERO task: each of the task's benchmark initial states once, the
chunk policy's points executed through vla_libero.rollout with the joint position controller.

The condition is the label report the policy was trained with (span, H, anchor), the server timing (rate, or
velocity timing capped by it with switch holds) and the execution (e, latency, mode). Results per trial and
a summary: successes with the Wilson 95% interval, completion frames of the successful trials (median,
p10, p90), failure phases, inference time.

Usage (simulator environment; a policy served by `run.py serve-points` per port):
    python -m vla_libero.evaluate --policy websocket --ports 8000 8001 --report <ckpt>/assets/<repo_id>/arc_resample.json \
        --spec robot_spec.json --rate 16 --execute 4 --latency 0 --mode async --velocity-fraction 1 --switch-hold 0.2 0.55 \
        --kp 400 --trials 0 49 --workers 10 --out closed_loop.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import multiprocessing
import time
from pathlib import Path

import numpy as np
from libero.libero import benchmark
from libero.libero import get_libero_path

from vla_libero.env import LiberoJointEnv
from vla_libero.interfaces import ChunkPolicy
from vla_libero.policy import StillPolicy
from vla_libero.policy import WebsocketChunkPolicy
from vla_libero.rollout import Trial
from vla_libero.rollout import rollout
from vla_tools.arc_interp import ArcInterpolator
from vla_tools.arc_interp import VelocityTiming
from vla_tools.robot_spec import RobotSpec


@dataclasses.dataclass(frozen=True)
class Condition:
    report: str
    rate: float
    execute: int
    latency_s: float
    latency_frames: int
    mode: str
    velocity_fraction: float | None
    switch_hold_s: tuple[float, float] | None
    kp: float
    settle_frames: int
    budget: int


@dataclasses.dataclass(frozen=True)
class Summary:
    trials: int
    successes: int
    success_interval: tuple[float, float]  # Wilson 95%
    completion_frames: tuple[float, float, float]  # median, p10, p90 over the successful trials
    phases: dict[str, int]
    infer_seconds: float  # median over trials of each trial's median
    wall_seconds: float


@dataclasses.dataclass(frozen=True)
class WorkerSetup:
    suite: str
    task: int
    policy: str
    host: str
    ports: tuple[int, ...]
    interp: ArcInterpolator
    condition: Condition


_WORKER: dict[str, object] = {}


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (centre - half, centre + half)


def task_files(suite: str, task: int) -> tuple[Path, np.ndarray, str]:
    """bddl file, (N, state) benchmark initial states and the language instruction."""
    tasks = benchmark.get_benchmark_dict()[suite]()
    t = tasks.get_task(task)
    bddl = Path(get_libero_path("bddl_files")) / t.problem_folder / t.bddl_file
    return bddl, np.asarray(tasks.get_task_init_states(task)), t.language


def run_trial(args: tuple[WorkerSetup, int, int]) -> Trial:
    setup, trial, worker_slot = args
    if "env" not in _WORKER:
        bddl, states, prompt = task_files(setup.suite, setup.task)
        _WORKER.update(env=LiberoJointEnv(bddl, "joint", setup.condition.kp), states=states, prompt=prompt)
        port = setup.ports[worker_slot % len(setup.ports)]
        policy: ChunkPolicy = (WebsocketChunkPolicy(setup.host, port) if setup.policy == "websocket"
                               else StillPolicy(setup.interp.action_horizon))  # fmt: skip
        _WORKER["policy"] = policy
    c = setup.condition
    return rollout(_WORKER["env"], _WORKER["policy"], setup.interp, _WORKER["prompt"], _WORKER["states"][trial],  # type: ignore[arg-type]
                   execute=c.execute, latency_frames=c.latency_frames, mode=c.mode, settle_frames=c.settle_frames,  # type: ignore[arg-type]
                   budget=c.budget, trial=trial)  # fmt: skip


def summarize(trials: list[Trial], wall: float) -> Summary:
    done = np.array([t.completion_frames for t in trials if t.success], dtype=np.float64)
    phases: dict[str, int] = {}
    for t in trials:
        phases[t.phase] = phases.get(t.phase, 0) + 1
    quantiles = tuple(float(x) for x in np.percentile(done, [50, 10, 90])) if len(done) else (0.0, 0.0, 0.0)
    return Summary(trials=len(trials), successes=len(done), success_interval=wilson(len(done), len(trials)),
                   completion_frames=quantiles, phases=phases,  # type: ignore[arg-type]
                   infer_seconds=float(np.median([t.infer_seconds for t in trials])), wall_seconds=wall)  # fmt: skip


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--policy", choices=["websocket", "still"], default="websocket")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--ports", type=int, nargs="+", default=[8000], help="one served policy per port")
    parser.add_argument("--report", type=Path, required=True, help="arc_resample.json in the checkpoint's assets")
    parser.add_argument("--spec", type=Path, required=True, help="robot spec with the joint velocity limits")
    parser.add_argument("--rate", type=float, default=1.0)
    parser.add_argument("--execute", type=int, required=True)
    parser.add_argument("--latency", type=float, default=0.0, help="inference latency (s)")
    parser.add_argument("--mode", choices=["async", "sync"], default="async")
    parser.add_argument("--velocity-fraction", type=float, default=None)
    parser.add_argument("--switch-hold", nargs=2, type=float, default=None, metavar=("RISE_S", "FALL_S"),
                        help="holds at command switches; default the robot spec's switch_hold_s (0 0 for none)")  # fmt: skip
    parser.add_argument("--kp", type=float, required=True)
    parser.add_argument("--suite", default="libero_object")
    parser.add_argument("--task", type=int, default=0)
    parser.add_argument("--trials", type=int, nargs=2, default=[0, 49], metavar=("FIRST", "LAST"))
    parser.add_argument("--settle-frames", type=int, default=10)
    parser.add_argument("--budget", type=int, default=280)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    spec = RobotSpec.load(args.spec)
    timing = None
    if args.velocity_fraction is not None:
        if args.switch_hold is None and not spec.switch_hold_s:
            raise SystemExit(f"--velocity-fraction needs --switch-hold or switch_hold_s in {args.spec}")
        hold = tuple(args.switch_hold) if args.switch_hold is not None else spec.switch_hold_s
        timing = VelocityTiming(tuple(spec.joint_velocity_limits), args.velocity_fraction, spec.fps, *hold)
    else:
        hold = (0.0, 0.0)
    interp = ArcInterpolator.from_report(args.report, args.rate, timing)
    condition = Condition(report=str(args.report), rate=args.rate, execute=args.execute, latency_s=args.latency,
                          latency_frames=round(args.latency * spec.fps), mode=args.mode,
                          velocity_fraction=args.velocity_fraction, switch_hold_s=hold if timing else None, kp=args.kp,
                          settle_frames=args.settle_frames, budget=args.budget)  # fmt: skip
    setup = WorkerSetup(args.suite, args.task, args.policy, args.host, tuple(args.ports), interp, condition)
    trials = list(range(args.trials[0], args.trials[1] + 1))
    began = time.perf_counter()
    with multiprocessing.get_context("spawn").Pool(args.workers) as pool:
        jobs = [(setup, t, i % args.workers) for i, t in enumerate(trials)]
        results = pool.map(run_trial, jobs, chunksize=1)
    summary = summarize(results, time.perf_counter() - began)
    low, high = summary.success_interval
    print(f"{summary.successes}/{summary.trials} succeed ({low:.2f}-{high:.2f}), completion frames median "
          f"{summary.completion_frames[0]:g} (p10 {summary.completion_frames[1]:g}, p90 {summary.completion_frames[2]:g}), "
          f"phases {summary.phases}, inference {summary.infer_seconds * 1000:.0f} ms, wall {summary.wall_seconds:.0f} s")  # fmt: skip
    payload = {"suite": args.suite, "task": args.task, "policy": args.policy, "condition": dataclasses.asdict(condition),
               "summary": dataclasses.asdict(summary), "trials": [dataclasses.asdict(t) for t in results]}  # fmt: skip
    args.out.write_text(json.dumps(payload, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
