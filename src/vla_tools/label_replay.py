"""Execute the LABELS closed loop, the way the deployment executes model output, for several execution lengths.

Per episode, starting at frame 0: take the label of the observed frame, place and interpolate it as the
server does at execution rate r (arc_interp.point_frames, to_frames) from the pose at the observation,
execute e targets, then observe again at the recorded frame nearest to the executed pose. The search runs
forward from the last observation, up to three frames past the label's last point, among frames whose
recorded command state equals the executed one, so a gripper that has closed is not observed again before
its recorded close; frames at the same distance (a still stretch) are resolved towards the frame the
recording would have reached after the chunk targets executed since the observation.

Inference latency d control frames (closed_loop's latency_frames; 0 by default):
  sync         at every observation the robot holds its pose for d frames, then executes the chunk's first e
               targets (fewer when the chunk is shorter)
  async        an observation is taken every e frames (e >= d: one inference in flight). Its chunk arrives d
               frames later and is rebuilt from the pose commanded at arrival (arc_interp.after_latency: the
               points not yet passed, their times shifted by d), then executed until the next chunk arrives:
               e - d targets before the next observation and d during its inference. The first chunk is
               built from the observed pose and starts at target 0 after the robot has held still for d
               frames (with e = d the second observation is then the first frame again: the robot has not
               moved); a chunk that runs out holds its last target. A chunk must keep a target after the
               latency: floor(last point frame) > d
With d = 0 and e no longer than the chunk both modes are the same. `executed` is the wall-clock sequence of targets, one per control frame,
holds included. The executed targets are what a perfect model would make the robot do; for static tasks
they are judged by geometry, not timing:

  path         distance of every executed target to the recorded path segment its chunk covers
               (the observed state followed by actions up to three frames past the label's last point):
               median / p90 / max
  seam         jump from the last executed target to the first target of the next chunk, path steps
  kf position  for each recorded command edge, distance between the executed target where that command
               switches and the recorded joints at the edge; an edge never switched counts as missed
  hold         executed frames that stay within a quarter path step around each command switch
  speed        largest per-frame step of the executed targets / the reference step
  inferences   chunks executed; frames: executed targets / recorded frames

Distances are joint-space Euclidean (rad), printed in path steps (median non-zero recorded per-frame
step); the reference step is the 95th percentile of the same steps. With metric "time" and
span = horizon the labels are the recording itself, every observation lands e frames later and the path
deviation is 0.

Usage:
    python -m vla_tools.label_replay --dataset <dir> --spec robot_spec.json --keyframes keyframes.json \
        --config config.json --execute 2 4 6 8 10 [--rate 1] [--out replay.json]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import typing
from pathlib import Path
from typing import Literal

import numpy as np

from vla_tools import arc_core
from vla_tools.arc_interp import after_latency
from vla_tools.arc_interp import point_frames
from vla_tools.arc_interp import to_frames
from vla_tools.episodes import Episode
from vla_tools.episodes import load_episodes
from vla_tools.keyframe_record import KeyframeRecord
from vla_tools.path_metrics import distance_to_path
from vla_tools.path_metrics import step_statistics
from vla_tools.resample_config import ResampleConfig
from vla_tools.robot_spec import RobotSpec

PATH_MARGIN_FRAMES = 3
HOLD_TOLERANCE = 0.25  # path steps

Mode = Literal["sync", "async"]


@dataclasses.dataclass(frozen=True)
class EpisodeReplay:
    executed: np.ndarray  # (n, action_dim) executed targets in order
    observations: list[int]  # observed frame of every chunk
    distances: np.ndarray  # (m,) each executed chunk target's distance to its chunk's recorded segment (rad)
    seams: list[float]  # rad, between consecutive chunks
    stall_frames: int  # frames held for inference latency or because a chunk ran out


@dataclasses.dataclass(frozen=True)
class EdgeCheck:
    positions: list[float]  # rad, per executed command switch: its joints against the recorded edge's
    holds: list[int]  # executed frames held at each executed switch (within the hold tolerance)
    recorded_holds: list[int]  # frames the recording holds at the same edges
    missed: int  # recorded edges never switched


@dataclasses.dataclass(frozen=True)
class ReplayRow:
    execute: int
    path: tuple[float, float, float]  # median, p90, max (rad)
    seam: tuple[float, float]  # median, max (path steps)
    kf_position: tuple[float, float]  # median, max (rad); nan when no edge was switched
    kf_missed: int
    hold_frames: tuple[float, float]  # median, min over command switches
    speed_ratio: tuple[float, float]  # median, max over episodes
    inferences: int
    executed_frames: int
    recorded_frames: int


def _observe(
    executed_pose: np.ndarray,
    poses: np.ndarray,
    joints: list[int],
    commands: list[int],
    lo: int,
    hi: int,
    expected: int,
) -> int:
    """Recorded frame in (lo, hi] nearest to the executed pose, among those with the same command state.

    Frames at the same distance (a still stretch records one pose many times) are resolved towards
    `expected`, the frame the recording would have reached after the executed targets.
    """
    candidates = np.arange(lo + 1, max(hi, lo + 1) + 1)
    if commands:
        state = poses[candidates][:, commands] > 0.5
        same = (state == (executed_pose[commands] > 0.5)).all(axis=1)
        if same.any():
            candidates = candidates[same]
    distance = np.linalg.norm(poses[candidates][:, joints] - executed_pose[joints], axis=1)
    tied = candidates[distance <= distance.min() + 1e-12]
    return int(tied[int(np.argmin(np.abs(tied - expected)))])


def closed_loop(
    e: Episode,
    spec: RobotSpec,
    record: KeyframeRecord,
    config: ResampleConfig,
    execute: int,
    rate: float,
    latency_frames: int = 0,
    mode: Mode = "sync",
) -> EpisodeReplay:
    if mode not in typing.get_args(Mode):
        raise ValueError(f"mode must be one of {typing.get_args(Mode)}, got {mode!r}")
    if execute < 1 or latency_frames < 0:
        raise ValueError(f"execute must be >= 1 and latency_frames >= 0, got {execute}, {latency_frames}")
    if mode == "async" and execute < latency_frames:
        raise ValueError(
            f"async needs execute >= latency_frames (one inference in flight), got {execute} < {latency_frames}"
        )
    times = point_frames(config.span_frames, config.action_horizon, config.anchor, rate)
    if np.floor(times[-1]) < 1 or (mode == "async" and np.floor(times[-1]) - latency_frames < 1):
        raise ValueError(
            f"a chunk of {times[-1]:.2f} frames keeps no target after {latency_frames if mode == 'async' else 0}"
            " frames of latency"
        )
    joints, commands = list(spec.joint_dims), list(spec.command_dims)
    actions = e.actions
    poses = np.stack([spec.current_pose(s) for s in e.states])
    _, ch = arc_core.chunks(actions, spec, config, record.windows(e.index), e.real)
    last_frame = e.real - 1
    t, pose = 0, poses[0]
    executed, observations, distances, seams = [], [], [], []
    stalls = 0
    while t < last_frame and len(observations) <= 4 * len(actions):
        observations.append(t)
        arrival = latency_frames if mode == "async" and executed else 0  # observation to the first target
        if arrival:
            chunk = after_latency(ch.points[t], times, executed[-1], arrival, spec.command_dims)
        else:
            chunk = to_frames(ch.points[t], pose, times, spec.command_dims)
        run = list(chunk[:execute])
        if mode == "async" and len(run) < execute:
            held = run[-1] if run else executed[-1]
            stalls += execute - len(run)
            run += [held.copy() for _ in range(execute - len(run))]
        targets = np.array(run)
        if executed:
            seams.append(float(np.linalg.norm(targets[0, joints] - executed[-1][joints])))
        if mode == "sync" or not executed:
            executed.extend(pose.copy() for _ in range(latency_frames))
            stalls += latency_frames
        end = min(len(actions), int(np.ceil(ch.frames[t, -1])) + PATH_MARGIN_FRAMES)
        segment = np.concatenate([poses[t, joints][None, :], actions[t:end, joints]])
        distances.append(distance_to_path(targets[:, joints], segment))
        split = len(targets) - latency_frames if mode == "async" else len(targets)
        executed.extend(targets[:split])
        pose = executed[-1].copy()
        if arrival + split > 0:
            expected = t + round(rate * (arrival + split))
            t = _observe(pose, poses, joints, commands, t, min(last_frame, end), expected)
        # else: async back to back (e = d) right after the first inference; the robot has not moved, so the
        # next observation is the same frame
        executed.extend(targets[split:])
    return EpisodeReplay(
        executed=np.array(executed),
        observations=observations,
        distances=np.concatenate(distances) if distances else np.zeros(0),
        seams=seams,
        stall_frames=stalls,
    )


def _switches(values: np.ndarray) -> list[int]:
    on = values > 0.5
    return (np.flatnonzero(np.diff(on.astype(np.int8)) != 0) + 1).tolist()


def _hold(executed: np.ndarray, joints: list[int], index: int, tolerance: float) -> int:
    """Executed frames around `index` within `tolerance` of the target at `index`."""
    anchor = executed[index, joints]
    count, lo, hi = 1, index - 1, index + 1
    while lo >= 0 and np.linalg.norm(executed[lo, joints] - anchor) <= tolerance:
        count, lo = count + 1, lo - 1
    while hi < len(executed) and np.linalg.norm(executed[hi, joints] - anchor) <= tolerance:
        count, hi = count + 1, hi + 1
    return count


def edge_check(e: Episode, spec: RobotSpec, executed: np.ndarray, tolerance: float) -> EdgeCheck:
    """Executed command switches against the recorded edges of every command dim, in order; holds within
    `tolerance` (rad) of the joints at the switch."""
    joints = list(spec.joint_dims)
    recorded_rows = e.actions[: e.real]
    positions: list[float] = []
    holds: list[int] = []
    recorded_holds: list[int] = []
    missed = 0
    for dim in spec.command_dims:
        done = _switches(executed[:, dim]) if len(executed) else []
        for k, edge in enumerate(arc_core.command_edges(recorded_rows, dim)):
            if k >= len(done):
                missed += 1
                continue
            positions.append(float(np.linalg.norm(executed[done[k], joints] - e.actions[edge, joints])))
            holds.append(_hold(executed, joints, done[k], tolerance))
            recorded_holds.append(_hold(recorded_rows, joints, edge, tolerance))
    return EdgeCheck(positions=positions, holds=holds, recorded_holds=recorded_holds, missed=missed)


def replay(
    episodes: list[Episode], spec: RobotSpec, record: KeyframeRecord, config: ResampleConfig, execute: int,
    rate: float, path_step: float, reference_step: float,
) -> ReplayRow:  # fmt: skip
    joints = list(spec.joint_dims)
    distances, seams, kf, holds, speeds = [], [], [], [], []
    missed = inferences = executed_frames = recorded_frames = 0
    for e in episodes:
        run = closed_loop(e, spec, record, config, execute, rate)
        distances.append(run.distances)
        seams += [s / path_step for s in run.seams]
        inferences += len(run.observations)
        executed_frames += len(run.executed)
        recorded_frames += e.real
        if len(run.executed) > 1:
            steps = np.linalg.norm(np.diff(run.executed[:, joints], axis=0), axis=1)
            speeds.append(float(steps.max()) / reference_step)
        edges = edge_check(e, spec, run.executed, HOLD_TOLERANCE * path_step)
        missed += edges.missed
        kf += edges.positions
        holds += edges.holds
    pooled = np.concatenate(distances) if distances else np.zeros(1)
    nan2 = (float("nan"), float("nan"))
    return ReplayRow(
        execute=execute,
        path=(float(np.median(pooled)), float(np.percentile(pooled, 90)), float(pooled.max())),
        seam=(float(np.median(seams)), float(np.max(seams))) if seams else (0.0, 0.0),
        kf_position=(float(np.median(kf)), float(np.max(kf))) if kf else nan2,
        kf_missed=missed,
        hold_frames=(float(np.median(holds)), float(np.min(holds))) if holds else nan2,
        speed_ratio=(float(np.median(speeds)), float(np.max(speeds))) if speeds else (0.0, 0.0),
        inferences=inferences,
        executed_frames=executed_frames,
        recorded_frames=recorded_frames,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--keyframes", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--execute", nargs="+", type=int, default=[2, 4, 6, 8, 10])
    parser.add_argument("--rate", type=float, default=1.0, help="execution rate relative to the recording's pace")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    spec = RobotSpec.load(args.spec)
    record = KeyframeRecord.load(args.keyframes)
    record.check_dataset(args.dataset)
    config = ResampleConfig.from_dict(json.loads(args.config.read_text(encoding="utf-8")))
    episodes = load_episodes(args.dataset, spec, only=sorted(record.episodes))
    joints = list(spec.joint_dims)
    path_step, reference_step = step_statistics([e.actions[: e.real, joints] for e in episodes])
    print(f"{len(episodes)} episodes, {config.to_dict()}, execution rate {args.rate}")
    print(f"path step {path_step:.4f} rad, reference step (p95) {reference_step:.4f} rad; path and kf in path steps")
    print(f"{'execute':>7} {'path med/p90/max':>17} {'seam med/max':>13} {'kf med/max':>11} {'missed':>6} "
          f"{'hold':>5} {'speed':>11} {'infer':>6} {'frames':>12}")  # fmt: skip
    rows = []
    for execute in args.execute:
        r = replay(episodes, spec, record, config, execute, args.rate, path_step, reference_step)
        rows.append(r)
        print(f"{r.execute:>7} {r.path[0] / path_step:5.2f}/{r.path[1] / path_step:5.2f}/{r.path[2] / path_step:5.2f} "
              f"{r.seam[0]:6.2f}/{r.seam[1]:6.2f} {r.kf_position[0] / path_step:5.2f}/{r.kf_position[1] / path_step:5.2f} "
              f"{r.kf_missed:>6} {r.hold_frames[0]:>5.0f} {r.speed_ratio[0]:5.2f}/{r.speed_ratio[1]:5.2f} "
              f"{r.inferences:>6} {r.executed_frames:>5}/{r.recorded_frames:<6}")  # fmt: skip
    if args.out is not None:
        payload = {
            "config": config.to_dict(),
            "rate": args.rate,
            "path_step_rad": path_step,
            "reference_step_rad": reference_step,
            "rows": [dataclasses.asdict(r) for r in rows],
        }
        args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
