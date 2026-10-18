"""Open-loop replays of a LIBERO recording from each demonstration's recorded initial simulator state.

    osc     the recorded 7-dim actions through the recorded OSC_POSE controller: does this simulator
            reproduce the recordings
    joint   the recorded joint positions as absolute targets through the joint position controller, one
            per frame (target t = joints of frame t + 1 and the command of frame t, the converted
            dataset's action rows), for each kp: success and tracking
    labels  the targets the label replay executes (vla_tools.label_replay.closed_loop: a label
            configuration's chunks rebuilt by the server timing at a rate, execution length, latency and
            mode) through the joint position controller: the labels themselves in physics. The converted
            dataset's episode e is demonstration e of the HDF5 file

After the last target the final target is held for --extra-frames. Per episode: the first frame at which
the task succeeds, the joint tracking error one frame after each target (max over joints; median and max),
the lag k in 0..5 frames that minimises the mean error of measured[t + k] against target t, and the mean
absolute difference (0-255) between the images rendered at the initial state and the recorded first frame;
the object's largest and final rise above its initial height, and the finger width 10 frames after the first
close command (about 0.063 m around the can, near 0 on nothing).

Usage (simulator environment, MUJOCO_GL=egl, OMP_NUM_THREADS=1, LIBERO on PYTHONPATH):
    python -m vla_libero.replay --hdf5 <task>_demo.hdf5 --source joint --kp 50 150 400 1000 --out sim_replay.json
    python -m vla_libero.replay --hdf5 <task>_demo.hdf5 --source labels --kp 1000 --dataset <timed> \
        --spec robot_spec.json --keyframes keyframes.json --label-config label_config.json --rate 16 --execute 4 \
        --latency 0.1 --mode async --velocity-fraction 1 --switch-hold 0.2 0.55 --out sim_label_replay.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import multiprocessing
from pathlib import Path

import h5py
import numpy as np
from libero.libero import get_libero_path

from vla_libero.env import LiberoJointEnv
from vla_tools import label_replay
from vla_tools.arc_interp import VelocityTiming
from vla_tools.episodes import load_episodes
from vla_tools.keyframe_record import KeyframeRecord
from vla_tools.path_metrics import step_statistics
from vla_tools.resample_config import ResampleConfig
from vla_tools.robot_spec import RobotSpec

LAGS = 6


@dataclasses.dataclass(frozen=True)
class EpisodeJob:
    bddl: Path
    controller: str
    kp: float
    episode: int
    initial_state: np.ndarray  # flattened simulator state of frame 0
    targets: np.ndarray  # (T, 8) joint targets or (T, 7) LIBERO actions
    recorded_images: tuple[np.ndarray, np.ndarray]  # agentview, wrist of frame 0, rotated 180 degrees
    extra_frames: int


@dataclasses.dataclass(frozen=True)
class LabelSetting:
    label_config: dict  # ResampleConfig.to_dict()
    rate: float
    execute: int
    latency_s: float
    mode: str
    velocity_fraction: float | None
    switch_hold_s: tuple[float, float] | None


@dataclasses.dataclass(frozen=True)
class EpisodeReplay:
    episode: int
    targets: int
    success_frame: int | None  # frames stepped when the task first succeeded
    replay_completion: int | None  # labels: the label replay's completion frames
    lift_max: float  # m, the object's largest rise above its initial height
    lift_final: float  # m, its rise at the last frame stepped
    grasp_width: float | None  # m, finger width 10 frames after the first close command
    tracking_median: float  # rad
    tracking_max: float  # rad
    lag_frames: int
    image_difference: float


@dataclasses.dataclass(frozen=True)
class ReplayResult:
    source: str
    kp: float | None
    episodes: int
    successes: int
    success_frames: tuple[float, float, float]  # median, p10, p90 over the successful episodes
    target_frames: float  # median
    tracking_median: float  # median over episodes, rad
    tracking_max: float  # max over episodes, rad
    lag_frames: float  # median over episodes
    image_difference: float  # max over episodes
    setting: LabelSetting | None
    per_episode: list[EpisodeReplay]


_ENV: dict[tuple[str, float], LiberoJointEnv] = {}  # this worker's one environment; one render context each


def replay_episode(job: EpisodeJob) -> EpisodeReplay:
    key = (job.controller, job.kp)
    if key not in _ENV:
        for old in _ENV.values():
            old.close()
        _ENV.clear()
        _ENV[key] = LiberoJointEnv(job.bddl, job.controller, job.kp)  # type: ignore[arg-type]
    env = _ENV[key]
    obs = env.reset(job.initial_state)
    image_difference = float(np.mean([np.abs(obs.agentview.astype(np.int16) - job.recorded_images[0]).mean(),
                                      np.abs(obs.wrist.astype(np.int16) - job.recorded_images[1]).mean()]))  # fmt: skip
    steps = list(job.targets) + [job.targets[-1]] * job.extra_frames
    measured, heights, success_frame = [], [], None
    base = env.object_height()
    close_frame, grasp_width = None, None
    for frame, target in enumerate(steps, start=1):
        obs = env.step(target)
        measured.append(env.joints())
        heights.append(env.object_height() - base)
        if close_frame is None and obs.state[7] > 0.5:
            close_frame = frame
        if close_frame is not None and grasp_width is None and frame == close_frame + 10:
            fingers = env.gripper()
            grasp_width = float(fingers[0] - fingers[1])
        if success_frame is None and env.success():
            success_frame = frame
            break
    tracking_median = tracking_max = 0.0
    lag = 0
    if job.controller == "joint":
        q = np.array(measured)
        n = min(len(q), len(job.targets))
        error = np.abs(q[:n] - job.targets[:n, :7]).max(axis=1)
        tracking_median, tracking_max = float(np.median(error)), float(error.max())
        means = [np.abs(q[k:n] - job.targets[: n - k, :7]).max(axis=1).mean() for k in range(min(LAGS, n))]
        lag = int(np.argmin(means))
    return EpisodeReplay(job.episode, len(job.targets), success_frame, None, float(max(heights)), float(heights[-1]),
                         grasp_width, tracking_median, tracking_max, lag, image_difference)  # fmt: skip


def summarize(source: str, kp: float | None, rows: list[EpisodeReplay], setting: LabelSetting | None) -> ReplayResult:
    frames = np.array([r.success_frame for r in rows if r.success_frame is not None], dtype=np.float64)
    quantiles = tuple(float(x) for x in np.percentile(frames, [50, 10, 90])) if len(frames) else (0.0, 0.0, 0.0)
    return ReplayResult(
        source=source,
        kp=kp,
        episodes=len(rows),
        successes=len(frames),
        success_frames=quantiles,  # type: ignore[arg-type]
        target_frames=float(np.median([r.targets for r in rows])),
        tracking_median=float(np.median([r.tracking_median for r in rows])),
        tracking_max=float(max(r.tracking_max for r in rows)),
        lag_frames=float(np.median([r.lag_frames for r in rows])),
        image_difference=float(max(r.image_difference for r in rows)),
        setting=setting,
        per_episode=rows,
    )


def recorded_targets(demo: h5py.Group, source: str) -> np.ndarray:
    actions = demo["actions"][:]
    if source == "osc":
        return actions.astype(np.float64)
    joints = demo["obs"]["joint_states"][:].astype(np.float64)
    command = (actions[:, 6] > 0).astype(np.float64)
    return np.concatenate([joints[1:], command[:-1, None]], axis=1)


def label_targets(
    args: argparse.Namespace, episodes: list[int]
) -> tuple[dict[int, np.ndarray], dict[int, int], LabelSetting]:
    """Per episode the label replay's executed targets and completion frames, and the setting."""
    spec = RobotSpec.load(args.spec)
    record = KeyframeRecord.load(args.keyframes)
    config = ResampleConfig.from_dict(json.loads(args.label_config.read_text(encoding="utf-8")))
    loaded = load_episodes(args.dataset, spec, only=sorted(record.episodes))
    joints = list(spec.joint_dims)
    path_step, _ = step_statistics([e.actions[: e.real, joints] for e in loaded])
    hold = tuple(args.switch_hold) if args.switch_hold else (0.0, 0.0)
    timing = None
    if args.velocity_fraction is not None:
        timing = VelocityTiming(tuple(spec.joint_velocity_limits), args.velocity_fraction, spec.fps, *hold)
    latency = round(args.latency * spec.fps)
    targets, completion = {}, {}
    for e in loaded:
        if e.index not in episodes:
            continue
        run = label_replay.closed_loop(e, spec, record, config, args.execute, args.rate, latency, args.mode,
                                       final_tolerance=label_replay.HOLD_TOLERANCE * path_step, timing=timing)  # fmt: skip
        targets[e.index], completion[e.index] = run.executed, run.completion_frames
    setting = LabelSetting(config.to_dict(), args.rate, args.execute, args.latency, args.mode, args.velocity_fraction,
                           hold if timing is not None else None)  # fmt: skip
    return targets, completion, setting


def bddl_path(attribute: str) -> Path:
    """The recording's bddl file, recorded relative to the LIBERO repository, in this LIBERO checkout."""
    return Path(get_libero_path("bddl_files")) / Path(attribute).relative_to("libero/libero/bddl_files")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hdf5", type=Path, required=True)
    parser.add_argument("--source", choices=["osc", "joint", "labels"], required=True)
    parser.add_argument("--kp", type=float, nargs="+", default=[1000.0], help="joint controller gains (joint, labels)")
    parser.add_argument("--episodes", type=int, nargs="*", default=None, help="default: every demonstration")
    parser.add_argument("--extra-frames", type=int, default=20)
    parser.add_argument("--workers", type=int, default=25)
    labels = parser.add_argument_group("labels")
    labels.add_argument("--dataset", type=Path, help="the converted, timed dataset the labels are built from")
    labels.add_argument("--spec", type=Path)
    labels.add_argument("--keyframes", type=Path)
    labels.add_argument("--label-config", type=Path)
    labels.add_argument("--rate", type=float, default=1.0)
    labels.add_argument("--execute", type=int, default=5)
    labels.add_argument("--latency", type=float, default=0.0, help="inference latency (s)")
    labels.add_argument("--mode", choices=["async", "sync"], default="sync")
    labels.add_argument("--velocity-fraction", type=float, default=None)
    labels.add_argument("--switch-hold", nargs=2, type=float, default=None, metavar=("RISE_S", "FALL_S"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    with h5py.File(args.hdf5, "r") as f:
        data = f["data"]
        bddl = bddl_path(data.attrs["bddl_file_name"])
        names = sorted(data, key=lambda n: int(n.split("_")[1]))
        chosen = range(len(names)) if args.episodes is None else args.episodes
        recordings = {
            e: (data[names[e]]["states"][0].astype(np.float64),
                recorded_targets(data[names[e]], args.source) if args.source != "labels" else None,
                (np.ascontiguousarray(data[names[e]]["obs"]["agentview_rgb"][0][::-1, ::-1]).astype(np.int16),
                 np.ascontiguousarray(data[names[e]]["obs"]["eye_in_hand_rgb"][0][::-1, ::-1]).astype(np.int16)))
            for e in chosen
        }  # fmt: skip
    setting, completion = None, {}
    if args.source == "labels":
        if None in (args.dataset, args.spec, args.keyframes, args.label_config):
            raise SystemExit("--source labels needs --dataset, --spec, --keyframes and --label-config")
        targets, completion, setting = label_targets(args, list(recordings))
        recordings = {e: (s, targets[e], images) for e, (s, _, images) in recordings.items()}
    gains = args.kp if args.source != "osc" else [0.0]
    controller = "osc" if args.source == "osc" else "joint"
    jobs = [EpisodeJob(bddl, controller, kp, e, s, t, images, args.extra_frames)
            for kp in gains for e, (s, t, images) in recordings.items()]  # fmt: skip
    with multiprocessing.get_context("spawn").Pool(args.workers) as pool:
        rows = pool.map(replay_episode, jobs, chunksize=1)
    rows = [dataclasses.replace(r, replay_completion=completion.get(r.episode)) for r in rows]
    results = []
    for kp in gains:
        r = summarize(
            args.source,
            kp if args.source != "osc" else None,
            [x for j, x in zip(jobs, rows, strict=True) if j.kp == kp],
            setting,
        )
        results.append(r)
        planned = [x.replay_completion for x in r.per_episode if x.replay_completion is not None]
        replay = f"; label replay completion median {np.median(planned):g}" if planned else ""
        print(f"{args.source} kp {kp:g}: {r.successes}/{r.episodes} succeed, success frame median {r.success_frames[0]:g} "
              f"(p10 {r.success_frames[1]:g}, p90 {r.success_frames[2]:g}) of {r.target_frames:g} targets{replay}; "
              f"tracking median {r.tracking_median:.4f} max {r.tracking_max:.4f} rad, lag {r.lag_frames:g} frames; "
              f"image difference max {r.image_difference:.2f}")  # fmt: skip
    payload = {"hdf5": str(args.hdf5), "bddl": str(bddl), "extra_frames": args.extra_frames,
               "results": [dataclasses.asdict(r) for r in results]}  # fmt: skip
    args.out.write_text(json.dumps(payload, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
