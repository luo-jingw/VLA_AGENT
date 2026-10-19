"""Open-loop evaluation of a served policy on recorded episodes, through the openpi websocket client.

Every observation sent is the recorded one, so a wrong prediction never feeds back: this measures
prediction error in the units the robot executes, not task success. Use held-out episodes; on training
episodes it only shows the pipeline works.

The server returns per-frame targets under "actions" (actions[j] is for frame t+j+1). A resampled
server also returns "arc_points"; when the dataset has an "action_chunk" column they are compared with
it, which separates model error from the server's placement error. A per-frame baseline returns only
"actions" and is scored on the same per-frame columns, which is what makes models comparable.

  points    model points vs label points (resampled servers only)
  head      per-frame targets vs recorded, first --execute frames
  stitched  infer every --execute frames, keep that many targets, compare the stitched trajectory
  edges     signed shift of each recorded command edge in the stitched trajectory
  static    largest move commanded on the static dims, rad
  hold      on rows marked is_padding (the task is done, the recording holds the final pose): how far
            the stitched targets drift from that pose, frames of motion. All other columns use real
            rows only
  ms        client round trip; under JAX, openpi's own infer_ms stops at dispatch and is not compute time

replay_spec.json:
    {"images": {"observation/image": "observation.images.cam_high", ...},
     "state": {"observation/state": "observation.state"}, "prompt_key": "prompt", "static_dims": [7, 8]}

Usage:
    python -m vla_tools.replay_eval --dataset <dir> --spec robot_spec.json --replay-spec replay_spec.json \
        --episodes 3 4 11 19 --execute 6 --host <server> --port 8000 --out eval.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from openpi_client import websocket_client_policy

from vla_tools import arc_core
from vla_tools.episodes import FrameReader
from vla_tools.episodes import episode_paths
from vla_tools.episodes import load_episode
from vla_tools.episodes import tasks
from vla_tools.robot_spec import RobotSpec


@dataclasses.dataclass(frozen=True)
class ReplaySpec:
    images: dict[str, str]
    state: dict[str, str]
    prompt_key: str
    static_dims: tuple[int, ...]

    @staticmethod
    def load(path: Path) -> ReplaySpec:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return ReplaySpec(
            images=dict(raw["images"]),
            state=dict(raw["state"]),
            prompt_key=str(raw.get("prompt_key", "prompt")),
            static_dims=tuple(int(d) for d in raw.get("static_dims", [])),
        )


@dataclasses.dataclass(frozen=True)
class EpisodeResult:
    episode: int
    frames: int
    points_err: list[float]  # rad, per inference
    head_err: list[float]
    stitched_err: list[float]  # rad, per real frame
    hold_err: list[float]  # rad, per padding frame
    edge_shifts: list[int]  # frames, stitched switch minus recorded edge
    edges_missed: int
    static_move: float
    latency_ms: list[float]


def evaluate_episode(
    client: websocket_client_policy.WebsocketClientPolicy,
    root: Path,
    episode: int,
    spec: RobotSpec,
    replay: ReplaySpec,
    execute: int,
) -> EpisodeResult:
    path = dict(episode_paths(root))[episode]
    e = load_episode(episode, path, spec)
    columns = [*replay.state.values(), "task_index"] + [
        c for c in ("action_chunk", "is_padding") if _has_column(path, c)
    ]
    table = pd.read_parquet(path, columns=columns)
    labels = np.stack(table["action_chunk"].to_numpy()) if "action_chunk" in table else None
    padding = (
        np.array([float(np.ravel(v)[0]) for v in table["is_padding"]]) > 0.5
        if "is_padding" in table
        else np.zeros(len(table), dtype=bool)
    )
    prompt = tasks(root)[int(table["task_index"].iloc[0])]
    reader = FrameReader(root)
    joints = list(spec.joint_dims)
    n = len(e.actions)

    answers, latency = {}, []
    for t in range(n):
        obs = {key: reader.frame(episode, column, t) for key, column in replay.images.items()}
        obs.update({key: np.asarray(table[column].iloc[t], dtype=np.float32) for key, column in replay.state.items()})
        obs[replay.prompt_key] = prompt
        start = time.monotonic()
        answers[t] = client.infer(obs)
        latency.append((time.monotonic() - start) * 1000)

    points_err, head_err, static_move = [], [], 0.0
    for t, answer in answers.items():
        if padding[t]:
            continue
        targets = np.asarray(answer["actions"], dtype=np.float64)[:, : spec.action_dim]
        m = min(execute, len(targets), n - t)
        head_err.append(float(np.abs(targets[:m, joints] - e.actions[t : t + m, joints]).max()))
        if replay.static_dims:
            pose = spec.current_pose(e.states[t])
            dims = list(replay.static_dims)
            static_move = max(static_move, float(np.abs(targets[:, dims] - pose[dims]).max()))
        if labels is not None and "arc_points" in answer:
            points = np.asarray(answer["arc_points"], dtype=np.float64)[:, : spec.action_dim]
            truth = labels[t].reshape(len(points), spec.action_dim)
            points_err.append(float(np.abs(points[:, joints] - truth[:, joints]).max()))

    stitched = np.concatenate(
        [np.asarray(answers[t]["actions"], dtype=np.float64)[:execute, : spec.action_dim] for t in range(0, n, execute)]
    )[:n]
    real = ~padding[: len(stitched)]
    errors = np.abs(stitched[:, joints] - e.actions[: len(stitched), joints]).max(axis=1)
    shifts, missed = edge_shifts(stitched[real], e.actions[: len(stitched)][real], spec.command_dims)
    return EpisodeResult(
        episode=episode,
        frames=n,
        points_err=points_err,
        head_err=head_err,
        stitched_err=errors[real].tolist(),
        hold_err=errors[~real].tolist(),
        edge_shifts=shifts,
        edges_missed=missed,
        static_move=static_move,
        latency_ms=latency,
    )


def edge_shifts(stitched: np.ndarray, recorded: np.ndarray, command_dims: Sequence[int]) -> tuple[list[int], int]:
    """Signed frame shift of each recorded command edge to the nearest stitched switch of the same direction
    within half the distance to the neighbouring recorded edges of its dim, and the count with none there."""
    shifts, missed = [], 0
    for dim in command_dims:
        switches = arc_core.command_edges(stitched, dim)
        edges = arc_core.command_edges(recorded, dim)
        for k, edge in enumerate(edges):
            low = (edges[k - 1] + edge) / 2 if k > 0 else 0
            high = (edge + edges[k + 1]) / 2 if k + 1 < len(edges) else len(stitched)
            rising = recorded[edge, dim] > 0.5
            same = [s for s in switches if low <= s <= high and (stitched[s, dim] > 0.5) == rising]
            if same:
                shifts.append(min(same, key=lambda s: abs(s - edge)) - edge)
            else:
                missed += 1
    return shifts, missed


def _has_column(path: Path, column: str) -> bool:
    return column in pq.read_schema(path).names


def _fmt(values: list[float], step: float) -> str:
    return "      -      " if not values else f"{np.median(values) / step:5.2f}/{np.percentile(values, 90) / step:5.2f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--replay-spec", type=Path, required=True)
    parser.add_argument("--episodes", nargs="+", type=int, required=True)
    parser.add_argument("--execute", type=int, default=6)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    spec, replay = RobotSpec.load(args.spec), ReplaySpec.load(args.replay_spec)
    client = websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)
    results = [evaluate_episode(client, args.dataset, ep, spec, replay, args.execute) for ep in args.episodes]
    paths = dict(episode_paths(args.dataset))
    steps = [
        np.abs(np.diff(load_episode(ep, paths[ep], spec).actions[:, list(spec.joint_dims)], axis=0)).max(axis=1)
        for ep in args.episodes
    ]
    step = float(np.median(np.concatenate(steps)))
    print(f"execute {args.execute}, 1 frame of motion = {step:.4f} rad; errors median/p90 in frames of motion")
    print(
        f"{'episode':>7} {'points':>13} {'head':>13} {'stitched':>13} {'edges':>14} {'static':>8} {'hold':>13} {'ms':>5}"
    )
    for r in results:
        edges = str([int(s) for s in r.edge_shifts]) + (f" miss {r.edges_missed}" if r.edges_missed else "")
        print(f"{r.episode:>7} {_fmt(r.points_err, step):>13} {_fmt(r.head_err, step):>13} "
              f"{_fmt(r.stitched_err, step):>13} {edges:>14} {r.static_move:>8.1e} {_fmt(r.hold_err, step):>13} "
              f"{np.median(r.latency_ms):>5.0f}")  # fmt: skip
    pool = lambda name: [v for r in results for v in getattr(r, name)]  # noqa: E731
    print(f"{'all':>7} {_fmt(pool('points_err'), step):>13} {_fmt(pool('head_err'), step):>13} "
          f"{_fmt(pool('stitched_err'), step):>13} {str([int(s) for s in pool('edge_shifts')]):>14} {'':>8} "
          f"{_fmt(pool('hold_err'), step):>13}")  # fmt: skip
    payload = {
        "execute": args.execute,
        "frame_of_motion_rad": step,
        "episodes": [dataclasses.asdict(r) for r in results],
    }
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
