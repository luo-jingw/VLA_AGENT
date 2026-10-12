"""Data audit: measure what every later step assumes, and list what needs a decision. Writes only a report.

    alignment     actions[t] vs state[t+1] and vs state[t]; which one matches is the index convention
    timestamps    frame-to-frame gaps against 1/fps
    nonfinite     NaN or inf anywhere in state or action
    static dims   joints that never move within any episode; their delta actions are encoder noise
                  that quantile normalization stretches to [-1, 1]
    commands      values of each command dim, edges per episode, pulses shorter than --pulse-frames
    jumps         one-frame joint changes above --jump-factor x the median step, classified as
                  isolated / continuing / discontinuity (see cleaning_ops.Jump)
    idle          frames below --idle-threshold and the distance they carry; leading/trailing idle
    frozen video  consecutive identical camera frames (parquet-embedded images only)
    duplicates    episodes whose action arrays are identical
    lengths       episode length distribution

--review-dir writes one image per jump: the joint traces around it and the camera frames before, at
and after it, so its kind can be confirmed by eye before cleaning decides anything.

Usage:
    python -m vla_tools.audit --dataset <dir> --spec robot_spec.json --out audit.json [--review-dir audit_review]
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from vla_tools import cleaning_ops
from vla_tools import timing_ops
from vla_tools.episodes import Episode
from vla_tools.episodes import FrameReader
from vla_tools.episodes import info as read_info
from vla_tools.episodes import load_episodes
from vla_tools.episodes import median_step
from vla_tools.robot_spec import RobotSpec


@dataclasses.dataclass(frozen=True)
class Finding:
    kind: str
    episode: int
    frame: int
    detail: str


@dataclasses.dataclass(frozen=True)
class AuditReport:
    episodes: int
    frames: int
    fps: float
    lengths: dict[str, float]
    next_state_error_max: float
    same_state_error_max: float
    timestamp_gap_max: float  # largest frame-to-frame gap, in frames
    median_step: float
    static_dims: list[int]
    command_values: dict[str, list[float]]
    command_edges_per_episode: dict[str, list[int]]
    idle_frame_fraction: float
    idle_distance_fraction: float
    findings: list[Finding]


def _frozen_runs(path: Path, column: str, min_run: int) -> list[tuple[int, int]]:
    """Runs of identical camera frames, ignoring rows marked as end padding (they repeat by design)."""
    columns = [column, "is_padding"] if "is_padding" in pq.read_schema(path).names else [column]
    table = pd.read_parquet(path, columns=columns)
    if "is_padding" in table:
        table = table[np.asarray([float(np.ravel(v)[0]) for v in table["is_padding"]]) < 0.5]
    cells = table[column]
    if not len(cells) or not isinstance(cells.iloc[0], dict):
        return []
    digests = [hashlib.md5(cell["bytes"]).hexdigest() for cell in cells]
    runs, start = [], 0
    for t in range(1, len(digests) + 1):
        if t == len(digests) or digests[t] != digests[start]:
            if t - start >= min_run:
                runs.append((start, t))
            start = t
    return runs


def audit(root: Path, spec: RobotSpec, args: argparse.Namespace) -> tuple[AuditReport, list[Episode]]:
    # End padding is written on purpose; every statistic here is about the recorded rows.
    episodes = [
        dataclasses.replace(e, states=e.states[: e.real], actions=e.actions[: e.real], padding=e.padding[: e.real])
        for e in load_episodes(root, spec)
    ]
    meta = read_info(root)
    joints, joint_states = list(spec.joint_dims), list(spec.joint_state_dims)
    step = median_step(episodes, spec)
    findings: list[Finding] = []

    next_err = max(float(np.abs(e.actions[:-1, joints] - e.states[1:, joint_states]).max()) for e in episodes)
    same_err = max(float(np.abs(e.actions[:, joints] - e.states[:, joint_states]).max()) for e in episodes)

    gap_max = 0.0
    for e in episodes:
        stamps = pd.read_parquet(e.path, columns=["timestamp"])["timestamp"].to_numpy()[: e.real]
        gaps = np.diff(stamps) * spec.fps
        gap_max = max(gap_max, float(gaps.max()) if len(gaps) else 0.0)
        for t in np.flatnonzero(gaps > 1.5):
            findings.append(Finding("timestamp_gap", e.index, int(t) + 1, f"{gaps[t]:.1f} frames"))
        if not (np.isfinite(e.actions).all() and np.isfinite(e.states).all()):
            findings.append(Finding("nonfinite", e.index, -1, "NaN or inf in state/action"))

    static = [d for d in spec.joint_dims if max(np.ptp(e.actions[:, d]) for e in episodes) < args.static_threshold]
    for dim in static:
        findings.append(
            Finding("static_dim", -1, -1, f"action dim {dim} never moves more than {args.static_threshold:g}")
        )

    command_values, command_edges = {}, {}
    for dim in spec.command_dims:
        command_values[str(dim)] = sorted({round(float(v), 4) for e in episodes for v in np.unique(e.actions[:, dim])})
        command_edges[str(dim)] = [
            int((np.diff((e.actions[:, dim] > 0.5).astype(np.int8)) != 0).sum()) for e in episodes
        ]
        if not set(command_values[str(dim)]) <= {0.0, 1.0}:
            findings.append(Finding("non_binary_command", -1, -1, f"dim {dim} values {command_values[str(dim)][:6]}"))
        for e in episodes:
            for start, end in cleaning_ops.command_pulses(e.actions, dim, args.pulse_frames):
                findings.append(Finding("command_pulse", e.index, start, f"dim {dim} held {end - start} frames"))

    for e in episodes:
        for jump in cleaning_ops.find_jumps(e.actions, spec, step, args.jump_factor):
            findings.append(
                Finding(f"jump_{jump.kind}", e.index, jump.frame, f"dim {jump.dim}, {jump.steps:.1f} steps")
            )
        lead, trail = timing_ops.moving_bounds(e.actions, spec, args.idle_threshold)
        if lead >= args.idle_report_frames or trail >= args.idle_report_frames:
            findings.append(Finding("idle_ends", e.index, -1, f"{lead} leading / {trail} trailing still frames"))
        for column in spec.image_keys:
            for start, end in _frozen_runs(e.path, column, args.frozen_frames):
                findings.append(
                    Finding("frozen_camera", e.index, start, f"{column} identical for {end - start} frames")
                )

    seen: dict[str, int] = {}
    for e in episodes:
        digest = hashlib.md5(e.actions.tobytes()).hexdigest()
        if digest in seen:
            findings.append(Finding("duplicate_episode", e.index, -1, f"same actions as episode {seen[digest]}"))
        seen.setdefault(digest, e.index)

    speed = np.concatenate([np.linalg.norm(np.diff(e.actions[:, joints], axis=0), axis=1) for e in episodes])
    idle = speed < args.idle_threshold
    lengths = np.array([len(e.actions) for e in episodes])
    report = AuditReport(
        episodes=len(episodes),
        frames=int(lengths.sum()),
        fps=float(meta.get("fps", spec.fps)),
        lengths={"min": float(lengths.min()), "median": float(np.median(lengths)), "max": float(lengths.max())},
        next_state_error_max=next_err,
        same_state_error_max=same_err,
        timestamp_gap_max=gap_max,
        median_step=step,
        static_dims=static,
        command_values=command_values,
        command_edges_per_episode=command_edges,
        idle_frame_fraction=float(idle.mean()),
        idle_distance_fraction=float(speed[idle].sum() / max(speed.sum(), 1e-12)),
        findings=findings,
    )
    return report, episodes


def review_jumps(
    root: Path, spec: RobotSpec, report: AuditReport, episodes: list[Episode], out: Path, camera: str | None
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    reader = FrameReader(root)
    by_index = {e.index: e for e in episodes}
    for finding in report.findings:
        if not finding.kind.startswith("jump_"):
            continue
        e, t = by_index[finding.episode], finding.frame
        lo, hi = max(t - 10, 0), min(t + 10, len(e.actions))
        columns = 4 if camera else 1
        fig, axes = plt.subplots(1, columns, figsize=(4 * columns + 2, 3), squeeze=False)
        for dim in spec.joint_dims:
            axes[0, 0].plot(range(lo, hi), e.actions[lo:hi, dim], lw=0.8)
        axes[0, 0].axvline(t, color="r", ls=":")
        axes[0, 0].set_title(f"episode {e.index}: {finding.kind} at {t} ({finding.detail})", fontsize=8)
        if camera:
            for k, frame in enumerate((max(t - 2, 0), t, min(t + 2, len(e.actions) - 1))):
                axes[0, k + 1].imshow(reader.frame(e.index, camera, frame))
                axes[0, k + 1].set_title(f"{camera} frame {frame}", fontsize=7)
                axes[0, k + 1].axis("off")
        fig.tight_layout()
        fig.savefig(out / f"ep{e.index:04d}_f{t:05d}_{finding.kind}.png", dpi=70)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--review-dir", type=Path, default=None)
    parser.add_argument("--review-camera", default=None, help="image column; default the spec's first")
    parser.add_argument("--static-threshold", type=float, default=1e-3)
    parser.add_argument("--idle-threshold", type=float, default=2e-3)
    parser.add_argument("--idle-report-frames", type=int, default=15)
    parser.add_argument("--jump-factor", type=float, default=8.0)
    parser.add_argument("--pulse-frames", type=int, default=3)
    parser.add_argument("--frozen-frames", type=int, default=5)
    args = parser.parse_args()

    spec = RobotSpec.load(args.spec)
    report, episodes = audit(args.dataset, spec, args)
    padding_rows = sorted({len(e.padding) - e.real for e in load_episodes(args.dataset, spec) if e.padding.any()})
    if padding_rows:
        print(f"end padding present ({padding_rows} rows per episode); every statistic below uses recorded rows only")
    r = report
    print(f"{r.episodes} episodes, {r.frames} frames, fps {r.fps:g}; length min/median/max "
          f"{r.lengths['min']:.0f}/{r.lengths['median']:.0f}/{r.lengths['max']:.0f}")  # fmt: skip
    print(
        f"max |action[t]-state[t+1]| {r.next_state_error_max:.2e}   max |action[t]-state[t]| {r.same_state_error_max:.2e}"
    )
    print(f"largest timestamp gap {r.timestamp_gap_max:.2f} frames; 1 frame of motion = {r.median_step:.4f}")
    print(
        f"static dims {r.static_dims}; idle {r.idle_frame_fraction:.1%} of frames, {r.idle_distance_fraction:.2%} of distance"
    )
    for dim, values in r.command_values.items():
        counts = {k: r.command_edges_per_episode[dim].count(k) for k in sorted(set(r.command_edges_per_episode[dim]))}
        print(f"command dim {dim}: values {values}, edges per episode {counts}")
    kinds: dict[str, int] = {}
    for f in r.findings:
        kinds[f.kind] = kinds.get(f.kind, 0) + 1
    print(f"\nfindings: {kinds}")
    for f in r.findings:
        if f.kind != "static_dim":
            print(f"  {f.kind:<20} episode {f.episode:>4} frame {f.frame:>5}  {f.detail}")
    args.out.write_text(json.dumps(dataclasses.asdict(report), indent=2), encoding="utf-8")
    if args.review_dir is not None:
        camera = args.review_camera or (spec.image_keys[0] if spec.image_keys else None)
        review_jumps(args.dataset, spec, report, episodes, args.review_dir, camera)
        print(f"\njump review images in {args.review_dir}")


if __name__ == "__main__":
    main()
