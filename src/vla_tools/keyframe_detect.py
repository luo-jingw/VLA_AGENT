"""Detect a task's keyframes in every episode, apply manual corrections, check consistency, write keyframes.json.

Checks, all per keyframe type:
  count      occurrences per episode against `expected`
  order      first occurrences follow the spec's `order`
  phase      position in the episode (frame / length); robust z beyond --outlier-z is flagged
  interval   frames between consecutive types in `order`; same outlier rule
  clipped    a window that runs past either end of the episode

Detection is never final on its own: confirm the semantics by eye with keyframe_review, record every
correction in the overrides file with a reason, and rerun until the issue list is empty or every
remaining issue is explained there.

keyframe_overrides.json:
    {"set":    [{"episode": 3, "type": "grasp", "occurrence": 0, "frame": 79, "reason": "..."}],
     "add":    [{"episode": 5, "type": "regrasp", "frame": 120, "reason": "..."}],
     "remove": [{"episode": 7, "type": "release", "occurrence": 1, "reason": "..."}],
     "accept": [{"episode": 9, "check": "interval", "reason": "operator paused before lifting"},
                {"episode": "*", "check": "clipped", "reason": "recording stops right after the release"}],
     "exclude_episodes": {"12": "grasp failed and was redone by hand"}}

Usage:
    python -m vla_tools.keyframe_detect --dataset <dir> --spec robot_spec.json --keyframes-spec spec.json \
        [--overrides overrides.json] --out keyframes.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np
import pandas as pd

from vla_tools.episodes import load_episodes
from vla_tools.keyframe_detectors import detect_episode
from vla_tools.keyframe_spec import KeyframeSpec
from vla_tools.robot_spec import RobotSpec


@dataclasses.dataclass(frozen=True)
class Keyframe:
    type: str
    frame: int
    window: tuple[int, int]  # [start, end) action indices
    source: str  # "detector" or "override: <reason>"


@dataclasses.dataclass(frozen=True)
class Issue:
    episode: int
    check: str
    detail: str
    accepted: str  # reason from the overrides file, or ""


def _apply_overrides(found: dict[str, list[int]], episode: int, overrides: dict) -> dict[str, str]:
    """Edit `found` in place; return frame-key -> reason for every overridden keyframe."""
    reasons: dict[str, str] = {}
    for item in overrides.get("set", []):
        if item["episode"] == episode:
            frames = found.setdefault(item["type"], [])
            k = int(item.get("occurrence", 0))
            if k < len(frames):
                frames[k] = int(item["frame"])
            else:
                frames.append(int(item["frame"]))
            reasons[f"{item['type']}:{item['frame']}"] = item["reason"]
    for item in overrides.get("add", []):
        if item["episode"] == episode:
            found.setdefault(item["type"], []).append(int(item["frame"]))
            reasons[f"{item['type']}:{item['frame']}"] = item["reason"]
    for item in overrides.get("remove", []):
        if item["episode"] == episode:
            frames = found.get(item["type"], [])
            k = int(item.get("occurrence", 0))
            if k < len(frames):
                frames.pop(k)
    for frames in found.values():
        frames.sort()
    return reasons


def _windows(kspec: KeyframeSpec, found: dict[str, list[int]], length: int, reasons: dict[str, str]) -> list[Keyframe]:
    out = []
    for ktype in kspec.types:
        for frame in found.get(ktype.name, []):
            end = frame + ktype.window.after
            if ktype.window.until is not None:
                later = [f for f in found.get(ktype.window.until, []) if f > frame]
                end = later[0] + ktype.window.after if later else end
            reason = reasons.get(f"{ktype.name}:{frame}")
            out.append(
                Keyframe(
                    type=ktype.name,
                    frame=int(frame),
                    window=(int(frame - ktype.window.before), int(end)),
                    source=f"override: {reason}" if reason else "detector",
                )
            )
    return sorted(out, key=lambda k: (k.frame, k.type))


def _robust_outliers(values: dict[int, float], z: float) -> dict[int, float]:
    if len(values) < 4:
        return {}
    array = np.array(list(values.values()))
    median = np.median(array)
    mad = 1.4826 * np.median(np.abs(array - median)) + 1e-9
    return {ep: (v - median) / mad for ep, v in values.items() if abs(v - median) / mad > z}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--keyframes-spec", type=Path, required=True)
    parser.add_argument("--overrides", type=Path, default=None)
    parser.add_argument("--outlier-z", type=float, default=3.5)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    spec, kspec = RobotSpec.load(args.spec), KeyframeSpec.load(args.keyframes_spec)
    overrides = json.loads(args.overrides.read_text(encoding="utf-8")) if args.overrides else {}
    excluded = {int(k): v for k, v in overrides.get("exclude_episodes", {}).items()}
    accepted = {(str(a["episode"]), a["check"]): a["reason"] for a in overrides.get("accept", [])}

    def acceptance(episode: int, check: str) -> str:
        return accepted.get((str(episode), check)) or accepted.get(("*", check), "")

    result: dict[str, dict] = {}
    issues: list[Issue] = []
    phases: dict[str, dict[int, float]] = {t.name: {} for t in kspec.types}
    intervals: dict[str, dict[int, float]] = {}
    for e in load_episodes(args.dataset, spec):
        if e.index in excluded:
            continue
        n = len(e.actions)
        extra = {}
        if kspec.columns():
            table = pd.read_parquet(e.path, columns=kspec.columns())
            extra = {c: np.stack(table[c].to_numpy()).astype(np.float64) for c in kspec.columns()}
        found = detect_episode(kspec, e.actions, e.states, spec, extra)
        reasons = _apply_overrides(found, e.index, overrides)
        keyframes = _windows(kspec, found, n, reasons)

        def issue(check: str, detail: str, episode: int = e.index) -> None:
            issues.append(Issue(episode, check, detail, acceptance(episode, check)))

        for ktype in kspec.types:
            frames = found.get(ktype.name, [])
            if ktype.expected >= 0 and len(frames) != ktype.expected:
                issue("count", f"{ktype.name}: {len(frames)} found, {ktype.expected} expected")
            if frames:
                phases[ktype.name][e.index] = frames[0] / max(n - 1, 1)
        firsts = [found[name][0] for name in kspec.order if found.get(name)]
        if firsts != sorted(firsts):
            issue("order", " < ".join(f"{name}@{found[name][0]}" for name in kspec.order if found.get(name)))
        for a, b in zip(kspec.order[:-1], kspec.order[1:], strict=True):
            if found.get(a) and found.get(b):
                intervals.setdefault(f"{a}->{b}", {})[e.index] = float(found[b][0] - found[a][0])
        for k in keyframes:
            if k.window[0] < 0 or k.window[1] > n - 1:
                issue("clipped", f"{k.type}@{k.frame} window {list(k.window)} in {n} frames")
        result[str(e.index)] = {"length": n, "keyframes": [dataclasses.asdict(k) for k in keyframes]}

    for name, values in phases.items():
        for ep, z in _robust_outliers(values, args.outlier_z).items():
            issues.append(
                Issue(
                    ep,
                    "phase",
                    f"{name} at {values[ep]:.2f} of the episode (z {z:+.1f})",
                    acceptance(ep, "phase"),
                )
            )
    for name, values in intervals.items():
        for ep, z in _robust_outliers(values, args.outlier_z).items():
            issues.append(
                Issue(ep, "interval", f"{name} {values[ep]:.0f} frames (z {z:+.1f})", acceptance(ep, "interval"))
            )

    print(f"{len(result)} episodes ({len(excluded)} excluded); task: {kspec.task}; effector: {kspec.effector}")
    print(f"{'type':<14} {'episodes':>8} {'count ok':>8} {'phase med [p10, p90]':>22}")
    for ktype in kspec.types:
        values = np.array(list(phases[ktype.name].values()))
        count_ok = sum(
            1
            for ep in result.values()
            if ktype.expected < 0 or sum(k["type"] == ktype.name for k in ep["keyframes"]) == ktype.expected
        )
        phase = (
            f"{np.median(values):.2f} [{np.percentile(values, 10):.2f}, {np.percentile(values, 90):.2f}]"
            if len(values)
            else "-"
        )
        print(f"{ktype.name:<14} {len(values):>8} {count_ok:>8} {phase:>22}")
    for name, values in intervals.items():
        v = np.array(list(values.values()))
        print(f"interval {name}: median {np.median(v):.0f} frames [{v.min():.0f}, {v.max():.0f}]")
    open_issues = [i for i in issues if not i.accepted]
    print(f"\nissues: {len(issues)} ({len(open_issues)} open)")
    for i in sorted(issues, key=lambda i: (i.episode, i.check)):
        print(
            f"  episode {i.episode:>4} {i.check:<9} {i.detail}" + (f"   accepted: {i.accepted}" if i.accepted else "")
        )

    payload = {
        "dataset": str(args.dataset),
        "keyframes_spec": json.loads(args.keyframes_spec.read_text(encoding="utf-8")),
        "overrides": overrides,
        "excluded": {str(k): v for k, v in excluded.items()},
        "episodes": result,
        "issues": [dataclasses.asdict(i) for i in issues],
    }
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
