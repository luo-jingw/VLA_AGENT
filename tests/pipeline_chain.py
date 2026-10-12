"""Run data stages S0-S7 on the synthetic dataset with the commands the skills document, writing what each stage writes.

Layout follows `.claude/skills/vla-pipeline/references/artifacts.md`: datasets under <project>/data/,
plans, reports and review images under <project>/runs/<task>/.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from synthetic_dataset import CAMERA
from synthetic_dataset import SyntheticDataset
from synthetic_dataset import write_pick_place
from synthetic_dataset import write_spec
from tool_runner import run_tool

TASK = "synthetic_pick"
EPISODES = 10
HOLDOUT = 2
HORIZON = 10
SPAN = 20
LABELS = f"time-s{SPAN}"

KEYFRAMES_SPEC = {
    "task": "pick the block and place it in the bin",
    "effector": "parallel gripper, binary close command, no feedback",
    "order": ["grasp", "lift", "release"],
    "types": [
        {
            "name": "grasp",
            "description": "close command issued",
            "detector": {"type": "command_edge", "signal": {"source": "action", "dim": 3}, "direction": "rise"},
            "window": {"before": 4, "after": 4},
            "expected": 1,
        },
        {
            "name": "lift",
            "description": "arm starts moving after the grasp",
            "detector": {
                "type": "motion_onset",
                "signal": {"source": "joint_speed"},
                "threshold": 0.001,
                "hold_frames": 2,
                "reference": "grasp",
                "search_frames": 20,
            },
            "window": {"before": 1, "after": 4},
            "expected": 1,
        },
        {
            "name": "release",
            "description": "open command issued",
            "detector": {"type": "command_edge", "signal": {"source": "action", "dim": 3}, "direction": "fall"},
            "window": {"before": 3, "after": 4},
            "expected": 1,
        },
    ],
}
TIMING_PLAN = {
    "threshold": 0.001,
    "min_frames": 4,
    "leading_keep": 2,
    "pause_keep": None,
    "trailing_keep": None,
    "end_padding": 10,
    "reasons": {
        "leading_keep": "the operator had not started; two frames keep the start pose visible",
        "end_padding": "recordings stop three frames after the release; padding covers one window",
    },
}
LABEL_CONFIG = {
    "metric": "time",
    "span_frames": SPAN,
    "action_horizon": HORIZON,
    "event_density": 0.0,
    "key_weight": 2.0,
    "pad_weight": 1.0,
    "deadzone": 0.0,
    "anchor": "state",
}


@dataclasses.dataclass(frozen=True)
class ChainRun:
    project: Path
    run_dir: Path
    source: SyntheticDataset
    spec: Path
    clean: Path
    timed: Path
    labels_train: Path
    labels_eval: Path
    keyframes_pass1: Path
    keyframes: Path
    holdout: list[int]
    clean_refusal: str  # output of clean run without jump decisions
    timing_report_mode: str  # output of episode_timing without a plan


def _write_json(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def run_chain(project: Path) -> ChainRun:
    data, run = project / "data", project / "runs" / TASK
    run.mkdir(parents=True)

    # S0 scope
    source = write_pick_place(data / "synth", EPISODES, seed=0, faults=True)
    spec = write_spec(run / "robot_spec.json")

    # S1 audit
    run_tool("audit", "--dataset", source.root, "--spec", spec, "--out", run / "audit.json",
             "--review-dir", run / "audit_review")  # fmt: skip
    findings = json.loads((run / "audit.json").read_text(encoding="utf-8"))["findings"]
    to_decide = [f for f in findings if f["kind"] in ("jump_continuing", "jump_discontinuity")]

    # S2 clean: refused until every non-isolated jump has a decision
    plan = {"jump_factor": 8.0, "repair_isolated_jumps": True, "command_pulse_frames": 3, "exclude_episodes": {}}
    clean = data / "synth-clean"
    refusal = run_tool("clean", "--dataset", source.root, "--spec", spec,
                       "--plan", _write_json(run / "cleaning_plan.json", plan), "--out", clean, expect_code=2)  # fmt: skip
    plan["jump_decisions"] = {
        f"{f['episode']}:{f['frame']}": "keep: fast joint 2 motion, traces checked in audit_review" for f in to_decide
    }
    run_tool("clean", "--dataset", source.root, "--spec", spec,
             "--plan", _write_json(run / "cleaning_plan.json", plan), "--out", clean)  # fmt: skip

    # S3 first keyframe pass
    kspec = _write_json(run / "keyframes_spec.json", KEYFRAMES_SPEC)
    pass1 = run / "keyframes_pass1.json"
    run_tool("keyframe_detect", "--dataset", clean, "--spec", spec, "--keyframes-spec", kspec, "--out", pass1)

    # S4 still frames and end padding
    timed = data / "synth-timed"
    report_mode = run_tool("episode_timing", "--dataset", clean, "--spec", spec, "--keyframes", pass1,
                           "--threshold", "0.001", "--min-frames", "4")  # fmt: skip
    run_tool("episode_timing", "--dataset", clean, "--spec", spec, "--keyframes", pass1,
             "--plan", _write_json(run / "timing_plan.json", TIMING_PLAN), "--out", timed)  # fmt: skip

    # S5 final keyframes and review sheets
    keyframes = run / "keyframes.json"
    run_tool("keyframe_detect", "--dataset", timed, "--spec", spec, "--keyframes-spec", kspec, "--out", keyframes)
    run_tool("keyframe_review", "--dataset", timed, "--spec", spec, "--keyframes", keyframes,
             "--type", "grasp", "--camera", CAMERA, "--out", run / "kf_review")  # fmt: skip

    # S6 diversity and held-out split
    run_tool("diversity", "--dataset", timed, "--spec", spec, "--keyframes", keyframes, "--holdout", str(HOLDOUT),
             "--out", run / "diversity.json", "--plot", run / "diversity.png", "--montage", run / "montage.png")  # fmt: skip
    holdout = json.loads((run / "diversity.json").read_text(encoding="utf-8"))["split"]["holdout"]

    # S7 labels
    run_tool("sweep", "--dataset", timed, "--spec", spec, "--keyframes", keyframes, "--horizon", str(HORIZON),
             "--metrics", "time", "curvature", "--spans", str(HORIZON), str(SPAN), "--out", run / "sweep.json")  # fmt: skip
    config = _write_json(run / "label_config.json", LABEL_CONFIG)
    run_tool("label_replay", "--dataset", timed, "--spec", spec, "--keyframes", keyframes, "--config", config,
             "--execute", "4", "8", "--out", run / f"replay_{LABELS}.json")  # fmt: skip
    train, evaluation = data / f"synth-{LABELS}-train", data / f"synth-{LABELS}-eval"
    run_tool("build_labels", "--dataset", timed, "--spec", spec, "--keyframes", keyframes, "--config", config,
             "--exclude", *(str(e) for e in holdout), "--out", train)  # fmt: skip
    run_tool("build_labels", "--dataset", timed, "--spec", spec, "--keyframes", keyframes, "--config", config,
             "--out", evaluation)  # fmt: skip

    return ChainRun(
        project=project,
        run_dir=run,
        source=source,
        spec=spec,
        clean=clean,
        timed=timed,
        labels_train=train,
        labels_eval=evaluation,
        keyframes_pass1=pass1,
        keyframes=keyframes,
        holdout=holdout,
        clean_refusal=refusal,
        timing_report_mode=report_mode,
    )
