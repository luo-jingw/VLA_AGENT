"""Apply a cleaning plan and write a cleaned dataset version. The source dataset is never modified.

Cleaning repairs values and drops episodes; it never adds or removes rows (idle frames and end padding
are episode_timing's job). Repairs that are safe by construction run when enabled; everything that
needs judgment needs a written decision, and the tool refuses to write until every such finding has one.

cleaning_plan.json:
    {
      "jump_factor": 8.0,
      "repair_isolated_jumps": true,          # isolated jumps come straight back: a glitch
      "jump_decisions": {"3:178": "keep: real fast wrist motion, frames 176-180 checked"},
                                              # required for every continuing / discontinuity jump
                                              # whose episode is not excluded; value starts with
                                              # "keep:" or "repair:" (interpolate across it)
      "command_pulse_frames": 3,              # remove command runs shorter than this; 0 disables
      "exclude_episodes": {"12": "operator restarted the task mid-episode"}
    }

Usage:
    python -m vla_tools.clean --dataset <dir> --spec robot_spec.json --plan cleaning_plan.json --out <new_dir>
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

from vla_tools import cleaning_ops
from vla_tools import dataset_writer
from vla_tools.episodes import load_episodes
from vla_tools.episodes import median_step
from vla_tools.robot_spec import RobotSpec


@dataclasses.dataclass(frozen=True)
class EpisodeLog:
    source: int
    repaired_jumps: list[int]
    kept_jumps: list[int]
    removed_pulses: list[list[int]]  # [dim, start, end)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    spec = RobotSpec.load(args.spec)
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    excluded = {int(k): v for k, v in plan.get("exclude_episodes", {}).items()}
    decisions = dict(plan.get("jump_decisions", {}))
    episodes = load_episodes(args.dataset, spec)
    step = median_step(episodes, spec)
    factor = float(plan.get("jump_factor", 8.0))

    undecided = [
        f"{e.index}:{j.frame} ({j.kind}, dim {j.dim}, {j.steps:.1f} steps)"
        for e in episodes
        if e.index not in excluded
        for j in cleaning_ops.find_jumps(e.actions, spec, step, factor)
        if j.kind != "isolated" and f"{e.index}:{j.frame}" not in decisions
    ]
    malformed = [k for k, v in decisions.items() if not v.startswith(("keep:", "repair:"))]
    if undecided or malformed:
        for item in undecided:
            print(f"needs a decision in jump_decisions: {item}")
        for item in malformed:
            print(f"decision must start with 'keep:' or 'repair:': {item}")
        sys.exit(2)

    plans, logs = [], []
    for e in episodes:
        if e.index in excluded:
            continue
        actions, states = e.actions.copy(), e.states.copy()
        repaired, kept = [], []
        for jump in cleaning_ops.find_jumps(e.actions, spec, step, factor):
            decision = decisions.get(f"{e.index}:{jump.frame}", "")
            isolated_ok = jump.kind == "isolated" and plan.get("repair_isolated_jumps", False)
            if (isolated_ok or decision.startswith("repair:")) and 0 < jump.frame < len(actions) - 1:
                cleaning_ops.repair_isolated(actions, states, spec, jump.frame)
                repaired.append(jump.frame)
            else:
                kept.append(jump.frame)
        pulses = []
        if int(plan.get("command_pulse_frames", 0)) > 0:
            for dim in spec.command_dims:
                for pulse in cleaning_ops.command_pulses(actions, dim, int(plan["command_pulse_frames"])):
                    cleaning_ops.remove_pulse(actions, states, spec, dim, pulse)
                    pulses.append([dim, *pulse])
        replace = {spec.action_key: actions, spec.state_key: states} if (repaired or pulses) else {}
        plans.append(dataset_writer.EpisodePlan(source=e.index, replace=replace))
        logs.append(EpisodeLog(e.index, repaired, kept, pulses))
        if repaired or pulses:
            print(f"  episode {e.index}: repaired jumps {repaired}, removed pulses {pulses}")

    report = {
        "plan": plan,
        "excluded": {str(k): v for k, v in excluded.items()},
        "episodes": [dataclasses.asdict(log) for log in logs],
    }
    dataset_writer.write(args.dataset, args.out, plans, {"cleaning_report.json": report})
    print(f"wrote {len(plans)} episodes ({len(excluded)} excluded) to {args.out}")


if __name__ == "__main__":
    main()
