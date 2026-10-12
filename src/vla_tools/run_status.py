"""Status of each data stage of a run, S0 scope to S7 labels, read from the run's artifacts. Read-only.

    passed   the stage's artifacts exist and every machine-checkable part of its gate holds
    open     the artifacts exist but a gate check fails; the reason names it
    missing  an artifact the stage writes is absent
    waiting  decisions.md holds a pending decision for the stage

The checks are the gates in the vla-pipeline skill that files can show: dataset lineage (each version
derived from the previous one), the index convention, written decisions, open keyframe issues, review
sheets present for every keyframe type, a held-out split covering every cluster, the per-frame baseline
among the label candidates, train and eval label sets consistent with the split. Judgments made by
looking (review images, cluster explanations) stay with the person and the agent.

Dataset paths recorded in artifacts are resolved against the project root (the parent of runs/), the
directory the tools are run from. Downstream (S8) and close (S9) are not tracked here.

Usage (from the project root):
    python -m vla_tools.run_status runs/<task>
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from collections.abc import Callable
from pathlib import Path
from typing import Literal

from vla_tools.decision_log import Decision
from vla_tools.decision_log import load_decisions
from vla_tools.robot_spec import RobotSpec
from vla_tools.run_index import RunIndex

Status = Literal["passed", "open", "missing", "waiting"]


@dataclasses.dataclass(frozen=True)
class StageStatus:
    stage: str
    name: str
    status: Status
    artifact: str
    reason: str


@dataclasses.dataclass(frozen=True)
class Check:
    status: Status
    artifact: str
    reason: str


class RunChecks:
    """One method per data stage, each judging the stage from its artifacts alone."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = Path(run_dir)
        self.index = RunIndex.load(self.run_dir) if RunIndex.file(self.run_dir).exists() else None

    def _json(self, path: Path) -> dict | None:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    def _path(self, recorded: str) -> Path:
        return RunIndex.resolve(self.run_dir, recorded)

    def _same(self, recorded: str, expected: str | None) -> bool:
        return expected is not None and self._path(recorded) == self._path(expected)

    def _holdout(self) -> list[int] | None:
        diversity = self._json(self.run_dir / "diversity.json")
        split = diversity.get("split") if diversity else None
        return sorted(split["holdout"]) if split else None

    def s0(self) -> Check:
        if self.index is None:
            return Check("missing", "run.json", "no run index")
        spec_path = self._path(self.index.spec)
        if not spec_path.exists():
            return Check("missing", self.index.spec, "no robot spec")
        try:
            spec = RobotSpec.load(spec_path)
        except (ValueError, KeyError) as error:
            return Check("open", self.index.spec, f"invalid robot spec: {error}")
        if not (self._path(self.index.source) / "meta" / "info.json").exists():
            return Check("open", "run.json", f"source {self.index.source} is not a dataset")
        return Check("passed", "run.json, robot_spec.json",
                     f"{spec.action_dim} action dims: joints {list(spec.joint_dims)}, commands {list(spec.command_dims)}")  # fmt: skip

    def s1(self) -> Check:
        audit = self._json(self.run_dir / "audit.json")
        if audit is None:
            return Check("missing", "audit.json", "no audit")
        nxt, same = audit["next_state_error_max"], audit["same_state_error_max"]
        if not nxt < same:
            return Check(
                "open",
                "audit.json",
                f"index convention not confirmed: |a[t]-s[t+1]| {nxt:.1e} >= |a[t]-s[t]| {same:.1e}",
            )
        kinds = sorted({f["kind"] for f in audit["findings"]})
        return Check(
            "passed",
            "audit.json",
            f"action[t] = state[t+1] ({nxt:.1e} vs {same:.1e}); findings: {', '.join(kinds) or 'none'}",
        )

    def s2(self) -> Check:
        if self.index is None or self.index.clean is None:
            return Check("missing", "run.json datasets.clean", "no cleaned version recorded")
        report = self._json(self._path(self.index.clean) / "meta" / "cleaning_report.json")
        if report is None:
            return Check("missing", "meta/cleaning_report.json", f"{self.index.clean} has no cleaning report")
        if not self._same(report["source_dataset"], self.index.source):
            return Check(
                "open", "meta/cleaning_report.json", f"cleaned from {report['source_dataset']}, not the source"
            )
        audit = self._json(self.run_dir / "audit.json")
        if audit is None:
            return Check("open", "audit.json", "no audit to check the jump decisions against")
        excluded = {int(k) for k in report["excluded"]}
        decided = set(report["plan"].get("jump_decisions", {}))
        undecided = [
            f"{f['episode']}:{f['frame']}"
            for f in audit["findings"]
            if f["kind"] in ("jump_continuing", "jump_discontinuity") and f["episode"] not in excluded
            and f"{f['episode']}:{f['frame']}" not in decided
        ]  # fmt: skip
        if undecided:
            return Check("open", "meta/cleaning_report.json", f"jumps without a written decision: {undecided}")
        repaired = sum(len(e["repaired_jumps"]) + len(e["removed_pulses"]) for e in report["episodes"])
        return Check("passed", "meta/cleaning_report.json",
                     f"{len(report['episodes'])} episodes, {len(excluded)} excluded; {repaired} repairs; {len(decided)} jump decisions")  # fmt: skip

    def s3(self) -> Check:
        record = self._json(self.run_dir / "keyframes_pass1.json")
        if record is None:
            return Check("missing", "keyframes_pass1.json", "no first keyframe pass")
        if self.index is None or not self._same(record["dataset"], self.index.clean):
            return Check("open", "keyframes_pass1.json", f"detected on {record['dataset']}, not the cleaned version")
        return Check("passed", "keyframes_pass1.json", "on the cleaned version; classifies still runs only")

    def s4(self) -> Check:
        if self.index is None or self.index.timed is None:
            return Check("missing", "run.json datasets.timed", "no timed version recorded")
        report = self._json(self._path(self.index.timed) / "meta" / "timing_report.json")
        if report is None:
            return Check("missing", "meta/timing_report.json", f"{self.index.timed} has no timing report")
        if not self._same(report["source_dataset"], self.index.clean):
            return Check(
                "open", "meta/timing_report.json", f"timed from {report['source_dataset']}, not the cleaned version"
            )
        plan = report["plan"]
        unexplained = [
            k for k in ("leading_keep", "pause_keep", "trailing_keep", "end_padding")
            if plan.get(k) not in (None, 0) and k not in plan.get("reasons", {})
        ]  # fmt: skip
        if unexplained:
            return Check("open", "meta/timing_report.json", f"plan entries without a reason: {unexplained}")
        dropped = sum(e["dropped"] for e in report["episodes"].values())
        return Check("passed", "meta/timing_report.json",
                     f"{dropped} still frames dropped, {plan.get('end_padding', 0)} padding rows per episode")  # fmt: skip

    def s5(self) -> Check:
        record = self._json(self.run_dir / "keyframes.json")
        if record is None:
            return Check("missing", "keyframes.json", "no final keyframes")
        if self.index is None or not self._same(record["dataset"], self.index.timed):
            return Check("open", "keyframes.json", f"detected on {record['dataset']}, not the timed version")
        open_issues = [i for i in record["issues"] if not i["accepted"]]
        if open_issues:
            return Check("open", "keyframes.json", f"{len(open_issues)} open issues")
        types = [t["name"] for t in record["keyframes_spec"]["types"]]
        unreviewed = [t for t in types if not list((self.run_dir / "kf_review").glob(f"{t}_page*.png"))]
        if unreviewed:
            return Check("open", "kf_review/", f"no review sheets for: {unreviewed}")
        return Check("passed", "keyframes.json, kf_review/", f"0 open issues; review sheets for {types}")

    def s6(self) -> Check:
        diversity = self._json(self.run_dir / "diversity.json")
        if diversity is None:
            return Check("missing", "diversity.json", "no diversity report")
        if self.index is None or not self._same(diversity["dataset"], self.index.timed):
            return Check("open", "diversity.json", f"computed on {diversity['dataset']}, not the timed version")
        split = diversity["split"]
        if split is None:
            return Check("open", "diversity.json", "no held-out split")
        if split["clusters_missing_from_train"]:
            return Check(
                "open", "diversity.json", f"clusters with no training episode: {split['clusters_missing_from_train']}"
            )
        clusters = len(set(diversity["combined_clusters"]))
        return Check(
            "passed",
            "diversity.json",
            f"{clusters} combined clusters; holdout {split['holdout']} leaves every cluster in training",
        )

    def s7(self) -> Check:
        sweep = self._json(self.run_dir / "sweep.json")
        if sweep is None:
            return Check("missing", "sweep.json", "no label sweep")
        if self.index is None or not self._same(sweep["dataset"], self.index.timed):
            return Check("open", "sweep.json", f"swept on {sweep['dataset']}, not the timed version")
        horizons = {int(c["config"]["action_horizon"]) for c in sweep["cells"]}
        baseline = any(
            c["config"]["metric"] == "time" and c["config"]["span_frames"] in horizons for c in sweep["cells"]
        )
        if not baseline or not any(int(span) in horizons for span in sweep["shortlist"]):
            return Check("open", "sweep.json", "the per-frame baseline (time, span = H) is not among the candidates")
        if not self.index.labels:
            return Check("missing", "run.json datasets.labels", "no label datasets recorded")
        holdout = self._holdout()
        problems: list[str] = []
        for name, versions in self.index.labels.items():
            train = self._json(self._path(versions.train) / "meta" / "arc_resample.json")
            evaluation = self._json(self._path(versions.eval) / "meta" / "arc_resample.json")
            if train is None or evaluation is None:
                problems.append(f"{name}: no arc_resample.json")
                continue
            if not (
                self._same(train["source_dataset"], self.index.timed)
                and self._same(evaluation["source_dataset"], self.index.timed)
            ):
                problems.append(f"{name}: not built from the timed version")
            if holdout is None or train["excluded_source_episodes"] != holdout:
                problems.append(f"{name}: train excludes {train['excluded_source_episodes']}, holdout is {holdout}")
            if evaluation["excluded_source_episodes"]:
                problems.append(f"{name}: eval excludes {evaluation['excluded_source_episodes']}")
            if not (self.run_dir / f"replay_{name}.json").exists():
                problems.append(f"{name}: no replay_{name}.json to choose the execution length from")
        if problems:
            return Check("open", "label datasets", "; ".join(problems))
        return Check(
            "passed",
            "sweep.json, label datasets",
            f"labels {sorted(self.index.labels)}; train excludes holdout {holdout}",
        )


def _with_decisions(stage: str, check: Check, decisions: list[Decision]) -> Check:
    pending = [d for d in decisions if d.stage == stage and d.status == "pending"]
    if pending:
        return Check("waiting", "decisions.md", "; ".join(f"{d.id} pending: {d.item}" for d in pending))
    unwritten = [d.id for d in decisions if d.stage == stage and not d.complete]
    if unwritten:
        return Check("open", "decisions.md", f"decided but not written into a file: {unwritten}")
    return check


def stage_statuses(run_dir: Path) -> list[StageStatus]:
    checks = RunChecks(run_dir)
    decisions = load_decisions(Path(run_dir) / "decisions.md")
    stages: tuple[tuple[str, str, Callable[[], Check]], ...] = (
        ("S0", "scope", checks.s0),
        ("S1", "audit", checks.s1),
        ("S2", "clean", checks.s2),
        ("S3", "keyframes pass 1", checks.s3),
        ("S4", "still frames, end padding", checks.s4),
        ("S5", "keyframes", checks.s5),
        ("S6", "diversity, held-out split", checks.s6),
        ("S7", "action labels", checks.s7),
    )
    out = []
    for stage, name, check_stage in stages:
        check = _with_decisions(stage, check_stage(), decisions)
        out.append(StageStatus(stage, name, check.status, check.artifact, check.reason))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()

    statuses = stage_statuses(args.run_dir)
    print(f"{'stage':<5} {'name':<26} {'status':<8} {'artifact':<28} reason")
    for s in statuses:
        print(f"{s.stage:<5} {s.name:<26} {s.status:<8} {s.artifact:<28} {s.reason}")
    first = next((s for s in statuses if s.status != "passed"), None)
    print(f"\nnext: {first.stage} {first.name} ({first.status})" if first else "\nnext: all data stages passed")
    report = args.run_dir / "data_report.md"
    print(f"data_report.md: {'present' if report.exists() else 'not written'}")


if __name__ == "__main__":
    main()
