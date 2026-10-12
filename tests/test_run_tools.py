"""run_index, decision_log, run_status and data_report on the synthetic S0-S7 run and on copies with one gate broken."""

from __future__ import annotations

import json
import re
import shutil
from collections.abc import Callable
from pathlib import Path

import pytest

from pipeline_chain import TASK
from pipeline_chain import ChainRun
from tool_runner import run_tool
from vla_tools.decision_log import load_decisions
from vla_tools.run_index import RunIndex
from vla_tools.run_status import StageStatus
from vla_tools.run_status import stage_statuses


def _rows(output: str) -> dict[str, str]:
    """run_status output -> stage -> its status word."""
    rows = (re.match(r"(S\d)\s.*?\s(passed|open|missing|waiting)\s", line) for line in output.splitlines())
    return {m.group(1): m.group(2) for m in rows if m}


def _copy_run(chain: ChainRun, tmp_path: Path) -> Path:
    """The run's directory copied into a new project whose data/ points at the chain's datasets."""
    project = tmp_path / "project"
    shutil.copytree(chain.run_dir, project / "runs" / TASK)
    (project / "data").symlink_to(chain.project / "data")
    return project / "runs" / TASK


def _status(run_dir: Path, stage: str) -> StageStatus:
    return next(s for s in stage_statuses(run_dir) if s.stage == stage)


def _edit_json(path: Path, edit: Callable[[dict], None]) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    edit(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_status_waits_on_a_pending_decision(chain: ChainRun) -> None:
    rows = _rows(chain.status_waiting)
    print(chain.status_waiting)
    assert rows["S0"] == rows["S1"] == "passed"
    assert rows["S2"] == "waiting"
    assert "D-001 pending" in chain.status_waiting
    assert "next: S2 clean (waiting)" in chain.status_waiting


def test_status_after_the_last_stage(chain: ChainRun) -> None:
    rows = _rows(chain.status_final)
    print(chain.status_final)
    assert rows == {f"S{i}": "passed" for i in range(8)}
    assert "next: all data stages passed" in chain.status_final
    assert "data_report.md: present" in chain.status_final


def _audit_convention_unconfirmed(run: Path) -> None:
    _edit_json(run / "audit.json", lambda a: a.update(next_state_error_max=1.0))


def _decision_not_written(run: Path) -> None:
    path = run / "decisions.md"
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            "- written_to: runs/synthetic_pick/cleaning_plan.json", "- written_to:"
        )
    )


def _first_pass_removed(run: Path) -> None:
    (run / "keyframes_pass1.json").unlink()


def _open_keyframe_issue(run: Path) -> None:
    issue = {"episode": 0, "check": "count", "detail": "grasp: 2 found, 1 expected", "accepted": ""}
    _edit_json(run / "keyframes.json", lambda k: k["issues"].append(issue))


def _review_sheets_missing(run: Path) -> None:
    for page in (run / "kf_review").glob("lift_page*.png"):
        page.unlink()


def _cluster_missing_from_train(run: Path) -> None:
    _edit_json(run / "diversity.json", lambda d: d["split"].update(clusters_missing_from_train=["motion cluster 2"]))


def _train_and_eval_swapped(run: Path) -> None:
    def swap(index: dict) -> None:
        versions = index["datasets"]["labels"]["time-s20"]
        versions["train"], versions["eval"] = versions["eval"], versions["train"]

    _edit_json(run / "run.json", swap)


def _replay_removed(run: Path) -> None:
    (run / "replay_time-s20.json").unlink()


@pytest.mark.parametrize(
    ("break_gate", "stage", "status", "reason"),
    [
        (_audit_convention_unconfirmed, "S1", "open", "index convention not confirmed"),
        (_decision_not_written, "S2", "open", "decided but not written"),
        (_first_pass_removed, "S3", "missing", "no first keyframe pass"),
        (_open_keyframe_issue, "S5", "open", "1 open issues"),
        (_review_sheets_missing, "S5", "open", "no review sheets for: ['lift']"),
        (_cluster_missing_from_train, "S6", "open", "clusters with no training episode"),
        (_train_and_eval_swapped, "S7", "open", "train excludes []"),
        (_replay_removed, "S7", "open", "no replay_time-s20.json"),
    ],
)
def test_status_reports_each_broken_gate(
    chain: ChainRun, tmp_path: Path, break_gate: Callable[[Path], None], stage: str, status: str, reason: str
) -> None:
    run = _copy_run(chain, tmp_path)
    assert all(s.status == "passed" for s in stage_statuses(run))
    break_gate(run)
    found = _status(run, stage)
    print(f"{found.stage} {found.status}: {found.reason}")
    assert found.status == status
    assert reason in found.reason


def test_data_report_lists_versions_decisions_and_coverage(chain: ChainRun) -> None:
    report = (chain.run_dir / "data_report.md").read_text(encoding="utf-8")
    index = RunIndex.load(chain.run_dir)
    versions = [
        index.source,
        index.clean,
        index.timed,
        *(v for pair in index.labels.values() for v in (pair.train, pair.eval)),
    ]
    assert all(f"| {v} |" in report for v in versions)
    plan = json.loads((chain.run_dir / "cleaning_plan.json").read_text(encoding="utf-8"))
    assert all(f"| {key} |" in report for key in plan["jump_decisions"])
    diversity = json.loads((chain.run_dir / "diversity.json").read_text(encoding="utf-8"))
    assert all(f"| {group} |" in report for group in diversity["groups"])
    assert f"{chain.holdout}" in report
    assert all(f"| {d.id} |" in report for d in load_decisions(chain.run_dir / "decisions.md"))
    for heading in (
        "## 数据集版本",
        "## 体检与清洗",
        "## 关键帧",
        "## 静止帧与末尾补帧",
        "## 覆盖与留出集",
        "## 动作标签",
        "## 人的决定",
    ):
        assert heading in report


def test_data_report_refuses_until_every_stage_passes(chain: ChainRun, tmp_path: Path) -> None:
    run = _copy_run(chain, tmp_path)
    (run / "data_report.md").unlink()
    (run / "keyframes.json").unlink()
    output = run_tool("data_report", run, expect_code=2)
    assert "S5 keyframes: missing" in output
    assert not (run / "data_report.md").exists()


def test_run_index_refuses_what_is_not_a_dataset(tmp_path: Path) -> None:
    run = tmp_path / "runs" / "t"
    run.mkdir(parents=True)
    spec = run / "robot_spec.json"
    spec.write_text("{}", encoding="utf-8")
    assert "needs --task" in run_tool("run_index", run, "--source", tmp_path, expect_code=1)
    output = run_tool("run_index", run, "--task", "t", "--spec", spec, "--source", tmp_path / "data", expect_code=1)
    assert "is not a LeRobot dataset" in output
    assert not RunIndex.file(run).exists()


def test_decision_log_rejects_malformed_entries(tmp_path: Path) -> None:
    path = tmp_path / "decisions.md"
    path.write_text(
        "## D-001\n- stage: S2\n- status: maybe\n- item: x\n- evidence: x\n- proposal: x\n- decision:\n- written_to:\n"
    )
    with pytest.raises(ValueError, match="pending or decided"):
        load_decisions(path)
    path.write_text("## D-001\n- stage: S2\n- status: pending\n")
    with pytest.raises(ValueError, match="lacks"):
        load_decisions(path)
    assert load_decisions(tmp_path / "absent.md") == []
