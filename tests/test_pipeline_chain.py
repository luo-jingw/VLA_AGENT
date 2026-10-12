"""Data stages S1-S7 end to end on a synthetic dataset with known faults and keyframes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline_chain import HORIZON
from pipeline_chain import SPAN
from pipeline_chain import TIMING_PLAN
from pipeline_chain import ChainRun
from synthetic_dataset import LEAD_FRAMES
from synthetic_dataset import SPEC
from tool_runner import run_tool
from vla_tools.arc_interp import ArcInterpolator
from vla_tools.episodes import load_episodes

JOINTS = list(SPEC.joint_dims)


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _dropped_leading() -> int:
    """Rows the timing plan drops from every episode: the leading still run minus the frames kept."""
    return (LEAD_FRAMES - 1) - TIMING_PLAN["leading_keep"]


def test_audit_reports_alignment_and_injected_faults(chain: ChainRun) -> None:
    report = _json(chain.run_dir / "audit.json")
    faults = chain.source.faults
    found = {(f["kind"], f["episode"], f["frame"]) for f in report["findings"]}
    print(
        f"next-state error {report['next_state_error_max']:.2e}, same-state error {report['same_state_error_max']:.2e}"
    )
    print(f"findings {sorted(found)}")
    assert report["next_state_error_max"] < 1e-6 < report["same_state_error_max"]
    assert ("jump_isolated", *faults.isolated_jump) in found
    assert ("command_pulse", faults.command_pulse[0], faults.command_pulse[1]) in found
    assert ("jump_continuing", *faults.fast_motion) in found
    episode, frame = faults.isolated_jump
    assert not any(f[1] == episode and f[2] == frame + 1 for f in found), "return of an isolated glitch reported"
    jumps = [f for f in report["findings"] if f["kind"].startswith("jump_")]
    assert len(list((chain.run_dir / "audit_review").glob("*.png"))) == len(jumps)


def test_clean_refuses_undecided_jumps(chain: ChainRun) -> None:
    episode, frame = chain.source.faults.fast_motion
    assert f"{episode}:{frame} (continuing" in chain.clean_refusal
    glitch_episode, glitch_frame = chain.source.faults.isolated_jump
    assert f"{glitch_episode}:{glitch_frame}" not in chain.clean_refusal
    assert f"{glitch_episode}:{glitch_frame + 1}" not in chain.clean_refusal


def test_clean_repairs_glitch_and_pulse_only(chain: ChainRun) -> None:
    source = {e.index: e for e in load_episodes(chain.source.root, SPEC)}
    clean = {e.index: e for e in load_episodes(chain.clean, SPEC)}
    episode, t = chain.source.faults.isolated_jump
    a, s = clean[episode].actions, clean[episode].states
    np.testing.assert_allclose(a[t, JOINTS], 0.5 * (a[t - 1, JOINTS] + a[t + 1, JOINTS]), atol=1e-6)
    np.testing.assert_allclose(s[t + 1, JOINTS], a[t, JOINTS], atol=1e-6)
    pulse_episode, start, end = chain.source.faults.command_pulse
    assert (clean[pulse_episode].actions[start:end, 3] == 0).all()
    assert (clean[pulse_episode].states[start + 1 : end + 1, 3] == 0).all()
    for index, e in source.items():
        assert len(clean[index].actions) == len(e.actions)
        if index not in (episode, pulse_episode):
            np.testing.assert_array_equal(clean[index].actions, e.actions)
    report = _json(chain.clean / "meta" / "cleaning_report.json")
    logs = {log["source"]: log for log in report["episodes"]}
    assert logs[episode]["repaired_jumps"] == [t]
    assert logs[pulse_episode]["removed_pulses"] == [[3, start, end]]


def _first_frames(keyframes: dict, episode: int) -> dict[str, int]:
    return {k["type"]: k["frame"] for k in keyframes["episodes"][str(episode)]["keyframes"]}


def test_first_keyframe_pass_matches_truth_and_flags_clipped_release(chain: ChainRun) -> None:
    pass1 = _json(chain.keyframes_pass1)
    for episode, truth in chain.source.truth.items():
        assert _first_frames(pass1, episode) == {"grasp": truth.grasp, "lift": truth.lift, "release": truth.release}
    checks = {(i["episode"], i["check"]) for i in pass1["issues"]}
    assert checks == {(e, "clipped") for e in chain.source.truth}


def test_timing_classifies_still_runs_and_pads_ends(chain: ChainRun) -> None:
    episodes = len(chain.source.truth)
    for kind in ("leading", "keyframe", "trailing"):
        assert f"{kind:<9} {episodes:>3} runs" in chain.timing_report_mode
    assert "pause       0 runs" in chain.timing_report_mode
    clean = {e.index: e for e in load_episodes(chain.clean, SPEC)}
    pad = TIMING_PLAN["end_padding"]
    for e in load_episodes(chain.timed, SPEC):
        source = clean[e.index]
        assert len(e.actions) == len(source.actions) - _dropped_leading() + pad
        assert e.padding[-pad:].all() and not e.padding[:-pad].any()
        np.testing.assert_array_equal(e.actions[-pad:], np.repeat(source.actions[-1:], pad, axis=0))
        np.testing.assert_array_equal(e.states[-pad:], np.repeat(source.actions[-1:], pad, axis=0))
        np.testing.assert_array_equal(e.actions[: e.real], source.actions[_dropped_leading() :])
        frames = pd.read_parquet(e.path, columns=["frame_index"])["frame_index"].to_numpy()
        np.testing.assert_array_equal(frames, np.arange(len(e.actions)))
    report = _json(chain.timed / "meta" / "timing_report.json")
    assert report["plan"]["reasons"] == TIMING_PLAN["reasons"]


def test_audit_after_padding_reports_the_recorded_rows_only(chain: ChainRun, tmp_path: Path) -> None:
    output = run_tool("audit", "--dataset", chain.timed, "--spec", chain.spec, "--out", tmp_path / "audit.json")
    assert f"end padding present ([{TIMING_PLAN['end_padding']}] rows per episode)" in output
    findings = {(f["kind"], f["episode"], f["frame"]) for f in _json(tmp_path / "audit.json")["findings"]}
    episode, frame = chain.source.faults.fast_motion
    before = {(k, e, f - _dropped_leading()) for k, e, f in _audit_findings(chain) if e == episode}
    print(f"after padding {sorted(findings)}")
    assert findings == before


def _audit_findings(chain: ChainRun) -> set[tuple[str, int, int]]:
    return {(f["kind"], f["episode"], f["frame"]) for f in _json(chain.run_dir / "audit.json")["findings"]}


def test_keyframes_from_another_version_are_refused(chain: ChainRun) -> None:
    output = run_tool("episode_timing", "--dataset", chain.source.root, "--spec", chain.spec,
                      "--keyframes", chain.keyframes_pass1, expect_code=1)  # fmt: skip
    assert "keyframes were detected on" in output


def test_labels_refuse_keyframes_with_open_issues(chain: ChainRun, tmp_path: Path) -> None:
    output = run_tool("build_labels", "--dataset", chain.clean, "--spec", chain.spec, "--keyframes",
                      chain.keyframes_pass1, "--config", chain.run_dir / "label_config.json",
                      "--out", tmp_path / "labels", expect_code=1)  # fmt: skip
    assert "open issues" in output
    assert not (tmp_path / "labels").exists()


def test_final_keyframes_follow_the_dropped_rows(chain: ChainRun) -> None:
    final = _json(chain.keyframes)
    assert final["issues"] == []
    shift = _dropped_leading()
    for episode, truth in chain.source.truth.items():
        expected = {"grasp": truth.grasp - shift, "lift": truth.lift - shift, "release": truth.release - shift}
        assert _first_frames(final, episode) == expected
    assert sorted(p.name for p in (chain.run_dir / "kf_review").glob("*.png"))


def test_diversity_split_covers_both_block_regions(chain: ChainRun) -> None:
    report = _json(chain.run_dir / "diversity.json")
    split = report["split"]
    regions = {chain.source.truth[e].region for e in split["holdout"]}
    print(f"combined clusters {report['combined_clusters']}; holdout {split['holdout']}; "
          f"distance to train {split['holdout_distance_to_train']}")  # fmt: skip
    assert regions == {0, 1}
    assert split["clusters_missing_from_train"] == []
    assert sorted(split["train"] + split["holdout"]) == sorted(chain.source.truth)
    assert (chain.run_dir / "diversity.png").exists() and (chain.run_dir / "montage.png").exists()


def test_sweep_frame_baseline_is_exact_and_largest_admissible_span_is_selected(chain: ChainRun) -> None:
    sweep = _json(chain.run_dir / "sweep.json")
    cells = {(c["config"]["metric"], c["config"]["span_frames"]): c for c in sweep["cells"]}
    baseline = cells[("time", HORIZON)]
    print({k: (round(c["head_err"][0], 6), round(c["full_err"][0], 6)) for k, c in cells.items()})
    assert baseline["head_err"][0] < 1e-9 and baseline["full_err"][0] < 1e-9
    assert sorted(int(s) for s in sweep["shortlist"]) == [HORIZON, SPAN]
    assert sweep["selected"]["span_frames"] == SPAN


def test_label_replay_reports_each_replan_interval(chain: ChainRun) -> None:
    replay = _json(chain.run_dir / f"replay_time-s{SPAN}.json")
    print([(r["execute"], r["err"], r["edges_missed"]) for r in replay["rows"]])
    assert [r["execute"] for r in replay["rows"]] == [4, 8]


def test_train_and_eval_label_datasets(chain: ChainRun) -> None:
    train = _json(chain.labels_train / "meta" / "arc_resample.json")
    evaluation = _json(chain.labels_eval / "meta" / "arc_resample.json")
    kept = [e for e in sorted(chain.source.truth) if e not in chain.holdout]
    assert train["excluded_source_episodes"] == sorted(chain.holdout)
    assert train["episode_map"] == {str(e): new for new, e in enumerate(kept)}
    assert evaluation["episode_map"] == {str(e): e for e in chain.source.truth}

    timed = {e.index: e for e in load_episodes(chain.timed, SPEC)}
    columns = ["action_chunk", "action_loss_weight", "episode_index"]
    source, new = kept[0], 0
    train_rows = pd.read_parquet(
        chain.labels_train / "data" / "chunk-000" / f"episode_{new:06d}.parquet", columns=columns
    )
    eval_rows = pd.read_parquet(
        chain.labels_eval / "data" / "chunk-000" / f"episode_{source:06d}.parquet", columns=columns
    )
    chunks = np.stack(train_rows["action_chunk"].to_numpy())
    weights = np.stack(train_rows["action_loss_weight"].to_numpy())
    assert chunks.shape == (len(timed[source].actions), HORIZON * SPEC.action_dim)
    assert weights.shape == (len(timed[source].actions), HORIZON)
    np.testing.assert_array_equal(chunks, np.stack(eval_rows["action_chunk"].to_numpy()))
    assert set(np.unique(weights)) <= {1.0, 2.0}

    interp = ArcInterpolator.from_report(chain.labels_train / "meta" / "arc_resample.json")
    actions, states = timed[source].actions, timed[source].states
    worst = 0.0
    for t in range(0, len(actions) - SPAN):
        targets = interp(chunks[t].reshape(HORIZON, SPEC.action_dim), states[t])
        at_points = (interp.frames() - 1).astype(int)  # out[j] is the target for action index t + j
        worst = max(worst, float(np.abs(targets[at_points][:, JOINTS] - actions[t + at_points][:, JOINTS]).max()))
    print(f"targets at the point frames vs recording: max {worst:.2e}")
    assert worst < 1e-6
