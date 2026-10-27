"""prompt_annotation on the synthetic dataset: task tables, task_index, coverage check, source-map renumbering."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from synthetic_dataset import SyntheticDataset
from tool_runner import run_tool
from vla_tools.episodes import episode_paths


def _meta(root: Path, name: str) -> list[dict]:
    return [json.loads(line) for line in (root / "meta" / name).read_text().splitlines() if line.strip()]


def test_instructions_land_in_the_task_tables_and_rows(plain_dataset: SyntheticDataset, tmp_path: Path) -> None:
    episodes = [index for index, _ in episode_paths(plain_dataset.root)]
    prompts = {str(e): {"scene": f"scene {e}", "prompts": [f"first {e % 2}", f"second {e}"]} for e in episodes}
    (tmp_path / "prompts.json").write_text(json.dumps(prompts))
    out = tmp_path / "annotated"
    run_tool("prompt_annotation", "--dataset", plain_dataset.root, "--prompts", tmp_path / "prompts.json", "--out", out)

    tasks = {row["task"]: row["task_index"] for row in _meta(out, "tasks.jsonl")}
    print(tasks)
    assert list(tasks)[:3] == ["first 0", "second 0", "first 1"]
    assert len(tasks) == 2 + len(episodes)
    for row in _meta(out, "episodes.jsonl"):
        e = row["episode_index"]
        assert row["tasks"] == [f"first {e % 2}", f"second {e}"]
        rows = pd.read_parquet(dict(episode_paths(out))[e], columns=["task_index"])
        assert set(rows["task_index"]) == {tasks[f"first {e % 2}"]}
    assert json.loads((out / "meta" / "info.json").read_text())["total_tasks"] == len(tasks)
    assert json.loads((out / "meta" / "prompt_annotation.json").read_text())["scenes"]["0"] == "scene 0"


def test_missing_episode_is_refused_and_source_map_renumbers(plain_dataset: SyntheticDataset, tmp_path: Path) -> None:
    episodes = [index for index, _ in episode_paths(plain_dataset.root)]
    partial = {str(e): {"scene": "s", "prompts": ["p"]} for e in episodes[1:]}
    (tmp_path / "partial.json").write_text(json.dumps(partial))
    output = run_tool("prompt_annotation", "--dataset", plain_dataset.root, "--prompts", tmp_path / "partial.json",
                      "--out", tmp_path / "refused", expect_code=1)  # fmt: skip
    assert "episodes without instructions" in output and not (tmp_path / "refused").exists()

    # keyed by source numbers 100, 101, ... plus one excluded source episode (99)
    by_source = {str(100 + e): {"scene": "s", "prompts": [f"task {e}"]} for e in episodes} | {
        "99": {"scene": "x", "prompts": ["gone"]}
    }
    (tmp_path / "by_source.json").write_text(json.dumps(by_source))
    (tmp_path / "report.json").write_text(json.dumps({"episode_map": {str(100 + e): e for e in episodes}}))
    out = tmp_path / "mapped"
    run_tool("prompt_annotation", "--dataset", plain_dataset.root, "--prompts", tmp_path / "by_source.json",
             "--source-map", tmp_path / "report.json", "--out", out)  # fmt: skip
    assert [row["tasks"] for row in _meta(out, "episodes.jsonl")] == [[f"task {e}"] for e in episodes]
    assert "gone" not in {row["task"] for row in _meta(out, "tasks.jsonl")}
