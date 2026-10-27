"""episode_sheets on the synthetic dataset: the moments shown and the sheet layout."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from synthetic_dataset import IMAGE_SHAPE
from synthetic_dataset import SPEC
from synthetic_dataset import SyntheticDataset
from synthetic_dataset import write_spec
from tool_runner import run_tool
from vla_tools.episode_sheets import moments
from vla_tools.episodes import load_episodes


def test_moments_are_start_each_edge_and_end(plain_dataset: SyntheticDataset) -> None:
    episode = load_episodes(plain_dataset.root, SPEC, only=[0])[0]
    marks = moments(episode, SPEC.command_dims, after=2)
    print(marks)
    truth = plain_dataset.truth[0]
    assert [m.label.split()[1] if m.label.startswith("d") else m.label.split()[0] for m in marks] == [
        "start", "close", "open", "end"
    ]  # fmt: skip
    assert marks[1].frame == truth.grasp + 1 + 2 and marks[2].frame == min(truth.release + 3, episode.real - 1)
    assert marks[-1].frame == episode.real - 1
    regular = moments(episode, SPEC.command_dims, after=2, every=10)
    frames = [m.frame for m in regular]
    assert (
        frames == sorted(frames)
        and {10, 20}.issubset(frames)
        and len(regular) == len(marks) + len(range(10, episode.real - 1, 10))
    )


def test_one_sheet_per_episode_with_a_column_per_moment(plain_dataset: SyntheticDataset, tmp_path: Path) -> None:
    out = tmp_path / "sheets"
    run_tool("episode_sheets", "--dataset", plain_dataset.root, "--spec", write_spec(tmp_path / "spec.json"),
             "--width", 64, "--episodes", 0, 1, "--out", out)  # fmt: skip
    files = sorted(out.glob("*.png"))
    assert [f.name for f in files] == ["episode_000000.png", "episode_000001.png"]
    with Image.open(files[0]) as img:
        print(img.size)
        assert img.size == (4 * 64, 16 + round(64 * IMAGE_SHAPE[0] / IMAGE_SHAPE[1]))
