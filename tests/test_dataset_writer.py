"""dataset_writer round trip: row edits, repeated rows, exclusion with renumbering, added and replaced columns."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from synthetic_dataset import CAMERA
from synthetic_dataset import SPEC
from synthetic_dataset import SyntheticDataset
from vla_tools import dataset_writer
from vla_tools.episodes import FrameReader
from vla_tools.episodes import episode_paths

ROWS = [0, 2, 2, 5]


def _table(root: Path, episode: int) -> pd.DataFrame:
    return pd.read_parquet(dict(episode_paths(root))[episode])


def _write(plain_dataset: SyntheticDataset, out: Path) -> dict[int, int]:
    replaced = np.arange(len(ROWS) * SPEC.action_dim, dtype=np.float64).reshape(len(ROWS), SPEC.action_dim)
    plans = [
        dataset_writer.EpisodePlan(
            source=1,
            rows=ROWS,
            replace={"action": replaced},
            add={"flag": dataset_writer.NewColumn([[0.0], [0.0], [1.0], [1.0]], {"dtype": "float32", "shape": [1]})},
        ),
        dataset_writer.EpisodePlan(
            source=3,
            add={
                "flag": dataset_writer.NewColumn(
                    [[0.0]] * plain_dataset.truth[3].length, {"dtype": "float32", "shape": [1]}
                )
            },
        ),
    ]
    return dataset_writer.write(plain_dataset.root, out, plans, {"note.json": {"purpose": "test"}})


def test_round_trip_renumbers_and_keeps_rows_consistent(plain_dataset: SyntheticDataset, tmp_path: Path) -> None:
    out = tmp_path / "derived"
    assert _write(plain_dataset, out) == {1: 0, 3: 1}

    source, first = _table(plain_dataset.root, 1), _table(out, 0)
    assert [cell["bytes"] for cell in first[CAMERA]] == [source[CAMERA].iloc[r]["bytes"] for r in ROWS]
    np.testing.assert_array_equal(np.stack(first["observation.state"].to_numpy()),
                                  np.stack(source["observation.state"].to_numpy())[ROWS])  # fmt: skip
    np.testing.assert_array_equal(np.stack(first["action"].to_numpy()).ravel(), np.arange(len(ROWS) * SPEC.action_dim))
    assert first["episode_index"].tolist() == [0] * len(ROWS)
    assert first["frame_index"].tolist() == list(range(len(ROWS)))
    assert first["index"].tolist() == list(range(len(ROWS)))
    np.testing.assert_allclose(first["timestamp"].to_numpy(), np.arange(len(ROWS)) / SPEC.fps, atol=1e-6)
    assert [v.tolist() for v in first["flag"]] == [[0.0], [0.0], [1.0], [1.0]]

    second = _table(out, 1)
    length = plain_dataset.truth[3].length
    assert second["index"].tolist() == list(range(len(ROWS), len(ROWS) + length))
    assert second["episode_index"].tolist() == [1] * length

    info = json.loads((out / "meta" / "info.json").read_text(encoding="utf-8"))
    assert (info["total_episodes"], info["total_frames"]) == (2, len(ROWS) + length)
    assert "flag" in info["features"]
    episodes = [json.loads(line) for line in (out / "meta" / "episodes.jsonl").read_text().splitlines()]
    assert [(e["episode_index"], e["length"]) for e in episodes] == [(0, len(ROWS)), (1, length)]
    note = json.loads((out / "meta" / "note.json").read_text(encoding="utf-8"))
    assert note["episode_map"] == {"1": 0, "3": 1}

    reader = FrameReader(out)
    np.testing.assert_array_equal(reader.frame(0, CAMERA, 2), FrameReader(plain_dataset.root).frame(1, CAMERA, 2))


def test_existing_output_is_refused(plain_dataset: SyntheticDataset, tmp_path: Path) -> None:
    (tmp_path / "derived").mkdir()
    with pytest.raises(FileExistsError):
        dataset_writer.write(plain_dataset.root, tmp_path / "derived", [dataset_writer.EpisodePlan(source=0)], {})


def test_row_edits_on_a_video_dataset_are_refused(plain_dataset: SyntheticDataset, tmp_path: Path) -> None:
    video = tmp_path / "video"
    shutil.copytree(plain_dataset.root, video)
    info = json.loads((video / "meta" / "info.json").read_text(encoding="utf-8"))
    info["video_path"] = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
    (video / "meta" / "info.json").write_text(json.dumps(info), encoding="utf-8")
    with pytest.raises(ValueError, match="re-encoding"):
        dataset_writer.write(video, tmp_path / "out", [dataset_writer.EpisodePlan(source=0, rows=[0, 1])], {})
    assert not (tmp_path / "out").exists()
