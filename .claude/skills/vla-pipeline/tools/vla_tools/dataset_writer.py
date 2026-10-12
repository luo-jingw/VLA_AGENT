"""Write a derived LeRobot v2.1 dataset. The single place that writes data; the source is never modified.

A derived dataset is described episode by episode: which source episode, which source rows make up
the new episode (dropped, kept, or repeated -- a repeated last row is how end padding is written), which
columns to replace and which to add. Kept episodes are renumbered 0..N-1, and `frame_index`,
`timestamp` and `index` are rewritten so the copy is contiguous. Parquet goes through `datasets` so the
`huggingface` schema block that declares image columns survives. Videos are copied with their episode
renumbered; changing the rows of a video dataset would need re-encoding and is refused.

`meta/episodes_stats.jsonl` is copied for kept episodes but not recomputed: openpi computes its own
normalization statistics and does not read it.
"""

from __future__ import annotations

import dataclasses
import json
import shutil
import uuid
from pathlib import Path

import datasets
import numpy as np

from vla_tools.episodes import episode_paths
from vla_tools.episodes import info as read_info


@dataclasses.dataclass(frozen=True)
class NewColumn:
    values: list[list[float]]  # one row per kept frame
    feature: dict  # the info.json feature entry, e.g. {"dtype": "float32", "shape": [140], "names": [...]}


@dataclasses.dataclass(frozen=True)
class EpisodePlan:
    source: int  # source episode_index
    rows: list[int] | None = None  # source row of every new row, in order; None keeps the episode as is
    replace: dict[str, np.ndarray] = dataclasses.field(default_factory=dict)  # column -> (kept rows, dim)
    add: dict[str, NewColumn] = dataclasses.field(default_factory=dict)


def write(src: Path, out: Path, plans: list[EpisodePlan], extra_meta: dict[str, dict]) -> dict[int, int]:
    """Write the derived dataset. Returns source episode -> new episode."""
    if out.exists():
        raise FileExistsError(f"{out} exists; remove it or choose another output")
    src_info = read_info(src)
    paths = dict(episode_paths(src))
    chunks_size = int(src_info.get("chunks_size", 1000))
    has_video = bool(src_info.get("video_path"))
    if has_video and any(p.rows is not None for p in plans):
        raise ValueError("changing the rows of a video dataset needs re-encoding; edit rows before conversion instead")

    # Transforms below are keyed by fingerprint; with caching on, a fingerprint reused from an earlier
    # run silently returns that run's table. Nothing here is worth caching.
    datasets.disable_caching()
    (out / "meta").mkdir(parents=True)
    fps = float(src_info["fps"])
    remap: dict[int, int] = {}
    lengths: dict[int, int] = {}
    running = 0
    added_features: dict[str, dict] = {}
    for new, plan in enumerate(plans):
        remap[plan.source] = new
        table = datasets.Dataset.from_parquet(str(paths[plan.source]))
        if plan.rows is not None:
            # select() only adds an index view; materialise it so later columns line up with the rows.
            table = table.select(plan.rows).flatten_indices()
        n = len(table)
        for column, values in plan.replace.items():
            feature = table.features[column]
            table = table.remove_columns([column]).add_column(
                column, [row.astype(np.float32).tolist() for row in values], new_fingerprint=uuid.uuid4().hex
            )
            table = table.cast_column(column, feature)
        for column, extra in plan.add.items():
            table = table.add_column(column, extra.values, new_fingerprint=uuid.uuid4().hex)
            added_features[column] = extra.feature
        rewrites = {
            "episode_index": [new] * n,
            "index": list(range(running, running + n)),
            "frame_index": list(range(n)),
            "timestamp": [i / fps for i in range(n)],
        }
        for column, values in rewrites.items():
            if column in table.column_names:
                feature = table.features[column]
                table = table.remove_columns([column]).add_column(column, values, new_fingerprint=uuid.uuid4().hex)
                table = table.cast_column(column, feature)
        target = out / "data" / f"chunk-{new // chunks_size:03d}" / f"episode_{new:06d}.parquet"
        target.parent.mkdir(parents=True, exist_ok=True)
        table.to_parquet(str(target))
        if has_video:
            _copy_videos(src, out, src_info, plan.source, new, chunks_size)
        lengths[new] = n
        running += n

    _write_meta(src, out, src_info, remap, lengths, added_features, chunks_size)
    for name, payload in extra_meta.items():
        payload = payload | {"source_dataset": str(src), "episode_map": {str(k): v for k, v in remap.items()}}
        (out / "meta" / name).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return remap


def _copy_videos(src: Path, out: Path, src_info: dict, old: int, new: int, chunks_size: int) -> None:
    keys = [k for k, f in src_info["features"].items() if f.get("dtype") == "video"]
    for key in keys:
        fmt = src_info["video_path"]
        source = src / fmt.format(episode_chunk=old // chunks_size, video_key=key, episode_index=old)
        target = out / fmt.format(episode_chunk=new // chunks_size, video_key=key, episode_index=new)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)


def _write_meta(
    src: Path,
    out: Path,
    src_info: dict,
    remap: dict[int, int],
    lengths: dict[int, int],
    added_features: dict[str, dict],
    chunks_size: int,
) -> None:
    for source in (src / "meta").iterdir():
        if not source.is_file() or source.name == "info.json":
            continue
        if source.name in ("episodes.jsonl", "episodes_stats.jsonl"):
            rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
            kept = []
            for row in rows:
                if row["episode_index"] not in remap:
                    continue
                new = remap[row["episode_index"]]
                row = row | {"episode_index": new}
                if "length" in row:
                    row["length"] = lengths[new]
                kept.append(row)
            kept.sort(key=lambda row: row["episode_index"])
            (out / "meta" / source.name).write_text("".join(json.dumps(r) + "\n" for r in kept), encoding="utf-8")
        else:
            shutil.copyfile(source, out / "meta" / source.name)

    new_info = dict(src_info)
    new_info["features"] = dict(src_info["features"]) | added_features
    episodes = len(remap)
    video_keys = [k for k, f in src_info["features"].items() if f.get("dtype") == "video"]
    new_info.update(
        total_episodes=episodes,
        total_frames=int(sum(lengths.values())),
        total_videos=episodes * len(video_keys),
        total_chunks=(episodes + chunks_size - 1) // chunks_size,
        splits={"train": f"0:{episodes}"},
    )
    (out / "meta" / "info.json").write_text(json.dumps(new_info, indent=2), encoding="utf-8")
