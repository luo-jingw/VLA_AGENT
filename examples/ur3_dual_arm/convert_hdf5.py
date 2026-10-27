"""Convert raw dual-arm UR3 recordings (one HDF5 file per episode, the `episode_writer.py` layout of
`embodyx/UR3-Whole_Process`) into a LeRobot v2.1 dataset with parquet-embedded images: the intake step of
vla-data-intake for this recorder.

Row t of an episode:
    observation.state (14) float32  left joints 0-5 (rad), left gripper 6, right joints 7-12, right gripper 13;
                                    a gripper is its command (0 open, 1 closed), as in examples/ur3_dual_arm
    action (14) float32             observation.state of row t+1; the last row repeats its own state
    observation.images.cam_high        <- observations/images/top          (240x320)
    observation.images.cam_left_wrist  <- observations/images/left_wrist   (240x424)
    observation.images.cam_right_wrist <- observations/images/right_wrist  (240x424)
                                    the recorded JPEG bytes, not re-encoded
    task_index                      0: one placeholder task (--task) for every episode until
                                    `vla_tools.prompt_annotation` writes each episode's instructions

Timestamps are rewritten contiguous (frame / 30), so a gap in the recording would no longer be visible after
conversion: an episode whose largest interval between stored steps (`timestamps/step`) exceeds 1.5 frames is
refused unless conversion.json excludes it.

Inputs:
    conversion.json  {"source": "<hub repo>@<revision>", "exclude": {"<episode>": "<reason>", ...}}

Writes meta/conversion_report.json: source, exclusions, source episode -> new episode (the `episode_map`
that `prompt_annotation --source-map` reads), and the largest step interval of every kept episode.

Usage (needs h5py: `pip install -e <VLA_AGENT>[hdf5]`):
    python examples/ur3_dual_arm/convert_hdf5.py --raw <raw_dir> --conversion runs/<task>/conversion.json \
        --task "<placeholder instruction>" --out data/<src>
"""

from __future__ import annotations

import argparse
import dataclasses
import io
import json
from pathlib import Path

import datasets
import h5py
import numpy as np
from PIL import Image

FPS = 30
CHUNKS_SIZE = 1000
ARMS = ("left", "right")
CAMERAS = {  # LeRobot key -> recorder camera
    "observation.images.cam_high": "top",
    "observation.images.cam_left_wrist": "left_wrist",
    "observation.images.cam_right_wrist": "right_wrist",
}
JOINTS = ("shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3", "gripper")
NAMES = [f"{arm}_{joint}" for arm in ARMS for joint in JOINTS]
IMAGE_SAMPLES = 20  # frames per episode and camera in meta/episodes_stats.jsonl
MAX_GAP_FRAMES = 1.5  # largest interval between stored steps of a kept episode, in frames


@dataclasses.dataclass(frozen=True)
class ConversionConfig:
    source: str
    exclude: dict[int, str]

    @staticmethod
    def load(path: Path) -> ConversionConfig:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return ConversionConfig(source=raw["source"], exclude={int(k): v for k, v in raw["exclude"].items()})


@dataclasses.dataclass(frozen=True)
class RecordedEpisode:
    source: int
    state: np.ndarray  # (T, 14) float32
    images: dict[str, list[bytes]]  # LeRobot key -> T JPEG byte strings
    max_step_s: float  # largest interval between consecutive stored steps

    @property
    def action(self) -> np.ndarray:
        return np.concatenate([self.state[1:], self.state[-1:]], axis=0)


def read_episode(path: Path, source: int) -> RecordedEpisode:
    with h5py.File(path, "r") as f:
        columns = []
        for arm in ARMS:
            columns.append(np.asarray(f[f"observations/{arm}/joint_pos"], dtype=np.float32))
            columns.append(np.asarray(f[f"command/{arm}/gripper"], dtype=np.float32)[:, None])
        state = np.concatenate(columns, axis=1)
        images = {}
        for key, camera in CAMERAS.items():
            data, lengths = f[f"observations/images/{camera}"][:], f[f"observations/images/{camera}_len"][:]
            images[key] = [data[t, : int(lengths[t])].tobytes() for t in range(len(lengths))]
        steps = np.asarray(f["timestamps/step"], dtype=np.float64)
    return RecordedEpisode(source=source, state=state, images=images, max_step_s=float(np.diff(steps).max()))


def image_shape(jpeg: bytes) -> list[int]:
    width, height = Image.open(io.BytesIO(jpeg)).size
    return [height, width, 3]


def features(shapes: dict[str, list[int]]) -> tuple[datasets.Features, dict]:
    """The parquet schema and the info.json feature block."""
    vector = {"dtype": "float32", "shape": [len(NAMES)], "names": [NAMES]}
    info = {key: {"dtype": "image", "shape": shapes[key], "names": ["height", "width", "channel"]} for key in CAMERAS}
    info |= {
        "observation.state": vector,
        "action": vector,
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
    }
    schema = {key: datasets.Image() for key in CAMERAS}
    schema |= {
        "observation.state": datasets.Sequence(datasets.Value("float32"), length=len(NAMES)),
        "action": datasets.Sequence(datasets.Value("float32"), length=len(NAMES)),
        "timestamp": datasets.Value("float32"),
        "frame_index": datasets.Value("int64"),
        "episode_index": datasets.Value("int64"),
        "index": datasets.Value("int64"),
        "task_index": datasets.Value("int64"),
    }
    return datasets.Features(schema), info


def numeric_stats(values: np.ndarray) -> dict[str, list[float]]:
    values = values.reshape(len(values), -1).astype(np.float64)
    return {
        "min": values.min(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
        "mean": values.mean(axis=0).tolist(),
        "std": values.std(axis=0).tolist(),
        "count": [len(values)],
    }


def image_stats(jpegs: list[bytes]) -> dict[str, list]:
    """Per-channel statistics of evenly spaced frames, in [0, 1], shaped (3, 1, 1) as lerobot expects."""
    picks = np.linspace(0, len(jpegs) - 1, min(IMAGE_SAMPLES, len(jpegs))).round().astype(int)
    pixels = np.stack([np.asarray(Image.open(io.BytesIO(jpegs[i])).convert("RGB")) for i in picks]) / 255.0
    flat = pixels.reshape(-1, 3)

    def shaped(v: np.ndarray) -> list:
        return v.reshape(3, 1, 1).tolist()

    return {
        "min": shaped(flat.min(axis=0)),
        "max": shaped(flat.max(axis=0)),
        "mean": shaped(flat.mean(axis=0)),
        "std": shaped(flat.std(axis=0)),
        "count": [len(picks)],
    }


def episode_columns(episode: RecordedEpisode, new: int, start: int, task: int) -> dict[str, list]:
    n = len(episode.state)
    columns: dict[str, list] = {key: [{"bytes": b, "path": None} for b in episode.images[key]] for key in CAMERAS}
    columns |= {
        "observation.state": episode.state.tolist(),
        "action": episode.action.tolist(),
        "timestamp": [i / FPS for i in range(n)],
        "frame_index": list(range(n)),
        "episode_index": [new] * n,
        "index": list(range(start, start + n)),
        "task_index": [task] * n,
    }
    return columns


def episode_stats(episode: RecordedEpisode, columns: dict[str, list]) -> dict[str, dict]:
    stats = {key: image_stats(episode.images[key]) for key in CAMERAS}
    for key in ("observation.state", "action", "timestamp", "frame_index", "episode_index", "index", "task_index"):
        stats[key] = numeric_stats(np.asarray(columns[key]))
    return stats


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw", type=Path, required=True)
    p.add_argument("--conversion", type=Path, required=True)
    p.add_argument("--task", required=True, help="placeholder instruction until prompt_annotation")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise SystemExit(f"{a.out} exists; remove it or choose another output")
    config = ConversionConfig.load(a.conversion)
    sources = sorted(
        (int(path.stem.split("_")[1]), path)
        for path in a.raw.glob("episode_*.hdf5")
        if int(path.stem.split("_")[1]) not in config.exclude
    )

    datasets.disable_caching()
    (a.out / "meta").mkdir(parents=True)
    schema, info_features = None, None
    episodes_rows, stats_rows, episode_map, max_step = [], [], {}, {}
    running = 0
    for new, (source, path) in enumerate(sources):
        episode = read_episode(path, source)
        if episode.max_step_s > MAX_GAP_FRAMES / FPS:
            raise SystemExit(
                f"episode {source:04d}: largest step interval {episode.max_step_s:.2f} s exceeds {MAX_GAP_FRAMES} "
                f"frames; exclude it in {a.conversion} with a reason"
            )
        if schema is None:
            schema, info_features = features({key: image_shape(episode.images[key][0]) for key in CAMERAS})
        columns = episode_columns(episode, new, running, 0)
        target = a.out / "data" / f"chunk-{new // CHUNKS_SIZE:03d}" / f"episode_{new:06d}.parquet"
        target.parent.mkdir(parents=True, exist_ok=True)
        datasets.Dataset.from_dict(columns, features=schema).to_parquet(str(target))
        n = len(episode.state)
        episodes_rows.append({"episode_index": new, "tasks": [a.task], "length": n})
        stats_rows.append({"episode_index": new, "stats": episode_stats(episode, columns)})
        episode_map[str(source)] = new
        max_step[str(source)] = round(episode.max_step_s, 4)
        running += n
        print(
            f"episode {source:04d} -> {new:3d}: {n} rows, largest step {episode.max_step_s * 1000:.1f} ms", flush=True
        )

    total = len(sources)
    info = {
        "codebase_version": "v2.1",
        "robot_type": "ur3_bimanual",
        "total_episodes": total,
        "total_frames": running,
        "total_tasks": 1,
        "total_videos": 0,
        "total_chunks": (total + CHUNKS_SIZE - 1) // CHUNKS_SIZE,
        "chunks_size": CHUNKS_SIZE,
        "fps": FPS,
        "splits": {"train": f"0:{total}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": None,
        "features": info_features,
    }
    meta = a.out / "meta"
    (meta / "info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    (meta / "tasks.jsonl").write_text(json.dumps({"task_index": 0, "task": a.task}) + "\n", encoding="utf-8")
    (meta / "episodes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in episodes_rows), encoding="utf-8")
    (meta / "episodes_stats.jsonl").write_text("".join(json.dumps(r) + "\n" for r in stats_rows), encoding="utf-8")
    report = {
        "source": config.source,
        "raw": str(a.raw),
        "excluded": {str(k): v for k, v in sorted(config.exclude.items())},
        "episode_map": episode_map,
        "max_step_s": max_step,
    }
    (meta / "conversion_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {total} episodes, {running} rows to {a.out}")


if __name__ == "__main__":
    main()
