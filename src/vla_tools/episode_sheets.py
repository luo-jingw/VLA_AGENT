"""One contact sheet per episode: what each episode shows, for writing its instructions and for spotting episodes
that must not be trained on (a person in the scene, a failed or undone task, a wrong object).

A column per moment -- the first frame, --after frames after every command edge (a gripper closing or
opening), the last recorded frame, and with --every N also every N-th frame -- and a row per camera. The
regular frames show what happens between the edges: a person reaching in during an automatic return home
leaves no trace at the edges or at the end. Reads the dataset only; works on any version.

Usage:
    python -m vla_tools.episode_sheets --dataset <dir> --spec robot_spec.json \
        --cameras observation.images.cam_high observation.images.cam_right_wrist [--after 5] [--every 40] \
        [--episodes 3 9] --out sheets/
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

from vla_tools.episodes import Episode
from vla_tools.episodes import FrameReader
from vla_tools.episodes import load_episodes
from vla_tools.robot_spec import RobotSpec

LABEL = 16


@dataclasses.dataclass(frozen=True)
class Moment:
    frame: int  # row shown
    label: str


def moments(episode: Episode, dims: tuple[int, ...], after: int, every: int = 0) -> list[Moment]:
    """First frame, `after` frames past each edge of the command dims, every `every`-th frame (0: none), last
    recorded frame, in frame order. An edge carried by actions[f] is reached at row f + 1."""
    last = episode.real - 1
    out = [Moment(0, "start")]
    edges = []
    for d in dims:
        values = episode.actions[: episode.real, d]
        for f in np.flatnonzero(np.abs(np.diff(values)) > 0.5) + 1:
            kind = "close" if values[f] > values[f - 1] else "open"
            edges.append(Moment(min(int(f) + 1 + after, last), f"d{d} {kind} @{int(f)}"))
    if every > 0:
        edges += [Moment(t, f"@{t}") for t in range(every, last, every)]
    out += sorted(edges, key=lambda m: m.frame)
    out.append(Moment(last, f"end @{last}"))
    return out


def sheet(reader: FrameReader, episode: Episode, cameras: list[str], marks: list[Moment], width: int) -> Image.Image:
    rows = []
    for camera in cameras:
        first = reader.frame(episode.index, camera, 0)
        height = round(width * first.shape[0] / first.shape[1])
        rows.append((camera, height))
    img = Image.new("RGB", (width * len(marks), LABEL + sum(h for _, h in rows)), (20, 20, 20))
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    for k, mark in enumerate(marks):
        draw.text((k * width + 4, 2), mark.label, font=font, fill=(255, 255, 255))
        y = LABEL
        for camera, height in rows:
            frame = Image.fromarray(reader.frame(episode.index, camera, mark.frame)).resize((width, height))
            img.paste(frame, (k * width, y))
            y += height
    return img


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--cameras", nargs="+", default=None, help="image columns, one row each; default the spec's")
    parser.add_argument("--after", type=int, default=5, help="frames past a command edge to show")
    parser.add_argument("--every", type=int, default=0, help="also every N-th frame; 0 shows none")
    parser.add_argument("--width", type=int, default=320, help="tile width in pixels")
    parser.add_argument("--episodes", nargs="*", type=int, default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    spec = RobotSpec.load(args.spec)
    cameras = args.cameras or list(spec.image_keys)
    if not cameras:
        raise SystemExit("no camera: pass --cameras or list image_keys in the spec")
    args.out.mkdir(parents=True, exist_ok=True)
    count = 0
    for episode in load_episodes(args.dataset, spec, only=args.episodes):
        marks = moments(episode, tuple(spec.command_dims), args.after, args.every)
        sheet(FrameReader(args.dataset), episode, cameras, marks, args.width).save(
            args.out / f"episode_{episode.index:06d}.png"
        )
        count += 1
    print(f"wrote {count} sheets to {args.out}")


if __name__ == "__main__":
    main()
