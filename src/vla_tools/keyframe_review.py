"""Contact sheets for checking keyframes by eye. Writes PNG pages; reads nothing but the dataset and keyframes.json.

One page per keyframe type and group of episodes. Each row is one episode: camera frames at the
chosen offsets from the keyframe (default: window start, the keyframe, window end) and a strip with the
command dims and joint speed, the keyframe line and its window shaded. What to look for depends on the
keyframe's meaning -- e.g. at a grasp, the fingers are around the object in the "0" frame and closed
on it at the window end; at a seal, the part has left the bin by the window end.

Usage:
    python -m vla_tools.keyframe_review --dataset <dir> --spec robot_spec.json --keyframes keyframes.json \
        --type grasp --camera observation.images.cam_left_wrist [--episodes 3 9] [--issues-only] --out review/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from vla_tools.episodes import FrameReader
from vla_tools.episodes import load_episodes
from vla_tools.keyframe_detectors import signal_values
from vla_tools.keyframe_spec import Signal
from vla_tools.robot_spec import RobotSpec


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--keyframes", type=Path, required=True)
    parser.add_argument("--type", required=True)
    parser.add_argument("--camera", required=True)
    parser.add_argument("--offsets", nargs="*", type=int, default=None, help="frames relative to the keyframe")
    parser.add_argument("--episodes", nargs="*", type=int, default=None)
    parser.add_argument("--issues-only", action="store_true")
    parser.add_argument("--per-page", type=int, default=8)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    spec = RobotSpec.load(args.spec)
    record = json.loads(args.keyframes.read_text(encoding="utf-8"))
    # An episode is flagged for this type only by open issues that name the type.
    flagged = {i["episode"] for i in record["issues"] if not i["accepted"] and args.type in i["detail"]}
    wanted = [int(k) for k in record["episodes"]]
    if args.episodes:
        wanted = [e for e in wanted if e in args.episodes]
    if args.issues_only:
        wanted = [e for e in wanted if e in flagged]
    episodes = {e.index: e for e in load_episodes(args.dataset, spec, only=wanted)}
    reader = FrameReader(args.dataset)

    rows = []
    for ep in wanted:
        for k in record["episodes"][str(ep)]["keyframes"]:
            if k["type"] == args.type:
                rows.append((ep, k))
    if not rows:
        print(f"no {args.type} keyframes in the selected episodes")
        return
    args.out.mkdir(parents=True, exist_ok=True)
    pages = [rows[i : i + args.per_page] for i in range(0, len(rows), args.per_page)]
    for page_index, page in enumerate(pages):
        _, sample = page[0]
        offsets = args.offsets or [sample["window"][0] - sample["frame"], 0, sample["window"][1] - sample["frame"]]
        fig, axes = plt.subplots(
            len(page), len(offsets) + 1, figsize=(2.6 * len(offsets) + 4.5, 2.0 * len(page)), squeeze=False,
            gridspec_kw={"width_ratios": [1] * len(offsets) + [1.8]},
        )  # fmt: skip
        for row, (ep, k) in enumerate(page):
            e = episodes[ep]
            n = len(e.actions)
            for col, offset in enumerate(offsets):
                frame = int(np.clip(k["frame"] + offset, 0, n - 1))
                axes[row, col].imshow(reader.frame(ep, args.camera, frame))
                axes[row, col].set_title(f"ep {ep} f{frame} ({offset:+d})", fontsize=7)
                axes[row, col].axis("off")
            ax = axes[row, -1]
            lo, hi = max(k["window"][0] - 15, 0), min(k["window"][1] + 15, n)
            speed = signal_values(Signal("joint_speed"), e.actions, e.states, spec, {})
            ax.plot(range(lo, hi), speed[lo:hi] / max(speed.max(), 1e-9), color="0.4", lw=0.8, label="joint speed")
            for dim in spec.command_dims:
                ax.step(range(lo, hi), e.actions[lo:hi, dim], where="post", lw=1.0, label=f"cmd {dim}")
            ax.axvspan(k["window"][0], k["window"][1], color="tab:orange", alpha=0.15)
            ax.axvline(k["frame"], color="tab:red", lw=1)
            ax.set_yticks([])
            ax.tick_params(labelsize=6)
            mark = "  [issue]" if ep in flagged else ""
            ax.set_title(f"{args.type}@{k['frame']} {k['source'][:40]}{mark}", fontsize=7)
        axes[0, -1].legend(fontsize=5, loc="upper right")
        fig.tight_layout()
        path = args.out / f"{args.type}_page{page_index:02d}.png"
        fig.savefig(path, dpi=65)
        plt.close(fig)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
