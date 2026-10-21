"""Norm stats, training and serving for the dual-arm UR3 example, through vla_openpi.launch.

Run from the openpi root, in an openpi environment where the harness is installed (`pip install -e <VLA_AGENT>`),
with openpi's scripts and this directory on PYTHONPATH:

    export PYTHONPATH=<openpi>/scripts:<VLA_AGENT>/examples/ur3_dual_arm
    uv run python <VLA_AGENT>/examples/ur3_dual_arm/run.py norm-stats --data-root <data> --repo-id <dataset>-train
    uv run python <VLA_AGENT>/examples/ur3_dual_arm/run.py train --data-root <data> --repo-id <dataset>-train --exp-name <run>
    uv run python <VLA_AGENT>/examples/ur3_dual_arm/run.py serve --data-root <data> --repo-id <dataset>-train \
        --checkpoint checkpoints/<name>/<run>/<step> --port 8000

`serve` needs the label dataset's meta/arc_resample.json copied into the checkpoint's assets/<repo_id>/.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from configs import train_config

from vla_openpi import launch
from vla_tools.arc_interp import DEPLOY_CAP
from vla_tools.arc_interp import DEPLOY_FRACTION


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["norm-stats", "train", "serve"])
    parser.add_argument("--name", default="pi05_ur3_dual_arm")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--exp-name", default="run")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--cap",
        type=float,
        default=DEPLOY_CAP,
        help="serve: velocity timing's cap, or the rate with --uniform, relative to the recording's pace",
    )
    parser.add_argument(
        "--velocity-fraction",
        type=float,
        default=DEPLOY_FRACTION,
        help="serve: fraction of each joint velocity limit (velocity timing, the default)",
    )
    parser.add_argument(
        "--uniform", action="store_true", help="serve: uniform timing at --cap instead of velocity timing"
    )
    args = parser.parse_args()

    config = train_config(args.name, args.data_root, args.repo_id, args.exp_name)
    if args.command == "norm-stats":
        print(f"wrote {launch.norm_stats(config)}")
    elif args.command == "train":
        launch.train(config)
    else:
        if args.checkpoint is None:
            raise SystemExit("serve needs --checkpoint")
        launch.serve(config, args.checkpoint, args.port, args.cap, None if args.uniform else args.velocity_fraction)


if __name__ == "__main__":
    main()
