"""Norm stats, training and serving for the dual-arm UR3 example, through vla_openpi.launch.

Run from the openpi root, in an openpi environment where the harness is installed (`pip install -e <VLA_AGENT>`),
with openpi's scripts and this directory on PYTHONPATH:

    export PYTHONPATH=<openpi>/scripts:<VLA_AGENT>/examples/ur3_dual_arm
    uv run python <VLA_AGENT>/examples/ur3_dual_arm/run.py norm-stats --data-root <data> --repo-id <dataset>-train \
        [--per-point-actions]
    uv run python <VLA_AGENT>/examples/ur3_dual_arm/run.py train --data-root <data> --repo-id <dataset>-train --exp-name <run> \
        [--resume] [--recipe <recipe>] [--init-checkpoint checkpoints/<name>/<run>/<step> [--lora]]
    uv run python <VLA_AGENT>/examples/ur3_dual_arm/run.py serve --data-root <data> --repo-id <dataset>-train \
        --checkpoint checkpoints/<name>/<run>/<step> --port 8000

`serve` needs the label dataset's meta/arc_resample.json copied into the checkpoint's assets/<repo_id>/.
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

from configs import train_config
from recipes import RECIPES

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
    parser.add_argument("--resume", action="store_true", help="train: continue from the run's latest checkpoint")
    parser.add_argument("--absolute-joints", action="store_true", help="joints as absolute positions, not deltas")
    parser.add_argument("--state-input", action="store_true", help="the state as discrete tokens in the prompt")
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        default=None,
        help="train: start from the params of this saved checkpoint step directory instead of pi05_base",
    )
    parser.add_argument(
        "--lora",
        action="store_true",
        help="train: LoRA adapters of the action expert only, on top of --init-checkpoint",
    )
    parser.add_argument(
        "--recipe",
        choices=sorted(RECIPES),
        default=None,
        help="train: steps, learning rate, warm-up and EMA decay (recipes.py); default full_20k, or lora_10k with --lora",
    )
    parser.add_argument(
        "--per-point-actions",
        action="store_true",
        help="norm-stats: action statistics per point of the chunk instead of pooled over its points",
    )
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

    config = train_config(
        args.name,
        args.data_root,
        args.repo_id,
        args.exp_name,
        args.absolute_joints,
        args.state_input,
        args.init_checkpoint,
        args.lora,
        recipe=RECIPES[args.recipe or ("lora_10k" if args.lora else "full_20k")],
    )
    if args.command == "norm-stats":
        print(f"wrote {launch.norm_stats(config, per_point_actions=args.per_point_actions)}")
    elif args.command == "train":
        launch.train(dataclasses.replace(config, resume=args.resume))
    else:
        if args.checkpoint is None:
            raise SystemExit("serve needs --checkpoint")
        launch.serve(config, args.checkpoint, args.port, args.cap, None if args.uniform else args.velocity_fraction)


if __name__ == "__main__":
    main()
