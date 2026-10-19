"""Norm stats, training and serving for the dual-arm UR3 example with the VLM frozen, through vla_openpi.launch.

Run from the openpi root, in an openpi environment where the harness is importable, with openpi's scripts
and this directory on PYTHONPATH:

    export PYTHONPATH=<openpi>/scripts:<VLA_AGENT>/src:<VLA_AGENT>/examples/ur3_optical_bench
    uv run python <VLA_AGENT>/examples/ur3_optical_bench/run.py norm-stats --name <config> --data-root <data> --repo-id <dataset>-train
    uv run python <VLA_AGENT>/examples/ur3_optical_bench/run.py train --name <config> --data-root <data> --repo-id <dataset>-train --exp-name <run>
    uv run python <VLA_AGENT>/examples/ur3_optical_bench/run.py serve --name <config> --data-root <data> --repo-id <dataset>-train \
        --checkpoint checkpoints/<config>/<run>/<step> --port 8000 [--rate 1]

`serve` answers with per-frame targets interpolated from the checkpoint's report (and the points);
`serve-points` answers with the (H, D) points only. Both need the label dataset's meta/arc_resample.json in
the checkpoint's assets/<repo_id>/.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from configs import train_config

from vla_openpi import launch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["norm-stats", "train", "serve", "serve-points"])
    parser.add_argument("--name", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--exp-name", default="run")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--rate", type=float, default=1.0, help="execution rate relative to the recording's pace")
    args = parser.parse_args()

    config = train_config(args.name, args.data_root, args.repo_id, args.exp_name)
    if args.command == "norm-stats":
        print(f"wrote {launch.norm_stats(config)}")
    elif args.command == "train":
        launch.train(config)
    else:
        if args.checkpoint is None:
            raise SystemExit(f"{args.command} needs --checkpoint")
        if args.command == "serve":
            launch.serve(config, args.checkpoint, args.port, args.rate)
        else:
            launch.serve_points(config, args.checkpoint, args.port)


if __name__ == "__main__":
    main()
