"""Norm stats, training and serving points for the LIBERO joint-space example, through vla_openpi.launch.

Run from the openpi root, in an openpi environment where the harness is importable, with openpi's scripts
and this directory on PYTHONPATH:

    export PYTHONPATH=<openpi>/scripts:<VLA_AGENT>/src:<VLA_AGENT>/examples/libero_object
    uv run python <VLA_AGENT>/examples/libero_object/run.py norm-stats --name <config> --data-root <data> --repo-id <dataset>-train
    uv run python <VLA_AGENT>/examples/libero_object/run.py train --name <config> --data-root <data> --repo-id <dataset>-train --exp-name <run>
    uv run python <VLA_AGENT>/examples/libero_object/run.py serve-points --name <config> --data-root <data> \
        --repo-id <dataset>-train --checkpoint checkpoints/<config>/<run>/<step> --port 8000

`serve-points` answers each observation with the model's (H, D) absolute points; the client interpolates them
(vla_libero.evaluate).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from configs import train_config

from vla_openpi import launch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["norm-stats", "train", "serve-points"])
    parser.add_argument("--name", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--exp-name", default="run")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--per-point-actions", action="store_true",
                        help="norm-stats: action statistics per point of the chunk instead of pooled over its points")
    args = parser.parse_args()

    config = train_config(args.name, args.data_root, args.repo_id, args.exp_name)
    if args.command == "norm-stats":
        print(f"wrote {launch.norm_stats(config, per_point_actions=args.per_point_actions)}")
    elif args.command == "train":
        launch.train(config)
    else:
        if args.checkpoint is None:
            raise SystemExit("serve-points needs --checkpoint")
        launch.serve_points(config, args.checkpoint, args.port)


if __name__ == "__main__":
    main()
