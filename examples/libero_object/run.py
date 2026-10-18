"""Norm stats and training for the LIBERO joint-space example, through vla_openpi.launch.

Run from the openpi root, in an openpi environment where the harness is importable, with openpi's scripts
and this directory on PYTHONPATH:

    export PYTHONPATH=<openpi>/scripts:<VLA_AGENT>/src:<VLA_AGENT>/examples/libero_object
    uv run python <VLA_AGENT>/examples/libero_object/run.py norm-stats --name <config> --data-root <data> --repo-id <dataset>-train
    uv run python <VLA_AGENT>/examples/libero_object/run.py train --name <config> --data-root <data> --repo-id <dataset>-train --exp-name <run>
"""

from __future__ import annotations

import argparse
from pathlib import Path

from configs import train_config

from vla_openpi import launch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["norm-stats", "train"])
    parser.add_argument("--name", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--exp-name", default="run")
    args = parser.parse_args()

    config = train_config(args.name, args.data_root, args.repo_id, args.exp_name)
    if args.command == "norm-stats":
        print(f"wrote {launch.norm_stats(config)}")
    else:
        launch.train(config)


if __name__ == "__main__":
    main()
