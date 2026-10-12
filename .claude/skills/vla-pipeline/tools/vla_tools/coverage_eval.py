"""Read model errors along data coverage: does a model get worse the farther an episode is from its training data?

Joins the split written by `diversity` (each held-out episode's distance to the nearest training episode)
with one or more `replay_eval` results, and reports per held-out episode and per model the stitched error
and command-edge shift, plus the Spearman correlation between distance and error. With few held-out
episodes the correlation is weak evidence; the per-episode table is the primary output.

Usage:
    python -m vla_tools.coverage_eval --diversity diversity.json --eval baseline=eval_a.json resampled=eval_b.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--diversity", type=Path, required=True)
    parser.add_argument("--eval", nargs="+", required=True, help="name=path pairs")
    args = parser.parse_args()

    split = json.loads(args.diversity.read_text(encoding="utf-8"))["split"]
    if split is None:
        raise SystemExit("the diversity file has no split; rerun diversity with --holdout")
    distance = {int(k): v / split["median_nearest_neighbour"] for k, v in split["holdout_distance_to_train"].items()}
    models = {}
    for item in args.eval:
        name, path = item.split("=", 1)
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        step = data["frame_of_motion_rad"]
        models[name] = {
            r["episode"]: (float(np.median(r["stitched_err"])) / step, r["edge_shifts"]) for r in data["episodes"]
        }

    header = f"{'episode':>7} {'distance':>8} " + " ".join(
        f"{name + ' err':>14} {name + ' edges':>16}" for name in models
    )
    print(
        "distance: to the nearest training episode, in median nearest-neighbour units; err: stitched median, frames of motion"
    )
    print(header)
    for ep in sorted(distance, key=distance.get):
        cells = []
        for results in models.values():
            if ep in results:
                err, edges = results[ep]
                cells.append(f"{err:>14.2f} {str([int(e) for e in edges]):>16}")
            else:
                cells.append(f"{'-':>14} {'-':>16}")
        print(f"{ep:>7} {distance[ep]:>8.2f} " + " ".join(cells))
    for name, results in models.items():
        common = [ep for ep in distance if ep in results]
        if len(common) >= 3:
            rho, p = spearmanr([distance[ep] for ep in common], [results[ep][0] for ep in common])
            print(f"{name}: Spearman(distance, error) = {rho:+.2f} (p {p:.2f}, n {len(common)})")


if __name__ == "__main__":
    main()
