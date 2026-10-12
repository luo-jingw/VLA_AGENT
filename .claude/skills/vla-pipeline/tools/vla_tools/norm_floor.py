"""Give near-constant dims a minimum normalisation range in an openpi norm_stats.json. Writes a new file.

Dims that never move (a parked arm) have delta-action quantiles at the level of encoder noise, e.g.
q99 - q01 ~ 2e-4 rad. Quantile normalisation divides by that range, so the model is trained to predict
noise stretched to [-1, 1]. Widening the range to --min-range around its centre maps the same noise to
values near 0; mean is unchanged and std is raised to at least min_range / 4.

Apply it after compute_norm_stats and before training, for the dims the audit marked static, and keep
the report: the checkpoint's assets carry the edited file.

Usage:
    python -m vla_tools.norm_floor --stats norm_stats.json --key actions --dims 7 8 9 10 11 12 \
        --min-range 0.05 --out norm_stats.floored.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--key", default="actions")
    parser.add_argument("--dims", nargs="+", type=int, required=True)
    parser.add_argument("--min-range", type=float, required=True, help="in the key's units (rad for joint deltas)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    raw = json.loads(args.stats.read_text(encoding="utf-8"))
    stats = raw["norm_stats"][args.key]
    print(f"{args.key}: dim  q01 -> q01'  q99 -> q99'  std -> std'")
    for d in args.dims:
        q01, q99, std = stats["q01"][d], stats["q99"][d], stats["std"][d]
        if q99 - q01 < args.min_range:
            centre = 0.5 * (q01 + q99)
            stats["q01"][d] = centre - 0.5 * args.min_range
            stats["q99"][d] = centre + 0.5 * args.min_range
        stats["std"][d] = max(std, args.min_range / 4)
        print(f"  {d:>3}  {q01:+.2e} -> {stats['q01'][d]:+.2e}  {q99:+.2e} -> {stats['q99'][d]:+.2e}  "
              f"{std:.2e} -> {stats['std'][d]:.2e}")  # fmt: skip
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(raw, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
