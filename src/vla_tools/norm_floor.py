"""Give near-constant dims a minimum normalisation range in an openpi norm_stats.json. Writes a new file.

Dims that never move (a parked arm) have delta-action quantiles at the level of encoder noise, e.g.
q99 - q01 ~ 2e-4 rad. Quantile normalisation divides by that range, so the model is trained to predict
noise stretched to [-1, 1]. Widening the range to --min-range around its centre maps the same noise to
values near 0; mean is unchanged and std is raised to at least min_range / 4.

Statistics are pooled, one value per dim (D,), or per point of the chunk, (H, D) (`run.py norm-stats
--per-point-actions`); a dim is the last axis in both, and every point of a floored dim is floored.

Apply it after compute_norm_stats and before training, for the dims the audit marked static, and keep
the report: the checkpoint's assets carry the edited file.

Usage:
    python -m vla_tools.norm_floor --stats norm_stats.json --key actions --dims 7 8 9 10 11 12 \
        --min-range 0.05 --out norm_stats.floored.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
from typing import TypedDict

import numpy as np


class QuantileStats(TypedDict):
    """One key of openpi's norm_stats.json; each entry is (D,) or (H, D) nested lists."""

    mean: list
    std: list
    q01: list
    q99: list


@dataclasses.dataclass(frozen=True)
class DimFloor:
    dim: int
    range_before: float  # smallest q99 - q01 over the dim's points
    range_after: float
    std_before: float  # smallest std over the dim's points
    std_after: float


def floor_dims(stats: QuantileStats, dims: list[int], min_range: float) -> tuple[QuantileStats, list[DimFloor]]:
    """The statistics with every point of `dims` given a q99 - q01 of at least `min_range` and a std of at
    least min_range / 4, and what changed per dim. The input is not modified."""
    q01, q99, std = (np.array(stats[key], dtype=np.float64) for key in ("q01", "q99", "std"))
    report = []
    for d in dims:
        lo, hi, sd = q01[..., d].copy(), q99[..., d].copy(), std[..., d].copy()
        narrow = hi - lo < min_range
        centre = 0.5 * (lo + hi)
        q01[..., d] = np.where(narrow, centre - 0.5 * min_range, lo)
        q99[..., d] = np.where(narrow, centre + 0.5 * min_range, hi)
        std[..., d] = np.maximum(sd, min_range / 4)
        report.append(
            DimFloor(
                dim=d,
                range_before=float(np.min(hi - lo)),
                range_after=float(np.min(q99[..., d] - q01[..., d])),
                std_before=float(np.min(sd)),
                std_after=float(np.min(std[..., d])),
            )
        )
    floored = QuantileStats(mean=stats["mean"], std=std.tolist(), q01=q01.tolist(), q99=q99.tolist())
    return floored, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--key", default="actions")
    parser.add_argument("--dims", nargs="+", type=int, required=True)
    parser.add_argument("--min-range", type=float, required=True, help="in the key's units (rad for joint deltas)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    raw = json.loads(args.stats.read_text(encoding="utf-8"))
    floored, report = floor_dims(raw["norm_stats"][args.key], args.dims, args.min_range)
    raw["norm_stats"][args.key] = raw["norm_stats"][args.key] | floored
    shape = np.shape(floored["q01"])
    print(f"{args.key} {shape}: dim  smallest q99-q01 -> after  smallest std -> after")
    for r in report:
        print(f"  {r.dim:>3}  {r.range_before:.2e} -> {r.range_after:.2e}  {r.std_before:.2e} -> {r.std_after:.2e}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(raw, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
