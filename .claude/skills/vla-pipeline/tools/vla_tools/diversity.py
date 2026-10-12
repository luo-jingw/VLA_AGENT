"""Data diversity and coverage: what the episodes span, where they cluster, what repeats, and a held-out split.

Per descriptor group (see diversity_descriptors): spread, clusters (Ward linkage, cut at the largest
gap in merge height), and the episodes farthest from all others. Over all groups combined (each group
standardised and weighted equally): clusters, near-duplicate pairs, isolated episodes, and a 2-D map.

The split keeps every cluster of every group represented in training and records, for each held-out
episode, its distance to the nearest training episode -- the axis model errors are later read along
(coverage_eval). Strategies:
  stratified     held-out drawn from each combined cluster in proportion: in-distribution generalisation
  extrapolation  the most isolated episodes: how the model degrades at the edge of the data

Usage:
    python -m vla_tools.diversity --dataset <dir> --spec robot_spec.json --keyframes keyframes.json \
        --holdout 3 --strategy stratified --out diversity.json --plot diversity.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform

from vla_tools.diversity_descriptors import Descriptors
from vla_tools.diversity_descriptors import describe
from vla_tools.diversity_descriptors import static_dims
from vla_tools.episodes import FrameReader
from vla_tools.episodes import load_episodes
from vla_tools.keyframe_record import KeyframeRecord
from vla_tools.robot_spec import RobotSpec


def standardise(matrix: np.ndarray) -> np.ndarray:
    """Zero mean, unit variance per column, then scaled so the whole group weighs as one unit."""
    std = matrix.std(axis=0)
    std[std < 1e-12] = 1.0
    return (matrix - matrix.mean(axis=0)) / std / np.sqrt(matrix.shape[1])


def clusters(matrix: np.ndarray, max_k: int) -> np.ndarray:
    """Ward clusters, cut where the merge height jumps most. All ones when no cut is clear."""
    if len(matrix) < 4 or np.allclose(matrix, matrix[0]):
        return np.ones(len(matrix), dtype=int)
    tree = hierarchy.linkage(matrix, method="ward")
    heights = tree[:, 2]
    top = heights[-(max_k):]
    gaps = np.diff(top)
    if len(gaps) == 0 or gaps.max() < 0.5 * heights[-1] / max(len(heights), 1):
        return np.ones(len(matrix), dtype=int)
    k = max_k - int(np.argmax(gaps))
    k = int(np.clip(k, 1, max_k))
    labels = hierarchy.fcluster(tree, t=k, criterion="maxclust")
    return labels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--keyframes", type=Path, required=True)
    parser.add_argument("--cameras", nargs="*", default=None, help="image columns; default the spec's first")
    parser.add_argument("--static-threshold", type=float, default=1e-3)
    parser.add_argument("--max-clusters", type=int, default=6)
    parser.add_argument("--duplicate-factor", type=float, default=0.25)
    parser.add_argument("--holdout", type=int, default=0)
    parser.add_argument("--strategy", choices=["stratified", "extrapolation"], default="stratified")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--plot", type=Path, default=None)
    parser.add_argument("--montage", type=Path, default=None, help="first frames grouped by combined cluster")
    args = parser.parse_args()

    spec = RobotSpec.load(args.spec)
    record = KeyframeRecord.load(args.keyframes)
    record.check_dataset(args.dataset)
    episodes = load_episodes(args.dataset, spec, only=sorted(record.episodes))
    static = static_dims(episodes, spec, args.static_threshold)
    cameras = args.cameras if args.cameras is not None else list(spec.image_keys[:1])
    reader = FrameReader(args.dataset)
    counts: dict[str, int] = {}
    for e in episodes:
        for k in record.episodes[e.index]:
            counts[k["type"]] = counts.get(k["type"], 0) + 1
    once = sorted(
        t
        for t, c in counts.items()
        if all(sum(k["type"] == t for k in record.episodes[e.index]) == 1 for e in episodes)
    )

    described: list[Descriptors] = []
    for e in episodes:
        frames = {t: next(k["frame"] for k in record.episodes[e.index] if k["type"] == t) for t in once}
        images = {c: reader.frame(e.index, c, 0) for c in cameras}
        described.append(describe(e, spec, frames, static, images))
    ids = np.array([d.episode for d in described])
    names = list(described[0].groups)
    groups = {g: np.stack([d.groups[g] for d in described]) for g in names}

    print(f"{len(episodes)} episodes; static joints {static}; groups: {', '.join(names)}")
    print(f"{'group':<34} {'dims':>4} {'spread':>9} {'clusters':>9}  sizes / most isolated episodes")
    group_labels: dict[str, list[int]] = {}
    for g, matrix in groups.items():
        z = standardise(matrix)
        labels = clusters(z, args.max_clusters)
        group_labels[g] = labels.tolist()
        sizes = sorted(np.bincount(labels)[1:].tolist(), reverse=True)
        distance = squareform(pdist(z))
        np.fill_diagonal(distance, np.inf)
        isolated = ids[np.argsort(-distance.min(axis=1))[:3]].tolist()
        spread = (
            float(np.sqrt((matrix.std(axis=0) ** 2).sum())) if not g.startswith("image") else float(np.median(pdist(z)))
        )
        print(f"{g:<34} {matrix.shape[1]:>4} {spread:>9.4f} {len(sizes):>9}  {sizes} / {isolated}")

    combined = np.concatenate([standardise(m) for m in groups.values()], axis=1)
    distance = squareform(pdist(combined))
    np.fill_diagonal(distance, np.inf)
    nearest = distance.min(axis=1)
    labels = clusters(combined, args.max_clusters)
    threshold = args.duplicate_factor * float(np.median(nearest))
    duplicates = sorted(
        {tuple(sorted((int(ids[i]), int(ids[j])))) for i, j in zip(*np.where(distance < threshold), strict=True)}
    )
    median_nn = float(np.median(nearest))
    mad = 1.4826 * float(np.median(np.abs(nearest - median_nn))) + 1e-9
    isolated = [int(ids[i]) for i in np.flatnonzero((nearest - median_nn) / mad > 3.5)]
    print(f"\ncombined: {len(set(labels))} clusters {sorted(np.bincount(labels)[1:].tolist(), reverse=True)}; "
          f"near-duplicate pairs {duplicates or 'none'}; isolated {isolated or 'none'}")  # fmt: skip

    split = None
    if args.holdout > 0:
        rng = np.random.default_rng(args.seed)
        if args.strategy == "extrapolation":
            chosen = list(np.argsort(-nearest)[: args.holdout])
        else:
            # Proportional allocation by largest remainder, never taking a cluster's last member.
            members = {c: rng.permutation(np.flatnonzero(labels == c)).tolist() for c in sorted(set(labels))}
            share = {c: len(m) / len(ids) * args.holdout for c, m in members.items()}
            alloc = {c: min(int(np.floor(share[c])), len(members[c]) - 1) for c in members}
            for c in sorted(members, key=lambda c: -(share[c] - np.floor(share[c]))):
                if sum(alloc.values()) < args.holdout and alloc[c] < len(members[c]) - 1:
                    alloc[c] += 1
            for c in sorted(members, key=lambda c: -len(members[c])):
                while sum(alloc.values()) < args.holdout and alloc[c] < len(members[c]) - 1:
                    alloc[c] += 1
            chosen = [i for c in members for i in members[c][: alloc[c]]]
        held = sorted(int(i) for i in chosen)
        train = [i for i in range(len(ids)) if i not in held]
        lost = [
            f"{g} cluster {c}"
            for g, lab in group_labels.items()
            for c in set(lab)
            if not any(lab[i] == c for i in train)
        ]
        to_train = {int(ids[i]): float(distance[i, train].min()) for i in held}
        split = {
            "strategy": args.strategy,
            "seed": args.seed,
            "train": [int(ids[i]) for i in train],
            "holdout": [int(ids[i]) for i in held],
            "holdout_distance_to_train": to_train,
            "median_nearest_neighbour": median_nn,
            "clusters_missing_from_train": lost,
        }
        print(f"\nsplit ({args.strategy}): holdout {split['holdout']}; distance to nearest train episode "
              f"{ {k: round(v / median_nn, 2) for k, v in to_train.items()} } (x median nearest-neighbour)")  # fmt: skip
        if lost:
            print(f"  clusters with no training episode: {lost}")

    if args.plot is not None:
        u, s, _ = np.linalg.svd(combined - combined.mean(axis=0), full_matrices=False)
        xy = u[:, :2] * s[:2]
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(xy[:, 0], xy[:, 1], c=labels, cmap="tab10", s=40)
        for (x, y), ep in zip(xy, ids, strict=True):
            held = split is not None and int(ep) in split["holdout"]
            ax.annotate(str(ep) + ("*" if held else ""), (x, y), fontsize=7)
        ax.set_title("episodes, all descriptor groups (PCA); * = held out", fontsize=9)
        fig.tight_layout()
        fig.savefig(args.plot, dpi=90)

    if args.montage is not None and cameras:
        order = np.argsort(labels, kind="stable")
        cols = 6
        rows = int(np.ceil(len(order) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(2.4 * cols, 2.0 * rows), squeeze=False)
        for ax in axes.ravel():
            ax.axis("off")
        for slot, i in enumerate(order):
            ax = axes.ravel()[slot]
            ax.imshow(reader.frame(int(ids[i]), cameras[0], 0))
            ax.set_title(f"ep {ids[i]}  cluster {labels[i]}", fontsize=7)
        fig.tight_layout()
        fig.savefig(args.montage, dpi=60)

    payload = {
        "dataset": str(args.dataset),
        "static_dims": static,
        "groups": {g: {"dims": int(m.shape[1]), "clusters": group_labels[g]} for g, m in groups.items()},
        "episodes": ids.tolist(),
        "combined_clusters": labels.tolist(),
        "nearest_neighbour": nearest.tolist(),
        "near_duplicates": duplicates,
        "isolated": isolated,
        "split": split,
    }
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
