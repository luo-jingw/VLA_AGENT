"""Time-free comparisons of deployed joint targets with a recorded joint path. Pure functions, no IO.

For static tasks the timing along a demonstrated path does not matter; where the arm goes and where it
switches a command does. Every function here compares geometry only:

    step statistics     size of one recorded per-frame step, the unit the other figures are read in
    distance to path    distance of each target to the recorded polyline (projection onto its segments)
    path deviation      the largest of those distances
    keyframe position   distance between the target at which a command first switches and the recorded
                        joints at the command edge
    speed ratio         largest per-frame step of the targets relative to a recorded reference step

All distances are joint-space Euclidean, in radians. Paths and targets are (N, D) arrays of joint values
only; command dims are passed separately where needed.
"""

from __future__ import annotations

import numpy as np


def _steps(path: np.ndarray) -> np.ndarray:
    """(N, D) path -> (N-1,) Euclidean length of each step."""
    return np.linalg.norm(np.diff(np.asarray(path, dtype=np.float64), axis=0), axis=1)


def step_statistics(paths: list[np.ndarray]) -> tuple[float, float]:
    """Median and 95th percentile of the non-zero per-frame steps over all paths.

    Zero steps (frames that repeat the previous one, e.g. end padding) are left out. Raises when no path
    moves at all, since there is then no step to measure anything in.
    """
    steps = np.concatenate([_steps(p) for p in paths]) if paths else np.zeros(0)
    moving = steps[steps > 0.0]
    if len(moving) == 0:
        raise ValueError("no non-zero step in the given paths")
    return float(np.median(moving)), float(np.percentile(moving, 95))


def distance_to_path(points: np.ndarray, path: np.ndarray) -> np.ndarray:
    """(M, D) points, (N, D) polyline -> (M,) distance of each point to its nearest point on the polyline.

    A one-row path is a single point. Segments of zero length reduce to their start point.
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, np.shape(path)[-1])
    path = np.asarray(path, dtype=np.float64)
    if len(path) == 0:
        raise ValueError("empty path")
    if len(path) == 1:
        return np.linalg.norm(points - path[0], axis=1)
    start, direction = path[:-1], np.diff(path, axis=0)  # (S, D)
    length2 = np.einsum("sd,sd->s", direction, direction)  # (S,)
    offset = points[:, None, :] - start[None, :, :]  # (M, S, D)
    along = np.einsum("msd,sd->ms", offset, direction)
    fraction = np.clip(np.divide(along, length2, out=np.zeros_like(along), where=length2 > 0.0), 0.0, 1.0)
    nearest = start[None, :, :] + fraction[:, :, None] * direction[None, :, :]
    return np.linalg.norm(points[:, None, :] - nearest, axis=2).min(axis=1)


def path_deviation(targets: np.ndarray, path: np.ndarray) -> float:
    """Largest distance of the (M, D) targets to the (N, D) polyline; 0 when there are no targets."""
    if len(targets) == 0:
        return 0.0
    return float(distance_to_path(targets, path).max())


def keyframe_position_error(
    targets_joints: np.ndarray,
    targets_cmd: np.ndarray,
    current_cmd: bool,
    recorded_joints: np.ndarray,
    recorded_edge_index: int,
) -> float | None:
    """Where the targets switch a command, against where the recording switched it.

    targets_joints       (M, D) joint targets
    targets_cmd          (M,) one command dim of the same targets; values above 0.5 are on
    current_cmd          the command before the recorded edge
    recorded_joints      (T, D) recorded joint path
    recorded_edge_index  row of `recorded_joints` at which the recorded command switches

    Returns the distance between the first target row whose command differs from `current_cmd` and
    recorded_joints[recorded_edge_index], or None when the targets never switch.
    """
    switched = np.flatnonzero((np.asarray(targets_cmd) > 0.5) != bool(current_cmd))
    if len(switched) == 0:
        return None
    row = int(switched[0])
    return float(
        np.linalg.norm(np.asarray(targets_joints[row], dtype=np.float64) - recorded_joints[recorded_edge_index])
    )


def speed_ratio(targets: np.ndarray, pose: np.ndarray, reference_step: float) -> float:
    """Largest Euclidean step along [pose, targets...] divided by `reference_step`.

    targets (M, D) joint targets, one per frame; pose (D,) the joints they start from.
    """
    if reference_step <= 0.0:
        raise ValueError(f"reference_step must be positive, got {reference_step}")
    path = np.concatenate([np.asarray(pose, dtype=np.float64)[None, :], np.asarray(targets, dtype=np.float64)])
    return float(_steps(path).max()) / reference_step if len(path) > 1 else 0.0
