"""Keyframe detectors: pure functions from one episode's arrays to keyframe frames. No IO.

`detect_episode` runs the spec's types in order, so a type can be searched relative to an earlier one.
"""

from __future__ import annotations

import numpy as np

from vla_tools.keyframe_spec import Detector
from vla_tools.keyframe_spec import KeyframeSpec
from vla_tools.keyframe_spec import Signal
from vla_tools.robot_spec import RobotSpec


def signal_values(
    signal: Signal, actions: np.ndarray, states: np.ndarray, spec: RobotSpec, columns: dict[str, np.ndarray]
) -> np.ndarray:
    if signal.source == "action":
        return actions[:, signal.dim]
    if signal.source == "state":
        return states[:, signal.dim]
    if signal.source == "column":
        values = columns[signal.name]
        return values if values.ndim == 1 else values[:, signal.dim]
    speed = np.abs(np.diff(actions[:, list(spec.joint_dims)], axis=0)).max(axis=1)
    return np.concatenate([[0.0], speed])


def _held(mask: np.ndarray, start: int, hold: int) -> int | None:
    """First index >= start where `mask` is True for `hold` consecutive frames."""
    run = 0
    for t in range(start, len(mask)):
        run = run + 1 if mask[t] else 0
        if run >= hold:
            return t - hold + 1
    return None


def _crossing_mask(values: np.ndarray, threshold: float, direction: str) -> np.ndarray:
    return values > threshold if direction == "rise" else values < threshold


def _edges(values: np.ndarray, direction: str) -> list[int]:
    on = values > 0.5
    change = np.flatnonzero(np.diff(on.astype(np.int8)) != 0) + 1
    return [int(t) for t in change if bool(on[t]) == (direction == "rise")]


def detect(
    detector: Detector,
    actions: np.ndarray,
    states: np.ndarray,
    spec: RobotSpec,
    found: dict[str, list[int]],
    columns: dict[str, np.ndarray],
) -> list[int]:
    """Frames of one keyframe type in one episode."""
    if detector.type == "manual":
        return []
    values = signal_values(detector.signal or Signal("joint_speed"), actions, states, spec, columns)
    n = len(values)
    references = found.get(detector.reference, []) if detector.reference else None
    search = detector.search_frames or n

    if detector.type == "command_edge":
        edges = _edges(values, detector.direction or "rise")
        if references is None:
            return edges
        out = []
        for r in references:
            hit = next((e for e in edges if r <= e < r + search), None)
            if hit is not None:
                out.append(hit)
        return out

    if detector.type == "speed_minimum":
        out = []
        for r in references or []:
            lo = max(r - search, 0)
            if r > lo:
                out.append(lo + int(np.argmin(values[lo:r])))
        return out

    if detector.type == "settle":
        change = np.abs(np.concatenate([[0.0], np.diff(values)]))
        mask = change < float(detector.threshold)
        return [
            t
            for t in (_held(mask[: r + search], r + 1, detector.hold_frames) for r in references or [])
            if t is not None
        ]

    if detector.type in ("threshold_cross", "motion_onset"):
        direction = "rise" if detector.type == "motion_onset" else (detector.direction or "rise")
        mask = _crossing_mask(values, float(detector.threshold), direction)
        if references is None:
            out, t = [], 1
            while t < n:
                hit = _held(mask, t, detector.hold_frames)
                if hit is None:
                    break
                if hit == 0 or not mask[hit - 1]:
                    out.append(hit)
                t = hit + 1
                while t < n and mask[t]:
                    t += 1
            return out
        hits = (_held(mask[: r + search], r + 1, detector.hold_frames) for r in references)
        return [t for t in hits if t is not None]

    raise ValueError(f"unknown detector type {detector.type!r}")


def detect_episode(
    kspec: KeyframeSpec, actions: np.ndarray, states: np.ndarray, spec: RobotSpec, columns: dict[str, np.ndarray]
) -> dict[str, list[int]]:
    found: dict[str, list[int]] = {}
    for ktype in kspec.types:
        found[ktype.name] = sorted(detect(ktype.detector, actions, states, spec, found, columns))
    return found
