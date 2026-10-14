"""arc_core labels: the per-frame baseline is the recording itself, anchors shift by one frame, weights follow windows;
the curvature metric's progress per unit path length depends on the path's shape, not on the teleop speed."""

from __future__ import annotations

import numpy as np
import pytest

from vla_tools import arc_core
from vla_tools.resample_config import ResampleConfig
from vla_tools.robot_spec import RobotSpec

SPEC = RobotSpec(action_dim=4, joint_dims=(0, 1, 2), command_dims=(3,), state_dims=(0, 1, 2, 3), fps=10.0)
H = 8


def _actions(frames: int = 40) -> np.ndarray:
    rng = np.random.default_rng(3)
    actions = np.zeros((frames, 4))
    actions[:, :3] = np.cumsum(rng.normal(0.0, 0.02, (frames, 3)), axis=0)
    actions[15:30, 3] = 1.0
    return actions


def _expected_baseline(actions: np.ndarray, offset: int) -> np.ndarray:
    """actions[t + offset + k], k = 0..H-1, clamped to the last frame."""
    n = len(actions)
    index = np.clip(np.arange(n)[:, None] + offset + np.arange(H)[None, :], 0, n - 1)
    return actions[index]


@pytest.mark.parametrize(("anchor", "offset"), [("state", 0), ("action", 1)])
def test_time_with_span_equal_to_horizon_is_the_recorded_chunk(anchor: str, offset: int) -> None:
    actions = _actions()
    config = ResampleConfig(metric="time", span_frames=H, action_horizon=H, anchor=anchor)
    _, chunks = arc_core.chunks(actions, SPEC, config, windows=[])
    difference = float(np.abs(chunks.points - _expected_baseline(actions, offset)).max())
    print(f"anchor {anchor}: max difference to actions[t+{offset}:t+{offset}+H] {difference:.1e}")
    assert difference < 1e-9


def test_padding_and_window_weights() -> None:
    actions = _actions()
    config = ResampleConfig(metric="time", span_frames=H, action_horizon=H, key_weight=3.0, pad_weight=0.5,
                            anchor="state")  # fmt: skip
    _, chunks = arc_core.chunks(actions, SPEC, config, windows=[(10, 15)])
    n = len(actions)
    target = np.arange(n)[:, None] + np.arange(H)[None, :]
    np.testing.assert_array_equal(chunks.padded, target > n - 1)
    expected = np.where(target > n - 1, 0.5, np.where((target >= 10) & (target < 15), 3.0, 1.0))
    np.testing.assert_array_equal(chunks.weights, expected)


def test_event_density_respaces_progress_inside_windows() -> None:
    config = ResampleConfig(metric="time", span_frames=2 * H, action_horizon=H, event_density=2.0, anchor="state")
    curve = arc_core.progress_curve(_actions(), SPEC, config, windows=[(10, 15)])
    np.testing.assert_allclose(curve.ds[10:15], 2.0 * curve.ds[0])
    np.testing.assert_allclose(np.delete(curve.ds, range(10, 15)), curve.ds[0])


@pytest.mark.parametrize("metric", ["curvature", "second_difference"])
def test_progress_metrics_give_monotone_frames_and_binary_commands(metric: str) -> None:
    actions = _actions()
    config = ResampleConfig(metric=metric, span_frames=2 * H, action_horizon=H, anchor="state")
    curve, chunks = arc_core.chunks(actions, SPEC, config, windows=[])
    assert (np.diff(curve.s) >= 0).all()
    assert (np.diff(chunks.frames, axis=1) >= 0).all()
    assert chunks.frames.min() >= 0.0 and chunks.frames.max() <= len(actions) - 1
    assert set(np.unique(chunks.points[:, :, 3])) <= {0.0, 1.0}


def test_recorded_config_without_anchor_is_refused() -> None:
    with pytest.raises(ValueError, match="anchor"):
        ResampleConfig.from_dict({"metric": "time", "span_frames": 20, "action_horizon": 10})


def _plane() -> tuple[np.ndarray, np.ndarray]:
    """Two orthonormal joint-space directions spanning a plane tilted against every joint axis."""
    e1 = np.array([1.0, 0.5, -0.3])
    e1 /= np.linalg.norm(e1)
    e2 = np.array([0.2, -0.1, 1.0])
    e2 -= e2.dot(e1) * e1
    return e1, e2 / np.linalg.norm(e2)


STRAIGHT, RADIUS, TURN = 0.4, 0.25, np.pi  # straight 0.4 rad, arc of radius 0.25 over pi, straight 0.4 rad
PATH_LENGTH = 2 * STRAIGHT + RADIUS * TURN


def _bent_path(sigma: np.ndarray) -> np.ndarray:
    """Joint positions at arc length sigma along: straight, a tangent arc, straight."""
    e1, e2 = _plane()
    start = np.array([0.1, -0.2, 0.3])
    centre = start + STRAIGHT * e1 + RADIUS * e2
    phase = np.clip((sigma - STRAIGHT) / RADIUS, 0.0, TURN)[:, None]
    on_arc = centre + RADIUS * (np.sin(phase) * e1 - np.cos(phase) * e2)
    before = np.minimum(sigma - STRAIGHT, 0.0)[:, None] * e1
    exit_direction = np.cos(TURN) * e1 + np.sin(TURN) * e2
    after = np.maximum(sigma - STRAIGHT - RADIUS * TURN, 0.0)[:, None] * exit_direction
    return on_arc + before + after


def _record(q: np.ndarray) -> np.ndarray:
    actions = np.zeros((len(q), 4))
    actions[:, :3] = q
    return actions


def _speed_profiles() -> dict[str, np.ndarray]:
    """Arc length along the path at every recorded frame, for three teleop speed profiles."""
    t = np.linspace(0.0, 1.0, 960)
    speed = 0.3 + np.minimum(t / 0.15, 1.0) * (1.0 + 0.8 * np.sin(2.0 * np.pi * 3.0 * t))
    speed[528:553] = 0.0  # a 25-frame pause on the arc
    varying = np.concatenate([[0.0], np.cumsum(speed)])
    return {
        "steady v": np.linspace(0.0, PATH_LENGTH, 801),
        "steady 2v": np.linspace(0.0, PATH_LENGTH, 401),
        "accelerating, slowing, paused": varying * PATH_LENGTH / varying[-1],
    }


def test_curvature_progress_per_unit_length_is_independent_of_teleop_speed() -> None:
    config = ResampleConfig(metric="curvature", span_frames=2 * H, action_horizon=H, anchor="state")
    grid = np.linspace(0.0, PATH_LENGTH, 2001)
    progress = {}
    for name, sigma in _speed_profiles().items():
        q = _bent_path(sigma)
        curve = arc_core.progress_curve(_record(q), SPEC, config, windows=[])
        distinct, first = np.unique(sigma, return_index=True)
        progress[name] = np.interp(grid, distinct, (curve.s / curve.total)[first])
        length = np.linalg.norm(np.diff(q, axis=0), axis=1)
        inside = (sigma[:-1] > STRAIGHT + 0.1) & (sigma[1:] < STRAIGHT + RADIUS * TURN - 0.1)
        window = config.curvature_window * float(np.median(length[length > 0]))
        print(f"{name}: {len(sigma)} frames, window {window:.4f} rad, progress per unit length inside the arc "
              f"{curve.ds[inside].sum() / length[inside].sum():.4f} (R^-1/3 {RADIUS ** (-1 / 3):.4f}), "
              f"total progress {curve.total:.4f}")  # fmt: skip
    names = list(progress)
    spread = 0.0
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            difference = float(np.abs(progress[a] - progress[b]).max())
            spread = max(spread, difference)
            print(f"  {a} vs {b}: max normalized progress difference {difference:.4f}")
    print(f"spread across speed profiles: {spread:.4f} of the total")
    assert spread <= 0.01


def test_curvature_is_zero_on_a_straight_segment() -> None:
    direction = np.array([0.37, -0.81, 0.45])
    direction /= np.linalg.norm(direction)
    distance = 0.6 * np.linspace(0.0, 1.0, 200) ** 2  # accelerating from rest
    actions = _record(np.array([0.12, 0.5, -0.7]) + distance[:, None] * direction)
    per_length = {}
    for metric in ("curvature", "second_difference"):
        config = ResampleConfig(metric=metric, span_frames=2 * H, action_horizon=H, anchor="state")
        per_length[metric] = float(arc_core._raw_metric(actions, SPEC, config).sum() / distance[-1])
        print(f"accelerating straight line, {metric}: raw per unit length {per_length[metric]:.2e}")
    assert per_length["curvature"] < 1e-3


@pytest.mark.parametrize("radius", [0.1, 0.2, 0.4])
def test_curvature_on_a_circle_is_the_cube_root_of_inverse_radius(radius: float) -> None:
    e1, e2 = _plane()
    t = np.linspace(0.0, 1.0, 400)
    angle = 1.75 * np.pi * (t + 0.04 * np.sin(2.0 * np.pi * 3.0 * t))  # speed varies 0.25x to 1.75x
    q = radius * (np.cos(angle)[:, None] * e1 + np.sin(angle)[:, None] * e2)
    config = ResampleConfig(metric="curvature", span_frames=2 * H, action_horizon=H, anchor="state")
    raw = arc_core._raw_metric(_record(q), SPEC, config)[1:-1]  # the end samples have one chord only
    length = np.linalg.norm(np.diff(q, axis=0), axis=1)[1:-1]
    expected = radius ** (-1 / 3)
    error = np.abs(raw / length / expected - 1.0)
    print(f"radius {radius}: raw per unit length {raw.sum() / length.sum():.4f}, R^-1/3 {expected:.4f}, "
          f"max relative error {error.max():.2e}")  # fmt: skip
    assert error.max() <= 0.02


def test_recorded_curvature_config_without_window_is_refused() -> None:
    with pytest.raises(ValueError, match="second_difference"):
        ResampleConfig.from_dict({"metric": "curvature", "span_frames": 20, "action_horizon": 10, "anchor": "state"})
    config = ResampleConfig(metric="curvature", span_frames=20, action_horizon=10, anchor="state")
    assert ResampleConfig.from_dict(config.to_dict()) == config


def test_arc_metric_is_refused() -> None:
    with pytest.raises(ValueError, match="arc"):
        ResampleConfig.from_dict({"metric": "arc", "span_frames": 20, "action_horizon": 10, "anchor": "state"})
    config = ResampleConfig(metric="arc", span_frames=20, action_horizon=10, anchor="state")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="arc"):
        arc_core.progress_curve(_actions(), SPEC, config, windows=[])
