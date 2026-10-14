"""arc_core labels: the per-frame baseline is the recording itself, anchors shift by one frame, weights follow windows;
each frame's window ends at its H-th point, keyframe density is set by event_density alone, still windows are spaced
in time; the curvature metric's progress per unit path length depends on the path's shape, not on the teleop speed."""

from __future__ import annotations

import numpy as np
import pytest

from vla_tools import arc_core
from vla_tools.resample_config import ResampleConfig
from vla_tools.robot_spec import RobotSpec

SPEC = RobotSpec(action_dim=4, joint_dims=(0, 1, 2), command_dims=(3,), state_dims=(0, 1, 2, 3), fps=10.0)
H = 8
LENGTH = 0.1  # curvature chord length (rad) for the random-walk episode


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
    _, chunks = arc_core.chunks(actions, SPEC, config, windows=[], recorded=len(actions))
    difference = float(np.abs(chunks.points - _expected_baseline(actions, offset)).max())
    print(f"anchor {anchor}: max difference to actions[t+{offset}:t+{offset}+H] {difference:.1e}")
    assert difference < 1e-9


def test_padding_and_window_weights() -> None:
    actions = _actions()
    config = ResampleConfig(metric="time", span_frames=H, action_horizon=H, key_weight=3.0, pad_weight=0.5,
                            anchor="state")  # fmt: skip
    _, chunks = arc_core.chunks(actions, SPEC, config, windows=[(10, 15)], recorded=len(actions))
    n = len(actions)
    target = np.arange(n)[:, None] + np.arange(H)[None, :]
    np.testing.assert_array_equal(chunks.padded, target > n - 1)
    expected = np.where(target > n - 1, 0.5, np.where((target >= 10) & (target < 15), 3.0, 1.0))
    np.testing.assert_array_equal(chunks.weights, expected)


def test_time_labels_respace_keyframe_windows_by_event_density() -> None:
    config = ResampleConfig(metric="time", span_frames=2 * H, action_horizon=H, event_density=2.0, anchor="state")
    _, chunks = arc_core.chunks(_actions(), SPEC, config, windows=[(10, 15)], recorded=40)
    gaps = np.diff(chunks.frames, axis=1)
    inside = (chunks.frames[:, :-1] >= 10.0) & (chunks.frames[:, 1:] <= 15.0) & ~chunks.padded[:, 1:]
    outside = ((chunks.frames[:, 1:] <= 10.0) | (chunks.frames[:, :-1] >= 15.0)) & ~chunks.padded[:, 1:]
    print(
        f"gaps inside the window {np.unique(np.round(gaps[inside], 9))}, outside {np.unique(np.round(gaps[outside], 9))}"
    )
    np.testing.assert_allclose(gaps[inside], 1.0, atol=1e-9)
    np.testing.assert_allclose(gaps[outside], 2.0, atol=1e-9)
    np.testing.assert_allclose(chunks.step, 2.0)


@pytest.mark.parametrize("metric", ["curvature", "second_difference"])
def test_progress_metrics_give_monotone_frames_and_binary_commands(metric: str) -> None:
    actions = _actions()
    length = LENGTH if metric == "curvature" else None
    config = ResampleConfig(metric=metric, span_frames=2 * H, action_horizon=H, anchor="state", curvature_length=length)
    curve, chunks = arc_core.chunks(actions, SPEC, config, windows=[], recorded=len(actions))
    assert (curve.free >= 0).all()
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


PATH_CHORD = 4.0 * PATH_LENGTH / 800  # four steps of the steady profile


def test_curvature_progress_per_unit_length_is_independent_of_teleop_speed() -> None:
    config = ResampleConfig(metric="curvature", span_frames=2 * H, action_horizon=H, anchor="state",
                            curvature_length=PATH_CHORD)  # fmt: skip
    grid = np.linspace(0.0, PATH_LENGTH, 2001)
    progress = {}
    for name, sigma in _speed_profiles().items():
        q = _bent_path(sigma)
        curve = arc_core.progress_curve(_record(q), SPEC, config, windows=[], recorded=len(q))
        s = np.concatenate([[0.0], np.cumsum(curve.free)])
        distinct, first = np.unique(sigma, return_index=True)
        progress[name] = np.interp(grid, distinct, (s / s[-1])[first])
        length = np.linalg.norm(np.diff(q, axis=0), axis=1)
        inside = (sigma[:-1] > STRAIGHT + 0.1) & (sigma[1:] < STRAIGHT + RADIUS * TURN - 0.1)
        print(f"{name}: {len(sigma)} frames, chord {PATH_CHORD:.4f} rad, progress per unit length inside the arc "
              f"{curve.free[inside].sum() / length[inside].sum():.4f} (R^-1/3 {RADIUS ** (-1 / 3):.4f}), "
              f"total progress {s[-1]:.4f}")  # fmt: skip
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
        length = 0.02 if metric == "curvature" else None
        config = ResampleConfig(
            metric=metric, span_frames=2 * H, action_horizon=H, anchor="state", curvature_length=length
        )
        per_length[metric] = float(arc_core._raw_metric(actions, SPEC, config, len(actions)).sum() / distance[-1])
        print(f"accelerating straight line, {metric}: raw per unit length {per_length[metric]:.2e}")
    assert per_length["curvature"] < 1e-3


@pytest.mark.parametrize("radius", [0.1, 0.2, 0.4])
def test_curvature_on_a_circle_is_the_cube_root_of_inverse_radius(radius: float) -> None:
    e1, e2 = _plane()
    t = np.linspace(0.0, 1.0, 400)
    angle = 1.75 * np.pi * (t + 0.04 * np.sin(2.0 * np.pi * 3.0 * t))  # speed varies 0.25x to 1.75x
    q = radius * (np.cos(angle)[:, None] * e1 + np.sin(angle)[:, None] * e2)
    steps = np.linalg.norm(np.diff(q, axis=0), axis=1)
    config = ResampleConfig(metric="curvature", span_frames=2 * H, action_horizon=H, anchor="state",
                            curvature_length=4.0 * float(np.median(steps)))  # fmt: skip
    raw = arc_core._raw_metric(_record(q), SPEC, config, len(q))[1:-1]  # the end samples have one chord only
    length = steps[1:-1]
    expected = radius ** (-1 / 3)
    error = np.abs(raw / length / expected - 1.0)
    print(f"radius {radius}: raw per unit length {raw.sum() / length.sum():.4f}, R^-1/3 {expected:.4f}, "
          f"max relative error {error.max():.2e}")  # fmt: skip
    assert error.max() <= 0.02


def test_curvature_config_needs_a_chord_length() -> None:
    raw = {"metric": "curvature", "span_frames": 20, "action_horizon": 10, "anchor": "state"}
    with pytest.raises(ValueError, match="second_difference"):
        ResampleConfig.from_dict(raw)
    with pytest.raises(ValueError, match="curvature_window"):
        ResampleConfig.from_dict({**raw, "curvature_window": 4.0})
    with pytest.raises(ValueError, match="curvature_length"):
        ResampleConfig(metric="curvature", span_frames=20, action_horizon=10, anchor="state")
    config = ResampleConfig(
        metric="curvature", span_frames=20, action_horizon=10, anchor="state", curvature_length=0.12
    )
    assert ResampleConfig.from_dict(config.to_dict()) == config


def test_arc_metric_is_refused() -> None:
    with pytest.raises(ValueError, match="arc"):
        ResampleConfig.from_dict({"metric": "arc", "span_frames": 20, "action_horizon": 10, "anchor": "state"})
    config = ResampleConfig(metric="arc", span_frames=20, action_horizon=10, anchor="state")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="arc"):
        arc_core.progress_curve(_actions(), SPEC, config, windows=[], recorded=40)


def _padded(actions: np.ndarray, rows: int) -> np.ndarray:
    """End padding as episode_timing writes it: rows holding the last recorded target."""
    return np.concatenate([actions, np.repeat(actions[-1:], rows, axis=0)])


def _config(metric: str, span: int, density: float) -> ResampleConfig:
    length = PATH_CHORD if metric == "curvature" else None
    return ResampleConfig(metric=metric, span_frames=span, action_horizon=H, event_density=density, anchor="state",
                          curvature_length=length)  # fmt: skip


@pytest.mark.parametrize("metric", ["time", "curvature", "second_difference"])
def test_end_padding_does_not_change_labels_of_recorded_frames(metric: str) -> None:
    sigma = _speed_profiles()["accelerating, slowing, paused"]
    recorded = _record(_bent_path(sigma))
    config = _config(metric, 40, 2.0)
    windows = [(300, 320)]
    results = {}
    for rows in (10, 40):
        actions = _padded(recorded, rows)
        results[rows] = arc_core.chunks(actions, SPEC, config, windows, recorded=len(recorded))
    (c10, k10), (c40, k40) = results[10], results[40]
    n = len(recorded)
    clean = ~k40.padded[:n].any(axis=1)
    progress = float(np.abs(c10.free[: n - 1] - c40.free[: n - 1]).max())
    steps = float(np.abs(k10.step[:n] - k40.step[:n]).max())
    points = float(np.abs(k10.points[:n][clean] - k40.points[:n][clean]).max())
    reach = np.median(k40.frames[:n][clean, -1] - np.arange(n)[clean] + 1)
    print(f"{metric}: progress over recorded rows differs by {progress:.1e}, steps by {steps:.1e}, labels of "
          f"{int(clean.sum())} unpadded frames by {points:.1e}; median reach {reach:.1f} frames for span 40")  # fmt: skip
    assert progress == 0.0
    assert steps == 0.0
    assert points <= 1e-9  # the strict-progress drift scales with the total, which the padding rows change


@pytest.mark.parametrize("metric", ["time", "curvature", "second_difference"])
def test_window_without_keyframes_ends_at_its_last_point(metric: str) -> None:
    sigma = _speed_profiles()["accelerating, slowing, paused"]
    actions = _padded(_record(_bent_path(sigma)), 40)
    n, span = len(sigma), 40
    _, chunks = arc_core.chunks(actions, SPEC, _config(metric, span, 0.0), [], recorded=n)
    frames = np.arange(1, n - span + 1)  # state row t-1 >= 0, window end t-1+span <= n-1
    end = actions[frames - 1 + span, :3]
    worst = float(np.abs(chunks.points[frames, -1, :3] - end).max())
    reach = chunks.frames[frames, -1] - (frames - 1)
    padded = int(chunks.padded[frames].sum())
    print(f"{metric}: {len(frames)} windows, H-th point vs the recorded row span frames after the state: max {worst:.1e}; "
          f"reach in rows p05 {np.percentile(reach, 5):.2f} p50 {np.median(reach):.2f} max {reach.max():.2f}; "
          f"padded points {padded}")  # fmt: skip
    assert worst <= 1e-9
    assert padded == 0  # the last of these windows ends on the last recorded row, not past it


@pytest.mark.parametrize("metric", ["time", "curvature"])
def test_keyframe_density_does_not_depend_on_the_rest_of_the_window(metric: str) -> None:
    sigma = _speed_profiles()["accelerating, slowing, paused"]
    actions = _record(_bent_path(sigma))
    span, density, window = 40, 2.0, (300, 320)
    _, chunks = arc_core.chunks(actions, SPEC, _config(metric, span, density), [window], recorded=len(sigma))
    gaps = np.diff(chunks.frames, axis=1)
    inside = (chunks.frames[:, :-1] >= window[0]) & (chunks.frames[:, 1:] <= window[1])
    expected = (span / H) / density
    a = np.arange(len(sigma)) - 1
    touching = (a + span > window[0]) & (a < window[1]) & (a >= 0) & (a + span <= len(sigma) - 1)
    reach = chunks.frames[touching, -1] - a[touching]
    print(f"{metric}: {int(inside.sum())} gaps inside the keyframe window, all {expected:g} rows: "
          f"max deviation {np.abs(gaps[inside] - expected).max():.1e}; windows touching it reach "
          f"p50 {np.median(reach):.1f} / max {reach.max():.1f} rows of span {span}")  # fmt: skip
    np.testing.assert_allclose(gaps[inside], expected, atol=1e-9)
    assert reach.max() <= span + 1e-9


def test_still_window_is_spaced_in_time_and_keeps_its_command() -> None:
    moving = np.linspace(0.0, 0.3, 30)[:, None] * np.array([1.0, 0.4, -0.2])
    still = np.repeat(moving[-1:], 40, axis=0)
    actions = _record(np.concatenate([moving, still, moving[-1] + moving[1:]]))
    actions[45:, 3] = 1.0  # the gripper closes while the arm stands still
    config = _config("curvature", 2 * H, 0.0)
    _, chunks = arc_core.chunks(actions, SPEC, config, [], recorded=len(actions))
    t = 36  # state row 35, window rows 35..51: inside the still stretch
    print(
        f"still window: frames {np.round(chunks.frames[t], 6)}, commands {chunks.points[t, :, 3]}, step {chunks.step[t]}"
    )
    np.testing.assert_allclose(chunks.frames[t], 35.0 + np.arange(1, H + 1) * 2.0, atol=1e-9)
    assert chunks.points[t, :, 3].max() == 1.0


def test_recorded_rows_must_be_within_the_episode() -> None:
    config = ResampleConfig(metric="time", span_frames=H, action_horizon=H, anchor="state")
    with pytest.raises(ValueError, match="recorded"):
        arc_core.chunks(_actions(), SPEC, config, windows=[], recorded=41)
