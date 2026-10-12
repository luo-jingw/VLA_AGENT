"""arc_core labels: the per-frame baseline is the recording itself, anchors shift by one frame, weights follow windows."""

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


@pytest.mark.parametrize("metric", ["arc", "curvature"])
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
