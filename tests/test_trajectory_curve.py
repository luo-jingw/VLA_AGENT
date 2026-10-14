"""trajectory_curve: the rebuilt curve passes through every recorded row, ignores end padding, and stays at the
last recorded row beyond it."""

from __future__ import annotations

import numpy as np
import pytest

from vla_tools import trajectory_curve


def _joints(rows: int) -> np.ndarray:
    rng = np.random.default_rng(11)
    return np.cumsum(rng.normal(0.0, 0.03, (rows, 3)), axis=0)


def test_curve_passes_through_every_recorded_row() -> None:
    joints = _joints(50)
    curve = trajectory_curve.build(joints, recorded=50)
    worst = float(np.abs(curve.joints(np.arange(50.0)) - joints).max())
    print(f"curve at the recorded rows vs the rows: max {worst:.1e}")
    assert worst <= 1e-12


def test_end_padding_does_not_shape_the_curve() -> None:
    joints = _joints(50)
    grid = np.linspace(0.0, 49.0, 981)
    curves = {rows: trajectory_curve.build(np.concatenate([joints, np.repeat(joints[-1:], rows, axis=0)]), 50)
              for rows in (10, 40)}  # fmt: skip
    difference = float(np.abs(curves[10].joints(grid) - curves[40].joints(grid)).max())
    beyond = float(np.abs(curves[40].joints(np.array([49.5, 60.0, 89.0])) - joints[-1]).max())
    print(f"10 vs 40 padding rows: max {difference:.1e}; past the last recorded row vs that row: max {beyond:.1e}")
    assert difference == 0.0
    assert beyond == 0.0


def test_single_recorded_row_is_a_constant_curve() -> None:
    joints = _joints(5)
    curve = trajectory_curve.build(joints, recorded=1)
    np.testing.assert_array_equal(curve.joints(np.array([0.0, 2.5])), np.repeat(joints[:1], 2, axis=0))


def test_recorded_rows_must_be_within_the_joints() -> None:
    with pytest.raises(ValueError, match="recorded"):
        trajectory_curve.build(_joints(5), recorded=6)
