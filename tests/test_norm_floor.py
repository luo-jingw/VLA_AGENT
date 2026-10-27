"""norm_floor on hand-built statistics: pooled (D,) and per-point (H, D)."""

from __future__ import annotations

import numpy as np

from vla_tools.norm_floor import QuantileStats
from vla_tools.norm_floor import floor_dims


def test_pooled_statistics_floor_only_the_named_narrow_dims() -> None:
    stats = QuantileStats(mean=[0.0, 0.0, 0.0], std=[1e-5, 1e-5, 0.3], q01=[-1e-4, 0.1, -0.5], q99=[1e-4, 0.1001, 0.5])
    floored, report = floor_dims(stats, [0, 2], min_range=0.05)
    print(floored, report)
    np.testing.assert_allclose(floored["q01"], [-0.025, 0.1, -0.5])
    np.testing.assert_allclose(floored["q99"], [0.025, 0.1001, 0.5])
    np.testing.assert_allclose(floored["std"], [0.0125, 1e-5, 0.3])
    assert [r.dim for r in report] == [0, 2]
    assert stats["q01"] == [-1e-4, 0.1, -0.5]


def test_per_point_statistics_floor_every_point_of_a_dim() -> None:
    q01 = [[-1e-4, -0.1], [0.2, -0.2], [-0.1, -0.3]]
    q99 = [[1e-4, 0.1], [0.2002, 0.2], [0.1, 0.3]]
    stats = QuantileStats(mean=[[0.0, 0.0]] * 3, std=[[1e-5, 0.05]] * 3, q01=q01, q99=q99)
    floored, report = floor_dims(stats, [0], min_range=0.05)
    print(floored, report)
    np.testing.assert_allclose(np.array(floored["q01"])[:, 0], [-0.025, 0.2001 - 0.025, -0.1])
    np.testing.assert_allclose(np.array(floored["q99"])[:, 0], [0.025, 0.2001 + 0.025, 0.1])
    np.testing.assert_allclose(np.array(floored["q01"])[:, 1], [-0.1, -0.2, -0.3])
    np.testing.assert_allclose(np.array(floored["std"])[:, 0], [0.0125] * 3)
    assert report[0].range_before < 1e-3 and report[0].range_after >= 0.05 - 1e-12
