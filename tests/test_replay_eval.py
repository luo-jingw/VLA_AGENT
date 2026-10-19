from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("openpi_client")

from vla_tools.replay_eval import edge_shifts  # noqa: E402


def _commands(length: int, switches: list[int], dims: int = 2, dim: int = 1) -> np.ndarray:
    """(length, dims) rows whose command dim toggles at each switch frame, starting open (0)."""
    rows = np.zeros((length, dims))
    value = 0.0
    start = 0
    for s in [*switches, length]:
        rows[start:s, dim] = value
        value = 1.0 - value
        start = s
    return rows


def test_shifts_are_signed_frames_to_the_switch_of_the_same_direction() -> None:
    recorded = _commands(60, [20, 40])
    stitched = _commands(60, [22, 39])
    assert edge_shifts(stitched, recorded, [1]) == ([2, -1], 0)


def test_an_edge_never_switched_near_it_is_missed() -> None:
    recorded = _commands(60, [20, 40])
    stitched = _commands(60, [21])  # closes, never opens
    assert edge_shifts(stitched, recorded, [1]) == ([1], 1)


def test_a_switch_beyond_half_way_to_the_next_edge_does_not_count() -> None:
    recorded = _commands(80, [20, 40])
    stitched = _commands(80, [32, 60])  # close 12 frames late: past the midpoint (30) towards the open edge
    assert edge_shifts(stitched, recorded, [1]) == ([20], 1)  # the open at 60 is within (30, 80]


def test_each_command_dim_is_scored_on_its_own_edges() -> None:
    recorded = np.concatenate([_commands(50, [10], dims=1, dim=0), _commands(50, [30], dims=1, dim=0)], axis=1)
    stitched = np.concatenate([_commands(50, [13], dims=1, dim=0), _commands(50, [28], dims=1, dim=0)], axis=1)
    assert edge_shifts(stitched, recorded, [0, 1]) == ([3, -2], 0)
