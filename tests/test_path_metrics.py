"""path_metrics on hand-built joint paths with known distances, edges and steps."""

from __future__ import annotations

import numpy as np

from vla_tools.path_metrics import distance_to_path
from vla_tools.path_metrics import keyframe_position_error
from vla_tools.path_metrics import path_deviation
from vla_tools.path_metrics import speed_ratio
from vla_tools.path_metrics import step_statistics

L_PATH = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])  # along x, then up y


def _line(steps: int, step: float) -> np.ndarray:
    """(steps + 1, 3) straight joint path at constant speed `step` per frame, not along an axis."""
    direction = np.array([1.0, -2.0, 2.0]) / 3.0
    return np.arange(steps + 1, dtype=np.float64)[:, None] * step * direction[None, :]


def test_distances_to_an_l_shaped_path() -> None:
    points = np.array(
        [
            [0.5, 0.5],  # inside the corner: 0.5 from both segments
            [0.5, -0.3],  # below the first segment
            [2.0, 0.5],  # right of the second segment
            [-1.0, 0.0],  # before the start
            [2.0, 2.0],  # past the end, diagonal
            [1.0, 0.0],  # the corner itself
            [0.25, 0.0],  # on the path
        ]
    )
    expected = np.array([0.5, 0.3, 1.0, 1.0, np.sqrt(2.0), 0.0, 0.0])
    found = distance_to_path(points, L_PATH)
    print(f"distances {np.round(found, 12)}; max error {np.abs(found - expected).max():.1e}")
    assert found.shape == (len(points),)
    np.testing.assert_allclose(found, expected, atol=1e-12)
    assert abs(path_deviation(points, L_PATH) - np.sqrt(2.0)) < 1e-12


def test_path_deviation_of_a_path_against_itself_is_zero() -> None:
    path = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.05], [0.25, 0.1, 0.05], [0.3, 0.3, 0.0]])
    between = 0.5 * (path[:-1] + path[1:])  # points on the segments, not on the vertices
    itself, on_segments = path_deviation(path, path), path_deviation(between, path)
    print(f"deviation of the path against itself {itself:.1e}, of segment midpoints {on_segments:.1e}")
    assert itself == 0.0
    assert on_segments < 1e-12
    assert path_deviation(np.zeros((0, 3)), path) == 0.0


def test_command_switched_two_rows_late_is_two_steps_off() -> None:
    step, edge, late = 0.01, 5, 2
    joints = _line(12, step)
    command = (np.arange(len(joints)) >= edge + late).astype(np.float64)
    error = keyframe_position_error(joints, command, False, joints, edge)
    on_time = keyframe_position_error(joints, (np.arange(len(joints)) >= edge).astype(np.float64), False, joints, edge)
    never = keyframe_position_error(joints, np.zeros(len(joints)), False, joints, edge)
    print(f"keyframe position error: {late} rows late {error / step:.6f} steps, on time {on_time}, never {never}")
    assert error is not None
    assert abs(error - late * step) < 1e-12
    assert on_time == 0.0
    assert never is None


def test_doubled_step_has_speed_ratio_two() -> None:
    step = 0.02
    path = _line(5, step)
    targets = np.concatenate([path[1:3], path[4:]])  # rows 1, 2, 4, 5: the step from 2 to 4 is doubled
    ratio = speed_ratio(targets, path[0], reference_step=step)
    steady = speed_ratio(path[1:], path[0], reference_step=step)
    print(f"speed ratio with one doubled step {ratio:.12f}, at constant speed {steady:.12f}")
    assert abs(ratio - 2.0) < 1e-12
    assert abs(steady - 1.0) < 1e-12


def test_step_statistics_ignore_repeated_frames() -> None:
    first = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 2.0]])  # steps 1, 0, 2
    second = np.array([[0.0, 0.0], [0.0, 1.0], [0.0, 1.0]])  # steps 1, 0
    median, p95 = step_statistics([first, second])
    print(f"non-zero steps [1, 2, 1]: median {median}, p95 {p95}")
    assert median == 1.0
    assert abs(p95 - float(np.percentile([1.0, 2.0, 1.0], 95))) < 1e-12
