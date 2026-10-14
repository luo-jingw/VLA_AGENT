"""path_curvature: 1/R on a circle at any traversal speed, 0 on a straight line, no spike where the path stops."""

from __future__ import annotations

import numpy as np

from vla_tools import path_curvature

RADIUS = 0.3
WINDOW_STEPS = 4.0


def _plane() -> tuple[np.ndarray, np.ndarray]:
    """Two orthonormal directions spanning a plane tilted against every joint axis."""
    e1 = np.array([1.0, 0.5, -0.3])
    e1 /= np.linalg.norm(e1)
    e2 = np.array([0.2, -0.1, 1.0])
    e2 -= e2.dot(e1) * e1
    return e1, e2 / np.linalg.norm(e2)


def _circle(angles: np.ndarray) -> np.ndarray:
    e1, e2 = _plane()
    centre = np.array([0.1, -0.2, 0.3])
    return centre + RADIUS * (np.cos(angles)[:, None] * e1 + np.sin(angles)[:, None] * e2)


def _arc_length(q: np.ndarray) -> np.ndarray:
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(q, axis=0), axis=1))])


def _window(q: np.ndarray) -> float:
    step = np.linalg.norm(np.diff(q, axis=0), axis=1)
    return WINDOW_STEPS * float(np.median(step[step > 0.0]))


def _interior(q: np.ndarray, window: float) -> np.ndarray:
    """Samples at least one window of arc length away from both ends."""
    arc = _arc_length(q)
    return (arc >= window) & (arc <= arc[-1] - window)


def _speed_profiles() -> dict[str, np.ndarray]:
    """Traversal parameter in [0, 1] per frame."""
    uniform = np.linspace(0.0, 1.0, 400)
    t = np.linspace(0.0, 1.0, 400)
    speed = 0.1 + 1.9 * (0.5 + 0.5 * np.sin(2.0 * np.pi * 4.0 * t))  # 0.1x to 2x, a 20x ratio
    varying = np.concatenate([[0.0], np.cumsum(speed)])
    return {"uniform": uniform, "varying": varying / varying[-1]}


def test_circle_gives_inverse_radius_at_any_speed() -> None:
    for name, u in _speed_profiles().items():
        q = _circle(1.5 * np.pi * u)
        window = _window(q)
        kappa = path_curvature.curvature(q, window)
        error = np.abs(kappa[_interior(q, window)] * RADIUS - 1.0)
        steps = np.linalg.norm(np.diff(q, axis=0), axis=1)
        print(f"{name}: steps {steps.min():.2e}..{steps.max():.2e}, kappa*R median {np.median(kappa * RADIUS):.4f}, "
              f"max relative error {error.max():.2e}")  # fmt: skip
        assert kappa.dtype == np.float64 and kappa.shape == (len(q),)
        assert error.max() <= 0.02


def test_straight_line_traversed_while_accelerating_has_no_curvature() -> None:
    direction = np.array([0.37, -0.81, 0.45])
    direction /= np.linalg.norm(direction)
    t = np.linspace(0.0, 1.0, 200)
    q = np.array([0.12, 0.5, -0.7]) + (0.6 * t**2)[:, None] * direction
    kappa = path_curvature.curvature(q, _window(q))
    print(f"accelerating straight line: max kappa {kappa.max():.2e}")
    assert kappa.max() < 1e-6


def test_stop_on_the_path_adds_no_curvature() -> None:
    angles = np.linspace(0.0, 1.5 * np.pi, 300)
    stopped = np.concatenate([angles[:150], np.full(40, angles[150]), angles[150:]])
    q = _circle(stopped)
    window = _window(q)
    kappa = path_curvature.curvature(q, window)
    at_stop = kappa[150:191] * RADIUS
    interior = kappa[_interior(q, window)] * RADIUS
    print(f"circle with a 40-sample stop: kappa*R at the stop {at_stop.min():.4f}..{at_stop.max():.4f}, "
          f"interior max {interior.max():.4f}")  # fmt: skip
    assert np.abs(at_stop - 1.0).max() <= 0.02
    assert interior.max() <= 1.02

    t = np.linspace(0.0, 1.0, 200)
    line = np.concatenate([t[:100], np.full(30, t[100]), t[100:]])
    q_line = np.array([0.12, 0.5, -0.7]) + (0.6 * line)[:, None] * np.array([0.6, -0.48, 0.64])
    kappa_line = path_curvature.curvature(q_line, _window(q_line))
    print(f"straight line with a 30-sample stop: max kappa {kappa_line.max():.2e}")
    assert kappa_line.max() < 1e-6
