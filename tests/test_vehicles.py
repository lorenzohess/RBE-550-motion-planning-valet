"""Vehicle geometry and the turning radius (model 1a)."""

import math

import numpy as np

from valet.vehicles import CAR, ROBOT, Body


def test_model_1a_turning_radius():
    # Inside front wheel at 30 deg; the rear axle's centre turns on L cot(30) + B / 2.
    expected = 2.8 / math.tan(math.radians(30.0)) + 0.5 * 60.0 * 0.0254
    assert math.isclose(CAR.min_turning_radius, expected)
    assert math.isclose(CAR.max_curvature, 1.0 / expected)
    # Wider than the bicycle form with the same angle.
    assert CAR.min_turning_radius > 2.8 / math.tan(math.radians(30.0))


def test_body_corners():
    b = Body(rear=1.0, front=3.0, half_width=0.5)
    c = b.corners(10.0, 5.0, 0.0)
    assert np.allclose(c, [[9.0, 4.5], [13.0, 4.5], [13.0, 5.5], [9.0, 5.5]])
    c = b.corners(0.0, 0.0, math.pi / 2)
    assert np.allclose(c[1], [0.5, 3.0])          # front right, facing north


def test_robot_turns_in_place():
    assert ROBOT.min_turning_radius == 0.0
