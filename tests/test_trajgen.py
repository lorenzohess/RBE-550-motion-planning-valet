"""Cubic curvature spirals with zero curvature at both ends."""

import math

import numpy as np
import pytest

from valet.trajgen import solve

CASES = [
    (2.0, 0.0, 0.0, 0.0),
    (3.0, 1.0, 0.0, math.atan2(1, 2)),
    (4.0, 2.0, 0.0, math.pi / 4),
    (3.0, 3.0, 0.0, math.pi / 2),
    (2.5, 0.3, 0.0, math.radians(7.125)),
    (0.4, 5.2, math.radians(90), math.radians(75.96)),
    (-3.0, -1.2, math.radians(-153.43), math.radians(-135.0)),
]


@pytest.mark.parametrize("x, y, psi0, psi1", CASES)
def test_lands_on_the_target_with_straight_wheels(x, y, psi0, psi1):
    sp = solve(x, y, psi0, psi1)
    assert sp is not None
    poses = sp.poses(0.01)
    assert math.hypot(poses[-1, 0] - x, poses[-1, 1] - y) < 1e-5
    assert abs(math.remainder(poses[-1, 2] - psi1, 2 * math.pi)) < 1e-9
    assert abs(float(sp.kappa(0.0))) < 1e-12
    assert abs(float(sp.kappa(sp.length))) < 1e-9
    # Path length agrees with the sampled poses.
    assert math.isclose(np.hypot(*np.diff(poses[:, :2], axis=0).T).sum(), sp.length, rel_tol=1e-4)


def test_poses_are_consistent_with_the_curvature():
    sp = solve(4.0, 2.0, 0.0, math.pi / 4)
    poses = sp.poses(0.001)
    heading_from_path = np.arctan2(np.diff(poses[:, 1]), np.diff(poses[:, 0]))
    mid = 0.5 * (poses[1:, 2] + poses[:-1, 2])
    assert np.allclose(heading_from_path, mid, atol=1e-5)


def test_degenerate_target():
    assert solve(0.0, 0.0, 0.0, 1.0) is None
