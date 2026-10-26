"""The trailer: its motion along edges (closed forms) and its contact with the truck."""

import math

import numpy as np

from valet import trailer
from valet.collision import contact_angle, convex_overlap
from valet.vehicles import TRUCK, Body

D1 = TRUCK.trailer.hitch_to_axle


def test_straight_matches_the_closed_form():
    # tan(beta / 2) = tan(beta0 / 2) exp(-sigma s / d1) on a straight.
    beta0 = np.array([-2.0, -0.6, 0.1, 1.0, 2.3])
    ds, n = 0.04, 250
    s = ds * np.arange(1, n + 1)
    for sigma in (1.0, -1.0):
        beta = trailer.integrate(0.3, np.full(n, 0.3), ds, sigma, beta0, D1)
        exact = 2 * np.arctan(np.tan(beta0[:, None] / 2) * np.exp(-sigma * s / D1))
        assert np.allclose(beta, exact, atol=1e-9)


def test_forward_lines_up_and_reverse_folds():
    beta = trailer.integrate(0.0, np.zeros(100), 0.04, 1.0, [0.3], D1)[0]
    assert np.all(np.diff(np.abs(beta)) < 0)
    beta = trailer.integrate(0.0, np.zeros(100), 0.04, -1.0, [0.3], D1)[0]
    assert np.all(np.diff(np.abs(beta)) > 0)


def test_held_turn_settles_at_the_steady_state_angle():
    # Rill Eq. 9.40 with b = 0: sin(beta_ss) = d1 / rho.
    rho = TRUCK.min_turning_radius
    ds = 0.05
    s = ds * np.arange(1, 2001)               # 100 m around the circle
    for sign in (1.0, -1.0):
        beta = trailer.integrate(0.0, sign * s / rho, ds, 1.0, [0.0], D1)[0]
        assert math.isclose(beta[-1], sign * math.asin(D1 / rho), abs_tol=1e-5)


def test_table_lookup_matches_direct_integration():
    from valet import lattice
    cs = lattice.build(TRUCK)
    batch = cs.batches[cs.nearest_bin(0.0)]
    for beta0 in (-1.1, -0.2, 0.0, 0.37, 1.9):
        looked_up = trailer.lookup(batch.beta, beta0)
        for i, e in enumerate(batch.edges):
            ds = e.prim.length / len(e.samples)
            direct = trailer.integrate(0.0, e.samples[:, 2], ds, -1.0 if e.reverse else 1.0, [beta0], D1)[0]
            got = looked_up[batch.starts[i]:batch.starts[i] + len(e.samples)]
            assert np.allclose(got, direct, atol=2e-3)


def test_contact_angle_matches_a_brute_force_sweep():
    margin = 0.05
    grow = lambda b: Body(b.rear + margin, b.front + margin, b.half_width + margin)  # noqa: E731
    truck, trl = grow(TRUCK.body), grow(TRUCK.trailer.body)
    corners = truck.corners(0.0, 0.0, 0.0)
    first = None
    for deg in np.arange(0.0, 180.0, 0.01):
        psi1 = -math.radians(deg)
        if convex_overlap(corners, trl.corners(*TRUCK.trailer.axle(0.0, 0.0, psi1), psi1)):
            first = deg
            break
    assert first is not None
    assert abs(math.degrees(contact_angle(TRUCK, margin)) - first) <= 0.01
    # The true rectangles first touch at about 140 deg.
    assert 139.0 < math.degrees(contact_angle(TRUCK, 0.0)) < 141.0
