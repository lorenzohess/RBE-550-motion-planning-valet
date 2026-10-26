"""Heading set and the robot's control set."""

import math

import numpy as np
import pytest

from valet import lattice
from valet.vehicles import ROBOT


@pytest.fixture(scope="module")
def cs():
    return lattice.build_diff_drive(ROBOT, resolution=0.1)


def test_heading_set_is_nav2_32_plus_four_near_east_and_west():
    headings = lattice.heading_set(32, ((8, 1),))
    degrees = sorted(round(math.degrees(h.psi), 3) for h in headings)
    assert len(headings) == 36
    for d in (0.0, 7.125, -7.125, 172.875, -172.875, -180.0, 14.036, 26.565, 45.0, 90.0):
        assert d in degrees
    nav2_16 = lattice.heading_set(16, ())
    assert {(h.gx, h.gy) for h in nav2_16} <= {(h.gx, h.gy) for h in headings}


def test_headings_are_reduced_grid_steps(cs):
    for h in cs.headings:
        assert math.gcd(h.gx, h.gy) == 1
        assert math.isclose(math.atan2(h.gy, h.gx), h.psi, abs_tol=1e-12) or \
            math.isclose(abs(h.psi), math.pi)


def test_every_heading_has_an_opposite(cs):
    for k in range(cs.n):
        j = cs.opposite(k)
        assert math.isclose(abs(lattice.wrap(cs.headings[j].psi - cs.headings[k].psi)), math.pi)


def test_robot_edges(cs):
    for k, batch in enumerate(cs.batches):
        h = cs.headings[k]
        rotations = [e for e in batch.edges if e.prim.kind == "rotate"]
        assert sorted(e.end_bin for e in rotations) == sorted([(k - 1) % cs.n, (k + 1) % cs.n])
        assert all((e.dx, e.dy) == (0, 0) and not e.reverse for e in rotations)
        for e in batch.edges:
            if e.prim.kind != "straight":
                continue
            assert e.end_bin == k                      # a straight keeps the heading
            m = e.dx / h.gx if h.gx else e.dy / h.gy
            sign = -1 if e.reverse else 1
            assert (e.dx, e.dy) == (round(sign * abs(m) * h.gx), round(sign * abs(m) * h.gy))
            # Body heading along a reverse edge is still the node's heading.
            assert np.allclose(lattice.wrap(e.samples[:, 2] - h.psi), 0.0)
        shortest = min(abs(e.prim.length) for e in batch.edges if e.prim.kind == "straight")
        assert math.isclose(shortest, math.hypot(h.gx, h.gy) * cs.resolution)


def test_samples_are_close_enough(cs):
    margin = 0.05
    for batch in cs.batches:
        for e in batch.edges:
            pts = np.vstack([[0.0, 0.0, e.samples[0, 2] - (e.prim.turn / len(e.samples) if e.prim.kind == "rotate" else 0.0)],
                             e.samples])
            if e.prim.kind == "straight":
                step = np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1])).max()
            else:
                step = ROBOT.body.max_radius() * np.abs(np.diff(np.unwrap(pts[:, 2]))).max()
            assert step <= 2 * margin + 1e-9


# --- car -----------------------------------------------------------------------------------------

from valet.vehicles import CAR  # noqa: E402


@pytest.fixture(scope="module")
def car_cs():
    return lattice.build_car(CAR)


def test_every_car_heading_turns_both_ways(car_cs):
    for k, batch in enumerate(car_cs.batches):
        curves = [e for e in batch.edges if e.prim.kind == "curve" and not e.reverse]
        assert any(e.prim.left_turn for e in curves), k
        assert any(not e.prim.left_turn for e in curves), k


def test_car_curves_respect_the_steering_limit_and_land_on_nodes(car_cs):
    kappa_max = 1.0 / CAR.min_turning_radius
    res = car_cs.resolution
    for k, batch in enumerate(car_cs.batches):
        for e in batch.edges:
            if e.prim.kind != "curve":
                continue
            sp = e.prim.spiral
            assert sp.max_abs_kappa() <= kappa_max * (1 + 1e-9)
            end = sp.poses(0.01)[-1]
            assert math.hypot(end[0] - e.dx * res, end[1] - e.dy * res) < 1e-4
            # end_bin is the body's heading at the end, forward or reverse.
            heading = car_cs.headings[e.end_bin].psi
            assert abs(float(lattice.wrap(e.samples[-1, 2] - heading))) < 1e-9


def test_car_reverse_edges_keep_the_body_heading_at_the_start(car_cs):
    for k, batch in enumerate(car_cs.batches):
        for e in batch.edges:
            if e.reverse and e.prim.kind == "curve":
                # The first sample is close to the start, so close to the node's heading.
                assert abs(float(lattice.wrap(e.samples[0, 2] - car_cs.headings[k].psi))) < 0.1
