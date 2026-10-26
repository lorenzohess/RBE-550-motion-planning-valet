"""The car's parking manoeuvre: unpark with full-lock shuffles, reverse it, join it to the lattice."""

import math

import numpy as np
import pytest

from valet import lattice, parking
from valet.collision import CollisionChecker
from valet.sim import kmh, simulate
from valet.vehicles import CAR
from valet.world import build_world


@pytest.fixture(scope="module")
def cs():
    return lattice.build_car(CAR)


def parked_ok(world, pose):
    x0, x1, y0, y1 = world.goal_box
    c = CAR.body.corners(*pose)
    heading_ok = min(abs(math.remainder(pose[2] - h, 2 * math.pi)) for h in CAR.goal_headings) <= CAR.heading_tolerance
    return c[:, 0].min() >= x0 and c[:, 0].max() <= x1 and c[:, 1].min() >= y0 and c[:, 1].max() <= y1 and heading_ok


@pytest.mark.parametrize("seed, rho, facing", [(0, 0.0, 0.0), (1, 0.1, math.pi)])
def test_manoeuvre_parks_without_collision(cs, seed, rho, facing):
    world = build_world(seed, rho, "5a", CAR.footprint(*CAR.start), CAR.goal_inset)
    checker = CollisionChecker(world, CAR.body)
    m = parking.find_maneuver(checker, CAR, cs, world.goal_box)
    assert m is not None
    assert parked_ok(world, m.parked)
    assert abs(math.remainder(m.parked[2] - facing, 2 * math.pi)) < 1e-9
    res = cs.resolution
    entry = (m.entry.x * res, m.entry.y * res, cs.headings[m.entry.psi].psi)
    assert checker.pose_free(*entry)
    traj = simulate(CAR, entry, m.edges, kmh(10.0), dt=0.005)
    # The time-stepped car follows the manoeuvre into the parked pose ...
    assert np.allclose(traj.state[-1, :2], m.parked[:2], atol=1e-3)
    assert abs(math.remainder(traj.state[-1, 2] - m.parked[2], 2 * math.pi)) < 1e-3
    # ... without touching anything on the way (checked with the grown rectangle).
    assert not checker.hits(traj.state[:, 0], traj.state[:, 1], traj.state[:, 2]).any()


def test_shuffle_uses_full_lock_and_alternates_gear(cs):
    world = build_world(0, 0.0, "5a", CAR.footprint(*CAR.start), CAR.goal_inset)
    checker = CollisionChecker(world, CAR.body)
    m = parking.find_maneuver(checker, CAR, cs, world.goal_box)
    kmax = CAR.max_curvature
    shuffle = m.edges[1:]
    for edge in shuffle:
        assert math.isclose(abs(edge.prim.spiral.k0), kmax)
    gears = [e.reverse for e in shuffle]
    # Backs in first, and the gear alternates through the shuffle.
    assert gears[0] is True
    connector = m.edges[0]
    assert not connector.reverse and connector.prim.spiral.max_abs_kappa() <= kmax


def test_steering_at_most_the_limit(cs):
    from valet.sim import segments
    world = build_world(0, 0.0, "5a", CAR.footprint(*CAR.start), CAR.goal_inset)
    m = parking.find_maneuver(CollisionChecker(world, CAR.body), CAR, cs, world.goal_box)
    for seg in segments(CAR, m.edges, kmh(10.0)):
        for tau in np.linspace(0.0, seg.duration, 5):
            assert abs(seg.control(tau)[1]) <= CAR.max_steer + 1e-9
