"""Coarse-to-fine planning: a 1 m lattice to a handoff pose, then the fine lattice."""

import math

import numpy as np
import pytest

from valet import lattice, parking, planning
from valet.collision import CollisionChecker
from valet.search import LatticePlanner, SearchParams
from valet.sim import kmh, simulate
from valet.vehicles import CAR, ROBOT
from valet.world import build_world


def test_start_off_the_grid_is_a_node():
    cs = lattice.build_diff_drive(ROBOT, resolution=1.0, n_base=16, extra=())
    world = build_world(0, 0.0, "5a", ROBOT.footprint(*ROBOT.start))
    p = LatticePlanner(world, ROBOT, cs, CollisionChecker(world, ROBOT.body), start=(1.5, 34.2, -math.pi / 2))
    assert np.allclose(p.pose(p.start), (1.5, 34.2, -math.pi / 2))
    # Every coarse node is on the 0.1 m grid.
    x, y, _ = p.pose(p.start._replace(x=p.start.x + 7, y=p.start.y - 3))
    assert math.isclose(x * 10, round(x * 10), abs_tol=1e-9) and math.isclose(y * 10, round(y * 10), abs_tol=1e-9)


def test_target_row_tolerance():
    cs = lattice.build_diff_drive(ROBOT, resolution=1.0, n_base=16, extra=())
    world = build_world(0, 0.0, "5a", ROBOT.footprint(*ROBOT.start))
    p = LatticePlanner(world, ROBOT, cs, CollisionChecker(world, ROBOT.body), target=(20.5, 4.0, 0.0),
                       target_tolerance=3.0)
    t = p.target
    assert p.is_goal_reached(t, None)
    assert p.is_goal_reached(t._replace(x=t.x - 3), None)          # behind, on the row
    assert p.is_goal_reached(t._replace(x=t.x + 2), None)          # ahead, on the row
    assert not p.is_goal_reached(t._replace(x=t.x - 4), None)      # too far
    assert not p.is_goal_reached(t._replace(y=t.y + 1), None)      # off the row
    assert not p.is_goal_reached(t._replace(psi=(t.psi + 1) % cs.n), None)   # wrong heading


@pytest.mark.parametrize("seed", [0, 2])
def test_robot_coarse_then_fine_parks(seed):
    fine = lattice.build_diff_drive(ROBOT)
    coarse = lattice.build_diff_drive(ROBOT, resolution=1.0, n_base=16, extra=())
    world = build_world(seed, 0.1, "5a", ROBOT.footprint(*ROBOT.start))
    checker = CollisionChecker(world, ROBOT.body)
    r = planning.plan(world, ROBOT, checker, fine, coarse, SearchParams())
    assert r.status == "found"
    assert [name for name, _ in r.stages] == ["coarse", "fine"]
    traj = simulate(ROBOT, ROBOT.start, r.edges, kmh(10.0))
    corners = ROBOT.body.corners(*traj.state[-1])
    x0, x1, y0, y1 = world.goal_box
    assert corners[:, 0].min() >= x0 - 1e-6 and corners[:, 0].max() <= x1 + 1e-6
    assert corners[:, 1].min() >= y0 - 1e-6 and corners[:, 1].max() <= y1 + 1e-6


def test_car_coarse_then_fine_reaches_the_entry_and_parks():
    fine = lattice.build_car(CAR)
    coarse = lattice.build(CAR, resolution=1.0, n_base=16, extra=())
    world = build_world(0, 0.1, "5a", CAR.footprint(*CAR.start))
    checker = CollisionChecker(world, CAR.body)
    m = parking.find_maneuver(checker, CAR, fine, world.goal_box)
    r = planning.plan(world, CAR, checker, fine, coarse, SearchParams(), maneuver=m)
    assert r.status == "found" and [name for name, _ in r.stages] == ["coarse", "fine"]
    traj = simulate(CAR, CAR.start, r.edges + m.edges, kmh(10.0), dt=0.005)
    assert np.allclose(traj.state[-1, :2], m.parked[:2], atol=2e-3)
    assert not checker.hits(traj.state[:, 0], traj.state[:, 1], traj.state[:, 2]).any()


def test_fallback_is_weighted_when_coarse_fails():
    from valet.lattice import ControlSet, _assemble, _rotation, heading_set
    fine = lattice.build_diff_drive(ROBOT, resolution=0.5)
    # A coarse lattice that can only turn on the spot never reaches the handoff.
    headings = heading_set(16, ())
    stuck = _assemble(ControlSet(1.0, headings),
                      [[_rotation(headings, k, +1), _rotation(headings, k, -1)] for k in range(16)],
                      ROBOT.body, 0.05)
    world = build_world(0, 0.0, "5a", ROBOT.footprint(*ROBOT.start), ROBOT.goal_inset)
    r = planning.plan(world, ROBOT, CollisionChecker(world, ROBOT.body), fine, stuck, SearchParams())
    assert r.status == "found"
    assert [name for name, _ in r.stages] == ["coarse", f"fine only, weighted x{planning.FALLBACK_WEIGHT:g}"]


def test_truck_parks_on_an_empty_lot():
    # One weighted fine search to a straight shot; the simulated rig ends inside the goal box.
    from valet.vehicles import TRUCK
    cs = lattice.build(TRUCK)
    world = build_world(0, 0.0, "5b", TRUCK.footprint(*TRUCK.start), TRUCK.goal_inset)
    checker = CollisionChecker(world, TRUCK.body)
    result = planning.plan(world, TRUCK, checker, cs, None, SearchParams(max_expansions=50_000))
    assert result.status == "found"
    assert result.edges[-1].prim.kind == "straight" and not result.edges[-1].reverse
    traj = simulate(TRUCK, TRUCK.start, result.edges, kmh(30.0))
    x, y, psi, psi1 = traj.state[-1]
    x0, x1, y0, y1 = world.goal_box
    for corners in TRUCK.footprint(x, y, psi, psi - psi1):
        assert corners[:, 0].min() >= x0 - 1e-3 and corners[:, 0].max() <= x1 + 1e-3
        assert corners[:, 1].min() >= y0 - 1e-3 and corners[:, 1].max() <= y1 + 1e-3
