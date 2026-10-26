"""The heuristic is a lower bound, and blocks only what the vehicle cannot reach."""

import math

import numpy as np

from valet import lattice
from valet.collision import CollisionChecker
from valet.heuristic import clearance_radius
from valet.search import LatticePlanner, Node
from valet.vehicles import CAR, ROBOT
from valet.world import build_world


def test_clearance_radius():
    assert math.isclose(clearance_radius(ROBOT), 0.285)
    assert math.isclose(clearance_radius(CAR), 0.9)


def test_lower_bound_on_the_robot_plan():
    cs = lattice.build_diff_drive(ROBOT, resolution=0.5)
    for seed in range(4):
        world = build_world(seed, 0.1, "5a", ROBOT.footprint(*ROBOT.start))
        p = LatticePlanner(world, ROBOT, cs, CollisionChecker(world, ROBOT.body))
        plan = p.plan()
        if plan.status != "found":
            continue
        # Every node on the path: heuristic <= cost still to go.
        costs = [p.edge_cost(e, plan.edges[i - 1] if i else None) for i, e in enumerate(plan.edges)]
        to_go = np.cumsum(costs[::-1])[::-1]
        for node, rest in zip(plan.nodes[:-1], to_go):
            assert p.heuristic(*node) <= rest + 1e-6


def test_rotation_term_and_unreachable_points():
    cs = lattice.build_diff_drive(ROBOT, resolution=0.5)
    world = build_world(0, 0.0, "5a", ROBOT.footprint(*ROBOT.start))
    p = LatticePlanner(world, ROBOT, cs, CollisionChecker(world, ROBOT.body))
    east, north = cs.nearest_bin(0.0), cs.nearest_bin(math.pi / 2)
    assert p.heuristic(54, 3, east) == 0.0                  # parked
    assert p.heuristic(54, 3, north) > 0.0                  # must still rotate
    assert math.isinf(p.heuristic(40, 3, east))             # inside the block west of the slot
    assert math.isinf(p.heuristic(0, 30, east))             # on the lot boundary
