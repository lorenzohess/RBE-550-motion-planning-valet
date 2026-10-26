"""Lattice A* on the robot. A 0.5 m lattice keeps these fast; the logic is the same at 0.1 m."""

import numpy as np
import pytest

from valet import lattice
from valet.collision import CollisionChecker
from valet.search import LatticePlanner, Node, SearchParams
from valet.vehicles import ROBOT
from valet.world import CELLS, LAYOUTS, World, build_world


@pytest.fixture(scope="module")
def cs():
    return lattice.build_diff_drive(ROBOT, resolution=0.5)


def planner(world, cs, **kw):
    return LatticePlanner(world, ROBOT, cs, CollisionChecker(world, ROBOT.body), SearchParams(**kw))


def test_empty_lot_parks(cs):
    world = build_world(0, 0.0, "5a", ROBOT.footprint(*ROBOT.start))
    p = planner(world, cs)
    plan = p.plan()
    assert plan.status == "found"
    assert p.parked(plan.nodes[-1])
    assert plan.nodes[0] == p.start
    for a, b, e in zip(plan.nodes, plan.nodes[1:], plan.edges):
        assert (a.x + e.dx, a.y + e.dy, e.end_bin) == tuple(b)


def test_walled_in_start_has_no_path(cs):
    world = build_world(0, 0.0, "5a", ROBOT.footprint(*ROBOT.start))
    world.occupied[1, 10] = world.occupied[1, 11] = world.occupied[0, 9] = True
    plan = planner(world, cs).plan()
    assert plan.status == "no_path"
    # The heuristic's grid search already finds the goal box cut off, so nothing is expanded.
    assert plan.expansions == 0


def test_expansion_cap(cs):
    world = build_world(0, 0.0, "5a", ROBOT.footprint(*ROBOT.start))
    assert planner(world, cs, max_expansions=10).plan().status == "expansion_cap"


def test_goal_box_and_heading_tolerance(cs):
    world = build_world(0, 0.0, "5a", ROBOT.footprint(*ROBOT.start))
    p = planner(world, cs)
    east = cs.nearest_bin(0.0)
    north = cs.nearest_bin(np.pi / 2)
    near_east = cs.nearest_bin(np.radians(7.125))
    assert p.parked(Node(54, 3, east))           # (27, 1.5) facing east
    assert p.parked(Node(54, 3, near_east))      # within 10 deg
    assert not p.parked(Node(54, 3, north))      # wrong heading
    assert not p.parked(Node(48, 3, east))       # body sticks out of the box at x = 24
    assert p.heuristic_cost_estimate(Node(54, 3, east), None) == 0.0
    # From y = 5 the centre must get within half a body width (0.285 m) of the box's top edge
    # (y = 3), less the 1 cm pad of parked_reference_box.
    assert p.heuristic_cost_estimate(Node(54, 10, east), None) == pytest.approx(5.0 - (3.0 - 0.285) - 0.01)


def test_path_is_collision_free(cs):
    world = build_world(3, 0.1, "5a", ROBOT.footprint(*ROBOT.start))
    p = planner(world, cs)
    plan = p.plan()
    assert plan.status == "found"
    checker = CollisionChecker(world, ROBOT.body)
    for node, edge in zip(plan.nodes, plan.edges):
        x, y, _ = p.pose(node)
        assert not checker.hits(x + edge.samples[:, 0], y + edge.samples[:, 1], edge.samples[:, 2]).any()


def test_change_penalty_only_between_opposite_turns():
    from valet.vehicles import CAR
    cs = lattice.build_car(CAR)
    world = build_world(0, 0.0, "5a", CAR.footprint(*CAR.start))
    p = LatticePlanner(world, CAR, cs, CollisionChecker(world, CAR.body))
    curves = [e for e in cs.batches[0].edges if e.prim.kind == "curve" and not e.reverse]
    left = next(e for e in curves if e.prim.left_turn)
    right = next(e for e in curves if not e.prim.left_turn)
    straight = next(e for e in cs.batches[0].edges if e.prim.kind == "straight" and not e.reverse)
    base = left.prim.length * p.params.non_straight_penalty
    assert p.edge_cost(left) == pytest.approx(base)
    assert p.edge_cost(left, left) == pytest.approx(base)
    assert p.edge_cost(left, straight) == pytest.approx(base)
    assert p.edge_cost(left, right) == pytest.approx(
        left.prim.length * (p.params.non_straight_penalty + p.params.change_penalty))
