"""Kinematic simulation: the time-stepped model lands on the planned nodes."""

import math

import numpy as np

from valet import lattice
from valet.search import Node
from valet.sim import kmh, simulate
from valet.vehicles import ROBOT


def drive(edges, start):
    return simulate(ROBOT, start, edges, kmh(10.0), dt=0.02)


def test_straights_and_rotations_land_on_nodes():
    cs = lattice.build_diff_drive(ROBOT, resolution=0.1)
    k = cs.nearest_bin(0.0)
    node = Node(100, 100, k)
    edges, nodes = [], [node]
    picks = ["straight", "rotate", "straight", "rotate", "rotate", "straight"]
    for kind in picks:
        batch = cs.batches[node.psi]
        edge = next(e for e in batch.edges if e.prim.kind == kind and not e.reverse
                    and (kind == "rotate" and e.prim.turn > 0 or kind == "straight"))
        edges.append(edge)
        node = Node(node.x + edge.dx, node.y + edge.dy, edge.end_bin)
        nodes.append(node)
    start = (nodes[0].x * 0.1, nodes[0].y * 0.1, cs.headings[nodes[0].psi].psi)
    traj = drive(edges, start)
    x, y, psi = traj.state[-1]
    assert math.isclose(x, node.x * 0.1, abs_tol=1e-9)
    assert math.isclose(y, node.y * 0.1, abs_tol=1e-9)
    assert math.isclose(float(lattice.wrap(psi - cs.headings[node.psi].psi)), 0.0, abs_tol=1e-9)


def test_reverse_moves_backwards_without_turning():
    cs = lattice.build_diff_drive(ROBOT, resolution=0.1)
    k = cs.nearest_bin(math.pi / 2)
    edge = next(e for e in cs.batches[k].edges if e.reverse)
    traj = drive([edge], (5.0, 5.0, math.pi / 2))
    x, y, psi = traj.state[-1]
    assert math.isclose(x, 5.0, abs_tol=1e-9)
    assert y < 5.0
    assert math.isclose(psi, math.pi / 2, abs_tol=1e-12)
    assert traj.reverse[1:].all()


def test_constant_speed():
    cs = lattice.build_diff_drive(ROBOT, resolution=0.1)
    k = cs.nearest_bin(0.0)
    edge = max((e for e in cs.batches[k].edges if e.prim.kind == "straight" and not e.reverse),
               key=lambda e: e.prim.length)
    traj = drive([edge], (5.0, 5.0, 0.0))
    speed = np.hypot(np.diff(traj.state[:, 0]), np.diff(traj.state[:, 1])) / np.diff(traj.t)
    assert np.allclose(speed, kmh(10.0))


def test_car_curves_land_on_nodes_forward_and_reverse():
    from valet.vehicles import CAR
    cs = lattice.build_car(CAR)
    k = cs.nearest_bin(0.0)
    for reverse in (False, True):
        edge = next(e for e in cs.batches[k].edges if e.prim.kind == "curve" and e.reverse == reverse)
        start = (10.0, 10.0, cs.headings[k].psi)
        traj = simulate(CAR, start, [edge], kmh(10.0), dt=0.01)
        x, y, psi = traj.state[-1]
        assert math.isclose(x, 10.0 + edge.dx * cs.resolution, abs_tol=1e-4)
        assert math.isclose(y, 10.0 + edge.dy * cs.resolution, abs_tol=1e-4)
        assert abs(float(lattice.wrap(psi - cs.headings[edge.end_bin].psi))) < 1e-6


def test_car_steering_stays_within_the_limit():
    from valet.sim import segments
    from valet.vehicles import CAR
    cs = lattice.build_car(CAR)
    edges = [e for b in cs.batches for e in b.edges if e.prim.kind == "curve"]
    for seg in segments(CAR, edges, kmh(20.0)):
        for tau in np.linspace(0.0, seg.duration, 11):
            assert abs(seg.control(tau)[1]) <= CAR.max_steer + 1e-9


def test_truck_trailer_matches_the_planner():
    # A rig on the bottom row west of the strip, trailer swung 20 deg, drives in. The time-stepped
    # model (RK4 on all four states) ends where the planner says, with the same hitch angle.
    from valet.collision import CollisionChecker
    from valet.search import TruckPlanner
    from valet.vehicles import TRUCK
    from valet.world import build_world
    cs = lattice.build(TRUCK)
    world = build_world(0, 0.0, "5b", TRUCK.footprint(*TRUCK.start), TRUCK.goal_inset)
    beta0 = math.radians(20.0)
    p = TruckPlanner(world, TRUCK, cs, CollisionChecker(world, TRUCK.body), start=(12.5, 1.5, 0.0),
                     beta0=beta0)
    plan = p.plan()
    assert plan.status == "found"
    x, y, psi, beta = p.rig_pose(plan.nodes[-1])
    traj = simulate(TRUCK, (12.5, 1.5, 0.0), plan.edges, kmh(10.0), beta0=beta0)
    sx, sy, spsi, spsi1 = traj.state[-1]
    assert math.isclose(sx, x, abs_tol=1e-4) and math.isclose(sy, y, abs_tol=1e-4)
    assert abs(float(lattice.wrap(spsi - spsi1 - beta))) < 1e-3
    x0, x1, y0, y1 = world.goal_box
    for corners in TRUCK.footprint(sx, sy, spsi, spsi - spsi1):
        assert corners[:, 0].min() >= x0 - 1e-6 and corners[:, 0].max() <= x1 + 1e-6
        assert corners[:, 1].min() >= y0 - 1e-6 and corners[:, 1].max() <= y1 + 1e-6
