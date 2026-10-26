"""Collision checking: separating axis test, the cell shortcut, edges in a batch."""

import math
import random

import numpy as np
import pytest

from valet import lattice
from valet.collision import CollisionChecker, convex_overlap, polygon_overlaps_box
from valet.vehicles import CAR, ROBOT, Body
from valet.world import CELLS, World


def make_world(cells=()):
    occupied = np.zeros((CELLS, CELLS), dtype=bool)
    for c in cells:
        occupied[c] = True
    return World(occupied=occupied, layout="5a", goal_box=(24.0, 30.0, 0.0, 3.0), seed=0, rho=0.0)


SQUARE = (3.0, 6.0, 3.0, 6.0)   # cell (1, 1)
UNIT = Body(rear=0.5, front=0.5, half_width=0.5)


def test_far_apart():
    assert not polygon_overlaps_box(UNIT.corners(10.0, 10.0, 0.3), SQUARE)


def test_overlapping():
    assert polygon_overlaps_box(UNIT.corners(4.5, 4.5, 0.3), SQUARE)


def test_touching_is_not_overlapping():
    assert not polygon_overlaps_box(UNIT.corners(6.5, 4.5, 0.0), SQUARE)


def test_rotated_rectangle_separated_only_by_its_own_axis():
    # A 45 deg diamond off the square's corner: the axis-aligned extents overlap, so only the
    # rectangle's own axis separates them.
    # Separated when the corner (6, 6) is farther than 0.5 m (half a side) from the centre.
    corners = UNIT.corners(6.55, 6.55, math.pi / 4)
    assert corners[:, 0].min() < 6.0 and corners[:, 1].min() < 6.0
    assert not polygon_overlaps_box(corners, SQUARE)
    checker = CollisionChecker(make_world([(1, 1)]), UNIT, margin=0.0)
    assert checker.pose_free(6.55, 6.55, math.pi / 4)
    assert not checker.pose_free(6.3, 6.3, math.pi / 4)


def test_convex_overlap_is_symmetric():
    a = UNIT.corners(0.0, 0.0, 0.2)
    b = UNIT.corners(0.9, 0.3, 1.1)
    assert convex_overlap(a, b) == convex_overlap(b, a)


def test_leaving_the_lot_collides():
    checker = CollisionChecker(make_world(), ROBOT.body)
    assert not checker.pose_free(0.2, 18.0, 0.0)
    assert checker.pose_free(1.0, 18.0, 0.0)


@pytest.mark.parametrize("body", [ROBOT.body, CAR.body])
def test_vectorized_matches_one_by_one(body):
    rng = random.Random(3)
    cells = [(rng.randrange(CELLS), rng.randrange(CELLS)) for _ in range(25)]
    world = make_world(cells)
    checker = CollisionChecker(world, body, margin=0.0)
    xs = np.array([rng.uniform(-1, 37) for _ in range(2000)])
    ys = np.array([rng.uniform(-1, 37) for _ in range(2000)])
    psis = np.array([rng.uniform(-math.pi, math.pi) for _ in range(2000)])
    fast = checker.hits(xs, ys, psis)
    for x, y, psi, hit in zip(xs, ys, psis, fast):
        corners = body.corners(x, y, psi)
        out = corners.min() < 0.0 or corners.max() > 36.0
        slow = out or any(polygon_overlaps_box(corners, world.cell_box(*c)) for c in set(cells))
        assert hit == slow


def test_batch_shortcut_matches_full_check():
    cs = lattice.build_diff_drive(ROBOT, resolution=0.1)
    world = make_world([(5, 5), (6, 5), (2, 8)])
    checker = CollisionChecker(world, ROBOT.body)
    rng = random.Random(1)
    for _ in range(300):
        x = round(rng.uniform(0, 36), 1)
        y = round(rng.uniform(0, 36), 1)
        batch = cs.batches[rng.randrange(cs.n)]
        fast = checker.first_hits(x, y, batch)
        hit = checker.hits(x + batch.x, y + batch.y, batch.psi)
        for i, start in enumerate(batch.starts):
            mine = hit[batch.owner == i]
            expected = int(np.argmax(mine)) if mine.any() else -1
            assert fast[i] == expected


def test_margin_grows_the_rectangle():
    world = make_world([(1, 1)])
    tight = CollisionChecker(world, UNIT, margin=0.0)
    grown = CollisionChecker(world, UNIT, margin=0.05)
    assert tight.pose_free(2.47, 4.5, 0.0)
    assert not grown.pose_free(2.47, 4.5, 0.0)
