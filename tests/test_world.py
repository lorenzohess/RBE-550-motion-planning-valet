"""World generation: seeded tetrominoes, parking layouts, start clearing."""

import random

import numpy as np
import pytest

from valet.vehicles import CAR, ROBOT
from valet.world import (CELLS, LAYOUTS, PARKING_CELLS, TETROMINOES, build_world, cells_under,
                         place_tetrominoes)


def test_twelve_free_tetromino_orientations():
    assert len(TETROMINOES) == 12
    assert all(len(shape) == 4 for shape in TETROMINOES)


def test_reproducible_and_seed_sensitive():
    a = build_world(7, 0.1, "5a", ROBOT.footprint(*ROBOT.start)).occupied
    b = build_world(7, 0.1, "5a", ROBOT.footprint(*ROBOT.start)).occupied
    c = build_world(8, 0.1, "5a", ROBOT.footprint(*ROBOT.start)).occupied
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


@pytest.mark.parametrize("rho", [0.0, 0.1, 0.3])
def test_density_is_a_floor(rho):
    for seed in range(50):
        occupied = place_tetrominoes(random.Random(seed), rho)
        assert occupied.mean() >= rho
        assert occupied.shape == (CELLS, CELLS)


class ScriptedRNG:
    """Always draws the vertical I piece; anchors come from a script."""

    def __init__(self, anchors):
        self.anchors = list(anchors)

    def choice(self, seq):
        return ((0, 0), (0, 1), (0, 2), (0, 3))

    def randrange(self, n):
        return self.anchors.pop(0)


def test_pieces_do_not_wrap():
    # First anchor (11, 11): the piece would run off the north edge. obstacleworld would wrap it
    # to row 0; here it is drawn again, at (0, 0).
    occupied = place_tetrominoes(ScriptedRNG([11, 11, 0, 0]), 4 / 144)
    assert {tuple(c) for c in np.argwhere(occupied)} == {(0, 0), (0, 1), (0, 2), (0, 3)}


@pytest.mark.parametrize("layout", ["5a", "5b"])
def test_parking_layout_overwrites_obstacles(layout):
    for seed in range(30):
        w = build_world(seed, 0.5, layout, ROBOT.footprint(*ROBOT.start))
        blocked = set(LAYOUTS[layout]["blocked"])
        for cell in PARKING_CELLS:
            assert w.occupied[cell] == (cell in blocked)


@pytest.mark.parametrize("vehicle", [ROBOT, CAR])
def test_start_cells_are_cleared(vehicle):
    footprint = vehicle.footprint(*vehicle.start)
    cells = {c for corners in footprint for c in cells_under(corners)}
    assert cells == {(0, 10), (0, 11)}      # the two coincident cells of Fig. 4
    for seed in range(30):
        w = build_world(seed, 0.5, vehicle.layout, footprint)
        assert not any(w.occupied[c] for c in cells)


def test_goal_boxes():
    assert LAYOUTS["5a"]["goal_box"] == (24.0, 30.0, 0.0, 3.0)
    assert LAYOUTS["5b"]["goal_box"] == (18.0, 30.0, 0.0, 3.0)


def test_goal_inset_shrinks_the_box():
    from valet.vehicles import CAR, ROBOT
    w = build_world(0, 0.1, "5a", ROBOT.footprint(*ROBOT.start), ROBOT.goal_inset)
    assert w.goal_box == pytest.approx((24.5, 29.5, 0.5, 2.5))
    w = build_world(0, 0.1, "5a", CAR.footprint(*CAR.start), CAR.goal_inset)
    x0, x1, y0, y1 = w.goal_box
    assert x1 - x0 >= CAR.body.length and y1 - y0 >= 2 * CAR.body.half_width
