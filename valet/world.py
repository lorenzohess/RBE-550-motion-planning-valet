"""The parking lot: a 12 x 12 grid of 3 m cells with tetromino obstacles, a parking area and a start.

Coordinates: x east, y north, metres, origin at the SW corner of the lot. Cells are indexed
(cx, cy) the same way, with cy = 0 the southernmost row, so `occupied[cx, cy]` covers
x in [3 cx, 3 cx + 3], y in [3 cy, 3 cy + 3]. North is drawn at the top of the window.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np

CELLS = 12
CELL_M = 3.0
SIZE_M = CELLS * CELL_M  # 36 m

# Free tetrominoes (assignment Fig. 3): the shapes of the obstacleworld library written for
# assignment 1. There is no O piece, and a free tetromino counts a reflection as the same piece.
L_TET = ((0, 0), (1, 0), (1, -1), (1, -2))
I_TET = ((0, 0), (0, 1), (0, 2), (0, 3))
S_TET = ((0, 0), (0, -1), (1, -1), (1, -2))
T_TET = ((0, 0), (1, 0), (1, 1), (1, -1))


def _normalize(cells):
    min_x = min(c[0] for c in cells)
    min_y = min(c[1] for c in cells)
    return tuple(sorted((c[0] - min_x, c[1] - min_y) for c in cells))


def _rotations(cells):
    """Distinct 90 degree rotations of a shape, each normalized to start at (0, 0)."""
    shapes = []
    current = list(cells)
    for _ in range(4):
        current = [(y, -x) for x, y in current]
        shape = _normalize(current)
        if shape not in shapes:
            shapes.append(shape)
    return shapes


# 12 orientations: L 4, I 2, S 2, T 4.
TETROMINOES = [shape for tet in (L_TET, I_TET, S_TET, T_TET) for shape in _rotations(tet)]

# Southeast 6 x 2 parking area (assignment Fig. 5): columns 6-11, the two southern rows.
PARKING_CELLS = [(cx, cy) for cx in range(6, 12) for cy in range(2)]
LAYOUTS = {
    # Fig. 5a, robot and car: blocks on both ends of a two-cell slot.
    "5a": {"blocked": [(6, 0), (7, 0), (10, 0), (11, 0)], "goal_box": (24.0, 30.0, 0.0, 3.0)},
    # Fig. 5b, truck and trailer: the bottom row is open to the west, blocked at the east end.
    "5b": {"blocked": [(10, 0), (11, 0)], "goal_box": (18.0, 30.0, 0.0, 3.0)},
}


@dataclass
class World:
    occupied: np.ndarray            # bool (CELLS, CELLS), indexed [cx, cy]
    layout: str                     # "5a" or "5b"
    goal_box: tuple[float, float, float, float]   # x_min, x_max, y_min, y_max
    seed: int
    rho: float                      # requested obstacle density

    def cell_box(self, cx: int, cy: int) -> tuple[float, float, float, float]:
        return (cx * CELL_M, (cx + 1) * CELL_M, cy * CELL_M, (cy + 1) * CELL_M)


def place_tetrominoes(rng: random.Random, rho: float) -> np.ndarray:
    """Place random tetrominoes until at least a fraction rho of the cells is occupied.

    Unlike obstacleworld, pieces do not wrap around the edges: a placement that would leave the
    lot is drawn again. Pieces may overlap each other, so coverage is measured, not counted.
    """
    if not 0.0 <= rho <= 1.0:
        raise ValueError("rho must be between 0 and 1")
    occupied = np.zeros((CELLS, CELLS), dtype=bool)
    while occupied.mean() < rho:
        shape = rng.choice(TETROMINOES)
        cx0 = rng.randrange(CELLS)
        cy0 = rng.randrange(CELLS)
        cells = [(cx0 + dx, cy0 + dy) for dx, dy in shape]
        if all(0 <= cx < CELLS and 0 <= cy < CELLS for cx, cy in cells):
            for cx, cy in cells:
                occupied[cx, cy] = True
    return occupied


def cells_under(corners: np.ndarray) -> list[tuple[int, int]]:
    """Cells whose square overlaps the convex polygon `corners` (k, 2). Used to clear the start."""
    from .collision import polygon_overlaps_box  # local import: collision imports nothing from here

    x_lo, y_lo = corners.min(axis=0)
    x_hi, y_hi = corners.max(axis=0)
    cells = []
    for cx in range(max(0, math.floor(x_lo / CELL_M)), min(CELLS, math.floor(x_hi / CELL_M) + 1)):
        for cy in range(max(0, math.floor(y_lo / CELL_M)), min(CELLS, math.floor(y_hi / CELL_M) + 1)):
            box = (cx * CELL_M, (cx + 1) * CELL_M, cy * CELL_M, (cy + 1) * CELL_M)
            if polygon_overlaps_box(corners, box):
                cells.append((cx, cy))
    return cells


def build_world(seed: int, rho: float, layout: str, start_footprint: list[np.ndarray],
                goal_inset: tuple[float, float] = (0.0, 0.0)) -> World:
    """Tetrominoes, then the parking area overwritten (Fig. 5), then the start cells cleared.

    start_footprint: the corners of every body of the vehicle at its start pose. Every cell they
    touch is cleared (two cells for the robot and the car, as the assignment asks).
    goal_inset: the goal box is the parking slot shrunk by this much at each end (along x) and at
    each side (along y).
    """
    rng = random.Random(seed)
    occupied = place_tetrominoes(rng, rho)
    for cx, cy in PARKING_CELLS:
        occupied[cx, cy] = False
    for cx, cy in LAYOUTS[layout]["blocked"]:
        occupied[cx, cy] = True
    for corners in start_footprint:
        for cx, cy in cells_under(corners):
            occupied[cx, cy] = False
    x0, x1, y0, y1 = LAYOUTS[layout]["goal_box"]
    along, across = goal_inset
    return World(occupied=occupied, layout=layout,
                 goal_box=(x0 + along, x1 - along, y0 + across, y1 - across), seed=seed, rho=rho)
