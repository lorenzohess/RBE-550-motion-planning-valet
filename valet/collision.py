"""Collision checking: vehicle rectangles against obstacle cells and the lot boundary.

The exact test is the separating axis theorem. Two convex shapes do not overlap if and only if
there is a direction along which their projections (their shadows on a line) do not overlap. For a
rectangle and an axis-aligned square, four directions are enough: the world x and y axes (the
square's edge directions) and the rectangle's own forward and sideways axes.

Shortcut: a rectangle can only touch the cells under its axis-aligned extent. Only occupied cells
in that range are tested, so poses in open space skip the exact test.

Along a motion the rectangle is checked at sample poses (lattice.py chooses them so no point of the
body moves more than 2 * margin between two samples). Each rectangle is grown by `margin` on every
side, so a corner cannot slip into a cell between two checked poses.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .vehicles import Body, Vehicle
from .world import CELL_M, CELLS, SIZE_M, World


def polygon_overlaps_box(corners: np.ndarray, box: tuple[float, float, float, float]) -> bool:
    """Separating axis test, convex polygon (k, 2) vs axis-aligned box (x_min, x_max, y_min, y_max).
    Touching is not overlapping."""
    x0, x1, y0, y1 = box
    if (corners[:, 0].max() <= x0 or corners[:, 0].min() >= x1
            or corners[:, 1].max() <= y0 or corners[:, 1].min() >= y1):
        return False
    square = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
    return convex_overlap(corners, square)


def convex_overlap(p: np.ndarray, q: np.ndarray) -> bool:
    """Separating axis test for two convex polygons (k, 2), corners in order. Touching is not
    overlapping. Candidate axes are the edge normals of both polygons."""
    for poly in (p, q):
        edges = np.roll(poly, -1, axis=0) - poly
        for ex, ey in edges:
            axis = np.array([-ey, ex])
            pp, qq = p @ axis, q @ axis
            if pp.max() <= qq.min() or qq.max() <= pp.min():
                return False
    return True


def contact_angle(vehicle: Vehicle, margin: float = 0.05, step: float = math.radians(0.5)) -> float:
    """Smallest hitch angle |beta| at which the trailer touches the truck, both rectangles grown by
    the margin (separating axis test). Scanned in steps, then bisected to 1e-6 rad. The rig is
    symmetric, so -beta touches at the same angle."""
    grow = lambda b: Body(b.rear + margin, b.front + margin, b.half_width + margin)  # noqa: E731
    truck, trailer = grow(vehicle.body), grow(vehicle.trailer.body)
    corners = truck.corners(0.0, 0.0, 0.0)

    def touches(beta):
        psi1 = -beta
        return convex_overlap(corners, trailer.corners(*vehicle.trailer.axle(0.0, 0.0, psi1), psi1))

    lo = 0.0
    while lo + step <= math.pi and not touches(lo + step):
        lo += step
    hi = min(lo + step, math.pi)
    while hi - lo > 1e-6:
        mid = 0.5 * (lo + hi)
        lo, hi = (lo, mid) if touches(mid) else (mid, hi)
    return hi


@dataclass
class Hit:
    """Where a motion was found to collide, for display."""
    sample: int            # index into the motion's samples
    corners: np.ndarray    # (4, 2) the grown rectangle at that sample
    cells: list            # occupied cells it overlaps; empty if it left the lot


class CollisionChecker:
    def __init__(self, world: World, body: Body, margin: float = 0.05):
        self.world = world
        self.margin = margin
        self.body = Body(rear=body.rear + margin, front=body.front + margin,
                         half_width=body.half_width + margin)
        self.occupied = world.occupied
        cx, cy = np.nonzero(self.occupied)
        self._cells = np.stack([cx, cy], axis=1)
        self._extents: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    # --- many poses at once ------------------------------------------------------------------

    def hits(self, x: np.ndarray, y: np.ndarray, psi: np.ndarray) -> np.ndarray:
        """Boolean (S,): does the grown rectangle at each pose overlap a cell or leave the lot?"""
        # Axis-aligned extent of each rectangle: centre +- (a |cos psi| + w |sin psi|,
        # a |sin psi| + w |cos psi|), a and w its half length and half width.
        b = self.body
        c, s = np.cos(psi), np.sin(psi)
        shift = 0.5 * (b.front - b.rear)
        a = 0.5 * (b.front + b.rear)
        ac, as_ = np.abs(c), np.abs(s)
        hx = a * ac + b.half_width * as_
        hy = a * as_ + b.half_width * ac
        mx, my = x + shift * c, y + shift * s
        lo = np.stack([mx - hx, my - hy], axis=1)
        hi = np.stack([mx + hx, my + hy], axis=1)
        result = (lo[:, 0] < 0.0) | (lo[:, 1] < 0.0) | (hi[:, 0] > SIZE_M) | (hi[:, 1] > SIZE_M)

        cells = self._candidate_cells(lo.min(axis=0), hi.max(axis=0))
        if len(cells) == 0:
            return result
        boxes_lo = cells * CELL_M                                  # (M, 2)
        boxes_hi = boxes_lo + CELL_M
        # World axes: the rectangle's extent against each square.
        near = ((lo[:, None, 0] < boxes_hi[None, :, 0]) & (hi[:, None, 0] > boxes_lo[None, :, 0])
                & (lo[:, None, 1] < boxes_hi[None, :, 1]) & (hi[:, None, 1] > boxes_lo[None, :, 1]))
        s_idx, m_idx = np.nonzero(near)
        if len(s_idx) == 0:
            return result
        # Rectangle axes: project the square's corners into the vehicle frame.
        offsets = np.array([[0, 0], [1, 0], [1, 1], [0, 1]]) * CELL_M
        sq = boxes_lo[m_idx][:, None, :] + offsets                 # (P, 4, 2)
        dx = sq[..., 0] - x[s_idx, None]
        dy = sq[..., 1] - y[s_idx, None]
        c, s = c[s_idx, None], s[s_idx, None]
        along = dx * c + dy * s
        across = -dx * s + dy * c
        overlap = ((along.max(axis=1) > -b.rear) & (along.min(axis=1) < b.front)
                   & (across.max(axis=1) > -b.half_width) & (across.min(axis=1) < b.half_width))
        hit_samples = s_idx[overlap]
        result[hit_samples] = True
        return result

    def _candidate_cells(self, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
        """Occupied cells under the extent [lo, hi], as (M, 2) integer (cx, cy)."""
        cx0 = max(0, math.floor(lo[0] / CELL_M))
        cy0 = max(0, math.floor(lo[1] / CELL_M))
        cx1 = min(CELLS - 1, math.floor(hi[0] / CELL_M))
        cy1 = min(CELLS - 1, math.floor(hi[1] / CELL_M))
        if cx1 < cx0 or cy1 < cy0:
            return np.zeros((0, 2), dtype=int)
        sub = self.occupied[cx0:cx1 + 1, cy0:cy1 + 1]
        cx, cy = np.nonzero(sub)
        return np.stack([cx + cx0, cy + cy0], axis=1).astype(float)

    def first_hits(self, x0: float, y0: float, batch) -> np.ndarray:
        """For every edge of a heading bin (lattice.BinBatch) applied at node position (x0, y0):
        the index of the edge's first colliding sample, or -1 if the edge is free.

        Shortcut first: if no occupied cell lies under the extent of all the edges' rectangles
        together, and that extent is inside the lot, every edge is free."""
        lo, hi = self._extent(batch)
        lo = lo + (x0, y0)
        hi = hi + (x0, y0)
        inside = lo[0] >= 0.0 and lo[1] >= 0.0 and hi[0] <= SIZE_M and hi[1] <= SIZE_M
        if inside and len(self._candidate_cells(lo, hi)) == 0:
            return np.full(len(batch.edges), -1, dtype=int)
        return first_per_edge(self.hits(x0 + batch.x, y0 + batch.y, batch.psi), batch)

    def _extent(self, batch) -> tuple[np.ndarray, np.ndarray]:
        """Axis-aligned extent of the grown rectangles at all samples of a batch, relative to the
        node. Cached per batch."""
        key = id(batch)
        if key not in self._extents:
            corners = self.body.corners(batch.x, batch.y, batch.psi).reshape(-1, 2)
            self._extents[key] = (corners.min(axis=0), corners.max(axis=0))
        return self._extents[key]

    # --- one pose, for display and tests -------------------------------------------------------

    def describe(self, x: float, y: float, psi: float) -> Hit | None:
        corners = self.body.corners(x, y, psi)
        cells = [(cx, cy) for cx, cy in self._cells
                 if polygon_overlaps_box(corners, self.world.cell_box(cx, cy))]
        out = corners.min() < 0.0 or corners.max() > SIZE_M
        if not cells and not out:
            return None
        return Hit(sample=-1, corners=corners, cells=cells)

    def pose_free(self, x: float, y: float, psi: float) -> bool:
        return not self.hits(np.array([x]), np.array([y]), np.array([psi]))[0]


def first_per_edge(hit: np.ndarray, batch) -> np.ndarray:
    """From a boolean over a batch's samples: each edge's first colliding sample, or -1."""
    idx = np.flatnonzero(hit)
    first = np.full(len(batch.edges), -1, dtype=int)
    if len(idx):
        owners, where = np.unique(batch.owner[idx], return_index=True)
        first[owners] = idx[where] - batch.starts[owners]
    return first
