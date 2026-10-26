"""A* heuristic: a lower bound on the cost from a node to a parked pose.

Two parts:

- Position: the shortest 8-connected grid path (Dijkstra, run once from the goal) for the
  reference point, around obstacles grown by the reference point's clearance: the radius of the
  largest circle centred on the reference point that fits in the body (0.285 m robot, 0.9 m car).
  The reference point can never be closer to an obstacle than that, so the grown obstacles block
  nothing the vehicle can reach. Grid paths are divided by 1.0824, the largest ratio of an
  8-connected path to the straight line it stands for, so they never overestimate (the basic
  simulation planner's h_2D, topographical-planner ssp/heuristic.py). The Euclidean distance to
  the goal box is the floor.

- Heading: the heading still to change before the vehicle is parked (east or west, within the
  tolerance).
    robot: each one-bin rotation costs rotation_cost, and rotations are the only way
           to turn, so the fewest rotations to a parked heading are added to the position term.
    car:   turning by dpsi takes at least rho_min * dpsi of driving (|kappa| <= 1 / rho_min). That
           driving also moves the car, so it is not added: h = max(position, rho_min * dpsi).

Nodes the position part cannot reach (blocked for the reference point, or cut off from the goal
box) get an infinite heuristic, and the search drops them.

The goal is the goal box or a single pose (a lattice node: the car's parking entry, or the handoff
between the coarse and the fine search); for a pose, the source is its one grid point and the
heading term aims at its heading exactly. For the goal box, the source is the part of the box where
the reference point of a parked vehicle can be (parked_reference_box): the truck's rear axle, for
one, can only be within 0.15 m of one spot in a 12 m strip.

The position part is always computed on a 0.1 m grid, whatever the lattice's resolution and origin,
and looked up at each node's position in metres.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

from .lattice import ControlSet, wrap
from .vehicles import Vehicle
from .world import CELL_M, SIZE_M, World

OCTILE_OVER_EUCLID = (math.sqrt(2) - 1) * math.sin(math.radians(22.5)) + math.cos(math.radians(22.5))
GRID = 0.1   # m, resolution of the position part


def clearance_radius(vehicle: Vehicle) -> float:
    b = vehicle.body
    return min(b.rear, b.front, b.half_width)


def reference_free(world: World, vehicle: Vehicle, x: float, y: float) -> bool:
    """Is (x, y) at least the clearance radius from every occupied cell and the lot boundary?"""
    r = clearance_radius(vehicle)
    if min(x, SIZE_M - x, y, SIZE_M - y) < r:
        return False
    for cx, cy in zip(*np.nonzero(world.occupied)):
        dx = max(cx * CELL_M - x, 0.0, x - (cx + 1) * CELL_M)
        dy = max(cy * CELL_M - y, 0.0, y - (cy + 1) * CELL_M)
        if math.hypot(dx, dy) < r:
            return False
    return True


def parked_reference_box(vehicle: Vehicle, goal_box, step: float = math.radians(0.5),
                         pad: float = 0.01) -> tuple[float, float, float, float]:
    """The box around every position of the reference point at which the vehicle is parked: each
    body inside the goal box, each heading (truck and trailer) within the tolerance of a goal
    heading. Headings scanned in `step`s, the box padded by `pad` for what the scan misses."""
    x0, x1, y0, y1 = goal_box
    tol = vehicle.heading_tolerance
    offsets = np.linspace(-tol, tol, 2 * math.ceil(tol / step) + 1)
    lo_x = lo_y = math.inf
    hi_x = hi_y = -math.inf
    for g in vehicle.goal_headings:
        for a in offsets:
            for b in (offsets if vehicle.trailer is not None else (a,)):
                corners = np.concatenate(vehicle.footprint(0.0, 0.0, g + a, a - b))
                ax0, ax1 = x0 - corners[:, 0].min(), x1 - corners[:, 0].max()
                ay0, ay1 = y0 - corners[:, 1].min(), y1 - corners[:, 1].max()
                if ax0 <= ax1 and ay0 <= ay1:
                    lo_x, hi_x = min(lo_x, ax0), max(hi_x, ax1)
                    lo_y, hi_y = min(lo_y, ay0), max(hi_y, ay1)
    if math.isinf(lo_x):
        return goal_box          # nothing fits; the search will report no path
    return (lo_x - pad, hi_x + pad, lo_y - pad, hi_y + pad)


def position_cost_to_go(world: World, vehicle: Vehicle, res: float,
                        target: tuple[int, int] | None = None,
                        box: tuple[float, float, float, float] | None = None) -> np.ndarray:
    """(n, n) array over grid points (x = i res, y = j res): grid-path distance to `box` (default
    the goal box), or to the grid point `target`, for the reference point; inf where it cannot be.
    Grid points within res / 2 of the box count as in it, so a box thinner than res still has some
    (that can only lower the distances)."""
    n = round(SIZE_M / res) + 1
    coords = np.arange(n) * res
    x, y = np.meshgrid(coords, coords, indexing="ij")
    # Clearance: distance to the nearest grid point in or on an occupied cell (exact, as cell
    # corners are grid points), and to the lot boundary.
    inside = np.zeros((n, n), dtype=bool)
    for cx, cy in zip(*np.nonzero(world.occupied)):
        inside |= ((x >= cx * CELL_M - 1e-9) & (x <= (cx + 1) * CELL_M + 1e-9)
                   & (y >= cy * CELL_M - 1e-9) & (y <= (cy + 1) * CELL_M + 1e-9))
    clearance = np.minimum.reduce([x, SIZE_M - x, y, SIZE_M - y])
    if inside.any():
        clearance = np.minimum(clearance, distance_transform_edt(~inside) * res)
    blocked = clearance < clearance_radius(vehicle)

    ids = np.arange(n * n).reshape(n, n)
    src, dst, wts = [], [], []
    for di, dj in ((1, 0), (0, 1), (1, 1), (1, -1)):
        a = (slice(0, n - di), slice(max(0, -dj), n - max(0, dj)))
        b = (slice(di, n), slice(max(0, dj), n - max(0, -dj)))
        ok = ~blocked[a] & ~blocked[b]
        src.append(ids[a][ok])
        dst.append(ids[b][ok])
        wts.append(np.full(int(ok.sum()), res * math.hypot(di, dj)))
    graph = coo_matrix((np.concatenate(wts), (np.concatenate(src), np.concatenate(dst))),
                       shape=(n * n, n * n)).tocsr()
    if target is not None:
        goal = ids[target] if not blocked[target] else np.zeros(0, dtype=int)
        goal = np.atleast_1d(goal)
    else:
        x0, x1, y0, y1 = box if box is not None else world.goal_box
        h = 0.5 * res
        goal = ids[(x >= x0 - h) & (x <= x1 + h) & (y >= y0 - h) & (y <= y1 + h) & ~blocked]
    if len(goal) == 0:
        return np.full((n, n), np.inf)
    dist = dijkstra(graph, directed=False, indices=goal, min_only=True).reshape(n, n)
    dist[blocked] = np.inf
    return dist / OCTILE_OVER_EUCLID


class Heuristic:
    def __init__(self, world: World, vehicle: Vehicle, cs: ControlSet, origin: tuple[float, float],
                 rotation_cost: float, target: tuple[float, float, float] | None = None):
        """origin: position of lattice node (0, 0) in metres. target: a pose to reach instead of the
        goal box."""
        self.res = cs.resolution
        self.origin = origin
        if target is None:
            self.goal_box = parked_reference_box(vehicle, world.goal_box)
            headings, tol = vehicle.goal_headings, vehicle.heading_tolerance
            self.position = position_cost_to_go(world, vehicle, GRID, box=self.goal_box)
        else:
            tx, ty, tpsi = target
            self.goal_box = (tx, tx, ty, ty)
            headings, tol = (tpsi,), 0.0
            self.position = position_cost_to_go(world, vehicle, GRID, (round(tx / GRID), round(ty / GRID)))
        turn = np.array([max(0.0, min(abs(float(wrap(h.psi - g))) for g in headings) - tol)
                         for h in cs.headings])
        self.additive = vehicle.model == "diff"
        if self.additive:
            goal_bins = [k for k, t in enumerate(turn) if t <= 1e-9]
            n = cs.n
            steps = [min(min((k - g) % n, (g - k) % n) for g in goal_bins) for k in range(n)]
            self.heading = np.array(steps, float) * rotation_cost
        else:
            self.heading = vehicle.min_turning_radius * turn

    def __call__(self, x: int, y: int, psi: int) -> float:
        """x, y: lattice node indices; psi a heading bin."""
        px = self.origin[0] + x * self.res
        py = self.origin[1] + y * self.res
        i, j = round(px / GRID), round(py / GRID)
        n = self.position.shape[0]
        if not (0 <= i < n and 0 <= j < n):
            return math.inf
        x0, x1, y0, y1 = self.goal_box
        euclid = math.hypot(max(x0 - px, 0.0, px - x1), max(y0 - py, 0.0, py - y1))
        pos = max(euclid, float(self.position[i, j]))
        if self.additive:
            return pos + float(self.heading[psi])
        return max(pos, float(self.heading[psi]))
