"""Lattice A*, using the `astar` package (PyPI, BSD) for the search itself.

A node is (x, y, psi): x and y in grid steps of the control set's resolution, psi the heading bin.
Gear is not part of the node: reverse edges are mirrored primitives, charged a reverse penalty, as
in the basic simulation planner (topographical-planner/basic-simulation-planner, ssp/search.py).

Edge costs follow Nav2's lattice planner (nav2_smac_planner node_lattice.cpp) and the basic
simulation planner, in metres:
  straight       length
  curve          length x non_straight_penalty, + change_penalty in the multiplier when it turns
                 the opposite way to the curve that reached the node
  rotation       rotation_cost, per heading bin (Nav2: rotation_penalty grid steps; 5 x 0.1 m)
  reverse        x reverse_penalty

The change penalty needs the edge that reached a node. The astar package keeps each node's best
parent (came_from) and a free `cache` slot, where we record that edge whenever the package takes a
cheaper path to the node (path_distance_between).

Heuristic: heuristic.py. Children it proves cannot reach the goal (infinite heuristic) are dropped.

The truck (TruckPlanner) adds the hitch angle to the node, rounded to 5 deg bins: two arrivals at
the same truck pose with the trailer at different angles are different states. The exact angle of
the best path to each node is kept, and the trailer is carried along every edge from it
(trailer.py), and checked against obstacles and the truck at every sample.

The truck parks by driving straight in along the strip's row: driving straight, the hitch angle
shrinks as tan(beta / 2) ~ exp(-s / d1), and nothing else straightens the trailer in a strip 0.35 m
longer than the rig. So its goal test is a straight shot (like Hybrid A*'s analytic expansion): a
node is a goal if it heads east on the parked row and driving straight on to a parked x is
collision-free and ends with the rig in the goal box. Its heuristic aims at a pre-parking point
PRE_PARK metres west of the parked rear axle on that row, far enough for the straight to bring
even the steady-state hitch angle (49 deg) under 2 deg.
"""

from __future__ import annotations

import heapq
import itertools
import math
import time
from dataclasses import dataclass, field
from typing import NamedTuple, Protocol

import astar
import numpy as np
from astar import AStar

from . import trailer
from .collision import CollisionChecker, contact_angle, first_per_edge
from .heuristic import Heuristic, parked_reference_box, reference_free
from .lattice import BinBatch, ControlSet, Edge, _samples, _straight, wrap
from .vehicles import Vehicle
from .world import World


class LazyOpenSet:
    """Drop-in for astar.OpenSet. The package's remove() (used when a node in the open set gets a
    cheaper path) finds the node with list.index, a scan of the whole open set, which took 70 % of
    the run time. Here remove() only marks the node, and pop() skips entries that are out of date.
    Ties in f go to the node with the larger g (deeper along its path), which keeps A* from
    widening along whole fronts of equal f. The search loop itself is the package's, unchanged."""

    def __init__(self):
        self.heap: list = []
        self.size = 0
        self._count = itertools.count()

    def push(self, item) -> None:
        item.in_openset = True
        heapq.heappush(self.heap, (item.fscore, -item.gscore, next(self._count), item))
        self.size += 1

    def pop(self):
        while True:
            fscore, _, _, item = heapq.heappop(self.heap)
            if item.in_openset and fscore == item.fscore:
                item.in_openset = False
                self.size -= 1
                return item

    def remove(self, item) -> None:
        item.in_openset = False
        self.size -= 1

    def __len__(self) -> int:
        return self.size


astar.OpenSet = LazyOpenSet   # AStar.astar() looks OpenSet up in the package module at call time


class Node(NamedTuple):
    x: int     # grid steps east
    y: int     # grid steps north
    psi: int   # heading bin


class RigNode(NamedTuple):
    """Truck and trailer."""
    x: int
    y: int
    psi: int    # truck heading bin
    beta: int   # hitch angle bin, round(beta / BETA_BIN)


BETA_BIN = math.radians(5.0)
PRE_PARK = 16.0   # m, the truck's heuristic aims this far west of the parked rear axle


@dataclass
class SearchParams:
    reverse_penalty: float = 2.0      # Nav2 default
    rotation_cost: float = 0.5        # m per one-bin rotation: Nav2's 5 grid steps at 0.1 m
    non_straight_penalty: float = 1.05   # Nav2 default
    change_penalty: float = 0.05      # Nav2 default
    max_expansions: int = 1_000_000
    heuristic_weight: float = 1.0     # > 1: weighted A* (faster, paths up to this factor costlier)


class Observer(Protocol):
    """Receives every expansion while the search runs (the live display): the node, its parent and
    the edge between them (None for the start), and the node's edges with their first collision."""
    def expanded(self, node: Node, parent: Node | None, incoming: Edge | None, batch: BinBatch,
                 first_hits: np.ndarray) -> None: ...


class Stop(Exception):
    """Raised from inside the search to end it early (expansion cap, or the user quit)."""
    def __init__(self, status: str):
        super().__init__(status)
        self.status = status


@dataclass
class Plan:
    status: str                    # "found" | "no_path" | "expansion_cap" | "quit"
    nodes: list[Node] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)   # edges[i] leads from nodes[i] to nodes[i + 1]
    expansions: int = 0
    seconds: float = 0.0
    cost: float = 0.0

    @property
    def length(self) -> float:
        return sum(e.prim.length for e in self.edges)


class LatticePlanner(AStar):
    def __init__(self, world: World, vehicle: Vehicle, cs: ControlSet, checker: CollisionChecker,
                 params: SearchParams | None = None, observer: Observer | None = None,
                 target: tuple[float, float, float] | None = None,
                 start: tuple[float, float, float] | None = None, target_tolerance: float = 0.0):
        """target: a pose to reach instead of the goal box (rounded to the nearest node). With
        target_tolerance > 0, any node with the target's heading, on the target's row (the line
        through it along that heading) and within target_tolerance metres of it along the row
        counts.
        start: where to start (default: the vehicle's start). The lattice is placed so that the
        start is a node: node (i, j) is at origin + (i, j) * resolution."""
        self.world = world
        self.vehicle = vehicle
        self.cs = cs
        self.checker = checker
        self.params = params or SearchParams()
        self.observer = observer
        self.expansions = 0
        self._costs: dict[Node, tuple[float, Edge]] = {}
        x, y, psi = start if start is not None else vehicle.start
        res = cs.resolution
        self.origin = (x - res * round(x / res), y - res * round(y / res))
        self.start = Node(round(x / res), round(y / res), cs.nearest_bin(psi))
        self.target = self.node_at(target) if target is not None else None
        self.target_tolerance = target_tolerance
        self.heuristic = Heuristic(world, vehicle, cs, self.origin, self.params.rotation_cost,
                                   self.heuristic_target())

    def heuristic_target(self):
        """The pose the heuristic aims at; None for the goal box."""
        return self.pose(self.target) if self.target is not None else None

    # --- geometry ------------------------------------------------------------------------------

    def pose(self, node: Node) -> tuple[float, float, float]:
        res = self.cs.resolution
        return (self.origin[0] + node.x * res, self.origin[1] + node.y * res,
                self.cs.headings[node.psi].psi)

    def node_at(self, pose) -> Node:
        """The node nearest to a pose."""
        res = self.cs.resolution
        return Node(round((pose[0] - self.origin[0]) / res), round((pose[1] - self.origin[1]) / res),
                    self.cs.nearest_bin(pose[2]))

    def edge_cost(self, edge: Edge, incoming: Edge | None = None) -> float:
        p = self.params
        if edge.prim.kind == "rotate":
            cost = p.rotation_cost
        elif edge.prim.kind == "curve":
            factor = p.non_straight_penalty
            if (incoming is not None and incoming.prim.kind == "curve"
                    and incoming.prim.left_turn != edge.prim.left_turn):
                factor += p.change_penalty
            cost = edge.prim.length * factor
        else:
            cost = edge.prim.length
        if edge.reverse:
            cost *= p.reverse_penalty
        return cost

    def parked(self, node: Node) -> bool:
        x, y, psi = self.pose(node)
        return self._parked(x, y, [psi], self.vehicle.footprint(x, y, psi))

    def _parked(self, x, y, headings, bodies) -> bool:
        x0, x1, y0, y1 = self.world.goal_box
        e = 1e-9      # round-off: lattice poses can sit exactly on the box edge
        for corners in bodies:
            if (corners[:, 0].min() < x0 - e or corners[:, 0].max() > x1 + e
                    or corners[:, 1].min() < y0 - e or corners[:, 1].max() > y1 + e):
                return False
        tol = self.vehicle.heading_tolerance + 1e-9
        return all(any(abs(float(wrap(psi - h))) <= tol for h in self.vehicle.goal_headings)
                   for psi in headings)

    # --- astar package interface -------------------------------------------------------------

    def heuristic_cost_estimate(self, current: Node, goal) -> float:
        return self.params.heuristic_weight * self.heuristic(current.x, current.y, current.psi)

    def in_goal_box(self, node: Node) -> bool:
        x, y, _ = self.pose(node)
        x0, x1, y0, y1 = self.world.goal_box
        return x0 <= x <= x1 and y0 <= y <= y1

    def is_goal_reached(self, current: Node, goal) -> bool:
        if self.target is not None:
            if self.target_tolerance <= 0.0:
                return current == self.target
            if current.psi != self.target.psi:
                return False
            res = self.cs.resolution
            dx, dy = (current.x - self.target.x) * res, (current.y - self.target.y) * res
            psi = self.cs.headings[self.target.psi].psi
            along = dx * math.cos(psi) + dy * math.sin(psi)
            across = -dx * math.sin(psi) + dy * math.cos(psi)
            return abs(across) <= 0.5 * res and abs(along) <= self.target_tolerance + 1e-9
        # The reference point is inside the body, so it must be in the goal box first.
        return self.in_goal_box(current) and self.parked(current)

    def path_neighbors(self, snode) -> list[Node]:
        node: Node = snode.data
        self.expansions += 1
        if self.expansions > self.params.max_expansions:
            raise Stop("expansion_cap")
        incoming: Edge | None = snode.cache
        x, y, _ = self.pose(node)
        batch = self.cs.batches[node.psi]
        first = self.checker.first_hits(x, y, batch)
        self._costs = {}
        for i, edge in enumerate(batch.edges):
            if first[i] >= 0:
                continue
            child = Node(node.x + edge.dx, node.y + edge.dy, edge.end_bin)
            if math.isinf(self.heuristic(child.x, child.y, child.psi)):
                continue
            cost = self.edge_cost(edge, incoming)
            if cost < self._costs.get(child, (math.inf, None))[0]:
                self._costs[child] = (cost, edge)
        if self.observer is not None:
            parent = snode.came_from.data if snode.came_from is not None else None
            self.observer.expanded(node, parent, incoming, batch, first)
        return list(self._costs)

    def path_distance_between(self, n1, n2) -> float:
        cost, edge = self._costs[n2.data]
        if n1.gscore + cost < n2.gscore:
            n2.cache = edge          # the package will take this path to n2 (see astar())
        return cost

    # --- running -----------------------------------------------------------------------------

    def edge_between(self, a: Node, b: Node, incoming: Edge | None) -> Edge:
        """The cheapest collision-free edge from a to b (used to recover the path's primitives)."""
        x, y, _ = self.pose(a)
        batch = self.cs.batches[a.psi]
        first = self.checker.first_hits(x, y, batch)
        best, best_cost = None, math.inf
        for i, edge in enumerate(batch.edges):
            if first[i] < 0 and (a.x + edge.dx, a.y + edge.dy, edge.end_bin) == tuple(b):
                cost = self.edge_cost(edge, incoming)
                if cost < best_cost:
                    best, best_cost = edge, cost
        if best is None:
            raise RuntimeError(f"no edge from {a} to {b}")
        return best

    def start_free(self) -> bool:
        return self.checker.pose_free(*self.pose(self.start))

    def path_edges(self, nodes) -> list[Edge]:
        edges: list[Edge] = []
        for a, b in zip(nodes, nodes[1:]):
            edges.append(self.edge_between(a, b, edges[-1] if edges else None))
        return edges

    def plan(self) -> Plan:
        t0 = time.perf_counter()
        try:
            s = self.start
            if not self.start_free() or math.isinf(self.heuristic(s.x, s.y, s.psi)):
                raise Stop("no_path")
            result = self.astar(self.start, self.world.goal_box)
        except Stop as stop:
            return Plan(stop.status, expansions=self.expansions, seconds=time.perf_counter() - t0)
        seconds = time.perf_counter() - t0
        if result is None:
            return Plan("no_path", expansions=self.expansions, seconds=seconds)
        nodes = list(result)
        edges = self.path_edges(nodes)
        cost = sum(self.edge_cost(e, edges[i - 1] if i else None) for i, e in enumerate(edges))
        return Plan("found", nodes, edges, self.expansions, seconds, cost)


class TruckPlanner(LatticePlanner):
    """Lattice A* for the truck and trailer. Nodes are RigNodes; self.beta holds the exact hitch
    angle of the best path found to each node (updated together with the package's came_from)."""

    def __init__(self, world: World, vehicle: Vehicle, cs: ControlSet, checker: CollisionChecker,
                 params: SearchParams | None = None, observer: Observer | None = None,
                 target: tuple[float, float, float] | None = None,
                 start: tuple[float, float, float] | None = None, target_tolerance: float = 0.0,
                 beta0: float = 0.0, target_beta: float = 0.5 * BETA_BIN):
        """beta0: hitch angle at the start. target_beta: with a target pose, the hitch angle must
        also be within this of zero (default: the zero bin)."""
        super().__init__(world, vehicle, cs, checker, params, observer, target, start, target_tolerance)
        self.d1 = vehicle.trailer.hitch_to_axle
        self.trailer_checker = CollisionChecker(world, vehicle.trailer.body, checker.margin)
        self.contact = contact_angle(vehicle, checker.margin)
        self.target_beta = target_beta
        n = self.start
        self.start = RigNode(n.x, n.y, n.psi, round(beta0 / BETA_BIN))
        self.beta: dict[RigNode, float] = {self.start: beta0}
        self._via: dict[RigNode, Edge] = {}
        self.last_trailer = None     # for the display: the trailer at the last expansion's samples
        self.shot = None             # (node, steps, beta at the end) of the straight shot found

    def heuristic_target(self):
        # Called from LatticePlanner.__init__: where the parked rear axle can be is needed first.
        self.reference_box = parked_reference_box(self.vehicle, self.world.goal_box)
        if self.target is not None:
            return super().heuristic_target()
        # The pre-parking point; moved east along the row if an obstacle covers it.
        x0, x1, y0, y1 = self.reference_box
        x, y, psi = 0.5 * (x0 + x1) - PRE_PARK, 0.5 * (y0 + y1), self.vehicle.goal_headings[0]
        while x < x0 and not reference_free(self.world, self.vehicle, x, y):
            x += self.cs.resolution
        return (x, y, psi)

    def straight_shot(self, node: RigNode) -> bool:
        """Can the rig park by driving straight on from this node? Records the shot in self.shot."""
        x, y, psi, beta = self.rig_pose(node)
        if abs(float(wrap(psi - self.vehicle.goal_headings[0]))) > 1e-9:
            return False
        rx0, rx1, ry0, ry1 = self.reference_box
        if not ry0 <= y <= ry1:
            return False
        res, d1 = self.cs.resolution, self.d1
        c, s = math.cos(psi), math.sin(psi)
        # Steps along the heading to each parked x, farthest first (the trailer has the most room).
        k0 = max(0, math.ceil((rx0 - x) / (res * c) - 1e-9))
        k1 = math.floor((rx1 - x) / (res * c) + 1e-9)
        for k in range(k1, k0 - 1, -1):
            dist = k * res
            # Driving straight: tan(beta / 2) = tan(beta0 / 2) exp(-s / d1).
            b_end = 2.0 * math.atan(math.tan(beta / 2) * math.exp(-dist / d1))
            px, py = x + dist * c, y + dist * s
            if not self._parked(px, py, [psi, psi - b_end], self.vehicle.footprint(px, py, psi, b_end)):
                continue
            n = max(2, math.ceil(dist * self.vehicle.trailer.travel_per_metre() / (2 * self.checker.margin)) + 1)
            d = np.linspace(0.0, dist, n)
            bb = 2.0 * np.arctan(np.tan(beta / 2) * np.exp(-d / d1))
            xs, ys, heading = x + d * c, y + d * s, np.full(n, psi)
            if self.checker.hits(xs, ys, heading).any():
                continue
            ax, ay = self.vehicle.trailer.axle(xs, ys, heading - bb)
            if self.trailer_checker.hits(ax, ay, heading - bb).any():
                continue
            self.shot = (node, k, b_end)
            return True
        return False

    def rig_pose(self, node: RigNode) -> tuple[float, float, float, float]:
        return (*self.pose(node), self.beta[node])

    def trailer_poses(self, x: float, y: float, batch, beta: np.ndarray):
        """Trailer axle poses at every sample of a batch applied at (x, y), given beta there."""
        psi1 = batch.psi - beta
        ax, ay = self.vehicle.trailer.axle(x + batch.x, y + batch.y, psi1)
        return ax, ay, psi1

    def rig_first_hits(self, x: float, y: float, batch, beta: np.ndarray):
        """First colliding sample of each edge (or -1), for the truck and for the trailer (against
        obstacles, the lot boundary, or the truck)."""
        truck = self.checker.first_hits(x, y, batch)
        ax, ay, psi1 = self.trailer_poses(x, y, batch, beta)
        hit = self.trailer_checker.hits(ax, ay, psi1) | (np.abs(beta) >= self.contact)
        trl = first_per_edge(hit, batch)
        first = np.where(truck < 0, trl, np.where(trl < 0, truck, np.minimum(truck, trl)))
        self.last_trailer = (ax, ay, psi1, truck, trl)
        return first

    def start_free(self) -> bool:
        x, y, psi = self.pose(self.start)
        beta = self.beta[self.start]
        psi1 = psi - beta
        ax, ay = self.vehicle.trailer.axle(x, y, psi1)
        return (self.checker.pose_free(x, y, psi) and self.trailer_checker.pose_free(ax, ay, psi1)
                and abs(beta) < self.contact)

    def parked(self, node: RigNode) -> bool:
        x, y, psi, beta = self.rig_pose(node)
        return self._parked(x, y, [psi, psi - beta], self.vehicle.footprint(x, y, psi, beta))

    def is_goal_reached(self, current: RigNode, goal) -> bool:
        if self.target is not None:
            if abs(self.beta[current]) > self.target_beta:
                return False
            return super().is_goal_reached(Node(current.x, current.y, current.psi), goal)
        return self.straight_shot(current)

    def path_neighbors(self, snode) -> list[RigNode]:
        node: RigNode = snode.data
        self.expansions += 1
        if self.expansions > self.params.max_expansions:
            raise Stop("expansion_cap")
        incoming: Edge | None = snode.cache
        x, y, _ = self.pose(node)
        batch = self.cs.batches[node.psi]
        beta = trailer.lookup(batch.beta, self.beta[node])
        first = self.rig_first_hits(x, y, batch, beta)
        ends = batch.ends()
        self._costs = {}
        for i, edge in enumerate(batch.edges):
            if first[i] >= 0:
                continue
            b_end = math.remainder(float(beta[ends[i]]), 2 * math.pi)
            child = RigNode(node.x + edge.dx, node.y + edge.dy, edge.end_bin, round(b_end / BETA_BIN))
            if math.isinf(self.heuristic(child.x, child.y, child.psi)):
                continue
            cost = self.edge_cost(edge, incoming)
            if cost < self._costs.get(child, (math.inf,))[0]:
                self._costs[child] = (cost, edge, b_end)
        if self.observer is not None:
            parent = snode.came_from.data if snode.came_from is not None else None
            self.observer.expanded(node, parent, incoming, batch, first)
        return list(self._costs)

    def path_distance_between(self, n1, n2) -> float:
        cost, edge, b_end = self._costs[n2.data]
        if n1.gscore + cost < n2.gscore:
            n2.cache = edge          # the package will take this path to n2 (see astar())
            self.beta[n2.data] = b_end
            self._via[n2.data] = edge
        return cost

    def path_edges(self, nodes) -> list[Edge]:
        return [self._via[n] for n in nodes[1:]]

    def plan(self) -> Plan:
        """The lattice path, then the straight shot as one more edge (a straight of k grid steps
        along the heading)."""
        plan = super().plan()
        if plan.status != "found" or self.target is not None:
            return plan
        node, k, b_end = self.shot
        if k > 0:
            h = self.cs.headings[node.psi]   # east: one grid step is (1, 0), so k steps
            prim = _straight(self.cs.headings, node.psi, k, self.cs.resolution)
            samples, poly = _samples(prim, h.psi, False, self.cs.resolution, self.vehicle.body,
                                     self.checker.margin, self.vehicle.trailer)
            edge = Edge(prim, False, node.psi, samples, poly)
            end = RigNode(node.x + prim.dx, node.y + prim.dy, node.psi, round(b_end / BETA_BIN))
            self.beta[end] = b_end
            plan.edges.append(edge)
            plan.nodes.append(end)
            plan.cost += self.edge_cost(edge, plan.edges[-2] if len(plan.edges) > 1 else None)
        return plan


def make_planner(world: World, vehicle: Vehicle, cs: ControlSet, checker: CollisionChecker,
                 params: SearchParams | None = None, observer: Observer | None = None,
                 target=None, start=None, target_tolerance: float = 0.0, beta0: float = 0.0):
    """The planner for a vehicle: TruckPlanner when it pulls a trailer."""
    if vehicle.trailer is not None:
        return TruckPlanner(world, vehicle, cs, checker, params, observer, target, start,
                            target_tolerance, beta0)
    return LatticePlanner(world, vehicle, cs, checker, params, observer, target, start, target_tolerance)
