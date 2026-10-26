"""Coarse-to-fine planning.

1. Coarse: a 1 m lattice with Nav2's 16 headings, from the start to a handoff pose near the parking
   area: 4 m before the car's parking entry, or above the robot's slot.
2. Fine: the 0.1 m lattice with 36 headings, from the handoff pose to the goal (the goal box, or
   the car's parking entry).
3. Car: the parking shuffle (parking.py).

The truck has no coarse stage: see below.

Every coarse node is also a fine node: 1 m is a multiple of 0.1 m, the 16 headings are among the 36,
and both lattices are placed so that their start is a node. So the fine search starts exactly where
the coarse path ends. A coarse search has about 1300 x 16 states in the whole lot, against about
130 000 x 36 at 0.1 m.

If either stage finds nothing, the plan falls back to one fine search from the start, as weighted
A* (heuristic x FALLBACK_WEIGHT). The coarse stage fails when the car starts boxed in and needs a
three-point turn its long turns can't make; a plain 0.1 m search over the whole lot then runs out of
expansions (seed 1: 1M in 7 min), the weighted one parks in about 2.5 min.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

from .collision import CollisionChecker
from .lattice import ControlSet
from .parking import Maneuver
from .search import Plan, SearchParams, make_planner
from .vehicles import Vehicle
from .world import World

HANDOFF = 4.0          # m, coarse path ends this far before the car's parking entry
ROBOT_HANDOFF = 2.0    # m, above the robot's goal box
HANDOFF_TOLERANCE = 3.0  # m: any coarse node on the handoff's row, facing the same way, within this
                         # distance along the row will do, i.e. 1 to 7 m before the car's entry, so the
                         # fine search just drives straight on. Aiming at one 1 m node exactly made
                         # the coarse path overshoot and back up; allowing nodes off the row left
                         # the fine search to undo the offset, with gear changes.
FALLBACK_WEIGHT = 2.0    # weighted A* for the fine search when coarse to fine fails
TRUCK_WEIGHT = 3.0       # the truck's single fine search (see below)


@dataclass
class Result:
    status: str
    edges: list = field(default_factory=list)
    stages: list[tuple[str, Plan]] = field(default_factory=list)

    @property
    def expansions(self) -> int:
        return sum(p.expansions for _, p in self.stages)

    @property
    def seconds(self) -> float:
        return sum(p.seconds for _, p in self.stages)


def handoff_pose(world: World, vehicle: Vehicle, fine: ControlSet, maneuver: Maneuver | None):
    if maneuver is not None:
        e = maneuver.entry
        psi = fine.headings[e.psi].psi
        return (e.x * fine.resolution - HANDOFF * math.cos(psi),
                e.y * fine.resolution - HANDOFF * math.sin(psi), psi)
    x0, x1, y0, y1 = world.goal_box
    return (0.5 * (x0 + x1), y1 + ROBOT_HANDOFF, -math.pi / 2)


def plan(world: World, vehicle: Vehicle, checker: CollisionChecker, fine: ControlSet,
         coarse: ControlSet | None, params: SearchParams, observer=None,
         maneuver: Maneuver | None = None) -> Result:
    target = None
    if maneuver is not None:
        e = maneuver.entry
        target = (e.x * fine.resolution, e.y * fine.resolution, fine.headings[e.psi].psi)
    result = Result("no_path")

    def run(name, planner):
        if observer is not None:
            observer.begin_search(planner)
        p = planner.plan()
        result.stages.append((name, p))
        return p

    if vehicle.trailer is not None:
        # The truck: one fine search, weighted, to a straight shot into the strip (search.py).
        # No coarse stage: a turn onto the strip's row must end 1.5-2.1 m above the lot edge (the
        # truck's front corner swings low at the end of a turn), which the 1 m lattice can't hit.
        # Weighted, as the heuristic can't see that, nor how far the trailer needs to straighten.
        weight = max(params.heuristic_weight, TRUCK_WEIGHT)
        p = run(f"fine, weighted x{weight:g}",
                make_planner(world, vehicle, fine, checker, replace(params, heuristic_weight=weight), observer))
        result.status = p.status
        result.edges = p.edges
        return result

    if coarse is not None:
        first = make_planner(world, vehicle, coarse, checker, params, observer,
                             target=handoff_pose(world, vehicle, fine, maneuver),
                             target_tolerance=HANDOFF_TOLERANCE)
        p1 = run("coarse", first)
        if p1.status == "quit":
            result.status = "quit"
            return result
        if p1.status == "found":
            second = make_planner(world, vehicle, fine, checker, params, observer, target=target,
                                  start=first.pose(p1.nodes[-1]))
            p2 = run("fine", second)
            if p2.status in ("found", "quit"):
                result.status = p2.status
                result.edges = p1.edges + p2.edges
                return result

    if coarse is None:
        single = make_planner(world, vehicle, fine, checker, params, observer, target=target)
        p = run("fine only", single)
    else:
        weight = max(params.heuristic_weight, FALLBACK_WEIGHT)
        single = make_planner(world, vehicle, fine, checker, replace(params, heuristic_weight=weight),
                              observer, target=target)
        p = run(f"fine only, weighted x{weight:g}", single)
    result.status = p.status
    result.edges = p.edges
    return result
