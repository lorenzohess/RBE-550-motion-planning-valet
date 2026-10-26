"""Command line: build the lot, plan with the lattice, simulate, and show or record it."""

from __future__ import annotations

import argparse
import sys

from . import lattice, parking, planning
from .collision import CollisionChecker
from .search import SearchParams, Stop
from .sim import kmh, simulate
from .vehicles import VEHICLES
from .world import build_world


def parse(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="valet", description=__doc__)
    p.add_argument("--vehicle", choices=sorted(VEHICLES), default="robot")
    p.add_argument("--seed", type=int, default=0, help="obstacle field seed")
    p.add_argument("--rho", type=float, default=0.10, help="obstacle density (default 0.10)")
    p.add_argument("--speed", type=float, default=10.0, choices=(10.0, 20.0, 30.0),
                   help="constant driving speed, km/h")
    p.add_argument("--record", metavar="MP4", help="write the whole run to an mp4 (needs ffmpeg)")
    p.add_argument("--headless", action="store_true", help="no window (use with --record)")
    p.add_argument("--no-search-view", action="store_true",
                   help="plan without drawing the search, then show the drive")
    p.add_argument("--collisions", action="store_true", help="start with the collision display on")
    p.add_argument("--max-expansions", type=int, default=SearchParams.max_expansions)
    p.add_argument("--coarse", type=float, default=1.0, metavar="RES",
                   help="coarse lattice resolution (m) for the first stage; 0 = one fine search only")
    p.add_argument("--weight", type=float, default=1.0,
                   help="heuristic weight; > 1 is weighted A* (faster, paths up to that factor costlier)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse(argv)
    vehicle = VEHICLES[args.vehicle]
    world = build_world(args.seed, args.rho, vehicle.layout, vehicle.footprint(*vehicle.start),
                        vehicle.goal_inset)
    cs = lattice.build(vehicle)
    coarse = None   # the truck has no coarse stage (planning.py)
    if args.coarse > 0 and vehicle.trailer is None:
        coarse = lattice.build(vehicle, resolution=args.coarse, n_base=16, extra=())
    checker = CollisionChecker(world, vehicle.body)
    show = not args.headless or args.record

    view = None
    if show:
        from .view import View  # pygame is only needed when drawing
        view = View(world, vehicle, cs, checker, args.speed, record=args.record,
                    headless=args.headless, collisions=args.collisions)
    try:
        # The car parks with a computed shuffle (parking.py); the lattice drives it to the entry pose.
        maneuver, target = None, None
        if vehicle.model == "car":
            maneuver = parking.find_maneuver(checker, vehicle, cs, world.goal_box)
            if maneuver is None:
                print(f"{vehicle.name} seed {args.seed}: no_path (no parking manoeuvre fits)")
                if view is not None:
                    view.show_map(0.5)
                    view.show_no_path("no_path")
                return 2
            target = maneuver.entry
            if view is not None:
                entry = (target.x * cs.resolution, target.y * cs.resolution, cs.headings[target.psi].psi)
                view.set_parking(entry, simulate(vehicle, entry, maneuver.edges, kmh(args.speed)))
        if view is not None:
            view.show_map(1.0)
        observer = None if (view is None or args.no_search_view) else view
        params = SearchParams(max_expansions=args.max_expansions, heuristic_weight=args.weight)
        plan = planning.plan(world, vehicle, checker, cs, coarse, params, observer, maneuver)
        edges = plan.edges + (maneuver.edges if maneuver is not None and plan.status == "found" else [])
        stages = ", ".join(f"{name} {p.status} {p.expansions} in {p.seconds:.1f} s" for name, p in plan.stages)
        summary = (f"{vehicle.name} seed {args.seed}: {plan.status}, {plan.expansions} expansions, "
                   f"{plan.seconds:.1f} s ({stages})")
        if plan.status == "found":
            cusps = sum(1 for a, b in zip(edges, edges[1:]) if a.reverse != b.reverse)
            summary += (f", path {sum(e.prim.length for e in edges):.1f} m, {len(edges)} edges, "
                        f"{cusps} gear changes")
            if maneuver is not None:
                summary += f" (parking: {maneuver.moves} moves, {maneuver.length:.1f} m)"
        print(summary)
        if view is not None:
            if plan.status == "found":
                traj = simulate(vehicle, vehicle.start, edges, kmh(args.speed))
                view.drive(traj, edges)
            elif plan.status != "quit":
                view.show_no_path(plan.status)
        return 0 if plan.status == "found" else 2
    except Stop:
        return 1
    finally:
        if view is not None:
            view.close()


if __name__ == "__main__":
    sys.exit(main())
