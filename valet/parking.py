"""Parking the car: the lattice drives to an entry pose in the aisle, then a computed shuffle parks.

The slot (Fig. 5a) is 6 m long and the car 5.2 m. Parked, the rear axle has about 0.5 m of fore-aft
room, less than any lattice turn with the wheels straight at both ends needs (1.06 m at least), so
no lattice edge leads into the slot. A driver parks there by turning the wheel at a stop and moving
short distances at full lock. That manoeuvre is computed backwards, by unparking, the iterative
back-and-forth of Paromtchik & Laugier (1996):

 1. From a parked pose facing east, drive forward at full left lock as far as the clearance allows,
    then backwards at full right lock as far as it allows; repeat.
 2. Before each forward move, try to leave: forward at full left lock up to some heading, then
    forward at full right lock back to east, ending in the aisle above the east block.
 3. Parking is that sequence reversed: each move driven the other way with the same steering.

The mirror image (parked facing west, leaving over the west block) is tried when facing east finds
no way out, e.g. when obstacles sit right above the aisle at the east end. It is computed in a frame
mirrored about the slot's centre line, where it is the same problem; mirroring flips the steering.

A connector, a spiral with straight wheels at both ends (trajgen.py), joins a lattice node, the
entry pose, to where the manoeuvre starts. Reversed, this is the classic parallel park: pull up past
the slot heading east, stop, back in, shuffle. The steering only changes where the car is stopped:
at the cusps, and where the connector meets the first full-lock arc.

Every move is checked against the actual lot with the planner's collision checker (grown
rectangle), at 1 cm steps, so the manoeuvre adapts to each seed's obstacles.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .collision import CollisionChecker
from .lattice import ControlSet, Edge, Primitive, wrap
from .search import Node
from .trajgen import Spiral, solve
from .vehicles import Vehicle

STEP = 0.01             # m, collision samples along a move
MAX_MOVE = 2.0          # m, longest shuffle move
MAX_CYCLES = 30
MAX_TURN = math.radians(80.0)   # give up a shuffle that has turned the car this far
CONNECTOR = 2.0         # m, length of the connector from the entry node
SLACK = 0.005           # m, parked pose kept this far inside the collision margin


@dataclass(frozen=True)
class Move:
    kappa: float       # steering curvature tan(phi) / L; > 0 steers left
    direction: int     # +1 forward, -1 reverse
    length: float      # m


@dataclass
class Maneuver:
    entry: Node              # lattice node where the lattice path ends
    parked: tuple            # final pose (x, y, psi)
    edges: list[Edge]        # connector, then the shuffle, in driving order
    moves: int               # moves in the shuffle (cusps + 1)

    @property
    def length(self) -> float:
        return sum(e.prim.length for e in self.edges)


def advance(pose, kappa: float, direction: int, s):
    """Pose(s) after driving s metres (array or scalar) with fixed steering."""
    x, y, psi = pose
    s = np.asarray(s, float)
    if abs(kappa) < 1e-12:
        return (x + direction * s * math.cos(psi), y + direction * s * math.sin(psi), psi + 0.0 * s)
    p = psi + direction * kappa * s
    return (x + (np.sin(p) - math.sin(psi)) / kappa, y - (np.cos(p) - math.cos(psi)) / kappa, p)


@dataclass(frozen=True)
class Frame:
    """Where the unparking is computed. side +1: the lot itself, parked facing east. side -1: the lot
    mirrored about x = axis (the slot's centre line), so that parked facing west becomes facing
    east. Poses are checked in the lot, so they are mapped back first."""
    side: int
    axis: float

    def to_lot(self, x, y, psi):
        if self.side > 0:
            return x, y, psi
        return 2 * self.axis - np.asarray(x), y, math.pi - np.asarray(psi)


def free_length(checker: CollisionChecker, frame: Frame, pose, kappa: float, direction: int,
                limit: float = MAX_MOVE) -> float:
    """The longest collision-free move (to STEP) with this steering and direction, up to limit."""
    s = np.arange(STEP, limit + STEP / 2, STEP)
    xs, ys, ps = frame.to_lot(*advance(pose, kappa, direction, s))
    hit = np.flatnonzero(checker.hits(np.asarray(xs, float), np.asarray(ys, float), np.asarray(ps, float)))
    if len(hit) == 0:
        return float(s[-1]) if len(s) else 0.0
    return float(s[hit[0] - 1]) if hit[0] > 0 else 0.0


def _move_free(checker, frame: Frame, pose, move: Move) -> bool:
    s = np.arange(STEP, move.length + STEP / 2, STEP)
    if len(s) == 0 or s[-1] < move.length:
        s = np.r_[s, move.length]
    xs, ys, ps = frame.to_lot(*advance(pose, move.kappa, move.direction, s))
    return not checker.hits(np.asarray(xs, float), np.asarray(ys, float), np.asarray(ps, float)).any()


def _after(pose, move: Move):
    x, y, p = advance(pose, move.kappa, move.direction, move.length)
    return (float(x), float(y), float(p))


def _try_exit(checker, frame: Frame, pose, kmax: float):
    """Forward at full left lock to a heading, then full right lock back to east. The first (lowest)
    heading for which both arcs are free."""
    psi = pose[2]
    for target in np.radians(np.arange(5.0, 85.1, 2.5)):
        if target <= psi + 1e-6:
            continue
        up = Move(kmax, +1, (target - psi) / kmax)
        down = Move(-kmax, +1, target / kmax)
        if _move_free(checker, frame, pose, up):
            mid = _after(pose, up)
            if _move_free(checker, frame, mid, down):
                return [up, down]
    return None


def unpark(checker: CollisionChecker, frame: Frame, vehicle: Vehicle, parked) -> list[Move] | None:
    """Moves (in the frame) that take the car from `parked` (facing east in the frame) out to the
    aisle, facing east."""
    kmax = vehicle.max_curvature
    pose = parked
    moves: list[Move] = []
    for _ in range(MAX_CYCLES):
        out = _try_exit(checker, frame, pose, kmax)
        if out is not None:
            return moves + out
        progress = False
        for kappa, direction in ((kmax, +1), (-kmax, -1)):
            s = free_length(checker, frame, pose, kappa, direction)
            if s >= STEP:
                move = Move(kappa, direction, s)
                moves.append(move)
                pose = _after(pose, move)
                progress = True
        if not progress or abs(pose[2]) > MAX_TURN:
            return None
    return None


def _edge(pose, move: Move) -> Edge:
    """A move as an Edge the simulation can drive. Reverse edges follow the lattice convention:
    the spiral describes the path in the direction of travel (heading + pi), and its curvature is
    the heading change per metre travelled, -kappa when reversing."""
    travel = pose[2] + (math.pi if move.direction < 0 else 0.0)
    spiral = Spiral(travel, 0.0, 0.0, 0.0, move.length, k0=move.kappa * move.direction)
    prim = Primitive(-1, -1, 0, 0, "curve", move.length, move.direction * move.kappa * move.length, spiral)
    poses = spiral.poses(0.05)
    return Edge(prim, move.direction < 0, -1, np.empty((0, 3)), poses[:, :2])


def _connector_edge(spiral: Spiral) -> Edge:
    kind = "straight" if spiral.max_abs_kappa() < 1e-12 else "curve"
    prim = Primitive(-1, -1, 0, 0, kind, spiral.length, 0.0, spiral)
    return Edge(prim, False, -1, np.empty((0, 3)), spiral.poses(0.05)[:, :2])


def _unpark_options(checker, frame: Frame, vehicle: Vehicle, goal_box) -> list:
    """Every way out found from parked poses facing east (in the frame) in the goal box, on a 5 cm
    grid: (parked pose, moves), fewest moves (then the shortest) first."""
    x0, x1, y0, y1 = goal_box
    body = vehicle.body
    xs = np.arange(x0 + body.rear + SLACK, x1 - body.front - SLACK + 1e-9, 0.05)
    ys = np.arange(y1 - body.half_width - SLACK, y0 + body.half_width + SLACK - 1e-9, -0.05)
    options = []
    for x, y in ((float(x), float(y)) for y in ys for x in xs):
        parked = (x, y, 0.0)
        if not checker.pose_free(*[float(v) for v in frame.to_lot(*parked)]):
            continue
        moves = unpark(checker, frame, vehicle, parked)
        if moves is None:
            continue
        options.append(((len(moves), sum(m.length for m in moves)), parked, moves))
    options.sort(key=lambda o: o[0])
    return [o[1:] for o in options]


def find_maneuver(checker: CollisionChecker, vehicle: Vehicle, cs: ControlSet,
                  goal_box) -> Maneuver | None:
    """The parking manoeuvre: facing east if possible, else facing west."""
    axis = 0.5 * (goal_box[0] + goal_box[1])
    for frame in (Frame(+1, axis), Frame(-1, axis)):
        # The best way out whose entry pose and connector are also free.
        for parked, moves in _unpark_options(checker, frame, vehicle, goal_box):
            m = _assemble(checker, frame, vehicle, cs, parked, moves)
            if m is not None:
                return m
    return None


def _assemble(checker, frame: Frame, vehicle: Vehicle, cs: ControlSet, parked, moves) -> Maneuver | None:
    # Where the unparking ends (in the frame): in the aisle, facing east.
    pose = parked
    for move in moves:
        pose = _after(pose, move)
    out_x, out_y, out_psi = (float(v) for v in frame.to_lot(*pose))     # in the lot
    heading = float(wrap(out_psi))
    res = cs.resolution
    back_x = out_x - CONNECTOR * math.cos(heading)
    entry = Node(round(back_x / res), round(out_y / res), cs.nearest_bin(heading))
    ex, ey = entry.x * res, entry.y * res
    if not checker.pose_free(ex, ey, heading):
        return None
    connector = solve(out_x - ex, out_y - ey, heading, heading)
    kmax = vehicle.max_curvature
    if connector is None or connector.max_abs_kappa() > kmax:
        return None
    path = connector.poses(STEP)
    if checker.hits(ex + path[:, 0], ey + path[:, 1], path[:, 2]).any():
        return None

    # Driving order: connector, then the unparking moves reversed and driven the other way. In the
    # lot, a mirrored frame flips the steering.
    edges = [_connector_edge(connector)]
    lot_pose = (out_x, out_y, heading)
    for move in reversed(moves):
        back = Move(frame.side * move.kappa, -move.direction, move.length)
        edges.append(_edge(lot_pose, back))
        lot_pose = _after(lot_pose, back)
    parked_lot = tuple(float(v) for v in frame.to_lot(*parked))
    return Maneuver(entry=entry, parked=parked_lot, edges=edges, moves=len(moves))
