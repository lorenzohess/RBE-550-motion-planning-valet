"""The control set: motion primitives per heading bin, and the edges the search applies.

Lattice nodes sit on a grid of `resolution` metres with headings from a fixed set. A primitive
starts at a node and ends exactly on another node. An edge is a primitive driven forward, or a
primitive of the opposite heading bin driven backwards (reverse): for a kinematic model, driving
the same steering backwards traces the same curve, so a mirrored primitive is exact.

Heading set: Nav2's discretization (nav2_smac_planner lattice_primitives,
_get_heading_discretization), whose headings point along small integer grid steps (gx, gy) so that
straight primitives land on nodes, plus extra headings near east and west.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .trajgen import Spiral
from .vehicles import Body, Trailer, Vehicle

LATTICE_DIR = Path(__file__).resolve().parent.parent / "lattices"


@dataclass(frozen=True)
class Heading:
    psi: float   # rad, in [-pi, pi)
    gx: int      # smallest grid step along this heading
    gy: int


def heading_set(n_base: int = 32, extra: tuple[tuple[int, int], ...] = ((8, 1),)) -> list[Heading]:
    """Nav2's n_base headings plus, for each (a, b) in `extra`, the four headings along
    (+-a, +-b): with (8, 1) that is +-7.1 deg about east and about west."""
    max_val = n_base // 8
    points = set()
    for i in range(-max_val, max_val + 1):
        points.update({(i, -max_val), (i, max_val), (-max_val, i), (max_val, i)})
    for a, b in extra:
        points.update({(a, b), (a, -b), (-a, b), (-a, -b)})
    steps = {}
    for gx, gy in points:
        g = math.gcd(gx, gy)
        gx, gy = gx // g, gy // g
        psi = math.atan2(gy, gx)
        if psi >= math.pi - 1e-12:
            psi -= 2 * math.pi
        steps[(gx, gy)] = psi
    return [Heading(psi, gx, gy) for (gx, gy), psi in sorted(steps.items(), key=lambda kv: kv[1])]


def wrap(angle):
    """Angle wrapped to [-pi, pi)."""
    return (np.asarray(angle) + np.pi) % (2 * np.pi) - np.pi


@dataclass
class Primitive:
    start_bin: int
    end_bin: int
    dx: int                 # end offset in grid steps
    dy: int
    kind: str               # "straight" | "rotate" | "curve"
    length: float           # path length (m); 0 for a rotation
    turn: float             # heading change (rad), signed; 0 for a straight
    spiral: Spiral | None = None   # curves: the steering profile

    @property
    def left_turn(self) -> bool:
        return self.turn > 0


@dataclass
class Edge:
    prim: Primitive
    reverse: bool
    end_bin: int
    samples: np.ndarray     # (S, 3): x, y offsets (m) from the start node, body heading (rad)
    polyline: np.ndarray    # (n, 2): x, y offsets (m) for drawing

    @property
    def dx(self) -> int:
        return self.prim.dx

    @property
    def dy(self) -> int:
        return self.prim.dy

    @property
    def gear(self) -> str:
        return "reverse" if self.reverse else "forward"


@dataclass
class BinBatch:
    """All edges from one heading bin, their samples concatenated for one collision call."""
    edges: list[Edge]
    x: np.ndarray
    y: np.ndarray
    psi: np.ndarray
    owner: np.ndarray
    starts: np.ndarray
    beta: np.ndarray | None = None   # truck: hitch angle at every sample per start value (trailer.py)

    def ends(self) -> np.ndarray:
        """Index of each edge's last sample."""
        return np.r_[self.starts[1:], len(self.x)] - 1


@dataclass
class ControlSet:
    resolution: float
    headings: list[Heading]
    batches: list[BinBatch] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.headings)

    def opposite(self, k: int) -> int:
        h = self.headings[k]
        return self.index(-h.gx, -h.gy)

    def index(self, gx: int, gy: int) -> int:
        for k, h in enumerate(self.headings):
            if (h.gx, h.gy) == (gx, gy):
                return k
        raise KeyError((gx, gy))

    def nearest_bin(self, psi: float) -> int:
        angles = np.array([h.psi for h in self.headings])
        return int(np.argmin(np.abs(wrap(angles - psi))))


# --- building ----------------------------------------------------------------------------------

def _straight(headings, k, m, res) -> Primitive:
    h = headings[k]
    dx, dy = h.gx * m, h.gy * m
    return Primitive(k, k, dx, dy, "straight", math.hypot(dx, dy) * res, 0.0)


def _rotation(headings, k, step) -> Primitive:
    n = len(headings)
    j = (k + step) % n
    turn = float(wrap(headings[j].psi - headings[k].psi))
    return Primitive(k, j, 0, 0, "rotate", 0.0, turn)


def _samples(prim: Primitive, psi0: float, reverse: bool, res: float, body: Body,
             margin: float, trailer: Trailer | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Collision samples (excluding the start, which is the parent's end) and a drawing polyline.
    Spacing: no point of the body, or of the trailer, moves more than 2 * margin between samples.
    Samples are equally spaced along the path."""
    # A trailer point moves at most trailer.travel_per_metre() per metre, whatever the hitch angle.
    swing = prim.length * trailer.travel_per_metre() if trailer is not None else 0.0
    if prim.kind == "curve":
        # Body points move at most (1 + r kappa) per metre driven.
        travel = max(swing, prim.length * (1.0 + body.max_radius() * prim.spiral.max_abs_kappa()))
        count = max(1, math.ceil(travel / (2.0 * margin)))
        poses = prim.spiral.poses(prim.length / count)[1:]
        poses[-1, :2] = (prim.dx * res, prim.dy * res)
        x, y, psi = poses[:, 0], poses[:, 1], poses[:, 2]
        polyline = prim.spiral.poses(0.25)[:, :2]
        polyline[-1] = (prim.dx * res, prim.dy * res)
    else:
        if prim.kind == "straight":
            travel = max(swing, prim.length)
        else:
            travel = body.max_radius() * abs(prim.turn)
        count = max(1, math.ceil(travel / (2.0 * margin)))
        t = np.arange(1, count + 1) / count
        x = t * prim.dx * res
        y = t * prim.dy * res
        psi = psi0 + t * prim.turn
        polyline = np.array([[0.0, 0.0], [prim.dx * res, prim.dy * res]])
    if reverse:
        psi = psi + math.pi
    samples = np.stack([x, y, wrap(psi)], axis=1)
    return samples, polyline


def _straights(headings, k, resolution, straight_lengths) -> list[Primitive]:
    """One and two grid steps along the heading (the shortest moves that land on a node), plus the
    multiples closest to `straight_lengths`."""
    h = headings[k]
    step = math.hypot(h.gx, h.gy) * resolution
    multiples = sorted({1, 2} | {max(1, round(length / step)) for length in straight_lengths})
    return [_straight(headings, k, m, resolution) for m in multiples]


def _assemble(cs: ControlSet, forward: list[list[Primitive]], body: Body, margin: float,
              trailer: Trailer | None = None) -> ControlSet:
    """Edges per bin: the bin's primitives forward, and the opposite bin's driven backwards.
    Rotations have no reverse; turning the other way is the other rotation."""
    res = cs.resolution
    for k in range(cs.n):
        edges = []
        for prim in forward[k]:
            samples, poly = _samples(prim, cs.headings[k].psi, False, res, body, margin, trailer)
            edges.append(Edge(prim, False, prim.end_bin, samples, poly))
        back = cs.opposite(k)
        for prim in forward[back]:
            if prim.kind == "rotate":
                continue
            samples, poly = _samples(prim, cs.headings[back].psi, True, res, body, margin, trailer)
            edges.append(Edge(prim, True, cs.opposite(prim.end_bin), samples, poly))
        cs.batches.append(_batch(edges))
    return cs


def build_diff_drive(vehicle: Vehicle, resolution: float = 0.1, n_base: int = 32,
                     extra: tuple[tuple[int, int], ...] = ((8, 1),),
                     straight_lengths: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0),
                     margin: float = 0.05) -> ControlSet:
    """Robot control set: in-place rotations of one heading bin either way, and straights."""
    headings = heading_set(n_base, extra)
    cs = ControlSet(resolution, headings)
    forward = [_straights(headings, k, resolution, straight_lengths)
               + [_rotation(headings, k, +1), _rotation(headings, k, -1)]
               for k in range(len(headings))]
    return _assemble(cs, forward, vehicle.body, margin)


def build_car(vehicle: Vehicle, resolution: float = 0.1, n_base: int = 32,
              extra: tuple[tuple[int, int], ...] = ((8, 1),),
              straight_lengths: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0),
              stopping_threshold: int = 10, distinctness: float = 0.9,
              margin: float = 0.05) -> ControlSet:
    """Car and truck control set: spiral turns chosen by Nav2's generator (nav2_spiral.py), cached
    as JSON in lattices/, plus straights. The truck's trailer only changes the sample spacing."""
    headings = heading_set(n_base, extra)
    cs = ControlSet(resolution, headings)
    curves = load_spirals(vehicle, headings, resolution, stopping_threshold, distinctness)
    forward = [_straights(headings, k, resolution, straight_lengths) for k in range(len(headings))]
    for spec in curves:
        sp = Spiral(spec["psi0"], spec["b"], spec["c"], spec["d"], spec["length"])
        turn = float(wrap(headings[spec["end_bin"]].psi - headings[spec["start_bin"]].psi))
        forward[spec["start_bin"]].append(Primitive(spec["start_bin"], spec["end_bin"], spec["dx"],
                                                    spec["dy"], "curve", sp.length, turn, sp))
    cs = _assemble(cs, forward, vehicle.body, margin, vehicle.trailer)
    if vehicle.trailer is not None:
        from . import trailer
        for k, batch in enumerate(cs.batches):
            batch.beta = trailer.batch_table(batch, cs.headings[k].psi, vehicle.trailer.hitch_to_axle)
    return cs


def spiral_file(vehicle: Vehicle, n_headings: int, resolution: float, stopping_threshold: int) -> Path:
    return LATTICE_DIR / (f"{vehicle.name}_res{resolution:g}_h{n_headings}_rho{vehicle.min_turning_radius:.3f}"
                          f"_stop{stopping_threshold}.json")


def load_spirals(vehicle: Vehicle, headings: list[Heading], resolution: float,
                 stopping_threshold: int, distinctness: float) -> list[dict]:
    """The spiral turns of the control set, generated once and cached."""
    path = spiral_file(vehicle, len(headings), resolution, stopping_threshold)
    if path.exists():
        data = json.loads(path.read_text())
        return data["primitives"]
    from . import nav2_spiral   # needs a navigation2 checkout (README)
    cs = ControlSet(resolution, headings)
    output = nav2_spiral.generate([h.psi for h in headings], resolution, vehicle.min_turning_radius,
                                  stopping_threshold, distinctness)
    specs = []
    for trajectories in output.values():
        for tr in trajectories:
            p, sp = tr.parameters, tr.parameters.spiral
            specs.append({"start_bin": cs.nearest_bin(p.start_angle), "end_bin": cs.nearest_bin(p.end_angle),
                          "dx": int(round(p.end_point[0] / resolution)),
                          "dy": int(round(p.end_point[1] / resolution)),
                          "psi0": sp.psi0, "b": sp.b, "c": sp.c, "d": sp.d, "length": sp.length})
    specs.sort(key=lambda d: (d["start_bin"], d["end_bin"], d["dx"], d["dy"]))
    LATTICE_DIR.mkdir(exist_ok=True)
    meta = {"vehicle": vehicle.name, "resolution": resolution,
            "headings_deg": [math.degrees(h.psi) for h in headings],
            "rho_min": vehicle.min_turning_radius, "stopping_threshold": stopping_threshold,
            "trajectory_distinctness_ratio": distinctness,
            "generator": "Nav2 LatticeGenerator selection, Kelly & Nagy cubic spirals (valet/nav2_spiral.py)"}
    path.write_text(json.dumps({"meta": meta, "primitives": specs}, indent=1))
    return specs


def _batch(edges: list[Edge]) -> BinBatch:
    starts, owner, chunks = [], [], []
    total = 0
    for i, e in enumerate(edges):
        starts.append(total)
        owner.append(np.full(len(e.samples), i))
        chunks.append(e.samples)
        total += len(e.samples)
    allsamples = np.concatenate(chunks)
    return BinBatch(edges=edges, x=allsamples[:, 0], y=allsamples[:, 1], psi=allsamples[:, 2],
                    owner=np.concatenate(owner), starts=np.array(starts))


def build(vehicle: Vehicle, **kwargs) -> ControlSet:
    if vehicle.model == "diff":
        return build_diff_drive(vehicle, **kwargs)
    if vehicle.model in ("car", "truck"):
        return build_car(vehicle, **kwargs)
    raise NotImplementedError(f"no control set for {vehicle.name}")
