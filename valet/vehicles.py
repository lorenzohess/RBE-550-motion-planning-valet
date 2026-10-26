"""Vehicle geometry and kinematic models.

Each vehicle has a reference point whose position is (x, y) and a heading psi (rad, east = 0,
counter-clockwise):
- robot: the centre of the wheel axle, which is also the centre of the body (Fig. 6b);
- car and truck: the centre of the rear axle (Fig. 7b). The truck's trailer is hitched there
  (Fig. 9, LaValle Fig. 13.13); its own reference point is the centre of its axle, d1 behind the
  hitch, with heading psi1. The hitch angle is beta = psi - psi1.

A body is a rectangle in the vehicle frame (x forward from the reference point, y to the left):
x in [-rear, front], y in [-half_width, half_width]. "rear" and "front" are distances from the
reference point to the rear and front of the body; for the car, front = wheelbase + front overhang
and rear = rear overhang.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Body:
    rear: float
    front: float
    half_width: float

    @property
    def length(self) -> float:
        return self.rear + self.front

    def corners(self, x, y, psi) -> np.ndarray:
        """Corners at one or many poses: (..., 4, 2), ordered rear-right, front-right, front-left,
        rear-left. x, y, psi broadcast against each other."""
        x, y, psi = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float), np.asarray(psi, float))
        local = np.array([[-self.rear, -self.half_width], [self.front, -self.half_width],
                          [self.front, self.half_width], [-self.rear, self.half_width]])
        c, s = np.cos(psi)[..., None], np.sin(psi)[..., None]
        cx = x[..., None] + local[:, 0] * c - local[:, 1] * s
        cy = y[..., None] + local[:, 0] * s + local[:, 1] * c
        return np.stack([cx, cy], axis=-1)

    def max_radius(self) -> float:
        """Distance from the reference point to the farthest corner."""
        return math.hypot(max(self.rear, self.front), self.half_width)


@dataclass(frozen=True)
class Trailer:
    body: Body              # about the trailer's axle centre
    hitch_to_axle: float    # d1 in Fig. 9 (m)

    def axle(self, x, y, psi1):
        """Axle centre of the trailer hitched at (x, y) with heading psi1."""
        return x - self.hitch_to_axle * np.cos(psi1), y - self.hitch_to_axle * np.sin(psi1)

    def reach(self) -> float:
        """Distance from the hitch to the trailer's farthest corner."""
        b = self.body
        return math.hypot(self.hitch_to_axle + b.rear, b.half_width)

    def travel_per_metre(self) -> float:
        """Most any trailer point moves per metre the hitch moves: the hitch moves 1 m and the
        trailer turns by at most 1 / d1 (|psi1'| = |v sin beta| / d1)."""
        return 1.0 + self.reach() / self.hitch_to_axle


@dataclass(frozen=True)
class Vehicle:
    name: str
    body: Body
    model: str                       # "diff" (diwheel, Fig. 6b), "car" (Ackermann bicycle, Fig. 7b)
                                     # or "truck" (the bicycle pulling a trailer, Fig. 9)
    start: tuple[float, float, float]   # x, y (m), psi (rad) of the reference point
    layout: str                      # parking layout, assignment Fig. 5a or 5b
    goal_headings: tuple[float, ...]    # parked headings allowed (rad)
    heading_tolerance: float = math.radians(10.0)
    half_track: float | None = None  # diff: d in Fig. 6b, centre of axle to each wheel (m)
    wheelbase: float | None = None   # car, truck: L in Fig. 7b (m)
    max_steer: float | None = None   # car, truck: steering limit of the inside front wheel (rad)
    track_width: float | None = None   # car, truck: front track B (m), for model 1a
    trailer: Trailer | None = None
    goal_inset: tuple[float, float] = (0.0, 0.0)   # goal box = parking slot shrunk by this much
                                                   # at each end and at each side (m)

    def footprint(self, x, y, psi, beta=0.0) -> list[np.ndarray]:
        """Corners of every body at a pose: the vehicle, then the trailer at hitch angle beta."""
        bodies = [self.body.corners(x, y, psi)]
        if self.trailer is not None:
            psi1 = np.asarray(psi) - beta
            bodies.append(self.trailer.body.corners(*self.trailer.axle(x, y, psi1), psi1))
        return bodies

    @property
    def min_turning_radius(self) -> float:
        """Model 1a (Ackermann geometry, Wong Fig. 5.2; vehicle-dynamics slides): the tightest turn is
        when the inside front wheel reaches its limit. The rear axle's centre then turns on
        rho_min = L cot(delta_i,max) + B / 2. (The slides give the radius at the CG,
        sqrt((L cot delta_i + B/2)^2 + l2^2); ours is the same geometry at the rear axle, the
        reference point.) Without a track width this is the bicycle form L / tan(phi_max)."""
        if self.wheelbase is None:
            return 0.0
        return self.wheelbase / math.tan(self.max_steer) + 0.5 * (self.track_width or 0.0)

    @property
    def max_curvature(self) -> float:
        return 1.0 / self.min_turning_radius


# --- kinematic models -----------------------------------------------------------------------

def diff_drive(state: np.ndarray, control: tuple[float, float]) -> np.ndarray:
    """Diwheel (Fig. 6b, LaValle Eq. 13.16) written with v = r (u_l + u_r) / 2 and
    omega = r (u_r - u_l) / (2 d): x' = v cos psi, y' = v sin psi, psi' = omega."""
    v, omega = control
    return np.array([v * math.cos(state[2]), v * math.sin(state[2]), omega])


def bicycle(state: np.ndarray, control: tuple[float, float], wheelbase: float) -> np.ndarray:
    """Ackermann bicycle (Fig. 7b, LaValle 13.1.2.1): x' = v cos psi, y' = v sin psi,
    psi' = (v / L) tan phi."""
    v, phi = control
    return np.array([v * math.cos(state[2]), v * math.sin(state[2]), v / wheelbase * math.tan(phi)])


def truck_trailer(state: np.ndarray, control: tuple[float, float], wheelbase: float,
                  hitch_to_axle: float) -> np.ndarray:
    """The bicycle pulling one trailer hitched at its rear axle (Fig. 9; LaValle Eq. 13.19 with
    k = 1; Rill Eq. 9.38 with b = 0). State x, y, psi (truck), psi1 (trailer):
    psi1' = (v / d1) sin(psi - psi1). The trailer does not change the truck's motion."""
    v, phi = control
    return np.array([v * math.cos(state[2]), v * math.sin(state[2]), v / wheelbase * math.tan(phi),
                     v / hitch_to_axle * math.sin(state[2] - state[3])])


# --- the vehicles -----------------------------------------------------------------------------

CAR_GOAL_INSET = (0.3, 0.5)   # the car is 5.2 m in a 6 m slot, so it can't be 0.5 m in at both ends

# Delivery robot, 0.7 x 0.57 m. The axle is assumed at the centre of the body, and the wheels
# 0.25 m either side of it (d in Fig. 6b, not given).
ROBOT = Vehicle(
    name="robot",
    body=Body(rear=0.35, front=0.35, half_width=0.285),
    model="diff",
    half_track=0.25,
    # Fig. 4: col 0, straddling rows 0 and 1, facing south.
    start=(1.5, 33.0, -math.pi / 2),
    layout="5a",
    goal_headings=(0.0, math.pi),
    goal_inset=(0.5, 0.5),
)

# Car, 5.2 x 1.8 m, L = 2.8 m. Front overhang 1.0 m (Rill Fig. 9.3, typical passenger car), so the
# rear overhang is 5.2 - 2.8 - 1.0 = 1.4 m. The inside wheel's 30 deg limit is an assumption; the
# front track is the M-body Diplomat's, 60.0 in. Model 1a: rho_min = 2.8 cot 30 + 0.762 = 5.61 m.
CAR = Vehicle(
    name="car",
    body=Body(rear=1.4, front=2.8 + 1.0, half_width=0.9),
    model="car",
    wheelbase=2.8,
    max_steer=math.radians(30.0),
    track_width=60.0 * 0.0254,
    goal_inset=CAR_GOAL_INSET,
    # Body centred on the two start cells (y 30-36): the body centre is 1.2 m ahead of the rear axle.
    start=(1.5, 34.2, -math.pi / 2),
    layout="5a",
    goal_headings=(0.0, math.pi),
)

# Truck, 5.4 x 2.0 m, L = 3.4 m, front overhang 1.0 m as for the car, so the rear overhang is
# 1.0 m. Inside wheel 30 deg as for the car; the front track is the Holden Rodeo TF's (the Fig. 8
# truck), 1390 mm. Model 1a: rho_min = 3.4 cot 30 + 0.695 = 6.58 m.
# Trailer 4.5 x 2.0 m, axle at its midpoint, d1 = 5.0 m from the hitch. The rig is
# 4.4 + 5.0 + 2.25 = 11.65 m long.
TRUCK = Vehicle(
    name="truck",
    body=Body(rear=1.0, front=3.4 + 1.0, half_width=1.0),
    model="truck",
    wheelbase=3.4,
    max_steer=math.radians(30.0),
    track_width=1.390,
    trailer=Trailer(body=Body(rear=2.25, front=2.25, half_width=1.0), hitch_to_axle=5.0),
    goal_inset=(0.1, 0.25),
    # Column 0 facing south, rig straight, centred on the four cells y 24-36: truck front at
    # y = 24.2, trailer rear (to the north) at y = 35.85.
    start=(1.5, 28.6, -math.pi / 2),
    layout="5b",
    goal_headings=(0.0,),
)

VEHICLES = {"robot": ROBOT, "car": CAR, "truck": TRUCK}
