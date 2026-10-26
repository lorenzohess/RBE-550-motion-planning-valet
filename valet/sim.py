"""Kinematic simulation: the planned edges become controls, integrated with RK4 at constant speed.

Robot (diwheel, Fig. 6b), controls (v, omega): a straight is (v, 0) for length / v seconds, driven
with v < 0 in reverse; an in-place rotation is (0, +-v / d), the wheels turning at the same rim
speed v in opposite directions.

Car (bicycle, Fig. 7b), controls (v, phi): a straight is (v, 0); a curve steers
phi(t) = atan(L kappa(s)) with s = |v| t along its spiral. In reverse the car drives the opposite
bin's primitive backwards (v < 0), and the steering is negated: the heading still changes by
kappa per metre along the path, and psi' = (v / L) tan phi with v < 0.

Truck: the car's controls, with the trailer heading psi1 as a fourth state (Fig. 9).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .lattice import Edge
from .vehicles import Vehicle, bicycle, diff_drive, truck_trailer


@dataclass
class Segment:
    duration: float          # s
    control: object          # function of the time since the segment began -> control tuple
    edge: int                # index of the edge in the plan
    reverse: bool


def _constant(control):
    return lambda tau: control


def _steering(spiral, speed: float, wheelbase: float, sign: float):
    return lambda tau: (sign * speed, sign * math.atan(wheelbase * float(spiral.kappa(speed * tau))))


@dataclass
class Trajectory:
    t: np.ndarray            # (n,) s
    state: np.ndarray        # (n, 3) x, y, psi (psi unwrapped); the truck adds psi1 (n, 4)
    edge: np.ndarray         # (n,) edge index of each sample
    reverse: np.ndarray      # (n,) bool

    @property
    def duration(self) -> float:
        return float(self.t[-1])

    def at(self, time: float) -> tuple[float, float, float, int]:
        """Pose at a time (linear interpolation between steps) and the edge being driven."""
        s, edge = self.state_at(time)
        return float(s[0]), float(s[1]), float(s[2]), edge

    def state_at(self, time: float) -> tuple[np.ndarray, int]:
        """Whole state at a time (linear interpolation between steps) and the edge being driven."""
        time = min(max(time, 0.0), self.duration)
        i = int(np.searchsorted(self.t, time, side="right")) - 1
        i = min(max(i, 0), len(self.t) - 2)
        dt = self.t[i + 1] - self.t[i]
        a = 0.0 if dt <= 0 else (time - self.t[i]) / dt
        return self.state[i] + a * (self.state[i + 1] - self.state[i]), int(self.edge[i + 1])


def kmh(speed_kmh: float) -> float:
    return speed_kmh / 3.6


def segments(vehicle: Vehicle, edges: list[Edge], speed: float) -> list[Segment]:
    """Controls for each edge at constant speed `speed` (m/s)."""
    out = []
    for i, e in enumerate(edges):
        sign = -1.0 if e.reverse else 1.0
        if e.prim.kind == "rotate":
            omega = math.copysign(speed / vehicle.half_track, e.prim.turn)
            out.append(Segment(abs(e.prim.turn) / abs(omega), _constant((0.0, omega)), i, False))
        elif e.prim.kind == "curve":
            out.append(Segment(e.prim.length / speed,
                               _steering(e.prim.spiral, speed, vehicle.wheelbase, sign), i, e.reverse))
        else:
            out.append(Segment(e.prim.length / speed, _constant((sign * speed, 0.0)), i, e.reverse))
    return out


def rk4(f, state: np.ndarray, control, tau: float, h: float) -> np.ndarray:
    """One RK4 step of x' = f(x, u(tau)), the control evaluated at each stage's time."""
    k1 = f(state, control(tau))
    k2 = f(state + 0.5 * h * k1, control(tau + 0.5 * h))
    k3 = f(state + 0.5 * h * k2, control(tau + 0.5 * h))
    k4 = f(state + h * k3, control(tau + h))
    return state + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)


def simulate(vehicle: Vehicle, start: tuple[float, float, float], edges: list[Edge],
             speed: float, dt: float = 0.02, beta0: float = 0.0) -> Trajectory:
    """Time-step the kinematic model through the plan. Each segment is split into equal steps of
    at most dt, so every segment ends exactly at its own end time. beta0: the truck's hitch angle
    at the start."""
    state = np.array(start, dtype=float)
    if vehicle.model == "diff":
        f = diff_drive
    elif vehicle.trailer is not None:
        d1 = vehicle.trailer.hitch_to_axle
        f = lambda state, control: truck_trailer(state, control, vehicle.wheelbase, d1)  # noqa: E731
        state = np.append(state, state[2] - beta0)
    else:
        f = lambda state, control: bicycle(state, control, vehicle.wheelbase)  # noqa: E731
    ts, states, owner, rev = [0.0], [state.copy()], [0], [False]
    t = 0.0
    for seg in segments(vehicle, edges, speed):
        steps = max(1, math.ceil(seg.duration / dt))
        h = seg.duration / steps
        for j in range(steps):
            state = rk4(f, state, seg.control, j * h, h)
            t += h
            ts.append(t)
            states.append(state.copy())
            owner.append(seg.edge)
            rev.append(seg.reverse)
    return Trajectory(np.array(ts), np.array(states), np.array(owner), np.array(rev))
