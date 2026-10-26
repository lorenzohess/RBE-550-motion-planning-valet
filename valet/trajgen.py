"""Car motion primitives from the vehicle model: cubic curvature spirals (Kelly & Nagy 2003).

A primitive starts at the origin with heading psi_s and must end exactly at (x_f, y_f, psi_f), with
the wheels straight at both ends, so that steering is continuous across lattice nodes without
putting curvature in the search state. Along the primitive

    kappa(s) = b s + c s^2 + d s^3          (kappa(0) = 0)
    psi(s)   = psi_s + b s^2/2 + c s^3/3 + d s^4/4
    x(s), y(s) = integrals of cos psi, sin psi

and the steering angle is phi(s) = atan(L kappa(s)) (bicycle model, Fig. 7b). The end conditions
kappa(s_f) = 0 and psi(s_f) = psi_f fix b and d given (c, s_f):

    d = -(b + c s_f) / s_f^2,   b = 4 dpsi / s_f^2 - c s_f / 3

which leaves two unknowns (c, s_f) for the two position conditions, solved by Newton's method with
a finite-difference Jacobian, as in PythonRobotics' ModelPredictiveTrajectoryGenerator (MIT). A
primitive is kept only if |kappa(s)| <= 1 / rho_min everywhere.

Which primitives make up the control set is decided by Nav2's lattice generator; see
nav2_spiral.py.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# 24-point Gauss-Legendre rule on [0, 1] for the endpoint integrals.
_GL_X, _GL_W = np.polynomial.legendre.leggauss(24)
_GL_U = 0.5 * (_GL_X + 1.0)
_GL_W = 0.5 * _GL_W


@dataclass(frozen=True)
class Spiral:
    psi0: float     # start heading (rad)
    b: float
    c: float
    d: float
    length: float   # s_f (m)
    k0: float = 0.0  # curvature at the start: 0 for lattice primitives; a full-lock arc is k0 alone

    def kappa(self, s):
        s = np.asarray(s, float)
        return self.k0 + s * (self.b + s * (self.c + s * self.d))

    def heading(self, s):
        s = np.asarray(s, float)
        return self.psi0 + s * (self.k0 + s * (self.b / 2 + s * (self.c / 3 + s * self.d / 4)))

    def poses(self, step: float) -> np.ndarray:
        """(n, 3) x, y, psi from s = 0 to s_f, at most `step` apart, the end included."""
        n = max(2, math.ceil(self.length / step) + 1)
        s = np.linspace(0.0, self.length, n)
        fine = np.linspace(0.0, self.length, 8 * (n - 1) + 1)
        psi = self.heading(fine)
        ds = np.diff(fine)
        x = np.r_[0.0, np.cumsum(0.5 * (np.cos(psi[1:]) + np.cos(psi[:-1])) * ds)]
        y = np.r_[0.0, np.cumsum(0.5 * (np.sin(psi[1:]) + np.sin(psi[:-1])) * ds)]
        return np.stack([x[::8], y[::8], self.heading(s)], axis=1)

    def max_abs_kappa(self) -> float:
        return float(np.abs(self.kappa(np.linspace(0.0, self.length, 201))).max())


def _coefficients(dpsi: float, c: float, sf: float) -> tuple[float, float]:
    b = 4.0 * dpsi / sf**2 - c * sf / 3.0
    d = -(b + c * sf) / sf**2
    return b, d


def _endpoint(dpsi: float, c: float, sf: float) -> np.ndarray:
    """End position in the start frame (start heading 0)."""
    b, d = _coefficients(dpsi, c, sf)
    s = sf * _GL_U
    psi = s * s * (b / 2 + s * (c / 3 + s * d / 4))
    return sf * np.array([_GL_W @ np.cos(psi), _GL_W @ np.sin(psi)])


def solve(x: float, y: float, psi0: float, psi1: float, tol: float = 1e-7,
          max_iter: int = 40) -> Spiral | None:
    """Spiral from (0, 0, psi0) to (x, y, psi1) with zero curvature at both ends, or None."""
    dpsi = math.remainder(psi1 - psi0, 2 * math.pi)
    c0, s0 = math.cos(psi0), math.sin(psi0)
    target = np.array([c0 * x + s0 * y, -s0 * x + c0 * y])     # in the start frame
    chord = float(np.hypot(*target))
    if chord < 1e-9:
        return None
    sf = chord * (1.0 + dpsi * dpsi / 24.0)
    c = -6.0 * dpsi / sf**3                                    # symmetric profile: kappa ~ s (sf - s)
    for _ in range(max_iter):
        r = _endpoint(dpsi, c, sf) - target
        if float(np.hypot(*r)) < tol:
            b, d = _coefficients(dpsi, c, sf)
            return Spiral(psi0, b, c, d, sf)
        hc = 1e-6 * max(1.0, abs(c))
        hs = 1e-6 * sf
        jac = np.column_stack([(_endpoint(dpsi, c + hc, sf) - target - r) / hc,
                               (_endpoint(dpsi, c, sf + hs) - target - r) / hs])
        try:
            step = np.linalg.solve(jac, -r)
        except np.linalg.LinAlgError:
            return None
        # Damped: never shrink s_f by more than half in one step.
        scale = 1.0
        if sf + step[1] < 0.5 * sf:
            scale = 0.5 * sf / -step[1]
        c += scale * step[0]
        sf += scale * step[1]
        if not math.isfinite(c) or sf <= 0.0 or sf > 4.0 * chord:
            return None
    return None
