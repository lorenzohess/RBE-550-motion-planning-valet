"""The trailer along a lattice edge.

Along an edge the truck's motion is fixed by the primitive. The trailer heading psi1 follows from
LaValle Eq. 13.19 (Rill Eq. 9.38 with b = 0), written per metre driven:

    d psi1 / ds = sigma sin(psi - psi1) / d1

with s the distance driven, sigma = +1 forward and -1 in reverse (v < 0), and psi the truck heading.
So the hitch angle beta = psi - psi1 along an edge depends only on the edge and on its value at the
start. On a straight, tan(beta / 2) = tan(beta0 / 2) exp(-sigma s / d1): forward the trailer lines
up behind the truck; in reverse it folds. Held on a circle of radius rho, beta settles at
asin(d1 / rho) (Rill Eq. 9.40).

For the search, every edge gets a table, computed once with the control set: beta at each of the
edge's collision samples, for start values beta0 on a 1 deg grid (RK4 from sample to sample, the
truck heading linear in between). An expansion interpolates the table at the node's exact beta.
Beta is kept unwrapped (continuous along the edge), so that interpolating never mixes angles on
either side of +-180 deg.
"""

from __future__ import annotations

import math

import numpy as np

BETA_STEP = math.radians(1.0)
BETA_MAX = math.radians(150.0)   # table range; beyond the truck-trailer contact (about 140 deg)
BETA_GRID = np.arange(-BETA_MAX, BETA_MAX + 0.5 * BETA_STEP, BETA_STEP)


def integrate(psi_start: float, psi: np.ndarray, ds: float, sigma: float, beta0, d1: float) -> np.ndarray:
    """Hitch angle at each sample of an edge.

    psi_start: truck heading at the start; psi (S,): truck heading at the samples, ds apart (m);
    sigma: +1 forward, -1 reverse; beta0: start value(s), shape (B,). Returns (B, S), unwrapped."""
    beta0 = np.atleast_1d(np.asarray(beta0, float))
    heading = np.unwrap(np.concatenate([[psi_start], psi]))
    psi1 = heading[0] - beta0
    k = sigma / d1
    out = np.empty((len(beta0), len(psi)))
    for i in range(len(psi)):
        a, b = heading[i], heading[i + 1]
        m = 0.5 * (a + b)
        k1 = k * np.sin(a - psi1)
        k2 = k * np.sin(m - psi1 - 0.5 * ds * k1)
        k3 = k * np.sin(m - psi1 - 0.5 * ds * k2)
        k4 = k * np.sin(b - psi1 - ds * k3)
        psi1 = psi1 + ds / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        out[:, i] = b - psi1
    return out


def edge_table(edge, psi_start: float, d1: float) -> np.ndarray:
    """(len(BETA_GRID), S) float32: beta at the edge's samples for each grid start value. The
    samples are equally spaced along the path (lattice._samples)."""
    ds = edge.prim.length / len(edge.samples)
    sigma = -1.0 if edge.reverse else 1.0
    return integrate(psi_start, edge.samples[:, 2], ds, sigma, BETA_GRID, d1).astype(np.float32)


def batch_table(batch, psi_start: float, d1: float) -> np.ndarray:
    """The tables of all edges of a heading bin, side by side, aligned with the batch's samples."""
    return np.concatenate([edge_table(e, psi_start, d1) for e in batch.edges], axis=1)


def lookup(table: np.ndarray, beta0: float) -> np.ndarray:
    """Beta at every column of a table for start value beta0 (linear in beta0)."""
    u = (min(max(beta0, -BETA_MAX), BETA_MAX) + BETA_MAX) / BETA_STEP
    i = min(int(u), len(BETA_GRID) - 2)
    w = u - i
    return (1.0 - w) * table[i] + w * table[i + 1]


def replay(edges, psi_start: float, beta0: float, d1: float) -> list[float]:
    """Exact beta at the end of each edge of a path (RK4, no table)."""
    out = []
    beta, psi = beta0, psi_start
    for e in edges:
        ds = e.prim.length / len(e.samples)
        sigma = -1.0 if e.reverse else 1.0
        beta = float(integrate(psi, e.samples[:, 2], ds, sigma, beta, d1)[0, -1])
        beta = math.remainder(beta, 2 * math.pi)
        psi = float(e.samples[-1, 2])
        out.append(beta)
    return out
