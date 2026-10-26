"""Nav2's control-set selection, with spiral trajectories from trajgen.py.

Nav2's LatticeGenerator (nav2_smac_planner/lattice_primitives, Apache-2.0, run unmodified) decides
which primitives make up the control set: expanding wavefronts of candidate end poses, pruning any
candidate that existing primitives already cover, and symmetry to fill the other quadrants. Its
trajectory generator is swapped for ours, so every primitive it keeps is a spiral of the vehicle
model. Straights are left out of the selection and added afterwards (see generate_trajectory). Nav2's own arc builder is kept as a quick first filter: if no single arc of radius >= rho_min
joins the two poses, no spiral with zero end curvature does either (it needs more room), so the
spiral solve is skipped.

Importing this module needs a navigation2 checkout on sys.path (README). Only regenerating the
control sets needs it; the generated sets are committed in lattices/.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass

import numpy as np

sys.dont_write_bytecode = True   # do not write __pycache__ into the external checkout
from nav2_smac_planner.lattice_primitives.lattice_generator import LatticeGenerator  # noqa: E402
from nav2_smac_planner.lattice_primitives.trajectory import Path, Trajectory  # noqa: E402
from nav2_smac_planner.lattice_primitives.trajectory_generator import TrajectoryGenerator  # noqa: E402

from .trajgen import Spiral, solve  # noqa: E402


@dataclass(frozen=True)
class SpiralParameters:
    """What Nav2's selection reads from a trajectory (start_angle, end_angle, total_length), plus
    the spiral itself."""
    start_angle: float
    end_angle: float
    end_point: np.ndarray
    total_length: float
    left_turn: bool
    spiral: Spiral


class SpiralTrajectoryGenerator:
    """Stands in for Nav2's TrajectoryGenerator inside its LatticeGenerator."""

    def __init__(self, config: dict, kappa_max: float):
        self.arcs = TrajectoryGenerator(config)
        self.kappa_max = kappa_max

    def generate_trajectory(self, end_point, start_angle, end_angle, primitive_resolution):
        # Straights are added afterwards (lattice.py). In Nav2's pruning a short straight's end pose
        # would remove every turn from that heading: a spiral leaves its start with zero curvature,
        # so it passes over that pose with nearly the same heading and counts as "composable".
        if start_angle == end_angle:
            return None
        if self.arcs._calculate_trajectory_params(end_point, start_angle, end_angle) is None:
            return None
        spiral = solve(float(end_point[0]), float(end_point[1]), start_angle, end_angle)
        if spiral is None or spiral.max_abs_kappa() > self.kappa_max * (1.0 + 1e-9):
            return None
        poses = spiral.poses(primitive_resolution)
        poses[-1, :2] = end_point
        params = SpiralParameters(start_angle, end_angle, np.asarray(end_point, float), spiral.length,
                                  math.remainder(end_angle - start_angle, 2 * math.pi) > 0, spiral)
        return Trajectory(path=Path(poses[:, 0], poses[:, 1], poses[:, 2]), parameters=params)


class SpiralLatticeGenerator(LatticeGenerator):
    """Nav2's generator with our heading set and our trajectories."""

    def __init__(self, config: dict, headings: list[float]):
        self._headings = list(headings)
        self._start_heading = 0.0
        super().__init__(config)
        self.trajectory_generator = SpiralTrajectoryGenerator(config, 1.0 / config["turning_radius"])

    def _get_heading_discretization(self, number_of_headings):
        return list(self._headings)

    def _compute_for_single_heading(self, start_heading):
        self._start_heading = start_heading
        return super()._compute_for_single_heading(start_heading)

    def _compute_min_trajectory_length(self):
        """Where Nav2 starts its wavefronts. Nav2 uses rho_min times the smallest heading change, the
        length of the shortest arc. A spiral with zero end curvature needs at least 1.5 rho_min dpsi
        (its peak curvature is 1.5 times its mean), and the smallest change depends on the heading:
        7.1 deg next to east, 14 deg next to north. Starting earlier only adds empty wavefronts,
        which count towards the stopping threshold, and from north they used it up before the first
        possible turn. The end point of that turn can sit on a closer wavefront (a wavefront is a
        square ring, so a diagonal end point at distance r is on ring r / sqrt(2)), hence the factor."""
        k = self._headings.index(self._start_heading)
        n = len(self._headings)
        gap = min(abs(math.remainder(self._headings[(k + 1) % n] - self._start_heading, 2 * math.pi)),
                  abs(math.remainder(self._headings[k - 1] - self._start_heading, 2 * math.pi)))
        return 1.5 * self.turning_radius * gap / math.sqrt(2)


def nav2_headings(psis: list[float]) -> list[float]:
    """Headings in Nav2's convention: sorted, in (-pi, pi]."""
    return sorted(math.pi if abs(p + math.pi) < 1e-12 else p for p in psis)


def generate(psis: list[float], resolution: float, rho_min: float, stopping_threshold: int = 5,
             distinctness: float = 0.9) -> dict[float, list]:
    """{start heading: [Trajectory]} for the whole heading set."""
    headings = nav2_headings(psis)
    config = {"motion_model": "ackermann", "turning_radius": rho_min, "grid_resolution": resolution,
              "stopping_threshold": stopping_threshold, "num_of_headings": len(headings),
              "trajectory_distinctness_ratio": distinctness}
    return SpiralLatticeGenerator(config, headings).run()
