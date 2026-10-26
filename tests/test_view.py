"""Headless smoke test of the window, the search drawing and the recorder."""

import os
import shutil

import pytest

os.environ["SDL_VIDEODRIVER"] = "dummy"

from valet import lattice  # noqa: E402
from valet.collision import CollisionChecker  # noqa: E402
from valet.search import LatticePlanner  # noqa: E402
from valet.sim import kmh, simulate  # noqa: E402
from valet.vehicles import ROBOT  # noqa: E402
from valet.world import build_world  # noqa: E402


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_headless_run_records(tmp_path):
    from valet.view import View
    world = build_world(0, 0.0, "5a", ROBOT.footprint(*ROBOT.start))
    cs = lattice.build_diff_drive(ROBOT, resolution=0.5)
    checker = CollisionChecker(world, ROBOT.body)
    out = tmp_path / "run.mp4"
    view = View(world, ROBOT, cs, checker, 10.0, record=str(out), headless=True, collisions=True,
                screenshot_dir=tmp_path)
    try:
        view.show_map(0.1)
        p = LatticePlanner(world, ROBOT, cs, checker, observer=view)
        view.begin_search(p)
        plan = p.plan()
        assert plan.status == "found"
        view.screenshot()
        traj = simulate(ROBOT, p.pose(p.start), plan.edges, kmh(30.0))
        view.drive(traj, plan.edges)
    finally:
        view.close()
    assert out.stat().st_size > 0
    assert list(tmp_path.glob("*.png"))


def test_truck_frames(tmp_path):
    # The trailer is drawn while searching, in the collision display and driving.
    import math
    from valet.search import TruckPlanner
    from valet.vehicles import TRUCK
    from valet.view import View
    world = build_world(0, 0.0, "5b", TRUCK.footprint(*TRUCK.start), TRUCK.goal_inset)
    cs = lattice.build(TRUCK)
    checker = CollisionChecker(world, TRUCK.body)
    view = View(world, TRUCK, cs, checker, 30.0, headless=True, collisions=True, screenshot_dir=tmp_path)
    try:
        p = TruckPlanner(world, TRUCK, cs, checker, observer=view, start=(12.5, 1.5, 0.0),
                         beta0=math.radians(20.0))
        view.begin_search(p)
        plan = p.plan()
        assert plan.status == "found"
        traj = simulate(TRUCK, (12.5, 1.5, 0.0), plan.edges, kmh(30.0), beta0=math.radians(20.0))
        view.drive(traj, plan.edges)
        view.screenshot()
    finally:
        view.close()
    assert list(tmp_path.glob("*.png"))
