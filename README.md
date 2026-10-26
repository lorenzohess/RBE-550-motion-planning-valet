# Valet

RBE-550 assignment 4: park a delivery robot, a car and a truck with a trailer in a 36 × 36 m lot,
with a state-lattice planner. **Status:** the robot parks on seeds 0–5 and the car on seeds 0–5;
the truck parks on seeds 0 and 3 (see `notes.org`, Phase 3 status).

## Setup

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

`ffmpeg` must be on `PATH` for `--record`.

The car's control set is generated once with Nav2's lattice generator and committed in `lattices/`,
so running needs nothing else. Regenerating it (after changing the car or the heading set) needs a
[navigation2](https://github.com/ros-navigation/navigation2) checkout on the path and `rtree`:

```bash
.venv/bin/python -m pip install rtree
echo /path/to/navigation2 > $(.venv/bin/python -c "import site; print(site.getsitepackages()[0])")/nav2.pth
```

## Running

```bash
.venv/bin/python main.py --seed 0                          # window: map, live search, drive
.venv/bin/python main.py --seed 0 --collisions             # start with the collision display on
.venv/bin/python main.py --seed 0 --no-search-view         # plan without drawing, then drive
.venv/bin/python main.py --seed 0 --headless --record runs/robot-seed0.mp4
.venv/bin/python main.py --vehicle car --seed 0                # the car
.venv/bin/python main.py --vehicle truck --seed 3              # the truck and trailer
.venv/bin/python -m pytest -q tests
```

Options: `--vehicle robot|car|truck`, `--seed N`, `--rho 0.10` (obstacle density), `--speed 10|20|30`
(km/h, constant), `--coarse RES` (first-stage lattice resolution, default 1 m; 0 = one fine search
only), `--max-expansions N`, `--weight W` (W > 1: weighted A*, faster, paths up to W times costlier;
plain A* by default).

Keys: `space` pause, `c` collision display, `s` screenshot (PNG in `runs/screenshots/`),
`+` / `-` search pace or playback speed, `esc` quit.

## What you see

1. The lot: 12 × 12 cells of 3 m, tetromino obstacles, the parking area (Fig. 5a), the goal box
   (green) and the start (grey outline, NW corner, facing south). For the car, also the parking
   entry (dashed purple) and the parking shuffle from it (thin purple).
2. The search, as it runs: the search tree, each expanded node drawn with the edge that reached
   it, coloured by heading. With `c`, the window says "Evaluating collisions" and shows the node
   being expanded with its rejected edges in red, each with the rectangle where it collides. The
   first three rejections are held on screen; nothing else accumulates.
3. The drive: the path coloured by gear (forward blue, reverse orange) and the vehicle rectangle
   with its front in yellow, time-stepped at constant speed.

If there is no path, the window says so.

## How it works

- **World** (`valet/world.py`): tetrominoes placed until the density is reached, without wrapping
  around the edges; the SE 6 × 2 cells overwritten with Fig. 5a; the cells under the start cleared.
- **Control set** (`valet/lattice.py`): 0.1 m grid; 36 headings (Nav2's 32 plus ±7.1° about east
  and west). Straights from every heading: one and two grid steps, and about 0.5, 1, 2 and 4 m.
  Reverse edges are the opposite heading's primitives driven backwards.
  - Robot: straights and in-place rotations of one heading bin.
  - Car: straights and turns. Each turn is a cubic curvature spiral (Kelly & Nagy 2003,
    `valet/trajgen.py`) with the wheels straight at both ends, solved by Newton's method so it ends
    exactly on a lattice node, and kept only if its radius is never below ρ_min = 5.61 m. That is
    model 1a (Ackermann geometry): inside front wheel at 30°, front track 60.0 in (M-body Dodge
    Diplomat), ρ_min = L cot 30° + B/2 at the rear axle. Which turns make
    up the set is decided by Nav2's lattice generator, run unmodified with our trajectories in
    place of its arcs (`valet/nav2_spiral.py`).
- **Collision** (`valet/collision.py`): the vehicle rectangle, grown by 5 cm, against occupied cells
  and the lot boundary with the separating axis theorem, at samples no more than 10 cm apart.
  Cells not under the rectangles' extent are skipped.
- **Coarse to fine** (`valet/planning.py`): first a 1 m lattice with Nav2's 16 headings, from the
  start to a handoff near the parking area (on the row 1 to 7 m before the car's entry, or above the
  robot's slot); then the 0.1 m lattice from there. Every coarse node is a fine node, so the fine
  search starts exactly where the coarse path ends. Falls back to one fine search if a stage fails.
- **Goal box**: the parking slot shrunk by 0.5 m all round for the robot, and by 0.3 m at the ends
  and 0.5 m at the sides for the car (it is 5.2 m long in a 6 m slot).
- **Search** (`valet/search.py`): A* from the `astar` package over (x, y, heading), to the goal box
  (robot) or to the parking entry node (car). Costs follow
  Nav2's lattice planner: length; ×1.05 for turns, +0.05 when a turn reverses the previous one's
  direction; 0.5 m per one-bin rotation; ×2 in reverse. Parked = the rectangle inside the goal box,
  heading within 10° of east or west.
- **Heuristic** (`valet/heuristic.py`): a lower bound on the cost to park. Position: grid Dijkstra
  from the goal box around obstacles grown by the reference point's clearance. Heading: the robot
  adds the fewest rotations still needed; the car takes the larger of the position term and
  ρ_min × the heading change still needed.
- **Car parking** (`valet/parking.py`): no lattice edge can enter the 6 m slot (the car has 0.5 m
  of fore-aft room there, and the shortest turn with straight wheels at both ends is 1.06 m). The
  car parks the way a driver does in a tight spot: turning the wheel at a stop, then short moves at
  full lock. The manoeuvre is computed backwards (Paromtchik & Laugier 1996): from a parked pose,
  alternate forward at full left lock and reverse at full right lock, each as far as the clearance
  allows, until the car can pull out into the aisle and straighten to east; parking is that
  sequence reversed. A connector spiral joins it to a lattice node, the entry pose, which is the
  lattice search's goal. If facing east has no way out (obstacles right above the aisle), the
  mirror image, parked facing west, is used. Every move is collision-checked against the actual
  lot.
- **Truck and trailer** (`valet/trailer.py`, `TruckPlanner` in `valet/search.py`): the node adds
  the hitch angle in 5° bins; the trailer is carried along every edge (per-edge tables of the hitch
  angle, RK4 of LaValle Eq. 13.19) and checked against obstacles and the truck. Driving straight is
  the only thing that straightens a trailer (tan(β/2) shrinks as e^(−s/d₁)), and the strip is only
  0.35 m longer than the rig, so the goal test is a straight shot: a node heading east on the parked
  row is a goal if driving straight on parks the rig. One 0.1 m search, weighted ×3, aimed at a point
  16 m west of the parked spot; no coarse stage (the truck's front corner swings low at the end of a
  turn into the bottom row, and the 1 m lattice can't end a turn high enough).
- **Simulation** (`valet/sim.py`): each edge becomes controls, (v, ω) for the robot (Fig. 6b) and
  (v, φ(s) = atan(L κ(s))) for the car (Fig. 7b) and the truck, integrated with RK4 at constant
  speed; the truck adds the trailer heading, ψ̇₁ = (v / d₁) sin(ψ − ψ₁).

## Third-party code and sources

- `astar` 1.1 (PyPI, BSD): the A* search. Its open set is swapped for one with lazy deletion
  (`valet/search.py`, `LazyOpenSet`) because its decrease-key scans the whole open list.
- Pillow: text rendering (this pygame build has no SDL_ttf) and screenshots.
- Nav2 `nav2_smac_planner/lattice_primitives` (Apache-2.0): its lattice generator selects the car's
  control set; the heading discretization and the edge costs follow it.
- Kelly & Nagy (2003), *Reactive nonholonomic trajectory generation via parametric optimal
  control*: the cubic spiral. PythonRobotics' ModelPredictiveTrajectoryGenerator (MIT) for the
  Newton iteration with a finite-difference Jacobian.
- SciPy: the heuristic's distance transform and Dijkstra.
- `valet/recorder.py` is from assignment 2 (Flatland); the tetromino shapes are from the
  obstacleworld library of assignment 1.
- LaValle, *Planning Algorithms*, 13.1.2 (vehicle kinematics); Rill, *Road Vehicle Dynamics*, 9.1.
- Paromtchik & Laugier (1996), *Autonomous parallel parking of a nonholonomic vehicle*: the
  iterative back-and-forth used to compute the car's parking shuffle.

## AI assistance

The code was written with Claude Code (Anthropic, Claude Opus 5.5), directed and reviewed by the
author.
