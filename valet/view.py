"""The pygame window: the lot, the search while it runs, the drive, and a small legend.

Keys: space pause, c collision display, s screenshot, + / - speed, esc quit.
Recording (--record) writes every rendered frame to an mp4.
"""

from __future__ import annotations

import math
import os
import time
from pathlib import Path

import numpy as np
import pygame

from .collision import CollisionChecker
from .lattice import BinBatch, ControlSet
from .recorder import Recorder
from .search import Node, Stop
from .sim import Trajectory
from .vehicles import Body, Vehicle
from .world import CELL_M, CELLS, SIZE_M, World

PX_PER_M = 20
MARGIN = 12
LEGEND_PX = 40
FPS = 30
EXAMPLES = 3            # rejected edges shown slowly each time the collision display is turned on
EXAMPLE_HOLD_S = 1.2

BACKGROUND = (232, 233, 238)
LOT = (250, 250, 252)
GRID = (214, 216, 224)
OBSTACLE = (62, 66, 76)
PARKING = (150, 154, 166)
GOAL = (40, 165, 70)
START = (120, 124, 136)
ENTRY = (150, 110, 200)
VEHICLE = (32, 92, 190)
FRONT = (255, 196, 0)
OUTLINE = (20, 24, 32)
FORWARD = (30, 120, 255)
REVERSE = (240, 120, 20)
REJECT = (215, 35, 35)
TEXT = (30, 32, 40)
MESSAGE_BG = (255, 255, 255, 215)

# Search pace while watching: expansions drawn per frame. None = as fast as the search runs,
# redrawing 30 times a second.
PACES = (1, 4, 16, 64, 256, None)


class Font:
    """Text rendered with Pillow's bundled FreeType font. The pygame builds used here lack SDL_ttf,
    so pygame.font is unavailable. Rendered strings are cached."""

    def __init__(self, size: int):
        from PIL import ImageFont
        self.font = ImageFont.load_default(size=size)
        self._cache: dict[tuple[str, tuple], pygame.Surface] = {}

    def render(self, text: str, _antialias: bool, colour) -> pygame.Surface:
        key = (text, tuple(colour))
        if key not in self._cache:
            from PIL import Image, ImageDraw
            left, top, right, bottom = self.font.getbbox(text)
            image = Image.new("RGBA", (max(1, right), max(1, bottom)), (0, 0, 0, 0))
            ImageDraw.Draw(image).text((0, 0), text, font=self.font, fill=tuple(colour)[:3] + (255,))
            self._cache[key] = pygame.image.frombytes(image.tobytes(), image.size, "RGBA")
        return self._cache[key]


def heading_colour(psi: float) -> pygame.Color:
    colour = pygame.Color(0)
    colour.hsva = (math.degrees(psi) % 360.0, 85, 85, 100)
    return colour


class View:
    def __init__(self, world: World, vehicle: Vehicle, cs: ControlSet, checker: CollisionChecker,
                 speed_kmh: float, record: str | None = None, headless: bool = False,
                 collisions: bool = False, px_per_m: int = PX_PER_M, fps: int = FPS,
                 screenshot_dir: Path = Path("runs/screenshots")):
        self.headless = headless
        if headless:
            os.environ["SDL_VIDEODRIVER"] = "dummy"
        pygame.init()
        pygame.display.set_caption(f"Valet: {vehicle.name}, seed {world.seed}")
        self.world, self.vehicle, self.cs, self.checker = world, vehicle, cs, checker
        self.speed_kmh = speed_kmh
        self.s = px_per_m
        self.fps = fps
        self.map_px = int(SIZE_M * px_per_m)
        width = self.map_px + 2 * MARGIN
        height = self.map_px + 2 * MARGIN + LEGEND_PX
        width += width % 2                       # libx264 with yuv420p needs even sizes
        height += height % 2
        self.size = (width, height)
        self.screen = pygame.display.set_mode(self.size)
        self.font = Font(15)
        self.big = Font(20)
        self.clock = pygame.time.Clock()
        self.recorder = Recorder(record, self.size, fps) if record else None
        self.screenshot_dir = screenshot_dir
        self.screenshots = 0

        self.static = self._draw_static()
        self.search_layer = self.static.copy()   # the map with the search tree drawn so far
        self._edge_px: dict[int, list] = {}
        self.path_layer = pygame.Surface(self.size, pygame.SRCALPHA)

        self.phase = "map"
        self.entry = None
        self.planner = None
        self.paused = False
        self.collisions = collisions
        self.examples_left = EXAMPLES if collisions else 0
        self.pace = len(PACES) - 1
        self.rate = 1.0                          # drive playback rate
        self.message: str | None = None
        self._last_render = 0.0
        self._since_render = 0
        self._last_expansion = None
        self._last_trailer = None

    # --- geometry -----------------------------------------------------------------------------

    def px(self, x, y):
        return (MARGIN + x * self.s, MARGIN + (SIZE_M - y) * self.s)

    def poly(self, corners: np.ndarray) -> list[tuple[float, float]]:
        return [self.px(x, y) for x, y in corners]

    # --- static map -------------------------------------------------------------------------------

    def _draw_static(self) -> pygame.Surface:
        surf = pygame.Surface(self.size)
        surf.fill(BACKGROUND)
        pygame.draw.rect(surf, LOT, (MARGIN, MARGIN, self.map_px, self.map_px))
        for i in range(CELLS + 1):
            a = MARGIN + i * CELL_M * self.s
            pygame.draw.line(surf, GRID, (a, MARGIN), (a, MARGIN + self.map_px))
            pygame.draw.line(surf, GRID, (MARGIN, a), (MARGIN + self.map_px, a))
        for cx, cy in zip(*np.nonzero(self.world.occupied)):
            x0, y1 = self.px(cx * CELL_M, (cy + 1) * CELL_M)
            pygame.draw.rect(surf, OBSTACLE, (x0, y1, CELL_M * self.s + 1, CELL_M * self.s + 1))
        # Parking area (the SE 6 x 2 cells) and the goal box.
        x0, y1 = self.px(18.0, 6.0)
        pygame.draw.rect(surf, PARKING, (x0, y1, 18.0 * self.s, 6.0 * self.s), 1)
        gx0, gx1, gy0, gy1 = self.world.goal_box
        x0, y1 = self.px(gx0, gy1)
        goal = pygame.Rect(x0, y1, (gx1 - gx0) * self.s, (gy1 - gy0) * self.s)
        fill = pygame.Surface(goal.size, pygame.SRCALPHA)
        fill.fill((*GOAL, 45))
        surf.blit(fill, goal.topleft)
        pygame.draw.rect(surf, GOAL, goal, 2)
        for corners in self.vehicle.footprint(*self.vehicle.start):
            pygame.draw.polygon(surf, START, self.poly(corners), 1)
        pygame.draw.rect(surf, OUTLINE, (MARGIN, MARGIN, self.map_px, self.map_px), 1)
        return surf

    # --- drawing helpers --------------------------------------------------------------------------

    def set_parking(self, entry, traj: Trajectory):
        """The car's entry pose (where the lattice path ends) and the parking shuffle from it,
        drawn faintly on the map."""
        self.entry = entry
        for corners in self.vehicle.footprint(*entry):
            self._dashed_polygon(self.static, ENTRY, self.poly(corners))
        points = [self.px(x, y) for x, y in traj.state[:, :2]]
        if len(points) > 1:
            pygame.draw.lines(self.static, ENTRY, False, points, 1)
        self.search_layer = self.static.copy()

    def _dashed_polygon(self, surf, colour, points, dash=5):
        for a, b in zip(points, points[1:] + points[:1]):
            length = math.hypot(b[0] - a[0], b[1] - a[1])
            n = max(1, int(length // dash))
            for i in range(0, n, 2):
                t0, t1 = i / n, min((i + 1) / n, 1.0)
                pygame.draw.line(surf, colour, (a[0] + (b[0] - a[0]) * t0, a[1] + (b[1] - a[1]) * t0),
                                 (a[0] + (b[0] - a[0]) * t1, a[1] + (b[1] - a[1]) * t1), 1)

    def _vehicle(self, surf, x, y, psi, psi1=None):
        """The vehicle with its front in the front colour; the truck also draws its trailer, the
        trailer's front (hitch end) in the front colour and the tongue from the hitch."""
        self._body(surf, self.vehicle.body, x, y, psi)
        trailer = self.vehicle.trailer
        if trailer is not None and psi1 is not None:
            ax, ay = trailer.axle(x, y, psi1)
            self._body(surf, trailer.body, float(ax), float(ay), psi1)
            fx = ax + trailer.body.front * math.cos(psi1)
            fy = ay + trailer.body.front * math.sin(psi1)
            pygame.draw.line(surf, OUTLINE, self.px(x, y), self.px(fx, fy), 2)

    def _body(self, surf, body: Body, x, y, psi):
        pygame.draw.polygon(surf, VEHICLE, self.poly(body.corners(x, y, psi)))
        tip = 0.2 * body.length
        front = Body(rear=-(body.front - tip), front=body.front, half_width=body.half_width)
        pygame.draw.polygon(surf, FRONT, self.poly(front.corners(x, y, psi)))
        pygame.draw.polygon(surf, OUTLINE, self.poly(body.corners(x, y, psi)), 1)

    def _legend(self, surf):
        y = MARGIN + self.map_px + 12
        x = MARGIN
        def item(swatch, label):
            nonlocal x
            swatch(x, y + 7)
            text = self.font.render(label, True, TEXT)
            surf.blit(text, (x + 26, y))
            x += 26 + text.get_width() + 22
        def line(colour, width=3):
            return lambda a, b: pygame.draw.line(surf, colour, (a, b), (a + 20, b), width)
        def box(colour, width):
            return lambda a, b: pygame.draw.rect(surf, colour, (a + 2, b - 6, 16, 12), width)
        def hue_bar(a, b):
            for i in range(20):
                pygame.draw.line(surf, heading_colour(math.radians(18 * i)), (a + i, b - 6), (a + i, b + 6))
        if self.phase == "search":
            item(hue_bar, "search tree, by heading")
            if self.collisions:
                item(line(REJECT), "rejected edge")
        elif self.phase == "drive":
            item(line(FORWARD), "forward")
            item(line(REVERSE), "reverse")
        item(box(GOAL, 2), "goal box")
        item(box(START, 1), "start")
        if self.entry is not None:
            item(box(ENTRY, 1), "parking entry")
        if self.phase == "drive":
            item(lambda a, b: None, f"v = {self.speed_kmh:g} km/h")

    def _message(self, surf, lines: list[str], colour=TEXT):
        texts = [self.big.render(t, True, colour) for t in lines]
        w = max(t.get_width() for t in texts) + 16
        h = sum(t.get_height() + 2 for t in texts) + 12
        box = pygame.Surface((w, h), pygame.SRCALPHA)
        box.fill(MESSAGE_BG)
        left = MARGIN + self.map_px - 8 - w     # top-right: the start is in the top-left corner
        surf.blit(box, (left, MARGIN + 8))
        yy = MARGIN + 14
        for t in texts:
            surf.blit(t, (left + 8, yy))
            yy += t.get_height() + 2

    # --- frames ---------------------------------------------------------------------------------

    def _present(self):
        if not self.headless:
            pygame.display.flip()
        if self.recorder is not None:
            self.recorder.add(self.screen)

    def _pump(self):
        """Handle input. Raises Stop("quit") on esc or window close."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise Stop("quit")
            if event.type != pygame.KEYDOWN:
                continue
            if event.key in (pygame.K_ESCAPE, pygame.K_q):
                raise Stop("quit")
            if event.key == pygame.K_SPACE:
                self.paused = not self.paused
            elif event.key == pygame.K_c:
                self.collisions = not self.collisions
                self.examples_left = EXAMPLES if self.collisions else 0
            elif event.key == pygame.K_s:
                self.screenshot()
            elif event.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                if self.phase == "drive":
                    self.rate = min(self.rate * 2, 16.0)
                else:
                    self.pace = min(self.pace + 1, len(PACES) - 1)
            elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                if self.phase == "drive":
                    self.rate = max(self.rate / 2, 0.125)
                else:
                    self.pace = max(self.pace - 1, 0)

    def screenshot(self):
        from PIL import Image
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)
        self.screenshots += 1
        path = self.screenshot_dir / f"{self.vehicle.name}-seed{self.world.seed}-{self.screenshots:02d}.png"
        Image.frombytes("RGB", self.size, pygame.image.tobytes(self.screen, "RGB")).save(path)
        print(f"screenshot: {path}")

    def hold(self, seconds: float, render):
        """Keep showing `render()` for a while (recorded as that many seconds of video)."""
        if self.headless:
            render()
            for _ in range(int(seconds * self.fps)):
                self._present()
            return
        end = time.perf_counter() + seconds
        while time.perf_counter() < end or self.paused:
            self._pump()
            render()
            self._present()
            self.clock.tick(self.fps)

    def wait_close(self, render):
        """Live window: keep the last frame up until esc or close."""
        if self.headless:
            return
        try:
            while True:
                self._pump()
                render()
                pygame.display.flip()
                self.clock.tick(self.fps)
        except Stop:
            pass

    def close(self):
        if self.recorder is not None:
            self.recorder.close()
            print(f"recorded: {self.recorder.path} ({self.recorder.frames} frames)")
        pygame.quit()

    # --- map --------------------------------------------------------------------------------------

    def render_map(self):
        self.screen.blit(self.static, (0, 0))
        self._legend(self.screen)
        if self.message:
            self._message(self.screen, [self.message])

    def show_map(self, seconds: float = 1.0):
        self.phase = "map"
        self.hold(seconds, self.render_map)

    # --- search (the planner's observer) ---------------------------------------------------------

    def _edge_offsets(self, edge) -> list[tuple[float, float]]:
        """An edge's polyline as pixel offsets from its start node (cached per edge)."""
        key = id(edge)
        if key not in self._edge_px:
            self._edge_px[key] = [(float(dx * self.s), float(-dy * self.s)) for dx, dy in edge.polyline]
        return self._edge_px[key]

    def _edge_points(self, x, y, edge):
        bx, by = self.px(x, y)
        return [(bx + ox, by + oy) for ox, oy in self._edge_offsets(edge)]

    def begin_search(self, planner) -> None:
        """The next expansions come from this planner (coarse or fine lattice)."""
        self.planner = planner

    def expanded(self, node: Node, parent: Node | None, incoming, batch: BinBatch,
                 first: np.ndarray) -> None:
        """Draws the search tree: each expanded node with the edge that reached it."""
        self.phase = "search"
        x, y, psi = self.planner.pose(node)
        if parent is not None and incoming is not None and incoming.prim.kind != "rotate":
            colour = heading_colour(psi)
            px, py, _ = self.planner.pose(parent)
            pygame.draw.lines(self.search_layer, colour, False, self._edge_points(px, py, incoming), 1)
        self._last_expansion = (x, y, psi, batch, first)
        trailer = getattr(self.planner, "last_trailer", None)
        self._last_trailer = None if trailer is None else (self.planner.beta[node], *trailer)

        if self.collisions and self.examples_left > 0 and (first >= 0).any():
            i = int(np.flatnonzero(first >= 0)[0])
            self.examples_left -= 1
            n = EXAMPLES - self.examples_left
            self.hold(EXAMPLE_HOLD_S, lambda: self._render_example(x, y, batch, i, int(first[i]), n))

        self._since_render += 1
        pace = PACES[self.pace]
        now = time.perf_counter()
        if pace is None:
            due = now - self._last_render >= 1.0 / self.fps
        else:
            due = self._since_render >= pace
        if due or self.paused:
            self._pump()
            self.render_search()
            self._present()
            self._last_render = now
            self._since_render = 0
            if pace is not None and not self.headless:
                self.clock.tick(self.fps)
            while self.paused:
                self._pump()
                self.render_search()
                pygame.display.flip()
                self.clock.tick(self.fps)

    def render_search(self, extra_lines: list[str] | None = None, messages: bool = True):
        self.screen.blit(self.search_layer, (0, 0))
        lines = list(extra_lines or [])
        if self.collisions:
            self._draw_checked()
            lines.insert(0, "Evaluating collisions")
        if self._last_expansion is not None:
            x, y = self._last_expansion[:2]
            pygame.draw.circle(self.screen, OUTLINE, self.px(x, y), 3)
        if self.paused:
            lines.append("Paused")
        if self.message:
            lines.append(self.message)
        self._legend(self.screen)
        if lines and messages:
            self._message(self.screen, lines)

    def _hit_body(self, x, y, batch, i, sample):
        """The rectangle that collides on edge i at `sample`: (grown corners, the checker that
        found it, its pose, which body). The trailer's when the trailer hits there first."""
        if self._last_trailer is not None:
            _, ax, ay, psi1, truck, trl = self._last_trailer
            if trl[i] >= 0 and (truck[i] < 0 or trl[i] <= truck[i]):
                k = batch.starts[i] + sample
                checker = self.planner.trailer_checker
                pose = (float(ax[k]), float(ay[k]), float(psi1[k]))
                return checker.body.corners(*pose), checker, pose, "trailer"
        sx, sy, spsi = batch.edges[i].samples[sample]
        pose = (x + sx, y + sy, spsi)
        return self.checker.body.corners(*pose), self.checker, pose, "vehicle"

    def _draw_checked(self):
        """The node just expanded: the vehicle there, and each of its edges that was rejected, in
        red, with the rectangle where it collides. Nothing accumulates."""
        if self._last_expansion is None:
            return
        x, y, psi, batch, first = self._last_expansion
        for corners in self.vehicle.footprint(x, y, psi, self._last_trailer[0] if self._last_trailer else 0.0):
            pygame.draw.polygon(self.screen, OUTLINE, self.poly(corners), 1)
        for i in np.flatnonzero(first >= 0):
            edge = batch.edges[i]
            if edge.prim.kind != "rotate":
                pygame.draw.lines(self.screen, REJECT, False, self._edge_points(x, y, edge), 2)
            corners = self._hit_body(x, y, batch, i, int(first[i]))[0]
            pygame.draw.polygon(self.screen, REJECT, self.poly(corners), 1)

    def _render_example(self, x, y, batch, i, hit, n):
        self.render_search(messages=False)
        edge = batch.edges[i]
        body = self.checker.body
        for sx, sy, spsi in edge.samples[:hit]:
            pygame.draw.polygon(self.screen, START, self.poly(body.corners(x + sx, y + sy, spsi)), 1)
        corners, checker, pose, which = self._hit_body(x, y, batch, i, hit)
        if edge.prim.kind != "rotate":
            pygame.draw.lines(self.screen, REJECT, False, self._edge_points(x, y, edge), 3)
        overlay = pygame.Surface(self.size, pygame.SRCALPHA)
        pygame.draw.polygon(overlay, (*REJECT, 90), self.poly(corners))
        self.screen.blit(overlay, (0, 0))
        pygame.draw.polygon(self.screen, REJECT, self.poly(corners), 2)
        hit_info = checker.describe(*pose)
        for cx, cy in (hit_info.cells if hit_info else []):
            x0, y1 = self.px(cx * CELL_M, (cy + 1) * CELL_M)
            pygame.draw.rect(self.screen, REJECT, (x0, y1, CELL_M * self.s, CELL_M * self.s), 3)
        if hit_info is None:
            where = "hits the truck"
        elif not hit_info.cells:
            where = "leaves the lot"
        else:
            where = "hits an obstacle"
        self._message(self.screen, ["Evaluating collisions",
                                    f"Edge rejected: the {which} {where} ({n} of {EXAMPLES})"], REJECT)

    def show_no_path(self, status: str):
        self.message = "No path" if status == "no_path" else f"Stopped: {status}"
        self.hold(3.0, lambda: self.render_search())
        self.wait_close(lambda: self.render_search())

    # --- drive ------------------------------------------------------------------------------------

    def drive(self, traj: Trajectory, edges):
        self.phase = "drive"
        self.message = None
        self.collisions = False
        self._draw_path(edges, traj)
        t = 0.0
        while True:
            self._pump()
            if not self.paused:
                t = min(t + self.rate / self.fps, traj.duration)
            self.render_drive(traj, t)
            self._present()
            if not self.headless:
                self.clock.tick(self.fps)
            if t >= traj.duration and not self.paused:
                break
        self.message = "Parked"
        self.hold(2.0, lambda: self.render_drive(traj, traj.duration))
        self.wait_close(lambda: self.render_drive(traj, traj.duration))

    def _draw_path(self, edges, traj: Trajectory):
        """The planned path, coloured by gear, from the simulated reference-point track."""
        for i, edge in enumerate(edges):
            if edge.prim.kind == "rotate":
                continue
            mask = traj.edge == i
            idx = np.flatnonzero(mask)
            idx = np.r_[idx[0] - 1, idx] if idx[0] > 0 else idx
            points = [self.px(x, y) for x, y in traj.state[idx, :2]]
            if len(points) > 1:
                pygame.draw.lines(self.path_layer, REVERSE if edge.reverse else FORWARD, False, points, 2)

    def render_drive(self, traj: Trajectory, t: float):
        self.screen.blit(self.static, (0, 0))
        self.screen.blit(self.path_layer, (0, 0))
        state, _ = traj.state_at(t)
        self._vehicle(self.screen, *(float(v) for v in state))
        self._legend(self.screen)
        lines = []
        if self.paused:
            lines.append("Paused")
        if self.message:
            lines.append(self.message)
        if lines:
            self._message(self.screen, lines)
