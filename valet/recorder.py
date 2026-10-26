#!/usr/bin/env python3
"""Frame capture to mp4 via ffmpeg (from assignment 2, Flatland: flatland/render/recorder.py).

This pygame build has no SDL_image, so ``pygame.image.save`` cannot write PNGs.
Raw RGB frames are piped straight into ffmpeg instead, which avoids intermediate
files entirely and is faster than writing them.
"""

import shutil
import subprocess
from pathlib import Path

import pygame

DEFAULT_FPS = 10
HOLD_SECONDS = 2.0


class Recorder:
    """Writes an H.264 mp4. Usable as a context manager."""

    def __init__(self, path, size, fps=DEFAULT_FPS):
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("ffmpeg not found on PATH; cannot record")

        # ffmpeg will not create intermediate directories, and it fails at open
        # time -- which we would not notice until close(), after the whole run.
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        self.path = str(path)
        self.width, self.height = size
        self.fps = fps
        self.frames = 0
        self._last = None

        self._process = subprocess.Popen(
            [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-f", "rawvideo", "-pixel_format", "rgb24",
                "-video_size", f"{self.width}x{self.height}",
                "-framerate", str(fps),
                "-i", "-",
                "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                "-pix_fmt", "yuv420p",
                self.path,
            ],
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def add(self, surface):
        """Append one frame."""
        if self._process.stdin is None or self._process.stdin.closed:
            return
        if surface.get_size() != (self.width, self.height):
            raise ValueError(
                f"frame size {surface.get_size()} != recorder size "
                f"{(self.width, self.height)}"
            )
        self._last = pygame.image.tobytes(surface, "RGB")
        self._process.stdin.write(self._last)
        self.frames += 1

    def hold(self, seconds=HOLD_SECONDS):
        """Repeat the final frame so the outcome is readable before the cut."""
        if self._last is None:
            return
        for _ in range(int(self.fps * seconds)):
            if self._process.stdin is None or self._process.stdin.closed:
                return
            self._process.stdin.write(self._last)
            self.frames += 1

    def close(self):
        """Flush and finalize. Safe to call more than once."""
        if self._process is None:
            return
        try:
            if self._process.stdin and not self._process.stdin.closed:
                self._process.stdin.close()
        except BrokenPipeError:
            pass
        _, stderr = self._process.communicate()
        code = self._process.returncode
        self._process = None
        if code != 0:
            raise RuntimeError(
                f"ffmpeg exited {code}: {stderr.decode(errors='replace').strip()}"
            )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        # Hold only on a clean finish; on an exception just flush what we have.
        if exc_type is None:
            self.hold()
        self.close()
        return False
