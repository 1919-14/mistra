"""
Renders the annotated frame directly to a monitor attached to the Pi's
own HDMI output. This is the only display path in the standalone
pipeline -- no laptop, no network socket, no JPEG encode/decode. That
overhead is gone entirely, which is the whole point: it frees CPU
budget on the Pi for inference instead.

Requires a GUI-capable OpenCV build (opencv-python, not -headless) and
a display to draw to -- a desktop/X session or a KMS/DRM console on
the Pi. If cv2.imshow can't open a window (e.g. a pure SSH session
with no display attached), construction raises immediately with a
clear message, rather than silently running with nothing on screen.
"""
from __future__ import annotations

import os
import time

import cv2
import numpy as np


def _has_display_target() -> bool:
    """Best-effort check for a GUI target before touching cv2 at all.

    This matters because OpenCV's Qt backend doesn't raise a catchable
    Python exception when there's no display -- it calls abort() and
    kills the whole process (SIGABRT), no matter what's wrapped in a
    try/except. Checking the environment first lets us fail with a
    clear message instead of a hard process crash.
    """
    import sys
    if sys.platform in ("win32", "darwin"):
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


class Display:
    def __init__(self, window_name: str = "MISTRA", fullscreen: bool = True):
        if not _has_display_target():
            raise RuntimeError(
                "no display target found (neither $DISPLAY nor "
                "$WAYLAND_DISPLAY is set); is a monitor plugged into the "
                "Pi's HDMI port, with a desktop/X session or Wayland "
                "compositor running? (If you're SSH'd in, this session "
                "has no GUI access to the Pi's own screen -- run this "
                "directly on the Pi's console/desktop instead, or "
                "SSH in with X forwarding as a fallback.)"
            )

        self.window_name = window_name
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        if fullscreen:
            cv2.setWindowProperty(
                self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
            )
        cv2.imshow(self.window_name, np.zeros((2, 2, 3), dtype=np.uint8))
        cv2.waitKey(1)

        self._last_fps_time = time.time()
        self._frame_count = 0
        self.display_fps = 0.0

    def show(self, image: np.ndarray, hud_lines: list[str]) -> bool:
        """Draws the HUD onto a copy of `image` and displays it.
        Returns False if the user pressed 'q' to quit."""
        self._frame_count += 1
        now = time.time()
        if now - self._last_fps_time >= 1.0:
            self.display_fps = self._frame_count / (now - self._last_fps_time)
            self._frame_count = 0
            self._last_fps_time = now

        frame = image.copy()
        lines = [f"display fps: {self.display_fps:.1f}"] + hud_lines
        for i, line in enumerate(lines):
            y = 20 + i * 20
            cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 1, cv2.LINE_AA)

        cv2.imshow(self.window_name, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            return False
        if key == ord("s"):
            fname = f"snapshot_{int(time.time())}.jpg"
            cv2.imwrite(fname, frame)
            print(f"[display] saved {fname}")
        return True

    def close(self) -> None:
        cv2.destroyWindow(self.window_name)
