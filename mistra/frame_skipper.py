"""
Decides, frame by frame, whether the Pi should run full YOLO inference or
reuse the previous detections -- and at what resolution -- so the
pipeline stays close to real-time on modest edge hardware (a Raspberry
Pi CPU) instead of falling further and further behind.

Strategy (simple additive/multiplicative controller, similar in spirit to
TCP congestion control):
  - Track a rolling average of recent inference times.
  - If the average exceeds the frame-time budget for the target FPS,
    first shrink the inference resolution scale; if already at the
    minimum, start skipping frames outright.
  - If there's headroom, do the opposite, up to configured maximums.
"""
from __future__ import annotations

from collections import deque


class DynamicFrameSkipper:
    def __init__(
        self,
        target_fps: float = 8.0,
        min_scale: float = 0.4,
        max_scale: float = 1.0,
        max_skip: int = 6,
        window: int = 15,
    ):
        self.target_frame_time = 1.0 / target_fps
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.max_skip = max_skip

        self._recent_times = deque(maxlen=window)
        self.scale = max_scale
        self.skip = 0
        self._since_last_process = 0

    def should_process(self) -> bool:
        """Call once per captured frame. Returns True if this frame should
        go through YOLO, False if it should reuse the last detections."""
        if self._since_last_process >= self.skip:
            self._since_last_process = 0
            return True
        self._since_last_process += 1
        return False

    def record_inference_time(self, seconds: float) -> None:
        """Call after running inference, with the wall-clock time it took,
        to adapt `skip` and `scale` for subsequent frames."""
        self._recent_times.append(seconds)
        avg = sum(self._recent_times) / len(self._recent_times)

        if avg > self.target_frame_time * 1.15:
            # Running behind: shrink resolution before skipping more
            # frames outright, since resolution mostly preserves
            # detection continuity better than dropping frames does.
            if self.scale > self.min_scale:
                self.scale = max(self.min_scale, round(self.scale - 0.1, 2))
            elif self.skip < self.max_skip:
                self.skip += 1
        elif avg < self.target_frame_time * 0.7:
            # Comfortable headroom: claw back skip first, then resolution.
            if self.skip > 0:
                self.skip -= 1
            elif self.scale < self.max_scale:
                self.scale = min(self.max_scale, round(self.scale + 0.1, 2))

    @property
    def status(self) -> str:
        return f"skip={self.skip} scale={self.scale:.2f}"
