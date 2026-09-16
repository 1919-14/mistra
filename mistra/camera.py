"""
Camera source abstraction, used only on the Pi side.

Prefers `picamera2` (the standard Raspberry Pi Camera Module driver).
Falls back to OpenCV `VideoCapture` (USB webcam) if picamera2 isn't
installed, so the rest of the pipeline can be developed/tested on a
development machine without any Pi hardware. `--synthetic` skips real cameras
entirely and generates a moving test pattern, for protocol/pipeline
testing on any machine with no camera at all.
"""
from __future__ import annotations

import time

import cv2
import numpy as np


class CameraSource:
    def __init__(self, width: int, height: int, device: int, synthetic: bool):
        self.width = width
        self.height = height
        self.synthetic = synthetic
        self._picam = None
        self._cap = None
        self._t0 = time.time()

        if synthetic:
            return

        try:
            from picamera2 import Picamera2  # type: ignore

            self._picam = Picamera2()
            cfg = self._picam.create_video_configuration(
                main={"size": (width, height), "format": "RGB888"}
            )
            self._picam.configure(cfg)
            self._picam.start()
            print("[camera] using picamera2 (Raspberry Pi Camera Module)")
        except Exception as exc:  # picamera2 missing or no CSI camera attached
            print(f"[camera] picamera2 unavailable ({exc}); "
                  f"falling back to OpenCV VideoCapture({device})")
            self._cap = cv2.VideoCapture(device)
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            if not self._cap.isOpened():
                raise RuntimeError(
                    "No picamera2 and no OpenCV-accessible camera device found. "
                    "Use --synthetic to test without any camera hardware."
                )

    def read(self) -> np.ndarray:
        if self.synthetic:
            return self._synthetic_frame()
        if self._picam is not None:
            frame = self._picam.capture_array()
            return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        ok, frame = self._cap.read()
        if not ok:
            raise RuntimeError("camera read failed")
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            frame = cv2.resize(frame, (self.width, self.height))
        return frame

    def _synthetic_frame(self) -> np.ndarray:
        t = time.time() - self._t0
        frame = np.full((self.height, self.width, 3), 30, dtype=np.uint8)
        cx = int(self.width / 2 + (self.width / 3) * np.sin(t))
        cy = int(self.height / 2 + (self.height / 3) * np.cos(t * 0.7))
        cv2.rectangle(frame, (cx - 40, cy - 25), (cx + 40, cy + 25), (0, 200, 0), -1)
        cv2.putText(frame, "SYNTHETIC FEED", (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, time.strftime("%H:%M:%S"), (10, self.height - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        return frame

    def release(self) -> None:
        if self._picam is not None:
            self._picam.stop()
        if self._cap is not None:
            self._cap.release()
