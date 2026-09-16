"""
Background capture+preprocess worker.

Runs camera.read() and Preprocessor.apply() continuously in their own
thread, decoupled from the main thread's inference/display cadence. On
a multi-core Pi this lets capture+preprocess for frame N+1 overlap with
YOLO inference on frame N instead of the two serializing on a single
core -- that overlap is the entire reason this exists. Camera read plus
preprocessing typically costs single-digit-to-low-teens ms; YOLO
inference costs tens of ms. Serialized, every frame pays both costs
back to back. Overlapped, the main thread's loop time approaches
max(capture+preprocess, inference) instead of their sum.

Only the *latest* processed frame is kept (a single-slot mailbox, not a
queue): if the main thread is still busy with a slow inference call
when a newer frame arrives, the older one is simply discarded rather
than piling up and being processed late. This mirrors the same
"shed backlog rather than fall behind" principle as
DynamicFrameSkipper, just at the capture stage instead of the
inference stage.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from mistra.camera import CameraSource
from mistra.preprocess import Preprocessor, PreprocessStats


@dataclass
class CapturedFrame:
    frame_id: int
    image: np.ndarray
    prep_stats: PreprocessStats
    captured_at: float


class CaptureWorker:
    def __init__(self, camera: CameraSource, preprocessor: Preprocessor,
                 max_capture_fps: float = 20.0, cv2_threads: int = 1):
        self.camera = camera
        self.preprocessor = preprocessor
        self.max_capture_fps = max_capture_fps
        self.cv2_threads = cv2_threads

        self._lock = threading.Lock()
        self._latest: Optional[CapturedFrame] = None
        self._frame_id = 0
        self._stop = threading.Event()
        self._error: Optional[BaseException] = None
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> "CaptureWorker":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)

    def get_latest(self) -> Optional[CapturedFrame]:
        with self._lock:
            return self._latest

    def raise_if_failed(self) -> None:
        """Call from the main thread each iteration -- re-raises any
        exception that killed the worker, instead of the main loop
        silently spinning on a stale frame forever."""
        if self._error is not None:
            raise self._error

    def _run(self) -> None:
        # Keep OpenCV single-threaded in *this* thread so its internal
        # thread pool doesn't compete with torch's inference threads
        # for CPU time in the main thread -- see --capture-cv2-threads.
        cv2.setNumThreads(self.cv2_threads)
        frame_interval = 1.0 / self.max_capture_fps if self.max_capture_fps > 0 else 0.0
        try:
            while not self._stop.is_set():
                t0 = time.time()
                image = self.camera.read()
                image, prep_stats = self.preprocessor.apply(image)
                with self._lock:
                    self._latest = CapturedFrame(
                        self._frame_id, image, prep_stats, time.time()
                    )
                    self._frame_id += 1
                elapsed = time.time() - t0
                sleep_for = frame_interval - elapsed
                if sleep_for > 0:
                    time.sleep(sleep_for)
        except BaseException as exc:  # surfaced to the main thread via raise_if_failed()
            self._error = exc
            self._stop.set()
