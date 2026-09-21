"""
Thin wrapper around a YOLOv8n (nano) model for on-device object detection.

Detections are returned as raw arrays (boxes/scores/class ids) rather
than a pre-rendered image, so the caller can redraw the last known boxes
onto every new raw frame -- including skipped ones -- instead of only
updating the picture whenever inference actually runs. See
frame_skipper.py for the skip/scale decision this feeds into.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import cv2
import numpy as np
from ultralytics import YOLO


@dataclass
class Detections:
    boxes: np.ndarray            # (N, 4) xyxy, pixel coords in the original frame
    scores: np.ndarray           # (N,)
    class_ids: np.ndarray        # (N,)
    names: Dict[int, str]
    inference_seconds: float = 0.0


def empty_detections() -> Detections:
    return Detections(
        boxes=np.empty((0, 4)),
        scores=np.empty((0,)),
        class_ids=np.empty((0,), dtype=int),
        names={},
    )


class YOLODetector:
    def __init__(self, weights: str = "yolov8n.pt", conf: float = 0.35,
                 device: str = "cpu", classes: Optional[List[int]] = None):
        self.model = YOLO(weights)
        self.conf = conf
        self.device = device
        self.classes = classes  # e.g. [0, 2, 7] -> person/car/truck in COCO

    def run(self, image: np.ndarray, imgsz: int) -> Detections:
        """Runs inference; `imgsz` controls the resolution YOLO actually
        processes at (ultralytics letterboxes/rescales internally and
        returns box coordinates back in the *original* image's pixel
        space, so no manual rescaling is needed here)."""
        t0 = time.time()
        results = self.model.predict(
            source=image, imgsz=imgsz, conf=self.conf,
            classes=self.classes, device=self.device, verbose=False,
        )
        elapsed = time.time() - t0
        result = results[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            det = empty_detections()
        else:
            det = Detections(
                boxes=boxes.xyxy.cpu().numpy(),
                scores=boxes.conf.cpu().numpy(),
                class_ids=boxes.cls.cpu().numpy().astype(int),
                names=result.names,
            )
        det.inference_seconds = elapsed
        return det


_BOX_COLOR = (0, 255, 0)


def draw_detections(image: np.ndarray, det: Detections) -> np.ndarray:
    frame = image.copy()
    for box, score, cls_id in zip(det.boxes, det.scores, det.class_ids):
        x1, y1, x2, y2 = [int(v) for v in box]
        label = f"{det.names.get(int(cls_id), str(cls_id))} {score:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), _BOX_COLOR, 2)
        cv2.putText(frame, label, (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, _BOX_COLOR, 2, cv2.LINE_AA)
    return frame
