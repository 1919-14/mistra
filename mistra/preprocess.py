"""
IR/low-visibility image preprocessing, applied on the Pi before a frame
goes to YOLO (and before it's shown on the Pi's own display). Mirrors
the pipeline in the project's knowledge doc:

    Frame Capture -> Noise Reduction -> CLAHE -> Dehazing/DCP (optional)
                   -> [resize/normalize -- handled internally by YOLO's
                      own `imgsz` argument in detector.py, not duplicated
                      here to avoid resizing twice] -> YOLO

Each step is independently toggleable and independently timed. Noise
reduction and CLAHE default ON (lightweight, generally beneficial for
low-visibility footage). Dehazing defaults OFF: the knowledge doc is
explicit that dark-channel dehazing "must be experimentally validated
for the actual IR/thermal imagery" and should not be assumed to always
help -- so it only runs when --dehaze is passed, not silently.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np


def reduce_noise(frame: np.ndarray, strength: int = 5) -> np.ndarray:
    """Edge-preserving denoise. Bilateral filtering rather than
    fastNlMeansDenoising: comparable noise reduction while preserving
    edges, at a fraction of the CPU cost -- worth it when this runs on
    every frame on Pi hardware."""
    return cv2.bilateralFilter(frame, d=strength, sigmaColor=50, sigmaSpace=50)


def apply_clahe(frame: np.ndarray, clip_limit: float = 2.0, tile_grid: int = 8) -> np.ndarray:
    """Contrast-Limited Adaptive Histogram Equalization on the luminance
    channel only, to lift local contrast in fog/haze without blowing out
    color or amplifying noise the way global histogram equalization does."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
    l_ch = clahe.apply(l_ch)
    lab = cv2.merge((l_ch, a_ch, b_ch))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def dehaze_dcp(frame: np.ndarray, patch_size: int = 15, omega: float = 0.95,
               t0: float = 0.1) -> np.ndarray:
    """Single-image dehazing via Dark Channel Prior (He et al., 2009).

    EXPERIMENTAL / opt-in -- see module docstring. DCP assumes an RGB
    haze-formation model that doesn't always hold for true IR/thermal
    imagery, so treat any visible improvement here as a hypothesis to
    validate on real camera footage, not a given. Also the most
    expensive of the three steps; expect it to dominate `preprocess_ms`
    when enabled.
    """
    img = frame.astype(np.float64) / 255.0

    # Dark channel: min over color channels, then min-filtered over a patch.
    min_channel = np.min(img, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
    dark_channel = cv2.erode(min_channel, kernel)

    # Atmospheric light: mean of the original-image pixels corresponding
    # to the brightest ~0.1% of dark-channel values.
    flat_dark = dark_channel.ravel()
    num_top = max(1, int(flat_dark.size * 0.001))
    top_idx = np.argpartition(flat_dark, -num_top)[-num_top:]
    flat_img = img.reshape(-1, 3)
    atmospheric_light = np.max(flat_img[top_idx], axis=0)

    # Transmission estimate and scene-radiance recovery.
    norm_img = img / np.clip(atmospheric_light, 1e-6, None)
    transmission = 1.0 - omega * cv2.erode(np.min(norm_img, axis=2), kernel)
    transmission = np.clip(transmission, t0, 1.0)

    recovered = np.empty_like(img)
    for c in range(3):
        recovered[:, :, c] = (
            (img[:, :, c] - atmospheric_light[c]) / transmission + atmospheric_light[c]
        )
    recovered = np.clip(recovered, 0.0, 1.0)
    return (recovered * 255).astype(np.uint8)


@dataclass
class PreprocessStats:
    denoise_ms: float = 0.0
    clahe_ms: float = 0.0
    dehaze_ms: float = 0.0

    @property
    def total_ms(self) -> float:
        return self.denoise_ms + self.clahe_ms + self.dehaze_ms


class Preprocessor:
    """Applies the enabled steps in order and times each one, so the
    per-stage cost shows up in the Pi's reported stats alongside
    inference time -- this is preprocessing overhead, not covered by
    the dynamic skipper, so it's worth seeing on its own."""

    def __init__(self, denoise: bool = True, clahe: bool = True, dehaze: bool = False,
                 denoise_strength: int = 5, clahe_clip: float = 2.0, dehaze_patch: int = 15):
        self.denoise = denoise
        self.clahe = clahe
        self.dehaze = dehaze
        self.denoise_strength = denoise_strength
        self.clahe_clip = clahe_clip
        self.dehaze_patch = dehaze_patch

    def apply(self, frame: np.ndarray) -> Tuple[np.ndarray, PreprocessStats]:
        stats = PreprocessStats()
        out = frame

        if self.denoise:
            t0 = time.perf_counter()
            out = reduce_noise(out, self.denoise_strength)
            stats.denoise_ms = (time.perf_counter() - t0) * 1000

        if self.clahe:
            t0 = time.perf_counter()
            out = apply_clahe(out, self.clahe_clip)
            stats.clahe_ms = (time.perf_counter() - t0) * 1000

        if self.dehaze:
            t0 = time.perf_counter()
            out = dehaze_dcp(out, self.dehaze_patch)
            stats.dehaze_ms = (time.perf_counter() - t0) * 1000

        return out, stats
