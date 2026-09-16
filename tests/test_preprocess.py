"""Sanity checks for the preprocessing steps. No camera or model
dependencies -- only cv2/numpy -- safe to run on any machine.

    python3 tests/test_preprocess.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))
from mistra.preprocess import Preprocessor, apply_clahe, dehaze_dcp, reduce_noise  # noqa: E402


def _fake_hazy_frame(h: int = 120, w: int = 160) -> np.ndarray:
    rng = np.random.default_rng(0)
    base = rng.integers(80, 140, size=(h, w, 3), dtype=np.uint8)
    haze = np.full((h, w, 3), 180, dtype=np.uint8)
    return (0.6 * base + 0.4 * haze).astype(np.uint8)


def test_each_step_preserves_shape_and_dtype() -> None:
    frame = _fake_hazy_frame()
    for fn in (reduce_noise, apply_clahe, dehaze_dcp):
        out = fn(frame)
        assert out.shape == frame.shape, f"{fn.__name__} changed shape"
        assert out.dtype == np.uint8, f"{fn.__name__} changed dtype"
    print("OK: each preprocessing step preserves shape/dtype")


def test_preprocessor_toggles_and_timing() -> None:
    frame = _fake_hazy_frame()

    all_off = Preprocessor(denoise=False, clahe=False, dehaze=False)
    out, stats = all_off.apply(frame)
    assert np.array_equal(out, frame)
    assert stats.total_ms == 0.0

    defaults = Preprocessor()  # denoise+clahe on, dehaze off
    out, stats = defaults.apply(frame)
    assert out.shape == frame.shape
    assert stats.denoise_ms > 0 and stats.clahe_ms > 0 and stats.dehaze_ms == 0.0

    all_on = Preprocessor(denoise=True, clahe=True, dehaze=True)
    out, stats = all_on.apply(frame)
    assert stats.dehaze_ms > 0
    assert stats.total_ms == stats.denoise_ms + stats.clahe_ms + stats.dehaze_ms
    print("OK: Preprocessor toggles and per-stage timing")


def test_clahe_increases_contrast_on_flat_image() -> None:
    flat = np.full((100, 100, 3), 120, dtype=np.uint8)
    flat[40:60, 40:60] = 135  # a faint low-contrast patch
    enhanced = apply_clahe(flat, clip_limit=3.0)
    assert enhanced.std() >= flat.std(), "CLAHE should not reduce local contrast here"
    print("OK: CLAHE increases local contrast on a low-contrast patch")


if __name__ == "__main__":
    test_each_step_preserves_shape_and_dtype()
    test_preprocessor_toggles_and_timing()
    test_clahe_increases_contrast_on_flat_image()
