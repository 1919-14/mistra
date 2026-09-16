# MISTRA Vision Module — Standalone Pi Camera + On-Device YOLOv8n

First working module toward MISTRA's camera pipeline (SIH26007, team
0818_Ananta): a Raspberry Pi captures frames, runs YOLOv8n **on the Pi
itself**, and renders the annotated feed straight to a monitor plugged
into the Pi's own HDMI port. Fully standalone — no laptop, no network
link, no streaming. All inference happens on the Pi on purpose — that's
the actual target hardware for MISTRA, so its real latency is the number
we need in order to know what to optimise before adding radar/GNSS
fusion on top.

```
Pi camera → capture → noise reduction → CLAHE → [dehaze, opt-in]
          → dynamic skip/scale decision → YOLOv8n (Pi CPU) → draw boxes
          → local HDMI display (Pi's own monitor)
```

Dropping the network layer isn't just simpler — it removes real
per-frame cost (JPEG encode, socket I/O) that used to compete with
inference for the Pi's CPU budget. That budget now goes entirely to
capture, preprocessing, and YOLO.

## Project structure

```
mistra-vision/
├── README.md
├── requirements.txt        # opencv (GUI build), numpy, ultralytics
├── mistra/
│   ├── config.py            # shared defaults (resolution, capture fps)
│   ├── camera.py            # picamera2 / USB webcam / synthetic test source
│   ├── preprocess.py        # noise reduction, CLAHE, optional DCP dehazing
│   ├── detector.py          # YOLOv8n wrapper + box-drawing
│   ├── frame_skipper.py     # dynamic skip + adaptive resolution controller
│   ├── display.py           # local HDMI display window + stats HUD
│   └── main.py              # capture → preprocess → infer → display (entry point)
└── tests/
    └── test_preprocess.py   # preprocessing step tests (cv2/numpy only)
```

## Setup

Run this directly on the Pi, with a monitor plugged into its HDMI port
(a desktop session or a bare KMS/DRM console both work — anything that
gives OpenCV's HighGUI a display to draw to).

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Only needed for the real Pi Camera Module (skip if using a USB webcam):
sudo apt install -y python3-picamera2
```

**Important:** `requirements.txt` installs plain `opencv-python`, not
`opencv-python-headless`. The headless wheel has no GUI/HighGUI support
at all, and `cv2.imshow` will fail — this build needs the GUI-capable
one specifically because it now renders locally.

`ultralytics` pulls in `torch`/`torchvision`, which can take a while to
install/build on a Pi. If that becomes a bottleneck, the natural next
step (once you've got a baseline `infer` number on-screen) is exporting
the model to NCNN or ONNX for a faster Pi-specific runtime — not in
scope for this first module, but `detector.py`'s interface is small
enough to swap out later without touching the rest of the pipeline.

The first run will auto-download `yolov8n.pt` (~6 MB) from Ultralytics
if it isn't already present in the working directory.

## Running it

On the Pi, with a monitor attached:

```bash
python3 -m mistra.main --target-fps 8
```

A fullscreen window opens on the Pi's own display showing the live
annotated feed with a HUD of live stats (preprocessing time, inference
time, frame skip/scale state, loop fps). Press `q` to quit, `s` to save
a snapshot into the working directory.

Use `--windowed` instead of fullscreen when developing on a regular
desktop rather than the Pi's dedicated monitor.

### Testing without Pi camera hardware

Camera and inference machinery both run fine on a development machine,
as long as it has its own display to render to:

```bash
# simulate the Pi using your own machine's webcam:
python3 -m mistra.main --camera-device 0 --windowed

# or with zero camera hardware, a synthetic test pattern:
python3 -m mistra.main --synthetic --windowed
```

### Running the preprocessing tests

```bash
python3 tests/test_preprocess.py
```

### No display attached?

`mistra.main` checks for `$DISPLAY`/`$WAYLAND_DISPLAY` up front and
fails with a clear error if neither is set, rather than letting
OpenCV's Qt backend hard-crash the process (which is what happens if
you try to open a GUI window with no display target — it's not a
catchable Python exception). If you're SSH'd into the Pi with no X
forwarding, this pipeline has nothing to draw to; run it from the Pi's
own console/desktop session instead.

## Dynamic frame skipping & adaptive resolution

`mistra/frame_skipper.py` implements a simple congestion-control-style
loop, driven by a rolling average of recent YOLO inference times
against a target frame budget (`--target-fps`):

- **Falling behind** → first shrink the inference resolution
  (`--base-imgsz × scale`, scale bottoming out at `--min-scale`); if
  already at minimum scale, start skipping frames outright (up to
  `--max-skip` consecutive skips).
- **Comfortable headroom** → claw back skipped frames first, then grow
  resolution back toward `--base-imgsz`.

On a skipped frame, the last known detection boxes are redrawn onto the
current raw frame — so the picture on screen stays live and
motion-smooth even when YOLO itself only updates the boxes every few
frames.

## Image preprocessing (`mistra/preprocess.py`)

Matches the pipeline in the knowledge doc's "Camera / Image Processing"
section, run on every captured frame before it goes to YOLO (and before
it's shown on the Pi's own display, since it's meant to help human
visibility too):

1. **Noise reduction** — edge-preserving bilateral filter (not the
   heavier `fastNlMeansDenoising`, to keep per-frame cost low on Pi CPU).
   On by default.
2. **CLAHE** — local contrast enhancement on the luminance channel only,
   to lift detail in fog/haze without blowing out color. On by default.
3. **Dehazing (Dark Channel Prior)** — **off by default, opt-in via
   `--dehaze`.** The knowledge doc is explicit that DCP dehazing "must be
   experimentally validated for the actual IR/thermal imagery" and
   shouldn't be assumed to help — it's implemented and available to test,
   not silently applied.

Resize/normalization (the last step before YOLO in the doc's diagram)
isn't a separate step here — it's handled internally by YOLO's own
`imgsz` argument in `detector.py`, so the frame isn't resized twice.

Each stage is independently timed and shown live in the on-screen HUD
(total preprocessing time, plus a denoise/CLAHE/dehaze breakdown)
alongside the inference stats, since this cost runs on every frame
regardless of the dynamic skipper — worth seeing on its own when
deciding what to trim.

## Key CLI flags (`mistra/main.py`)

| Flag | Default | Meaning |
|---|---|---|
| `--target-fps` | `8` | inference throughput the skipper aims to sustain |
| `--base-imgsz` | `640` | YOLO input size at full scale |
| `--min-scale` | `0.4` | smallest resolution fraction before frames start skipping |
| `--max-skip` | `6` | max consecutive frames skipped before forcing inference |
| `--conf` | `0.35` | YOLO confidence threshold |
| `--classes` | (all) | comma-separated COCO ids to keep, e.g. `0,2,7` for person,car,truck |
| `--no-denoise` | (denoise on) | disable bilateral-filter noise reduction |
| `--no-clahe` | (CLAHE on) | disable CLAHE contrast enhancement |
| `--dehaze` | off | enable DCP dehazing — experimental, see above |
| `--windowed` | off (fullscreen) | show the display in a normal window instead of fullscreen |
| `--synthetic` | off | no-camera test mode |

Run `python3 -m mistra.main --help` for the full list.

## What this module is (and isn't)

This is the **camera/detection slice** of MISTRA's architecture — the
`IR Camera → Frame Capture → OpenCV preprocessing → YOLO` part of the
pipeline described in the project's knowledge base, generalized to any
camera source for now (real IR camera hardware TBD). It doesn't yet
include:

- Radar or GNSS branches (parallel to this one per the architecture doc)
- Sensor fusion / Kalman tracking across sensors
- The rule-based risk engine and driver-facing risk decisions
- SQLite event logging
- Multi-camera stitching (the doc lists this as "if required" / an
  implementation choice — currently single-camera only)
- Remote monitoring/logging off-device — this module is intentionally
  standalone; if a remote dashboard is needed later, it would be a
  separate add-on rather than something this pipeline depends on

Those are natural next modules once this one gives a real number for
"how fast can a Pi actually run YOLOv8n on live frames, end to end,
including getting the result on screen."

## Honesty note

Per the project's own guidance: treat everything here as **prototype /
being benchmarked**, not "validated." The `loop_fps` and `infer`
numbers you'll see in the on-screen HUD depend heavily on which Pi
model you run this on — report them as measured, not projected.
#   m i s t r a  
 