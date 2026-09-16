"""
One-time helper: export a trained/pretrained YOLO .pt model for faster
CPU inference on the Pi. Supports two targets -- read this before
picking one, the right choice depends on what precision you want:

  --format ncnn (default)   FP16 or FP32 only. NCNN is built specifically
                             for ARM and is generally the fastest FP16/FP32
                             CPU backend on a Pi -- but as of this
                             ultralytics version, it does NOT support INT8
                             export (`quantize=8` is rejected outright for
                             this format; verified against the installed
                             exporter, not assumed from docs, since some
                             older docs/discussions claim otherwise).

  --format litert            TFLite/LiteRT, via XNNPACK's NEON-optimized
                              ARM kernels. This is the actual INT8 path for
                              a generic Pi CPU (no NPU/TensorRT/IMX
                              accelerator) -- ultralytics' own INT8-capable
                              format list is onnx/openvino/engine/coreml/
                              saved_model/edgetpu/mnn/imx/rknn/hailo/litert,
                              and litert is the one that targets a plain
                              ARM CPU the way NCNN does for FP16/FP32.

INT4 is not offered by any ultralytics export target (checked the
exporter source: valid --quantize values across every supported format
are 8, 16, w8a16, w8a32 -- nothing below INT8). This lines up with
published results elsewhere too: a 2025 study quantizing YOLO-family
detectors down to INT4 on ARM/edge hardware found it "practically
unusable due to high latency and lack of runtime support," even though
it looked fine in simulation (https://openreview.net/pdf?id=legjTSXjbD).
INT8 is the practical floor for object detection on this kind of
hardware -- don't spend time chasing INT4 here.

Usage:
    # FP16 NCNN (fast, no accuracy-losing quantization, safe default;
    # runs fine directly on the Pi, lightweight toolchain):
    python3 tools/export_ncnn.py --model yolov8n.pt --half

    # INT8 LiteRT (smallest/fastest, but DOES lose some accuracy, and
    # needs real calibration images -- see below):
    python3 tools/export_ncnn.py --model yolov8n.pt --format litert \
        --quantize 8 --calib-data path/to/your_data.yaml

Then point mistra.main at whichever output directory/file was produced
-- ultralytics' YOLO() auto-detects the format, nothing else changes:
    python3 -m mistra.main --model <exported_path>

IMPORTANT -- calibration data for INT8:
  INT8 needs a representative calibration pass (ultralytics warns if you
  give it fewer than ~300 images). Using generic COCO images
  (coco8.yaml, the tiny built-in default) calibrates activation ranges
  against clean daylight photos, not your actual fog/haze/IR conditions
  -- the statistics PTQ relies on may not match what the model will
  actually see in the mine. Build a small (300+ image) calibration set
  from real or representative footage and point --calib-data at a yaml
  describing it once you have that; treat a coco8.yaml run as a
  smoke-test of the export pipeline, not a calibration you'd deploy with.

IMPORTANT -- do this export on a laptop/PC, not the Pi:
  The LiteRT export path pulls in a heavy conversion toolchain (torch,
  jax, the litert converter, several hundred MB of downloads) that you
  don't want to fight for space/time on the Pi itself. Export on a dev
  machine, then copy only the resulting model file/directory over to
  the Pi -- the Pi only needs to *run* the exported model, not convert
  to it. NCNN's FP16 export is much lighter and fine to run on-device
  if you'd rather do it there.

Either way: benchmark whatever you export with
`--headless --max-frames 60` against your current baseline before
trusting it -- see the README's optimization section for why NCNN's
own numbers are inconsistent between the C++ and Python-wrapper paths.
"""
from __future__ import annotations

import argparse

from ultralytics import YOLO


def main() -> None:
    p = argparse.ArgumentParser(description="Export a YOLO .pt model for faster Pi inference.")
    p.add_argument("--model", default="yolov8n.pt", help="source .pt weights")
    p.add_argument("--format", choices=["ncnn", "litert"], default="ncnn",
                    help="ncnn: FP16/FP32 only, fastest ARM CPU backend, no INT8 support here. "
                         "litert: TFLite/XNNPACK, the actual INT8-on-CPU path")
    p.add_argument("--imgsz", type=int, default=640, help="export input size")
    p.add_argument("--half", action="store_true",
                    help="FP16 export (ncnn default choice if you're not quantizing to INT8)")
    p.add_argument("--quantize", choices=["8"], default=None,
                    help="INT8 quantization. Only valid with --format litert -- ncnn will "
                         "reject this outright (see module docstring)")
    p.add_argument("--calib-data", default="coco8.yaml",
                    help="calibration dataset yaml for --quantize 8 (default is ultralytics' "
                         "tiny built-in coco8.yaml -- fine for a smoke test, NOT recommended "
                         "for a real deployed model; see module docstring)")
    args = p.parse_args()

    if args.quantize == "8" and args.format == "ncnn":
        raise SystemExit(
            "ERROR: --format ncnn does not support --quantize 8 in this ultralytics "
            "version (NCNN export only supports FP16/FP32 here). Use --format litert "
            "for INT8, or drop --quantize and use --half for an NCNN FP16 export instead."
        )

    model = YOLO(args.model)
    export_kwargs = {"format": args.format, "imgsz": args.imgsz}
    if args.quantize == "8":
        export_kwargs["quantize"] = 8
        export_kwargs["data"] = args.calib_data
        print(f"[export] INT8 quantizing with calibration data: {args.calib_data}")
        if args.calib_data == "coco8.yaml":
            print("[export] WARNING: using the tiny built-in coco8.yaml (8 generic images). "
                  "This is a smoke test of the export pipeline, not a real calibration -- "
                  "see the module docstring before deploying this model.")
    elif args.half:
        export_kwargs["half"] = True

    out_path = model.export(**export_kwargs)
    print(f"[export] wrote {out_path}")
    print(f"[export] benchmark it: python3 -m mistra.main --model {out_path} "
          f"--headless --max-frames 60")


if __name__ == "__main__":
    main()
