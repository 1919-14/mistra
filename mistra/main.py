"""
Standalone entry point. Runs entirely on the Raspberry Pi with a
monitor plugged into its own HDMI port -- no laptop, no network
socket, no streaming. Camera capture, dynamic frame skipping, YOLOv8n
inference, box drawing, and local display all happen in one loop.

Dropping the network layer isn't just simpler -- it removes real
per-frame cost (JPEG encode, socket I/O) that used to compete with
inference for the Pi's CPU budget. The dynamic frame skipper now only
has to answer to the local display's own refresh, not to a remote
consumer's bandwidth.

Usage (on the Pi, with the real camera):
    python3 -m mistra.main --model yolov8n.pt --target-fps 8

Testing without Pi camera hardware (from this project's root, on any
machine with a display):
    python3 -m mistra.main --camera-device 0      # USB webcam
    python3 -m mistra.main --synthetic            # no camera at all
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from mistra import config
from mistra.camera import CameraSource
from mistra.detector import YOLODetector, draw_detections, empty_detections
from mistra.display import Display
from mistra.frame_skipper import DynamicFrameSkipper
from mistra.preprocess import Preprocessor


def round_to_stride(value: int, stride: int = 32) -> int:
    """YOLO wants imgsz to be a multiple of its stride (32 for v8)."""
    return max(stride, int(round(value / stride)) * stride)


def run(args: argparse.Namespace) -> None:
    camera = CameraSource(args.width, args.height, args.device, args.synthetic)
    classes = [int(c) for c in args.classes.split(",")] if args.classes else None
    detector = YOLODetector(
        weights=args.model, conf=args.conf, device=args.yolo_device, classes=classes
    )
    skipper = DynamicFrameSkipper(
        target_fps=args.target_fps, min_scale=args.min_scale, max_skip=args.max_skip,
    )
    preprocessor = Preprocessor(
        denoise=args.denoise, clahe=args.clahe, dehaze=args.dehaze,
        denoise_strength=args.denoise_strength, clahe_clip=args.clahe_clip,
    )
    display = None if args.headless else Display(fullscreen=not args.windowed)

    print(f"[mistra] running standalone ({args.width}x{args.height}, "
          f"model={args.model}, target_fps={args.target_fps}, "
          f"headless={args.headless})")

    last_detections = empty_detections()
    frame_id = 0
    loop_times: list[float] = []

    try:
        while True:
            loop_start = time.time()
            frame = camera.read()
            frame, prep_stats = preprocessor.apply(frame)

            inference_ms = 0.0
            imgsz_used = 0
            skipped = True
            if skipper.should_process():
                imgsz_used = round_to_stride(int(args.base_imgsz * skipper.scale))
                last_detections = detector.run(frame, imgsz=imgsz_used)
                skipper.record_inference_time(last_detections.inference_seconds)
                inference_ms = last_detections.inference_seconds * 1000
                skipped = False

            annotated = draw_detections(frame, last_detections)

            loop_times.append(time.time() - loop_start)
            if len(loop_times) > 30:
                loop_times.pop(0)
            total_time = sum(loop_times)
            loop_fps = len(loop_times) / total_time if total_time > 0 else 0.0

            hud = [
                f"preprocess={prep_stats.total_ms:.1f}ms "
                f"(denoise={prep_stats.denoise_ms:.1f} clahe={prep_stats.clahe_ms:.1f} "
                f"dehaze={prep_stats.dehaze_ms:.1f})",
                f"infer={inference_ms:.1f}ms imgsz={imgsz_used} skipped={skipped} "
                f"dets={len(last_detections.boxes)}",
                f"skip_n={skipper.skip} scale={skipper.scale:.2f} "
                f"loop_fps={loop_fps:.1f} frame={frame_id}",
            ]
            if display is not None:
                if not display.show(annotated, hud):
                    print("[mistra] 'q' pressed, shutting down")
                    break
            else:
                if frame_id % int(max(1, args.target_fps)) == 0 or not skipped:
                    print(f"[mistra #{frame_id:04d}] fps={loop_fps:.1f} | "
                          f"infer={inference_ms:.1f}ms (imgsz={imgsz_used}) | "
                          f"prep={prep_stats.total_ms:.1f}ms | "
                          f"dets={len(last_detections.boxes)}", flush=True)

            frame_id += 1
            if args.max_frames > 0 and frame_id >= args.max_frames:
                print(f"[mistra] reached max frames ({args.max_frames}), shutting down", flush=True)
                break

            elapsed = time.time() - loop_start
            sleep_for = (1.0 / args.max_capture_fps) - elapsed
            if sleep_for > 0:
                time.sleep(sleep_for)
    except KeyboardInterrupt:
        print("\n[mistra] shutting down")
    finally:
        camera.release()
        if display is not None:
            display.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Standalone Pi capture + YOLOv8n inference + local HDMI display."
    )
    p.add_argument("--width", type=int, default=config.DEFAULT_WIDTH)
    p.add_argument("--height", type=int, default=config.DEFAULT_HEIGHT)
    p.add_argument("--max-capture-fps", type=float, default=config.DEFAULT_CAPTURE_FPS,
                    help="hard cap on the capture/infer/display loop rate")
    p.add_argument("--camera-device", type=int, default=0, dest="device",
                    help="OpenCV VideoCapture index, used only if picamera2 is unavailable")
    p.add_argument("--synthetic", action="store_true",
                    help="generate a synthetic test pattern instead of a real camera "
                         "(no hardware needed)")
    p.add_argument("--windowed", action="store_true",
                    help="show the display in a normal window instead of fullscreen "
                         "(useful when testing on a desktop, not the Pi's own monitor)")
    p.add_argument("--headless", action="store_true",
                    help="run without opening a GUI window, printing benchmark telemetry "
                         "to the terminal (useful over SSH or when no monitor is attached)")
    p.add_argument("--max-frames", type=int, default=0,
                    help="stop after processing N frames (0 = run indefinitely, useful for benchmarks)")

    p.add_argument("--no-denoise", dest="denoise", action="store_false",
                    help="disable bilateral-filter noise reduction (on by default)")
    p.add_argument("--denoise-strength", type=int, default=5,
                    help="bilateral filter diameter; higher = smoother but slower")
    p.add_argument("--no-clahe", dest="clahe", action="store_false",
                    help="disable CLAHE local contrast enhancement (on by default)")
    p.add_argument("--clahe-clip", type=float, default=2.0,
                    help="CLAHE clip limit; higher = stronger local contrast boost")
    p.add_argument("--dehaze", action="store_true",
                    help="enable Dark Channel Prior dehazing (OFF by default -- "
                         "experimental, validate on real IR footage before trusting it)")
    p.set_defaults(denoise=True, clahe=True)

    p.add_argument("--model", default="yolov8n.pt", help="ultralytics weights path/name")
    p.add_argument("--conf", type=float, default=0.35)
    p.add_argument("--yolo-device", default="cpu", dest="yolo_device",
                    help="'cpu', 'cuda', or a device index -- 'cpu' for a stock Raspberry Pi")
    p.add_argument("--classes", default="",
                    help="comma-separated COCO class ids to keep, e.g. '0,2,7' "
                         "for person,car,truck")

    p.add_argument("--target-fps", type=float, default=8.0,
                    help="inference throughput the dynamic skipper aims to sustain")
    p.add_argument("--base-imgsz", type=int, default=640, help="YOLO input size at scale=1.0")
    p.add_argument("--min-scale", type=float, default=0.4,
                    help="smallest fraction of base-imgsz the skipper may drop to")
    p.add_argument("--max-skip", type=int, default=6,
                    help="max consecutive frames skipped before forcing inference")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
