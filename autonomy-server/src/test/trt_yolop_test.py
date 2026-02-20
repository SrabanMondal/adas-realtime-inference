import cv2
import numpy as np
import os
import time

from src.inference.tensorrt_engine import InferenceEngine
from src.utils.image import letterbox, unletterbox   # your exact functions


# -----------------------------------------------------------------------------
# Small logger
# -----------------------------------------------------------------------------
def log(msg):
    print(f"[DEBUG] {msg}")


# -----------------------------------------------------------------------------
# RUN VIDEO DEBUG
# -----------------------------------------------------------------------------
def run_yolop_debug(
    engine_path="src/weights/yolop/yolop.engine",
    video_path="src/data/input30.mp4",
    output_path="src/data/test_debug_yolop.mp4",
    input_size=320
):

    # Validate files
    if not os.path.exists(engine_path):
        raise FileNotFoundError(f"Cannot find engine: {engine_path}")
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Input video missing: {video_path}")

    # Load engine
    log(f"Loading TensorRT YOLOP engine")
    engine = InferenceEngine(engine_path)

    # Load video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Failed to open input video")

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS = cap.get(cv2.CAP_PROP_FPS)

    log(f"Video resolution: {W}x{H}, {FPS} FPS")

    # Prepare writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, FPS, (W, H))

    frame_id = 0
    infer_times = []

    # -------------------------------------------------------------------------
    # PROCESS VIDEO
    # -------------------------------------------------------------------------
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1
        log(f"Frame {frame_id}")

        # Letterbox
        lb = letterbox(frame, size=input_size)

        # Inference
        t0 = time.perf_counter()
        mask_output = engine.infer(lb)   # (C,320,320)
        infer_ms = (time.perf_counter() - t0) * 1000
        infer_times.append(infer_ms)

        log(f" - Inference took {infer_ms:.2f} ms")

        if mask_output.ndim != 3:
            log(f"[ERROR] Unexpected mask shape: {mask_output.shape}")
            continue

        C, Hm, Wm = mask_output.shape
        log(f" - Mask shape: C={C}, H={Hm}, W={Wm}")

        # If 2 channels → probably (drive, lane). Visualize each.
        if mask_output.shape[0] == 1:
            drive_mask_320 = (mask_output[0] > 0).astype(np.uint8)
        else:
            drive_mask_320 = (mask_output[1] > mask_output[0]).astype(np.uint8)


        # Unletterbox mask to original size
        mask_resized = unletterbox(drive_mask_320, (H, W), size=input_size)

        # Visualization overlay
        colored = cv2.applyColorMap(mask_resized, cv2.COLORMAP_JET)
        debug_frame = cv2.addWeighted(frame, 0.65, colored, 0.35, 0)

        writer.write(debug_frame)

    # -------------------------------------------------------------------------
    # FINISH
    # -------------------------------------------------------------------------
    cap.release()
    writer.release()

    log("-------------------------------------")
    log(f"Output video saved: {output_path}")
    if infer_times:
        log(f"Average inference: {np.mean(infer_times):.2f} ms")
        log(f"Max inference:     {np.max(infer_times):.2f} ms")
    log("-------------------------------------")


if __name__ == "__main__":
    run_yolop_debug()