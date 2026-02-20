import os
import cv2
import numpy as np

from src.inference.trt_object_engine import TRTObjectInferenceEngine
from src.utils.image import letterbox, unletterbox, scale_boxes


DEBUG_OUTPUT_PATH = "src/data/output_debug.mp4"


class TRTDebugRunner:

    def __init__(self, engine_path: str):
        self.engine = TRTObjectInferenceEngine(engine_path)

    def run_on_video(self, mp4_path: str):
        assert os.path.exists(mp4_path), f"[ERROR] MP4 file does not exist: {mp4_path}"

        cap = cv2.VideoCapture(mp4_path)
        if not cap.isOpened():
            raise RuntimeError("[ERROR] Failed to open video")

        fps = cap.get(cv2.CAP_PROP_FPS)
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer = cv2.VideoWriter(
            DEBUG_OUTPUT_PATH,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (W, H)
        )

        print(f"[INFO] Running debug inference on video: {mp4_path}")
        print(f"[INFO] Original resolution = {W}x{H}")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            try:
                # -----------------------------
                # 1. LETTERBOX to 320x320
                # -----------------------------
                img320 = letterbox(frame, size=320)

                # -----------------------------
                # 2. Run TensorRT inference
                # -----------------------------
                raw = self.engine.infer(img320)

                # -----------------------------
                # 3. Filter for valid boxes
                # Format per row = [cx, cy, w, h, conf, cls_probs...]
                # -----------------------------
                dets = []
                for row in raw:
                    conf = row[4]
                    if conf < 0.25:
                        continue

                    cls_id = np.argmax(row[5:])
                    # Convert YOLO center format → xyxy
                    cx, cy, w, h = row[:4]
                    x1 = cx - w / 2
                    y1 = cy - h / 2
                    x2 = cx + w / 2
                    y2 = cy + h / 2

                    dets.append([x1, y1, x2, y2, conf, cls_id])

                dets = np.array(dets)

                # -----------------------------
                # 4. Scale boxes back to original resolution
                # -----------------------------
                dets_scaled = scale_boxes(dets, (H, W), size=320)

                # -----------------------------
                # 5. Draw boxes
                # -----------------------------
                for x1, y1, x2, y2, conf, cls_id in dets_scaled:
                    cv2.rectangle(frame, (int(x1), int(y1)),
                                  (int(x2), int(y2)),
                                  (0, 255, 0), 2)

                    cv2.putText(
                        frame,
                        f"{cls_id}:{conf:.2f}",
                        (int(x1), int(y1) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1
                    )

            except Exception as e:
                print(f"[ERROR] Inference error: {e}")

            writer.write(frame)

        cap.release()
        writer.release()

        print(f"[INFO] Debug video written to {DEBUG_OUTPUT_PATH}")


if __name__ == "__main__":
    runner = TRTDebugRunner("weights/yolo/yolo.engine")
    runner.run_on_video("src/data/input30.mp4")