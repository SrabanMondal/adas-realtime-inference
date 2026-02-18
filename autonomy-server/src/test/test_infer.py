import cv2
import numpy as np
import time
import argparse
from tqdm import tqdm

from src.inference.openvino_engine import InferenceEngine
from src.utils.image import letterbox, unletterbox, scale_boxes
from src.adas.perception.road.segmentation import clean_road_mask
from src.adas.control.mpcv2 import CenterlineMPC
from src.adas.perception.road.road_v2 import RoadPerception
from src.inference.object_engine import ObjectInferenceEngine, ObjectPerception

INPUT_VIDEO = "src/data/input30.mp4"
OUTPUT_VIDEO = "src/data/letterbox_test_7.avi"
YOLOP_MODEL_PATH = "src/weights/yolop/yolopv2_fast_fp16.xml"
YOLO_MODEL_PATH = "src/weights/yolo/yolo26n.xml"
DEVICE = "GPU"
IMG_SIZE = 320

def color_for(src):
    if src == "LANE_DETECTED": return (255,255,0)
    if src == "ROAD_ESTIMATE": return (255,255,255)
    if src == "MEMORY_HOLD": return (0,255,255)
    return (0,0,128)

def draw_steering_arrow(
    img: np.ndarray,
    steering: float,
    max_steering: float = np.deg2rad(25),
    color=(0, 255, 255),
    thickness: int = 6,
):
    h, w = img.shape[:2]
    x0 = w // 2
    y0 = h - 10
    # Normalize steering
    s = np.clip(steering / max_steering, -1.0, 1.0)
    # Arrow length
    L = int(h * 0.35)
    # End point (rotate forward vector)
    angle = -np.pi / 2 + s * np.deg2rad(40)   # 40 deg visual exaggeration
    x1 = int(x0 + L * np.cos(angle))
    y1 = int(y0 + L * np.sin(angle))

    # Control point for curvature
    # Push control point sideways to show turning intent
    ctrl_x = int(x0 + s * L * 0.6)
    ctrl_y = int(y0 - L * 0.5)

    # Draw quadratic Bezier curve
    pts = []
    for t in np.linspace(0, 1, 30):
        x = int((1-t)**2 * x0 + 2*(1-t)*t * ctrl_x + t**2 * x1)
        y = int((1-t)**2 * y0 + 2*(1-t)*t * ctrl_y + t**2 * y1)
        pts.append((x, y))

    # Draw curve
    for i in range(len(pts)-1):
        cv2.line(img, pts[i], pts[i+1], color, thickness)

    # Draw arrow head
    cv2.arrowedLine(
        img,
        pts[-2],
        pts[-1],
        color,
        thickness,
        tipLength=0.4
    )

    # Draw base circle
    cv2.circle(img, (x0, y0), 8, (255,255,255), -1)


def main(demo: bool, morph: bool):
    cap = cv2.VideoCapture(INPUT_VIDEO)
    if not cap.isOpened():
        raise RuntimeError("Could not open input video")

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("[CONFIG] h: ",h," w: ",w)
    writer = None
    overlay = None

    if demo:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (w, h))
        overlay = np.zeros((h, w, 3), dtype=np.uint8)

    engine = InferenceEngine(YOLOP_MODEL_PATH, device=DEVICE)
    road_engine = RoadPerception()
    object_engine = ObjectInferenceEngine(YOLO_MODEL_PATH, device="CPU")
    object_perception = ObjectPerception(w, h)
    mpc = CenterlineMPC()

    # timing accumulators
    t_pre = t_inf = t_post = t_write = 0.0
    n = 0

    for _ in tqdm(range(frame_count)):
        
        t0 = time.perf_counter()
        ret, frame = cap.read()
        vis = frame
        if not ret:
            break

        boxed = letterbox(frame)

        t1 = time.perf_counter()

        drive_logits = engine.infer(boxed)
        object_outputs = object_engine.get_perception(boxed)

        t2 = time.perf_counter()

        # drive_logits = outputs["drive"][0]
        # lane_logits = outputs["lane"][0]

        # fast binary masks in model space
        # if drive_logits.ndim == 4: # [1, 2, 320, 320]
        #     # Use argmax to get the best class per pixel
        #     drive_mask_320 = np.argmax(drive_logits[0], axis=0).astype(np.uint8)
        # elif drive_logits.ndim == 3: # [1, 320, 320]
        #     drive_mask_320 = (drive_logits[0] > 0).astype(np.uint8)
        # else:
        #     drive_mask_320 = (drive_logits > 0).astype(np.uint8)
        if drive_logits.shape[0] == 1:
            drive_mask_320 = (drive_logits[0] > 0).astype(np.uint8)
        else:
            drive_mask_320 = (drive_logits[1] > drive_logits[0]).astype(np.uint8)

        # if lane_logits.shape[0] == 1:
        #     lane_mask_320 = (lane_logits[0] > 0).astype(np.uint8)
        # else:
        #     lane_mask_320 = (lane_logits[1] > lane_logits[0]).astype(np.uint8)
        
        drive_mask = unletterbox(drive_mask_320, frame.shape[:2])
        if morph:
            drive_mask = clean_road_mask(drive_mask)
        out = road_engine.process(drive_mask)
        unletterboxed_objs = scale_boxes(object_outputs, frame.shape[:2])
        brake, dist = object_perception.filter_and_control(unletterboxed_objs,10)
        center_pts = out["center_points"]
        steer, traj = mpc.compute(
                road_mask = drive_mask,
                center_points = center_pts,
                gps_bias = 0
            )

        if demo and overlay is not None:
            # lane_mask_orig  = unletterbox(lane_mask_640,  frame.shape[:2])
            overlay.fill(0)
            # overlay[lane_mask_orig == 1]  = (0, 0, 255)
                
            overlay[drive_mask == 1] = (255, 0, 0)
            center_pts = out["center_points"]           

            vis = cv2.addWeighted(frame, 1.0, overlay, 0.4, 0)
            # draw lanes
            if len(center_pts) > 1:
                pts = np.array(center_pts, dtype=np.int32)
                cv2.polylines(vis, [pts], False, (255,255,255), 5)
                
            h, w = vis.shape[:2]
            
            
            if len(traj) > 1:
                pts = np.array(traj, dtype=np.int32)
                cv2.polylines(vis, [pts], False, (0, 0, 0), 5)

            # origin_x = w // 2
            # origin_y = h - 1
            # if len(traj) > 1:
            #     pts = np.array(
            #         [(origin_x + dx, origin_y + dy) for dx, dy in traj],
            #         dtype=np.int32
            #     )
            #     cv2.polylines(vis, [pts], False, (0, 0, 0), 5)
            
            roi = object_perception.roi_poly.reshape((-1, 1, 2))
            cv2.polylines(vis, [roi], True, (0, 255, 255), 2)

            # Draw detections
            for det in unletterboxed_objs:
                x1, y1, x2, y2, conf, cls = det
                if conf > 0.25:
                    color = (0, 255, 0)
                    # If this car is the "brake trigger", make it red
                    if dist < object_perception.safe_distance and y2 == max([d[3] for d in unletterboxed_objs if d[4]>0.25]):
                         color = (0, 0, 255)
                    
                    cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    cv2.putText(vis, f"{dist:.1f}m", (int(x1), int(y1)-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
            if brake > 0:
                bar_h = int(h * 0.4 * brake)
                cv2.rectangle(vis, (w-50, h-50), (w-20, h-50-bar_h), (0, 0, 255), -1)
                cv2.putText(vis, "BRAKE", (w-80, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            draw_steering_arrow(vis, steer)
        t3 = time.perf_counter()

        if demo and writer:
            writer.write(vis)

        t4 = time.perf_counter()

        t_pre  += (t1 - t0)
        t_inf  += (t2 - t1)
        t_post += (t3 - t2)
        t_write+= (t4 - t3)
        n += 1

    cap.release()
    if writer:
        writer.release()

    print("\n---- AVG PER FRAME ----")
    print(f"Preprocess : {t_pre/n*1000:.2f} ms")
    print(f"Inference  : {t_inf/n*1000:.2f} ms")
    print(f"Postprocess: {t_post/n*1000:.2f} ms")
    print(f"Write      : {t_write/n*1000:.2f} ms")
    total = (t_pre + t_inf + t_post + t_write) / n
    print(f"Total      : {total*1000:.2f} ms  ({1000/total/1000:.2f} FPS)")
    if demo:
        print("Saved:", OUTPUT_VIDEO)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true",
                        help="Enable unletterbox + overlay + video write")
    parser.add_argument("--morph", action="store_true", help="Cleans the road mask")
    args = parser.parse_args()

    main(demo=args.demo, morph=args.morph)
