import cv2
import numpy as np
from tqdm import tqdm

# ----------------------------
# Internal Imports (Your Code)
# ----------------------------
from src.engine import InferenceEngine
from src.utils.image import letterbox_480_to_640, crop_mask_640_to_480, letterbox, unletterbox
from src.adas.perception import perceive_lanes
from src.adas.segmentation import clean_road_mask
from src.adas.checkpoint import CheckpointManager
from src.adas.control import CostController
from src.models import GpsData
import math

class SimulatedGPS:
    def __init__(self, lat, lon, dlat, dlon):
        self.lat = lat
        self.lon = lon
        self.dlat = dlat
        self.dlon = dlon
        self.heading = math.atan2(dlon, dlat)

    def step(self, turn_rate=0.0, speed_scale=1.0):
        """
        turn_rate: radians per frame (+ = left, - = right)
        speed_scale: multiplier for speed
        """

        # Rotate heading
        self.heading += turn_rate

        # Move forward
        speed = math.hypot(self.dlat, self.dlon) * speed_scale

        self.lat += math.cos(self.heading) * speed
        self.lon += math.sin(self.heading) * speed

        return (self.lat, self.lon, self.heading)

# ----------------------------
# Config
# ----------------------------
INPUT_VIDEO = "src/data/input.mp4"
OUTPUT_VIDEO = "src/data/output_autonomy.avi"

# Hardcoded route (same as server)
route_data = [(37.7749, -122.4194), (37.7750, -122.4195)]

# ----------------------------
# Initialize Modules
# ----------------------------
print("🚀 Loading model...")
engine = InferenceEngine("src/data/yolopv2fp16.xml", device="GPU")

navigator = CheckpointManager(route_data)
pilot = CostController()

# ----------------------------
# Video IO
# ----------------------------
cap = cv2.VideoCapture(INPUT_VIDEO)

if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {INPUT_VIDEO}")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"🎥 Video: {width}x{height} @ {fps} FPS | {frame_count} frames")

fourcc = cv2.VideoWriter_fourcc(*"XVID")
writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))
gps_sim = SimulatedGPS(
    lat=37.7749,
    lon=-122.4194,
    dlat=0.0000005,
    dlon=0.0
)

# ----------------------------
# Visualization Helpers
# ----------------------------
def draw_trajectory(img, traj, color=(0, 255, 0)):
    for i in range(1, len(traj)):
        p1 = (int(traj[i-1][0]), int(traj[i-1][1]))
        p2 = (int(traj[i][0]), int(traj[i][1]))
        cv2.line(img, p1, p2, color, 2)

def draw_steering_arrow(img, steering, color=(0, 0, 255)):
    h, w, _ = img.shape
    center = (w // 2, h - 50)

    length = 100
    angle = -steering  # flip if needed

    end = (
        int(center[0] + length * np.sin(angle)),
        int(center[1] - length * np.cos(angle))
    )

    cv2.arrowedLine(img, center, end, color, 4, tipLength=0.3)

def draw_gps_hud(img, gps, heading=None):
    lat, lon = gps

    y0 = 120
    dy = 35

    cv2.putText(img, f"GPS:", (20, y0),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    cv2.putText(img, f"Lat: {lat:.7f}", (20, y0 + dy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    cv2.putText(img, f"Lon: {lon:.7f}", (20, y0 + 2*dy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    if heading is not None:
        cv2.putText(img, f"Heading: {np.degrees(heading):.2f} deg",
                    (20, y0 + 3*dy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)


# ----------------------------
# Main Loop
# ----------------------------
print("🧠 Processing video...")

for _ in tqdm(range(frame_count)):
    ret, frame = cap.read()
    if not ret:
        break
    shape = frame.shape[:2]

    # 1. Preprocess (any_frame -> 640)
    img_640 = letterbox(frame)

    # 2. Inference
    outputs = engine.infer(img_640)

    lane_logits = outputs["lane"][0]   # (2, 640, 640)
    road_logits = outputs["drive"][0]

    lane_mask_640 = np.argmax(lane_logits, axis=0).astype(np.uint8)
    road_mask_640 = np.argmax(road_logits, axis=0).astype(np.uint8)

    # 3. Crop back to 480
    lane_mask = unletterbox(lane_mask_640, shape)
    road_mask = unletterbox(road_mask_640, shape)

    road_mask_clean = clean_road_mask(road_mask)

    # 4. Lane perception
    left_lane, right_lane = perceive_lanes(lane_mask, road_mask_clean)

    # 5. Navigation update (FAKE GPS for offline demo)
    gps = gps_sim.step(turn_rate=0.0)
    nav_status, nav_bias = navigator.update(GpsData(
        lat=gps[0], lon=gps[1]
    ))

    if nav_status == "FINISHED":
        steering = 0.0
        traj = []
        status_msg = "FINISHED"
    else:
        steering, traj = pilot.calculate_optimal_steering(
            road_mask_clean,
            left_lane,
            right_lane,
            nav_bias
        )
        status_msg = "NORMAL"

    # ----------------------------
    # Visualization Overlay
    # ----------------------------
    vis = frame.copy()

    # Road mask overlay (blue)
    road_overlay = np.zeros_like(vis)
    road_overlay[road_mask_clean == 1] = (255, 0, 0)
    vis = cv2.addWeighted(vis, 1.0, road_overlay, 0.3, 0)

    # Lane mask overlay (green)
    lane_overlay = np.zeros_like(vis)
    lane_overlay[lane_mask == 1] = (0, 255, 0)
    vis = cv2.addWeighted(vis, 1.0, lane_overlay, 0.5, 0)

    # Draw trajectory
    if len(traj) > 1:
        draw_trajectory(vis, traj)

    # Draw steering arrow
    draw_steering_arrow(vis, steering)
    draw_gps_hud(vis, (gps[0],gps[1]), gps[2])

    # Status text
    cv2.putText(vis, f"Status: {status_msg}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

    cv2.putText(vis, f"Steering: {steering:.3f}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

    # Write frame
    writer.write(vis)

# ----------------------------
# Cleanup
# ----------------------------
cap.release()
writer.release()
print("✅ Done! Saved to:", OUTPUT_VIDEO)
