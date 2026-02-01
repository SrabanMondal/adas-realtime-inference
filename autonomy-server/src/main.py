import asyncio
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import numpy as np

# API Models & Codecs
from src.api.models import (
    SensorMessage, AutonomyMessage, AutonomyState, Control,
    encode_msgpack, decode_msgpack, decode_jpeg_bytes
)

# Inference Engines
from src.inference.openvino_engine import InferenceEngine
from src.inference.object_engine import ObjectInferenceEngine, ObjectPerception

# Perception Pipeline (matching test_infer.py)
from src.adas.perception.road.segmentation import clean_road_mask
from src.adas.perception.road.road_v2 import RoadPerception

# Control
from src.adas.control.mpcv2 import CenterlineMPC

# Utils
from src.utils.image import letterbox, unletterbox, scale_boxes

# =============================================================================
# Configuration
# =============================================================================
YOLOP_MODEL_PATH = "src/weights/yolop/yolopv2fp16.xml"
YOLO_MODEL_PATH = "src/weights/yolo/yolo26n.xml"
DEVICE = "GPU"

# =============================================================================
# FastAPI Application
# =============================================================================
app = FastAPI(title="Autonomy Server", version="2.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"])

# =============================================================================
# Initialize Pipeline Components (Global Singletons)
# =============================================================================
print("🚀 Initializing pipeline...")
engine = InferenceEngine(YOLOP_MODEL_PATH, device=DEVICE)
object_engine = ObjectInferenceEngine(YOLO_MODEL_PATH, device="CPU")
road_perception = RoadPerception()
mpc = CenterlineMPC()
print("✅ Pipeline ready")

# =============================================================================
# WebSocket Flow Control
# =============================================================================
latest_packet: bytes | None = None


async def receiver_task(ws: WebSocket):
    """Constantly drains socket to keep 'latest_packet' fresh."""
    global latest_packet
    try:
        while True:
            latest_packet = await ws.receive_bytes()
    except Exception:
        pass


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    global latest_packet
    await ws.accept()
    print("✅ Frontend Connected")

    # Start background receiver
    asyncio.create_task(receiver_task(ws))

    try:
        while True:
            # 1. Wait for a frame
            if latest_packet is None:
                await asyncio.sleep(0.005)  # Prevent CPU spin
                continue

            # 2. Grab & Clear (Atomic-ish op)
            raw_data = latest_packet
            latest_packet = None

            # 3. Decode incoming message
            msg = decode_msgpack(raw_data, SensorMessage)
            frame = decode_jpeg_bytes(msg.payload.image)
            frame_shape = frame.shape[:2]  # (H, W)

            # 4. Preprocess (any resolution -> 640x640)
            boxed = letterbox(frame)

            # 5. Inference
            outputs = engine.infer(boxed)
            object_outputs = object_engine.get_perception(boxed)

            # 6. Post-process segmentation masks
            drive_logits = outputs["drive"][0]
            lane_logits = outputs["lane"][0]

            # Binary masks at model resolution
            if drive_logits.shape[0] == 1:
                drive_mask_640 = (drive_logits[0] > 0).astype(np.uint8)
            else:
                drive_mask_640 = (drive_logits[1] > drive_logits[0]).astype(np.uint8)

            if lane_logits.shape[0] == 1:
                lane_mask_640 = (lane_logits[0] > 0).astype(np.uint8)
            else:
                lane_mask_640 = (lane_logits[1] > lane_logits[0]).astype(np.uint8)

            # Unletterbox to original resolution
            drive_mask = unletterbox(drive_mask_640, frame_shape)
            lane_mask = unletterbox(lane_mask_640, frame_shape)

            # Clean road mask (optional morphology)
            drive_mask = clean_road_mask(drive_mask)

            # 7. Road Perception -> Centerline
            road_out = road_perception.process(drive_mask)
            center_pts = road_out["center_points"]

            # 8. Object Detection -> Braking
            h, w = frame_shape
            object_perception = ObjectPerception(w, h)
            scaled_objs = scale_boxes(object_outputs, frame_shape)
            brake_force, closest_dist = object_perception.filter_and_control(scaled_objs, 10)

            # 9. MPC Control -> Steering
            gps_bias = 0.0  # TODO: Integrate GPS from msg.payload.gps
            steering, trajectory = mpc.compute(
                road_mask=drive_mask,
                center_points=center_pts,
                gps_bias=gps_bias
            )

            # 10. Determine status
            if brake_force > 0.8:
                status_msg = "WARNING"
            elif brake_force > 0.5:
                status_msg = "WARNING"
            else:
                status_msg = "NORMAL"

            # 11. Build response
            response = AutonomyMessage(
                type="autonomy",
                payload=AutonomyState(
                    laneLines=[],  # Lane lines not used in current pipeline
                    trajectory=trajectory,
                    control=Control(
                        steeringAngle=float(steering),
                        confidence=road_out["confidence"]
                    ),
                    status=status_msg
                )
            )

            await ws.send_bytes(encode_msgpack(response))

    except Exception as e:
        print(f"❌ Connection error: {e}")