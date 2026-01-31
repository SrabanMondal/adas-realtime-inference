import asyncio
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import numpy as np

# Internal Imports
from src.models import (
    SensorMessage, AutonomyMessage, AutonomyState, Control,
    encode_msgpack, decode_msgpack, decode_jpeg_bytes
)
from src.engine import InferenceEngine
from src.utils.image import letterbox_480_to_640, crop_mask_640_to_480
from src.adas.perception import perceive_lanes
from src.adas.segmentation import clean_road_mask
from src.adas.checkpoint import CheckpointManager
from src.adas.control import CostController

# Initialize Logic Modules
# Hardcoded route for demo (Lat, Lon)
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"])

# Initialize Engine (Global Singleton)
# Ensure yolopv2fp16.xml is in your 'data' folder
engine = InferenceEngine("src/data/yolopv2fp16.xml", device="GPU")
route_data = [(37.7749, -122.4194), (37.7750, -122.4195)] 
navigator = CheckpointManager(route_data)
pilot = CostController()

# --- Flow Control ---
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
                await asyncio.sleep(0.005) # Prevent CPU spin
                continue
                
            # 2. Grab & Clear (Atomic-ish op)
            raw_data = latest_packet
            latest_packet = None
            
            # 3. Decode
            msg = decode_msgpack(raw_data, SensorMessage)
            img_480 = decode_jpeg_bytes(msg.payload.image)
            
            # 4. Preprocess (480 -> 640)
            img_640 = letterbox_480_to_640(img_480)
            
            # 5. Inference (Iris GPU)
            outputs = engine.infer(img_640)
            
            lane_logits = outputs["lane"][0] # (2, 640, 640)
            road_logits = outputs["drive"][0]
            mask_640 = np.argmax(lane_logits, axis=0).astype(np.uint8)
            lane_mask = crop_mask_640_to_480(np.argmax(lane_logits, axis=0).astype(np.uint8))
            road_mask = crop_mask_640_to_480(np.argmax(road_logits, axis=0).astype(np.uint8))
            road_mask_clean = clean_road_mask(road_mask)
            # 5. Pure Logic (Operates on 640x480)
            left_lane, right_lane = perceive_lanes(lane_mask, road_mask_clean)
            lane_lines = [left_lane, right_lane]
            
            nav_status, nav_bias = navigator.update(msg.payload.gps)
            if nav_status == "FINISHED":
                 # Stop the car
                 steering = 0.0
                 traj = []
                 status_msg = "FINISHED"
            else:
                 # 2. Control Logic (Using Perception + Nav Bias)
                 # left_lane, right_lane came from perceive_lanes()
                 steering, traj = pilot.calculate_optimal_steering(road_mask_clean, left_lane, right_lane, nav_bias)
                 status_msg = "NORMAL"
            
            # 7. Send Response
            response = AutonomyMessage(
                type="autonomy",
                payload=AutonomyState(
                    laneLines=lane_lines,
                    trajectory=traj, # Populate with bicycle model later
                    control=Control(steeringAngle=steering, confidence=0.9),
                    status=status_msg
                )
            )
            
            await ws.send_bytes(encode_msgpack(response))

    except Exception as e:
        print(f"❌ Connection error: {e}")