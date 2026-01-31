import time
import cv2
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

from models import SensorMessage
from ws_codec import decode_msgpack, encode_msgpack
from image_utils import decode_jpeg_bytes
from autonomy_demo import fake_autonomy_output


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

video_writer = None
frame_count = 0
start_time = time.time()


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    global video_writer, frame_count

    await ws.accept()
    print("✅ WebSocket connected")

    try:
        while True:
            data = await ws.receive_bytes()

            # ---------- Decode client message ----------
            msg = decode_msgpack(data, SensorMessage)
            packet = msg.payload

            # ---------- Decode image ----------
            img = decode_jpeg_bytes(packet.image)
            h, w = img.shape[:2]

            # ---------- Video writer ----------
            if video_writer is None:
                video_writer = cv2.VideoWriter(
                    "output/video.mp4",
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    10,
                    (w, h),
                )

            video_writer.write(img)
            frame_count += 1

            gps = packet.gps
            print(
                f"[Frame {frame_count}] "
                f"lat={gps.lat:.6f} lon={gps.lon:.6f} "
                f"w={w} h={h}"
            )

            # ---------- Send autonomy ----------
            t = time.time() - start_time
            response = fake_autonomy_output(t, w, h)
            await ws.send_bytes(encode_msgpack(response))

    except Exception as e:
        print("❌ WebSocket error:", e)

    finally:
        if video_writer:
            video_writer.release()
        print("🔌 WebSocket closed")
