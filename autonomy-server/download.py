import os
import urllib.request

WEIGHTS = {
    "yolov5nu.onnx": "https://github.com/SrabanMondal/adas-realtime-inference/releases/download/v1.0.0/yolov5nu.onnx",
    "yolopv2.onnx": "https://github.com/SrabanMondal/adas-realtime-inference/releases/download/v1.0.0/yolopv2.onnx",
}

TARGET_DIR = "src/weights"
os.makedirs(TARGET_DIR, exist_ok=True)

for name, url in WEIGHTS.items():
    dest = os.path.join(TARGET_DIR, name)
    if not os.path.exists(dest):
        print(f"[INFO] Downloading {name}...")
        urllib.request.urlretrieve(url, dest)
    else:
        print(f"[INFO] {name} already exists.")