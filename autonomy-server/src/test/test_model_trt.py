import time
import numpy as np
from tqdm import tqdm
import cv2

# Ensure this matches your actual filename and class name
from src.inference.tensorrt_engine import TRTInferenceEngine

ENGINE_PATH = "src/weights/yolop/yolop.engine"
WARMUP = 20
ITERATIONS = 200

def main():
    print(f"[INFO] Initializing TensorRT Engine: {ENGINE_PATH}")
    try:
        # Initializing the engine
        engine = TRTInferenceEngine(ENGINE_PATH)
    except Exception as e:
        print(f"[ERROR] Could not load engine: {e}")
        import traceback
        traceback.print_exc()
        return

    # Create dummy frame (H, W, C) 
    # Using 640x640 to match the expected model input and minimize resize overhead
    dummy = np.random.randint(0, 256, (640, 640, 3), dtype=np.uint8)

    print(f"Warming up for {WARMUP} iterations...")
    for _ in range(WARMUP):
        # CHANGED: engine.infer() instead of get_perception()
        _ = engine.infer(dummy)

    print("Benchmarking...")
    latencies = []

    for _ in tqdm(range(ITERATIONS)):
        t0 = time.perf_counter()
        
        # This includes: BGR->RGB, float16 conversion, H2D, Inference, D2H
        _ = engine.infer(dummy)
        
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)  # Convert to ms

    avg_latency = np.mean(latencies)
    median_latency = np.median(latencies)
    p95_latency = np.percentile(latencies, 95)
    fps = 1000.0 / avg_latency

    print("\n" + "="*40)
    print("       TensorRT Benchmark Results")
    print("="*40)
    print(f" Device       : NVIDIA Jetson Nano (Maxwell)")
    print(f" Avg Latency  : {avg_latency:.2f} ms")
    print(f" Median       : {median_latency:.2f} ms")
    print(f" P95 Latency  : {p95_latency:.2f} ms")
    print(f" FPS (avg)    : {fps:.2f}")
    print("="*40)

if __name__ == "__main__":
    main()
