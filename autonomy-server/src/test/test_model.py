import time
import numpy as np
from tqdm import tqdm

# import your engine
from src.inference.openvino_engine import InferenceEngine

MODEL_PATH = "src/data/yolopv2fp16.xml"
DEVICE = "GPU"   # ya "CPU"
WARMUP = 10
ITERATIONS = 100

def main():
    engine = InferenceEngine(MODEL_PATH, device=DEVICE)

    # random garbage frame (model input size)
    dummy = np.random.randint(0, 256, (640, 640, 3), dtype=np.uint8)

    print("Warming up...")
    for _ in range(WARMUP):
        engine.infer(dummy)

    print("Benchmarking...")
    times = []

    for _ in tqdm(range(ITERATIONS)):
        t0 = time.perf_counter()
        _ = engine.infer(dummy)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)  # ms

    avg = sum(times) / len(times)
    p95 = sorted(times)[int(0.95 * len(times))]

    print(f"\nResults on {DEVICE}:")
    print(f"  Avg latency : {avg:.2f} ms")
    print(f"  P95 latency : {p95:.2f} ms")
    print(f"  FPS (avg)   : {1000.0 / avg:.2f}")

if __name__ == "__main__":
    main()
