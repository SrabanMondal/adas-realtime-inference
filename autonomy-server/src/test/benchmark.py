import os
import time
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

# -------------------------------------------------------------------------
# Engine paths (adjust as needed)
# -------------------------------------------------------------------------
YOLO_ENGINE  = "src/weights/yolo/yolo.engine"
YOLOP_ENGINE = "src/weights/yolop/yolop.engine"


# -------------------------------------------------------------------------
# Pretty logging helpers
# -------------------------------------------------------------------------
def line():
    print("-" * 80)

def green(msg):  print(f"\033[92m{msg}\033[0m")
def red(msg):    print(f"\033[91m{msg}\033[0m")
def yellow(msg): print(f"\033[93m{msg}\033[0m")
def blue(msg):   print(f"\033[94m{msg}\033[0m")


# -------------------------------------------------------------------------
# Load TensorRT engine safely
# -------------------------------------------------------------------------
def load_engine(engine_path):
    line()
    blue(f"🔍 Loading TensorRT Engine: {engine_path}")
    line()

    if not os.path.exists(engine_path):
        red(f"❌ ERROR: Engine file does not exist.")
        raise FileNotFoundError(f"Engine not found: {engine_path}")

    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)

    try:
        with open(engine_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
    except Exception as e:
        red("❌ ERROR: Failed to deserialize engine.")
        print("Possible reasons:")
        print("• Engine built with a different TensorRT version")
        print("• Engine corrupted on disk")
        print("• Engine architecture mismatch (Jetson vs PC)")
        line()
        raise e

    if engine is None:
        red("❌ ERROR: deserialize_cuda_engine() returned None.")
        raise RuntimeError("Engine failed to load (corrupt or incompatible).")

    green("✅ Engine successfully loaded.")
    return engine


# -------------------------------------------------------------------------
# Allocate device/host buffers for TRT execution
# -------------------------------------------------------------------------
def allocate_buffers(engine):
    context = engine.create_execution_context()
    stream = cuda.Stream()

    host_inputs, device_inputs = [], []
    host_outputs, device_outputs = [], []
    bindings = []

    for i in range(engine.num_bindings):
        name = engine.get_binding_name(i)
        is_input = engine.binding_is_input(i)
        shape = engine.get_binding_shape(i)
        dtype = engine.get_binding_dtype(i)
        np_dtype = trt.nptype(dtype)

        size = trt.volume(shape)
        host_mem = cuda.pagelocked_empty(size, np_dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)

        bindings.append(int(device_mem))

        if is_input:
            host_inputs.append(host_mem)
            device_inputs.append(device_mem)
            yellow(f"• Input  '{name}'  shape: {shape}")
        else:
            host_outputs.append(host_mem)
            device_outputs.append(device_mem)
            yellow(f"• Output '{name}'  shape: {shape}")

        if size == 0:
            red(f"❌ ERROR: Binding {name} has size 0 — this engine is invalid.")
            raise ValueError("Invalid engine binding shape = 0.")

    return context, stream, host_inputs, device_inputs, host_outputs, device_outputs, bindings


# -------------------------------------------------------------------------
# Run a single inference pass
# -------------------------------------------------------------------------
def trt_infer(context, stream, host_inputs, device_inputs, host_outputs, device_outputs, bindings):
    # Copy host → device
    cuda.memcpy_htod_async(device_inputs[0], host_inputs[0], stream)
    
    # Execute inference
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)

    # Device → host
    for h_out, d_out in zip(host_outputs, device_outputs):
        cuda.memcpy_dtoh_async(h_out, d_out, stream)

    # Wait for GPU
    stream.synchronize()


# -------------------------------------------------------------------------
# Perform Benchmark
# -------------------------------------------------------------------------
def benchmark_engine(engine_path, iterations=100):
    line()
    blue(f"🚀 Benchmarking Engine: {engine_path}")
    line()

    engine = load_engine(engine_path)

    # Allocate buffers
    context, stream, host_inputs, device_inputs, host_outputs, device_outputs, bindings = allocate_buffers(engine)

    # ---------------------------------------------------------------------
    # Create dummy input that matches expected (1,3,320,320)
    # ---------------------------------------------------------------------
    dummy = np.random.rand(1, 3, 320, 320).astype(np.float32)
    np.copyto(host_inputs[0], dummy.ravel())

    green("✓ Dummy input created.")

    # ---------------------------------------------------------------------
    # Warm-up
    # ---------------------------------------------------------------------
    yellow("⚡ Warming up GPU (10 runs)...")
    for _ in range(10):
        trt_infer(context, stream, host_inputs, device_inputs, host_outputs, device_outputs, bindings)

    green("✓ Warm-up complete.")

    # ---------------------------------------------------------------------
    # Benchmark
    # ---------------------------------------------------------------------
    yellow(f"⚡ Running benchmark: {iterations} iterations...")
    latencies = []

    for i in range(iterations):
        t0 = time.perf_counter()
        trt_infer(context, stream, host_inputs, device_inputs, host_outputs, device_outputs, bindings)
        t1 = time.perf_counter()

        latencies.append((t1 - t0) * 1000)

    # ---------------------------------------------------------------------
    # Results
    # ---------------------------------------------------------------------
    avg = np.mean(latencies)
    p99 = np.percentile(latencies, 99)
    min_latency = np.min(latencies)
    max_latency = np.max(latencies)
    fps = 1000 / avg

    line()
    green("🏁 BENCHMARK RESULTS")
    line()
    print(f"Engine Path     : {engine_path}")
    print(f"Iterations      : {iterations}")
    print(f"Average Latency : {avg:.2f} ms")
    print(f"Peak Throughput : {fps:.2f} FPS")
    print(f"Min Latency     : {min_latency:.2f} ms")
    print(f"Max Latency     : {max_latency:.2f} ms")
    print(f"99th Percentile : {p99:.2f} ms")
    line()


# -------------------------------------------------------------------------
# Run benchmarks for both YOLO and YOLOP models
# -------------------------------------------------------------------------
if __name__ == "__main__":
    benchmark_engine(YOLO_ENGINE, iterations=100)
    benchmark_engine(YOLOP_ENGINE, iterations=100)