import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import os
import time

YOLO_ENGINE  = "src/weights/yolo/yolov.engine"
YOLOP_ENGINE = "src/weights/yolop/yolop.engine"


# ------------------------------------------------------------------------------
# Pretty Logging Helpers
# ------------------------------------------------------------------------------
def line():
    print("-" * 80)

def green(msg):  print(f"\033[92m{msg}\033[0m")
def red(msg):    print(f"\033[91m{msg}\033[0m")
def yellow(msg): print(f"\033[93m{msg}\033[0m")
def blue(msg):   print(f"\033[94m{msg}\033[0m")


# ------------------------------------------------------------------------------
# Dummy Input Creation (Matches OpenVINO Input)
# ------------------------------------------------------------------------------
def make_dummy_input():
    """
    Creates a dummy (1,3,320,320) float32 tensor.
    Equivalent to: RGB normalized input.
    """
    dummy = np.random.rand(1, 3, 320, 320).astype(np.float32)
    return dummy


# ------------------------------------------------------------------------------
# Engine Probe + Dummy Inference
# ------------------------------------------------------------------------------
def probe_engine(engine_path: str):
    line()
    blue(f"🔍 Probing TensorRT Engine: {engine_path}")
    line()

    if not os.path.exists(engine_path):
        red(f"❌ Engine file does NOT exist: {engine_path}")
        return

    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)

    # Load Engine
    try:
        with open(engine_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
    except Exception as e:
        red(f"❌ Failed to load engine: {e}")
        return

    if engine is None:
        red("❌ Engine load returned None (corrupt engine?)")
        return

    green("✅ Engine loaded successfully!")

    print(f"• Number of bindings : {engine.num_bindings}")
    print(f"• Max batch size     : {engine.max_batch_size}")
    print(f"• Optimization profiles: {engine.num_optimization_profiles}")
    print(f"• FP16 enabled: {engine.has_fast_fp16}")
    print(f"• INT8 enabled: {engine.has_fast_int8}")

    line()
    yellow("🔎 Binding Details:")
    line()

    bindings_info = []
    for i in range(engine.num_bindings):
        name = engine.get_binding_name(i)
        is_input = engine.binding_is_input(i)
        shape = engine.get_binding_shape(i)
        dtype = engine.get_binding_dtype(i)

        role = "INPUT  " if is_input else "OUTPUT "
        print(f"[{i}] {role} | {name:25} | Shape: {shape} | DataType: {dtype}")
        bindings_info.append((i, name, is_input, shape, dtype))

    # Count inputs/outputs
    n_inputs = sum(b[2] for b in bindings_info)
    n_outputs = engine.num_bindings - n_inputs

    line()
    print(f"Inputs Detected : {n_inputs}")
    print(f"Outputs Detected: {n_outputs}")
    line()

    # ------------------------------------------------------------------------------
    # Dummy inference test
    # ------------------------------------------------------------------------------
    yellow("⚡ Running Dummy Inference Test...")
    line()

    context = engine.create_execution_context()
    stream = cuda.Stream()

    # Allocate memory for each binding
    device_buffers = []
    host_buffers = []

    for i, name, is_input, shape, dtype in bindings_info:
        np_dtype = trt.nptype(dtype)
        size = trt.volume(shape)

        host_mem = cuda.pagelocked_empty(size, np_dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)

        host_buffers.append(host_mem)
        device_buffers.append(device_mem)

    # Fill input 0 with dummy data
    dummy_input = make_dummy_input().ravel()
    np.copyto(host_buffers[0], dummy_input)

    # H2D
    cuda.memcpy_htod_async(device_buffers[0], host_buffers[0], stream)

    # Execute
    t0 = time.time()
    context.execute_async_v2(bindings=[int(d) for d in device_buffers],
                             stream_handle=stream.handle)
    stream.synchronize()
    t1 = time.time()

    # D2H outputs
    for i, (_, _, is_input, shape, _) in enumerate(bindings_info):
        if not is_input:
            cuda.memcpy_dtoh_async(host_buffers[i], device_buffers[i], stream)

    stream.synchronize()

    green(f"✅ Dummy inference successful! Total time: {(t1 - t0)*1000:.2f} ms")

    # Print output shapes
    for i, (_, name, is_input, shape, _) in enumerate(bindings_info):
        if not is_input:
            arr = host_buffers[i].reshape(shape)
            print(f"Output[{name}] shape = {arr.shape}")

    line()
    print("Done.")
    line()


# ------------------------------------------------------------------------------
# Run Debug on Both Engines
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    probe_engine(YOLO_ENGINE)
    probe_engine(YOLOP_ENGINE)