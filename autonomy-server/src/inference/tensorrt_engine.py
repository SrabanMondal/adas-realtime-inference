import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # noqa
import numpy as np
import cv2
from typing import Dict

WANTED_OUTPUTS = {"lane_line_seg", "drive_area_seg"}

class TRTInferenceEngine:
    def __init__(self, engine_path: str):
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)

        print(f"[INFO] Loading TensorRT engine: {engine_path}")
        with open(engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()

        self._allocate_buffers()
        print("[INFO] TensorRT Engine Ready.")

    def _allocate_buffers(self):
        self.inputs = []
        self.outputs = []
        self.bindings = []

        for binding in self.engine:
            shape = self.engine.get_binding_shape(binding)
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            size = trt.volume(shape)

            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            self.bindings.append(int(device_mem))

            if self.engine.binding_is_input(binding):
                self.input_shape = shape
                self.inputs.append((host_mem, device_mem))
            else:
                if binding in WANTED_OUTPUTS:
                    self.outputs.append((binding, host_mem, device_mem, shape))


    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        """
        Input:
            img: (640, 640, 3) BGR uint8
        Output:
            (1, 3, 640, 640) float16 RGB
        """
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float16) / 255.0
        img = np.transpose(img, (2, 0, 1))  # HWC → CHW
        img = np.expand_dims(img, axis=0)
        return img

    def infer(self, img_640: np.ndarray) -> Dict[str, np.ndarray]:
        input_data = self._preprocess(img_640)

        # Copy input to host buffer
        np.copyto(self.inputs[0][0], input_data.ravel())

        # H2D
        cuda.memcpy_htod_async(
            self.inputs[0][1],
            self.inputs[0][0],
            self.stream
        )

        # Inference
        self.context.execute_async_v2(
            bindings=self.bindings,
            stream_handle=self.stream.handle
        )

        # D2H
        results = {}
        for name, host_mem, device_mem, shape in self.outputs:
            cuda.memcpy_dtoh_async(host_mem, device_mem, self.stream)
            results[name] = host_mem.reshape(shape)

        self.stream.synchronize()

        return {
            "drive": results.get("drive_area_seg",[]),
            "lane": results.get("lane_line_seg",[]),
        }

