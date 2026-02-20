import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2

class InferenceEngine:
    def __init__(self, engine_path: str):
        """
        TensorRT equivalent of the OpenVINO InferenceEngine.
        Accepts (320, 320, 3) BGR uint8 input.
        Returns output mask [C, 320, 320] -- no batch dim.
        """

        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)

        print(f"[INFO] Loading TensorRT engine: {engine_path}")
        with open(engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()

        # Allocate buffers
        self.bindings = []
        self.host_inputs = []
        self.cuda_inputs = []
        self.host_outputs = []
        self.cuda_outputs = []

        for i in range(self.engine.num_bindings):
            dtype = trt.nptype(self.engine.get_binding_dtype(i))
            shape = self.engine.get_binding_shape(i)
            size = trt.volume(shape)

            host_mem = cuda.pagelocked_empty(size, dtype)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)

            self.bindings.append(int(cuda_mem))

            if self.engine.binding_is_input(i):
                self.host_inputs.append(host_mem)
                self.cuda_inputs.append(cuda_mem)
                self.input_shape = shape  # (1,3,320,320)
            else:
                self.host_outputs.append(host_mem)
                self.cuda_outputs.append(cuda_mem)
                self.output_shape = shape  # (1,C,320,320)

        print(f"[INFO] TRT Ready. Output shape = {self.output_shape}")

    # -------------------------------------------------------
    # Preprocess (must match OpenVINO PrePostProcessor exactly)
    # -------------------------------------------------------
    def _preprocess(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        Input:  (320, 320, 3) BGR uint8
        Output: (1, 3, 320, 320) float32 normalized RGB
        """

        # BGR -> RGB
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Normalize
        img_rgb = img_rgb.astype(np.float32) / 255.0

        # NHWC -> NCHW
        nchw = np.transpose(img_rgb, (2, 0, 1))

        return np.expand_dims(nchw, 0)  # (1,3,H,W)

    # -------------------------------------------------------
    # Inference (OpenVINO-compatible API)
    # -------------------------------------------------------
    def infer(self, img_320_bgr: np.ndarray) -> np.ndarray:
        """
        Input: (320,320,3) uint8 BGR
        Output: (C,320,320) raw drive mask
        """

        # 1. Preprocess
        input_tensor = self._preprocess(img_320_bgr)

        # 2. Copy to device
        np.copyto(self.host_inputs[0], input_tensor.ravel())
        cuda.memcpy_htod_async(self.cuda_inputs[0], self.host_inputs[0], self.stream)

        # 3. Execute TRT
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)

        # 4. Copy back
        cuda.memcpy_dtoh_async(self.host_outputs[0], self.cuda_outputs[0], self.stream)
        self.stream.synchronize()

        # 5. Reshape output (1,C,320,320)
        output = self.host_outputs[0].reshape(self.output_shape)

        # 6. Remove batch dim → (C,320,320)
        return output[0]