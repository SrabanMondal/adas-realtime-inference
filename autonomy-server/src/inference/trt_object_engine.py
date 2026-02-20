import cv2
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt


class TRTObjectInferenceEngine:
    """
    Clean, production-safe TensorRT inference engine.
    Fully compatible with OpenVINO output format.
    """

    def __init__(self, engine_path: str):
        self.logger = trt.Logger(trt.Logger.ERROR)

        with open(engine_path, "rb") as f:
            self.runtime = trt.Runtime(self.logger)
            self.engine = self.runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()

        self.bindings = []
        self.host_inputs = []
        self.cuda_inputs = []
        self.host_outputs = []
        self.cuda_outputs = []

        for binding in self.engine:
            idx = self.engine.get_binding_index(binding)
            shape = self.engine.get_binding_shape(idx)
            dtype = trt.nptype(self.engine.get_binding_dtype(idx))
            size = trt.volume(shape)

            host_mem = cuda.pagelocked_empty(size, dtype)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(cuda_mem))

            if self.engine.binding_is_input(idx):
                self.input_shape = shape  # (1, 3, 320, 320)
                self.host_inputs.append(host_mem)
                self.cuda_inputs.append(cuda_mem)
            else:
                self.output_shape = shape  # e.g. (1, 336, 84)
                self.host_outputs.append(host_mem)
                self.cuda_outputs.append(cuda_mem)

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess for TensorRT engine.
        """
        _, _, H, W = self.input_shape
        img = cv2.resize(frame, (W, H))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        return np.ascontiguousarray(np.expand_dims(img, 0))

    def infer(self, frame: np.ndarray) -> np.ndarray:
        """
        Returns (N, 84) detections exactly like OpenVINO.
        """
        img = self._preprocess(frame)
        np.copyto(self.host_inputs[0], img.ravel())

        cuda.memcpy_htod_async(self.cuda_inputs[0], self.host_inputs[0], self.stream)
        self.context.execute_async_v2(self.bindings, self.stream.handle, None)
        cuda.memcpy_dtoh_async(self.host_outputs[0], self.cuda_outputs[0], self.stream)
        self.stream.synchronize()

        # Expect engine output shape = (1, 336, 84)
        arr = self.host_outputs[0].reshape(self.output_shape)
        return arr[0]  # (336, 84)