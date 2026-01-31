import openvino as ov
from openvino.preprocess import PrePostProcessor, ColorFormat
from openvino import Type
import numpy as np
from typing import Dict

class InferenceEngine:
    def __init__(self, model_path: str, device: str = "GPU"):
        self.core = ov.Core()
        self.core.set_property({'CACHE_DIR': './model_cache'})
        
        print(f"[INFO] Loading {model_path} to {device}...")
        raw_model = self.core.read_model(model_path)
        
        # --- PrePostProcessor Optimization ---
        # Bakes normalization and layout conversion into the graph
        ppp = PrePostProcessor(raw_model)
        
        # Input: (1, 640, 640, 3) U8 BGR
        ppp.input().tensor() \
            .set_element_type(Type.u8) \
            .set_layout(ov.Layout('NHWC')) \
            .set_color_format(ColorFormat.BGR)
        
        # Process: U8->F32, BGR->RGB, Scale 0-1
        ppp.input().preprocess() \
            .convert_element_type(Type.f16) \
            .convert_color(ColorFormat.RGB) \
            .scale([255., 255., 255.])
        
        # Model expects: NCHW
        ppp.input().model().set_layout(ov.Layout('NCHW'))
        
        self.compiled_model = self.core.compile_model(
            ppp.build(), 
            device, 
            {"PERFORMANCE_HINT": "LATENCY"}
        )
        self.infer_request = self.compiled_model.create_infer_request()
        print("[INFO] Engine Ready.")

    def infer(self, img_640: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Args:
            img_640: (640, 640, 3) BGR image
        Returns:
            Dict containing raw logits/masks from model
        """
        input_tensor = np.expand_dims(img_640, 0)
        results = self.compiled_model(input_tensor)
        
        return {
            "drive": results["drive_area_seg"],
            "lane": results["lane_line_seg"]
        }