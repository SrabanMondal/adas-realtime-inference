import openvino as ov
from openvino.preprocess import PrePostProcessor, ColorFormat
from openvino import Type, Layout
import numpy as np
import os

class InferenceEngine:
    def __init__(self, model_path: str, device: str = "GPU"):
        self.core = ov.Core()
        
        # GPU Cache to prevent the 10-second "startup lag"
        if not os.path.exists('model_cache'): os.makedirs('model_cache', exist_ok=True)
        self.core.set_property({'CACHE_DIR': './model_cache'})
        
        print(f"[INFO] Loading Drive-Only YOLOPv2: {model_path} to {device}...")
        raw_model = self.core.read_model(model_path)
        
        # --- PrePostProcessor Optimization ---
        ppp = PrePostProcessor(raw_model)
        
        # 1. Input: (1, 320, 320, 3) UINT8 BGR
        ppp.input().tensor() \
            .set_element_type(Type.u8) \
            .set_layout(Layout('NHWC')) \
            .set_color_format(ColorFormat.BGR)
        
        # 2. Preprocessing: Color swap + Normalize (0-1)
        # Doing this here is much faster than doing it in your Python loop
        ppp.input().preprocess() \
            .convert_element_type() \
            .convert_color(ColorFormat.RGB) \
            .scale(255.0)
        
        # 3. Model Expects: NCHW
        ppp.input().model().set_layout(Layout('NCHW'))
        
        # Build and Compile
        self.compiled_model = self.core.compile_model(
            ppp.build(), 
            device, 
            {"PERFORMANCE_HINT": "LATENCY"}
        )

        # Get the specific output port for 'drive_area'
        # Using any_name ensures we match the "drive_area" name from your export script
        self.output_layer = self.compiled_model.output("drive_area")
        
        print(f"[INFO] Engine Ready. Output layer: {self.output_layer.any_name}")

    def infer(self, img_320: np.ndarray) -> np.ndarray:
        """
        Args: img_320 (320, 320, 3) BGR image
        Returns: Raw drive mask [1, 2, 320, 320] or [1, 1, 320, 320]
        """
        # NHWC expansion
        input_tensor = np.expand_dims(img_320, 0)
        
        # Run inference
        results = self.compiled_model(input_tensor)
        
        # Return only the drive mask
        return results[self.output_layer][0]