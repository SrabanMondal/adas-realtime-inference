import numpy as np
import cv2

class ObjectPerception:
    def __init__(self, frame_w, frame_h):
        self.w = frame_w
        self.h = frame_h
        # Define Trapezoidal ROI (bottom-heavy)
        self.roi_poly = np.array([
            [int(frame_w * 0.20), int(frame_h * 0.40)],  # Top-left (wide)
            [int(frame_w * 0.80), int(frame_h * 0.40)],  # Top-right (wide)
            [int(frame_w * 0.70), frame_h],              # Bottom-right (narrow)
            [int(frame_w * 0.30), frame_h],              # Bottom-left (narrow)
        ], dtype=np.int32)

        
        # Calibration: How many pixels = 1 meter? (Needs real-world testing)
        self.pixel_to_meter_ratio = 0.05 
        self.safe_distance = 15.0 # meters
        self.emergency_stop = 3.0 # meters

    def filter_and_control(self, detections, current_speed_kmh):
        closest_dist = float('inf')
        brake_force = 0.0 # 0.0 to 1.0

        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            if conf < 0.25: continue # Only cars (class 2 in COCO)

            # 1. Check if center-bottom of car is in Trapezoid
            # cx = (x1 + x2) / 2 * (self.w / 640)
            # by = y2 * (self.h / 640)
            cx = (x1 + x2) / 2
            by = y2

            is_inside = cv2.pointPolygonTest(self.roi_poly, (cx, by), False) >= 0
            
            if is_inside:
                # 2. Estimate Distance (Simple Geometry: Inverse of box height)
                box_h = (y2 - y1)
                distance = 1000 / box_h # Rough estimate formula
                
                if distance < closest_dist:
                    closest_dist = distance

        # 3. Braking Logic (TTC)
        if closest_dist < self.safe_distance:
            # Linear ramp: 0 force at 15m, 1.0 force at 3m
            brake_force = np.clip((self.safe_distance - closest_dist) / 
                                  (self.safe_distance - self.emergency_stop), 0, 1)

        return brake_force, closest_dist