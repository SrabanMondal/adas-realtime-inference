import numpy as np
import cv2
from typing import Tuple, List, Optional, Dict

# ============================================================
# Types & Constants
# ============================================================

LanePoints = List[List[int]]
Coefficients = Tuple[float, float, float]

class LaneSource:
    DETECTED = "LANE_DETECTED"      # High confidence
    ROAD_REF = "ROAD_ESTIMATE"      # Inferred from road edge
    MEMORY   = "MEMORY_HOLD"        # Stale data (coast mode)
    NONE     = "NO_LANE"            # Safety failure

# ============================================================
# 1. State Estimation (Memory & Watchdog)
# ============================================================

class LaneState:
    def __init__(self, max_age=15):
        # Rolling average of SIGNED offset (Lane X - Road X)
        self.avg_signed_offset = 0.0 
        self.count = 0
        
        # Memory & Watchdog
        self.last_coeffs: Optional[Coefficients] = None
        self.missing_frames = 0
        self.MAX_AGE = max_age  # Kill signal after ~0.5s of bad data

    def update_learning(self, lane_coeffs, road_coeffs, h):
        """
        Learn the signed distance between road edge and lane center.
        """
        # Evaluate at car hood level (bottom of frame)
        y_eval = h - 10
        l_x = lane_coeffs[0]*y_eval**2 + lane_coeffs[1]*y_eval + lane_coeffs[2]
        r_x = road_coeffs[0]*y_eval**2 + road_coeffs[1]*y_eval + road_coeffs[2]
        
        # Signed difference: (Lane - Road)
        current_offset = l_x - r_x
        
        # Sanity check: Offset must be reasonable (e.g., 10px to 250px)
        # Note: We check abs() for magnitude, but store signed value
        if 10 < abs(current_offset) < 250:
            if self.count == 0:
                self.avg_signed_offset = current_offset
            else:
                # Slow EMA to resist noise
                self.avg_signed_offset = 0.95 * self.avg_signed_offset + 0.05 * current_offset
            self.count += 1

    def get_fallback_model(self, road_coeffs) -> Optional[Coefficients]:
        """
        Synthesize a lane by applying the learned signed offset to the road edge.
        """
        if self.count < 5 or road_coeffs is None:
            return None # Not confident enough yet
            
        a, b, c = road_coeffs
        # Apply signed offset directly (works for both Left and Right sides)
        return (a, b, c + self.avg_signed_offset)

    def check_temporal_consistency(self, new_coeffs: Coefficients, threshold: int = 50) -> bool:
        """
        Returns True if the new lane is geometrically close to the last valid lane.
        """
        if self.last_coeffs is None:
            return True # First frame is always "consistent"
        
        # Check bottom intercept (c-term)
        # This detects lateral jumps
        delta_c = abs(new_coeffs[2] - self.last_coeffs[2])
        if delta_c > threshold:
            return False
            
        # Optional: Check curvature (a-term) if needed
        # delta_a = abs(new_coeffs[0] - self.last_coeffs[0])
        return True

# ============================================================
# 2. Robust Math & Extraction
# ============================================================

def robust_polyfit(y: np.ndarray, x: np.ndarray, w: int) -> Optional[Coefficients]:
    """
    Fit polynomial with adaptive threshold based on image width.
    """
    if len(x) < 50: return None
    
    # Adaptive Threshold (e.g., 1.5% of image width)
    # 640px -> ~9.6px, 1920px -> ~28px
    thresh_val = max(5.0, 0.015 * w) 

    try:
        # Pass 1: Rough Fit
        c = np.polyfit(y, x, 2)
        
        # Pass 2: Filter Outliers
        pred = c[0]*y**2 + c[1]*y + c[2]
        err = np.abs(x - pred)
        
        mask = err < thresh_val
        if np.sum(mask) < 40: return None
        
        c_final = np.polyfit(y[mask], x[mask], 2)
        return (float(c_final[0]), float(c_final[1]), float(c_final[2]))
    except:
        return None

def extract_road_edge_safe(road_mask: np.ndarray, side: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Robust extraction using boolean casting to prevent argmax bugs.
    """
    h, w = road_mask.shape
    
    # 1. Identify rows that have ANY road data
    row_sums = np.sum(road_mask, axis=1)
    valid_rows_idx = np.where(row_sums > 0)[0]
    
    if len(valid_rows_idx) < 50:
        return np.array([]), np.array([])
        
    # 2. Convert to Boolean to ensure argmax finds first *nonzero* pixel
    # (Fixes the bug where argmax finds the peak value 255 instead of edge value 1)
    valid_rows_bool = (road_mask[valid_rows_idx] > 0)
    
    if side == "left":
        # First True from left
        x_vals = np.argmax(valid_rows_bool, axis=1)
    else:
        # First True from right (using flip trick)
        x_vals = (w - 1) - np.argmax(np.flip(valid_rows_bool, axis=1), axis=1)
    
    # 3. Filter Frame Borders (Phantom Edge Fix)
    mask = (x_vals > 5) & (x_vals < w - 5)
    
    return valid_rows_idx[mask], x_vals[mask]

# ============================================================
# 3. Validation Logic
# ============================================================

def validate_lane_geometry(lane_c: Coefficients, road_c: Coefficients, h: int, side: str, w: int) -> bool:
    """
    Geometric Check: Is the lane inside the road corridor?
    """
    # Check at multiple vertical points
    for y_fac in [0.5, 0.75, 0.95]:
        y = int(h * y_fac)
        lx = lane_c[0]*y**2 + lane_c[1]*y + lane_c[2]
        rx = road_c[0]*y**2 + road_c[1]*y + road_c[2]
        
        # 1. Boundary Check
        if side == "left":
            if lx < rx: return False # Left lane is outside (left of) road edge
        else:
            if lx > rx: return False # Right lane is outside (right of) road edge
            
        # 2. Sanity Check (Center Crossing)
        if side == "left" and lx > w * 0.6: return False
        if side == "right" and lx < w * 0.4: return False
            
    return True

# ============================================================
# 4. Master Pipeline
# ============================================================

class LanePerception:
    def __init__(self):
        self.left_state = LaneState()
        self.right_state = LaneState()

    def process_side(self, lane_mask, road_mask, side: str):
        h, w = lane_mask.shape
        state = self.left_state if side == "left" else self.right_state
        
        # --- A. Extraction ---
        ly, lx = np.nonzero(lane_mask)
        ry, rx = extract_road_edge_safe(road_mask, side)
        
        lane_model = robust_polyfit(ly, lx, w)
        road_model = robust_polyfit(ry, rx, w)
        
        final_model = None
        current_source = LaneSource.NONE
        
        # --- B. Validation & Logic Switch ---
        
        # 1. Try to validate the detected Lane
        is_lane_valid = False
        if lane_model is not None:
            # Check 1: Geometry (if road exists)
            if road_model is not None:
                if validate_lane_geometry(lane_model, road_model, h, side, w):
                    is_lane_valid = True
            # Check 2: Temporal (if road missing) - Fixing the "Blind Trust" bug
            else:
                if state.check_temporal_consistency(lane_model):
                    is_lane_valid = True
                else:
                    is_lane_valid = False # Reject sudden hallucination if no road to verify

        # 2. Select Output
        if is_lane_valid:
            final_model = lane_model
            current_source = LaneSource.DETECTED
            state.missing_frames = 0
            
            # Learn metrics if we have good road data too
            if road_model is not None:
                state.update_learning(lane_model, road_model, h)

        elif road_model is not None:
            # Fallback: Synthesize from Road
            fallback = state.get_fallback_model(road_model)
            if fallback is not None:
                # Check consistency of the fallback too!
                if state.check_temporal_consistency(fallback, threshold=80):
                    final_model = fallback
                    current_source = LaneSource.ROAD_REF
                    state.missing_frames = 0
                
        # 3. Last Resort: Memory
        if final_model is None:
            if state.last_coeffs is not None and state.missing_frames < state.MAX_AGE:
                final_model = state.last_coeffs
                current_source = LaneSource.MEMORY
                state.missing_frames += 1
            else:
                current_source = LaneSource.NONE
        
        # Update History
        if final_model is not None:
            state.last_coeffs = final_model
            
        return final_model, current_source

    def perceive(self, lane_mask, road_mask) -> Tuple[Dict, Dict]:
        """
        Returns dictionaries containing points and metadata for controller.
        """
        h, w = lane_mask.shape
        mid = w // 2
        
        # Split & Process
        l_coeffs, l_src = self.process_side(lane_mask[:, :mid], road_mask[:, :mid], "left")
        r_coeffs, r_src = self.process_side(lane_mask[:, mid:], road_mask[:, mid:], "right")
        
        # Generate Output Packets
        left_out = {
            "points": self._gen(l_coeffs, h),
            "source": l_src,
            "coeffs": l_coeffs
        }
        
        right_out = {
            "points": self._gen(r_coeffs, h, x_off=mid),
            "source": r_src,
            "coeffs": r_coeffs
        }
        
        return left_out, right_out

    def _gen(self, coeffs, h, x_off=0):
        if coeffs is None: return []
        a, b, c = coeffs
        ys = np.linspace(h*0.45, h-1, 40)
        xs = a*ys**2 + b*ys + c + x_off
        return np.column_stack((xs, ys)).astype(int).tolist()