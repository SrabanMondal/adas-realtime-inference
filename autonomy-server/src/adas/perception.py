import cv2
import numpy as np
from typing import List, Tuple, Optional

# --- Types & Config ---
LanePoints = List[List[int]] 
Coefficients = Tuple[float, float, float]

# --- 1. Smoothing (EMA + Kalman) ---
# We keep the Kalman Filter simple for smoothing the coefficients over time
class PolynomialKalmanFilter:
    def __init__(self):
        self.state = np.zeros(3, dtype=np.float32)
        self.P = np.eye(3, dtype=np.float32) * 1.0 
        # Low Process Noise (Q) = Trust history (Smoothness)
        self.Q = np.eye(3, dtype=np.float32) * 0.001     
        # High Measurement Noise (R) = Don't jump for single-frame outliers
        self.R = np.eye(3, dtype=np.float32) * 10.0       
        self.F = np.eye(3, dtype=np.float32) 
        self.H = np.eye(3, dtype=np.float32) 
        self.initialized = False
        self.missed_frames = 0
        self.avg_coeffs = None # For Exponential Moving Average

    def update(self, measured_coeffs: Optional[Coefficients]) -> Coefficients:
        if measured_coeffs is None:
            self.missed_frames += 1
            if self.missed_frames > 30: # Reset if lost for too long
                self.initialized = False
                self.avg_coeffs = None
            return self._get_smoothed_output() # Return last known good state

        z = np.array(measured_coeffs, dtype=np.float32)
        self.missed_frames = 0

        if not self.initialized:
            self.state = z
            self.avg_coeffs = z
            self.initialized = True
            return tuple(self.state)

        # Kalman Predict & Update
        x_pred = self.F @ self.state
        P_pred = self.F @ self.P @ self.F.T + self.Q
        y = z - (self.H @ x_pred)
        S = self.H @ P_pred @ self.H.T + self.R
        K = P_pred @ self.H.T @ np.linalg.inv(S)
        self.state = x_pred + (K @ y)
        self.P = (np.eye(3) - (K @ self.H)) @ P_pred

        return self._get_smoothed_output()

    def _get_smoothed_output(self) -> Coefficients:
        # Exponential Moving Average for visual buttery smoothness
        current = self.state
        if self.avg_coeffs is None:
            self.avg_coeffs = current
        else:
            # alpha 0.15 = mostly history, slow smooth updates
            alpha = 0.15
            self.avg_coeffs = (alpha * current) + ((1 - alpha) * self.avg_coeffs)
        
        return (float(self.avg_coeffs[0]), float(self.avg_coeffs[1]), float(self.avg_coeffs[2]))

left_kf = PolynomialKalmanFilter()
right_kf = PolynomialKalmanFilter()


# --- 2. The Core Logic (Your Algorithm) ---

def _get_row_centroid(mask_row: np.ndarray) -> Optional[int]:
    """ Returns the middle column index of non-zero pixels in a row. """
    indices = np.flatnonzero(mask_row)
    if len(indices) == 0:
        return None
    return int(np.mean(indices))

def _is_road_boundary_valid(road_x: int, y: int, lane_points: dict, check_range: int = 50) -> bool:
    """
    Checks if a road boundary point aligns with the trajectory of existing true lane points.
    It looks for a true lane point above and below and checks deviation.
    """
    # 1. Find nearest True Lane Pixel ABOVE
    y_above = None
    for search_y in range(y - 1, y - check_range, -1):
        if search_y in lane_points:
            y_above = search_y
            break
            
    # 2. Find nearest True Lane Pixel BELOW
    y_below = None
    for search_y in range(y + 1, y + check_range):
        if search_y in lane_points:
            y_below = search_y
            break
    
    # Logic:
    # If we have BOTH top and bottom, interpolate and check deviation
    if y_above is not None and y_below is not None:
        x_above = lane_points[y_above]
        x_below = lane_points[y_below]
        
        # Linear Interpolation: expected_x
        ratio = (y - y_above) / (y_below - y_above)
        expected_x = x_above + ratio * (x_below - x_above)
        
        # Allow small deviation (e.g., 20px)
        return abs(road_x - expected_x) < 20

    # If we only have ONE (start or end of line), check simple proximity
    if y_above is not None:
        return abs(road_x - lane_points[y_above]) < 25
    if y_below is not None:
        return abs(road_x - lane_points[y_below]) < 25

    # If no lane points nearby, we can't trust the road boundary blindly
    return False

def _extract_hybrid_points(lane_mask: np.ndarray, road_mask: np.ndarray, is_left_lane: bool) -> List[List[int]]:
    h, w = lane_mask.shape
    
    # 1. First Pass: Collect ALL True Lane Points (The "Gold Standard")
    # We store them in a dict {y: x} for fast lookup during validation
    true_lane_points = {}
    
    # Scan every Nth row
    scan_step = 5 
    # Only scan bottom 60% of screen typically
    start_row = int(h * 0.4)
    
    for y in range(start_row, h, scan_step):
        # Handle Lane Mask
        if is_left_lane:
            # For left lane, search left side of histogram/image generally
            # But here we just take the row centroid of the specific mask provided
            x_lane = _get_row_centroid(lane_mask[y, :])
        else:
            x_lane = _get_row_centroid(lane_mask[y, :])
            
        if x_lane is not None:
            true_lane_points[y] = x_lane

    # 2. Second Pass: Fill gaps with Road Boundaries
    final_points = []
    
    for y in range(start_row, h, scan_step):
        # A. Try True Lane Pixel
        if y in true_lane_points:
            final_points.append([true_lane_points[y], y])
            continue
            
        # B. Fallback: Try Road Boundary
        row = road_mask[y, :]
        nonzero = np.flatnonzero(row)
        
        if len(nonzero) > 0:
            if is_left_lane:
                x_road = nonzero[0] # Left edge of road
            else:
                x_road = nonzero[-1] # Right edge of road
                
            # Rule 1: Ignore if touching frame boundary
            margin = 5
            if x_road <= margin or x_road >= (w - margin):
                continue
                
            # Rule 2: Check Deviation against True Lane Pixels
            if _is_road_boundary_valid(x_road, y, true_lane_points):
                final_points.append([x_road, y])
                
    return final_points


# --- 3. Fitting & Generation ---

def _fit_poly(points: List[List[int]]) -> Optional[Coefficients]:
    if len(points) < 5: return None # Need very few points now since they are high quality
    pts = np.array(points)
    try:
        # Fit y vs x (x = f(y))
        fit = np.polyfit(pts[:, 1], pts[:, 0], 2)
        return (fit[0], fit[1], fit[2])
    except:
        return None

def _generate_curve(coeffs: Coefficients, height: int) -> LanePoints:
    a, b, c = coeffs
    # Generate smooth visualization points
    plot_y = np.linspace(height * 0.45, height - 1, 50)
    plot_x = a * plot_y**2 + b * plot_y + c
    
    points = []
    for px, py in zip(plot_x, plot_y):
        points.append([int(px), int(py)])
    return points


# --- 4. Master Function ---

def perceive_lanes(lane_mask: np.ndarray, road_mask: np.ndarray) -> Tuple[LanePoints, LanePoints]:
    h, w = lane_mask.shape

    # 1. Extract Points using the "Hybrid Row-by-Row" logic
    # We assume lane_mask contains BOTH lanes? 
    # Usually strictly better if lane_mask is split, but if it's one mask:
    # We split it by screen center for the "True Lane" search.
    
    # Left Side
    l_mask_half = lane_mask.copy()
    l_mask_half[:, w//2:] = 0
    l_points_raw = _extract_hybrid_points(l_mask_half, road_mask, is_left_lane=True)
    
    # Right Side
    r_mask_half = lane_mask.copy()
    r_mask_half[:, :w//2] = 0
    r_points_raw = _extract_hybrid_points(r_mask_half, road_mask, is_left_lane=False)

    # 2. Fit Polynomials
    l_coeffs = _fit_poly(l_points_raw)
    r_coeffs = _fit_poly(r_points_raw)

    # 3. Smooth Updates (Kalman + EMA)
    l_smooth_coeffs = left_kf.update(l_coeffs)
    r_smooth_coeffs = right_kf.update(r_coeffs)

    # 4. Generate Curve
    final_left = _generate_curve(l_smooth_coeffs, h)
    final_right = _generate_curve(r_smooth_coeffs, h)

    return final_left, final_right