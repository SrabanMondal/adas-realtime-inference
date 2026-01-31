import cv2
import numpy as np
from typing import List, Tuple, Optional

# --- Types & Config ---
LanePoints = List[List[int]] 
Coefficients = Tuple[float, float, float]
LANE_WEIGHT = 2  # Lane mask points count 2x compared to road boundary

# --- 1. Kalman Filter Class (The Memory) ---
class PolynomialKalmanFilter:
    def __init__(self):
        # State: [a, b, c] for equation x = ay^2 + by + c
        self.state = np.zeros(3, dtype=np.float32)
        self.P = np.eye(3, dtype=np.float32) * 1.0       # Error Covariance
        self.Q = np.eye(3, dtype=np.float32) * 0.005     # Process Noise (Smoothness)
        self.R = np.eye(3, dtype=np.float32) * 0.2       # Measurement Noise
        self.F = np.eye(3, dtype=np.float32)             # Transition Matrix
        self.H = np.eye(3, dtype=np.float32)             # Measurement Matrix
        self.initialized = False
        self.missed_frames = 0

    def update(self, measured_coeffs: Optional[Coefficients]) -> Coefficients:
        # If we lost the lane, rely on prediction (up to a limit)
        if measured_coeffs is None:
            self.missed_frames += 1
            if self.missed_frames > 10: # Reset if lost for too long
                self.initialized = False
            return (float(self.state[0]), float(self.state[1]), float(self.state[2]))

        z = np.array(measured_coeffs, dtype=np.float32)
        self.missed_frames = 0

        # Initialize if first time
        if not self.initialized:
            self.state = z
            self.initialized = True
            return tuple(self.state)

        # Predict
        x_pred = self.F @ self.state
        P_pred = self.F @ self.P @ self.F.T + self.Q

        # Update
        y = z - (self.H @ x_pred)
        S = self.H @ P_pred @ self.H.T + self.R
        K = P_pred @ self.H.T @ np.linalg.inv(S)
        
        self.state = x_pred + (K @ y)
        self.P = (np.eye(3) - (K @ self.H)) @ P_pred

        return (float(self.state[0]), float(self.state[1]), float(self.state[2]))

# Instantiate filters as module-level singletons
left_kf = PolynomialKalmanFilter()
right_kf = PolynomialKalmanFilter()


# --- 2. Raw Data Extraction Helpers ---

def _get_road_boundaries(road_mask: np.ndarray) -> Tuple[List[List[int]], List[List[int]]]:
    """ Extract raw [x,y] points from road boundaries. """
    h, w = road_mask.shape
    contours, _ = cv2.findContours(road_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours: return [], []
    
    largest_contour = max(contours, key=cv2.contourArea)
    points = largest_contour.squeeze()
    if points.ndim != 2: return [], []

    left_pts, right_pts = [], []
    
    # Decimate (every 5th point) and Filter (Bottom 60% only)
    for x, y in points[::5]:
        if y < h * 0.4: continue
        if x < w // 2:
            left_pts.append([x, y])
        else:
            right_pts.append([x, y])
            
    return left_pts, right_pts

def _extract_lane_pixels(lane_mask: np.ndarray) -> Tuple[List[List[int]], List[List[int]]]:
    """ Extract raw [x,y] pixels from the lane mask using histogram peaks. """
    h, w = lane_mask.shape
    
    # Simple cleaning locally if needed, though main.py should pass clean masks
    # We assume mask is binary 0/1 here.
    
    # Histogram Search
    histogram = np.sum(lane_mask[h//2:, :], axis=0)
    midpoint = int(histogram.shape[0] / 2)
    left_peak = np.argmax(histogram[:midpoint])
    right_peak = np.argmax(histogram[midpoint:]) + midpoint

    if histogram[left_peak] < 50: left_peak = None
    if histogram[right_peak] < 50: right_peak = None

    nonzero = lane_mask.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    margin = 80
    left_pts, right_pts = [], []

    if left_peak is not None:
        valid_l = ((nonzerox > (left_peak - margin)) & (nonzerox < (left_peak + margin)))
        # Stack into [[x,y], [x,y]...]
        if np.any(valid_l):
            left_pts = np.column_stack((nonzerox[valid_l], nonzeroy[valid_l])).tolist()

    if right_peak is not None:
        valid_r = ((nonzerox > (right_peak - margin)) & (nonzerox < (right_peak + margin)))
        if np.any(valid_r):
            right_pts = np.column_stack((nonzerox[valid_r], nonzeroy[valid_r])).tolist()

    return left_pts, right_pts


# --- 3. Fitting & Generation Logic ---

def _fit_poly_coefficients(points: List[List[int]]) -> Optional[Coefficients]:
    """ Fits x = ay^2 + by + c to a list of [x,y] points. """
    if len(points) < 50: return None
    
    pts = np.array(points)
    x = pts[:, 0]
    y = pts[:, 1]
    
    try:
        # Fit 2nd degree polynomial
        fit = np.polyfit(y, x, 2)
        return (fit[0], fit[1], fit[2])
    except np.linalg.LinAlgError:
        return None

def _generate_smooth_points(coeffs: Coefficients, height: int) -> LanePoints:
    """ Generates clean points for frontend from coeffs. """
    a, b, c = coeffs
    
    # Generate 10 points for the bottom half of the screen
    plot_y = np.linspace(height // 2, height - 1, num=10)
    plot_x = a * plot_y**2 + b * plot_y + c
    
    points = []
    for px, py in zip(plot_x, plot_y):
        if 0 <= px < 640:
            points.append([int(px), int(py)])
    return points


# --- 4. The Master Function (Public API) ---

def perceive_lanes(lane_mask: np.ndarray, road_mask: np.ndarray) -> Tuple[LanePoints, LanePoints]:
    """
    1. Extracts raw pixels from Binary Lane Mask of original image resolution
    2. Extracts raw boundaries Binary from Road Mask of original image resolution
    3. Fuses them (Weighted)
    4. Fits Polynomial -> Smooths with Kalman Filter
    5. Returns Points
    """
    h, w = lane_mask.shape

    # 1. Extract Raw Data
    l_lane_raw, r_lane_raw = _extract_lane_pixels(lane_mask)
    l_road_raw, r_road_raw = _get_road_boundaries(road_mask)

    # 2. Fuse Data (Concatenate Lists)
    # We duplicate lane points to give them higher weight
    l_fused = (l_lane_raw * LANE_WEIGHT) + l_road_raw
    r_fused = (r_lane_raw * LANE_WEIGHT) + r_road_raw

    # 3. Fit Coefficients (Noisy)
    l_coeffs_noisy = _fit_poly_coefficients(l_fused)
    r_coeffs_noisy = _fit_poly_coefficients(r_fused)

    # 4. Kalman Smoothing (Memory)
    l_coeffs_smooth = left_kf.update(l_coeffs_noisy)
    r_coeffs_smooth = right_kf.update(r_coeffs_noisy)

    # 5. Generate Final Points
    final_left = _generate_smooth_points(l_coeffs_smooth, h)
    final_right = _generate_smooth_points(r_coeffs_smooth, h)

    return final_left, final_right