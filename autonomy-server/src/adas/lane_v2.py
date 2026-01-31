import numpy as np
import cv2
from typing import Tuple, List, Optional

# Type aliases for clarity
LanePoints = List[List[int]]
Coefficients = Tuple[float, float, float]

# ---------------------------------------------------------
# 1. Temporal Smoothing Class
# ---------------------------------------------------------
class PolyEMA:
    """ 
    Exponential Moving Average for polynomial coefficients. 
    Helps stabilize jitter between frames.
    """
    def __init__(self, alpha=0.25):
        self.alpha = alpha
        self.prev: Optional[np.ndarray] = None

    def update(self, coeffs: Optional[Coefficients]) -> Optional[Coefficients]:
        # If detection failed, hold previous value (or return None if no history)
        if coeffs is None:
            return tuple(self.prev) if self.prev is not None else None
        
        c = np.array(coeffs, dtype=np.float32)
        if self.prev is None:
            self.prev = c
        else:
            self.prev = self.alpha * c + (1 - self.alpha) * self.prev
        
        return tuple(self.prev)

# Instantiate smoothers for Left and Right lanes
left_ema = PolyEMA(alpha=0.3)
right_ema = PolyEMA(alpha=0.3)

# ---------------------------------------------------------
# 2. Math Helpers (Vectorized)
# ---------------------------------------------------------
def fast_robust_fit(y: np.ndarray, x: np.ndarray) -> Optional[Coefficients]:
    """
    A faster alternative to RANSAC.
    Uses a 2-pass Standard Deviation filter to reject noise.
    """
    if len(x) < 5: 
        return None
    
    # Pass 1: Initial fit
    try:
        c_init = np.polyfit(y, x, 2)
    except np.linalg.LinAlgError:
        return None
        
    # Calculate error (residuals)
    pred = c_init[0]*y**2 + c_init[1]*y + c_init[2]
    error = np.abs(x - pred)
    
    # Filter: Keep points within 1.5 standard deviations
    std = np.std(error)
    # +2.0 adds a small buffer so we don't discard everything if the line is perfect (std~0)
    mask = error < (1.5 * std + 2.0)
    
    # Pass 2: Refined fit on inliers
    if np.sum(mask) < 5: 
        return None
        
    c_final = np.polyfit(y[mask], x[mask], 2)
    return (float(c_final[0]), float(c_final[1]), float(c_final[2]))

def _get_road_edges(road_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts road boundary edges while rejecting frame borders.
    """
    # 1. Morphological Gradient to find edges
    kernel = np.ones((3,3), np.uint8)
    edges = cv2.morphologyEx(road_mask, cv2.MORPH_GRADIENT, kernel)
    
    # 2. Safety: Reject edges that touch the image border 
    # (prevents "road mask hitting the wall" bugs)
    h, w = road_mask.shape
    border_mask = np.zeros_like(edges)
    border_mask[1:h-1, 1:w-1] = 1 # Keep only internal pixels
    
    clean_edges = cv2.bitwise_and(edges, edges, mask=border_mask)
    return np.nonzero(clean_edges)

def _generate_curve_points(coeffs: Optional[Coefficients], h: int, steps=40) -> LanePoints:
    """Generates (x,y) points from polynomial coefficients for visualization."""
    if coeffs is None:
        return []
    
    a, b, c = coeffs
    # Generate points from 40% height down to bottom
    ys = np.linspace(h * 0.40, h - 1, steps)
    xs = a * ys**2 + b * ys + c
    
    # Format as list of [x, y]
    pts = np.column_stack((xs, ys)).astype(int)
    return pts.tolist()

# ---------------------------------------------------------
# 3. Core Logic: The Fused Perceiver
# ---------------------------------------------------------
def perceive_one_side_fused(lane_mask: np.ndarray, road_mask: np.ndarray, is_left: bool) -> Optional[Coefficients]:
    h, w = lane_mask.shape
    
    # --- A. Extract Raw Points ---
    ly, lx = np.nonzero(lane_mask)
    
    # --- B. Primary Fit (Lane Only) ---
    lane_model = fast_robust_fit(ly, lx)
    
    # --- C. Scenario 1: Lane Found (Try to improve/extend it) ---
    if lane_model is not None:
        # Check for "Bottom Gap" (e.g., hood to 10 meters is missing)
        lowest_lane_y = np.max(ly)
        gap_threshold_y = h - 60  # If lane ends 60px above bottom, we have a gap
        
        if lowest_lane_y < gap_threshold_y:
            # Get Road Edges
            ey, ex = _get_road_edges(road_mask)
            
            # Mask only the bottom gap area (below the lowest lane point)
            gap_mask = ey > lowest_lane_y
            gap_y = ey[gap_mask]
            gap_x = ex[gap_mask]
            
            if len(gap_y) > 10:
                # DYNAMIC OFFSET CALCULATION
                # 1. Project where the lane *should* be at the cutoff point
                proj_x = lane_model[0]*lowest_lane_y**2 + lane_model[1]*lowest_lane_y + lane_model[2]
                
                # 2. Find where the road edge *actually* is near that cutoff
                # Look at road pixels within 5 rows of the cutoff
                nearby_mask = np.abs(gap_y - lowest_lane_y) < 10
                
                if np.any(nearby_mask):
                    real_road_x = np.mean(gap_x[nearby_mask])
                    
                    # 3. Calculate Shift needed to glue them together
                    # Shift = (Target Lane X) - (Road Edge X)
                    current_offset = proj_x - real_road_x
                    
                    # Apply shift to road points
                    shifted_ex = gap_x + current_offset
                    
                    # 4. Merge Data: True Lane (Top) + Shifted Road (Bottom)
                    combined_y = np.concatenate((ly, gap_y))
                    combined_x = np.concatenate((lx, shifted_ex))
                    
                    # Refit on the cleaner, longer dataset
                    return fast_robust_fit(combined_y, combined_x)
                    
        return lane_model

    # --- D. Scenario 2: Lane Lost (Fallback to Road Edge) ---
    else:
        # Strictly use road edge with a "Safe Default" offset
        ey, ex = _get_road_edges(road_mask)
        road_model = fast_robust_fit(ey, ex)
        
        if road_model is not None:
            # We don't have a true lane to calculate dynamic offset, 
            # so we use a constant heuristic.
            # NOTE: This is less accurate but safe for "Limp Home" mode.
            default_offset = -30 if is_left else 30
            
            # Adjust the 'c' term (intercept)
            # Warning: This is a 2D shift, not perfect perspective, but okay for fallback.
            return (road_model[0], road_model[1], road_model[2] + default_offset)
            
    return None

# ---------------------------------------------------------
# 4. Master Function
# ---------------------------------------------------------
def perceive_lanes(lane_mask: np.ndarray, road_mask: np.ndarray) -> Tuple[LanePoints, LanePoints]:
    """
    Main entry point.
    Input: Binary masks (0 or 255/1) for Lane Lines and Road Drivable Area.
    Output: smoothed left and right lane points [[x,y], ...]
    """
    h, w = lane_mask.shape
    mid = w // 2

    # 1. Split Image (Left / Right)
    # Using copies to ensure memory safety when modifying
    l_lane = lane_mask[:, :mid]
    l_road = road_mask[:, :mid]
    
    r_lane = lane_mask[:, mid:]
    r_road = road_mask[:, mid:]

    # 2. Calculate Models (Fused Logic)
    l_coeffs = perceive_one_side_fused(l_lane, l_road, is_left=True)
    r_coeffs = perceive_one_side_fused(r_lane, r_road, is_left=False)

    # 3. Temporal Smoothing
    l_smooth = left_ema.update(l_coeffs)
    r_smooth = right_ema.update(r_coeffs)

    # 4. Generate Points for Output
    l_pts = _generate_curve_points(l_smooth, h)
    r_pts = _generate_curve_points(r_smooth, h)

    # 5. Fix Right Lane Coordinates 
    # (Since we processed the right half at x=0, we must add 'mid' back)
    if r_pts:
        for p in r_pts:
            p[0] += mid

    return l_pts, r_pts