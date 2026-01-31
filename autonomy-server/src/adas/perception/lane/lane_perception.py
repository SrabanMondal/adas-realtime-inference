import numpy as np
import cv2
from typing import List, Tuple, Optional

LanePoints = List[List[int]]
Coefficients = Tuple[float, float, float]

class PolyEMA:
    """ Exponential Moving Average for polynomial coefficients. """
    def __init__(self, alpha=0.25):
        self.alpha = alpha
        self.prev: Optional[np.ndarray] = None

    def update(self, coeffs: Optional[Coefficients]) -> Optional[Coefficients]:
        if coeffs is None:
            return tuple(self.prev) if self.prev is not None else None
        
        c = np.array(coeffs, dtype=np.float32)
        if self.prev is None:
            self.prev = c
        else:
            self.prev = self.alpha * c + (1 - self.alpha) * self.prev
        return tuple(self.prev)

left_ema = PolyEMA(0.3)
right_ema = PolyEMA(0.3)

# ------------------ Vectorized Core ------------------

def _get_lane_points_vectorized(
    lane_mask_area: np.ndarray, 
    road_mask_area: np.ndarray, 
    is_left_lane: bool,
    offset_px: int = 15,    # Distance to shift road edge to align with lane center
    max_dist_from_true: int = 60 # "Not too far from true lane" threshold
) -> np.ndarray:
    
    # 1. Extract TRUE Lane points (Vectorized)
    # ------------------------------------------------
    true_y, true_x = np.nonzero(lane_mask_area)
    
    # 2. Extract ROAD EDGE points (Vectorized)
    # ------------------------------------------------
    # Use morphological gradient to instantly find all edges
    kernel = np.ones((3,3), np.uint8)
    road_edges = cv2.morphologyEx(road_mask_area, cv2.MORPH_GRADIENT, kernel)
    
    # Remove Bottom/Side Frame Edges (Your specific constraint)
    h, w = road_mask_area.shape
    # Create a mask that is 0 at borders and 1 inside
    border_mask = np.zeros_like(road_edges)
    border_mask[0:h-1, 1:w-1] = 1 # Skip bottom row (h-1) and side cols
    
    clean_edges = cv2.bitwise_and(road_edges, road_edges, mask=border_mask)
    edge_y, edge_x = np.nonzero(clean_edges)
    
    if len(edge_y) == 0:
        # If no road edges, return just true points or empty
        if len(true_x) == 0: return np.empty((0, 2))
        return np.column_stack((true_x, true_y))

    # 3. Apply Geometry Correction (The Offset)
    # ------------------------------------------------
    # Road mask is the "inner" area. 
    # Left Lane: Road is to the Right -> Edge is Inner Right -> Shift Left (-offset)
    # Right Lane: Road is to the Left -> Edge is Inner Left -> Shift Right (+offset)
    shift = -offset_px if is_left_lane else offset_px
    edge_x = edge_x + shift

    # 4. Filter Edges: "Is it close to a True Lane?"
    # ------------------------------------------------
    # If we have NO true lane points, we can't validate the road edges safely.
    # But if we have SOME, we use them as an anchor.
    if len(true_x) > 0:
        # Create a distance map from True Lane Points
        # Invert lane mask: 0=Lane, 255=Background
        # (We need a full size mask for distanceTransform)
        full_mask = np.zeros_like(lane_mask_area)
        full_mask[true_y, true_x] = 1
        inv_mask = cv2.bitwise_not(full_mask * 255)
        
        # dist_map[y,x] = distance to nearest true lane pixel
        dist_map = cv2.distanceTransform(inv_mask, cv2.DIST_L2, 5)
        
        # Check distance of every road edge pixel to the nearest true lane pixel
        dists = dist_map[edge_y, (edge_x - shift)] # check original coords
        
        # Keep only edges that are "pseudo lane pixels" (close to established trend)
        valid_indices = dists < max_dist_from_true
        
        final_edge_x = edge_x[valid_indices]
        final_edge_y = edge_y[valid_indices]
    else:
        # Fallback: If no true lane detected, trust the road edge strictly?
        # Or reject? Usually safer to reject or use stricter constraints.
        # For now, we accept them if they look reasonable (heuristic).
        final_edge_x = edge_x
        final_edge_y = edge_y

    # 5. Combine High Confidence (True) and Low Confidence (Road)
    # ------------------------------------------------
    all_x = np.concatenate((true_x, final_edge_x))
    all_y = np.concatenate((true_y, final_edge_y))
    
    return np.column_stack((all_x, all_y))

def _fit_poly_robust(points: np.ndarray) -> Optional[Coefficients]:
    if points.shape[0] < 10: # Need reasonable number of points
        return None
        
    x = points[:, 0]
    y = points[:, 1]
    
    try:
        # fit generic polynomial
        f = np.polyfit(y, x, 2)
        return (float(f[0]), float(f[1]), float(f[2]))
    except:
        return None

def _generate_curve(coeffs: Optional[Coefficients], h: int) -> LanePoints:
    if coeffs is None:
        return []
    a, b, c = coeffs
    # Generate y values from bottom to roi_start
    ys = np.linspace(h * 0.45, h - 1, 40)
    xs = a * ys**2 + b * ys + c
    
    # Cast to int and list format
    return np.column_stack((xs, ys)).astype(int).tolist()

# ------------------ Master Pipeline ------------------

def perceive_lanes(lane_mask: np.ndarray, road_mask: np.ndarray) -> Tuple[LanePoints, LanePoints]:
    h, w = lane_mask.shape
    mid = w // 2

    # Split masks (Standard approach, assumes car is roughly centered)
    # Note: Copying is safer to avoid modifying original masks
    l_lane = lane_mask[:, :mid]
    l_road = road_mask[:, :mid]
    
    r_lane = lane_mask[:, mid:]
    r_road = road_mask[:, mid:]

    # Extract points using Vectorized Logic
    # Note: Offset is critical. Adjust '15' based on your camera resolution/calibration
    l_pts = _get_lane_points_vectorized(l_lane, l_road, is_left_lane=True, offset_px=15)
    r_pts = _get_lane_points_vectorized(r_lane, r_road, is_left_lane=False, offset_px=15)

    # Adjust Right Lane X coordinates (since we cropped the image)
    if len(r_pts) > 0:
        r_pts[:, 0] += mid

    # Fit Polynomials
    l_fit = _fit_poly_robust(l_pts)
    r_fit = _fit_poly_robust(r_pts)

    # Smooth
    l_s = left_ema.update(l_fit)
    r_s = right_ema.update(r_fit)

    # Generate Output
    left_curve = _generate_curve(l_s, h)
    right_curve = _generate_curve(r_s, h)

    return left_curve, right_curve