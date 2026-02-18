import cv2
import numpy as np

def clean_road_mask(mask: np.ndarray) -> np.ndarray:
    """
    Optimized for 320x320 resolution. 
    Fills internal holes without erasing the distant horizon line.
    """
    # Use smaller kernels for 320px to preserve distant road details
    kernel_small = np.ones((3, 3), np.uint8)

    # 1. Closing (Fills holes/shadows)
    # This is critical for YOLOP to ignore shadows under cars
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_small)
    
    # 2. Opening (Removes noise)
    # We do this after closing to ensure we don't 'eat' the narrow road ahead
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small)

    return mask