import cv2
import numpy as np

def clean_road_mask(mask: np.ndarray) -> np.ndarray:
    """
    Cleans the raw binary road mask using morphological operations
    to fill holes and smooth jagged edges.
    """
    # 1. Define Kernels
    # A larger kernel fills larger gaps but might round off sharp corners too much
    # 5x5 is a sweet spot for 640x480 resolution
    kernel_close = np.ones((5, 5), np.uint8)
    kernel_open = np.ones((3, 3), np.uint8)

    # 2. Morphological Closing (Dilation -> Erosion)
    # This fills black holes inside the white road blob (e.g., passing cars, shadows)
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
    
    # 3. Morphological Opening (Erosion -> Dilation)
    # This removes white noise specs outside the road (e.g., random false positives)
    cleaned = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open)

    return cleaned