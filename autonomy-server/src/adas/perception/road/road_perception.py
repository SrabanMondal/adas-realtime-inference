import numpy as np
import cv2
from typing import List, Tuple, Optional

Point = Tuple[int, int]

class RoadPerception:
    def __init__(
        self,
        ema_alpha_mask: float = 0.2,      # How fast geometry adapts
        min_row_width: int = 20,          # Ignore very thin road rows
        downsample_factor: int = 2,       # Downsample mask for EMA
        morph_kernel_size: int = 3,       # Optional morphological kernel size
        row_step: int = 4,                # Step for centerline row sampling
        enable_morph: bool = True         # Enable/disable morphology
    ):
        self.ema_alpha_mask: float = ema_alpha_mask
        self.min_row_width: int = min_row_width
        self.downsample_factor: int = downsample_factor
        self.morph_kernel_size: int = morph_kernel_size
        self.row_step: int = row_step
        self.enable_morph: bool = enable_morph

        self.slow_mask: Optional[np.ndarray] = None

    # --------------------------------------------------------
    # Public API
    # --------------------------------------------------------
    def process(self, road_mask: np.ndarray) -> dict:
        """
        Process a binary road mask to produce centerline points with confidence.
        
        Args:
            road_mask: uint8 binary mask (0/1 or 0/255)
        
        Returns:
            dict with keys:
                - center_points: List of (x,y) tuples
                - confidence: float (0.0–1.0)
                - slow_mask: EMA-smoothed mask (binary)
        """
        # 1. Clean mask (optional)
        fast_mask = self._clean_mask(road_mask) if self.enable_morph else (road_mask > 0).astype(np.uint8)

        # 2. Downsample EMA for speed
        slow_mask = self._update_slow_mask(fast_mask)

        # 3. Extract centerline (vectorized)
        center_pts = self._extract_centerline(slow_mask)

        # 4. Confidence heuristic
        confidence = min(1.0, len(center_pts) / 30.0)

        return {
            "center_points": center_pts,
            "confidence": confidence,
            "slow_mask": slow_mask
        }

    # --------------------------------------------------------
    # Internals
    # --------------------------------------------------------
    def _clean_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Morphological cleaning to remove small holes & noise.
        Faster than large kernel or multiple operations.
        """
        m: np.ndarray = (mask > 0).astype(np.uint8)
        kernel = np.ones((self.morph_kernel_size, self.morph_kernel_size), np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)
        return m

    def _update_slow_mask(self, fast: np.ndarray) -> np.ndarray:
        """
        EMA smoothing of mask for geometric stability.
        Downsamples mask to reduce computation.
        """
        # Downsample mask
        h, w = fast.shape
        small_h, small_w = h // self.downsample_factor, w // self.downsample_factor
        fast_small: np.ndarray = cv2.resize(fast, (small_w, small_h), interpolation=cv2.INTER_NEAREST).astype(np.float32)

        # Initialize or update EMA
        if self.slow_mask is None:
            self.slow_mask = fast_small
        else:
            self.slow_mask = (1.0 - self.ema_alpha_mask) * self.slow_mask + self.ema_alpha_mask * fast_small

        # Upsample back to original size and binarize
        slow_bin = cv2.resize(self.slow_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        slow_bin = (slow_bin > 0.5).astype(np.uint8)
        return slow_bin

    def _extract_centerline(self, mask: np.ndarray) -> List[Point]:
        """
        Vectorized extraction of rough road centerline.
        Skips top 30% and bottom 10% of frame.
        """
        h, w = mask.shape
        y_start = int(h * 0.4)   # bottom 60% of frame

        ys = np.arange(y_start, h, self.row_step)
        mask_sub = mask[ys]  # shape: (num_rows, width)

        # Find left/right bounds for each row
        lefts = np.argmax(mask_sub > 0, axis=1)
        rights = mask_sub.shape[1] - 1 - np.argmax(np.flip(mask_sub, axis=1) > 0, axis=1)
        widths = rights - lefts

        # Only keep rows with sufficient width
        valid = widths >= self.min_row_width
        center_xs = (lefts[valid] + rights[valid]) // 2
        valid_ys = ys[valid]

        center_pts: List[Point] = list(zip(center_xs, valid_ys))
        return center_pts
