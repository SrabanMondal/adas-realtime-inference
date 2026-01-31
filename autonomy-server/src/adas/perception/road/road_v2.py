import numpy as np
from typing import List, Tuple, Optional

Point = Tuple[int, int]


class RoadPerception:
    """
    Ultra-fast road perception for single-lane driving.

    Design:
    - No morphology
    - No mask EMA
    - Geometry-only temporal smoothing
    - Row-wise span extraction
    """

    def __init__(
        self,
        ema_alpha: float = 0.25,     # Geometry smoothing
        min_row_width: int = 20,     # Ignore narrow rows
        row_step: int = 4            # Vertical sampling stride
    ):
        self.ema_alpha = ema_alpha
        self.min_row_width = min_row_width
        self.row_step = row_step

        # Temporal geometry state
        self.left_ema: Optional[np.ndarray] = None
        self.right_ema: Optional[np.ndarray] = None

    # =========================================================
    # Public API
    # =========================================================
    def process(self, road_mask: np.ndarray) -> dict:
        """
        Args:
            road_mask: uint8 binary mask (0/1 or 0/255)

        Returns:
            dict with:
                - center_points: List[(x,y)]
                - confidence: float
        """
        center_pts = self._extract_centerline_fast(road_mask)

        confidence = min(1.0, len(center_pts) / 30.0)

        return {
            "center_points": center_pts,
            "confidence": confidence
        }

    # =========================================================
    # Core logic
    # =========================================================
    def _extract_centerline_fast(self, mask: np.ndarray) -> List[Point]:
        h, w = mask.shape

        # Bottom 60% of image
        y_start = int(h * 0.4)
        ys = np.arange(y_start, h, self.row_step)
        N = len(ys)

        if N == 0:
            return []

        # Boolean mask for speed
        mask_sub = (mask[ys] > 0)  # shape: (N, W)

        # Rows containing road pixels
        has_road = mask_sub.any(axis=1)
        if not has_road.any():
            return []

        # Initialize per-row geometry
        lefts = np.zeros(N, dtype=np.float32)
        rights = np.zeros(N, dtype=np.float32)

        valid_rows = np.where(has_road)[0]
        rows = mask_sub[valid_rows]

        # Row-wise span detection
        lefts[valid_rows] = np.argmax(rows, axis=1)
        rights[valid_rows] = w - 1 - np.argmax(rows[:, ::-1], axis=1)

        widths = rights - lefts
        valid_geom = has_road & (widths >= self.min_row_width)

        # -----------------------------------------------------
        # Geometry EMA (ONLY on valid rows)
        # -----------------------------------------------------
        if self.left_ema is None or self.right_ema is None:
            self.left_ema = lefts.copy()
            self.right_ema = rights.copy()
        else:
            a = self.ema_alpha
            v = valid_geom
            self.left_ema[v] = (1.0 - a) * self.left_ema[v] + a * lefts[v]
            self.right_ema[v] = (1.0 - a) * self.right_ema[v] + a * rights[v]

        # Compute centers from smoothed geometry
        centers = ((self.left_ema + self.right_ema) * 0.5).astype(np.int32)

        # Build output centerline
        center_pts: List[Point] = [
            (int(centers[i]), int(ys[i]))
            for i in range(N)
            if valid_geom[i]
        ]

        return center_pts
