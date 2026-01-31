import numpy as np
from typing import List, Tuple, Optional

Point = Tuple[float, float]
Polyline = List[Point]


class FastLaneExtractor:
    def __init__(
        self,
        lane_width_px: float,
        row_step: int = 4,
        min_points: int = 30,
    ):
        self.lane_width_px = lane_width_px
        self.row_step = row_step
        self.min_points = min_points

        self.prev_left: Optional[Polyline] = None
        self.prev_right: Optional[Polyline] = None

    def process(self, lane_mask: np.ndarray, road_mask: np.ndarray) -> Tuple[Polyline, Polyline]:
        h, w = lane_mask.shape
        ego_x = w * 0.5

        left_pts: Polyline = []
        right_pts: Polyline = []

        # --- scan rows bottom-up ---
        for y in range(h - 1, 0, -self.row_step):
            row = lane_mask[y]

            xs = np.where(row > 0)[0]
            if len(xs) == 0:
                continue

            # find segments
            splits = np.where(np.diff(xs) > 1)[0] + 1
            segments = np.split(xs, splits)

            centers = [float((seg[0] + seg[-1]) * 0.5) for seg in segments if len(seg) > 5]

            # pick left and right relative to ego
            left_candidates = [c for c in centers if c < ego_x]
            right_candidates = [c for c in centers if c > ego_x]

            if left_candidates:
                lx = max(left_candidates)
                left_pts.append((lx, float(y)))

            if right_candidates:
                rx = min(right_candidates)
                right_pts.append((rx, float(y)))

        left_pts.reverse()
        right_pts.reverse()

        # --- fallback logic ---
        left = self._finalize_side(left_pts, right_pts, self.prev_left, road_mask, "left")
        right = self._finalize_side(right_pts, left_pts, self.prev_right, road_mask, "right")

        self.prev_left = left
        self.prev_right = right

        return left, right

    def _finalize_side(
        self,
        detected: Polyline,
        other: Polyline,
        prev: Optional[Polyline],
        road_mask: np.ndarray,
        side: str,
    ) -> Polyline:

        h, w = road_mask.shape

        if len(detected) >= self.min_points:
            return detected

        if prev is not None and len(prev) > 0:
            return prev

        if len(other) > 0:
            sign = -1.0 if side == "left" else 1.0
            return [(x + sign * self.lane_width_px, y) for x, y in other]

        # last resort: use road
        poly: Polyline = []
        for y in range(h):
            xs = np.where(road_mask[y] > 0)[0]
            if len(xs) == 0:
                continue

            if side == "left":
                x = float(xs[0] + 5)
            else:
                x = float(xs[-1] - 5)

            poly.append((x, float(y)))

        return poly
