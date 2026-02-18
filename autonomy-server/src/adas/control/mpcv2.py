import numpy as np
import math
from typing import List, Tuple

Point = Tuple[int, int]


class CenterlineMPC:
    def __init__(self):
        # --- Config ---
        self.num_candidates = 12
        self.max_steering = math.radians(28)

        self.lookahead_steps = 40
        self.step_length = 15.0
        self.wheelbase = 60.0  # pixels

        # --- Weights ---
        self.W_CENTER = 1.0
        self.W_GPS = 0.3
        self.W_SMOOTH = 0.2

        # --- Steering smoothing ---
        self.steering_ema_alpha = 0.25
        self.last_steering = 0.0
        self.last_output = 0.0

        # Steering candidates
        self.candidates = np.linspace(
            -self.max_steering,
            self.max_steering,
            self.num_candidates
        )

        # ✅ Precompute trajectories (FLOAT, not int)
        self.traj_cache = [
            self._precompute_trajectory(angle)
            for angle in self.candidates
        ]

    # =========================================================
    # Public API
    # =========================================================
    def compute(
        self,
        road_mask: np.ndarray,
        center_points: List[Point],
        gps_bias: float
    ) -> Tuple[float, List[Point]]:

        if len(center_points) < 6:
            return self.last_output, []

        h, w = road_mask.shape
        ox = w // 2
        oy = h - 1

        centerline = np.asarray(center_points, dtype=np.int32)

        best_cost = float("inf")
        best_angle = 0.0
        best_traj = None

        # Explicit GPS steering target (guarded semantics)
        gps_target = gps_bias * self.max_steering

        for angle, traj in zip(self.candidates, self.traj_cache):

            # --- Fast road check ---
            if not self._trajectory_inside_road_fast(traj, road_mask, ox, oy):
                continue

            c_center = self._cost_center_fast(traj, centerline, ox, oy)
            c_gps = abs(angle - gps_target)
            c_smooth = abs(angle - self.last_steering)

            total = (
                self.W_CENTER * c_center +
                self.W_GPS * c_gps +
                self.W_SMOOTH * c_smooth
            )

            if total < best_cost:
                best_cost = total
                best_angle = angle
                best_traj = traj

        if best_traj is None:
            out = self.last_output * 0.9
            self.last_output = out
            return out, []

        self.last_steering = best_angle
        out = (
            (1.0 - self.steering_ema_alpha) * self.last_output +
            self.steering_ema_alpha * best_angle
        )
        self.last_output = out

        traj_pts = [
            (int(ox + dx), int(oy + dy))
            for dx, dy in best_traj
        ]

        return out, traj_pts

    # =========================================================
    # Precompute
    # =========================================================
    def _precompute_trajectory(self, steering: float) -> np.ndarray:
        """
        Precompute trajectory using the SAME incremental bicycle model
        as the original MPC. This is critical for correctness.
        Returns (N,2) float32 array of (dx, dy).
        """
        x = 0.0
        y = 0.0
        theta = -math.pi / 2

        pts = []

        k = math.tan(steering) / self.wheelbase

        for _ in range(self.lookahead_steps):
            if abs(k) < 1e-6:
                dx = self.step_length * math.cos(theta)
                dy = self.step_length * math.sin(theta)
                x += dx
                y += dy
            else:
                dtheta = self.step_length * k
                r = 1.0 / k

                cx = x - r * math.sin(theta)
                cy = y + r * math.cos(theta)

                theta += dtheta
                x = cx + r * math.sin(theta)
                y = cy - r * math.cos(theta)

            pts.append((x, y))

        return np.asarray(pts, dtype=np.float32)


    # =========================================================
    # Fast checks & costs
    # =========================================================
    def _trajectory_inside_road_fast(
        self,
        traj: np.ndarray,
        mask: np.ndarray,
        ox: int,
        oy: int
    ) -> bool:

        xs = (ox + traj[:, 0]).astype(np.int32)
        ys = (oy + traj[:, 1]).astype(np.int32)

        valid = (
            (xs >= 0) & (xs < mask.shape[1]) &
            (ys >= 0) & (ys < mask.shape[0])
        )

        if not valid.all():
            return False

        return bool(np.all(mask[ys, xs] > 0))

    def _cost_center_fast(
        self,
        traj: np.ndarray,
        centerline: np.ndarray,
        ox: int,
        oy: int
    ) -> float:
        """
        Fast lateral deviation cost
        """
        traj_x = ox + traj[:, 0]
        traj_y = oy + traj[:, 1]

        cl_x = centerline[:, 0]
        cl_y = centerline[:, 1]

        cost = 0.0
        for x, y in zip(traj_x, traj_y):
            idx = np.argmin(np.abs(cl_y - y))
            cost += abs(cl_x[idx] - x)

        return cost / len(traj)
