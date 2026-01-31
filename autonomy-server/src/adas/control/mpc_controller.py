import numpy as np
import math
from typing import List, Tuple, Optional

try:
    from scipy.spatial import cKDTree
    HAS_KDTREE = False
except Exception:
    HAS_KDTREE = False

Point = Tuple[int, int]

class CenterlineMPC:
    def __init__(self):
        # --- Config ---
        self.num_candidates = 10
        self.max_steering = math.radians(28)

        # Longer horizon, finer resolution
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

        best_cost = float("inf")
        best_angle = 0.0
        best_traj: List[Point] = []

        center_arr = np.asarray(center_points, dtype=np.float32)

        if HAS_KDTREE:
            tree = cKDTree(center_arr)
        else:
            tree = None

        candidates = np.linspace(-self.max_steering, self.max_steering, self.num_candidates)

        for angle in candidates:
            angle = float(angle)
            traj = self._project_trajectory(angle)

            if not self._trajectory_inside_road(traj, road_mask):
                continue

            c_center = self._cost_centerline(traj, center_arr, tree, road_mask.shape)
            c_gps = self._cost_gps(angle, gps_bias)
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

        if not best_traj:
            # graceful decay instead of lock-in
            out = self.last_output * 0.9
            self.last_output = out
            return out, []

        self.last_steering = best_angle

        out = (
            (1.0 - self.steering_ema_alpha) * self.last_output +
            self.steering_ema_alpha * best_angle
        )
        self.last_output = out

        return out, best_traj

    # =========================================================
    # Trajectory simulation (improved bicycle model)
    # =========================================================

    def _project_trajectory(self, steering: float) -> List[Point]:
        """
        Integrates true circular arc motion instead of Euler drift.
        """
        x = 0.0
        y = 0.0
        theta = -math.pi / 2  # facing upward

        pts: List[Point] = []

        k = math.tan(steering) / self.wheelbase  # curvature

        for _ in range(self.lookahead_steps):
            if abs(k) < 1e-6:
                # Straight motion
                dx = self.step_length * math.cos(theta)
                dy = self.step_length * math.sin(theta)
                x += dx
                y += dy
            else:
                # Arc motion
                dtheta = self.step_length * k
                r = 1.0 / k

                cx = x - r * math.sin(theta)
                cy = y + r * math.cos(theta)

                theta += dtheta
                x = cx + r * math.sin(theta)
                y = cy - r * math.cos(theta)

            pts.append((int(x), int(y)))

        return pts

    # =========================================================
    # Costs
    # =========================================================

    def _trajectory_inside_road(self, traj: List[Point], mask: np.ndarray) -> bool:
        h, w = mask.shape
        origin_x = w // 2
        origin_y = h - 1

        for dx, dy in traj:
            x = origin_x + dx
            y = origin_y + dy

            if not (0 <= x < w and 0 <= y < h):
                return False
            if mask[y, x] == 0:
                return False

        return True

    def _cost_centerline(
        self,
        traj: List[Point],
        centerline: np.ndarray,
        tree,
        mask_shape
    ) -> float:

        h, w = mask_shape
        origin_x = w // 2
        origin_y = h - 1

        y_cut = int(0.9 * h)
        cl = centerline[centerline[:, 1] < y_cut]
        if len(cl) == 0:
            cl = centerline

        cost = 0.0

        for dx, dy in traj:
            x = origin_x + dx
            y = origin_y + dy

            if tree is not None:
                d, _ = tree.query((x, y))
                cost += float(d)
            else:
                dists = (cl[:, 0] - x) ** 2 + (cl[:, 1] - y) ** 2
                cost += math.sqrt(float(np.min(dists)))

        return cost / len(traj)

    def _cost_gps(self, steering: float, gps_bias: float) -> float:
        target = gps_bias * self.max_steering
        return abs(steering - target)
