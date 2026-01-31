import math
from typing import List, Tuple

class CheckpointManager:
    def __init__(
        self,
        route: List[Tuple[float, float]],
        reach_threshold_m: float = 5.0,
        max_bias_distance_m: float = 10.0
    ):
        """
        route: list of (lat, lon) GPS checkpoints
        reach_threshold_m: distance to advance checkpoint
        max_bias_distance_m: distance at which bias saturates to ±1
        """
        assert len(route) >= 2, "Route must have at least 2 points"

        self.route = route
        self.idx = 1  # current goal
        self.reach_threshold = reach_threshold_m
        self.max_bias_dist = max_bias_distance_m

    def update(self, lat: float, lon: float):
        """
        Returns:
            gps_bias ∈ [-1, 1]
            distance_to_goal (meters)
        """
        prev = self.route[self.idx - 1]
        goal = self.route[self.idx]

        # Distance to goal
        dist = self._haversine(lat, lon, goal[0], goal[1])

        # Advance checkpoint if close enough
        if dist < self.reach_threshold and self.idx < len(self.route) - 1:
            self.idx += 1
            prev = self.route[self.idx - 1]
            goal = self.route[self.idx]

        # Signed cross-track error
        cte = self._cross_track_error(prev, goal, (lat, lon))

        # Normalize into [-1, 1]
        bias = max(-1.0, min(1.0, cte / self.max_bias_dist))

        return bias, dist

    # ------------------ Geometry ------------------

    def _cross_track_error(self, p1, p2, p):
        """
        Signed perpendicular distance (meters)
        Left of path = negative
        Right of path = positive
        """
        x1, y1 = self._to_xy(*p1)
        x2, y2 = self._to_xy(*p2)
        x, y = self._to_xy(*p)

        dx = x2 - x1
        dy = y2 - y1

        # Cross product sign gives left/right
        return ((x - x1) * dy - (y - y1) * dx) / math.hypot(dx, dy)

    def _to_xy(self, lat, lon):
        """
        Equirectangular projection (local, fast, good for small areas)
        """
        R = 6371000
        lat0 = math.radians(self.route[0][0])
        x = R * math.radians(lon) * math.cos(lat0)
        y = R * math.radians(lat)
        return x, y

    def _haversine(self, lat1, lon1, lat2, lon2):
        R = 6371000
        φ1, φ2 = math.radians(lat1), math.radians(lat2)
        dφ = math.radians(lat2 - lat1)
        dλ = math.radians(lon2 - lon1)

        a = math.sin(dφ / 2) ** 2 + math.cos(φ1) * math.cos(φ2) * math.sin(dλ / 2) ** 2
        return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))
