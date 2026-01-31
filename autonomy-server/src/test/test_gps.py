from src.adas.gps.checkpoint import CheckpointManager

# ---------------------------------------
# Route: east then north (L-shape)
# ---------------------------------------
route = [
    (37.0000, -122.0000),  # P0
    (37.0000, -121.9990),  # P1 (~88m east)
    (37.0010, -121.9990),  # P2 (~111m north)
]

mgr = CheckpointManager(
    route=route,
    reach_threshold_m=5.0,
    max_bias_distance_m=10.0
)

# ---------------------------------------
# Sequential GPS updates (simulated drive)
# ---------------------------------------
gps_sequence = [
    ("Start near P0, centered",   37.0000, -121.9999),
    ("Right of P0→P1 segment",    37.00005, -121.9997),
    ("Approaching P1",            37.0000, -121.99905),
    ("Within 5m of P1 (switch)",  37.0000, -121.99901),
    ("Just after P1, slight left",37.00005, -121.9990),
    ("On P1→P2 segment",          37.0005, -121.9990),
    ("Near final checkpoint",     37.00095, -121.9990),
]

# ---------------------------------------
# Run test
# ---------------------------------------
for step, lat, lon in gps_sequence:
    bias, dist = mgr.update(lat, lon)
    goal_idx = mgr.idx
    goal = mgr.route[goal_idx]

    print(
        f"{step:30s} | "
        f"Goal idx: {goal_idx} | "
        f"Goal: {goal} | "
        f"Bias: {bias:+.3f} | "
        f"Dist: {dist:6.2f} m"
    )
