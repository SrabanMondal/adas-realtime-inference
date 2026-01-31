import math
from typing import List
from models import AutonomyMessage, AutonomyState, Control


def fake_autonomy_output(
    t: float,
    width: int,
    height: int,
) -> AutonomyMessage:
    steering = 20 * math.sin(t * 0.7)

    left_lane = [
        [int(width * 0.35), height],
        [int(width * 0.40), int(height * 0.6)],
        [int(width * 0.45), int(height * 0.3)],
    ]

    right_lane = [
        [int(width * 0.65), height],
        [int(width * 0.60), int(height * 0.6)],
        [int(width * 0.55), int(height * 0.3)],
    ]

    traj: List[List[int]] = []
    for i in range(6):
        y = height - i * (height // 8)
        x = int(width / 2 + steering * 2 * (i / 6))
        traj.append([x, y])

    return AutonomyMessage(
        type="autonomy",
        payload=AutonomyState(
            laneLines=[left_lane, right_lane],
            trajectory=traj,
            control=Control(
                steeringAngle=steering,
                confidence=0.9,
            ),
            status="NORMAL",
        ),
    )
