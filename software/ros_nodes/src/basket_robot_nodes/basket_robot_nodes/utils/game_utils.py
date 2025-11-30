import numpy as np
from typing import Tuple


def normalize_velocity(vx: float, vy: float, max_speed: float) -> Tuple[float, float]:
    """
    Normalize velocity vector to a maximum speed while preserving direction.

    Inputs:
        vx: velocity in x direction (m/s)
        vy: velocity in y direction (m/s)
        max_speed: maximum allowed speed (m/s)
    Returns:
        Normalized (vx, vy) tuple
    """
    speed = np.linalg.norm([vx, vy])
    if speed > max_speed:
        scale = float(max_speed / speed)
        vx *= scale
        vy *= scale
    return vx, vy
