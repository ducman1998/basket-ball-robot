from time import time
from typing import Tuple

import numpy as np


def clip_int16(v: int) -> int:
    return max(-32768, min(32767, int(v)))


def clip_uint16(v: int) -> int:
    return max(0, min(65535, int(v)))


def get_angle_diff(cur_angle: float, pre_angle: float) -> float:
    """
    Calculates the shortest angular difference (target - current) in degrees,
    normalized to the range [-180, 180].
    """
    raw_error = cur_angle - pre_angle
    if raw_error > 180.0:
        raw_error -= 360.0
    elif raw_error < -180.0:
        raw_error += 360.0
    return raw_error


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


def warm_up_xy(
    vx: float, vy: float, start_time: float, ramp_duration: float = 0.5
) -> Tuple[float, float]:
    """
    Gradually ramp up linear speed over the specified duration.
    Inputs:
        vx: target velocity in x direction (m/s)
        vy: target velocity in y direction (m/s)
        start_time: time when ramp-up started (seconds)
        ramp_duration: duration over which to ramp up speed (seconds)
    Returns:
        (vx, vy) tuple after applying ramp-up scaling
    """
    elapsed_time = time() - start_time
    if elapsed_time < ramp_duration:
        scale = elapsed_time / ramp_duration
        vx *= scale
        vy *= scale
    return vx, vy


def warm_up_angular(wz: float, start_time: float, ramp_duration: float = 0.5) -> float:
    """
    Gradually ramp up angular speed over the specified duration.
    Inputs:
        wz: target angular velocity (rad/s)
        start_time: time when ramp-up started (seconds)
        ramp_duration: duration over which to ramp up speed (seconds)
    Returns:
        wz after applying ramp-up scaling
    """
    elapsed_time = time() - start_time
    if elapsed_time < ramp_duration:
        scale = elapsed_time / ramp_duration
        wz *= scale
    return float(wz)


class FrameStabilityCounter:
    def __init__(self, threshold: int) -> None:
        self.threshold = threshold
        self.count = 0

    def update(self, condition: bool) -> bool:
        """Returns True once condition has been True for `threshold` consecutive frames."""
        if condition:
            self.count += 1
            if self.count >= self.threshold:
                self.count = 0
                return True
        else:
            self.count = 0
        return False
