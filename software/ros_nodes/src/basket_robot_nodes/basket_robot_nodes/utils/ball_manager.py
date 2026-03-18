import math
from time import time
from typing import List, Optional, Tuple, cast

import numpy as np


class BallManager:
    def __init__(
        self,
        num_stored_balls: int = 6,
        dist_threshold_mm: float = 100.0,
        alive_time_s: float = 15.0,
    ):
        self.num_stored_balls = num_stored_balls
        self.dist_threshold_mm = dist_threshold_mm
        self.alive_time = alive_time_s

        # Stored as (x, y, last_seen_time)
        self.stored_balls: List[Tuple[float, float, float]] = []

    def update_balls(self, ball_odom_positions: List[Tuple[float, float]]) -> None:
        now = time()

        # ---- 1. Remove expired balls ----
        self.stored_balls = [sb for sb in self.stored_balls if now - sb[2] <= self.alive_time]

        # ---- 2. Match new detections to stored balls ----
        for bx, by in ball_odom_positions:
            matched = False

            for i, (sx, sy, _) in enumerate(self.stored_balls):
                dist = np.linalg.norm([bx - sx, by - sy])
                if dist < self.dist_threshold_mm:
                    # Update existing ball
                    self.stored_balls[i] = (bx, by, now)
                    matched = True
                    break

            # ---- 3. Add new ball if no match ----
            if not matched:
                self.stored_balls.append((bx, by, now))

        # ---- 4. Keep only closest N balls to origin ----
        self.stored_balls.sort(key=lambda b: np.linalg.norm([b[0], b[1]]))
        self.stored_balls = self.stored_balls[: self.num_stored_balls]

    def get_rotation_angle_to_candidate(
        self, robot_pos_odom: Tuple[float, float], t_r_odom: np.ndarray
    ) -> Optional[float]:
        """Returns the heading error (in degrees) to the closest stored
        ball from the robot's position. Note that, T_r_odom is size 3x3.
        """
        if not self.stored_balls:
            return None

        rx, ry = robot_pos_odom
        candidate_odom = min(
            self.stored_balls,
            key=lambda b: np.linalg.norm([b[0] - rx, b[1] - ry]),
        )
        candidate_robot = t_r_odom @ np.array([*candidate_odom[:2], 1])  # ignore timestamp
        heading_error = np.rad2deg(-math.atan2(candidate_robot[0], candidate_robot[1]))
        return cast(Optional[float], heading_error)
