from time import time
from typing import Optional, Tuple, cast

import modern_robotics as mr
import numpy as np
from basket_robot_nodes.state_handlers.actions import ManipulationAction
from basket_robot_nodes.state_handlers.parameters import Parameters
from basket_robot_nodes.state_handlers.ret_code import RetCode
from basket_robot_nodes.utils.number_utils import warm_up_angular, warm_up_xy
from basket_robot_nodes.utils.peripheral_manager import PeripheralManager


class ManpulationHandler:
    """Handler for robot manipulation tasks (e.g., grabbing and throwing balls)."""

    def __init__(self, peripheral_manager: PeripheralManager, maneuver_rate: int = 60) -> None:
        self.peripheral_manager = peripheral_manager
        self.maneuver_rate = maneuver_rate  # control loop rate for manipulation tasks
        self.start_time: Optional[float] = None  # time when the handler is initialized
        self.current_action: Optional[ManipulationAction] = None
        # internal PID control variables
        self.prev_vx_error: float = 0.0
        self.prev_vy_error: float = 0.0
        self.prev_wz_error: float = 0.0
        self.cumm_vx_error: float = 0.0
        self.cumm_vy_error: float = 0.0
        self.cumm_wz_error: float = 0.0

        self.timeout: float = 10.0  # maximum allowed time for the handler

    def initialize(
        self,
        action: ManipulationAction,
    ) -> None:
        """Initialize the handler state."""
        self.reset()
        self.current_action = action
        self.start_time = time()

    def reset(self) -> None:
        """Reset the handler state."""
        self.start_time = None
        self.current_action = None
        self.prev_vx_error = 0.0
        self.prev_vy_error = 0.0
        self.prev_wz_error = 0.0
        self.cumm_vx_error = 0.0
        self.cumm_vy_error = 0.0
        self.cumm_wz_error = 0.0

    def approach_target(self, target_pos_robot_mm: Tuple[float, float]) -> RetCode:
        """Approach target for manipulation."""
        pass
        return RetCode.DOING

    def approach_ball(
        self,
        warm_start: bool = Parameters.MANI_APPROACH_WARMUP_ENABLED,
        ramp_duration: float = Parameters.MANI_WARMUP_RAMP_DURATION,
    ) -> RetCode:
        """Approach ball for manipulation."""
        pass
        return RetCode.DOING

    def approach_basket(
        self,
        warm_start: bool = Parameters.MANI_APPROACH_WARMUP_ENABLED,
        ramp_duration: float = Parameters.MANI_WARMUP_RAMP_DURATION,
    ) -> RetCode:
        """Approach basket for manipulation."""
        pass
        return RetCode.DOING

    def move_forward_to_grab(
        self,
        warm_start: bool = Parameters.MANI_GRAB_WARMUP_ENABLED,
        ramp_duration: float = Parameters.MANI_WARMUP_RAMP_DURATION,
    ) -> RetCode:
        """Move forward to grab the ball."""
        pass
        return RetCode.DOING

    def align_with_basket(
        self,
        warm_start: bool = Parameters.MANI_ALIGN_WARMUP_ENABLED,
        ramp_duration: float = Parameters.MANI_WARMUP_RAMP_DURATION,
    ) -> RetCode:
        """Align with the basket for scoring."""
        pass
        return RetCode.DOING

    def throw_ball(self) -> RetCode:
        """Throw the ball into the basket."""
        pass
        return RetCode.DOING

    def compute_control_signals(
        self, x_la_target: np.ndarray, pid_gains: dict[str, Tuple[float, float, float]]
    ) -> Tuple[float, float, float]:
        """Compute control signals (vx, vy, wz) using PID control.
        Inputs:
            x_la_target: target pose w.r.t. robot's look-ahead frame (4x4 numpy array)
        Returns:
            (vx, vy, wz): tuple of control signals
        """
        assert x_la_target.shape == (4, 4), "look-ahead_T_target must be a 4x4 matrix."

        xe_log = mr.MatrixLog6(x_la_target)
        xe_vec = mr.se3ToVec(xe_log)
        vx_error = xe_vec[3]  # velocity error in x
        vy_error = xe_vec[4]  # velocity error in y
        wz_error = xe_vec[2]  # angular velocity error in z

        kp_xy, ki_xy, kd_xy = pid_gains["linear"]
        kp_rot, ki_rot, kd_rot = pid_gains["angular"]

        vx = (
            kp_xy * vx_error
            + kd_xy * (vx_error - self.prev_vx_error) * self.maneuver_rate
            + ki_xy * self.cumm_vx_error
        )
        vy = (
            kp_xy * vy_error
            + kd_xy * (vy_error - self.prev_vy_error) * self.maneuver_rate
            + ki_xy * self.cumm_vy_error
        )
        wz = (
            kp_rot * wz_error
            + kd_rot * (wz_error - self.prev_wz_error) * self.maneuver_rate
            + ki_rot * self.cumm_wz_error
        )

        self.prev_vx_error = vx_error
        self.prev_vy_error = vy_error
        self.prev_wz_error = wz_error
        self.cumm_vx_error += vx_error / self.maneuver_rate
        self.cumm_vy_error += vy_error / self.maneuver_rate
        self.cumm_wz_error += wz_error / self.maneuver_rate
        return vx, vy, wz

    # def compute_pid(
    #     self, vx_err: float, vy_err: float, wz_err: float
    # ) -> Tuple[float, float, float]:
    #     """Compute PID control for manipulation tasks."""

    def get_pid_gains(self) -> None:
        """Retrieve PID gains for manipulation tasks."""
        pass

    def warm_up_linear_speed(
        self, vx: float, vy: float, ramp_duration: float = 0.5
    ) -> Tuple[float, float]:
        assert self.start_time is not None, "Handler not initialized."
        return cast(
            Tuple[float, float],
            warm_up_xy(vx, vy, self.start_time, ramp_duration),
        )

    def warm_up_turn_speed(self, wz: float, ramp_duration: float = 0.5) -> float:
        assert self.start_time is not None, "Handler not initialized."
        return cast(
            float,
            warm_up_angular(wz, self.start_time, ramp_duration),
        )
