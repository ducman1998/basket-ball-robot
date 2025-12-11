import math
from time import time
from typing import Optional, Tuple, Union, cast

import numpy as np
from basket_robot_nodes.state_handlers.actions import BaseAction
from basket_robot_nodes.state_handlers.parameters import Parameters
from basket_robot_nodes.state_handlers.ret_code import RetCode
from basket_robot_nodes.utils.number_utils import (
    get_angle_diff,
    warm_up_angular,
    warm_up_xy,
)
from basket_robot_nodes.utils.peripheral_manager import PeripheralManager


class BaseHandler:
    """Handler for robot odometry-based movements."""

    def __init__(self, peripheral_manager: PeripheralManager, maneuver_rate: int = 60) -> None:
        self.peripheral_manager = peripheral_manager
        self.maneuver_rate = maneuver_rate  # control loop rate for manipulation tasks
        # general state variables
        self.start_time: Optional[float] = None  # time when the handler is initialized
        self.start_yaw: Optional[float] = None  # yaw at the start of the handler
        self.previous_yaw: Optional[float] = None  # yaw in the previous control loop
        self.start_pose: Optional[np.ndarray] = None  # size (4x4)
        self.start_position: Optional[Tuple[float, float]] = None  # size (2,)
        self.cummulative_yaw_change: float = 0.0  # total yaw change since start
        # target cummulative yaw change for turning actions
        self.target_cummulative_yaw_change: Optional[float] = None
        # a list of transformation matrice of target pose, size (4x4)
        # note: sampling multiple target poses to force robot rollow a straight line
        self.target_pose: Optional[np.ndarray] = None
        self.target_distance_mm: Optional[float] = None  # in mm, for MOVE_XY action
        # internal variables for discrete turn handling
        self.turn_discrete_new_cycle_start_time: Optional[float] = None
        self.turn_discrete_stop_timestamp: Optional[float] = None
        self.num_discrete_turns: Optional[int] = None
        self.finished_discrete_turns: int = 0

        # pid control variables
        self.prev_wz_error: float = 0.0
        self.cumm_wz_error: float = 0.0

        self.timeout: float = 10.0  # maximum allowed time for the handler

    def initialize(
        self,
        action: BaseAction,
        angle_deg: Optional[Union[float, int]] = None,  # in degrees
        offset_y_mm: Optional[float] = None,  # in mm
        timeout: float = 10.0,  # seconds
    ) -> None:
        """Initialize the handler state."""
        if action == BaseAction.TURN_CONTINUOUS or action == BaseAction.TURN_DISCRETE:
            assert angle_deg is not None, "Angle must be provided for turning actions."
        if action == BaseAction.MOVE_FORWARD:
            assert (
                offset_y_mm is not None
            ), "Offset y must be provided for MOVE_FORWARD action (in mm)."
            assert angle_deg is None, "Angle should not be provided for MOVE_FORWARD action."

        self.reset()
        self.start_time = time()
        self.start_yaw = self.peripheral_manager.get_odom_yaw()
        self.previous_yaw = self.start_yaw
        self.start_pose = self.peripheral_manager.get_robot_to_odom_transform(include_z=True)
        # set target variables
        self.target_cummulative_yaw_change = angle_deg
        # preserve the code structure for future use of MOVE_XY action
        if action == BaseAction.MOVE_FORWARD:
            t_offset = np.eye(4)
            t_offset[0:2, 3] = [0.0, offset_y_mm]
            self.target_pose = self.start_pose @ t_offset
            assert offset_y_mm is not None, "Offsets cannot be None."
            self.target_distance_mm = offset_y_mm

        self.start_position = self.peripheral_manager.get_robot_odom_position()

        # set internal variables for discrete turn handling
        if action == BaseAction.TURN_DISCRETE and angle_deg is not None:
            self.num_discrete_turns = math.ceil(
                angle_deg / Parameters.BASE_DISCRETE_TURN_SUB_ANGLE_DEG
            )
            self.turn_discrete_new_cycle_start_time = time()

        self.timeout = timeout

    def reset(self) -> None:
        self.start_time = None
        self.start_yaw = None
        self.previous_yaw = None
        self.start_pose = None
        self.start_position = None
        self.cummulative_yaw_change = 0.0
        # reset target variables
        self.target_cummulative_yaw_change = None
        self.target_pose = None
        self.target_distance_mm = None
        # reset internal variables for discrete turn handling
        self.num_discrete_turns = None
        self.finished_discrete_turns = 0
        self.turn_discrete_stop_timestamp = None
        # reset pid's variables
        self.prev_wz_error = 0.0
        self.cumm_wz_error = 0.0

        self.timeout = 10.0

    def move_robot_forward(
        self,
        max_speed: float = Parameters.BASE_MOVE_Y_MAX_SPEED,
        distance_threshold_mm: float = Parameters.BASE_MOVE_Y_DIS_THRESHOLD_MM,
        warm_start: bool = Parameters.BASE_MOVE_Y_WARMUP_ENABLED,
        ramp_duration: float = Parameters.BASE_WARMUP_RAMP_DURATION,
    ) -> RetCode:
        """Perform Y-directioin movement action."""
        assert (
            self.start_time is not None
            and self.start_pose is not None
            and self.target_pose is not None
            and self.start_position is not None
        ), "Handler not initialized."
        assert self.target_distance_mm is not None, "Target distance not set."

        t_ro = self.peripheral_manager.get_odom_to_robot_transform(include_z=True)
        # handle multiple target poses for better straight line following

        t_error = t_ro @ self.target_pose  # target w.r.t. robot frame
        if np.linalg.norm(t_error[:2, 3]) <= abs(distance_threshold_mm):
            return RetCode.SUCCESS

        distance_moved_mm = np.linalg.norm(
            np.array(self.peripheral_manager.get_robot_odom_position())
            - np.array(self.start_position)
        )
        if distance_moved_mm >= self.target_distance_mm:
            return RetCode.SUCCESS

        (vx, vy, wz) = self.compute_control_signals(t_error)
        if warm_start:
            vx, vy = self.warm_up_linear_speed(vx, vy, ramp_duration)
            wz = self.warm_up_turn_speed(wz, ramp_duration)

        self.peripheral_manager.move_robot_normalized(vx, vy, wz, max_xy_speed=max_speed)

        if time() - self.start_time >= self.timeout:
            return RetCode.TIMEOUT

        return RetCode.DOING

    def turn_robot_cont(
        self,
        wz: float = Parameters.BASE_CONTINUOUS_TURN_SPEED,
        warm_start: bool = Parameters.BASE_TURN_WARMUP_ENABLED,
        ramp_duration: float = Parameters.BASE_WARMUP_RAMP_DURATION,
    ) -> RetCode:
        """Perform continuous turning action."""
        assert self.start_time is not None, "Handler not initialized."
        assert (
            self.start_yaw is not None and self.previous_yaw is not None
        ), "Handler not initialized."
        assert self.target_cummulative_yaw_change is not None, "Target angle not set."
        assert self.num_discrete_turns is None, "This is not a continuous turn action."

        if time() - self.start_time >= self.timeout:
            return RetCode.TIMEOUT

        if warm_start:
            wz = self.warm_up_turn_speed(wz, ramp_duration)

        self.peripheral_manager.move_robot(
            0.0, 0.0, wz * np.sign(self.target_cummulative_yaw_change)
        )
        current_yaw = self.peripheral_manager.get_odom_yaw()
        diff = get_angle_diff(current_yaw, self.previous_yaw)
        self.previous_yaw = current_yaw
        self.cummulative_yaw_change += diff

        if abs(self.cummulative_yaw_change) >= abs(self.target_cummulative_yaw_change):
            return RetCode.SUCCESS

        return RetCode.DOING

    def turn_robot_disc(
        self,
        wz: float = Parameters.BASE_DISCRETE_TURN_SPEED,
        warm_start: bool = Parameters.BASE_TURN_WARMUP_ENABLED,
        ramp_duration: float = Parameters.BASE_DISCRETE_WARMUP_RAMP_DURATION,
    ) -> RetCode:
        """Perform discrete turning action."""
        assert self.start_time is not None, "Handler not initialized."
        assert (
            self.start_yaw is not None and self.previous_yaw is not None
        ), "Handler not initialized."
        assert self.target_cummulative_yaw_change is not None, "Target angle not set."
        assert (
            self.num_discrete_turns is not None
        ), "Number of discrete turns not set. Please initialize with TURN_DISCRETE action."
        assert self.turn_discrete_new_cycle_start_time is not None, "Discrete turn not initialized."

        if time() - self.start_time >= self.timeout:
            return RetCode.TIMEOUT

        if warm_start:
            wz = self.warm_up_discrete_turn_speed(wz, ramp_duration)

        current_yaw = self.peripheral_manager.get_odom_yaw()
        diff = get_angle_diff(current_yaw, self.previous_yaw)
        self.previous_yaw = current_yaw
        self.cummulative_yaw_change += diff
        # determine if we need to start a new discrete turn, or stop for an interval?
        next_threshold = (
            self.finished_discrete_turns + 1
        ) * Parameters.BASE_DISCRETE_TURN_SUB_ANGLE_DEG
        if abs(self.cummulative_yaw_change) >= next_threshold:
            if self.turn_discrete_stop_timestamp is None:
                # finished a discrete turn, start stopping
                self.turn_discrete_stop_timestamp = time()
            else:
                # waiting during stop period
                if (
                    time() - self.turn_discrete_stop_timestamp
                    >= Parameters.BASE_DISCRETE_TURN_STOP_DURATION
                ):
                    self.finished_discrete_turns += 1
                    self.turn_discrete_stop_timestamp = None
                    # reset timer for new warm-up cycle
                    self.turn_discrete_new_cycle_start_time = time()

            self.peripheral_manager.move_robot(0.0, 0.0, 0.0)
        else:
            # continue turning
            self.peripheral_manager.move_robot(
                0.0, 0.0, wz * np.sign(self.target_cummulative_yaw_change)
            )

        if abs(self.cummulative_yaw_change) >= abs(self.target_cummulative_yaw_change):
            return RetCode.SUCCESS

        return RetCode.DOING

    def warm_up_linear_speed(
        self, vx: float, vy: float, ramp_duration: float = 0.5
    ) -> Tuple[float, float]:
        """Warm up linear speed for xy movement action."""
        assert self.start_time is not None, "Handler not initialized."
        return cast(
            Tuple[float, float],
            warm_up_xy(vx, vy, self.start_time, ramp_duration),
        )

    def warm_up_turn_speed(self, wz: float, ramp_duration: float = 0.5) -> float:
        """Warm up turn speed for continuous turning action."""
        assert self.start_time is not None, "Handler not initialized."
        return cast(
            float,
            warm_up_angular(wz, self.start_time, ramp_duration),
        )

    def warm_up_discrete_turn_speed(self, wz: float, ramp_duration: float = 0.25) -> float:
        """Warm up turn speed for discrete turning action."""
        assert self.turn_discrete_new_cycle_start_time is not None, "Discrete turn not initialized."
        return cast(
            float,
            warm_up_angular(wz, self.turn_discrete_new_cycle_start_time, ramp_duration),
        )

    def compute_control_signals(self, x_la_target: np.ndarray) -> Tuple[float, float, float]:
        """Compute control signals (vx, vy, wz) using PID control.
        Inputs:
            x_la_target: target pose w.r.t. robot's look-ahead frame (4x4 numpy array)
        Returns:
            (vx, vy, wz): tuple of control signals
        """
        assert x_la_target.shape == (4, 4), "look-ahead_T_target must be a 4x4 matrix."

        # z always 0 for planar movement, no need to convert
        r11 = x_la_target[0, 0]
        r21 = x_la_target[1, 0]
        wz_error = math.atan2(r21, r11)  # yaw error

        kp_wz, ki_wz, kd_wz = Parameters.BASE_PID_ANGULAR

        # TODO: experiment with different vx, vy control strategies
        vx = Parameters.BASE_MOVE_Y_MAX_SPEED * math.sin(-wz_error) * 1.25
        vy = Parameters.BASE_MOVE_Y_MAX_SPEED * math.cos(+wz_error)
        wz = (
            kp_wz * wz_error
            + kd_wz * (wz_error - self.prev_wz_error) * self.maneuver_rate
            + ki_wz * self.cumm_wz_error
        )

        self.prev_wz_error = wz_error
        self.cumm_wz_error += wz_error / self.maneuver_rate
        return vx, vy, wz
