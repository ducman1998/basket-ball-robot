import math
import numpy as np
import modern_robotics as mr
from time import time
from typing import Optional, Tuple, Union, cast, List

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
        self.start_time: Optional[float] = None  # time when the handler is initialized
        self.start_yaw: Optional[float] = None  # yaw at the start of the handler
        self.previous_yaw: Optional[float] = None  # yaw in the previous control loop
        self.start_pose: Optional[np.ndarray] = None  # size (4x4)
        self.cummulative_yaw_change: float = 0.0  # total yaw change since start
        # target cummulative yaw change for turning actions
        self.target_cummulative_yaw_change: Optional[float] = None
        # a list of transformation matrice of target pose, size (4x4)
        # note: sampling multiple target poses to force robot rollow a straight line
        self.target_poses: Optional[List[np.ndarray]] = None
        self.num_completed_poses: int = 0

        # internal variables for discrete turn handling
        self.turn_discrete_new_cycle_start_time: Optional[float] = None
        self.turn_discrete_stop_timestamp: Optional[float] = None
        self.num_discrete_turns: Optional[int] = None
        self.finished_discrete_turns: int = 0

        # pid control variables
        self.prev_vx_error: float = 0.0
        self.prev_vy_error: float = 0.0
        self.prev_wz_error: float = 0.0
        self.cumm_vx_error: float = 0.0
        self.cumm_vy_error: float = 0.0
        self.cumm_wz_error: float = 0.0

        self.timeout: float = 10.0  # maximum allowed time for the handler

    def initialize(
        self,
        action: BaseAction,
        angle_deg: Optional[Union[float, int]] = None,  # in degrees
        offset_x_mm: Optional[float] = None,  # in mm
        offset_y_mm: Optional[float] = None,  # in mm
        timeout: float = 10.0,  # seconds
    ) -> None:
        """Initialize the handler state."""
        if action == BaseAction.TURN_CONTINUOUS or action == BaseAction.TURN_DISCRETE:
            assert angle_deg is not None, "Angle must be provided for turning actions."
        if action == BaseAction.MOVE_XY:
            assert (
                offset_x_mm is not None and offset_y_mm is not None
            ), "Offset x and y must be provided for MOVE_XY action (in mm)."
            assert angle_deg is None, "Angle should not be provided for MOVE_XY action."

        self.reset()
        self.start_time = time()
        self.start_yaw = self.peripheral_manager.get_odom_yaw()
        self.previous_yaw = self.start_yaw
        self.start_pose = self.peripheral_manager.get_robot_to_odom_transform(include_z=True)
        # set target variables
        self.target_cummulative_yaw_change = angle_deg
        if action == BaseAction.MOVE_XY:
            # sampling multiple target poses along the path to ensure straight line movement
            num_samples = max(
                1,
                math.ceil(
                    math.hypot(offset_x_mm, offset_y_mm)
                    / Parameters.BASA_MOVE_XX_PACE_TRAJECTORY_SAMPLING_MM
                ),
            )
            self.target_poses = []
            for i in range(1, num_samples + 1):
                t_offset = np.eye(4)
                t_offset[0:2, 3] = [
                    offset_x_mm * i / num_samples,
                    offset_y_mm * i / num_samples,
                ]
                self.target_poses.append(self.start_pose @ t_offset)

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
        self.cummulative_yaw_change = 0.0
        # reset target variables
        self.target_cummulative_yaw_change = None
        self.target_poses = None
        self.num_completed_poses = 0
        # reset internal variables for discrete turn handling
        self.num_discrete_turns = None
        self.finished_discrete_turns = 0
        self.turn_discrete_stop_timestamp = None
        # reset pid's variables
        self.prev_vx_error: float = 0.0
        self.prev_vy_error: float = 0.0
        self.prev_wz_error: float = 0.0
        self.cumm_vx_error: float = 0.0
        self.cumm_vy_error: float = 0.0
        self.cumm_wz_error: float = 0.0

        self.timeout = 10.0

    def move_robot_xy(
        self,
        max_speed: float = Parameters.BASE_MOVE_XY_MAX_SPEED,
        distance_threshold_mm: float = Parameters.BASE_MOVE_XY_DIS_THRESHOLD_MM,
        warm_start: bool = Parameters.BASE_MOVE_XY_WARMUP_ENABLED,
        ramp_duration: float = Parameters.BASE_WARMUP_RAMP_DURATION,
    ) -> RetCode:
        assert (
            self.start_time is not None
            and self.start_pose is not None
            and self.target_poses is not None
        ), "Handler not initialized."
        t_ro = self.peripheral_manager.get_odom_to_robot_transform(include_z=True)
        # handle multiple target poses for better straight line following

        target_pose = self.target_poses[self.num_completed_poses]
        t_error = t_ro @ target_pose  # target w.r.t. robot frame
        if np.linalg.norm(t_error[:2, 3]) <= abs(distance_threshold_mm):
            self.num_completed_poses += 1
            if self.num_completed_poses >= len(self.target_poses):
                return RetCode.SUCCESS
            else:
                target_pose = self.target_poses[self.num_completed_poses]
                t_error = t_ro @ target_pose  # target w.r.t. robot frame

        (vx, vy, wz) = self.compute_control_signals(
            t_error, Parameters.BASE_PID_LINEAR, Parameters.BASE_PID_ANGULAR
        )
        if warm_start:
            vx, vy = self.warm_up_linear_speed(vx, vy, ramp_duration)
            wz = self.warm_up_turn_speed(wz, ramp_duration)
        print(f"Moving with vx: {vx:.2f} m/s, vy: {vy:.2f} m/s")
        self.peripheral_manager.move_robot_normalized(vx, vy, wz, max_speed, max_speed)

        if time() - self.start_time >= self.timeout:
            return RetCode.TIMEOUT

        return RetCode.DOING

    def turn_robot_cont(
        self,
        wz: float = Parameters.BASE_CONTINUOUS_TURN_SPEED,
        warm_start: bool = Parameters.BASE_TURN_WARMUP_ENABLED,
        ramp_duration: float = Parameters.BASE_WARMUP_RAMP_DURATION,
    ) -> RetCode:
        assert self.start_time is not None, "Handler not initialized."
        assert (
            self.start_yaw is not None and self.previous_yaw is not None
        ), "Handler not initialized."
        assert self.target_cummulative_yaw_change is not None, "Target angle not set."
        assert self.num_discrete_turns is None, "This is not a continuous turn action."

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

        if time() - self.start_time >= self.timeout:
            return RetCode.TIMEOUT

        return RetCode.DOING

    def turn_robot_disc(
        self,
        wz: float = Parameters.BASE_DISCRETE_TURN_SPEED,
        warm_start: bool = Parameters.BASE_TURN_WARMUP_ENABLED,
        ramp_duration: float = Parameters.BASE_DISCRETE_WARMUP_RAMP_DURATION,
    ) -> RetCode:
        assert self.start_time is not None, "Handler not initialized."
        assert (
            self.start_yaw is not None and self.previous_yaw is not None
        ), "Handler not initialized."
        assert self.target_cummulative_yaw_change is not None, "Target angle not set."
        assert (
            self.num_discrete_turns is not None
        ), "Number of discrete turns not set. Please initialize with TURN_DISCRETE action."
        assert self.turn_discrete_new_cycle_start_time is not None, "Discrete turn not initialized."

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

        if time() - self.start_time >= self.timeout:
            return RetCode.TIMEOUT

        return RetCode.DOING

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

    def warm_up_discrete_turn_speed(self, wz: float, ramp_duration: float = 0.25) -> float:
        assert self.turn_discrete_new_cycle_start_time is not None, "Discrete turn not initialized."
        return cast(
            float,
            warm_up_angular(wz, self.turn_discrete_new_cycle_start_time, ramp_duration),
        )

    def compute_control_signals(
        self,
        x_la_target: np.ndarray,
        linear_pid: Tuple[float, float, float],
        angular_pid: Tuple[float, float, float],
    ) -> Tuple[float, float, float]:
        """Compute control signals (vx, vy, wz) using PID control.
        Inputs:
            x_la_target: target pose w.r.t. robot's look-ahead frame (4x4 numpy array)
        Returns:
            (vx, vy, wz): tuple of control signals
        """
        assert x_la_target.shape == (4, 4), "look-ahead_T_target must be a 4x4 matrix."

        # z always 0 for planar movement, no need to convert
        x_la_target[0, 3] /= 1000.0  # convert mm to m
        x_la_target[1, 3] /= 1000.0  # convert mm to m
        xe_log = mr.MatrixLog6(x_la_target)
        xe_vec = mr.se3ToVec(xe_log)
        vx_error = xe_vec[3]  # velocity error in x
        vy_error = xe_vec[4]  # velocity error in y
        wz_error = xe_vec[2]  # angular velocity error in z

        kp_xy, ki_xy, kd_xy = linear_pid
        kp_rz, ki_rz, kd_rz = angular_pid

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
            kp_rz * wz_error
            + kd_rz * (wz_error - self.prev_wz_error) * self.maneuver_rate
            + ki_rz * self.cumm_wz_error
        )

        self.prev_vx_error = vx_error
        self.prev_vy_error = vy_error
        self.prev_wz_error = wz_error
        self.cumm_vx_error += vx_error / self.maneuver_rate
        self.cumm_vy_error += vy_error / self.maneuver_rate
        self.cumm_wz_error += wz_error / self.maneuver_rate
        return vx, vy, wz
