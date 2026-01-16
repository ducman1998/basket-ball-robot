import math
from collections import deque
from time import time
from typing import List, Literal, Optional, Tuple, Union, cast

import modern_robotics as mr
import numpy as np
from basket_robot_nodes.state_handlers.actions import ManipulationAction
from basket_robot_nodes.state_handlers.parameters import Parameters
from basket_robot_nodes.state_handlers.ret_code import RetCode
from basket_robot_nodes.utils.constants import (
    CALIB_THROWER_MOTOR_PERCENTS,
    CALIB_THROWER_MOTOR_OFFSETS,
)
from basket_robot_nodes.utils.image_info import Marker
from basket_robot_nodes.utils.number_utils import (
    get_rotation_matrix,
    warm_up_angular,
    warm_up_xy,
    get_angle_diff,
)
from basket_robot_nodes.utils.peripheral_manager import PeripheralManager


class ManpulationHandler:
    """Handler for robot manipulation tasks (e.g., grabbing and throwing balls)."""

    def __init__(self, peripheral_manager: PeripheralManager, maneuver_rate: int = 60) -> None:
        self.peripheral_manager = peripheral_manager
        self.maneuver_rate = maneuver_rate  # control loop rate for manipulation tasks
        self.start_time: Optional[float] = None  # time when the handler is initialized
        self.current_action: Optional[ManipulationAction] = None

        # basket alignment variables
        self.basket_color: Optional[str] = None
        self.base_thrower_percent: Optional[float] = None
        self.is_basket_aligned_queue: deque = deque(
            maxlen=Parameters.MANI_SEARCH_BASKET_NUM_CONSECUTIVE_VALID_FRAMES
        )
        # advanced basket alignment variables
        self.is_marker_pose_extracted: bool = False
        self.previous_yaw: Optional[float] = None
        self.cummulative_yaw_change: float = 0.0
        # transformation from odometry to desired throwing position
        self.t_odom_tp: Optional[np.ndarray] = None

        # throw the ball
        self.calculated_thrower_percent: Optional[float] = None

        # internal PID control variables
        self.prev_vx_error: float = 0.0
        self.prev_vy_error: float = 0.0
        self.prev_wz_error: float = 0.0
        self.cumm_vx_error: float = 0.0
        self.cumm_vy_error: float = 0.0
        self.cumm_wz_error: float = 0.0

        self.timeout: float = 10.0  # maximum allowed time for the handler
        # experimental data points for thrower speed calibration
        # (distance in milimeters, motor percent)
        self.data_points = CALIB_THROWER_MOTOR_PERCENTS
        self.offset_points = CALIB_THROWER_MOTOR_OFFSETS

    def initialize(
        self,
        action: ManipulationAction,
        basket_color: Optional[str] = None,
        base_thrower_percent: Optional[float] = None,
        turning_basket_direction: Optional[float] = None,
        timeout: float = 10.0,
        timeout_refine_angle: Optional[float] = None,
    ) -> None:
        """Initialize the handler state.
        Note that last option only used for ALIGN_BASKET_ADVANCED,
        ensure enough time for aligning basket angle deviation before throwing.
        """
        if action in (ManipulationAction.ALIGN_BASKET, ManipulationAction.ALIGN_BASKET_ADVANCED):
            assert (
                basket_color is not None
            ), "Basket color must be provided for ALIGN_BASKET & ALIGN_BASKET_ADVANCED action."

        if action == ManipulationAction.ALIGN_BASKET_ADVANCED:
            assert (
                timeout_refine_angle is not None
            ), "Specify timeout for refining angle error in ALIGN_BASKET_ADVANCED action."
            assert (
                timeout - timeout_refine_angle > 0.0
            ), "Total timeout must be greater than timeout for aligning angle error."

        if action == ManipulationAction.TURN_AROUND_BASKET:
            assert (
                turning_basket_direction is not None
            ), "Turning direction must be provided for TURN_AROUND_BASKET action."

        assert timeout > 0.0, "Timeout must be positive."

        self.reset()
        self.start_time = time()
        self.current_action = action
        self.basket_color = basket_color
        self.base_thrower_percent = base_thrower_percent
        self.turning_basket_direction = turning_basket_direction
        self.timeout = timeout  # total timeout for the manipulation action
        self.timeout_refine_angle = timeout_refine_angle
        self.align_basket_turning_direction: Optional[Literal[-1, 1]] = None
        self.previous_yaw = self.peripheral_manager.get_odom_yaw()
        self.cummulative_yaw_change = 0.0

    def reset(self) -> None:
        """Reset the handler state."""
        self.start_time = None
        self.current_action = None
        self.basket_color = None
        self.base_thrower_percent = None
        self.is_basket_aligned_queue.clear()
        self.is_marker_pose_extracted = False
        self.t_odom_tp = None
        self.calculated_thrower_percent = None
        self.turning_basket_direction = None
        self.timeout = 10.0
        self.timeout_refine_angle = None
        self.align_basket_turning_direction = None
        self.previous_yaw = None
        self.cummulative_yaw_change = 0.0
        self.reset_pid_errors()

    def reset_pid_errors(self) -> None:
        """Reset PID error accumulators."""
        self.prev_vx_error = 0.0
        self.prev_vy_error = 0.0
        self.prev_wz_error = 0.0
        self.cumm_vx_error = 0.0
        self.cumm_vy_error = 0.0
        self.cumm_wz_error = 0.0

    def compute_control_lookahead_woffset(
        self,
        pos_robot_mm: Tuple[float, float],
        pid_gains: Tuple[List[float], List[float]],
        la_dis_mm: float = 200.0,
        angle_offset_deg: float = 0.0,
    ) -> Tuple[float, float, float]:
        """
        Approach a target position for manipulation.
        Inputs:
            pos_robot_mm: target position in robot frame (x_mm, y_mm)
            pid_gains: tuple of (linear_gains, angular_gains) for PID control
            la_dis_mm: look-ahead distance in mm
            angle_offset_deg: angle offset in degrees to adjust heading
        Returns:
            RetCode indicating the status of the approach
        """
        heading_err_rad = -math.atan2(pos_robot_mm[0], pos_robot_mm[1])
        adjusted_pos_robot_mm = (pos_robot_mm[0], pos_robot_mm[1] - la_dis_mm)
        t_error = self.position_to_pose(adjusted_pos_robot_mm)
        r_mtx = get_rotation_matrix(heading_err_rad + np.deg2rad(angle_offset_deg))
        t_error[0:2, 0:2] = r_mtx
        (vx, vy, wz) = self.compute_control_signals(t_error, pid_gains)
        return vx, vy, wz

    def compute_aligment_errors(self, pos_robot_mm: Tuple[float, float]) -> Tuple[float, float]:
        """Compute positional and angular alignment errors to a target position."""
        distance_mm = np.hypot(
            pos_robot_mm[0], pos_robot_mm[1] - Parameters.MANI_ALIGN_BALL_LOOKAHEAD_DIS_MM
        )
        angle_rad = abs(math.atan2(pos_robot_mm[0], pos_robot_mm[1]))
        return distance_mm, np.rad2deg(angle_rad)

    def align_to_ball(
        self,
        warm_start: bool = Parameters.MANI_ALIGN_WARMUP_ENABLED,
        ramp_duration: float = Parameters.MANI_WARMUP_RAMP_DURATION,
    ) -> RetCode:
        """Approach ball for manipulation."""
        assert self.start_time is not None, "Handler not initialized."

        if self.peripheral_manager.is_balls_not_detected_in_nframes(
            Parameters.MANI_MAX_CONSECUTIVE_FRAMES_NO_BALL
        ):
            return RetCode.FAILED_BALL_LOST

        if time() - self.start_time > self.timeout:
            return RetCode.TIMEOUT

        if self.peripheral_manager.is_ball_detected():
            b_pos_robot_mm = self.peripheral_manager.get_closest_ball_position()
            if b_pos_robot_mm is None:
                # exceptional case -- should not happen due to is_ball_detected() check
                return RetCode.DOING

            p_error, o_error = self.compute_aligment_errors(b_pos_robot_mm)
            if (
                p_error <= Parameters.MANI_BALL_ALIGN_DIS_THRESHOLD_MM
                and o_error <= Parameters.MANI_BALL_ALIGN_ANGLE_THRESHOLD_DEG
            ):
                return RetCode.SUCCESS

            (vx, vy, wz) = self.compute_control_lookahead_woffset(
                b_pos_robot_mm,
                (
                    Parameters.MANI_PID_LINEAR_ALIGN,
                    Parameters.MANI_PID_ANGULAR_ALIGN,
                ),
                la_dis_mm=Parameters.MANI_ALIGN_BALL_LOOKAHEAD_DIS_MM,
            )
            if p_error > Parameters.MANI_BALL_ALIGN_ENABLED_DIST_PID_MM:
                # normalize vx, vy to max speed
                cur_speed = np.hypot(vx, vy)
                if cur_speed < Parameters.MANI_MAX_ALIGN_LINEAR_SPEED:
                    scale = Parameters.MANI_MAX_ALIGN_LINEAR_SPEED / cur_speed
                    vx *= scale
                    vy *= scale

            if warm_start:
                vx, vy = self.warm_up_linear_speed(vx, vy, ramp_duration)
                wz = self.warm_up_turn_speed(wz, ramp_duration)

            self.peripheral_manager._node.get_logger().info(
                f"Speed commands: vx={vx:.2f}, vy={vy:.2f}, wz={wz:.2f}"
            )
            self.peripheral_manager.move_robot_adv(
                vx,
                vy,
                wz,
                thrower_percent=0.0,
                servo_speed=Parameters.MANI_GRAB_BALL_SERVO_SPEED,
                normalize=True,
                max_xy_speed=Parameters.MANI_MAX_ALIGN_LINEAR_SPEED,
                max_rot_speed=Parameters.MANI_MAX_ALIGN_ANGULAR_SPEED,
            )
        return RetCode.DOING

    def align_to_basket(self, disable_timeout: bool = False) -> RetCode:
        """Align to the basket for scoring, the robot will rotate to face the basket"""
        assert self.start_time is not None, "Handler not initialized."
        assert self.basket_color is not None, "Basket color not set."

        if not disable_timeout:
            if time() - self.start_time > self.timeout:
                return RetCode.TIMEOUT

        basket_pos_robot_mm: Optional[Tuple[float, float]] = None
        if (
            self.peripheral_manager.is_basket_detected()
            and self.peripheral_manager.get_basket_color() == self.basket_color
        ):
            basket_pos_robot_mm = self.peripheral_manager.get_basket_position_2d()
            if basket_pos_robot_mm is not None:
                o_error_deg = np.rad2deg(math.atan2(basket_pos_robot_mm[0], basket_pos_robot_mm[1]))
                is_aligned = abs(o_error_deg) <= Parameters.MANI_BASKET_ALIGN_ANGLE_THRESHOLD_DEG
                self.is_basket_aligned_queue.append(is_aligned)
                if (
                    self.is_basket_aligned_queue.count(True)
                    == Parameters.MANI_SEARCH_BASKET_NUM_CONSECUTIVE_VALID_FRAMES
                ):
                    return RetCode.SUCCESS
        elif self.align_basket_turning_direction is None:
            # no basket deteted, using stored position to align the robot towards the basket
            basket_pos_robot_mm_temp: Optional[Tuple[float, float]] = None
            if self.peripheral_manager.get_target_basket_color() == self.basket_color:
                basket_pos_robot_mm_temp = self.peripheral_manager.get_stored_target_basket_pos(
                    True, Parameters.MANI_STORED_BASKET_TIMEOUT
                )
                self.peripheral_manager._node.get_logger().info(
                    "Using stored target basket position for alignment."
                )
            else:
                basket_pos_robot_mm_temp = self.peripheral_manager.get_stored_opponent_basket_pos(
                    True, Parameters.MANI_STORED_BASKET_TIMEOUT
                )
                self.peripheral_manager._node.get_logger().info(
                    "Using stored target basket position for alignment."
                )
            if basket_pos_robot_mm_temp is not None:
                x, y = basket_pos_robot_mm_temp
                if -np.arctan2(x, y) >= 0:
                    self.align_basket_turning_direction = 1
                else:
                    self.align_basket_turning_direction = -1

        vy = 0.0
        if basket_pos_robot_mm is not None:
            # control to face the basket
            heading_err_rad = -math.atan2(basket_pos_robot_mm[0], basket_pos_robot_mm[1])
            self.peripheral_manager._node.get_logger().info(
                f"Error to basket: {np.rad2deg(heading_err_rad):.2f} deg"
            )
            if abs(heading_err_rad) < np.deg2rad(
                Parameters.MAIN_BASKET_ALIGN_FINE_GRAINED_THRESHOLD_DEG
            ):
                # fine-grained alignment with slight forward movement
                # ti improve accuracy when aligning to basket
                vy = Parameters.MAIN_BASKET_ALIGN_Y_SPEED

            wz = self.compute_pid(
                2,
                heading_err_rad,
                Parameters.MANI_PID_ANGULAR_ALIGN_BASKET,
            )
        else:
            # update cummulated yaw change
            assert self.previous_yaw is not None, "Previous yaw not initialized."
            current_yaw = self.peripheral_manager.get_odom_yaw()
            diff = get_angle_diff(current_yaw, self.previous_yaw)
            self.previous_yaw = current_yaw
            self.cummulative_yaw_change += diff
            if (
                abs(self.cummulative_yaw_change)
                >= Parameters.MANI_SEARCH_BASKET_HIGH_SPEED_MAX_TURNING_DEG
            ):
                idx = 0
            else:
                idx = 1
            # keep rotating to search for the basket
            if self.align_basket_turning_direction is None:
                self.align_basket_turning_direction = 1  # default direction
            wz = (
                abs(Parameters.MANI_SEARCH_BASKET_ANGULAR_SPEEDS[idx])
                * self.align_basket_turning_direction
            )

        # Set a base thrower speed to avoid sudden changes that can cause the ball to get stuck.
        self.peripheral_manager.move_robot_adv(
            0.0,
            vy,
            wz,
            thrower_percent=0.0 if self.base_thrower_percent is None else self.base_thrower_percent,
            servo_speed=0,
            normalize=True,
            max_rot_speed=Parameters.MANI_SEARCH_BASKET_MAX_ANGULAR_SPEED,
        )

        return RetCode.DOING

    def align_to_basket_advanced(self) -> RetCode:
        """Align the basket with the ball for scoring."""
        # TODO: implement basket alignment with ball, can be similar to align_to_basket(), but
        # more complex logic to consider ball position, marker pose, etc. The robot may need to
        # move to preferred position based on detected markers before aligning the basket.
        assert self.start_time is not None, "Handler not initialized."
        assert self.timeout_refine_angle is not None, "Refine angle timeout not set."
        assert self.basket_color is not None, "Basket color not set."

        elapsed_time = time() - self.start_time
        if elapsed_time > self.timeout:
            return RetCode.TIMEOUT

        marker_based_timeout = self.timeout - self.timeout_refine_angle
        if elapsed_time > marker_based_timeout:
            self.peripheral_manager._node.get_logger().info(
                "Marker-based alignment is timeout. Switching to traditional basket alignment."
            )
            return self.align_to_basket(disable_timeout=True)  # return either SUCCESS or DOING

        # control robot based on marker poses if detected
        if (
            self.peripheral_manager.is_marker_detected(self.basket_color)
            and not self.is_marker_pose_extracted
        ):
            basket_dis_mm = self.peripheral_manager.get_basket_distance()
            if basket_dis_mm is not None:
                desired_dis_mm = basket_dis_mm
                if basket_dis_mm < Parameters.MANI_ALIGN_BASKET_ADV_VALID_DISTS_MM[0]:
                    desired_dis_mm = Parameters.MANI_ALIGN_BASKET_ADV_VALID_DISTS_MM[0]
                elif basket_dis_mm > Parameters.MANI_ALIGN_BASKET_ADV_VALID_DISTS_MM[1]:
                    desired_dis_mm = Parameters.MANI_ALIGN_BASKET_ADV_VALID_DISTS_MM[1]

                markers = self.peripheral_manager.get_detected_markers()
                if markers:
                    t_robot_desired_pos = self.get_expected_transformation_by_markers(
                        markers, desired_dis_mm
                    )
                    self.t_odom_tp = (
                        self.peripheral_manager.get_robot_to_odom_transform(True)
                        @ t_robot_desired_pos
                    )
                    if (
                        np.linalg.norm(t_robot_desired_pos[0:2, 3])
                        < Parameters.MANI_ALIGN_BASKET_ADV_DIS_THRESHOLD_MM
                    ):
                        self.is_marker_pose_extracted = True
                        self.peripheral_manager._node.get_logger().info("fixed position extracted.")

        if self.t_odom_tp is not None:
            # TODO: update t_odom_basket frequently to improve accuracy
            t_error = self.peripheral_manager.get_odom_to_robot_transform(True) @ self.t_odom_tp
            t_error[0:2, 0:2] = np.eye(2)  # ignore orientation error for position alignment
            dis_to_target = np.linalg.norm(t_error[0:2, 3])
            if dis_to_target > Parameters.MANI_ALIGN_BASKET_ADV_DIS_ODOM_THRESHOLD_MM:
                (vx, vy, wz) = self.compute_control_signals(
                    t_error,
                    (
                        Parameters.MANI_PID_LINEAR_ALIGN_BASKET_ADV,
                        Parameters.MANI_PID_ANGULAR_ALIGN_BASKET_ADV,
                    ),
                )

                if dis_to_target > Parameters.MANI_ALIGN_BASKET_ADV_ENABLED_DIST_PID_MM:
                    # normalize vx, vy to max speed
                    cur_speed = np.hypot(vx, vy)
                    if cur_speed < Parameters.MANI_ALIGN_BASKET_ADV_MAX_LINEAR_SPEED:
                        scale = Parameters.MANI_ALIGN_BASKET_ADV_MAX_LINEAR_SPEED / cur_speed
                        vx *= scale
                        vy *= scale

                self.peripheral_manager.move_robot_adv(
                    vx,
                    vy,
                    0.0,
                    thrower_percent=(
                        self.base_thrower_percent if self.base_thrower_percent is not None else 0.0
                    ),
                    servo_speed=0,
                    normalize=True,
                    max_xy_speed=Parameters.MANI_ALIGN_BASKET_ADV_MAX_LINEAR_SPEED,
                    max_rot_speed=Parameters.MANI_ALIGN_BASKET_ADV_MAX_ANGULAR_SPEED,
                )
                self.peripheral_manager._node.get_logger().info(
                    "Aligning to basket using marker pose on odom frame."
                )
                return RetCode.DOING
            else:
                self.t_odom_tp = None  # reached desired position
                self.reset_pid_errors()
                self.peripheral_manager._node.get_logger().info(
                    "Turning to traditional basket alignment."
                )

        self.peripheral_manager._node.get_logger().info(
            f"Marker based alignment is complete. Timestamp: {time() - self.start_time:.2f}s"
        )
        return self.align_to_basket(disable_timeout=True)  # return either SUCCESS or DOING

    def move_forward_to_grab(self) -> RetCode:
        """Move forward to grab the ball."""
        assert self.start_time is not None, "Handler not initialized."

        if time() - self.start_time > self.timeout:
            return RetCode.TIMEOUT

        if self.peripheral_manager.is_ball_grabbed():
            return RetCode.SUCCESS

        self.peripheral_manager.move_robot_adv(
            0.0,
            Parameters.MANI_GRAB_BALL_Y_SPEED,
            0.0,
            thrower_percent=0.0,
            servo_speed=Parameters.MANI_GRAB_BALL_SERVO_SPEED,
        )
        return RetCode.DOING

    def throw_ball(self) -> RetCode:
        """Throw the ball into the basket."""
        assert self.start_time is not None, "Handler not initialized."

        if time() - self.start_time > self.timeout:
            return RetCode.TIMEOUT

        # TURN OFF to avoid robot keep turning outside of the court
        # if self.peripheral_manager.is_basket_not_detected_in_nframes(
        #     Parameters.MANI_MAX_CONSECUTIVE_FRAMES_NO_BASKET
        # ):
        #     return RetCode.FAILED_BASKET_LOST

        if self.calculated_thrower_percent is None:
            thrower_percent = 50.0  # default thrower percent
            basket_distance_mm = self.peripheral_manager.get_basket_distance(
                avg_mode=True, num_samples=5
            )
            if basket_distance_mm is not None:
                thrower_percent = self.get_thrower_percent(
                    basket_distance_mm, offset_constant=Parameters.MAIN_THROW_BALL_OFFSET_PERCENT
                )
                self.calculated_thrower_percent = thrower_percent
                self.peripheral_manager._node.get_logger().info(
                    f"Basket dis: {basket_distance_mm} mm, thrower per.: {thrower_percent:.2f}%"
                )
        else:
            thrower_percent = self.calculated_thrower_percent

        basket_distance_mm = self.peripheral_manager.get_basket_distance(
            avg_mode=True, num_samples=5
        )
        self.peripheral_manager._node.get_logger().info(
            f"Basket dis: {basket_distance_mm} mm, thrower per.: {thrower_percent:.2f}%"
        )
        servo_speed = Parameters.MANI_THROW_BALL_SERVO_SPEED
        if time() - self.start_time < 0.5:
            servo_speed = 0  # avoid sudden movement at the start
        self.peripheral_manager.move_robot_adv(
            0.0,
            0.0,
            0.0,
            thrower_percent=thrower_percent,
            servo_speed=servo_speed,
        )
        return RetCode.DOING

    def clear_stuck_ball(self) -> RetCode:
        """Clear a stuck ball in the thrower mechanism."""
        assert self.start_time is not None, "Handler not initialized."

        if time() - self.start_time > self.timeout:
            return RetCode.TIMEOUT

        self.peripheral_manager.move_robot_adv(
            0.0,
            0.0,
            0.0,
            thrower_percent=100,
            servo_speed=Parameters.MANI_THROW_BALL_SERVO_SPEED,
        )
        return RetCode.DOING

    def turn_around_basket(self) -> RetCode:
        """Turn around the basket by a specified degree."""
        assert self.start_time is not None, "Handler not initialized."
        assert self.turning_basket_direction is not None, "Turning direction not set."

        if time() - self.start_time > self.timeout:
            self.peripheral_manager._node.get_logger().info("Turn around basket timeout.")
            return RetCode.TIMEOUT

        basket_pos_robot_mm: Optional[Tuple[float, float]] = None
        if self.peripheral_manager.is_basket_detected():
            basket_pos_robot_mm = self.peripheral_manager.get_basket_position_2d()

        if basket_pos_robot_mm is not None:
            la_dis_mm = min(
                Parameters.MANI_TURN_AROUND_BASKET_DISTANCE_MM,
                np.hypot(basket_pos_robot_mm[0], basket_pos_robot_mm[1]),
            )
            (vx, vy, wz) = self.compute_control_lookahead_woffset(
                basket_pos_robot_mm,
                (
                    Parameters.MANI_PID_LINEAR_ALIGN,
                    Parameters.MANI_PID_ANGULAR_ALIGN,
                ),
                la_dis_mm=la_dis_mm,
                angle_offset_deg=60 * self.turning_basket_direction,
            )
            self.peripheral_manager.move_robot_adv(
                vx,
                vy,
                wz,
                thrower_percent=0.0,
                servo_speed=0,
                normalize=True,
                max_xy_speed=Parameters.MANI_TURN_AROUND_BASKET_MAX_LINEAR_SPEED,
                max_rot_speed=Parameters.MANI_TURN_AROUND_BASKET_MAX_ANGULAR_SPEED,
            )
        return RetCode.DOING

    def position_to_pose(self, pos_robot_mm: Tuple[float, float]) -> np.ndarray:
        """Convert position in robot frame to a 4x4 pose matrix."""
        x_mm, y_mm = pos_robot_mm
        pose_robot = np.eye(4)
        pose_robot[0, 3] = x_mm
        pose_robot[1, 3] = y_mm
        return pose_robot

    def compute_control_signals(
        self, x_la_target: np.ndarray, pid_gains: Tuple[List[float], List[float]]
    ) -> Tuple[float, float, float]:
        """Compute control signals (vx, vy, wz) using PID control.
        Inputs:
            x_la_target: target pose w.r.t. robot's look-ahead frame (4x4 numpy array)
        Returns:
            (vx, vy, wz): tuple of control signals
        """
        assert x_la_target.shape == (4, 4), "look-ahead_T_target must be a 4x4 matrix."

        x_la_target[0, 3] /= 1000.0  # convert mm to m
        x_la_target[1, 3] /= 1000.0  # convert mm to m
        xe_log = mr.MatrixLog6(x_la_target)
        xe_vec = mr.se3ToVec(xe_log)
        vx_error = xe_vec[3]  # velocity error in x
        vy_error = xe_vec[4]  # velocity error in y
        wz_error = xe_vec[2]  # angular velocity error in z

        vx = self.compute_pid(0, vx_error, pid_gains[0])
        vy = self.compute_pid(1, vy_error, pid_gains[0])
        wz = self.compute_pid(2, wz_error, pid_gains[1])
        return vx, vy, wz

    def compute_pid(
        self,
        var: Literal[0, 1, 2],
        error: float,
        pid_gains: Union[List[float], Tuple[float, float, float]],
    ) -> float:
        """Compute PID control for manipulation tasks (vx=0, vy=1, wz=2)."""
        assert var in (0, 1, 2), "var must be 0 (vx), 1 (vy), or 2 (wz)."
        kp, ki, kd = pid_gains
        output = kp * error
        if var == 0:  # vx
            output += (
                kd * (error - self.prev_vx_error) * self.maneuver_rate + ki * self.cumm_vx_error
            )
            self.prev_vx_error = error
            self.cumm_vx_error += error / self.maneuver_rate

        elif var == 1:  # vy
            output += (
                kd * (error - self.prev_vy_error) * self.maneuver_rate + ki * self.cumm_vy_error
            )
            self.prev_vy_error = error
            self.cumm_vy_error += error / self.maneuver_rate
        else:  # wz
            output += (
                kd * (error - self.prev_wz_error) * self.maneuver_rate + ki * self.cumm_wz_error
            )
            self.prev_wz_error = error
            self.cumm_wz_error += error / self.maneuver_rate

        return float(output)

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

    def get_thrower_percent(self, dis_to_basket_mm: float, offset_constant: float = 0.0) -> float:
        """Get the thrower speed percentage based on distance to basket."""
        # Experimental data points: (distance in mm, motor percent)
        distances_mm = [dp[0] for dp in self.data_points]
        percents = [dp[1] for dp in self.data_points]
        # np.interp performs linear interpolation and handles extrapolation at boundaries
        percent = float(np.interp(dis_to_basket_mm, distances_mm, percents))
        # calculate offset dynamically based on distance
        distances_mm = [op[0] for op in self.offset_points]
        offsets = [op[1] for op in self.offset_points]

        if dis_to_basket_mm < distances_mm[0]:
            offset_dynamic = 0.0
        elif dis_to_basket_mm > distances_mm[-1]:
            offset_dynamic = offsets[-1]
        else:
            offset_dynamic = float(np.interp(dis_to_basket_mm, distances_mm, offsets))

        return percent + offset_constant + offset_dynamic

    def get_expected_transformation_by_markers(
        self, markers: List[Marker], prefered_dist_mm: float
    ) -> np.ndarray:
        """Get the expected transformation from robot base to new position based on marker pose."""
        assert len(markers) in (1, 2), "Number of markers must be 1 or 2."
        markers = sorted(markers, key=lambda m: m.id)
        if len(markers) == 1:
            marker = markers[0]
            t_newr_marker = np.eye(4)
            t_newr_marker[1, 3] = prefered_dist_mm
            if marker.id % 2 != 0:
                # left markers
                t_newr_marker[0, 3] = -Parameters.MANI_ALIGN_BASKET_ADV_MARKER_OFFSET_X_MM
            else:
                # right markers
                t_newr_marker[0, 3] = Parameters.MANI_ALIGN_BASKET_ADV_MARKER_OFFSET_X_MM
            t_r_marker = np.eye(4)
            t_r_marker[0:2, 3] = marker.position_2d
            self.peripheral_manager._node.get_logger().info(
                f"Marker ID {marker.id} position_2d: {marker.position_2d}, theta: {marker.theta}"
            )
            # clip theta to prevent robot from turning behind the basket
            t_r_marker[:2, :2] = get_rotation_matrix(np.deg2rad(np.clip(marker.theta, -85, 85)))
            return t_r_marker @ np.linalg.inv(t_newr_marker)
        else:
            # average the transformations from multiple markers
            t_newr_basket = np.eye(4)
            t_newr_basket[1, 3] = prefered_dist_mm
            t_r_basket = np.eye(4)
            t_r_basket[0:2, 3] = np.mean([marker.position_2d for marker in markers], axis=0)
            # compute angle based on two markers to avoid errors in single marker detection
            theta = math.atan2(
                markers[1].position_2d[1] - markers[0].position_2d[1],
                markers[1].position_2d[0] - markers[0].position_2d[0],
            )
            self.peripheral_manager._node.get_logger().info(
                f"Marker IDs {[marker.id for marker in markers]},"
                + f" thetas: {[marker.theta for marker in markers]},"
                + f" computed theta: {np.rad2deg(theta):.2f}"
            )
            t_r_basket[:2, :2] = get_rotation_matrix(
                np.clip(theta, -85 * math.pi / 180, 85 * math.pi / 180)
            )
            return t_r_basket @ np.linalg.inv(t_newr_basket)
