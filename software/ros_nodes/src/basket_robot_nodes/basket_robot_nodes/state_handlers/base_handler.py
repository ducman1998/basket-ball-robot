import math
from time import time
from typing import Optional, Tuple, Union, cast

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

    def __init__(self, peripheral_manager: PeripheralManager) -> None:
        self.peripheral_manager = peripheral_manager
        self.start_time: Optional[float] = None  # time when the handler is initialized
        self.start_yaw: Optional[float] = None  # yaw at the start of the handler
        self.previous_yaw: Optional[float] = None  # yaw in the previous control loop
        self.start_pos: Optional[Tuple[float, float]] = None  # (x, y) at the start of the handler
        self.cummulative_yaw_change: float = 0.0  # total yaw change since start
        # target variables
        self.target_cummulative_yaw_change: Optional[float] = None
        self.target_distance: Optional[float] = None
        self.timeout: float = 10.0  # maximum allowed time for the handler

        # internal variables for discrete turn handling
        self.turn_discrete_new_cycle_start_time: Optional[float] = None
        self.turn_discrete_stop_timestamp: Optional[float] = None
        self.num_discrete_turns: Optional[int] = None
        self.finished_discrete_turns: int = 0

    def initialize(
        self,
        action: BaseAction,
        angle: Optional[Union[float, int]] = None,
        distance: Optional[float] = None,
        timeout: float = 10.0,  # seconds
    ) -> None:
        """Initialize the handler state."""
        if action == BaseAction.TURN_CONTINUOUS or action == BaseAction.TURN_DISCRETE:
            assert angle is not None, "Angle must be provided for rotation actions."
        if action == BaseAction.MOVE_FORWARD or action == BaseAction.MOVE_SIDEWAY:
            assert distance is not None, "Distance must be provided for movement actions."

        # reset internal states
        self.reset()

        if angle is not None:
            angle = abs(angle)

        self.start_time = time()
        self.start_yaw = self.peripheral_manager.get_odom_yaw()
        self.previous_yaw = self.start_yaw
        self.start_pos = self.peripheral_manager.get_robot_odom_position()
        # set target variables
        self.target_cummulative_yaw_change = angle
        self.target_distance = distance
        self.timeout = timeout

        # set internal variables for discrete turn handling
        if action == BaseAction.TURN_DISCRETE and angle is not None:
            self.num_discrete_turns = math.ceil(angle / Parameters.BASE_DISCRETE_TURN_SUB_ANGLE_DEG)
            self.turn_discrete_new_cycle_start_time = time()

    def reset(self) -> None:
        self.start_time = None
        self.start_yaw = None
        self.previous_yaw = None
        self.start_pos = None
        self.cummulative_yaw_change = 0.0
        # reset target variables
        self.target_cummulative_yaw_change = None
        self.target_distance = None
        # reset internal variables for discrete turn handling
        self.num_discrete_turns = None
        self.finished_discrete_turns = 0
        self.turn_discrete_stop_timestamp = None

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

        if warm_start:
            wz = self.warm_up_turn_speed(wz, ramp_duration)

        self.peripheral_manager.move_robot(0.0, 0.0, wz)
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
        assert self.num_discrete_turns is not None, "Number of discrete turns not set."
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
            self.peripheral_manager.move_robot(0.0, 0.0, wz)

        if abs(self.cummulative_yaw_change) >= abs(self.target_cummulative_yaw_change):
            return RetCode.SUCCESS

        if time() - self.start_time >= self.timeout:
            return RetCode.TIMEOUT

        return RetCode.DOING

    def move_robot_sideway(
        self,
        speed: float = Parameters.BASE_MOVE_SIDEWAY_SPEED,
        warm_start: bool = Parameters.BASE_MOVE_SIDEWAY_WARMUP_ENABLED,
        ramp_duration: float = Parameters.BASE_WARMUP_RAMP_DURATION,
    ) -> RetCode:
        assert (
            self.start_time is not None
            and self.target_distance is not None
            and self.start_pos is not None
        ), "Handler not initialized."

        vx = speed
        vy = 0.0
        if warm_start:
            vx, vy = self.warm_up_linear_speed(vx, vy, ramp_duration)

        self.peripheral_manager.move_robot(vx, vy, 0.0)
        current_pos = self.peripheral_manager.get_robot_odom_position()
        distance_moved = math.sqrt(
            (current_pos[0] - self.start_pos[0]) ** 2 + (current_pos[1] - self.start_pos[1]) ** 2
        )
        if distance_moved >= abs(self.target_distance):
            return RetCode.SUCCESS

        if time() - self.start_time >= self.timeout:
            return RetCode.TIMEOUT

        return RetCode.DOING

    def move_robot_forward(
        self,
        speed: float = Parameters.BASE_MOVE_FORWARD_SPEED,
        warm_start: bool = Parameters.BASE_MOVE_FORWARD_WARMUP_ENABLED,
        ramp_duration: float = Parameters.BASE_WARMUP_RAMP_DURATION,
    ) -> RetCode:
        assert (
            self.start_time is not None
            and self.target_distance is not None
            and self.start_pos is not None
        ), "Handler not initialized."

        vx = 0.0
        vy = speed
        if warm_start:
            vx, vy = self.warm_up_linear_speed(vx, vy, ramp_duration)

        self.peripheral_manager.move_robot(vx, vy, 0.0)
        current_pos = self.peripheral_manager.get_robot_odom_position()
        distance_moved = math.sqrt(
            (current_pos[0] - self.start_pos[0]) ** 2 + (current_pos[1] - self.start_pos[1]) ** 2
        )
        if distance_moved >= abs(self.target_distance):
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
