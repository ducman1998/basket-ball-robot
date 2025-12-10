import math
from collections import deque
from time import time
from typing import List, Literal, Optional, Sequence, Tuple, Union

import modern_robotics as mr
import numpy as np
import rclpy
from basket_robot_nodes.utils.constants import BASE_FRAME_ID
from basket_robot_nodes.utils.image_info import GreenBall
from basket_robot_nodes.utils.number_utils import FrameStabilityCounter
from basket_robot_nodes.utils.peripheral_manager import PeripheralManager
from basket_robot_nodes.utils.referee_client import RefereeClient
from basket_robot_nodes.utils.ros_utils import (
    int_descriptor,
    log_initialized_parameters,
    parse_log_level,
    str_descriptor,
)
from nav_msgs.msg import Odometry
from rclpy.clock import Clock
from rclpy.node import Node
from scipy.spatial.transform import Rotation as R

IS_DEV = True
SAMPLING_RATE = 60  # Hz
FT_DETECTED_SEARCH_BALL = 6  # frames
FT_UNDETECTED_REACH_BALL = 15  # frames
FT_UNDETECTED_ALIGN_BALL = 20  # frames
FT_DETECTED_APPROACH_BASKET = 40  # frames
FT_UNDETECTED_IDLE = 120  # frames
FT_DETECTED_STABLE_ALIGN_BALL = 10  # frames
TOTAL_ROT_DEGREE_THRESHOLD = 270.0  # seconds
ALIGNING_TIMEOUT = 4.5  # seconds
BASKET_DISTANCE_QUEUE_SIZE = 20  # frames for moving average


class GameState:
    """Enum-like class for game states."""

    INIT: int = 0
    SEARCH_BALL: int = 1
    REACH_BALL: int = 2
    ALIGN_TO_BASKET: int = 3
    THROW_BALL: int = 4
    APROACH_BASKET: int = 5
    IDLE: int = 6

    def __init__(self) -> None:
        self.state2name = {v: k for k, v in vars(GameState).items() if isinstance(v, int)}

    def get_state_name(self, state_value: int) -> str:
        return self.state2name.get(state_value, "UNKNOWN")


class GameLogicController(Node):
    def __init__(self) -> None:
        # Initialize the Game Logic Controller node
        super().__init__("game_logic_controller_node")
        # declare parameters
        self._declare_node_parameter()
        # read parameters
        self._read_node_parameters()
        # for checking: log all initialized parameters
        log_initialized_parameters(self)

        # subscriptions to sensors, offering convenient access/control sensors
        self.periph_manager = PeripheralManager(self)

        # game timer to run game logic at 20Hz
        self.game_timer = self.create_timer(1 / SAMPLING_RATE, self.game_logic_loop)

        self.cur_state = GameState.INIT
        self.timestamp = time()

        # state variables
        # INIT: initial state
        # TODO: add other state variables as needed
        # SEARCH_BALL: look for balls
        self.last_angle: float = 0.0  # starting angle for searching
        self.cummulative_rotation: float = 0.0  # track rotation during searching

        # REACH_BALL: move towards the closest ball
        self.prev_vx_error: float = 0.0
        self.prev_vy_error: float = 0.0
        self.prev_wz_error: float = 0.0
        self.cumm_vx_error: float = 0.0
        self.cumm_vy_error: float = 0.0
        self.cumm_wz_error: float = 0.0
        # ALIGN_TO_BASKET: align to basket (not implemented)
        # TODO: add other state variables as needed
        self.align_start_time: Optional[float] = None
        # APPROACH_BASKET: move towards court center
        self.fathest_basket_pos: Optional[Tuple[float, float]] = None
        self.fathest_basket_dis: float = 0.0
        # THROW_BALL: throw the ball
        self.throw_start_pos: Optional[Tuple[float, float]] = None
        self.basket_distance_queue: deque = deque(maxlen=BASKET_DISTANCE_QUEUE_SIZE)
        # IDLE: stop
        # TODO: add other state variables as needed

        # counters for frame stability, count consecutive frames of detection/loss
        self.fcounter_search_state = FrameStabilityCounter(FT_DETECTED_SEARCH_BALL)
        self.fcounter_reach_state = FrameStabilityCounter(FT_UNDETECTED_REACH_BALL)
        self.fcounter_align_state = FrameStabilityCounter(FT_UNDETECTED_ALIGN_BALL)
        self.fcounter_align_stable_state = FrameStabilityCounter(FT_DETECTED_STABLE_ALIGN_BALL)
        self.fcounter_approach_state = FrameStabilityCounter(FT_DETECTED_APPROACH_BASKET)
        self.fcounter_idle_state = FrameStabilityCounter(FT_UNDETECTED_IDLE)

        # Referee client state
        self.is_game_started = False  # True when referee sends START, False when STOP
        self.opponent_basket_color = "n/a"  # default opponent basket color
        if IS_DEV:
            self.get_logger().warn("Running in DEV mode: Referee signals are simulated.")
            self.opponent_basket_color = "green"  # for testing

        # Initialize and start referee client
        self.referee_client = RefereeClient(
            robot_id=self.robot_id,
            referee_ip=self.referee_ip_address,
            referee_port=self.referee_port,
            on_signal=self.handle_referee_signals,
            logger=self.get_logger(),
        )
        self.referee_client.start()
        self.get_logger().info(f"Started referee client for robot ID: {self.robot_id}")

    def _declare_node_parameter(self) -> None:
        """Declare parameters with descriptors."""
        self.declare_parameter("referee_ip_address", descriptor=str_descriptor)
        self.declare_parameter("referee_port", descriptor=int_descriptor)
        self.declare_parameter("robot_id", descriptor=str_descriptor)
        self.declare_parameter("log_level", descriptor=str_descriptor)

    def _read_node_parameters(self) -> None:
        """Read parameters into class variables."""
        # Read all parameters
        self.referee_ip_address = (
            self.get_parameter("referee_ip_address").get_parameter_value().string_value
        )
        self.referee_port = self.get_parameter("referee_port").get_parameter_value().integer_value
        self.robot_id = self.get_parameter("robot_id").get_parameter_value().string_value
        log_level = self.get_parameter("log_level").get_parameter_value().string_value

        # Validate all parameters
        self._validate_parameters()

        # Set logging level
        self.get_logger().set_level(parse_log_level(log_level))
        self.get_logger().info(f"Set node {self.get_name()} log level to {log_level}.")

    def _validate_parameters(self) -> None:
        """Validate all node parameters."""
        # Validate referee connection
        if not self.referee_ip_address or not isinstance(self.referee_port, int):
            raise ValueError("Invalid referee IP address or port number.")

        if not self.robot_id or not isinstance(self.robot_id, str):
            raise ValueError("Invalid robot ID.")

    def handle_referee_signals(
        self, signal: Literal["start", "stop"], basket: Optional[str]
    ) -> None:
        """Handle START/STOP signal from referee."""
        if signal == "start":
            if self.is_game_started:
                self.get_logger().warn("Received duplicate START signal from referee --> Ignored.")
            else:
                self.opponent_basket_color = basket if basket is not None else "n/a"
                self.is_game_started = True
        else:
            self.is_game_started = False
            # Stop robot movement immediately
            self.stop_robot()

    def game_logic_loop(self) -> None:
        """Main game logic loop, called periodically by a timer."""
        start_time = time()
        self.print_current_state()

        # If game is not active (referee hasn't started or has stopped), don't execute game logic
        if not self.is_game_started and not IS_DEV:
            # if we were playing and referee stopped us, ensure robot is stopped
            self.stop_robot()
            self.cur_state = GameState.INIT
            return

        match self.cur_state:
            case GameState.INIT:
                self.handle_init_state()

            case GameState.SEARCH_BALL:
                self.handle_search_ball_state()

            case GameState.REACH_BALL:
                self.handle_reach_ball_state()

            case GameState.ALIGN_TO_BASKET:
                self.handle_align_to_basket_state()

            case GameState.THROW_BALL:
                self.throw_ball_state()

            case GameState.APROACH_BASKET:
                self.handle_approach_basket_state()

            case GameState.IDLE:
                self.handle_idle_state()

            case _:
                raise RuntimeError("Unknown game state!")

        end_time = time()
        self.get_logger().info(f"Game logic loop took {end_time - start_time:.4f} seconds.")

    # state handlers
    def handle_init_state(self) -> None:
        for _ in range(3):
            self.stop_robot()

        if self.odom_msg and self.image_info_msg:
            self.transition_to_state(GameState.SEARCH_BALL)
        return None

    def handle_search_ball_state(self) -> None:
        # if a ball is detected, transition to REACH_BALL state
        if self.fcounter_search_state.update(self.periph_manager.is_ball_detected()):
            self.transition_to_state(GameState.REACH_BALL)
            return None

        # If no ball detected after rotating {TOTAL_ROT_DEGREE_THRESHOLD} degree --> IDLE
        cumulative_rotation = self.compute_cumulative_rotation()
        if abs(cumulative_rotation) >= TOTAL_ROT_DEGREE_THRESHOLD:
            if self.fathest_basket_pos is not None:
                self.transition_to_state(GameState.APROACH_BASKET)
                return None
            else:
                self.transition_to_state(GameState.IDLE)
                return None

        # rotate the robot to find balls
        self.rotate_robot(self.search_rot_speed)

        # update the fatherest basket if possible
        if (
            self.image_info_msg.basket is not None
            and self.image_info_msg.basket.position_2d is not None
        ):
            basket_pos = self.image_info_msg.basket.position_2d  # in mm
            basket_dis = float(np.linalg.norm(basket_pos))
            if self.fathest_basket_pos and basket_dis <= self.fathest_basket_dis:
                return None  # not farther than previous

            # transformation from robot base to odometry frame
            t_o_r = np.eye(4)
            quat = self.odom_msg.pose.pose.orientation
            t_o_r[0, 3] = self.odom_msg.pose.pose.position.x
            t_o_r[1, 3] = self.odom_msg.pose.pose.position.y
            t_o_r[:3, :3] = R.from_quat([quat.x, quat.y, quat.z, quat.w]).as_matrix()
            # scale basket position to a fixed distance for better accuracy (2 meters)
            scaled_ratio = 2000.0 / basket_dis
            fathest_basket_r = np.array(
                [
                    basket_pos[0] / 1000.0 * scaled_ratio,  # in meters
                    basket_pos[1] / 1000.0 * scaled_ratio,  # in meters
                    0.0,
                    1.0,
                ]
            ).reshape((4, 1))

            target_pos_o = t_o_r @ fathest_basket_r
            self.fathest_basket_pos = tuple(target_pos_o.ravel()[:2] * 1000.0)  # in mm
            self.fathest_basket_dis = basket_dis
        self.get_logger().info(
            f"Searching... Current yaw: {self.last_angle:.2f} degrees, "
            f"Cumulative rotation: {self.cummulative_rotation:.2f} degrees."
        )
        return None

    def handle_approach_basket_state(self) -> None:
        # If a ball is detected, transition to REACH_BALL state
        if self.fcounter_approach_state.update(self.is_ball_in_view()):
            self.get_logger().info("Ball detected! Transitioning to REACH_BALL state.")
            self.transition_to_state(GameState.REACH_BALL)
            return None

        best_basket_rob_frame = self.get_target_in_robot_frame(self.fathest_basket_pos)
        if best_basket_rob_frame:
            vx, vy, wz = self.compute_control_signals(best_basket_rob_frame, look_ahead_dis=0.15)
            self.move_robot(vx, vy, wz, 0, normalize=True)
            self.get_logger().info(
                f"Moving towards the farthest basket: vx={vx:.2f}, vy={vy:.2f}, wz={wz:.2f}, "
                f"pos=({best_basket_rob_frame[0]:.1f}, {best_basket_rob_frame[1]:.1f})mm"
            )
            if np.sqrt(best_basket_rob_frame[0] ** 2 + best_basket_rob_frame[1] ** 2) <= 200.0:
                self.transition_to_state(GameState.SEARCH_BALL)
                self.get_logger().info(
                    "Reached the farthest basket! Transitioning to SEARCH_BALL state."
                )
            return None
        else:
            self.transition_to_state(GameState.IDLE)
            self.get_logger().info("No basket information available. Transitioning to IDLE state.")
            return None

    def handle_align_to_basket_state(self) -> None:
        assert self.image_info_msg is not None
        assert self.image_size is not None

        # Initialize alignment start time on first entry
        if self.align_start_time is None:
            self.align_start_time = time()

        # Check for timeout
        elapsed_time = time() - self.align_start_time
        if elapsed_time > ALIGNING_TIMEOUT:
            self.get_logger().warn(
                f"Alignment timeout ({ALIGNING_TIMEOUT}s) exceeded! "
                f"Transitioning to THROW_BALL state."
            )
            self.transition_to_state(GameState.THROW_BALL)
            return None

        if self.fcounter_align_state.update(not self.is_ball_in_view()):
            self.get_logger().info("Lost sight of the ball. Returning to SEARCH_BALL state.")
            self.transition_to_state(GameState.SEARCH_BALL)
            return None
        elif self.is_ball_in_view():
            closet_ball: GreenBall = min(
                self.image_info_msg.balls,
                key=lambda b: math.hypot(*b.position_2d) if b.position_2d else float("inf"),
            )
            if (
                self.image_info_msg.basket is not None
                and self.image_info_msg.basket.position_2d is not None
                and self.image_info_msg.basket.color == self.opponent_basket_color
            ):
                basket_pos = self.image_info_msg.basket.position_2d  # in mm
                basket_distance_m = float(np.linalg.norm(basket_pos) / 1000.0)
                alignment_threshold = self.get_alignment_threshold(basket_distance_m)

                offset_angle = self.measure_angle_error(self.image_info_msg.basket.center)
                self.get_logger().info(
                    f"Offset angle: {offset_angle:.2f}°, threshold: {alignment_threshold:.2f}°, "
                    f"distance: {basket_distance_m:.2f}m"
                )
                vx, vy, wz = self.compute_control_signals(
                    closet_ball.position_2d, angle_offset=offset_angle, look_ahead_dis=0.15
                )
                self.move_robot(vx, vy, wz, 0, normalize=True, override_max_xy_speed=0.4)
                if self.fcounter_align_stable_state.update(abs(offset_angle) < alignment_threshold):
                    self.transition_to_state(GameState.THROW_BALL)
                return None
            else:
                vx, vy, wz = self.compute_control_signals(
                    closet_ball.position_2d, angle_offset=90.0, look_ahead_dis=0.20
                )
                self.move_robot(vx, vy, wz, 0, normalize=True)
                self.get_logger().info(
                    f"Aligning to basket: vx={vx:.2f}, vy={vy:.2f}, wz={wz:.2f}, "
                    f"pos=({closet_ball.position_2d[0]:.1f}, {closet_ball.position_2d[1]:.1f})mm"
                )
                return None
        else:
            self.get_logger().info("No ball detected. Remaining in ALIGN_TO_BASKET state.")
            return None

    def throw_ball_state(self) -> None:
        assert self.odom_msg is not None
        assert self.image_info_msg is not None

        current_pos_odom = (
            self.odom_msg.pose.pose.position.x,
            self.odom_msg.pose.pose.position.y,
        )
        if self.throw_start_pos is None:
            self.throw_start_pos = current_pos_odom

        # move forward a bit to throw the ball into the basket
        if self.image_info_msg.basket is None or self.image_info_msg.basket.position_2d is None:
            self.move_robot(0.0, self.throwing_xy_speed, 0.0, 60.0, normalize=False)
            self.get_logger().info(
                "No basket detected! Throwing ball with default motor percent=60."
            )
        else:
            basket_pos = self.image_info_msg.basket.position_2d  # in mm
            basket_distance = float(np.linalg.norm(basket_pos) / 1000.0)  # convert to meters

            # Add current distance to queue for moving average
            self.basket_distance_queue.append(basket_distance)

            # Calculate moving average distance
            avg_basket_distance = float(np.mean(self.basket_distance_queue))

            motor_percent = self.motor_percent_from_basket(avg_basket_distance)
            offset_angle = self.measure_angle_error(self.image_info_msg.basket.center)
            self.get_logger().info(
                f"Throwing: offset angle to basket: {offset_angle:.2f} degrees. "
                f"Distance: current={basket_distance:.2f}m, avg={avg_basket_distance:.2f}m "
                f"(n={len(self.basket_distance_queue)})"
            )
            vx, vy, wz = self.compute_control_signals(
                (0, 500),  # target 500mm (0.5m) forward in robot frame
                angle_offset=offset_angle * 1.5,
                look_ahead_dis=0.25,
            )
            # normalize vx, vy to a fixed speed of 0.4 m/s
            vx, vy = self.normalize_velocity(vx, vy, self.throwing_xy_speed)
            self.move_robot(vx, vy, wz, motor_percent, normalize=False)
            self.get_logger().info(
                f"Throwing ball: vx={vx:.2f}, vy={vy:.2f}, wz={wz:.2f}, "
                f"basket pos=({basket_pos[0]:.1f}, {basket_pos[1]:.1f})mm, motor={motor_percent}%"
            )
        if (
            np.linalg.norm(np.array(current_pos_odom) - np.array(self.throw_start_pos)) >= 0.5
        ):  # in meters
            self.transition_to_state(GameState.SEARCH_BALL)
        return None

    def handle_reach_ball_state(self) -> None:
        assert self.image_info_msg is not None

        if self.fcounter_reach_state.update(not self.is_ball_in_view()):
            self.get_logger().info("Lost sight of the ball. Returning to SEARCH_BALL state.")
            self.transition_to_state(GameState.SEARCH_BALL)
            return None
        elif self.is_ball_in_view():
            # move towards the closest ball
            closet_ball: GreenBall = min(
                self.image_info_msg.balls,
                key=lambda b: math.hypot(*b.position_2d) if b.position_2d else float("inf"),
            )
            stop_condition = self.check_stop_condition(closet_ball)
            if stop_condition:
                self.transition_to_state(GameState.ALIGN_TO_BASKET)
                self.get_logger().info(
                    "Reached the ball! Stopping and returning to ALIGN_TO_BASKET state."
                )
                return None

            # transformation matrix from the ball frame to robot base_footprint frame
            ball_pos = closet_ball.position_2d  # in mm
            # desired target pose (4x4 matrix) in the robot base_foot
            vx, vy, wz = self.compute_control_signals(ball_pos)
            # move the robot towards the ball
            self.move_robot(vx, vy, wz, 0, normalize=True)
            self.get_logger().info(
                f"Moving towards ball: vx={vx:.2f}, vy={vy:.2f}, wz={wz:.2f}, "
                f"pos=({ball_pos[0]:.1f}, {ball_pos[1]:.1f})mm"
            )
            return None
        else:
            self.get_logger().info("No ball detected. Remaining in REACH_BALL state.")
            return None

    def handle_idle_state(self) -> None:
        assert self.image_info_msg is not None

        self.stop_robot()
        # transition to searching state if the ball is disappeared
        if self.fcounter_idle_state.update(not self.is_ball_in_view()):
            self.transition_to_state(GameState.SEARCH_BALL)
            self.get_logger().info(
                "No ball detected for a while in IDLE state. "
                + "Transitioning to SEARCH_BALL state."
            )
            return None
        elif self.is_ball_in_view():
            # move towards the closest ball
            closet_ball: GreenBall = min(
                self.image_info_msg.balls,
                key=lambda b: math.hypot(*b.position_2d) if b.position_2d else float("inf"),
            )
            stop_condition = self.check_stop_condition(closet_ball)
            if not stop_condition:
                self.transition_to_state(GameState.REACH_BALL)
                return None
        else:
            self.get_logger().info("No ball detected. Remaining in IDLE state.")
            return None

    def move_robot(
        self,
        vx: float,
        vy: float,
        wz: float,
        thrower_percent: float,
        normalize: bool = False,
        override_max_xy_speed: Optional[float] = None,
    ) -> None:
        """Send velocity commands to the robot. vx, vy in m/s, wz in rad/s."""
        if normalize:
            if override_max_xy_speed is not None and override_max_xy_speed > 0:
                vx, vy = self.normalize_velocity(vx, vy, override_max_xy_speed)
            else:
                vx, vy = self.normalize_velocity(vx, vy, self.max_xy)
        else:
            if override_max_xy_speed is not None and override_max_xy_speed > 0:
                vx = np.clip(vx, -override_max_xy_speed, override_max_xy_speed)
                vy = np.clip(vy, -override_max_xy_speed, override_max_xy_speed)
            else:
                vx = np.clip(vx, -self.max_xy, self.max_xy)
                vy = np.clip(vy, -self.max_xy, self.max_xy)

        wz = np.clip(wz, -self.max_rot, self.max_rot)
        thrower_percent = np.clip(thrower_percent, 0, 100)

        self.control_msg.header.stamp = Clock().now().to_msg()
        self.control_msg.header.frame_id = BASE_FRAME_ID
        self.control_msg.twist.linear.x = float(vx)
        self.control_msg.twist.linear.y = float(vy)
        self.control_msg.twist.linear.z = 0.0
        self.control_msg.twist.angular.x = 0.0
        self.control_msg.twist.angular.y = 0.0
        self.control_msg.twist.angular.z = float(wz)
        self.control_msg.thrower_percent = float(thrower_percent)
        self.mainboard_controller_pub.publish(self.control_msg)

    def stop_robot(self) -> None:
        """Stop the robot by sending zero velocity commands."""
        self.move_robot(0.0, 0.0, 0.0, 0.0)

    def rotate_robot(self, wz: float) -> None:
        """Rotate the robot at a given angular velocity wz (rad/s)."""
        self.move_robot(0.0, 0.0, wz, 0.0)

    def get_target_in_robot_frame(
        self, target_pos_o: Optional[Tuple[float, float]]
    ) -> Optional[Tuple[float, float]]:
        """
        Get the target ball position in robot base frame (mm).
        Inputs:
            target_pos_o: target position in odometry frame (mm).
        Returns:
            target_pos_r: target position in robot base frame (mm).
        """
        if target_pos_o is None or self.odom_msg is None:
            return None

        t_o_r = np.eye(4)  # transformation from robot base to odometry frame
        quat = self.odom_msg.pose.pose.orientation
        t_o_r[0, 3] = self.odom_msg.pose.pose.position.x * 1000.0
        t_o_r[1, 3] = self.odom_msg.pose.pose.position.y * 1000.0
        t_o_r[:3, :3] = R.from_quat([quat.x, quat.y, quat.z, quat.w]).as_matrix()
        # target position in robot base-footprint frame
        target_pos_homo = np.array([*target_pos_o, 0.0, 1.0]).reshape((4, 1))
        target_pos_r = (np.linalg.pinv(t_o_r) @ target_pos_homo).ravel()
        return (target_pos_r[0], target_pos_r[1])

    def compute_control_signals(
        self,
        target_pos: Union[List[float], Tuple[float, float]],
        angle_offset: float = 0.0,  # offset angle in degrees
        look_ahead_dis: float = 0.25,
    ) -> Tuple[float, float, float]:
        """
        Compute control signals (vx, vy, wz) to reach the target position.
        Inputs:
            target_pos: target position in robot base frame (mm).
            angle_offset: angle offset in degrees.
            look_ahead_dis: look-ahead distance in meters.
        Returns:
            Control signals (vx, vy, wz) to reach the target position.
        """
        target_pos_m = [target_pos[0] / 1000.0, target_pos[1] / 1000.0]  # convert to meters
        x_desired = np.eye(4)
        heading_error = -math.atan2(target_pos_m[0], target_pos_m[1]) + math.radians(angle_offset)
        x_desired[0, :] = [
            np.cos(heading_error),
            -np.sin(heading_error),
            0.0,
            float(target_pos_m[0]),  # target x in m
        ]
        # set look-ahead point at 150mm in front of the robot
        x_desired[1, :] = [
            np.sin(heading_error),
            np.cos(heading_error),
            0.0,
            target_pos_m[1] - look_ahead_dis,  # target y in m
        ]
        x_desired[2, :] = [0.0, 0.0, 1.0, 0.0]
        x_desired[3, :] = [0.0, 0.0, 0.0, 1.0]
        xe_log = mr.MatrixLog6(x_desired)
        xe_vec = mr.se3ToVec(xe_log)
        vx_error = xe_vec[3]  # velocity error in x
        vy_error = xe_vec[4]  # velocity error in y
        wz_error = xe_vec[2]  # angular velocity error in z

        # PD control
        if self.cur_state == GameState.ALIGN_TO_BASKET:
            kp_xy, ki_xy, kd_xy = self.pid_linear_align_basket
            kp_rot, ki_rot, kd_rot = self.pid_angular_align_basket
        elif self.cur_state == GameState.THROW_BALL:
            kp_xy, ki_xy, kd_xy = self.pid_linear_throw_ball
            kp_rot, ki_rot, kd_rot = self.pid_angular_throw_ball
        else:
            kp_xy, ki_xy, kd_xy = self.pid_linear_common
            kp_rot, ki_rot, kd_rot = self.pid_angular_common

        vx = (
            kp_xy * vx_error
            + kd_xy * (vx_error - self.prev_vx_error) * SAMPLING_RATE
            + ki_xy * self.cumm_vx_error
        )
        vy = (
            kp_xy * vy_error
            + kd_xy * (vy_error - self.prev_vy_error) * SAMPLING_RATE
            + ki_xy * self.cumm_vy_error
        )
        wz = (
            kp_rot * wz_error
            + kd_rot * (wz_error - self.prev_wz_error) * SAMPLING_RATE
            + ki_rot * self.cumm_wz_error
        )

        self.prev_vx_error = vx_error
        self.prev_vy_error = vy_error
        self.prev_wz_error = wz_error
        self.cumm_vx_error += vx_error / SAMPLING_RATE
        self.cumm_vy_error += vy_error / SAMPLING_RATE
        self.cumm_wz_error += wz_error / SAMPLING_RATE
        return vx, vy, wz

    def yaw_from_odom(self, odom_msg: Odometry) -> float:
        """
        Extract yaw angle in degrees from odometry message.
        Returns yaw in [0, 360) degrees.
        """
        if not odom_msg:
            raise ValueError("No odometry message available.")

        quat = odom_msg.pose.pose.orientation
        yaw_rad = 2.0 * math.atan2(quat.z, quat.w)
        yaw_deg = math.degrees(yaw_rad)
        yaw_deg = (yaw_deg + 360) % 360
        return yaw_deg

    def shortest_angular_difference(self, cur_angle: float, prev_angle: float) -> float:
        """
        Calculates the shortest angular difference (target - current) in degrees,
        normalized to the range [-180, 180].
        """
        raw_error = cur_angle - prev_angle
        if raw_error > 180.0:
            raw_error -= 360.0
        elif raw_error < -180.0:
            raw_error += 360.0
        return raw_error

    def check_stop_condition(self, closest_ball: GreenBall) -> bool:
        """Check if the robot has reached the ball based on distance and angle thresholds."""
        ball_pos = closest_ball.position_2d
        # if close enough to the ball, stop
        # 350mm distance threshold (robot center to the ball)
        distance_check = math.hypot(ball_pos[0], ball_pos[1]) <= 350.0
        angle_check = abs(math.atan2(ball_pos[0], ball_pos[1])) <= math.radians(3)  # 3 degrees
        return distance_check and angle_check

    def is_ball_in_view(self) -> bool:
        """Check if any ball is currently detected in the image info."""
        if self.image_info_msg and len(self.image_info_msg.balls) > 0:
            return True
        return False

    def transition_to_state(self, new_state: int) -> None:
        """Transition to a new game state."""
        pre_state = self.cur_state
        self.cur_state = new_state
        if new_state == GameState.SEARCH_BALL:
            self.last_angle = self.yaw_from_odom(self.odom_msg)
            self.cummulative_rotation = 0.0
            self.fathest_basket_pos = None
            self.fathest_basket_dis = 0.0
        elif new_state == GameState.IDLE:
            self.stop_robot()
        elif new_state == GameState.THROW_BALL:
            self.throw_start_pos = None
            self.basket_distance_queue.clear()
        elif new_state == GameState.ALIGN_TO_BASKET:
            self.align_start_time = None

        if pre_state != new_state:
            self.prev_vx_error = 0.0
            self.prev_vy_error = 0.0
            self.prev_wz_error = 0.0
            self.cumm_vx_error = 0.0
            self.cumm_vy_error = 0.0
            self.cumm_wz_error = 0.0

        pre_state_name = GameState().get_state_name(pre_state)
        new_state_name = GameState().get_state_name(new_state)
        self.get_logger().info(f"Transitioning from state {pre_state_name} to {new_state_name}.")

    def print_current_state(self) -> None:
        """Log the current state and relevant information."""
        state_name = GameState().get_state_name(self.cur_state)
        if not self.referee_client.is_connected():
            status = "DISCONNECTED"
        else:
            status = "STARTED" if self.is_game_started else "INACTIVE"
        self.get_logger().info(
            f"Current State: {state_name} | Referee Status: {status}"
            + f" | Color: {self.opponent_basket_color}"
        )

    def compute_cumulative_rotation(self) -> float:
        yaw_now = self.yaw_from_odom(self.odom_msg)
        delta_yaw = self.shortest_angular_difference(yaw_now, self.last_angle)
        self.cummulative_rotation += delta_yaw
        self.last_angle = yaw_now
        return self.cummulative_rotation

    def measure_angle_error(
        self,
        basket_center: Union[Sequence[float], np.ndarray],
    ) -> float:
        """
        Calculate the signed angle between vectors OA and OB.

        Inputs:
            basket_center: basket center point as (x, y) in image coordinates
        Returns:
            Signed angle in degrees in range [-180, 180]
        """
        assert self.image_size is not None

        p1 = np.array(basket_center, dtype=float)
        p2 = np.array([self.image_size[0] / 2, 0], dtype=float)
        o = np.array([self.image_size[0] / 2, self.image_size[1] * 3 / 4], dtype=float)

        oa_vec = p1 - o
        ob_vec = p2 - o

        dot = np.dot(oa_vec, ob_vec)
        det = oa_vec[0] * ob_vec[1] - oa_vec[1] * ob_vec[0]

        angle_rad = np.arctan2(det, dot)
        angle_deg = np.degrees(angle_rad)

        return float(angle_deg)

    def get_alignment_threshold(
        self, basket_dis: float, min_v: float = 0.5, max_v: float = 1.0
    ) -> float:
        """
        Calculate distance-dependent alignment threshold.
        Closer baskets need larger tolerances due to bigger angular size.

        Uses exponential decay: threshold = 4.5 * exp(-0.9 * distance) + 0.15

        Distance → Threshold:
        - 1.0m → 3.00°
        - 2.6m → 0.60°
        - 4.2m → 0.24°

        Inputs:
            basket_pos: (x, y) in meters relative to robot base center
        Returns:
            Alignment threshold in degrees
        """

        # Exponential decay function: stricter tolerance at longer distances
        threshold = 4.5 * math.exp(-0.9 * basket_dis) + 0.15

        # Clamp to reasonable bounds
        return max(min_v, min(max_v, threshold))

    def motor_percent_from_basket(self, basket_dis: float, offset_val: int = 0) -> float:
        """
        Estimate motor percent needed to throw ball into basket based on distance.
        Inputs:
            basket_dis: distance to basket in meters
            offset_val: offset to add to motor percent (can be negative)
        Returns:
            Estimated motor percent (0-100)
        """
        # Experimental data points: (distance in meters, motor percent)
        data_points = [
            (0.935, 42.0),
            (1.28, 41.0),
            (1.43, 43.0),
            (1.55, 43.0),
            (2.30, 50.0),
            (2.63, 50.0),
            (2.90, 54.0),
            (3.00, 54.0),
            (3.20, 56.0),
            (3.32, 56.0),
            (3.40, 59.0),
            (4.00, 63.0),
            (4.20, 65.0),
        ]

        # If distance is below minimum measured, extrapolate using first two points
        if basket_dis <= data_points[0][0]:
            d1, p1 = data_points[0]
            d2, p2 = data_points[1]
            slope = (p2 - p1) / (d2 - d1)
            percent = p1 + slope * (basket_dis - d1) + offset_val
            return max(0, min(100, percent))

        # If distance is above maximum measured, extrapolate using last two points
        if basket_dis >= data_points[-1][0]:
            d1, p1 = data_points[-2]
            d2, p2 = data_points[-1]
            slope = (p2 - p1) / (d2 - d1)
            percent = p2 + slope * (basket_dis - d2) + offset_val
            return max(0, min(100, percent))

        # Linear interpolation between two nearest data points
        for i in range(len(data_points) - 1):
            d1, p1 = data_points[i]
            d2, p2 = data_points[i + 1]

            if d1 <= basket_dis <= d2:
                # Linear interpolation: p = p1 + (p2 - p1) * (d - d1) / (d2 - d1)
                percent = p1 + (p2 - p1) * (basket_dis - d1) / (d2 - d1) + offset_val
                return percent

        # Fallback (should never reach here)
        return 50.0

    def normalize_velocity(self, vx: float, vy: float, max_speed: float) -> Tuple[float, float]:
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


def main() -> None:
    rclpy.init()
    node = GameLogicController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Stop referee client before destroying node
        if hasattr(node, "referee_client"):
            node.referee_client.stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
