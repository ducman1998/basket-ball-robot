import math
from time import time
from typing import List, Optional, Sequence, Tuple, Union

import modern_robotics as mr
import numpy as np
import rclpy
from basket_robot_nodes.utils.constants import BASE_FRAME_ID, QOS_DEPTH
from basket_robot_nodes.utils.image_info import GreenBall, ImageInfo
from basket_robot_nodes.utils.number_utils import FrameStabilityCounter
from basket_robot_nodes.utils.ros_utils import (
    float_array_descriptor,
    float_descriptor,
    log_initialized_parameters,
    parse_log_level,
    str_descriptor,
)
from nav_msgs.msg import Odometry
from rclpy.clock import Clock
from rclpy.node import Node
from rclpy.qos import QoSProfile
from scipy.spatial.transform import Rotation as R
from shared_interfaces.msg import TwistStamped  # a custom message with thrower_percent
from std_msgs.msg import String

OPPONENT_BASKET_COLOR = "magenta"  # color of the opponent's basket
SAMPLING_RATE = 60  # Hz
FT_DETECTED_SEARCH_BALL = 6  # frames
FT_UNDETECTED_REACH_BALL = 15  # frames
FT_UNDETECTED_ALIGN_BALL = 20  # frames
FT_DETECTED_APPROACH_BASKET = 40  # frames
FT_UNDETECTED_IDLE = 120  # frames
FT_DETECTED_STABLE_ALIGN_BALL = 6  # frames
TOTAL_ROT_DEGREE_THRESHOLD = 270.0  # seconds
# thrower motor parameters
KV = 1200.0  # RPM/V
V_BATT = 5.5  # volts
RPM_MAX = KV * V_BATT
RPM_MIN = 0.1 * RPM_MAX  # approx 5% of max
THROWER_WHEEL_RADIUS = 0.014  # meters
THROWER_ANGLE_DEG = 55.0  # degrees
BASKET_HEIGHT = 0.55  # meters


class GameState:
    """Enum-like class for game states."""

    INIT: int = 0
    SEARCH_BALL: int = 1
    REACH_BALL: int = 2
    ALIGN_TO_BASKET: int = 3
    THROW_BALL: int = 4
    APROACH_BASKET: int = 5
    IDLE: int = 6


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

        # init subscribers and publishers
        self.odom_sub = self.create_subscription(
            Odometry, "/odom", self.odom_callback, QoSProfile(depth=QOS_DEPTH)
        )
        self.image_info_sub = self.create_subscription(
            String, "/image/info", self.image_info_callback, QoSProfile(depth=QOS_DEPTH)
        )
        self.mainboard_controller_pub = self.create_publisher(
            TwistStamped, "cmd_vel", QoSProfile(depth=QOS_DEPTH)
        )
        # game timer to run game logic at 20Hz
        self.game_timer = self.create_timer(1 / SAMPLING_RATE, self.game_logic_loop)

        self.cur_state = GameState.INIT
        self.timestamp = time()

        # internal variables for robot control
        self.odom_msg: Optional[Odometry] = None
        self.last_odom_time: float = 0.0

        self.image_info_msg: Optional[ImageInfo] = None
        self.last_image_info_time: float = 0.0

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
        # APPROACH_BASKET: move towards court center
        self.fathest_basket_pos: Optional[Tuple[float, float]] = None
        self.fathest_basket_dis: float = 0.0
        # THROW_BALL: throw the ball
        self.throw_start_pos: Optional[Tuple[float, float]] = None
        # IDLE: stop
        # TODO: add other state variables as needed

        # counters for frame stability, count consecutive frames of detection/loss
        self.fcounter_search_state = FrameStabilityCounter(FT_DETECTED_SEARCH_BALL)
        self.fcounter_reach_state = FrameStabilityCounter(FT_UNDETECTED_REACH_BALL)
        self.fcounter_align_state = FrameStabilityCounter(FT_UNDETECTED_ALIGN_BALL)
        self.fcounter_align_stable_state = FrameStabilityCounter(FT_DETECTED_STABLE_ALIGN_BALL)
        self.fcounter_approach_state = FrameStabilityCounter(FT_DETECTED_APPROACH_BASKET)
        self.fcounter_idle_state = FrameStabilityCounter(FT_UNDETECTED_IDLE)

        # reusable control msg
        self.control_msg = TwistStamped()

    def _declare_node_parameter(self) -> None:
        """Declare parameters with descriptors."""
        self.declare_parameter("max_rot_speed", descriptor=float_descriptor)
        self.declare_parameter("max_xy_speed", descriptor=float_descriptor)
        self.declare_parameter("search_ball_rot_speed", descriptor=float_descriptor)
        self.declare_parameter("pid_linear_common", descriptor=float_array_descriptor)
        self.declare_parameter("pid_angular_common", descriptor=float_array_descriptor)
        self.declare_parameter("pid_linear_align_basket", descriptor=float_array_descriptor)
        self.declare_parameter("pid_angular_align_basket", descriptor=float_array_descriptor)
        self.declare_parameter("log_level", descriptor=str_descriptor)

    def _read_node_parameters(self) -> None:
        """Read parameters into class variables."""
        self.max_rot = self.get_parameter("max_rot_speed").get_parameter_value().double_value
        self.max_xy = self.get_parameter("max_xy_speed").get_parameter_value().double_value
        self.search_rot_speed = (
            self.get_parameter("search_ball_rot_speed").get_parameter_value().double_value
        )
        self.pid_linear_common = (
            self.get_parameter("pid_linear_common").get_parameter_value().double_array_value
        )
        self.pid_angular_common = (
            self.get_parameter("pid_angular_common").get_parameter_value().double_array_value
        )
        self.pid_linear_align_basket = (
            self.get_parameter("pid_linear_align_basket").get_parameter_value().double_array_value
        )
        self.pid_angular_align_basket = (
            self.get_parameter("pid_angular_align_basket").get_parameter_value().double_array_value
        )
        # validate PID parameters
        if len(self.pid_linear_common) != 3 or len(self.pid_angular_common) != 3:
            raise ValueError("PID common parameters must be lists of three floats: [kp, ki, kd].")
        if len(self.pid_linear_align_basket) != 3 or len(self.pid_angular_align_basket) != 3:
            raise ValueError("PID align parameters must be lists of three floats: [kp, ki, kd].")

        # read and set logging level
        log_level = self.get_parameter("log_level").get_parameter_value().string_value
        self.get_logger().set_level(parse_log_level(log_level))
        self.get_logger().info(f"Set node {self.get_name()} log level to {log_level}.")

    def odom_callback(self, msg: Odometry) -> None:
        """Handle incoming odometry messages."""
        # Process odometry data as needed
        self.odom_msg = msg
        self.last_odom_time = time()

    def image_info_callback(self, msg: String) -> None:
        """Handle incoming image info messages."""
        # Process image info data as needed
        self.image_info_msg = ImageInfo.from_json(msg.data)
        self.last_image_info_time = time()

    def game_logic_loop(self) -> None:
        """Main game logic loop, called periodically by a timer."""
        start_time = time()
        self.print_current_state()

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
        assert self.image_info_msg is not None
        assert self.odom_msg is not None

        # if a ball is detected, transition to REACH_BALL state
        if self.fcounter_search_state.update(self.is_ball_in_view()):
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
                and self.image_info_msg.basket.color == OPPONENT_BASKET_COLOR
            ):
                offset_angle = self.angle_at_point(
                    closet_ball.position_2d, self.image_info_msg.basket.position_2d
                )
                self.get_logger().info(f"Offset angle to basket: {offset_angle:.2f} degrees.")
                vx, vy, wz = self.compute_control_signals(
                    closet_ball.position_2d, angle_offset=offset_angle, look_ahead_dis=0.2
                )
                self.move_robot(vx, vy, wz, 0, normalize=True)
                if self.fcounter_align_stable_state.update(abs(offset_angle) < 1.5):
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
        vx, vy, wz = 0.0, 0.4, 0.0
        if self.image_info_msg.basket is None or self.image_info_msg.basket.position_2d is None:
            self.move_robot(vx, vy, wz, 60, normalize=False)
            self.get_logger().info(
                "No basket detected! Throwing ball with default motor percent=60."
            )
        else:
            basket_pos = self.image_info_msg.basket.position_2d  # in mm
            motor_percent = self.motor_percent_from_basket(
                [basket_pos[0] / 1000, basket_pos[1] / 1000, BASKET_HEIGHT]
            )
            self.move_robot(vx, vy, wz, motor_percent, normalize=False)
            self.get_logger().info(
                f"Throwing ball towards basket at pos=({basket_pos[0]:.1f}, {basket_pos[1]:.1f})mm "
                f"with motor percent={motor_percent}."
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
        self, vx: float, vy: float, wz: float, thrower_percent: int, normalize: bool = False
    ) -> None:
        """Send velocity commands to the robot. vx, vy in m/s, wz in rad/s."""
        if normalize:
            if np.linalg.norm([vx, vy]) > self.max_xy:
                scale = self.max_xy / np.linalg.norm([vx, vy])
                vx *= scale
                vy *= scale
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
        self.control_msg.thrower_percent = int(thrower_percent)
        self.mainboard_controller_pub.publish(self.control_msg)

    def stop_robot(self) -> None:
        """Stop the robot by sending zero velocity commands."""
        self.move_robot(0.0, 0.0, 0.0, 0)

    def rotate_robot(self, wz: float) -> None:
        """Rotate the robot at a given angular velocity wz (rad/s)."""
        self.move_robot(0.0, 0.0, wz, 0)

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

        if pre_state != new_state:
            self.prev_vx_error = 0.0
            self.prev_vy_error = 0.0
            self.prev_wz_error = 0.0
            self.cumm_vx_error = 0.0
            self.cumm_vy_error = 0.0
            self.cumm_wz_error = 0.0
        self.get_logger().info(f"Transitioning from state {pre_state} to {new_state}.")

    def print_current_state(self) -> None:
        """Log the current state and relevant information."""
        state_name = next((k for k, v in vars(GameState).items() if v == self.cur_state), "UNKNOWN")
        self.get_logger().info(f"Current State: {state_name}")

    def compute_cumulative_rotation(self) -> float:
        yaw_now = self.yaw_from_odom(self.odom_msg)
        delta_yaw = self.shortest_angular_difference(yaw_now, self.last_angle)
        self.cummulative_rotation += delta_yaw
        self.last_angle = yaw_now
        return self.cummulative_rotation

    def angle_at_point(
        self, p1: Union[Sequence[float], np.ndarray], p2: Union[Sequence[float], np.ndarray]
    ) -> float:
        """
        Calculate the signed angle at point A between vectors OP1 and P1P2.

        Inputs:
            p1: Point P1 as (x, y)
            p2: Point P2 as (x, y)
        Returns:
            Signed angle in degrees in range [-180, 180]
        """
        p1 = np.array(p1, dtype=float)
        p2 = np.array(p2, dtype=float)
        origin = np.array([0, 0], dtype=float)

        oa_vec = p1 - origin
        ab_vec = p2 - p1

        dot = np.dot(oa_vec, ab_vec)
        det = oa_vec[0] * ab_vec[1] - oa_vec[1] * ab_vec[0]

        angle_rad = np.arctan2(det, dot)
        angle_deg = np.degrees(angle_rad)

        return float(angle_deg)

    def motor_percent_from_basket(
        self,
        basket_pos: Union[List[float], Tuple[float, float, float]],
        h0: float = 0.07,
        g: float = 9.81,
    ) -> int:
        """
        Compute motor percent for throwing wheel to hit a basket.

        Inputs:
            basket_pos: (x, y, z) in meters relative to launcher
            throw_angle_deg: throwing angle from horizontal
            r_wheel: radius of the throwing wheel in meters
            h0: launcher height
            g: gravity
        Returns:
            Motor speed percent [0-100] or None if impossible
        """
        x, y, z = basket_pos
        theta = math.radians(THROWER_ANGLE_DEG)
        y -= 0.010  # adjust for robot center to launcher y-offset
        d = math.sqrt(x**2 + y**2)

        denominator = d * math.tan(theta) - (z - h0)
        if denominator <= 0:
            return 50  # throw impossible at this angle

        v_ball = math.sqrt((g * d**2) / (2 * math.cos(theta) ** 2 * denominator))

        omega_wheel = v_ball / THROWER_WHEEL_RADIUS  # rad/s
        rpm = omega_wheel * 60 / (2 * math.pi)

        percent = (rpm - RPM_MIN) / (RPM_MAX - RPM_MIN) * 100
        percent = max(0.0, min(100.0, percent))

        return int(percent)


def main() -> None:
    rclpy.init()
    node = GameLogicController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
