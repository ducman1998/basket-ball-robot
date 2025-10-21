import math
from time import time
from typing import List, Optional, Tuple, Union

import modern_robotics as mr
import numpy as np
import rclpy
from basket_robot_nodes.utils.constant_utils import BASE_FRAME_ID
from basket_robot_nodes.utils.image_info import GreenBall, ImageInfo
from basket_robot_nodes.utils.ros_utils import (
    log_initialized_parameters,
    parse_log_level,
)
from nav_msgs.msg import Odometry
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
from rclpy.clock import Clock
from rclpy.node import Node
from rclpy.qos import QoSProfile
from shared_interfaces.msg import TwistStamped  # a custom message with thrower_percent
from std_msgs.msg import String

SAMPLING_RATE = 30  # Hz
FT_DETECTED_SEARCHING = 5  # frames
FT_UNDETECTED_REACHING = 5  # frames
FT_DETECTED_ENTERING = 15  # frames
FT_UNDETECTED_IDLE = 60  # frames
TOTAL_ROT_DEGREE_THRESHOLD = 270.0  # seconds


class GameState:
    """Enum-like class for game states."""

    INIT: int = 0
    SEARCHING_BALL: int = 1
    REACHING_BALL: int = 2
    ENTERING_COURT_CENTER: int = 3
    IDLE: int = 4


class RefFrame:
    ROBOT_BASE: int = 0
    ODOMETRY: int = 1


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
            Odometry, "/odom", self.odom_callback, QoSProfile(depth=3)
        )
        self.image_info_sub = self.create_subscription(
            String, "/image/info", self.image_info_callback, QoSProfile(depth=3)
        )
        self.mainboard_controller_pub = self.create_publisher(
            TwistStamped, "cmd_vel", QoSProfile(depth=3)
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
        # SEARCHING_BALL: look for balls
        self.last_angle: float = 0.0  # starting angle for searching
        self.cummulative_rotation: float = 0.0  # track rotation during searching
        self.frame_count_search_state: int = 0
        # REACHING_BALL: move towards the closest ball
        self.prev_vx_error: float = 0.0
        self.prev_vy_error: float = 0.0
        self.prev_wz_error: float = 0.0
        self.frame_count_reach_state: int = 0
        # ENTERING_COURT_CENTER: move towards court center
        self.court_center: Optional[Tuple[float, float]] = None
        self.court_area: float = 0.0
        self.best_court_center: Optional[Tuple[float, float]] = None
        self.best_court_area: float = 0.0
        self.frame_count_enter_state: int = 0
        # IDLE: stop
        # count of consecutive frames with ball detected in IDLE state
        self.frame_count_idle_state: int = 0
        # reusable control msg
        self.control_msg = TwistStamped()

    def _declare_node_parameter(self) -> None:
        """Declare parameters with descriptors."""
        float_descriptor = ParameterDescriptor(
            type=ParameterType.PARAMETER_DOUBLE, description="A floating point parameter."
        )
        bool_descriptor = ParameterDescriptor(
            type=ParameterType.PARAMETER_BOOL, description="A boolean parameter."
        )
        str_descriptor = ParameterDescriptor(
            type=ParameterType.PARAMETER_STRING, description="A string parameter."
        )
        self.declare_parameter("max_rot_speed", descriptor=float_descriptor)
        self.declare_parameter("max_xy_speed", descriptor=float_descriptor)
        self.declare_parameter("search_ball_rot_speed", descriptor=float_descriptor)
        self.declare_parameter("kp_xy", descriptor=float_descriptor)
        self.declare_parameter("kd_xy", descriptor=float_descriptor)
        self.declare_parameter("kp_rot", descriptor=float_descriptor)
        self.declare_parameter("kd_rot", descriptor=float_descriptor)
        self.declare_parameter("norm_xy_speed", descriptor=bool_descriptor)
        self.declare_parameter("log_level", descriptor=str_descriptor)

    def _read_node_parameters(self) -> None:
        """Read parameters into class variables."""
        self.max_rot = self.get_parameter("max_rot_speed").get_parameter_value().double_value
        self.max_xy = self.get_parameter("max_xy_speed").get_parameter_value().double_value
        self.search_rot = (
            self.get_parameter("search_ball_rot_speed").get_parameter_value().double_value
        )
        self.kp_xy = self.get_parameter("kp_xy").get_parameter_value().double_value
        self.kd_xy = self.get_parameter("kd_xy").get_parameter_value().double_value
        self.kp_rot = self.get_parameter("kp_rot").get_parameter_value().double_value
        self.kd_rot = self.get_parameter("kd_rot").get_parameter_value().double_value
        self.norm_xy_speed = self.get_parameter("norm_xy_speed").get_parameter_value().bool_value
        # set logging level
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
        self.court_center = self.image_info_msg.court_center
        if self.image_info_msg.court_area is not None:
            self.court_area = self.image_info_msg.court_area
        self.last_image_info_time = time()

    def game_logic_loop(self) -> None:
        """Main game logic loop, called periodically by a timer."""
        # Implement game logic here
        # For example, decide on robot movement based on state and sensor data
        start_time = time()
        match self.cur_state:
            case GameState.INIT:
                self.handle_init_state()

            case GameState.SEARCHING_BALL:
                self.handle_searching_ball_state()

            case GameState.REACHING_BALL:
                self.handle_reaching_ball_state()

            case GameState.ENTERING_COURT_CENTER:
                self.handle_entering_court_center_state()

            case GameState.IDLE:
                self.handle_idle_state()
            case _:
                raise RuntimeError("Unknown game state!")

        end_time = time()
        self.get_logger().info(f"Game logic loop took {end_time - start_time:.4f} seconds.")

    # state handlers
    def handle_init_state(self) -> None:
        self.get_logger().info("Game State: INIT")
        for _ in range(3):
            self.move_robot(0.0, 0.0, 0.0, 0)  # stop robot

        if self.odom_msg:
            yaw = self.yaw_from_odom(self.odom_msg)  # yaw is always not None here
            self.last_angle = yaw
            self.cummulative_rotation = 0.0
            self.get_logger().info(f"Last angle set to {self.last_angle:.2f} degrees.")
            # transition to searching state
            self.cur_state = GameState.SEARCHING_BALL
            self.get_logger().info("Transitioning to SEARCHING_BALL state.")
        return None

    def handle_searching_ball_state(self) -> None:
        self.get_logger().info("Game State: SEARCHING_BALL")
        # Here, you would check image_info_msg for detected balls
        # If a ball is detected, transition to REACHING_BALL state
        if self.image_info_msg and len(self.image_info_msg.balls) > 0:
            self.frame_count_search_state += 1
            if self.frame_count_search_state >= FT_DETECTED_SEARCHING:
                self.move_robot(0.0, 0.0, 0.0, 0)  # stop rotation
                self.cur_state = GameState.REACHING_BALL
                self.get_logger().info("Ball detected! Transitioning to REACHING_BALL state.")
                return None
        else:
            self.frame_count_search_state = 0  # reset count if no ball detected

        # If no ball detected after a full rotation, transition to IDLE state
        yaw_now = self.yaw_from_odom(self.odom_msg)
        delta_yaw = self.shortest_angular_difference(yaw_now, self.last_angle)  # TODO: check
        self.cummulative_rotation += delta_yaw
        self.last_angle = yaw_now
        if abs(self.cummulative_rotation) >= TOTAL_ROT_DEGREE_THRESHOLD:
            self.move_robot(0.0, 0.0, 0.0, 0)  # stop rotation
            if self.best_court_center is not None:
                self.cur_state = GameState.ENTERING_COURT_CENTER
                self.get_logger().info(
                    "No ball after full rotation. Transitioning to ENTERING_COURT_CENTER state."
                )
                return None
            else:
                self.cur_state = GameState.IDLE
                self.get_logger().info(
                    "No ball found after full rotation. Transitioning to IDLE state."
                )
                return None

        # rotate the robot to find balls
        self.move_robot(0.0, 0.0, self.search_rot, 0)
        self.get_logger().info(f"Court center: {self.court_center}, area: {self.court_area}")
        # update best court center if current area is larger
        if (
            self.odom_msg is not None
            and self.court_center is not None
            and self.court_area > self.best_court_area
        ):
            t_robot_to_odom = np.eye(4)
            quat = self.odom_msg.pose.pose.orientation
            yaw = 2.0 * math.atan2(quat.z, quat.w)
            t_robot_to_odom[0, 3] = self.odom_msg.pose.pose.position.x
            t_robot_to_odom[1, 3] = self.odom_msg.pose.pose.position.y
            t_robot_to_odom[0, :2] = [np.cos(yaw), -np.sin(yaw)]
            t_robot_to_odom[1, :2] = [np.sin(yaw), np.cos(yaw)]
            # target position in robot base-footprint frame
            court_center_robot_frame = np.array([0.0, 1.5, 0.0, 1.0])
            target_pos_odom_frame = t_robot_to_odom @ court_center_robot_frame
            self.best_court_center = target_pos_odom_frame[:2].tolist()
            self.best_court_area = self.court_area
        self.get_logger().info(
            f"Searching... Current yaw: {yaw_now:.2f} degrees, "
            f"Cumulative rotation: {self.cummulative_rotation:.2f} degrees."
        )
        return None

    def handle_entering_court_center_state(self) -> None:
        self.get_logger().info("Game State: FOLLOW_COURT_CENTER")
        if self.image_info_msg and len(self.image_info_msg.balls) > 0:
            self.frame_count_enter_state += 1
            if self.frame_count_enter_state >= FT_DETECTED_ENTERING:
                self.get_logger().info("Ball detected! Transitioning to REACHING_BALL state.")
                self.cur_state = GameState.REACHING_BALL
                return None
        else:
            self.frame_count_enter_state = 0  # reset count if ball detected

        if self.best_court_center is not None:
            vx, vy, wz = self.compute_control_signals(self.best_court_center, RefFrame.ODOMETRY)
            self.move_robot(vx, vy, wz, 0, normalize=True)
            self.get_logger().info(
                f"Moving towards court center: vx={vx:.2f}, vy={vy:.2f}, wz={wz:.2f}, "
                f"pos=({self.best_court_center[0]:.1f}, {self.best_court_center[1]:.1f})mm"
            )
            self.get_logger().info(
                f"Court center position: {self.best_court_center}, area: {self.best_court_area}"
            )
            return None
        else:
            self.move_robot(0.0, 0.0, 0.0, 0)  # stop robot
            self.cur_state = GameState.IDLE
            self.get_logger().info(
                "No court center information available. Transitioning to IDLE state."
            )
            return None

    def handle_reaching_ball_state(self) -> None:
        self.get_logger().info("Game State: REACHING_BALL")
        if self.image_info_msg and len(self.image_info_msg.balls) > 0:
            self.frame_count_reach_state = 0
            # move towards the closest ball
            closet_ball: GreenBall = min(
                self.image_info_msg.balls,
                key=lambda b: math.hypot(*b.position_2d) if b.position_2d else float("inf"),
            )
            stop_condition = self.check_stop_condition(closet_ball)
            if stop_condition:
                self.get_logger().info("Reached the ball! Stopping and returning to IDLE state.")
                self.cur_state = GameState.IDLE
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
            self.frame_count_reach_state += 1
            if self.frame_count_reach_state >= FT_UNDETECTED_REACHING:
                self.get_logger().info("Lost sight of the ball. Returning to SEARCHING_BALL state.")
                self.cur_state = GameState.SEARCHING_BALL
                self.reset_to_search_state()
                return None

    def handle_idle_state(self) -> None:
        self.get_logger().info("Game State: IDLE")
        self.move_robot(0.0, 0.0, 0.0, 0)  # stop robot
        # transition to searching state if the ball is disappeared
        if self.image_info_msg and len(self.image_info_msg.balls) > 0:
            self.frame_count_idle_state = 0  # reset count if ball detected
            # move towards the closest ball
            closet_ball: GreenBall = min(
                self.image_info_msg.balls,
                key=lambda b: math.hypot(*b.position_2d) if b.position_2d else float("inf"),
            )
            stop_condition = self.check_stop_condition(closet_ball)
            if not stop_condition:
                self.cur_state = GameState.REACHING_BALL
                self.reset_to_search_state()
                self.get_logger().info(
                    "Ball movement detected! Transitioning to REACHING_BALL state."
                )
                return None
        else:
            self.frame_count_idle_state += 1
            if self.frame_count_idle_state >= FT_UNDETECTED_IDLE:
                self.cur_state = GameState.SEARCHING_BALL
                self.reset_to_search_state()
                self.get_logger().info(
                    "No ball detected for a while in IDLE state. "
                    + "Transitioning to SEARCHING_BALL state."
                )
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

    def compute_control_signals(
        self,
        target_pos: Union[List[float], Tuple[float, float]],
        ref_frame: int = RefFrame.ROBOT_BASE,
    ) -> Tuple[float, float, float]:
        """
        Compute control signals (vx, vy, wz) to reach the target position.
        target_pos: (x, y) position of the target in mm in robot base frame.
        Returns: (vx, vy, wz) control signals.
        """
        x_desired = np.eye(4)
        if ref_frame == RefFrame.ROBOT_BASE:
            r_target_pos = [target_pos[0] / 1000.0, target_pos[1] / 1000.0]
        else:
            # calculate transformation from robot base to odometry frame
            if self.odom_msg is None:
                raise RuntimeError("No odometry message available for transformation.")

            t_robot_to_odom = np.eye(4)
            quat = self.odom_msg.pose.pose.orientation
            yaw = 2.0 * math.atan2(quat.z, quat.w)
            t_robot_to_odom[0, 3] = self.odom_msg.pose.pose.position.x
            t_robot_to_odom[1, 3] = self.odom_msg.pose.pose.position.y
            t_robot_to_odom[0, :2] = [np.cos(yaw), -np.sin(yaw)]
            t_robot_to_odom[1, :2] = [np.sin(yaw), np.cos(yaw)]
            # target position in robot base-footprint frame
            target_pos_robot_frame = np.linalg.pinv(t_robot_to_odom) @ np.array(
                [target_pos[0] / 1000.0, target_pos[1] / 1000.0, 0.0, 1.0]
            )
            r_target_pos = target_pos_robot_frame[:2].tolist()
            self.get_logger().info(
                "Transformed target position to robot frame: "
                + f"({r_target_pos[0]:.1f}, {r_target_pos[1]:.1f})m"
            )

        heading_error = -math.atan2(r_target_pos[0], r_target_pos[1])
        x_desired[0, :] = [
            np.cos(heading_error),
            -np.sin(heading_error),
            0.0,
            float(r_target_pos[0]),  # target x in m
        ]
        x_desired[1, :] = [
            np.sin(heading_error),
            np.cos(heading_error),
            0.0,
            r_target_pos[1] - 0.25,  # stop 250mm before the ball
        ]
        x_desired[2, :] = [0.0, 0.0, 1.0, 0.0]
        x_desired[3, :] = [0.0, 0.0, 0.0, 1.0]
        xe_log = mr.MatrixLog6(x_desired)
        xe_vec = mr.se3ToVec(xe_log)
        vx_error = xe_vec[3]  # velocity error in x
        vy_error = xe_vec[4]  # velocity error in y
        wz_error = xe_vec[2]  # angular velocity error in z

        # PD control
        vx = self.kp_xy * vx_error + self.kd_xy * (vx_error - self.prev_vx_error) * SAMPLING_RATE
        vy = self.kp_xy * vy_error + self.kd_xy * (vy_error - self.prev_vy_error) * SAMPLING_RATE
        wz = self.kp_rot * wz_error + self.kd_rot * (wz_error - self.prev_wz_error) * SAMPLING_RATE

        self.prev_vx_error = vx_error
        self.prev_vy_error = vy_error
        self.prev_wz_error = wz_error
        return vx, vy, wz

    def yaw_from_odom(self, odom_msg: Odometry) -> float:
        """Extract yaw angle in degrees from odometry message."""
        if not odom_msg:
            raise ValueError("No odometry message available.")

        quat = odom_msg.pose.pose.orientation
        yaw_rad = 2.0 * math.atan2(quat.z, quat.w)
        yaw_deg = math.degrees(yaw_rad)
        return yaw_deg

    def reset_to_search_state(self) -> None:
        """Reset variables to prepare for searching state."""
        self.last_angle = self.yaw_from_odom(self.odom_msg)
        self.cummulative_rotation = 0.0

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
        ball_pos = closest_ball.position_2d
        # if close enough to the ball, stop
        # 350mm distance threshold (robot center to the ball)
        distance_check = math.hypot(ball_pos[0], ball_pos[1]) <= 250.0
        angle_check = abs(math.atan2(ball_pos[0], ball_pos[1])) <= math.radians(3)  # 3 degrees
        if distance_check and angle_check:
            self.get_logger().info(
                "Reached the ball! Stopping and returning to SEARCHING_BALL state."
            )
        return distance_check and angle_check


def main() -> None:
    rclpy.init()
    node = GameLogicController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
