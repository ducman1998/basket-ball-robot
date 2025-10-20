import math
from time import time
from typing import Optional

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
CONSEC_DETECTED_FRAME_THRESHOLD_IN_GS = 5  # frames
CONSEC_DETECTED_FRAME_THRESHOLD_IN_IDLE = 60  # frames


class GameState:
    """Enum-like class for game states."""

    INIT: int = 0
    GO_STRAIGHT: int = 1
    SEARCHING_BALL: int = 2
    REACHING_BALL: int = 3
    IDLE: int = 4


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
        # GO_STRAIGHT: move forward for 1 meter
        self.init_pose: Optional[Odometry] = None
        self.cons_detected_frame_count_gs: int = 0  # count of consecutive frames with ball detected
        # SEARCHING_BALL: look for balls
        self.last_angle: float = 0.0  # starting angle for searching
        self.cummulative_rotation: float = 0.0  # track rotation during searching

        # REACHING_BALL: move towards the closest ball
        self.prev_vx_error: float = 0.0
        self.prev_vy_error: float = 0.0
        self.prev_wz_error: float = 0.0
        # IDLE: stop
        # count of consecutive frames with ball detected in IDLE state
        self.cons_detected_frame_count_idle: int = 0
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
        if self.init_pose is None:
            self.init_pose = msg  # store the initial pose

    def image_info_callback(self, msg: String) -> None:
        """Handle incoming image info messages."""
        # Process image info data as needed
        self.image_info_msg = ImageInfo.from_json(msg.data)
        self.last_image_info_time = time()

    def game_logic_loop(self) -> None:
        """Main game logic loop, called periodically by a timer."""
        # Implement game logic here
        # For example, decide on robot movement based on state and sensor data
        start_time = time()
        match self.cur_state:
            case GameState.INIT:
                self.handle_init_state()

            case GameState.GO_STRAIGHT:
                self.handle_go_straight_state()

            case GameState.SEARCHING_BALL:
                self.handle_searching_ball_state()

            case GameState.REACHING_BALL:
                self.handle_reaching_ball_state()

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
            self.cur_state = GameState.GO_STRAIGHT
            self.get_logger().info("Transitioning to GO_STRAIGHT state.")
        return None

    def handle_go_straight_state(self) -> None:
        self.get_logger().info("Game State: GO_STRAIGHT")
        self.move_robot(0.0, 0.25, 0.0, 0)  # move forward at 0.25 m/s
        if self.odom_msg and self.init_pose:
            dy = self.odom_msg.pose.pose.position.y - self.init_pose.pose.pose.position.y
            dx = self.odom_msg.pose.pose.position.x - self.init_pose.pose.pose.position.x
            distance_moved = np.sqrt(dy**2 + dx**2)
            self.get_logger().info(f"Distance moved: {distance_moved:.2f} meters.")
            # check for ball detection during moving forward
            if self.image_info_msg and len(self.image_info_msg.detected_balls) > 0:
                self.cons_detected_frame_count_gs += 1
                if self.cons_detected_frame_count_gs >= CONSEC_DETECTED_FRAME_THRESHOLD_IN_GS:
                    self.move_robot(0.0, 0.0, 0.0, 0)  # stop running
                    self.cur_state = GameState.REACHING_BALL
                    self.get_logger().info(
                        "Ball detected during moving forward in "
                        + f"{self.cons_detected_frame_count_gs} frames! "
                        + "Transitioning to REACHING_BALL state."
                    )
                    self.reset_to_search_state()
                    return None
            else:
                self.cons_detected_frame_count_gs = 0  # reset count if no ball detected

            if distance_moved >= 2.0:  # move forward for 1 meter
                self.cur_state = GameState.SEARCHING_BALL
                self.get_logger().info("Reached 1 meter. Transitioning to SEARCHING_BALL state.")
                return None
        return None

    def handle_searching_ball_state(self) -> None:
        self.get_logger().info("Game State: SEARCHING_BALL")
        # Here, you would check image_info_msg for detected balls
        # If a ball is detected, transition to REACHING_BALL state
        if self.image_info_msg and len(self.image_info_msg.detected_balls) > 0:
            self.move_robot(0.0, 0.0, 0.0, 0)  # stop rotation
            self.cur_state = GameState.REACHING_BALL
            self.get_logger().info("Ball detected! Transitioning to REACHING_BALL state.")
            return None

        # If no ball detected after a full rotation, transition to IDLE state
        yaw_now = self.yaw_from_odom(self.odom_msg)
        delta_yaw = self.shortest_angular_difference(yaw_now, self.last_angle)
        self.cummulative_rotation += delta_yaw
        self.last_angle = yaw_now
        if abs(self.cummulative_rotation) >= 360.0:
            self.move_robot(0.0, 0.0, 0.0, 0)  # stop rotation
            self.cur_state = GameState.IDLE
            self.get_logger().info(
                "No ball found after full rotation. Transitioning to IDLE state."
            )

        # rotate the robot to find balls
        self.move_robot(0.0, 0.0, self.search_rot, 0)
        self.get_logger().info(
            f"Searching... Current yaw: {yaw_now:.2f} degrees, "
            f"Cumulative rotation: {self.cummulative_rotation:.2f} degrees."
        )
        return None

    def handle_reaching_ball_state(self) -> None:
        self.get_logger().info("Game State: REACHING_BALL")
        if self.image_info_msg and len(self.image_info_msg.detected_balls) > 0:
            # move towards the closest ball
            closet_ball: GreenBall = min(
                self.image_info_msg.detected_balls,
                key=lambda b: math.hypot(*b.position_2d) if b.position_2d else float("inf"),
            )
            stop_condition = self.check_stop_condition(closet_ball)
            if stop_condition:
                self.get_logger().info("Reached the ball! Stopping and returning to IDLE state.")
                self.cur_state = GameState.IDLE
                self.reset_to_search_state()
                return None

            # transformation matrix from the ball frame to robot base_footprint frame
            ball_pos = closet_ball.position_2d  # in mm
            # desired target pose (4x4 matrix) in the robot base_foot
            x_desired = np.eye(4)
            heading_error = -math.atan2(ball_pos[0], ball_pos[1])
            x_desired[0, :] = [
                np.cos(heading_error),
                -np.sin(heading_error),
                0.0,
                ball_pos[0] / 1000.0,
            ]
            x_desired[1, :] = [
                np.sin(heading_error),
                np.cos(heading_error),
                0.0,
                ball_pos[1] / 1000.0 - 0.25,  # stop 250mm before the ball
            ]
            x_desired[2, :] = [0.0, 0.0, 1.0, 0.0]
            x_desired[3, :] = [0.0, 0.0, 0.0, 1.0]
            xe_log = mr.MatrixLog6(x_desired)
            xe_vec = mr.se3ToVec(xe_log)
            vx_error = xe_vec[3]  # velocity error in x
            vy_error = xe_vec[4]  # velocity error in y
            wz_error = xe_vec[2]  # angular velocity error in z

            # PD control
            vx = (
                self.kp_xy * vx_error + self.kd_xy * (vx_error - self.prev_vx_error) * SAMPLING_RATE
            )
            vy = (
                self.kp_xy * vy_error + self.kd_xy * (vy_error - self.prev_vy_error) * SAMPLING_RATE
            )
            wz = (
                self.kp_rot * wz_error
                + self.kd_rot * (wz_error - self.prev_wz_error) * SAMPLING_RATE
            )

            self.prev_vx_error = vx_error
            self.prev_vy_error = vy_error
            self.prev_wz_error = wz_error
            # move the robot towards the ball
            self.move_robot(vx, vy, wz, 0, normalize=True)
            self.get_logger().info(
                f"Moving towards ball: vx={vx:.2f}, vy={vy:.2f}, wz={wz:.2f}, "
                f"pos=({ball_pos[0]:.1f}, {ball_pos[1]:.1f})mm"
            )
            return None
        else:
            self.get_logger().info("Lost sight of the ball. Returning to SEARCHING_BALL state.")
            self.reset_to_search_state()
            self.cur_state = GameState.SEARCHING_BALL
            return None

    def handle_idle_state(self) -> None:
        self.get_logger().info("Game State: IDLE")
        self.move_robot(0.0, 0.0, 0.0, 0)  # stop robot
        # transition to searching state if the ball is disappeared
        if self.image_info_msg and len(self.image_info_msg.detected_balls) > 0:
            self.cons_detected_frame_count_idle = 0  # reset count if ball detected
            # move towards the closest ball
            closet_ball: GreenBall = min(
                self.image_info_msg.detected_balls,
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
            self.cons_detected_frame_count_idle += 1
            if self.cons_detected_frame_count_idle >= CONSEC_DETECTED_FRAME_THRESHOLD_IN_IDLE:
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
