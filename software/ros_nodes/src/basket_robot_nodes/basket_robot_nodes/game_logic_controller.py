import math
from time import time
from typing import Optional

import modern_robotics as mr
import numpy as np
import rclpy
from basket_robot_nodes.utils.image_info import GreenBall, ImageInfo
from basket_robot_nodes.utils.ros_utils import log_initialized_parameters
from nav_msgs.msg import Odometry
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
from rclpy.clock import Clock
from rclpy.node import Node
from rclpy.qos import QoSProfile
from shared_interfaces.msg import TwistStamped  # a custom message with thrower_percent
from std_msgs.msg import String

SAMPLING_RATE = 20  # Hz


class GameState:
    """Enum-like class for game states."""

    INIT: int = 0
    SEARCHING_BALL: int = 1
    REACHING_BALL: int = 2
    END: int = 3


class GameLogicController(Node):
    def __init__(self) -> None:
        # Initialize the Game Logic Controller node
        super().__init__("game_logic_controller")
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
        self.start_angle: Optional[float] = None  # starting angle for searching
        self.cummulative_rotation: float = 0.0  # track rotation during searching

        # REACHING_BALL: move towards the closest ball
        self.prev_vx_error: float = 0.0
        self.prev_vy_error: float = 0.0
        self.prev_wz_error: float = 0.0
        # END: stop

    def _declare_node_parameter(self) -> None:
        """Declare parameters with descriptors."""
        float_descriptor = ParameterDescriptor(
            type=ParameterType.PARAMETER_DOUBLE, description="A floating point parameter."
        )
        self.declare_parameter("max_rot_speed", descriptor=float_descriptor)
        self.declare_parameter("max_xy_speed", descriptor=float_descriptor)
        self.declare_parameter("search_ball_rot_speed", descriptor=float_descriptor)
        self.declare_parameter("kp_xy", descriptor=float_descriptor)
        self.declare_parameter("kd_xy", descriptor=float_descriptor)
        self.declare_parameter("kp_rot", descriptor=float_descriptor)
        self.declare_parameter("kd_rot", descriptor=float_descriptor)

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
        # Implement game logic here
        # For example, decide on robot movement based on state and sensor data
        start_time = time()
        if self.cur_state == GameState.INIT:
            self.handle_init_state()

        elif self.cur_state == GameState.SEARCHING_BALL:
            self.handle_searching_ball_state()

        elif self.cur_state == GameState.REACHING_BALL:
            self.handle_reaching_ball_state()

        else:  # GameState.END
            self.handle_end_state()

        vx, vy, wz, thrower_percent = 0.0, 0.0, 0.0, 0  # Example values
        self.move_robot(vx, vy, wz, thrower_percent)
        end_time = time()
        self.get_logger().info(f"Game logic loop took {end_time - start_time:.4f} seconds.")

    # state handlers
    def handle_init_state(self) -> None:
        self.get_logger().info("Game State: INIT")
        for _ in range(3):
            self.move_robot(0.0, 0.0, 0.0, 0)  # stop robot

        if self.odom_msg:
            yaw = self.yaw_from_odom(self.odom_msg)  # yaw is always not None here
            self.start_angle = yaw
            self.cummulative_rotation = 0.0
            self.get_logger().info(f"Starting angle set to {self.start_angle:.2f} degrees.")
            # transition to searching state
            self.cur_state = GameState.SEARCHING_BALL
            self.get_logger().info("Transitioning to SEARCHING_BALL state.")

    def handle_searching_ball_state(self) -> None:
        self.get_logger().info("Game State: SEARCHING_BALL")
        self.move_robot(0.0, 0.0, self.search_rot, 0)  # rotate in place (e.g., 0.5 rad/s)
        # Here, you would check image_info_msg for detected balls
        # If a ball is detected, transition to REACHING_BALL state
        if self.image_info_msg and len(self.image_info_msg.detected_balls) > 0:
            self.move_robot(0.0, 0.0, 0.0, 0)  # stop rotation
            self.cur_state = GameState.REACHING_BALL
            self.get_logger().info("Ball detected! Transitioning to REACHING_BALL state.")
            return None

        # If no ball detected after a full rotation, transition to END state
        yaw_now = self.yaw_from_odom(self.odom_msg)
        if self.start_angle is None:
            raise ValueError("start_angle should have been set in INIT state.")

        self.cummulative_rotation += yaw_now - self.start_angle
        if abs(self.cummulative_rotation) >= 360.0:
            self.move_robot(0.0, 0.0, 0.0, 0)  # stop rotation
            self.cur_state = GameState.END
            self.get_logger().info("No ball found after full rotation. Transitioning to END state.")

    def handle_reaching_ball_state(self) -> None:
        self.get_logger().info("Game State: REACHING_BALL")
        if self.image_info_msg and len(self.image_info_msg.detected_balls) > 0:
            # move towards the closest ball
            closet_ball: GreenBall = min(
                self.image_info_msg.detected_balls,
                key=lambda b: math.hypot(*b.position_2d) if b.position_2d else float("inf"),
            )
            ball_pos = closet_ball.position_2d
            # if close enough to the ball, stop
            if math.hypot(ball_pos[0], ball_pos[1]) <= 100.0 and math.atan2(
                ball_pos[0], ball_pos[1]
            ) <= math.radians(5):
                self.get_logger().info(
                    "Reached the ball! Stopping and returning to SEARCHING_BALL state."
                )
                self.cur_state = GameState.END
                self.reset_to_search_state()
                return None

            # transformation matrix from the ball frame to robot base_footprint frame
            x_desired = np.eye(4)
            heading_error = -math.atan2(ball_pos[0], ball_pos[1])
            x_desired[0, :3] = [
                np.cos(heading_error),
                -np.sin(heading_error),
                0.0,
                ball_pos[0],
            ]
            x_desired[1, :3] = [
                np.sin(heading_error),
                np.cos(heading_error),
                0.0,
                ball_pos[1],
            ]
            x_desired[2, :3] = [0.0, 0.0, 1.0, 0.0]
            x_desired[3, :3] = [0.0, 0.0, 0.0, 1.0]
            xe_log = mr.MatrixLog6(x_desired)
            xe_vec = mr.se3ToVec(xe_log)
            vx_error = xe_vec[3]  # velocity error in x
            vy_error = xe_vec[4]  # velocity error in y
            wz_error = xe_vec[2]  # angular velocity error in z

            # PD control
            vx = np.clip(
                self.kp_xy * vx_error
                + self.kd_xy * (vx_error - self.prev_vx_error) * SAMPLING_RATE,
                -self.max_xy,
                self.max_xy,
            )
            vy = np.clip(
                self.kp_xy * vy_error
                + self.kd_xy * (vy_error - self.prev_vy_error) * SAMPLING_RATE,
                -self.max_xy,
                self.max_xy,
            )
            wz = np.clip(
                self.kp_rot * wz_error
                + self.kd_rot * (wz_error - self.prev_wz_error) * SAMPLING_RATE,
                -self.max_rot,
                self.max_rot,
            )
            self.prev_vx_error = vx_error
            self.prev_vy_error = vy_error
            self.prev_wz_error = wz_error
            # move the robot
            self.move_robot(vx, vy, wz, 0)
            self.get_logger().info(
                f"Moving towards ball: vx={vx:.2f}, vy={vy:.2f}, wz={wz:.2f}, "
                f"pos=({ball_pos[0]:.1f}, {ball_pos[1]:.1f})mm"
            )

        else:
            self.get_logger().info("Lost sight of the ball. Returning to SEARCHING_BALL state.")
            self.reset_to_search_state()
            self.cur_state = GameState.SEARCHING_BALL
            return None

    def handle_end_state(self) -> None:
        self.get_logger().info("Game State: END")
        self.reset_to_search_state()
        self.move_robot(0.0, 0.0, 0.0, 0)  # stop robot
        exit(0)
        # Game over, do nothing or reset

    def move_robot(self, vx: float, vy: float, wz: float, thrower_percent: int) -> None:
        """Send velocity commands to the robot."""
        out = TwistStamped()
        out.header.stamp = Clock().now().to_msg()
        out.header.frame_id = "base_footprint"
        out.twist.linear.y = float(vx)
        out.twist.linear.x = float(vy)
        out.twist.linear.z = 0.0
        out.twist.angular.x = 0.0
        out.twist.angular.y = 0.0
        out.twist.angular.z = float(wz)
        out.thrower_percent = int(thrower_percent)
        self.mainboard_controller_pub.publish(out)

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
        self.start_angle = None
        self.cummulative_rotation = 0.0


def main() -> None:
    rclpy.init()
    node = GameLogicController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
