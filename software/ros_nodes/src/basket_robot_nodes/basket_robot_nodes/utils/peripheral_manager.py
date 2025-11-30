import numpy as np
from time import time
from typing import List, Optional, Tuple, Union

from basket_robot_nodes.utils.constants import QOS_DEPTH
from basket_robot_nodes.utils.image_info import Basket, GreenBall, ImageInfo
from basket_robot_nodes.utils.game_utils import normalize_velocity
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSProfile
from std_msgs.msg import String
from shared_interfaces.msg import TwistStamped  # a custom message with thrower_percent


class PeripheralManager:
    """
    Manages sensor subscriptions and provides convenient access to sensor data.

    This class handles:
    - Odometry data subscription and storage
    - Image information subscription and parsing
    - Thread-safe access to latest sensor data
    - Timestamp tracking for sensor freshness
    """

    def __init__(self, node: Node) -> None:
        """
        Initialize the PeripheralManager.

        Args:
            node: ROS2 Node instance for creating subscriptions
        """
        self._node = node

        # Odometry data
        self._odom_msg: Optional[Odometry] = None
        self._last_odom_time: float = 0.0

        # Image information data
        self._image_info_msg: Optional[ImageInfo] = None
        self._last_image_info_time: float = 0.0
        self._image_size: Optional[Tuple[int, int]] = None

        # Create subscriptions
        self._odom_sub = node.create_subscription(
            Odometry, "/odom", self._odom_callback, QoSProfile(depth=QOS_DEPTH)
        )

        self._image_info_sub = node.create_subscription(
            String, "/image/info", self._image_info_callback, QoSProfile(depth=QOS_DEPTH)
        )

        self.mainboard_controller_pub = node.create_publisher(
            TwistStamped, "cmd_vel", QoSProfile(depth=QOS_DEPTH)
        )
        # reusable control msg
        self.control_msg = TwistStamped()
        node.get_logger().info("SensorManager initialized")

    # ==================== Callback Methods ====================

    def _odom_callback(self, msg: Odometry) -> None:
        """Handle incoming odometry messages."""
        self._odom_msg = msg
        self._last_odom_time = time()

    def _image_info_callback(self, msg: String) -> None:
        """Handle incoming image info messages."""
        self._image_info_msg = ImageInfo.from_json(msg.data)

        # Store image size on first receipt
        if self._image_size is None and self._image_info_msg.image_size is not None:
            self._image_size = self._image_info_msg.image_size

        self._last_image_info_time = time()

    # ==================== Data Availability Methods ====================

    def is_sensor_data_available(self) -> bool:
        """
        Check if both odometry and image info data are available.

        Returns:
            True if both sensor data types have been received at least once
        """
        return self._odom_msg is not None and self._image_info_msg is not None

    def is_odom_available(self) -> bool:
        """
        Check if odometry data is available.

        Returns:
            True if odometry has been received at least once
        """
        return self._odom_msg is not None

    def is_image_info_available(self) -> bool:
        """
        Check if image info data is available.

        Returns:
            True if image info has been received at least once
        """
        return self._image_info_msg is not None

    # ==================== Odometry Access Methods ====================

    def get_odom_msg(self) -> Optional[Odometry]:
        """
        Get the latest odometry message.

        Returns:
            Latest Odometry message, or None if not available
        """
        return self._odom_msg

    def get_robot_position(self) -> Optional[Tuple[float, float, float]]:
        """
        Get the robot's current position from odometry.

        Returns:
            Tuple of (x, y, z) in meters, or None if odometry not available
        """
        if self._odom_msg is None:
            return None

        pos = self._odom_msg.pose.pose.position
        return (pos.x, pos.y, pos.z)

    def get_robot_orientation(self) -> Optional[Tuple[float, float, float, float]]:
        """
        Get the robot's current orientation quaternion from odometry.

        Returns:
            Tuple of (x, y, z, w) quaternion, or None if odometry not available
        """
        if self._odom_msg is None:
            return None

        quat = self._odom_msg.pose.pose.orientation
        return (quat.x, quat.y, quat.z, quat.w)

    def get_robot_velocity(self) -> Optional[Tuple[float, float, float]]:
        """
        Get the robot's current linear velocity from odometry.

        Returns:
            Tuple of (vx, vy, vz) in m/s, or None if odometry not available
        """
        if self._odom_msg is None:
            return None

        vel = self._odom_msg.twist.twist.linear
        return (vel.x, vel.y, vel.z)

    def get_odom_timestamp(self) -> float:
        """
        Get the timestamp of the last received odometry message.

        Returns:
            Time in seconds since epoch, or 0.0 if no message received
        """
        return self._last_odom_time

    # ==================== Image Info Access Methods ====================

    def get_image_info_msg(self) -> Optional[ImageInfo]:
        """
        Get the latest image information message.

        Returns:
            Latest ImageInfo object, or None if not available
        """
        return self._image_info_msg

    def is_ball_detected(self) -> bool:
        """
        Check if any green ball is detected in the latest image info.

        Returns:
            True if at least one green ball is detected, False otherwise
        """
        if self._image_info_msg is None:
            return False
        return len(self._image_info_msg.balls) > 0

    def get_detected_balls(self) -> Union[List[GreenBall], Tuple[GreenBall, ...]]:
        """
        Get the list of detected green balls from latest image info.

        Returns:
            List of GreenBall objects, empty list if no image info available
        """
        if self._image_info_msg is None:
            return []
        return self._image_info_msg.balls

    def get_detected_basket(self) -> Optional[Basket]:
        """
        Get the detected basket from latest image info.

        Returns:
            Basket object if detected, None otherwise
        """
        if self._image_info_msg is None:
            return None
        return self._image_info_msg.basket

    def get_basket_position_2d(self) -> Optional[Tuple[float, float]]:
        """
        Get the 2D position of the detected basket in robot frame.

        Returns:
            Tuple of (x, y) in mm, or None if basket not detected or no position
        """
        if self._image_info_msg is None or self._image_info_msg.basket is None:
            return None
        return self._image_info_msg.basket.position_2d

    def get_basket_center_pixel(self) -> Optional[Tuple[int, int]]:
        """
        Get the pixel coordinates of the basket center in the image.

        Returns:
            Tuple of (x, y) pixel coordinates, or None if basket not detected
        """
        if self._image_info_msg is None or self._image_info_msg.basket is None:
            return None
        return self._image_info_msg.basket.center

    def get_basket_color(self) -> Optional[str]:
        """
        Get the color of the detected basket.

        Returns:
            Basket color string ("blue" or "magenta"), or None if not detected
        """
        if self._image_info_msg is None or self._image_info_msg.basket is None:
            return None
        return self._image_info_msg.basket.color

    def get_image_size(self) -> Optional[Tuple[int, int]]:
        """
        Get the size of the camera image.

        Returns:
            Tuple of (width, height) in pixels, or None if not available
        """
        return self._image_size

    def get_image_info_timestamp(self) -> float:
        """
        Get the timestamp of the last received image info message.

        Returns:
            Time in seconds since epoch, or 0.0 if no message received
        """
        return self._last_image_info_time

    # ============== Mainboard Control Methods ================
    def move_robot(
        self,
        vx: float,
        vy: float,
        wz: float,
        thrower_percent: float,
        max_xy_speed: float,
        max_rot_speed: float,
        normalize: bool = False,
    ) -> None:
        """Send velocity commands to the robot. vx, vy in m/s, wz in rad/s."""
        if normalize:
            vx, vy = normalize_velocity(vx, vy, max_xy_speed)
        else:
            vx = np.clip(vx, -max_xy_speed, max_xy_speed)
            vy = np.clip(vy, -max_xy_speed, max_xy_speed)

        wz = np.clip(wz, -max_rot_speed, max_rot_speed)
        thrower_percent = np.clip(thrower_percent, 0, 100)

        self.publish_velocity_command(vx, vy, wz, thrower_percent)

    def publish_velocity_command(
        self, vx: float, vy: float, wz: float, thrower_percent: float
    ) -> None:
        """
        Publish a velocity command to the mainboard controller.

        Args:
            vx: Linear velocity in x (m/s)
            vy: Linear velocity in y (m/s)
            wz: Angular velocity around z (rad/s)
            thrower_percent: Thrower speed percentage (0-100)
        """
        self.control_msg.twist.linear.x = float(vx)
        self.control_msg.twist.linear.y = float(vy)
        self.control_msg.twist.angular.z = float(wz)
        self.control_msg.thrower_percent = float(thrower_percent)

        self.mainboard_controller_pub.publish(self.control_msg)

    # ==================== Utility Methods ====================

    def is_basket_detected_with_position(self, expected_color: str) -> bool:
        """
        Check if a basket is detected with valid 2D position.

        Args:
            expected_color: Optional basket color to match ("blue" or "magenta")

        Returns:
            True if basket is detected with position and matches expected color (if specified)
        """
        if self._image_info_msg is None or self._image_info_msg.basket is None:
            return False

        basket = self._image_info_msg.basket

        # Check if position is available
        if basket.position_2d is None:
            return False

        # Check color if specified
        if basket.color != expected_color:
            return False

        return True

    def get_closest_ball(self) -> Optional[GreenBall]:
        """
        Get the closest detected ball based on 2D distance from robot.

        Returns:
            GreenBall object of closest ball, or None if no balls detected
        """
        balls = self.get_detected_balls()
        if not balls:
            return None

        # Find ball with smallest distance (norm of position_2d)

        closest_ball = min(balls, key=lambda b: b.position_2d[0] ** 2 + b.position_2d[1] ** 2)
        return closest_ball
