import math
from collections import deque
from time import time
from typing import Deque, List, Literal, Optional, Tuple, Union, cast

import numpy as np
from basket_robot_nodes.utils.constants import QOS_DEPTH
from basket_robot_nodes.utils.image_info import Basket, GreenBall, ImageInfo, Marker
from basket_robot_nodes.utils.number_utils import normalize_velocity
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSProfile
from shared_interfaces.msg import TwistStamped  # a custom message with thrower_percent
from std_msgs.msg import Bool, String


class PeripheralManager:
    def __init__(self, node: Node, queue_size: int = 20) -> None:
        """
        Initialize the PeripheralManager.

        Args:
            node: ROS2 Node instance for creating subscriptions
        """
        self._node = node

        # --- Queues for recent sensor data (fixed size) ---
        # Each entry is a tuple (message, timestamp)
        self._is_balls_detected_queue: Deque[Tuple[bool, float]] = deque(maxlen=queue_size)
        self._is_basket_detected_queue: Deque[Tuple[bool, float]] = deque(maxlen=queue_size)
        self._is_markers_detected_queue: Deque[Tuple[bool, float]] = deque(maxlen=queue_size)

        # Convenience references to the most recent messages (backwards compat)
        self._odom_msg: Optional[Odometry] = None
        self._last_odom_time: float = 0.0

        self._image_info_msg: Optional[ImageInfo] = None
        self._last_image_info_time: float = 0.0
        self._image_size: Optional[Tuple[int, int]] = None

        self._ir_sensor_status: Optional[bool] = None

        # Create subscriptions
        self._odom_sub = node.create_subscription(
            Odometry, "/odom", self._odom_callback, QoSProfile(depth=QOS_DEPTH)
        )

        self._image_info_sub = node.create_subscription(
            String, "/image/info", self._image_info_callback, QoSProfile(depth=QOS_DEPTH)
        )

        self._ir_sensor_sub = node.create_subscription(
            Bool, "/sensors/ir_sensor", self.ir_sensor_callback, QoSProfile(depth=QOS_DEPTH)
        )

        self.mainboard_controller_pub = node.create_publisher(
            TwistStamped, "cmd_vel", QoSProfile(depth=QOS_DEPTH)
        )
        # reusable control msg
        self.control_msg = TwistStamped()

        # internal variables
        self.target_basket_color: Optional[str] = None  # either "magenta" or "blue"
        self.latest_target_basket_pos_odom: Optional[Tuple[float, float]] = None  # in mm
        self.latest_target_basket_timestamp: float = 0.0
        self.latest_opponent_basket_pos_odom: Optional[Tuple[float, float]] = None  # in mm
        self.latest_opponent_basket_timestamp: float = 0.0

        node.get_logger().info("PeripheralManager initialized.")

    def reset(self) -> None:
        """Reset the PeripheralManager state."""
        self._is_balls_detected_queue.clear()
        self._is_basket_detected_queue.clear()
        self._is_markers_detected_queue.clear()

        self._odom_msg = None
        self._last_odom_time = 0.0

        self._image_info_msg = None
        self._last_image_info_time = 0.0
        self._image_size = None

        self._ir_sensor_status = None

        self.target_basket_color = None
        self.latest_target_basket_pos_odom = None
        self.latest_target_basket_timestamp = 0.0
        self.latest_opponent_basket_pos_odom = None
        self.latest_opponent_basket_timestamp = 0.0

    def set_target_basket_color(self, color: str) -> None:
        """Set the target basket color."""
        self.target_basket_color = color

    def get_target_basket_color(self) -> str:
        """Get the target basket color."""
        assert self.target_basket_color is not None, "Target basket color is not set."

        return self.target_basket_color

    def basket_color_to_marker_ids(self, color: str) -> List[int]:
        """Convert basket color to corresponding marker IDs."""
        if color == "magenta":
            return [11, 12]  # IDs for magenta basket markers
        else:  # blue
            return [21, 22]  # IDs for blue basket markers

    # ==================== Callback Methods ====================
    def _odom_callback(self, msg: Odometry) -> None:
        """Handle incoming odometry messages."""
        self._odom_msg = msg
        self._last_odom_time = time()

    def _image_info_callback(self, msg: String) -> None:
        """Handle incoming image info messages."""
        image_info = ImageInfo.from_json(msg.data)
        self._image_info_msg = cast(ImageInfo, image_info)

        ts = time()
        # push to fixed-size queue and update last refs
        self._is_balls_detected_queue.append((self.is_ball_detected(), ts))
        self._is_basket_detected_queue.append((self.is_basket_detected(), ts))
        self._is_markers_detected_queue.append((self.is_marker_detected(), ts))

        # store image size on first receipt
        if self._image_size is None and image_info.image_size is not None:
            self._image_size = image_info.image_size

        # update latest target basket position in odom frame (in mm)
        if (
            self.target_basket_color is not None
            and self.is_odom_ready()
            and image_info.basket is not None
            and image_info.basket.position_2d is not None
        ):
            if image_info.basket.color == self.target_basket_color:
                # store latest target basket position in odom frame
                pos_robot_mm = image_info.basket.position_2d  # (x, y) in robot frame (mm)
                self.latest_target_basket_pos_odom = self.robot_to_odom_coords(pos_robot_mm)
                self.latest_target_basket_timestamp = ts

            else:  # another/opponent's basket
                pos_robot_mm = image_info.basket.position_2d  # (x, y) in robot frame (mm)
                self.latest_opponent_basket_pos_odom = self.robot_to_odom_coords(pos_robot_mm)
                self.latest_opponent_basket_timestamp = ts

        self._last_image_info_time = ts

    def ir_sensor_callback(self, msg: Bool) -> None:
        """Handle incoming IR sensor messages."""
        self._ir_sensor_status = bool(msg.data)

    # ==================== Data Availability Methods ====================
    def is_ready(self) -> bool:
        """Check if all essential peripheral data is ready."""
        return self.is_odom_ready() and self.is_perception_ready() and self.is_ir_sensor_ready()

    def is_odom_ready(self) -> bool:
        return self._odom_msg is not None

    def is_perception_ready(self) -> bool:
        return self._image_info_msg is not None

    def is_ir_sensor_ready(self) -> bool:
        return self._ir_sensor_status is not None

    # ==================== Odometry Access Methods ====================
    def get_odom_msg(self) -> Optional[Odometry]:
        return self._odom_msg

    def get_robot_odom_position(self) -> Tuple[float, float]:
        assert self._odom_msg is not None, "Odometry message is not available."

        pos = self._odom_msg.pose.pose.position
        return (pos.x * 1000, pos.y * 1000)  # convert to mm

    def get_robot_quat(self) -> Tuple[float, float, float, float]:
        assert self._odom_msg is not None, "Odometry message is not available."

        quat = self._odom_msg.pose.pose.orientation
        return (quat.x, quat.y, quat.z, quat.w)

    def get_odom_yaw(self) -> float:
        """Get robot yaw in degrees from odometry."""
        assert self._odom_msg is not None, "Odometry message is not available."
        quat = self._odom_msg.pose.pose.orientation
        yaw_rad = 2.0 * math.atan2(quat.z, quat.w)
        yaw_deg = math.degrees(yaw_rad)
        yaw_deg = (yaw_deg + 360) % 360
        return yaw_deg

    def get_odom_timestamp(self) -> float:
        return self._last_odom_time

    # ==================== Image Info Access Methods ====================
    def get_image_info_msg(self) -> Optional[ImageInfo]:
        return self._image_info_msg

    def is_ball_detected(self) -> bool:
        if self._image_info_msg is None:
            return False
        return len(self._image_info_msg.balls) > 0

    def is_basket_detected(self) -> bool:
        if self._image_info_msg is None:
            return False
        return self._image_info_msg.basket is not None

    def is_marker_detected(self, basket_color: Optional[str] = None) -> bool:
        if self._image_info_msg is None:
            return False
        if basket_color is None:
            return len(self._image_info_msg.markers) > 0
        else:
            assert basket_color in ["magenta", "blue"], "Invalid basket color."
            marker_ids = self.basket_color_to_marker_ids(basket_color)
            for marker in self._image_info_msg.markers:
                if marker.id in marker_ids:
                    return True
            return False

    def get_detected_balls(self) -> Union[List[GreenBall], Tuple[GreenBall, ...]]:
        if self._image_info_msg is None:
            return []
        detected_balls = self._image_info_msg.balls
        return cast(Union[List[GreenBall], Tuple[GreenBall, ...]], detected_balls)

    def get_detected_basket(self) -> Optional[Basket]:
        if self._image_info_msg is None:
            return None
        return self._image_info_msg.basket

    def get_detected_markers(self) -> List[Marker]:
        if self._image_info_msg is None:
            return []

        detected_markers = self._image_info_msg.markers
        detected_markers = sorted(detected_markers, key=lambda m: m.id)
        return cast(List[Marker], detected_markers)

    def get_basket_position_2d(self) -> Optional[Tuple[float, float]]:
        if self._image_info_msg is None or self._image_info_msg.basket is None:
            return None

        pos_2d = self._image_info_msg.basket.position_2d
        return cast(Optional[Tuple[float, float]], pos_2d)

    def get_basket_distance(self) -> Optional[float]:
        basket_pos = self.get_basket_position_2d()
        if basket_pos is None:
            return None
        return float(np.linalg.norm(basket_pos))

    def get_basket_center_pixel(self) -> Optional[Tuple[int, int]]:
        if self._image_info_msg is None or self._image_info_msg.basket is None:
            return None
        center = self._image_info_msg.basket.center
        return cast(Tuple[int, int], center)

    def get_basket_color(self) -> Optional[str]:
        if self._image_info_msg is None or self._image_info_msg.basket is None:
            return None
        return cast(str, self._image_info_msg.basket.color)

    def get_image_size(self) -> Tuple[int, int]:
        assert self._image_size is not None, "Image size is not available yet."

        return self._image_size

    def get_image_info_timestamp(self) -> float:
        return self._last_image_info_time

    # =================== IR Sensor Accessor ====================
    def is_ball_grabbed(self) -> bool:
        assert self._ir_sensor_status is not None, "IR sensor status is not available yet."

        return self._ir_sensor_status

    # ==================== Queue Accessors & Stability Checks ====================
    def is_balls_detected_in_nframes(self, nframes: int) -> bool:
        """Check if balls were detected in the last nframes."""
        return self._is_detected_in_nframes(self._is_balls_detected_queue, nframes)

    def is_balls_not_detected_in_nframes(self, nframes: int) -> bool:
        """Check if balls were not detected in the last nframes."""
        return self._is_not_detected_in_nframes(self._is_balls_detected_queue, nframes)

    def is_basket_detected_in_nframes(self, nframes: int) -> bool:
        """Check if basket was detected in the last nframes."""
        return self._is_detected_in_nframes(self._is_basket_detected_queue, nframes)

    def is_basket_not_detected_in_nframes(self, nframes: int) -> bool:
        """Check if basket was not detected in the last nframes."""
        return self._is_not_detected_in_nframes(self._is_basket_detected_queue, nframes)

    def is_markers_detected_in_nframes(self, nframes: int) -> bool:
        """Check if markers were detected in the last nframes."""
        return self._is_detected_in_nframes(self._is_markers_detected_queue, nframes)

    def is_marker_not_detected_in_nframes(self, nframes: int) -> bool:
        """Check if markers were not detected in the last nframes."""
        return self._is_not_detected_in_nframes(self._is_markers_detected_queue, nframes)

    def _is_not_detected_in_nframes(
        self, det_queue: Deque[Tuple[bool, float]], nframes: int
    ) -> bool:
        """Generic method to check if detection was negative in the last nframes."""
        recent_items = list(det_queue)[-nframes:]
        return sum(it[0] for it in recent_items) == 0

    def _is_detected_in_nframes(self, det_queue: Deque[Tuple[bool, float]], nframes: int) -> bool:
        """Generic method to check if detection was positive in the last nframes."""
        recent_items = list(det_queue)[-nframes:]
        if len(recent_items) < nframes:
            return False
        return sum(it[0] for it in recent_items) == nframes

    # ============== Mainboard Control Methods ================
    def stop_robot(self) -> None:
        """Stop the robot by sending zero velocities."""
        self.move_robot_adv(0.0, 0.0, 0.0, 0.0, 0)

    def move_robot(
        self,
        vx: float,
        vy: float,
        wz: float,
    ) -> None:
        """Send velocity commands to the robot. vx, vy in m/s, wz in rad/s."""
        self.move_robot_adv(
            vx,
            vy,
            wz,
            thrower_percent=0.0,
            servo_speed=0,
            max_xy_speed=float("inf"),
            max_rot_speed=float("inf"),
            normalize=False,
        )

    def move_robot_wthrower(self, vx: float, vy: float, wz: float, thrower_percent: float) -> None:
        """Send velocity commands to the robot with thrower speed. vx, vy in m/s, wz in rad/s."""
        self.move_robot_adv(
            vx,
            vy,
            wz,
            thrower_percent=thrower_percent,
            servo_speed=0,
            max_xy_speed=float("inf"),
            max_rot_speed=float("inf"),
            normalize=False,
        )

    def move_robot_normalized(
        self,
        vx: float,
        vy: float,
        wz: float,
        max_xy_speed: Optional[float] = None,
        max_rot_speed: Optional[float] = None,
    ) -> None:
        """Send normalized velocity commands to the robot. vx, vy in m/s, wz in rad/s."""
        self.move_robot_adv(
            vx,
            vy,
            wz,
            thrower_percent=0.0,
            servo_speed=0,
            max_xy_speed=max_xy_speed,
            max_rot_speed=max_rot_speed,
            normalize=True,
        )

    def move_robot_adv(
        self,
        vx: float,
        vy: float,
        wz: float,
        thrower_percent: float,
        servo_speed: int,
        normalize: bool = False,
        max_xy_speed: Optional[float] = None,
        max_rot_speed: Optional[float] = None,
    ) -> None:
        """Send velocity commands to the robot. vx, vy in m/s, wz in rad/s."""
        if normalize and max_xy_speed is not None:
            vx, vy = normalize_velocity(vx, vy, max_xy_speed)
        elif max_xy_speed is not None:
            vx = np.clip(vx, -max_xy_speed, max_xy_speed)
            vy = np.clip(vy, -max_xy_speed, max_xy_speed)

        if max_rot_speed is not None:
            wz = np.clip(wz, -max_rot_speed, max_rot_speed)
        thrower_percent = np.clip(thrower_percent, 0, 100)
        servo_speed = int(np.clip(servo_speed, 0, 20000))

        self.publish_velocity_command(vx, vy, wz, thrower_percent, servo_speed)

    def publish_velocity_command(
        self, vx: float, vy: float, wz: float, thrower_percent: float, servo_speed: int
    ) -> None:
        """
        Publish a velocity command to the mainboard controller.

        Args:
            vx: Linear velocity in x (m/s)
            vy: Linear velocity in y (m/s)
            wz: Angular velocity around z (rad/s)
            thrower_percent: Thrower speed percentage (0-100)
            servo_speed: Servo speed (0-20000)
        """
        self.control_msg.twist.linear.x = float(vx)
        self.control_msg.twist.linear.y = float(vy)
        self.control_msg.twist.angular.z = float(wz)
        self.control_msg.thrower_percent = float(thrower_percent)
        self.control_msg.servo_speed = int(servo_speed)

        self.mainboard_controller_pub.publish(self.control_msg)

    # ==================== Utility Methods ====================
    def is_basket_detected_with_position(self, expected_color: str) -> bool:
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
        balls = self.get_detected_balls()
        if len(balls) == 0:
            return None

        # Find ball with smallest distance (norm of position_2d)
        closest_ball = min(balls, key=lambda b: b.position_2d[0] ** 2 + b.position_2d[1] ** 2)
        return closest_ball

    def get_closest_ball_position(self) -> Optional[Tuple[float, float]]:
        """Get the position_2d of the closest detected ball (unit: mm in robot frame)"""
        closest_ball = self.get_closest_ball()
        if closest_ball is None:
            return None
        return cast(Tuple[float, float], closest_ball.position_2d)

    def get_robot_to_odom_transform(self, include_z: bool = False) -> np.ndarray:
        """
        Get the transformation matrix from robot frame to odometry frame (trans in mm).
        If include_z is True, returns a 4x4 matrix; else returns a 3x3 matrix.
        """
        assert self._odom_msg is not None, "Odometry message is not available."

        pos = self._odom_msg.pose.pose.position
        quat = self._odom_msg.pose.pose.orientation

        # Convert quaternion to yaw
        yaw_rad = 2.0 * math.atan2(quat.z, quat.w)

        # Construct transformation matrix
        cos_yaw = math.cos(yaw_rad)
        sin_yaw = math.sin(yaw_rad)
        if include_z:
            transform = np.array(
                [
                    [cos_yaw, -sin_yaw, 0, pos.x * 1000],
                    [sin_yaw, cos_yaw, 0, pos.y * 1000],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ]
            )

        else:
            transform = np.array(
                [
                    [cos_yaw, -sin_yaw, pos.x * 1000],
                    [sin_yaw, cos_yaw, pos.y * 1000],
                    [0, 0, 1],
                ]
            )
        return transform

    def get_odom_to_robot_transform(self, include_z: Literal[False]) -> np.ndarray:
        """
        Get the transformation matrix from odometry frame to robot frame (trans in mm).
        If include_z is True, returns a 4x4 matrix; else returns a 3x3 matrix.
        """
        robot_to_odom = self.get_robot_to_odom_transform(include_z)
        odom_to_robot = np.linalg.inv(robot_to_odom)
        return odom_to_robot

    def robot_to_odom_coords(self, pos_robot_mm: Tuple[float, float]) -> Tuple[float, float]:
        """Convert robot-base frame to Odom frame. Both input & output are in mm"""

        t_or = self.get_robot_to_odom_transform()  # translation in mm
        pos_odom_mm = t_or @ np.array([pos_robot_mm[0], pos_robot_mm[1], 1.0])
        return (pos_odom_mm[0], pos_odom_mm[1])

    def odom_to_robot_coords(self, pos_odom_mm: Tuple[float, float]) -> Tuple[float, float]:
        """Convert Odom frame to robot-base frame. Both input & output are in mm"""
        t_ro = self.get_odom_to_robot_transform(False)  # translation in mm
        pos_robot_mm = t_ro @ np.array([pos_odom_mm[0], pos_odom_mm[1], 1.0])
        return (pos_robot_mm[0], pos_robot_mm[1])

    def get_stored_target_basket_pos(
        self, in_robot_frame: bool = False, max_age_s: float = 8.0
    ) -> Optional[Tuple[float, float]]:
        """Get the latest stored target basket position.
        Inputs:
            in_robot_frame: If True, return position in robot frame; else in odom frame.
            max_age_s: Maximum age of the stored position to be considered valid.
        Returns:
            Tuple of (x, y) position in mm, or None if data is too old.
        """
        assert (
            self.target_basket_color is not None
        ), "Target basket color is not set. Please call function 'set_target_basket_color' first."

        if time() - self.latest_target_basket_timestamp > max_age_s:
            return None

        if self.latest_target_basket_pos_odom is None:
            return None

        if in_robot_frame:
            return self.odom_to_robot_coords(self.latest_target_basket_pos_odom)
        else:
            return self.latest_target_basket_pos_odom

    def get_stored_opponent_basket_pos(
        self, in_robot_frame: bool = False, max_age_s: float = 8.0
    ) -> Optional[Tuple[float, float]]:
        """Get the latest stored opponent basket position.
        Inputs:
            in_robot_frame: If True, return position in robot frame; else in odom frame.
            max_age_s: Maximum age of the stored position to be considered valid.
        Returns:
            Tuple of (x, y) position in mm, or None if data is too old.
        """
        if time() - self.latest_opponent_basket_timestamp > max_age_s:
            return None

        if self.latest_opponent_basket_pos_odom is None:
            return None

        if in_robot_frame:
            return self.odom_to_robot_coords(self.latest_opponent_basket_pos_odom)
        else:
            return self.latest_opponent_basket_pos_odom
