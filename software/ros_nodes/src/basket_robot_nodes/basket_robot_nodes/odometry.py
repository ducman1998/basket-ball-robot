from typing import Optional

import numpy as np
import rclpy
import tf2_ros
from basket_robot_nodes.utils.robot_motion import OmniMotionRobot
from geometry_msgs.msg import Quaternion, TransformStamped
from nav_msgs.msg import Odometry
from numpy.typing import NDArray
from rclpy.node import Node
from rclpy.qos import QoSProfile
from shared_interfaces.msg import WheelPositions
from std_msgs.msg import Header


def euler_to_quaternion(yaw: float) -> Quaternion:
    q = Quaternion()
    q.x = 0.0
    q.y = 0.0
    q.z = np.sin(yaw * 0.5)
    q.w = np.cos(yaw * 0.5)
    return q


def compute_dpos(new_pos: NDArray[np.int32], last_pos: NDArray[np.int32]) -> NDArray[np.int32]:
    """
    Inputs:
        new_pos: current encoder positions (ticks)
        last_pos: previous encoder positions (ticks)

    Returns:
        dpos: change in encoder positions (ticks), accounting for wraparound

    This function computes the difference between new_pos and last_pos,
    handling wraparound for int16 encoder values.
    """
    # Handles wraparound for int16 encoder values robustly
    diff = new_pos - last_pos
    # Wraparound correction for int16
    # Assuming encoder values are in range [-32768, 32767]
    diff = (diff + 32768) % 65536 - 32768
    return diff


class OdometryNode(Node):
    def __init__(self) -> None:
        super().__init__("odometry_node")
        # Declare and fetch parameters as in mainboard_controller, ignoring unused ones
        self.declare_parameter("wheel_radius", 0.035)
        self.declare_parameter("c2w_dis", 0.1295)
        self.declare_parameter("motor_01", [240.0, 150.0])
        self.declare_parameter("motor_02", [0.0, 270.0])
        self.declare_parameter("motor_03", [120.0, 30.0])
        self.declare_parameter("gear_ratio", 18.75)
        self.declare_parameter("encoder_resolution", 64)
        self.declare_parameter("pid_control_freq", 100)

        wheel_radius = self.get_parameter("wheel_radius").get_parameter_value().double_value
        c2w_dis = self.get_parameter("c2w_dis").get_parameter_value().double_value
        motor_01_angles = self.get_parameter("motor_01").get_parameter_value().double_array_value
        motor_02_angles = self.get_parameter("motor_02").get_parameter_value().double_array_value
        motor_03_angles = self.get_parameter("motor_03").get_parameter_value().double_array_value
        gear_ratio = self.get_parameter("gear_ratio").get_parameter_value().double_value
        encoder_res = self.get_parameter("encoder_resolution").get_parameter_value().integer_value
        pid_freq = self.get_parameter("pid_control_freq").get_parameter_value().integer_value

        # Set up kinematics
        self.controller_kin = OmniMotionRobot(
            wheel_radius=wheel_radius,
            c2w_dis=c2w_dis,
            motor_01_angles=motor_01_angles.tolist(),
            motor_02_angles=motor_02_angles.tolist(),
            motor_03_angles=motor_03_angles.tolist(),
            gear_ratio=gear_ratio,
            encoder_resolution=encoder_res,
            pid_control_freq=pid_freq,
        )

        self.last_pos: Optional[np.ndarray] = None
        self.last_time: Optional[float] = None

        self.x: float = 0.0
        self.y: float = 0.0
        self.yaw: float = 0.0

        self.odom_pub = self.create_publisher(Odometry, "odom", QoSProfile(depth=10))
        self.sub = self.create_subscription(
            WheelPositions, "wheel_positions", self.wheel_callback, QoSProfile(depth=10)
        )
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

    def publish_tf(self, x: float, y: float, yaw: float, stamp: float) -> None:
        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = "odom"
        t.child_frame_id = "base_footprint"
        t.transform.translation.x = x
        t.transform.translation.y = y
        t.transform.translation.z = 0.0
        q = euler_to_quaternion(yaw)
        t.transform.rotation = q
        self.tf_broadcaster.sendTransform(t)
        self.get_logger().info(f"Published TF: x={x:.2f}, y={y:.2f}, yaw={yaw:.2f}")

    def wheel_callback(self, msg: WheelPositions) -> None:
        # msg.pos1, msg.pos2, msg.pos3 are encoder positions (ticks)
        pos: NDArray[np.int32] = np.array([msg.pos1, msg.pos2, msg.pos3], dtype=np.int32)
        cur_time = self.get_clock().now().nanoseconds * 1e-9

        if self.last_pos is not None and self.last_time is not None:
            dt = cur_time - self.last_time
            if dt <= 0.0:
                return  # skip invalid dt

            dpos = compute_dpos(pos, self.last_pos)
            # Convert encoder ticks to wheel angular displacement (rad)
            ticks_per_rev = self.controller_kin.encoder_resolution * self.controller_kin.gear_ratio
            dtheta = dpos * 2 * np.pi / ticks_per_rev

            # Wheel angular velocities (rad/s), considering dt = 1.0s
            w_speeds = dtheta.reshape((3, 1))
            # Inverse kinematics: robot velocities in base frame
            # [rot_speed, x_speed, y_speed] = inv_jacobian @ wheel_speeds
            v_robot = self.controller_kin.inv_jacobian @ w_speeds

            # Integrate to update (x, y, yaw)
            omega_z, v_x, v_y = v_robot.ravel().tolist()
            if abs(omega_z) < 1e-6:
                dx = v_x
                dy = v_y
                dyaw = 0
            else:
                dx = (v_x * np.sin(omega_z) + v_y * (np.cos(omega_z) - 1)) / omega_z
                dy = (v_y * np.sin(omega_z) + v_x * (1 - np.cos(omega_z))) / omega_z
                dyaw = omega_z

            # Integrate robot frame movement to world frame
            self.x += dx * np.cos(self.yaw) - dy * np.sin(self.yaw)
            self.y += dx * np.sin(self.yaw) + dy * np.cos(self.yaw)
            self.yaw += dyaw

            # Publish odometry
            odom_msg = Odometry()
            odom_msg.header = Header()
            odom_msg.header.stamp = self.get_clock().now().to_msg()
            odom_msg.header.frame_id = "odom"
            odom_msg.child_frame_id = "base_footprint"
            odom_msg.pose.pose.position.x = float(self.x)
            odom_msg.pose.pose.position.y = float(self.y)
            odom_msg.pose.pose.orientation = euler_to_quaternion(float(self.yaw))
            odom_msg.twist.twist.linear.x = float(v_robot[1])
            odom_msg.twist.twist.linear.y = float(v_robot[2])
            odom_msg.twist.twist.angular.z = float(v_robot[0])

            self.odom_pub.publish(odom_msg)
            self.publish_tf(self.x, self.y, self.yaw, odom_msg.header.stamp)

        self.last_pos = pos
        self.last_time = cur_time


def main() -> None:
    rclpy.init()
    node = OdometryNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
