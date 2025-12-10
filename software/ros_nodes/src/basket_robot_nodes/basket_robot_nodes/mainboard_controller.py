import time
from typing import Optional

import rclpy
from basket_robot_nodes.utils.constants import QOS_DEPTH
from basket_robot_nodes.utils.feedback import FeedbackSerial
from basket_robot_nodes.utils.robot_motion import OmniMotionRobot
from basket_robot_nodes.utils.ros_utils import (
    float_array_descriptor,
    float_descriptor,
    int_descriptor,
    log_initialized_parameters,
    parse_log_level,
    str_descriptor,
)
from rclpy.node import Node
from rclpy.qos import QoSProfile
from shared_interfaces.msg import TwistStamped, WheelPositions
from std_msgs.msg import Bool


class MainboardController(Node):
    def __init__(self) -> None:
        """
        Mainboard controller node.
        Subscribes to "cmd_vel" topic to receive velocity commands and thrower percentage.
        Communicates with the mainboard to control the robot's motion and thrower.
        Publishes wheel positions for odometry.
        """
        super().__init__("mainboard_controller_node")
        # declare params (defaults are used only if launch doesn't override)
        self.declare_common_parameters(self)

        # setup ommiwheel robot controller
        self.controller_kin: OmniMotionRobot = self.init_ommi_controller(self)
        self.controller_kin.open()  # open serial port

        # create a timer to keep the node alive
        self.timer = self.create_timer(0.1, self.controller_placeholder_func)
        self.cmd_vel_sub = self.create_subscription(
            TwistStamped, "cmd_vel", self.control_callback, QoSProfile(depth=QOS_DEPTH)
        )

        self.timestamp = time.time()

        # Publisher for wheel positions
        self.wheel_pos_pub = self.create_publisher(
            WheelPositions, "wheel_positions", QoSProfile(depth=QOS_DEPTH)
        )
        self.sensor_status_pub = self.create_publisher(
            Bool, "sensors/ir_sensor", QoSProfile(depth=QOS_DEPTH)
        )
        # for checking: log all initialized parameters
        log_initialized_parameters(self)

    @staticmethod
    def declare_common_parameters(node: Node) -> None:
        """Declare parameters common to both MainboardControllerNode and OdometryNode."""
        node.declare_parameter("wheel_radius", descriptor=float_descriptor)
        node.declare_parameter("center_to_wheel_dis", descriptor=float_descriptor)
        node.declare_parameter("motor_01", descriptor=float_array_descriptor)
        node.declare_parameter("motor_02", descriptor=float_array_descriptor)
        node.declare_parameter("motor_03", descriptor=float_array_descriptor)
        node.declare_parameter("gear_ratio", descriptor=float_descriptor)
        node.declare_parameter("encoder_resolution", descriptor=int_descriptor)
        node.declare_parameter("pid_control_freq", descriptor=int_descriptor)
        node.declare_parameter("max_rot_speed", descriptor=float_descriptor)
        node.declare_parameter("max_xy_speed", descriptor=float_descriptor)
        node.declare_parameter("hwid", descriptor=str_descriptor)
        node.declare_parameter("baudrate", descriptor=int_descriptor)
        node.declare_parameter("polarity", descriptor=int_descriptor)
        node.declare_parameter("serial_timeout", descriptor=float_descriptor)
        node.declare_parameter("port", descriptor=str_descriptor)
        node.declare_parameter("cmd_fmt", descriptor=str_descriptor)
        node.declare_parameter("fbk_fmt", descriptor=str_descriptor)
        node.declare_parameter("delimiter", descriptor=int_descriptor)
        node.declare_parameter("log_level", descriptor=str_descriptor)

    @staticmethod
    def init_ommi_controller(node: Node) -> OmniMotionRobot:
        # read params once at startup
        w_radius = node.get_parameter("wheel_radius").get_parameter_value().double_value
        c2w_dis = node.get_parameter("center_to_wheel_dis").get_parameter_value().double_value
        motor_01_angles = node.get_parameter("motor_01").get_parameter_value().double_array_value
        motor_02_angles = node.get_parameter("motor_02").get_parameter_value().double_array_value
        motor_03_angles = node.get_parameter("motor_03").get_parameter_value().double_array_value
        gear_ratio = node.get_parameter("gear_ratio").get_parameter_value().double_value
        en_resolution = node.get_parameter("encoder_resolution").get_parameter_value().integer_value
        pid_freq = node.get_parameter("pid_control_freq").get_parameter_value().integer_value
        max_rot_speed = node.get_parameter("max_rot_speed").get_parameter_value().double_value
        max_xy_speed = node.get_parameter("max_xy_speed").get_parameter_value().double_value
        hwid = node.get_parameter("hwid").get_parameter_value().string_value
        baudrate = node.get_parameter("baudrate").get_parameter_value().integer_value
        polarity = node.get_parameter("polarity").get_parameter_value().integer_value
        serial_timeout = node.get_parameter("serial_timeout").get_parameter_value().double_value
        port = node.get_parameter("port").get_parameter_value().string_value
        cmd_fmt = node.get_parameter("cmd_fmt").get_parameter_value().string_value
        fbk_fmt = node.get_parameter("fbk_fmt").get_parameter_value().string_value
        delimieter = node.get_parameter("delimiter").get_parameter_value().integer_value
        log_level = node.get_parameter("log_level").get_parameter_value().string_value

        # read and set logging level
        node.get_logger().set_level(parse_log_level(log_level))
        node.get_logger().info(f"Set node {node.get_name()} log level to {log_level}.")

        controller = OmniMotionRobot(
            wheel_radius=w_radius,
            c2w_dis=c2w_dis,
            motor_01_angles=motor_01_angles.tolist(),
            motor_02_angles=motor_02_angles.tolist(),
            motor_03_angles=motor_03_angles.tolist(),
            gear_ratio=gear_ratio,
            encoder_resolution=en_resolution,
            pid_control_freq=pid_freq,
            max_rot_speed=max_rot_speed,
            max_xy_speed=max_xy_speed,
            hwid=hwid,
            cmd_fmt=cmd_fmt,
            fbk_fmt=fbk_fmt,
            delimieter=delimieter,
            polarity=polarity,
            baudrate=baudrate,
            timeout=serial_timeout,
            port=port,
        )
        return controller

    def controller_placeholder_func(self) -> None:
        # do nothing to keep this package running
        pass

    def control_callback(self, msg: TwistStamped) -> None:
        """Callback for "cmd_vel" topic subscription.
        Inputs:
            msg: TwistStamped message containing velocity commands and thrower percentage.
        """
        self.timestamp = time.time()
        # extract values from message
        vx: float = msg.twist.linear.x
        vy: float = msg.twist.linear.y
        wz: float = msg.twist.angular.z
        tp: float = msg.thrower_percent
        servo_speed: int = msg.servo_speed
        thrower_percent = max(0, min(100, tp))  # clip to [0, 100]
        servo_speed = max(0, min(20000, servo_speed))  # clip to [0, 20000]
        self.get_logger().info(
            f"vx: {vx:.2f}, vy: {vy:.2f}, wz: {wz:.2f}, "
            + f"tp: {thrower_percent:.2f}, servo_speed: {servo_speed}"
        )
        feedback: Optional[FeedbackSerial] = self.controller_kin.move(
            x_speed=vx,
            y_speed=vy,
            rot_speed=wz,
            thrower_speed_percent=thrower_percent,
            servo1=servo_speed,
            read_feedback=True,
        )

        if not feedback:
            self.get_logger().warn("No feedback from mainboard!")
        else:
            # log feedback for debugging
            self.get_logger().info(
                f"Feedback: pos1: {feedback.pos1}, pos2: {feedback.pos2}, pos3: {feedback.pos3}"
            )
            # Publish wheel positions for odometry
            wheel_msg = WheelPositions()
            wheel_msg.pos1 = feedback.pos1
            wheel_msg.pos2 = feedback.pos2
            wheel_msg.pos3 = feedback.pos3
            self.wheel_pos_pub.publish(wheel_msg)

            sensor_status_msg = Bool()
            sensor_status_msg.data = feedback.sensors > 0
            self.sensor_status_pub.publish(sensor_status_msg)

        return


def main() -> None:
    rclpy.init()
    controller = MainboardController()
    rclpy.spin(controller)
    controller.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
