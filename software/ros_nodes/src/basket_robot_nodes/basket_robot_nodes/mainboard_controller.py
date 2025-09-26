import time
from typing import Optional

import rclpy
from basket_robot_nodes.utils.feedback import FeedbackSerial
from basket_robot_nodes.utils.number_utils import thrower_clip
from basket_robot_nodes.utils.robot_motion import OmniMotionRobot
from rclpy.node import Node
from rclpy.qos import QoSProfile
from shared_interfaces.msg import TwistStamped, WheelPositions


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
        self.declare_parameter("wheel_radius", 0.035)
        self.declare_parameter("c2w_dis", 0.1295)
        self.declare_parameter("motor_01", [240.0, 150.0])
        self.declare_parameter("motor_02", [0.0, 270.0])
        self.declare_parameter("motor_03", [120.0, 30.0])
        self.declare_parameter("gear_ratio", 18.75)
        self.declare_parameter("encoder_resolution", 64)
        self.declare_parameter("pid_control_freq", 100)
        self.declare_parameter("max_rot_speed", 2.0)
        self.declare_parameter("max_xy_speed", 2.5)
        self.declare_parameter("hwid", "USB VID:PID=0483:5740")

        # read params once at startup
        w_radius = self.get_parameter("wheel_radius").get_parameter_value().double_value
        c2w_dis = self.get_parameter("c2w_dis").get_parameter_value().double_value
        motor_01_angles = self.get_parameter("motor_01").get_parameter_value().double_array_value
        motor_02_angles = self.get_parameter("motor_02").get_parameter_value().double_array_value
        motor_03_angles = self.get_parameter("motor_03").get_parameter_value().double_array_value
        gear_ratio = self.get_parameter("gear_ratio").get_parameter_value().double_value
        encoder_resolution = (
            self.get_parameter("encoder_resolution").get_parameter_value().integer_value
        )
        pid_control_freq = (
            self.get_parameter("pid_control_freq").get_parameter_value().integer_value
        )
        max_rot_speed = self.get_parameter("max_rot_speed").get_parameter_value().double_value
        max_xy_speed = self.get_parameter("max_xy_speed").get_parameter_value().double_value
        hwid = self.get_parameter("hwid").get_parameter_value().string_value

        self.timer = self.create_timer(0.1, self.controller_placeholder_func)
        self.cmd_vel_sub = self.create_subscription(
            TwistStamped, "cmd_vel", self.control_callback, QoSProfile(depth=10)
        )
        # setup ommiwheel robot controller
        self.controller_kin = OmniMotionRobot(
            wheel_radius=w_radius,
            c2w_dis=c2w_dis,
            motor_01_angles=motor_01_angles.tolist(),
            motor_02_angles=motor_02_angles.tolist(),
            motor_03_angles=motor_03_angles.tolist(),
            gear_ratio=gear_ratio,
            encoder_resolution=encoder_resolution,
            pid_control_freq=pid_control_freq,
            max_rot_speed=max_rot_speed,
            max_xy_speed=max_xy_speed,
            hwid=hwid,
        )
        self.controller_kin.open()  # open serial port
        self.timestamp = time.time()

        # Publisher for wheel positions
        self.wheel_pos_pub = self.create_publisher(
            WheelPositions, "wheel_positions", QoSProfile(depth=10)
        )

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
        tp = msg.thrower_percent
        thrower_percent = thrower_clip(tp)
        self.get_logger().info(
            f"vx: {vx:.2f}, vy: {vy:.2f}, wz: {wz:.2f}, tp: {thrower_percent:.2f}"
        )
        feedback: Optional[FeedbackSerial] = self.controller_kin.move(
            x_speed=vx,
            y_speed=vy,
            rot_speed=wz,
            thrower_speed_percent=thrower_percent,
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

        return


def main() -> None:
    rclpy.init()
    controller = MainboardController()
    rclpy.spin(controller)
    controller.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
