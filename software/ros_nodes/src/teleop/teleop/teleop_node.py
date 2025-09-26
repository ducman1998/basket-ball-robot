import sys
import termios

import rclpy
from rclpy.clock import Clock
from rclpy.qos import QoSProfile
from shared_interfaces.msg import TwistStamped  # a custom message with thrower_percent
from teleop.utils import (
    check_ang_limit,
    check_lin_limit,
    constrain,
    get_key,
    make_simple_profile,
    print_status,
)

# Limits
MAX_LIN_VEL = 1.0  # applies to both vx and vy
MAX_ANG_VEL = 1.5  # wz

LIN_VEL_STEP_SIZE = 0.05
ANG_VEL_STEP_SIZE = 0.1

# Thrower settings (percent, 0..100)
THROWER_MIN = 0.0
THROWER_MAX = 100.0
THROWER_STEP = 5.0

msg = """
Control Your Basket Robot (Ommiwheels) + Thrower
-----------------------------------------------
Axes (as per your robot mapping):
  vy: forward/backward
  vx: strafe right/left
  wz: yaw (rotation)

Keys:
  vy  :   w / x   (forward / back)
  vx  :   d / a   (right / left)
  wz  :   e / q   (rotate CW / CCW)

Thrower:
  + or = : increase thrower %
  -      : decrease thrower %

space or s : stop all (vx=vy=wz=0 and thrower=0)
CTRL-C     : quit
"""


def main() -> None:
    """
    Main function to run the teleop keyboard node.
    Captures keyboard input and publishes TwistStamped messages to control the robot.
    """
    settings = termios.tcgetattr(sys.stdin)  # save terminal settings

    rclpy.init()
    qos = QoSProfile(depth=10)
    node = rclpy.create_node("teleop_keyboard")  # create ROS2 node

    pub = node.create_publisher(TwistStamped, "cmd_vel", qos)  # publisher for cmd_vel

    status = 0  # loop counter for printing help

    # Targets (what keys set)
    target_vy = 0.0  # forward/backward
    target_vx = 0.0  # right/left
    target_wz = 0.0  # yaw

    # Controls (smoothed outputs we publish)
    ctrl_vy = 0.0
    ctrl_vx = 0.0
    ctrl_wz = 0.0

    thrower_percent = 0.0

    try:
        print(msg)
        while rclpy.ok():
            key = get_key(settings)

            if key == "w":
                target_vy = check_lin_limit(target_vy + LIN_VEL_STEP_SIZE, MAX_LIN_VEL)
                status += 1
            elif key == "x":
                target_vy = check_lin_limit(target_vy - LIN_VEL_STEP_SIZE, MAX_LIN_VEL)
                status += 1
            elif key == "d":
                target_vx = check_lin_limit(target_vx + LIN_VEL_STEP_SIZE, MAX_LIN_VEL)
                status += 1
            elif key == "a":
                target_vx = check_lin_limit(target_vx - LIN_VEL_STEP_SIZE, MAX_LIN_VEL)
                status += 1
            elif key == "e":
                # CW (negative)
                target_wz = check_ang_limit(target_wz - ANG_VEL_STEP_SIZE, MAX_ANG_VEL)
                status += 1
            elif key == "q":
                # CCW (positive)
                target_wz = check_ang_limit(target_wz + ANG_VEL_STEP_SIZE, MAX_ANG_VEL)
                status += 1
            elif key in ("+", "="):
                thrower_percent = constrain(
                    thrower_percent + THROWER_STEP, THROWER_MIN, THROWER_MAX
                )
            elif key == "-":
                thrower_percent = constrain(
                    thrower_percent - THROWER_STEP, THROWER_MIN, THROWER_MAX
                )
            elif key in (" ", "s"):
                target_vy = target_vx = target_wz = 0.0
                ctrl_vy = ctrl_vx = ctrl_wz = 0.0
                thrower_percent = 0.0
            elif key == "\x03":  # CTRL-C
                break

            if status == 20:
                print(msg)
                status = 0

            # Smooth
            ctrl_vy = make_simple_profile(ctrl_vy, target_vy, LIN_VEL_STEP_SIZE / 2.0)
            ctrl_vx = make_simple_profile(ctrl_vx, target_vx, LIN_VEL_STEP_SIZE / 2.0)
            ctrl_wz = make_simple_profile(ctrl_wz, target_wz, ANG_VEL_STEP_SIZE / 2.0)

            print_status(ctrl_vy, ctrl_vx, ctrl_wz, thrower_percent)

            # Publish (your custom TwistStamped has thrower_percent)
            out = TwistStamped()
            out.header.stamp = Clock().now().to_msg()
            out.header.frame_id = "base_footprint"
            out.twist.linear.y = ctrl_vy
            out.twist.linear.x = ctrl_vx
            out.twist.linear.z = 0.0
            out.twist.angular.x = 0.0
            out.twist.angular.y = 0.0
            out.twist.angular.z = ctrl_wz
            out.thrower_percent = int(thrower_percent)
            pub.publish(out)

    except (KeyboardInterrupt, EOFError):
        print("Exiting...")
    except Exception as e:
        print("Error:", e)
    finally:
        # Zero on exit
        out = TwistStamped()
        out.header.stamp = Clock().now().to_msg()
        out.header.frame_id = ""
        out.twist.linear.x = 0.0
        out.twist.linear.y = 0.0
        out.twist.linear.z = 0.0
        out.twist.angular.x = 0.0
        out.twist.angular.y = 0.0
        out.twist.angular.z = 0.0
        out.thrower_percent = 0
        pub.publish(out)
        # cleanup
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)


if __name__ == "__main__":
    main()
