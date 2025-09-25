import os
import select
import sys
import termios
import tty

import rclpy
from rclpy.clock import Clock
from rclpy.qos import QoSProfile
from shared_interfaces.msg import TwistStamped

MAX_LIN_VEL = 0.5
MAX_ANG_VEL = 1.5

LIN_VEL_STEP_SIZE = 0.01
ANG_VEL_STEP_SIZE = 0.1

# Thrower settings (percent, 0..100)
THROWER_MIN = 0.0
THROWER_MAX = 100.0
THROWER_STEP = 5.0

msg = """
Control Your Basker Robot + Thrower!
-----------------------------------
Moving around:
        w
   a    s    d
        x

Thrower control:
    + / = : increase thrower percent
    -     : decrease thrower percent

w/x : increase/decrease linear velocity (~ 0.5 m/s)
a/d : increase/decrease angular velocity (~ 1.5 rad/s)
space key, s : force stop (velocities to zero)

CTRL-C to quit
"""


def get_key(settings):
    tty.setraw(sys.stdin.fileno())
    rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
    if rlist:
        key = sys.stdin.read(1)
    else:
        key = ""
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key


def print_status(target_linear_velocity, target_angular_velocity, thrower_percent):
    print(
        "currently:\tlinear {0:.2f}\tangular {1:.2f}\tthrower {2:.0f}%".format(
            target_linear_velocity, target_angular_velocity, thrower_percent
        )
    )


def make_simple_profile(output_vel, input_vel, slop):
    if input_vel > output_vel:
        output_vel = min(input_vel, output_vel + slop)
    elif input_vel < output_vel:
        output_vel = max(input_vel, output_vel - slop)
    else:
        output_vel = input_vel
    return output_vel


def constrain(val, low, high):
    return max(low, min(high, val))


def check_linear_limit_velocity(velocity):
    return constrain(velocity, -MAX_LIN_VEL, MAX_LIN_VEL)


def check_angular_limit_velocity(velocity):
    return constrain(velocity, -MAX_ANG_VEL, MAX_ANG_VEL)


def main():
    settings = None
    if os.name != "nt":
        settings = termios.tcgetattr(sys.stdin)

    rclpy.init()
    qos = QoSProfile(depth=10)
    node = rclpy.create_node("teleop_keyboard")

    # Velocity publisher (TwistStamped)
    vel_and_thrower_pub = node.create_publisher(TwistStamped, "cmd_vel", qos)

    status = 0
    target_linear_velocity = 0.0
    target_angular_velocity = 0.0
    control_linear_velocity = 0.0
    control_angular_velocity = 0.0

    # Thrower state
    thrower_percent = 0.0

    try:
        print(msg)
        while rclpy.ok():
            key = get_key(settings)

            # Movement keys
            if key == "w":
                target_linear_velocity = check_linear_limit_velocity(
                    target_linear_velocity + LIN_VEL_STEP_SIZE
                )
                status += 1
                print_status(target_linear_velocity, target_angular_velocity, thrower_percent)

            elif key == "x":
                target_linear_velocity = check_linear_limit_velocity(
                    target_linear_velocity - LIN_VEL_STEP_SIZE
                )
                status += 1
                print_status(target_linear_velocity, target_angular_velocity, thrower_percent)

            elif key == "a":
                target_angular_velocity = check_angular_limit_velocity(
                    target_angular_velocity + ANG_VEL_STEP_SIZE
                )
                status += 1
                print_status(target_linear_velocity, target_angular_velocity, thrower_percent)

            elif key == "d":
                target_angular_velocity = check_angular_limit_velocity(
                    target_angular_velocity - ANG_VEL_STEP_SIZE
                )
                status += 1
                print_status(target_linear_velocity, target_angular_velocity, thrower_percent)

            # Stop velocities
            elif key == " " or key == "s":
                target_linear_velocity = 0.0
                control_linear_velocity = 0.0
                target_angular_velocity = 0.0
                control_angular_velocity = 0.0
                thrower_percent = 0.0
                print_status(target_linear_velocity, target_angular_velocity, thrower_percent)

            # Thrower: + / = increase, - decrease
            elif key in ("+", "="):
                thrower_percent = constrain(
                    thrower_percent + THROWER_STEP, THROWER_MIN, THROWER_MAX
                )
                print_status(target_linear_velocity, target_angular_velocity, thrower_percent)

            elif key == "-":
                thrower_percent = constrain(
                    thrower_percent - THROWER_STEP, THROWER_MIN, THROWER_MAX
                )
                print_status(target_linear_velocity, target_angular_velocity, thrower_percent)

            else:
                if key == "\x03":  # CTRL-C
                    break

            if status == 20:
                print(msg)
                status = 0

            # Smooth the velocity commands
            control_linear_velocity = make_simple_profile(
                control_linear_velocity,
                target_linear_velocity,
                (LIN_VEL_STEP_SIZE / 2.0),
            )

            control_angular_velocity = make_simple_profile(
                control_angular_velocity,
                target_angular_velocity,
                (ANG_VEL_STEP_SIZE / 2.0),
            )

            # Publish velocities
            twist_stamped = TwistStamped()
            twist_stamped.header.stamp = Clock().now().to_msg()
            twist_stamped.header.frame_id = ""
            twist_stamped.twist.linear.x = control_linear_velocity
            twist_stamped.twist.linear.y = 0.0
            twist_stamped.twist.linear.z = 0.0
            twist_stamped.twist.angular.x = 0.0
            twist_stamped.twist.angular.y = 0.0
            twist_stamped.twist.angular.z = control_angular_velocity
            twist_stamped.thrower_percent = int(thrower_percent)
            vel_and_thrower_pub.publish(twist_stamped)
            # rclpy.spin_once(node)
    except Exception as e:
        print("Error:", e)

    finally:
        # On exit: zero velocities and thrower
        twist_stamped = TwistStamped()
        twist_stamped.header.stamp = Clock().now().to_msg()
        twist_stamped.header.frame_id = ""
        twist_stamped.twist.linear.x = 0.0
        twist_stamped.twist.linear.y = 0.0
        twist_stamped.twist.linear.z = 0.0
        twist_stamped.twist.angular.x = 0.0
        twist_stamped.twist.angular.y = 0.0
        twist_stamped.twist.angular.z = 0.0
        twist_stamped.thrower_percent = 0
        vel_and_thrower_pub.publish(twist_stamped)

        if settings is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)


if __name__ == "__main__":
    main()
