import logging
import sys
import time

import fix_import  # noqa: F401 # ensures software/ is in sys.path
from utils.custom_exceptions import SerialPortNotFound
from utils.robot_motion import OmniMotionRobot

logging.basicConfig(level=logging.INFO)

# Example usage for this file:
#   python test/test_robot_motion.py or python test_robot_motion.py

bot = OmniMotionRobot(
    wheel_radius=0.035,
    c2w_dis=0.1295,
    motor_01_angles=[240.0, 150.0],
    motor_02_angles=[0.0, 270.0],
    motor_03_angles=[120.0, 30.0],
    hwid="USB VID:PID=0483:5740",
    max_rot_speed=2.0,
    max_xy_speed=2.5,
    pid_control_freq=100,
    gear_ratio=18.75,
    encoder_resolution=64,
)  # defaults to /dev/ttyACM0 @ 115200
try:
    bot.open()
    print("Connected to robot.")
except SerialPortNotFound as e:
    print(f"Error: {e}")
    sys.exit(1)
except RuntimeError as e:
    print(f"Error: {e}")
    sys.exit(1)

try:
    # Send raw wheel speeds directly
    bot.send_command(
        speed1=0,
        speed2=0,
        speed3=0,
        thrower_speed_percent=50,
        servo1=0,
        servo2=0,
        disable_failsafe=False,
    )

    # Or use the high-level move() (replace mapping with your kinematics)
    for i in range(10):
        bot.move(x_speed=0, y_speed=0.0, rot_speed=0.5)  # forward
        time.sleep(0.1)
finally:
    bot.close()
