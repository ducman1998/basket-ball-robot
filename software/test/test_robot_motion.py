import sys, pathlib
import time 
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1])) # add software/ to path
import logging
from utils.robot_motion import OmniMotionRobot
from utils.custom_exceptions import SerialPortNotFound
logging.basicConfig(level=logging.INFO)

# Example usage for this file: 
#   python test/test_robot_motion.py or python test_robot_motion.py

bot = OmniMotionRobot()  # defaults to /dev/ttyACM0 @ 115200
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
    # bot.send_command(speed1=0, speed2=0, speed3=0,
    #                  thrower_speed_percent=50, servo1=0, servo2=0,
    #                  disable_failsafe=0)

    # Or use the high-level move() (replace mapping with your kinematics)
    for i in range(50):
        bot.move(x_speed=0, y_speed=0.5, rot_speed=0)  # forward
        time.sleep(0.1)
finally:
    bot.close()