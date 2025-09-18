import sys, pathlib
import logging
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from utils.robot_motion import OmniMotionRobot
logging.basicConfig(level=logging.INFO)


bot = OmniMotionRobot()  # defaults to /dev/ttyACM0 @ 115200
bot.open()
try:
    # Send raw wheel speeds directly
    bot.send_command(speed1=0, speed2=0, speed3=0,
                     thrower_speed_percent=100, servo1=0, servo2=0,
                     disable_failsafe=0)

    # Or use the high-level move() (replace mapping with your kinematics)
    bot.move(x_speed=0, y_speed=-1, rot_speed=0)  # forward
finally:
    bot.close()