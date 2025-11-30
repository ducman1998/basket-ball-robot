from utils.robot_motion import OmniMotionRobot
from utils.feedback import FeedbackSerial
from typing import Tuple, Optional
import time
import sys


if __name__ == "__main__":
    robot = OmniMotionRobot(
        wheel_radius=0.035,
        c2w_dis=0.1295,
        motor_01_angles=[240.0, 150.0],
        motor_02_angles=[0.0, 270.0],
        motor_03_angles=[120.0, 30.0],
        gear_ratio=18.75,
        encoder_resolution=64,
        pid_control_freq=100,
        max_rot_speed=3.0,
        max_xy_speed=1.0,
        hwid="USB VID:PID=0483:5740",
        cmd_fmt="<hhhHHHBH",
        fbk_fmt="<hhhhhhBH",
        delimieter=0xAAAA,
        polarity=1,
        baudrate=115200,
        timeout=0.1,
        port="auto",
    )
    robot.open()
    is_detected = False
    init_thrower_counter = 0
    started_throwing_time = None
    MIN_THROW_PERCENT = 10.0
    THROW_PERCENT = 75.0
    try:
        while True:
            if started_throwing_time is not None:
                if time.time() - started_throwing_time > 4.0:
                    # reset after 3 seconds of throwing
                    is_detected = False
                    init_thrower_counter = 0
                    started_throwing_time = None
                    vy = 0.1  # resume forward motion
                    thrower_percent = MIN_THROW_PERCENT
                    print("Resetting thrower and resuming motion.")
                elif thrower_percent < THROW_PERCENT:
                    raise ValueError("Thrower percent should not decrease during throwing phase.")

            # Example motion command
            vx = 0.0  # m/s
            vy = 0.1  # m/s
            wz = 0.0  # rad/s

            # Determine thrower speed based on state
            if not is_detected:
                thrower_percent = MIN_THROW_PERCENT
            else:
                # Already threw the ball, stop thrower
                vy = 0.0
                thrower_percent = THROW_PERCENT

            if is_detected:
                if init_thrower_counter == 0:
                    print("Initializing thrower...")
                init_thrower_counter += 1
                if init_thrower_counter < 60:  # run thrower for 60 cycles (~1s)
                    robot.move(0.0, 0.0, 0.0, thrower_percent, servo1=0, read_feedback=True)
                    time.sleep(0.015)
                    continue
                elif started_throwing_time is None:
                    started_throwing_time = time.time()
                    print("Thrower initialized. Throwing the ball....")

            print(f"Sending command: vx={vx}, vy={vy}, wz={wz}, thrower_percent={thrower_percent}")
            fb: FeedbackSerial = robot.move(
                x_speed=vx,
                # y_speed=vy,
                y_speed=0.0,
                rot_speed=wz,
                thrower_speed_percent=thrower_percent,
                servo1=3000,
                read_feedback=True,
            )
            if fb:
                print(f"Feedback: {fb.sensors}, Pos1: {fb.pos1}, Pos2: {fb.pos2}, Pos3: {fb.pos3}")
                if fb.sensors > 0 and not is_detected:
                    print("Obstacle detected!")
                    print("Waiting 5 seconds before next command...")
                    started_time = time.time()
                    is_detected = True
                    while time.time() - started_time < 5.0:
                        robot.move(
                            0.0, 0.0, 0.0, thrower_speed_percent=MIN_THROW_PERCENT
                        )  # stop the robot
                        time.sleep(0.015)  # wait before next command
            else:
                print("No feedback received.")

            time.sleep(0.015)  # send command every second
    except KeyboardInterrupt:
        print("Stopping robot...")
    finally:
        robot.move(0.0, 0.0, 0.0, 0.0)  # stop the robot
        robot.close()
