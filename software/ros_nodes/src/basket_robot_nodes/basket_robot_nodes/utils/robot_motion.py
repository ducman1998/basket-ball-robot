import logging
import struct
import time
from typing import List, Literal, Optional, Union

import numpy as np
import serial
from serial.serialutil import SerialException
from serial.tools import list_ports

from .custom_exceptions import SerialPortNotFoundError
from .feedback import FeedbackSerial
from .number_utils import clip_int16, clip_uint16


class IRobotMotion:
    def open(self) -> None:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError

    def move(
        self,
        x_speed: float,
        y_speed: float,
        rot_speed: float,
        thrower_speed_percent: float = 0,
        servo1: int = 0,
        servo2: int = 0,
        read_feedback: bool = False,
    ) -> Optional[FeedbackSerial]:
        raise NotImplementedError


class OmniMotionRobot(IRobotMotion):
    """
    Sends packed commands to mainboard controlled by STM32 chip on the Basketball Robot:

    struct Command (packed, little-endian)
        int16_t  speed1
        int16_t  speed2
        int16_t  speed3
        uint16_t throwerSpeed      # 48-2047 (0-100% mapped to 48-2047)
        uint16_t servo1
        uint16_t servo2
        uint8_t  disableFailsafe   # 1 to disable failsafe, else enable
        uint16_t delimiter         # 0xAAAA

    Python pack format: '<hhhHHHBH'

    This class will try to connect to the default port (e.g. /dev/ttyACM0),
    if not found, it will scan for a likely port.
    """

    logger = logging.getLogger(__name__)

    def __init__(
        self,
        # robot configuration settings
        wheel_radius: float,  # wheel radius in meters
        c2w_dis: float,  # distance from center to wheel in meters
        motor_01_angles: List[Union[float, int]],  # [beta, alpha] in degrees
        motor_02_angles: List[Union[float, int]],  # [beta, alpha] in degrees
        motor_03_angles: List[Union[float, int]],  # [beta, alpha] in degrees
        gear_ratio: float,  # gear ratio (wheel to motor)
        encoder_resolution: int,  # encoder ticks per motor revolution
        pid_control_freq: int,  # PID control frequency in Hz
        max_rot_speed: float,  # rad/s
        max_xy_speed: float,  # m/s
        max_servo_speed: int,
        # serial settings
        hwid: str,
        cmd_fmt: str,
        fbk_fmt: str,
        delimieter: int,
        polarity: int,  # set to -1 if motors spin in reverse
        baudrate: int,
        timeout: float,
        port: Union[Literal["auto"], str],
    ) -> None:
        self.max_rot_speed = max_rot_speed
        self.max_xy_speed = max_xy_speed
        self.max_servo_speed = max_servo_speed
        # reference: https://hades.mech.northwestern.edu/images/7/7f/MR.pdf
        # ==> Chapter 13. Wheeled Mobile Robots
        # wheel 1: beta = 180+60=240 deg, alpha = 150 deg
        # wheel 2: beta = 0 deg, alpha = 270 deg
        # wheel 3: beta = 120 deg, alpha = 30 deg

        # rotation shift between X axis of robot base-frame vs drived direction of wheel (deg)
        # angle of wheel (from robot center to the wheel) vs X axis of robot base-frame (deg)
        m1_beta, m1_alpha = motor_01_angles
        m2_beta, m2_alpha = motor_02_angles
        m3_beta, m3_alpha = motor_03_angles

        self.encoder_resolution = encoder_resolution
        self.gear_ratio = gear_ratio
        self.hwid = hwid
        self.cmd_fmt = cmd_fmt
        self.fbk_fmt = fbk_fmt
        self.fbk_struct = struct.Struct(fbk_fmt)
        self.fbk_size = struct.calcsize(fbk_fmt)
        self.delimiter = delimieter
        self.polarity = 1 if polarity >= 0 else -1
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self._ser: Optional[serial.Serial] = None

        # Jacobian matrix to convert robot velocity to wheel speeds, using in high-level API move()
        # column vector v = [rot_speed, x_speed, y_speed].T as robot velocity in robot base frame
        # wheel speeds: [u1, u2, u3] = H * v
        # setting with the test robot:
        #   motor 01: [d*sin(240-150), cos(240), sin(240)]
        #   motor 02: [d*sin(0-270),   cos(0),   sin(0)]
        #   motor 03: [d*sin(120-30),  cos(120), sin(120)]
        self.jacobian: np.ndarray = (
            1
            / wheel_radius
            * np.array(
                [
                    [
                        c2w_dis * np.sin(np.deg2rad(m1_beta - m1_alpha)),
                        np.cos(np.deg2rad(m1_beta)),
                        np.sin(np.deg2rad(m1_beta)),
                    ],
                    [
                        c2w_dis * np.sin(np.deg2rad(m2_beta - m2_alpha)),
                        np.cos(np.deg2rad(m2_beta)),
                        np.sin(np.deg2rad(m2_beta)),
                    ],
                    [
                        c2w_dis * np.sin(np.deg2rad(m3_beta - m3_alpha)),
                        np.cos(np.deg2rad(m3_beta)),
                        np.sin(np.deg2rad(m3_beta)),
                    ],
                ]
            )
        )
        self.inv_jacobian = np.linalg.pinv(self.jacobian)

        # convert wheel angular speed (rad/s) to mainboard units (ticks/s)
        self.wheel_to_mb_unit: float = (
            gear_ratio * encoder_resolution / (2 * np.pi * pid_control_freq)
        )
        self.logger.info(
            f"Wheel speed to mainboard units: {self.wheel_to_mb_unit:.3f} ticks/s per rad/s"
        )
        self.logger.info(f"Jacobian matrix:\n{self.jacobian}")

    # ---------- lifecycle ----------
    def open(self) -> None:
        if self.port != "auto":
            self.logger.info(f"Using specified serial port: {self.port}")
            try:
                self._ser = self._open_sport(self.port)
                return
            except (FileNotFoundError, SerialException):
                raise SerialPortNotFoundError(f"Failed to open specified serial port {self.port}.")

        auto_port = self._autoselect_port()
        if not auto_port:
            raise SerialPortNotFoundError("No serial ports found.")
        self.logger.info(f"Opening scanned serial port {auto_port} @ {self.baudrate}…")

        try:
            self._ser = self._open_sport(auto_port)
            return
        except (FileNotFoundError, SerialException) as e:
            raise SerialPortNotFoundError(f"Failed to open serial port {auto_port}: {e}")

    def close(self) -> None:
        self.logger.info("Shutting down…")
        if self._ser and self._ser.is_open:
            try:
                self._ser.flush()
            finally:
                self._ser.close()
        self._ser = None

    # ---------- port scanning ----------
    def _autoselect_port(self) -> Optional[str]:
        """Auto-scan for a likely serial port based on hardware ID."""
        candidates = list_ports.comports()
        if candidates is None or len(candidates) == 0:
            self.logger.error("No serial ports found.")
            return None  # none found

        for device, _, hwid in candidates:
            if self.hwid in hwid:
                self.logger.info(f"Found port: {device}, hwid: {hwid}")
                return str(device)

        return None  # not valid port found

    def _open_sport(self, port: str) -> serial.Serial:
        ser = serial.Serial(port, self.baudrate, timeout=self.timeout)
        ser.reset_input_buffer()
        ser.reset_output_buffer()
        self.logger.info("Serial ready.")
        return ser

    def clear_rx(self) -> None:
        """Drop any stale bytes waiting in the input buffer."""
        if not (self._ser and self._ser.is_open):
            return
        self._ser.reset_input_buffer()

    def send_command(
        self,
        speed1: int,  # range -32768 to 32767 (int16)
        speed2: int,  # range -32768 to 32767 (int16)
        speed3: int,  # range -32768 to 32767 (int16)
        thrower_speed_percent: float = 0,  # range 0-100 (%)
        servo1: int = 0,  # range 0-65535 (uint16)
        servo2: int = 0,  # range 0-65535 (uint16)
        disable_failsafe: bool = False,
    ) -> None:
        """Pack and send one command frame."""
        if not (self._ser and self._ser.is_open):
            raise RuntimeError("Serial port not open. Call .open() first.")

        # Apply polarity to wheel speeds
        s1 = clip_int16(self.polarity * speed1)
        s2 = clip_int16(self.polarity * speed2)
        s3 = clip_int16(self.polarity * speed3)

        thrower_speed = 48 + int(
            thrower_speed_percent / 100 * (2047 - 48)
        )  # scale 0-100% to 0-65535
        thrower_speed = clip_uint16(thrower_speed)
        sv1 = min(self.max_servo_speed, max(0, servo1))
        sv2 = min(self.max_servo_speed, max(0, servo2))
        sv1 = clip_uint16(servo1)
        sv2 = clip_uint16(servo2)
        df = 1 if disable_failsafe else 0  # exactly 1 disables, else enables

        frame = struct.pack(self.cmd_fmt, s1, s2, s3, thrower_speed, sv1, sv2, df, self.delimiter)
        self.clear_rx()
        self._ser.write(frame)
        # Optional: self._ser.flush()  # uncomment if you need blocking send

        self.logger.debug(
            "TX frame: s1=%d s2=%d s3=%d thrower=%d servo1=%d servo2=%d df=%d delim=0x%04X",
            s1,
            s2,
            s3,
            thrower_speed,
            sv1,
            sv2,
            df,
            self.delimiter,
        )

    def read_feedback(self, timeout: float = 0.05) -> list[Union[int, float]]:
        """
        Read one feedback frame with delimiter-based resynchronization.

        Returns tuple:
          (speed1, speed2, speed3, pos1, pos2, pos3, sensors, feedback_delimiter)

        Raises TimeoutError if no full frame before timeout.
        """
        if not (self._ser and self._ser.is_open):
            raise RuntimeError("Serial port not open. Call .open() first.")

        buf = bytearray()
        deadline = time.monotonic() + timeout
        delimiter = self.delimiter.to_bytes(2, byteorder="little")
        while time.monotonic() < deadline:
            n_waiting = self._ser.in_waiting or 1
            chunk = self._ser.read(n_waiting)
            if not chunk:
                continue
            buf += chunk

            # update read timestamp
            timestamp = time.time()

            # search for delimiter occurrences
            search_from = 0
            while True:
                idx = buf.find(delimiter, search_from)
                if idx == -1:
                    # trim runaway buffer
                    if len(buf) > self.fbk_size * 4:
                        buf[:] = buf[-self.fbk_size * 2 :]
                    break

                frame_start = idx - (self.fbk_size - 2)
                frame_end = idx + 2  # exclusive
                if frame_start >= 0 and frame_end <= len(buf):
                    frame = bytes(buf[frame_start:frame_end])
                    if len(frame) == self.fbk_size:
                        vals = self.fbk_struct.unpack(frame)
                        fb_delim = vals[-1]
                        if fb_delim == self.delimiter:
                            # consume up to frame_end and return
                            del buf[:frame_end]
                            return [*vals, timestamp]
                        else:
                            # false positive; move search window
                            search_from = idx + 1
                            continue
                else:
                    # not enough bytes yet
                    break

        raise TimeoutError("No complete feedback frame before timeout.")

    # ---------- high-level API ----------
    def move(
        self,
        x_speed: float,
        y_speed: float,
        rot_speed: float,
        thrower_speed_percent: float = 0,
        servo1: int = 0,
        servo2: int = 0,
        read_feedback: bool = False,
    ) -> Optional[FeedbackSerial]:
        """
        Example shim: convert abstract (rot,x,y) into 3 wheel speeds.
        If you already have wheel speeds, call send_command(...) directly.
        """
        if abs(rot_speed) > self.max_rot_speed:
            rot_speed = np.sign(rot_speed) * self.max_rot_speed
        if abs(x_speed) > self.max_xy_speed:
            x_speed = np.sign(x_speed) * self.max_xy_speed
        if abs(y_speed) > self.max_xy_speed:
            y_speed = np.sign(y_speed) * self.max_xy_speed

        v = np.array([rot_speed, x_speed, y_speed])  # desired robot velocity (m/s and rad/s)
        wheel_speeds = self.wheel_to_mb_unit * (
            self.jacobian @ v
        )  # matrix multiply to get wheel speeds (rad/s)

        s1, s2, s3 = (int(round(ws)) for ws in wheel_speeds)
        # No servos/thrower; failsafe enabled by default (False)
        self.send_command(
            s1,
            s2,
            s3,
            thrower_speed_percent=thrower_speed_percent,
            servo1=servo1,
            servo2=servo2,
            disable_failsafe=False,
        )

        if not read_feedback:
            return None

        try:
            feedback_values = self.read_feedback(timeout=self.timeout)
            (
                actual_s1,
                actual_s2,
                actual_s3,
                pos1,
                pos2,
                pos3,
                sensors,
                fb_delim,
                timestamp,
            ) = feedback_values
            return FeedbackSerial(
                int(actual_s1),
                int(actual_s2),
                int(actual_s3),
                int(pos1),
                int(pos2),
                int(pos3),
                int(sensors),
                int(fb_delim),
                timestamp,
            )
        except TimeoutError:
            return None
