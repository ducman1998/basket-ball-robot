import logging
import struct
from typing import Any, Optional

import numpy as np
import serial
from serial.serialutil import SerialException
from serial.tools import list_ports
from utils.config_utils import load_settings
from utils.custom_exceptions import SerialPortNotFound
from utils.number_utils import clip_int16, clip_uint16


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
        thrower_speed_percent: int = 0,
        servo1: int = 0,
        servo2: int = 0,
    ) -> None:
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

    PACK_FMT: str = "<hhhHHHBH"
    DELIMITER: int = 0xAAAA
    CONF: dict[str, Any] = load_settings().get("robot_configuration", {})
    STM_32_HWID: str = CONF.get(
        "hwid", "USB VID:PID=0483:5740"
    )  # default STM32CubeProgrammer USB PID/VID

    def __init__(
        self,
        polarity: int = 1,  # set to -1 if motors spin in reverse
        port: str = "auto",
        baudrate: int = 115200,
        timeout: float = 0.1,
    ) -> None:
        self.polarity = 1 if polarity >= 0 else -1
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self._ser: Optional[serial.Serial] = None

        # reference: https://hades.mech.northwestern.edu/images/7/7f/MR.pdf
        # ==> Chapter 13. Wheeled Mobile Robots
        # wheel 1: beta = 180+60=240 deg, alpha = 150 deg
        # wheel 2: beta = 0 deg, alpha = 270 deg
        # wheel 3: beta = 120 deg, alpha = 30 deg

        wheel_radius: float = self.CONF["wheel_radius"]  # wheel radius in meters
        dis: float = self.CONF["center_distance"]  # distance from center to wheel in meters
        # rotation shift between X axis of robot base-frame vs drived direction of wheel (deg)
        # angle of wheel (from robot center to the wheel) vs X axis of robot base-frame (deg)
        m1_beta: float = self.CONF["motor_1"]["beta"]
        m1_alpha: float = self.CONF["motor_1"]["alpha"]
        m2_beta: float = self.CONF["motor_2"]["beta"]
        m2_alpha: float = self.CONF["motor_2"]["alpha"]
        m3_beta: float = self.CONF["motor_3"]["beta"]
        m3_alpha: float = self.CONF["motor_3"]["alpha"]
        # Jacobian matrix to convert robot velocity to wheel speeds, using in high-level API move()
        # column vector v = [rot_speed, x_speed, y_speed].T as robot velocity in robot base frame
        # wheel speeds: [u1, u2, u3] = H * v
        # setting with the test robot:
        #   motor 1: [d*sin(240-150), cos(240), sin(240)]
        #   motor 2: [d*sin(0-270),   cos(0),   sin(0)]
        #   motor 3: [d*sin(120-30),  cos(120), sin(120)]
        self.H: np.ndarray = (
            1
            / wheel_radius
            * np.array(
                [
                    [
                        -dis * np.sin(np.deg2rad(m1_beta - m1_alpha)),
                        np.cos(np.deg2rad(m1_beta)),
                        np.sin(np.deg2rad(m1_beta)),
                    ],
                    [
                        -dis * np.sin(np.deg2rad(m2_beta - m2_alpha)),
                        np.cos(np.deg2rad(m2_beta)),
                        np.sin(np.deg2rad(m2_beta)),
                    ],
                    [
                        -dis * np.sin(np.deg2rad(m3_beta - m3_alpha)),
                        np.cos(np.deg2rad(m3_beta)),
                        np.sin(np.deg2rad(m3_beta)),
                    ],
                ]
            )
        )

        gear_ratio: float = self.CONF["gear_ratio"]  # gear ratio (wheel to motor)
        encoder_resolution: int = self.CONF[
            "encoder_resolution"
        ]  # encoder ticks per motor revolution
        pid_contro_freq: int = self.CONF["pid_contro_freq"]  # PID control frequency in Hz
        # convert wheel angular speed (rad/s) to mainboard units (ticks/s)
        self.wheel_to_mainboard_unit: float = (
            gear_ratio * encoder_resolution / (2 * np.pi * pid_contro_freq)
        )
        self.logger.info(
            f"Wheel speed to mainboard units: {self.wheel_to_mainboard_unit:.3f} ticks/s per rad/s"
        )

        # max speeds from config
        self.max_rot_speed: float = self.CONF.get("max_rot_speed", 2.0)  # rad/s
        self.max_xy_speed: float = self.CONF.get("max_xy_speed", 2.5)  # m/s

    # ---------- lifecycle ----------
    def open(self) -> None:
        if self.port != "auto":
            self.logger.info(f"Using specified serial port: {self.port}")
            try:
                self._ser = self._open_sport(self.port)
                return
            except (FileNotFoundError, SerialException):
                raise SerialPortNotFound(f"Failed to open specified serial port {self.port}.")

        auto_port = self._autoselect_port()
        if not auto_port:
            raise SerialPortNotFound("No serial ports found.")
        self.logger.info(f"Opening scanned serial port {auto_port} @ {self.baudrate}…")

        try:
            self._ser = self._open_sport(auto_port)
            return
        except (FileNotFoundError, SerialException) as e:
            raise SerialPortNotFound(f"Failed to open serial port {auto_port}: {e}")

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
            if self.STM_32_HWID in hwid:
                self.logger.info(f"Found port: {device}, hwid: {hwid}")
                return str(device)

        return None  # not valid port found

    def _open_sport(self, port: str) -> serial.Serial:
        ser = serial.Serial(port, self.baudrate, timeout=self.timeout)
        ser.reset_input_buffer()
        ser.reset_output_buffer()
        self.logger.info("Serial ready.")
        return ser

    def send_command(
        self,
        speed1: int,  # range -32768 to 32767 (int16)
        speed2: int,  # range -32768 to 32767 (int16)
        speed3: int,  # range -32768 to 32767 (int16)
        thrower_speed_percent: int = 0,  # range 0-100 (%)
        servo1: int = 1500,  # range 0-65535 (uint16)
        servo2: int = 1500,  # range 0-65535 (uint16)
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
        sv1 = clip_uint16(servo1)
        sv2 = clip_uint16(servo2)
        df = 1 if disable_failsafe else 0  # exactly 1 disables, else enables

        frame = struct.pack(self.PACK_FMT, s1, s2, s3, thrower_speed, sv1, sv2, df, self.DELIMITER)
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
            self.DELIMITER,
        )

    # ---------- high-level API ----------
    def move(
        self,
        x_speed: float,
        y_speed: float,
        rot_speed: float,
        thrower_speed_percent: int = 0,
        servo1: int = 0,
        servo2: int = 0,
    ) -> None:
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
        wheel_speeds = self.wheel_to_mainboard_unit * (
            self.H @ v
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
