import numpy as np 
import logging
import struct
import time
import serial  # pip install pyserial
from typing import Optional


class IRobotMotion:
    def open(self) -> None:
        raise NotImplementedError
    def close(self) -> None:
        raise NotImplementedError
    def move(self, x_speed: float, y_speed: float, rot_speed: float) -> None:
        raise NotImplementedError


class OmniMotionRobot(IRobotMotion):
    """
    Sends packed commands:

    struct Command (packed, little-endian)
        int16_t  speed1
        int16_t  speed2
        int16_t  speed3
        uint16_t throwerSpeed
        uint16_t servo1
        uint16_t servo2
        uint8_t  disableFailsafe   # 1 to disable failsafe, else enable
        uint16_t delimiter         # 0xAAAA

    Python pack format: '<hhhHHHBH'
    """
    logger = logging.getLogger(__name__)

    PACK_FMT = "<hhhHHHBH"
    DELIMITER = 0xAAAA

    def __init__(
        self,
        polarity: int = 1,
        port: str = "/dev/ttyACM0",
        baudrate: int = 115200,
        timeout: float = 0.1,
        dtr_reset: bool = True,
    ) -> None:
        self.polarity = 1 if polarity >= 0 else -1
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.dtr_reset = dtr_reset
        self._ser: Optional[serial.Serial] = None
        
        # wheel 1: beta = 180+60=240, alpha = 150
        # wheel 2: beta = 0, alpha = 270
        # wheel 3: beta = 120, alpha = 30

        # 1: [d*sin(240-150), cos(240), sin(240)] 
        # 2: [d*sin(0-270), cos(0), sin(0)]
        # 3: [d*sin(120-30), cos(120), sin(120)]
        wheel_radius = 35/1000  # wheel radius in meters
        d = 129.5/1000  # distance from center to wheel in meters
        self.H = 1/wheel_radius*np.array([
            [-d*np.sin(np.deg2rad(240-150)), np.cos(np.deg2rad(240)), np.sin(np.deg2rad(240))],
            [-d*np.sin(np.deg2rad(0-270)),   np.cos(np.deg2rad(0)),   np.sin(np.deg2rad(0))],
            [-d*np.sin(np.deg2rad(120-30)),  np.cos(np.deg2rad(120)),  np.sin(np.deg2rad(120))]
        ])

    # ---------- lifecycle ----------
    def open(self) -> None:
        self.logger.info(f"Opening serial port {self.port} @ {self.baudrate}…")
        self._ser = serial.Serial(self.port, self.baudrate, timeout=self.timeout)
        
        if self.dtr_reset:
            try:
                self._ser.dtr = False
                time.sleep(0.05)
                self._ser.dtr = True
                time.sleep(0.25)  # small settle time
            except Exception:
                pass
        self._ser.reset_input_buffer()
        self._ser.reset_output_buffer()
        self.logger.info("Serial ready.")

    def close(self) -> None:
        self.logger.info("Shutting down…")
        if self._ser and self._ser.is_open:
            try:
                self._ser.flush()
            finally:
                self._ser.close()
        self._ser = None

    # ---------- command I/O ----------
    @staticmethod
    def _clip_int16(v: int) -> int:
        return max(-32768, min(32767, int(v)))

    @staticmethod
    def _clip_uint16(v: int) -> int:
        return max(0, min(65535, int(v)))

    def send_command(
        self,
        speed1: int,
        speed2: int,
        speed3: int,
        thrower_speed_percent: int = 0,
        servo1: int = 1500,
        servo2: int = 1500,
        disable_failsafe: int = 0,
    ) -> None:
        """Pack and send one command frame."""
        if not (self._ser and self._ser.is_open):
            raise RuntimeError("Serial port not open. Call .open() first.")

        # Apply polarity to wheel speeds
        s1 = self._clip_int16(self.polarity * speed1)
        s2 = self._clip_int16(self.polarity * speed2)
        s3 = self._clip_int16(self.polarity * speed3)

        thrower_speed = 48 + int(thrower_speed_percent/100 * (2047-48))  # scale 0-100% to 0-65535
        t  = self._clip_uint16(thrower_speed)
        sv1 = self._clip_uint16(servo1)
        sv2 = self._clip_uint16(servo2)
        df  = 1 if disable_failsafe == 1 else 0  # exactly 1 disables, else enables

        frame = struct.pack(self.PACK_FMT, s1, s2, s3, t, sv1, sv2, df, self.DELIMITER)
        self._ser.write(frame)
        # Optional: self._ser.flush()  # uncomment if you need blocking send

        self.logger.debug(
            "TX frame: s1=%d s2=%d s3=%d thrower=%d servo1=%d servo2=%d df=%d delim=0x%04X",
            s1, s2, s3, t, sv1, sv2, df, self.DELIMITER
        )

    # ---------- high-level API ----------
    def move(self, x_speed: float, y_speed: float, rot_speed: float) -> None:
        """
        Example shim: convert abstract (x,y,rot) into 3 wheel speeds.
        If you already have wheel speeds, call send_command(...) directly.

        This uses a simple placeholder mapping; adjust to your kinematics.
        """
        v = np.array([rot_speed, x_speed, y_speed])  # desired robot velocity
        wheel_speeds = self.H @ v  # matrix multiply to get wheel speeds
        s1, s2, s3 = (int(ws) for ws in wheel_speeds)
        # No servos/thrower; failsafe enabled by default (0)
        self.send_command(s1, s2, s3, thrower_speed_percent=0, servo1=1500, servo2=1500, disable_failsafe=0)
