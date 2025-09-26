import select
import sys
import termios
import tty
from typing import Any, List


def get_key(settings: List[Any]) -> str:
    """
    Inputs:
        settings: termios settings to restore at the end

    Get a single keypress from stdin, with a timeout of 0.1s.
    Returns an empty string if no key was pressed within the timeout.
    """
    tty.setraw(sys.stdin.fileno())
    rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
    key = sys.stdin.read(1) if rlist else ""
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key


def print_status(vy: float, vx: float, wz: float, thrower_percent: float) -> None:
    """
    Inputs:
        vy: forward/backward speed
        vx: right/left speed
        wz: yaw (rotation) speed
        thrower_percent: thrower speed in percent (0..100)

    Print the current control status.
    """
    print(f"currently:\tvy {vy:.2f}\tvx {vx:.2f}\twz {wz:.2f}\tthrower {thrower_percent:.0f}%")


def make_simple_profile(output_val: float, input_val: float, slop: float) -> float:
    """
    Inputs:
        output_val: current value
        input_val: target value
        slop: maximum change allowed (per call)

    Smoothly change output_val towards input_val by at most slop.
    """
    if input_val > output_val:
        return min(input_val, output_val + slop)
    if input_val < output_val:
        return max(input_val, output_val - slop)
    return input_val


def constrain(val: float, low: float, high: float) -> float:
    """
    Inputs:
        val: value to constrain
        low: minimum allowed value
        high: maximum allowed value

    Constrain val to be within the range [low, high].
    """
    return max(low, min(high, val))


def check_lin_limit(v: float, max_lin_vel: float) -> float:
    """
    Inputs:
        v: linear velocity to check

    Constrain linear velocity v to be within [-MAX_LIN_VEL, MAX_LIN_VEL].
    """
    return constrain(v, -max_lin_vel, max_lin_vel)


def check_ang_limit(w: float, max_ang_vel: float) -> float:
    """
    Inputs:
        w: angular velocity to check

    Constrain angular velocity w to be within [-MAX_ANG_VEL, MAX_ANG_VEL].
    """
    return constrain(w, -max_ang_vel, max_ang_vel)
