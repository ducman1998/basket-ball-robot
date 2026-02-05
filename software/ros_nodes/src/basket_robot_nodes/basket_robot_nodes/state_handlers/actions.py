from enum import IntEnum


class BaseAction(IntEnum):
    MOVE_FORWARD = 0
    TURN_CONTINUOUS = 1
    TURN_DISCRETE = 2


class ManipulationAction(IntEnum):
    ALIGN_BALL = 0
    ALIGN_BASKET = 1
    GRAB_BALL = 2
    ALIGN_BASKET_WBALL = 3
    THROW_BALL = 4
    # based on markers to align the robot to the court center for better accuracy
    ALIGN_BASKET_ADVANCED = 5
    CLEAR_STUCK_BALL = 6
    TURN_AROUND_BASKET = 7
