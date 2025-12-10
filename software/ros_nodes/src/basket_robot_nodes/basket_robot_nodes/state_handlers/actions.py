from enum import IntEnum


class BaseAction(IntEnum):
    MOVE_FORWARD = 0
    MOVE_SIDEWAY = 1
    TURN_CONTINUOUS = 2
    TURN_DISCRETE = 3


class ManipulationAction(IntEnum):
    APPROACH_BALL = 0
    APPROACH_BASKET = 1
    GRAB_BALL = 2
    ALIGN_BASKET = 3
    THROW_BALL = 4
