from enum import IntEnum


class BaseAction(IntEnum):
    MOVE_XY = 0
    TURN_CONTINUOUS = 1
    TURN_DISCRETE = 2


class ManipulationAction(IntEnum):
    APPROACH_BALL = 0
    APPROACH_BASKET = 1
    GRAB_BALL = 2
    ALIGN_BASKET = 3
    THROW_BALL = 4
