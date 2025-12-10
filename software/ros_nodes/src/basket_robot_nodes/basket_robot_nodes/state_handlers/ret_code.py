from enum import IntEnum


class RetCode(IntEnum):
    ERROR = -2
    SUCCESS = 0
    DOING = 1
    BALL_LOST = 2
    BASKET_LOST = 3
    ROBOT_STUCK = 4
    TIMEOUT = 10
