from enum import IntEnum


class RetCode(IntEnum):
    ERROR = -2
    SUCCESS = 0
    DOING = 1
    FAILED_BALL_LOST = 2
    FAILED_BASKET_LOST = 3
    fAILED_ROBOT_STUCK = 4
    TIMEOUT = 10
