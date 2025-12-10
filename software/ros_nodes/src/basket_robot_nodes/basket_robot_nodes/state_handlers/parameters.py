from typing import Final


class Parameters:
    # 1. parameters for odometry-based movement handlers
    BASE_TURN_WARMUP_ENABLED: Final[bool] = True
    BASE_MOVE_XY_WARMUP_ENABLED: Final[bool] = True
    BASE_WARMUP_RAMP_DURATION: Final[float] = 0.5  # seconds
    BASE_DISCRETE_WARMUP_RAMP_DURATION: Final[float] = 0.25  # seconds
    BASE_DISCRETE_TURN_SUB_ANGLE_DEG: Final[float] = 30.0  # degrees
    BASE_CONTINUOUS_TURN_SPEED: Final[float] = 1.25  # rad/s
    BASE_DISCRETE_TURN_SPEED: Final[float] = 5.0  # rad/s
    BASE_DISCRETE_TURN_STOP_DURATION: Final[float] = 0.5  # seconds
    BASE_MOVE_XY_MAX_SPEED: Final[float] = 2.0  # m/s
    BASE_MOVE_XY_DIS_THRESHOLD_MM: Final[float] = 2.0  # mm
    # PID parameters: [Kp, Ki, Kd] for forward, sideway, angular movements
    BASE_PID_ANGULAR: Final[list[float]] = [20.0, 0.0, 0.01]

    # 2. parameters for manipulation handlers
    MANI_APPROACH_WARMUP_ENABLED: Final[bool] = True
    MANI_GRAB_WARMUP_ENABLED: Final[bool] = True
    MANI_ALIGN_WARMUP_ENABLED: Final[bool] = False
    MANI_WARMUP_RAMP_DURATION: Final[float] = 0.5  # seconds
    ## 2.1. Maximum speeds for manipulation tasks
    MANI_MAX_APPROACH_LINEAR_SPEED: Final[float] = 2  # m/s
    MANI_MAX_APPROACH_ANGULAR_SPEED: Final[float] = 3.0  # rad/s
    MANI_MAX_ALIGN_LINEAR_SPEED: Final[float] = 1.5  # m/s
    MANI_MAX_ALIGN_ANGULAR_SPEED: Final[float] = 2.0  # rad/s
    MANI_MAX_THROW_LINEAR_SPEED: Final[float] = 1.0  # m/s
    MANI_MAX_THROW_ANGULAR_SPEED: Final[float] = 1.0  # rad/s
    ## 2.2. PID parameters for manipulation tasks
    MANI_PID_LINEAR_APPROACH: Final[list[float]] = [1.5, 0.001, 0.1]  # [Kp, Ki, Kd]
    MANI_PID_ANGULAR_APPROACH: Final[list[float]] = [2.0, 0.001, 0.1]  # [Kp, Ki, Kd]
    MANI_PID_LINEAR_ALIGN_BASKET_COARSE: Final[list[float]] = [3.0, 0.000, 0.01]  # [Kp, Ki, Kd]
    MANI_PID_ANGULAR_ALIGN_BASKET_COARSE: Final[list[float]] = [2.0, 0.000, 0.01]  # [Kp, Ki, Kd]
    MANI_PID_LINEAR_ALIGN_BASKET_FINE: Final[list[float]] = [1.5, 0.000, 0.01]  # [Kp, Ki, Kd]
    MANI_PID_ANGULAR_ALIGN_BASKET_FINE: Final[list[float]] = [1.5, 0.000, 0.01]  # [Kp, Ki, Kd]
    MANI_PID_LINEAR_THROW_BALL: Final[list[float]] = [0.5, 0.0, 0.05]  # [Kp, Ki, Kd]
    MANI_PID_ANGULAR_THROW_BALL: Final[list[float]] = [1.0, 0.0, 0.1]  # [Kp, Ki, Kd]

    # # linear & rotational PID parameters: [Kp, Ki, Kd]
    # PID_LINEAR_COMMON = [1.5, 0.001, 0.1]
    # PID_ANGULAR_COMMON = [2.0, 0.001, 0.1]
    # PID_LINEAR_ALIGN_BASKET = [3.0, 0.000, 0.01]
    # PID_ANGULAR_ALIGN_BASKET = [2.0, 0.000, 0.01]
    # PID_LINEAR_THROW_BALL = [0.5, 0.0, 0.05]
    # PID_ANGULAR_THROW_BALL = [1.0, 0.0, 0.1]
