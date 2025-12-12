from typing import Final


class Parameters:
    # 1. parameters for odometry-based movement handlers
    BASE_TURN_WARMUP_ENABLED: Final[bool] = True
    BASE_MOVE_Y_WARMUP_ENABLED: Final[bool] = True
    BASE_WARMUP_RAMP_DURATION: Final[float] = 0.5  # seconds
    BASE_DISCRETE_WARMUP_RAMP_DURATION: Final[float] = 0.25  # seconds
    BASE_DISCRETE_TURN_SUB_ANGLE_DEG: Final[float] = 30.0  # degrees
    BASE_CONTINUOUS_TURN_SPEED: Final[float] = 1.5  # rad/s
    BASE_DISCRETE_TURN_SPEED: Final[float] = 5.0  # rad/s
    BASE_DISCRETE_TURN_STOP_DURATION: Final[float] = 0.5  # seconds
    BASE_MOVE_Y_MAX_SPEED: Final[float] = 1.5  # m/s
    BASE_MOVE_Y_DIS_THRESHOLD_MM: Final[float] = 20.0  # mm
    # PID parameters: [Kp, Ki, Kd] for forward, sideway, angular movements
    BASE_PID_ANGULAR: Final[list[float]] = [20.0, 0.0, 0.01]

    # 2. parameters for manipulation handlers
    MANI_ALIGN_WARMUP_ENABLED: Final[bool] = True
    MANI_WARMUP_RAMP_DURATION: Final[float] = 0.5  # seconds
    ## 2.1. Maximum speeds for manipulation tasks
    MANI_MAX_ALIGN_LINEAR_SPEED: Final[float] = 1.0  # m/s
    MANI_MAX_ALIGN_ANGULAR_SPEED: Final[float] = 1.0  # rad/s
    MANI_MAX_ALIGN_WBALL_LINEAR_SPEED: Final[float] = 1.5  # m/s
    MANI_MAX_ALIGN_WBALL_ANGULAR_SPEED: Final[float] = 2.0  # rad/s
    MANI_MAX_THROW_LINEAR_SPEED: Final[float] = 1.0  # m/s
    MANI_MAX_THROW_ANGULAR_SPEED: Final[float] = 1.0  # rad/s
    ## 2.2. PID parameters for manipulation tasks
    MANI_PID_LINEAR_ALIGN: Final[list[float]] = [2.0, 0.001, 0.001]  # [Kp, Ki, Kd]
    MANI_PID_ANGULAR_ALIGN: Final[list[float]] = [1.0, 0.0, 0.0]  # [Kp, Ki, Kd]
    MANI_PID_ANGULAR_ALIGN_BASKET: Final[list[float]] = [20.0, 0.01, 0.01]  # [Kp, Ki, Kd]
    MANI_PID_LINEAR_THROW_BALL: Final[list[float]] = [0.5, 0.0, 0.05]  # [Kp, Ki, Kd]
    MANI_PID_ANGULAR_THROW_BALL: Final[list[float]] = [1.0, 0.0, 0.1]  # [Kp, Ki, Kd]
    ## 2.3. Specifict thresholds for aligning with ball
    MANI_ALIGN_BALL_LOOKAHEAD_DIS_MM: Final[float] = 200.0  # mm
    MANI_MAX_CONSECUTIVE_FRAMES_NO_BALL: Final[int] = 5
    MANI_BALL_ALIGN_DIS_THRESHOLD_MM: Final[float] = 20.0  # mm
    MANI_BALL_ALIGN_ANGLE_THRESHOLD_DEG: Final[float] = 3.0  # degrees
    ## 2.5. Specific thresholds for aligning basket
    MANI_MAX_CONSECUTIVE_FRAMES_NO_BASKET: Final[int] = 5
    MANI_BASKET_ALIGN_ANGLE_THRESHOLD_DEG: Final[float] = 0.5  # degrees
    MANI_STORED_BASKET_TIMEOUT: Final[float] = 15.0  # seconds
    MANI_SEARCH_BASKET_ANGULAR_SPEED: Final[float] = 1.25  # rad/s
    MANI_SEARCH_BASKET_MAX_ANGULAR_SPEED: Final[float] = 1.25  # rad/s
    MANI_SEARCH_BASKET_NUM_CONSECUTIVE_VALID_FRAMES: Final[int] = 10
    ## 2.4. Specific thresholds for grabbing ball
    MANI_GRAB_BALL_Y_SPEED: Final[float] = 0.2  # m/s
    MANI_GRAB_BALL_SERVO_SPEED: Final[int] = 3000  # servo speed for grabbing
    ## 2.5. Specific thresholds for aligning basket with ball

    ## 2.6. Specific parameters for throwing ball
    MANI_THROW_BALL_SERVO_SPEED: Final[int] = 4000  # servo speed for throwing

    ## 2.7. Specific parameters for aligning to basket in advanced mode
    MANI_ALIGN_BASKET_ADV_VALID_DISTS_MM: Final[tuple[float, float]] = (
        1500,
        3000,
    )  # min, max valid distances in mm
    MANI_ALIGN_BASKET_ADV_PREFERRED_DIST_MM: Final[
        float
    ] = 2300.0  # preferred distance to basket in mm
    # mm offset along x-axis from marker center to basket center
    MANI_ALIGN_BASKET_ADV_MARKER_OFFSET_X_MM = 230
    MANI_ALIGN_BASKET_ADV_MAX_LINEAR_SPEED: Final[float] = 1.5  # m/s
    MANI_ALIGN_BASKET_ADV_MAX_ANGULAR_SPEED: Final[float] = 1.5  # rad/s
    MANI_ALIGN_BASKET_ADV_DIS_THRESHOLD_MM: Final[float] = 50.0  # mm
    # parameters for main state machine
    # TODO: add more parameters if needed
    MAIN_TURNING_DEGREE: Final[float] = 340.0  # degrees to turn when searching for ball
