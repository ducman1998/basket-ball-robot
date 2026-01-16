from typing import Final, Tuple


class Parameters:
    # 1. parameters for odometry-based movement handlers
    BASE_TURN_WARMUP_ENABLED: Final[bool] = True
    BASE_MOVE_Y_WARMUP_ENABLED: Final[bool] = True
    BASE_WARMUP_RAMP_DURATION: Final[float] = 0.1  # seconds
    BASE_DISCRETE_WARMUP_RAMP_DURATION: Final[float] = 0.15  # seconds
    BASE_DISCRETE_TURN_SUB_ANGLE_DEG: Final[float] = 75.0  # degrees
    BASE_CONTINUOUS_TURN_SPEED: Final[float] = 1.5  # rad/s
    BASE_DISCRETE_TURN_SPEED: Final[float] = 8.0  # rad/s
    BASE_DISCRETE_TURN_STOP_DURATION: Final[float] = 0.25  # seconds
    BASE_MOVE_Y_MAX_SPEED: Final[float] = 2.0  # m/s
    BASE_MOVE_Y_DIS_THRESHOLD_MM: Final[float] = 20.0  # mm
    # PID parameters: [Kp, Ki, Kd] for forward, sideway, angular movements
    BASE_PID_ANGULAR: Final[list[float]] = [20.0, 0.0, 0.01]

    # 2. parameters for manipulation handlers
    MANI_ALIGN_WARMUP_ENABLED: Final[bool] = True
    MANI_WARMUP_RAMP_DURATION: Final[float] = 0.25  # seconds
    ## 2.1. Maximum speeds for manipulation tasks
    MANI_MAX_ALIGN_LINEAR_SPEED: Final[float] = 1.5  # m/s
    MANI_MAX_ALIGN_ANGULAR_SPEED: Final[float] = 3.0  # rad/s
    ## 2.2. PID parameters for manipulation tasks
    MANI_PID_LINEAR_ALIGN: Final[list[float]] = [2.0, 0.0, 0.1]  # [Kp, Ki, Kd]
    MANI_PID_ANGULAR_ALIGN: Final[list[float]] = [12.0, 0.0, 0.1]  # [Kp, Ki, Kd]
    MANI_PID_ANGULAR_ALIGN_BASKET: Final[list[float]] = [10.0, 0.0, 0.01]  # [Kp, Ki, Kd]
    MANI_PID_LINEAR_ALIGN_BASKET_ADV: Final[list[float]] = [1.5, 0.0, 0.05]  # [Kp, Ki, Kd]
    MANI_PID_ANGULAR_ALIGN_BASKET_ADV: Final[list[float]] = [10.0, 0.0, 0.1]  # [Kp, Ki, Kd]
    MANI_PID_LINEAR_THROW_BALL: Final[list[float]] = [0.5, 0.0, 0.05]  # [Kp, Ki, Kd]
    MANI_PID_ANGULAR_THROW_BALL: Final[list[float]] = [1.0, 0.0, 0.1]  # [Kp, Ki, Kd]
    ## 2.3. Specifict thresholds for aligning with ball
    MANI_ALIGN_BALL_LOOKAHEAD_DIS_MM: Final[float] = 200.0  # mm
    MANI_MAX_CONSECUTIVE_FRAMES_NO_BALL: Final[int] = 5
    MANI_BALL_ALIGN_DIS_THRESHOLD_MM: Final[float] = 60.0  # mm
    MANI_BALL_ALIGN_ANGLE_THRESHOLD_DEG: Final[float] = 5.0  # degrees
    MANI_BALL_ALIGN_ENABLED_DIST_PID_MM: Final[float] = 400.0  # mm
    ## 2.4. Specific thresholds for pre-aligning with basket
    MANI_PRE_ALIGN_BASKET_ANGULAR_SPEED: Final[float] = 8.0  # rad/s
    ## 2.5. Specific thresholds for aligning basket
    MANI_MAX_CONSECUTIVE_FRAMES_NO_BASKET: Final[int] = 5
    MANI_BASKET_ALIGN_ANGLE_THRESHOLD_DEG: Final[float] = 0.5  # degrees
    MANI_STORED_BASKET_TIMEOUT: Final[float] = 30.0  # seconds
    MANI_SEARCH_BASKET_ANGULAR_SPEEDS: Final[Tuple[float, float]] = (2.5, 6.0)  # rad/s
    MANI_SEARCH_BASKET_MAX_ANGULAR_SPEED: Final[float] = 6.0  # rad/s
    MANI_SEARCH_BASKET_NUM_CONSECUTIVE_VALID_FRAMES: Final[int] = 5
    MANI_SEARCH_BASKET_HIGH_SPEED_MAX_TURNING_DEG: Final[float] = 360.0  # degrees
    # slight forward movement when throwing to help accuracy
    MAIN_BASKET_ALIGN_FINE_GRAINED_THRESHOLD_DEG: Final[float] = 5.0  # degrees
    MAIN_BASKET_ALIGN_Y_SPEED: Final[float] = 0.05  # m/s
    ## 2.5. Specific thresholds for grabbing ball
    MANI_GRAB_BALL_Y_SPEED: Final[float] = 0.5  # m/s
    MANI_GRAB_BALL_SERVO_SPEED: Final[int] = 3000  # servo speed for grabbing
    ## 2.6. Specific parameters for throwing ball
    MANI_THROW_BALL_SERVO_SPEED: Final[int] = 3000  # servo speed for throwing
    MAIN_THROW_BALL_OFFSET_PERCENT: Final[float] = 0.0  # percent added to base thrower power
    ## 2.7. Specific parameters for aligning to basket in advanced mode
    MANI_ALIGN_BASKET_ADV_VALID_DISTS_MM: Final[tuple[float, float]] = (
        1800,
        3200,
    )  # min, max valid distances in mm
    # mm offset along x-axis from marker center to basket center
    MANI_ALIGN_BASKET_ADV_MARKER_OFFSET_X_MM = 230
    MANI_ALIGN_BASKET_ADV_MAX_LINEAR_SPEED: Final[float] = 2.0  # m/s
    MANI_ALIGN_BASKET_ADV_MAX_ANGULAR_SPEED: Final[float] = 8.0  # rad/s
    MANI_ALIGN_BASKET_ADV_DIS_THRESHOLD_MM: Final[float] = 500.0  # mm
    MANI_ALIGN_BASKET_ADV_DIS_ODOM_THRESHOLD_MM: Final[float] = 150.0  # mm
    MANI_ALIGN_BASKET_ADV_ENABLED_DIST_PID_MM: Final[float] = 200.0  # m
    # 2.8. Specific parameters for turning around basket
    MANI_TURN_AROUND_BASKET_MAX_LINEAR_SPEED: Final[float] = 1.5  # m/s
    MANI_TURN_AROUND_BASKET_MAX_ANGULAR_SPEED: Final[float] = 1.0  # rad/s
    # mm away from basket when turning around
    MANI_TURN_AROUND_BASKET_DISTANCE_MM: Final[float] = 800.0

    # 3. parameters for main state machine
    MAIN_TURNING_DEGREE: Final[float] = 300.0  # degrees to turn when searching for ball
    MAIN_BASE_THROWER_PERCENT: Final[float] = 20.0  # base power percent for throwing ball
    MAIN_TIMEOUT_SEARCH_BALL_TURN_DISC: Final[float] = 2.0  # seconds before re-searching for ball
    MAIN_TIMEOUT_SEARCH_BALL_MOVE_FW: Final[float] = 2.0  # seconds before re-searching for ball
    MAIN_TIMEOUT_ALIGN_BALL: Final[float] = 3.0  # seconds before re-searching for ball
    MAIN_TIMEOUT_GRAB_BALL: Final[float] = 1.5  # seconds before re-searching for ball
    MAIN_TIMEOUT_ALIGN_BASKET: Final[float] = 3.0  # seconds before re-searching for basket
    # seconds before re-searching for basket in advanced alignment mode
    # marker-based alignment = timeout - refine angle timeout
    MAIN_TIMEOUT_ALIGN_BASKET_ADVANCED_TOTAL: Final[float] = 4.0  # timeout
    # seconds for removing angle error in advanced basket alignment
    MAIN_TIMEOUT_ALIGN_BASKET_ADVANCED_REFINE_ANGLE: Final[float] = 2.0  # refine angle timeout
    MAIN_TIMEOUT_THROW_BALL: Final[float] = 1.75  # seconds before re-trying to throw ball
    MAIN_TIMEOUT_CLEAR_STUCK_BALL: Final[float] = 1.5  # seconds to clear stuck ball
    MAIN_TURNING_ANGULAR_SPEED_TO_CANDIDATE_BALL: Final[float] = 8.0  # rad/s
    MAIN_TIMEOUT_TURN_TO_CANDIDATE_BALL: Final[float] = 0.5  # seconds before re-searching for ball
    MAIN_TIMEOUT_TURN_AROUND_BASKET: Final[float] = 1.5  # timeout for turning to avoid collision
    MAIN_TIMEOUT_PRE_ALIGN_BASKET: Final[float] = 0.4  # seconds for pre-aligning to basket
