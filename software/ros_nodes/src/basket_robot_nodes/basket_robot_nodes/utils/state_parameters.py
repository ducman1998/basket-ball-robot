class StateParameters:
    MAX_ROT_SPEED = 2.0  # rad/s
    MAX_TRANS_SPEED = 1.0  # m/s
    THROWING_TRANS_SPEED = 0.4  # m/s
    SEARCH_BALL_ROT_SPEED = -1.25  # rad/s

    # linear & rotational PID parameters: [Kp, Ki, Kd]
    PID_LINEAR_COMMON = [1.5, 0.001, 0.1]
    PID_ANGULAR_COMMON = [2.0, 0.001, 0.1]
    PID_LINEAR_ALIGN_BASKET = [3.0, 0.000, 0.01]
    PID_ANGULAR_ALIGN_BASKET = [2.0, 0.000, 0.01]
    PID_LINEAR_THROW_BALL = [0.5, 0.0, 0.05]
    PID_ANGULAR_THROW_BALL = [1.0, 0.0, 0.1]
