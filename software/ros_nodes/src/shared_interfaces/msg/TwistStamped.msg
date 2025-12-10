# Header stamp: time of validity, usually now()
# Header frame_id: the reference coordinate frame (e.g. "base_link")
std_msgs/Header header

# Desired twist of the robot base in that frame
geometry_msgs/Twist twist

# Thrower motor duty cycle / percent (0–100)
float32 thrower_percent   # 0–100 (%)

# Servo rotation speed [0,20000] (0 = stopped)
uint16 servo_speed        # 0–20000 (units depend on servo)
