# Header stamp: time of validity, usually now()
# Header frame_id: the reference coordinate frame (e.g. "base_link")
std_msgs/Header header

# Desired twist of the robot base in that frame
geometry_msgs/Twist twist

# Thrower motor duty cycle / percent (0–100)
uint8 thrower_percent   # 0–100 (%)
