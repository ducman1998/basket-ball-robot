from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    default_param_file = PathJoinSubstitution(
        [FindPackageShare("basket_robot_bringup"), "params", "setting.yaml"]
    )

    param_file_arg = DeclareLaunchArgument(
        "param_file",
        default_value=default_param_file,
        description="Path to the shared YAML parameter file.",
    )

    launch_cfg = [LaunchConfiguration("param_file")]

    # Set custom logger format
    set_logger_format = SetEnvironmentVariable(
        name="RCUTILS_CONSOLE_OUTPUT_FORMAT", value="[{severity}] {name}: {message}"
    )

    # Define your nodes
    nodes = [
        Node(
            package="basket_robot_nodes",
            executable="start_game_logic_controller",
            name="game_logic_controller_node",
            parameters=[launch_cfg],
            output="screen",
        )
    ]

    # Return the LaunchDescription object
    return LaunchDescription([param_file_arg, set_logger_format] + nodes)
