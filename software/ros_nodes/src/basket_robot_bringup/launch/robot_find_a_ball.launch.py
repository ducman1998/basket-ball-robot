from typing import Any, Dict, List, Tuple

import yaml
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def _load_params_from_yaml(param_file_path: str) -> Tuple[Dict[str, Any], dict[str, str]]:
    with open(param_file_path, "r") as f:
        cfg = yaml.safe_load(f) or {}

    log_cfg = cfg.get("logging", {})
    log_level_dict: dict[str, str] = {}
    for k in log_cfg.keys():
        log_level_dict[k] = str(log_cfg.get(k, "INFO")).lower()

    try:
        params = {
            "max_rot_speed": cfg["shared_robot_setting"]["max_rot_speed"],
            "max_xy_speed": cfg["shared_robot_setting"]["max_xy_speed"],
            "search_ball_rot_speed": cfg["game_logic_controller"]["search_ball_rot_speed"],
            "kp_xy": cfg["game_logic_controller"]["kp_xy"],
            "kd_xy": cfg["game_logic_controller"]["kd_xy"],
            "kp_rot": cfg["game_logic_controller"]["kp_rot"],
            "kd_rot": cfg["game_logic_controller"]["kd_rot"],
        }
    except (KeyError, ValueError) as e:
        raise RuntimeError(f"Error in 'shared_robot_setting' in {param_file_path}: {e}")

    return params, log_level_dict


def _launch_setup(context: Any, *args: Any, **kwargs: Any) -> List[Node]:
    # Resolve arguments at runtime
    # I want to read param file at params/setting.yaml
    param_file = LaunchConfiguration("param_file").perform(context)

    common_params, log_level_dict = _load_params_from_yaml(param_file)

    # Launch the game logic controller node
    nodes = [
        Node(
            package="basket_robot_nodes",
            executable="start_game_logic_controller",
            name="game_logic_controller",
            parameters=[common_params],
            arguments=[
                "--ros-args",
                "--log-level",
                log_level_dict.get("game_logic_controller", "info"),
            ],
            output="screen",
        )
    ]

    return nodes


def generate_launch_description() -> LaunchDescription:
    default_param_file = PathJoinSubstitution(
        [FindPackageShare("basket_robot_bringup"), "params", "setting.yaml"]
    )

    param_file_arg = DeclareLaunchArgument(
        "param_file",
        default_value=default_param_file,
        description="Path to the YAML parameter file.",
    )

    return LaunchDescription(
        [
            param_file_arg,
            OpaqueFunction(function=_launch_setup),
        ]
    )
