from typing import Any, Dict, List, Tuple, Union

import yaml
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def _as_float_list(v: List[Union[float, int]]) -> List[float]:
    # Accept [240, 150] or [240.0, 150.0] → [240, 150]
    if not isinstance(v, list):
        raise ValueError(f"Expected a list, got {type(v)}")
    if not all(isinstance(x, (float, int)) for x in v):
        raise ValueError("All elements in the list must be float or int")
    return [float(x) for x in list(v)]


def _load_params_from_yaml(param_file_path: str) -> Tuple[Dict[str, Any], dict[str, str]]:
    with open(param_file_path, "r") as f:
        cfg = yaml.safe_load(f) or {}

    rcfg = cfg.get("shared_robot_setting") or {}
    if not rcfg:
        raise RuntimeError(f"'shared_robot_setting' missing in {param_file_path}")

    log_cfg = cfg.get("logging", {})
    log_level_dict: dict[str, str] = {}
    for k in log_cfg.keys():
        log_level_dict[k] = str(log_cfg.get(k, "INFO")).lower()

    # Map structured YAML -> flat ROS params in "shared_robot_setting"
    try:
        params = {
            "hwid": rcfg["hwid"],
            "cmd_fmt": rcfg["cmd_fmt"],
            "fbk_fmt": rcfg["fbk_fmt"],
            "delimiter": rcfg["delimiter"],
            "wheel_radius": rcfg["wheel_radius"],
            "center_to_wheel_dis": rcfg["center_to_wheel_dis"],
            "motor_01": _as_float_list(rcfg["motor_01"]),
            "motor_02": _as_float_list(rcfg["motor_02"]),
            "motor_03": _as_float_list(rcfg["motor_03"]),
            "gear_ratio": rcfg["gear_ratio"],
            "encoder_resolution": rcfg["encoder_resolution"],
            "pid_control_freq": rcfg["pid_control_freq"],
            "max_rot_speed": rcfg["max_rot_speed"],
            "max_xy_speed": rcfg["max_xy_speed"],
            "baudrate": rcfg["baudrate"],
            "polarity": rcfg["polarity"],
            "serial_timeout": rcfg["serial_timeout"],
            "port": rcfg["port"],
        }
    except (KeyError, ValueError) as e:
        raise Exception(f"Missing required parameter in {param_file_path}: {e}")
    return params, log_level_dict


def _launch_setup(context: Any, *args: Any, **kwargs: Any) -> List[Node]:
    # Resolve arguments at runtime
    # I want to read param file at params/setting.yaml
    param_file = LaunchConfiguration("param_file").perform(context)

    params, log_level_dict = _load_params_from_yaml(param_file)

    # Mainboard controller node (matches your console_scripts entry: start_mainboard_controller)
    nodes = [
        Node(
            package="basket_robot_nodes",
            executable="start_mainboard_controller",
            name="mainboard_controller_node",
            parameters=[params],
            arguments=[
                "--ros-args",
                "--log-level",
                log_level_dict.get("mainboard_controller_node", "info"),
            ],
            output="screen",
        ),
        Node(
            package="basket_robot_nodes",
            executable="start_odometry",
            name="odometry_node",
            parameters=[params],
            arguments=["--ros-args", "--log-level", log_level_dict.get("odometry_node", "info")],
            output="screen",
        ),
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
