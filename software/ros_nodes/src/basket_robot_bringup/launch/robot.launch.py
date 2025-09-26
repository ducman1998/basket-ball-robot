#!/usr/bin/env python3
from typing import Any, Dict, List, Tuple, Union

import yaml
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def _as_float_list(v: List[Union[float, int]]) -> List[float]:
    # Accept [240, 150] or [240.0, 150.0] → [240.0, 150.0]
    return [float(x) for x in list(v)]


def _load_params_from_yaml(param_file_path: str) -> Tuple[Dict[str, Any], str]:
    with open(param_file_path, "r") as f:
        cfg = yaml.safe_load(f) or {}

    rcfg = cfg.get("robot_configuration") or {}
    if not rcfg:
        raise RuntimeError(f"robot_configuration missing in {param_file_path}")

    log_cfg = cfg.get("logging", {})
    log_level = str(log_cfg.get("log_level", "INFO")).lower()

    # Map structured YAML → flat ROS params that the basket_robot_nodes declares
    params = {
        "wheel_radius": float(rcfg.get("wheel_radius", 0.035)),
        "c2w_dis": float(rcfg.get("center_to_wheel_dis", 0.1295)),
        "motor_01": _as_float_list(rcfg.get("motor_01", [240.0, 150.0])),
        "motor_02": _as_float_list(rcfg.get("motor_02", [0.0, 270.0])),
        "motor_03": _as_float_list(rcfg.get("motor_03", [120.0, 30.0])),
        "gear_ratio": float(rcfg.get("gear_ratio", 18.75)),
        "encoder_resolution": int(rcfg.get("encoder_resolution", 64)),
        "pid_control_freq": int(rcfg.get("pid_control_freq", 100)),
        "max_rot_speed": float(rcfg.get("max_rot_speed", 2.0)),
        "max_xy_speed": float(rcfg.get("max_xy_speed", 2.5)),
        "hwid": str(rcfg.get("hwid", "USB VID:PID=0483:5740")),
    }
    return params, log_level


def _launch_setup(context: Any, *args: Any, **kwargs: Any) -> List[Node]:
    # Resolve arguments at runtime
    # I want to read param file at params/setting.yaml
    param_file = LaunchConfiguration("param_file").perform(context)

    params, log_level = _load_params_from_yaml(param_file)

    # Mainboard controller node (matches your console_scripts entry: start_base_controller)
    nodes = [
        Node(
            package="basket_robot_nodes",
            executable="start_base_controller",
            name="mainboard_controller",
            parameters=[params],
            arguments=["--ros-args", "--log-level", log_level],
            output="screen",
        ),
        Node(
            package="basket_robot_nodes",
            executable="start_odometry",
            name="odometry_node",
            parameters=[params],
            arguments=["--ros-args", "--log-level", log_level],
            output="screen",
        ),
    ]

    return nodes


def generate_launch_description() -> LaunchDescription:
    default_param_file = PathJoinSubstitution(
        [FindPackageShare("basket_robot_bringup"), "params", "setting.yaml"]
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "param_file",
                default_value=default_param_file,
                description="Path to structured YAML (with robot_configuration).",
            ),
            OpaqueFunction(function=_launch_setup),
        ]
    )
