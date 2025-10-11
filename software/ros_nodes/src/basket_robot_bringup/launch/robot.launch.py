from typing import Any, Dict, List, Literal, Tuple, Union

import yaml
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def _as_list(
    v: List[Union[float, int]], type: Union[Literal["int"], Literal["float"]]
) -> List[Union[int, float]]:
    """Helper to ensure a parameter is a list of the specified type (int or float)."""
    # Accept [240, 150] or [240.0, 150.0] → int: [240, 150]/float: [240.0, 150.0]
    if type not in ("int", "float"):
        raise ValueError(f"Type must be 'int' or 'float', got {type}")
    if not isinstance(v, list):
        raise ValueError(f"Expected a list, got {type(v)}")
    if not all(isinstance(x, (float, int)) for x in v):
        raise ValueError("All elements in the list must be float or int")
    if type == "int":
        return [int(x) for x in v]
    else:  # type == 'float'
        return [float(x) for x in v]


def _load_params_from_yaml(param_file_path: str, key: str) -> Tuple[Dict[str, Any], dict[str, str]]:
    with open(param_file_path, "r") as f:
        cfg = yaml.safe_load(f) or {}

    log_cfg = cfg.get("logging", {})
    log_level_dict: dict[str, str] = {}
    for k in log_cfg.keys():
        log_level_dict[k] = str(log_cfg.get(k, "INFO")).lower()

    rcfg = cfg.get(key) or {}
    if not rcfg:
        raise RuntimeError(f"'{key}' missing in {param_file_path}")
    if not isinstance(rcfg, dict):
        raise RuntimeError(f"'{key}' in {param_file_path} is not a dictionary")
    if key == "shared_robot_setting":
        # Map structured YAML -> flat ROS params in "shared_robot_setting"
        try:
            params = {
                "hwid": rcfg["hwid"],
                "cmd_fmt": rcfg["cmd_fmt"],
                "fbk_fmt": rcfg["fbk_fmt"],
                "delimiter": rcfg["delimiter"],
                "wheel_radius": rcfg["wheel_radius"],
                "center_to_wheel_dis": rcfg["center_to_wheel_dis"],
                "motor_01": _as_list(rcfg["motor_01"], "float"),
                "motor_02": _as_list(rcfg["motor_02"], "float"),
                "motor_03": _as_list(rcfg["motor_03"], "float"),
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
            raise RuntimeError(f"Error in 'shared_robot_setting' in {param_file_path}: {e}")
    elif key == "image_processor_setting":
        try:
            params = {
                "ref_colors_flat": _as_list(rcfg["ref_colors_flat"], "int"),
                "ref_court_color": _as_list(rcfg["ref_court_color"], "int"),
                "resolution": _as_list(rcfg["resolution"], "int"),
                "fps": rcfg["fps"],
                "enable_depth": rcfg["enable_depth"],
                "publish_viz_image": rcfg["publish_viz_image"],
                "publish_viz_fps": rcfg["publish_viz_fps"],
                "publish_viz_resize": rcfg["publish_viz_resize"],
            }
        except (KeyError, ValueError) as e:
            raise RuntimeError(f"Error in 'image_processor_setting' in {param_file_path}: {e}")
    return params, log_level_dict


def _launch_setup(context: Any, *args: Any, **kwargs: Any) -> List[Node]:
    # Resolve arguments at runtime
    # I want to read param file at params/setting.yaml
    param_file = LaunchConfiguration("param_file").perform(context)

    common_params, log_level_dict = _load_params_from_yaml(param_file, "shared_robot_setting")
    image_processor_params, _ = _load_params_from_yaml(param_file, "image_processor_setting")

    # Mainboard controller node (matches your console_scripts entry: start_mainboard_controller)
    nodes = [
        Node(
            package="basket_robot_nodes",
            executable="start_mainboard_controller",
            name="mainboard_controller_node",
            parameters=[common_params],
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
            parameters=[common_params],
            arguments=["--ros-args", "--log-level", log_level_dict.get("odometry_node", "info")],
            output="screen",
        ),
        Node(
            package="basket_robot_nodes",
            executable="start_image_processor",
            name="image_processor",
            parameters=[image_processor_params],
            arguments=["--ros-args", "--log-level", log_level_dict.get("image_processor", "info")],
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
