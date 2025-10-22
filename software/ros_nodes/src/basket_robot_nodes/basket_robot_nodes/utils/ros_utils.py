from rclpy.logging import LoggingSeverity
from rclpy.node import Node


def log_initialized_parameters(node: Node) -> None:
    """Fetches and logs all parameters initialized for this node."""

    # Using an empty string prefix ('') retrieves ALL initialized parameters
    all_params = node.get_parameters_by_prefix("")

    if not all_params:
        node.get_logger().warn("No parameters were initialized for this node.")
        return

    node.get_logger().info("--- Initialized parameters loaded from YAML file---")

    for name, param in all_params.items():
        # The value is accessed via the 'value' property of the rclpy.Parameter object
        node.get_logger().info(f"  {name}: {param.value}")

    node.get_logger().info("------------------------------")
    return None


def parse_log_level(level_str: str) -> LoggingSeverity:
    """Map string log level to ROS 2 LoggingSeverity enum."""
    level_str = level_str.lower()
    mapping = {
        "debug": LoggingSeverity.DEBUG,
        "info": LoggingSeverity.INFO,
        "warn": LoggingSeverity.WARN,
        "warning": LoggingSeverity.WARN,
        "error": LoggingSeverity.ERROR,
        "fatal": LoggingSeverity.FATAL,
    }
    return mapping.get(level_str, LoggingSeverity.INFO)
