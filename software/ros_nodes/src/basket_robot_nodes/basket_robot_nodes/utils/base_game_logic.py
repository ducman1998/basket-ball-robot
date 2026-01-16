from typing import Literal, Optional

from basket_robot_nodes.utils.referee_client import RefereeClient
from basket_robot_nodes.utils.ros_utils import (
    int_descriptor,
    log_initialized_parameters,
    parse_log_level,
    str_descriptor,
)
from rclpy.node import Node


class BaseGameLogicController(Node):
    def __init__(self, node_name: str, dev_mode: bool = False) -> None:
        # Initialize the Game Logic Controller node
        super().__init__(node_name)
        # declare parameters
        self._declare_node_parameter()
        # read parameters
        self._read_node_parameters()
        # for checking: log all initialized parameters
        log_initialized_parameters(self)

        # Referee client state
        self.is_game_started = False  # True when referee sends START, False when STOP/
        # default opponent basket color
        self.target_basket_color = "n/a" if not dev_mode else "blue"

        # Initialize and start referee client
        self.referee_client = RefereeClient(
            robot_id=self.robot_id,
            referee_ip=self.referee_ip_address,
            referee_port=self.referee_port,
            on_signal=self.handle_referee_signals,
            logger=self.get_logger(),
        )
        self.referee_client.start()
        self.get_logger().info(f"Started referee client for robot ID: {self.robot_id}")

    def _declare_node_parameter(self) -> None:
        """Declare parameters with descriptors."""
        self.declare_parameter("referee_ip_address", descriptor=str_descriptor)
        self.declare_parameter("referee_port", descriptor=int_descriptor)
        self.declare_parameter("robot_id", descriptor=str_descriptor)
        self.declare_parameter("log_level", descriptor=str_descriptor)

    def _read_node_parameters(self) -> None:
        """Read parameters into class variables."""
        # Read all parameters
        self.referee_ip_address = (
            self.get_parameter("referee_ip_address").get_parameter_value().string_value
        )
        self.referee_port = self.get_parameter("referee_port").get_parameter_value().integer_value
        self.robot_id = self.get_parameter("robot_id").get_parameter_value().string_value
        log_level = self.get_parameter("log_level").get_parameter_value().string_value

        # Validate parameters
        if not self.referee_ip_address or not isinstance(self.referee_port, int):
            raise ValueError("Invalid referee IP address or port number.")

        # Set logging level
        self.get_logger().set_level(parse_log_level(log_level))
        self.get_logger().info(f"Set node {self.get_name()} log level to {log_level}.")

    def handle_referee_signals(
        self, signal: Literal["start", "stop"], basket: Optional[str]
    ) -> None:
        """Handle START/STOP signal from referee."""
        if signal == "start":
            if self.is_game_started:
                self.get_logger().warn("Received duplicate START signal from referee --> Ignored.")
            else:
                self.target_basket_color = basket if basket is not None else "n/a"
                self.is_game_started = True
        else:
            self.is_game_started = False

    def get_target_basket_color(self) -> str:
        """Get the opponent basket color."""
        return self.target_basket_color

    def get_opponent_basket_color(self) -> str:
        """Get the opponent basket color based on the target basket color."""
        if self.target_basket_color == "magenta":
            return "blue"
        else:  # blue
            return "magenta"
