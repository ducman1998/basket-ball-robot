from time import time
from typing import cast

import rclpy
from basket_robot_nodes.state_handlers.actions import BaseAction
from basket_robot_nodes.state_handlers.base_handler import BaseHandler
from basket_robot_nodes.state_handlers.ret_code import RetCode
from basket_robot_nodes.utils.base_game_logic import BaseGameLogicController
from basket_robot_nodes.utils.peripheral_manager import PeripheralManager

DEV_MODE = True
SAMPLING_RATE = 60  # Hz


class GameState:
    """Enum-like class for game states."""

    UNDEFINED = -1
    INIT = 0
    TAKE_ACTION = 1
    STOP = 2

    @staticmethod
    def get_name(state_value: int) -> str:
        state2name = {v: k for k, v in vars(GameState).items() if isinstance(v, int)}
        return state2name.get(state_value, "UNKNOWN")


class GameLogicController(BaseGameLogicController):
    def __init__(self) -> None:
        # Initialize the Game Logic Controller node
        super().__init__("game_logic_controller_node", dev_mode=DEV_MODE)

        # subscriptions to sensors, offering convenient access/control sensors & actuators
        self.periph_manager = PeripheralManager(self)
        self.base_handler = BaseHandler(self.periph_manager)
        # self.manipulation_handler = ManpulationHandler(self.periph_manager)

        self.game_timer = self.create_timer(1 / SAMPLING_RATE, self.game_logic_loop)

        self.pre_state = GameState.UNDEFINED
        self.cur_state = GameState.INIT
        self.timestamp = time()
        self.get_logger().info(f"Dev mode: {DEV_MODE}")

    def game_logic_loop(self) -> None:
        """Main game logic loop, called periodically by a timer."""
        start_time = time()
        self.print_current_state()

        # If game is not active (referee hasn't started or has stopped), don't execute game logic
        if not self.is_game_started and not DEV_MODE:
            # if we were playing and referee stopped us, ensure robot is stopped
            self.stop_robot()
            self.cur_state = GameState.INIT
            self.periph_manager.reset()
            return

        match self.cur_state:
            case GameState.INIT:
                self.periph_manager.stop_robot()  # trigger mainboard sends sensor data

                if self.periph_manager.is_ready():
                    self.periph_manager.set_target_basket_color(self.opponent_basket_color)
                    self.transition_to(GameState.TAKE_ACTION)
                    self.base_handler.initialize(
                        BaseAction.MOVE_FORWARD, distance=1000.0, timeout=10.0
                    )

            case GameState.TAKE_ACTION:
                ret = self.base_handler.move_robot_forward()
                if ret == RetCode.SUCCESS:
                    self.get_logger().info("Successfully moved forward 1 meter.")
                    self.transition_to(GameState.STOP)
                elif ret == RetCode.TIMEOUT:
                    self.get_logger().warn("Timeout while moving forward.")
                    self.transition_to(GameState.STOP)

            case GameState.STOP:
                self.periph_manager.stop_robot()

            case _:
                raise RuntimeError("Unknown game state!")

        end_time = time()
        self.get_logger().info(f"Game logic loop took {end_time - start_time:.4f} seconds.")

    # TODO: implement state transition handlers here
    def can_transition_to_move(self) -> bool:
        """Check if conditions are met to transition to MOVE state."""
        # Example condition: peripheral manager is ready and odometry is valid
        return cast(bool, self.periph_manager.is_ready())

    def transition_to(self, new_state: int) -> None:
        """Handle transition to a new state."""
        cur_state_name = GameState.get_name(self.cur_state)
        new_state_name = GameState.get_name(new_state)
        if new_state == self.cur_state:
            self.get_logger().info(f"Already in state {cur_state_name}, no transition needed.")
            return

        self.pre_state = self.cur_state
        self.cur_state = new_state
        self.get_logger().info(f"Transitioning from {cur_state_name} to {new_state_name}.")


def main() -> None:
    rclpy.init()
    node = GameLogicController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Stop referee client before destroying node
        if hasattr(node, "referee_client"):
            node.referee_client.stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
