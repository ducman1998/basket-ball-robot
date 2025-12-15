from typing import Union

import rclpy
from basket_robot_nodes.state_handlers.actions import BaseAction, ManipulationAction
from basket_robot_nodes.state_handlers.base_handler import BaseHandler
from basket_robot_nodes.state_handlers.manipulation_handler import ManpulationHandler
from basket_robot_nodes.state_handlers.parameters import Parameters
from basket_robot_nodes.state_handlers.ret_code import RetCode
from basket_robot_nodes.utils.base_game_logic import BaseGameLogicController
from basket_robot_nodes.utils.peripheral_manager import PeripheralManager

DEV_MODE = True
ENABLE_ADVANCED_BASKET_ALIGNMENT = True  # enable advanced basket alignment mode
SAMPLING_RATE = 60  # Hz


class GameState:
    """Enum-like class for game states."""

    UNDEFINED = -1
    INIT = 0
    SEARCH_BALL = 1
    ALIGN_BALL = 2
    GRAB_BALL = 3
    ALIGN_BASKET = 4
    ALIGN_BASKET_ADVANCED = 5
    THROW_BALL = 6
    CLEAR_STUCK_BALL = 7
    TURN_TO_CANDIDATE_BALL = 8
    TURN_AROUND_BASKET = 9

    @staticmethod
    def get_name(state_value: int) -> str:
        state2name = {v: k for k, v in vars(GameState).items() if isinstance(v, int)}
        return state2name.get(state_value, "UNKNOWN")


class SearchSubState:
    """Enum-like class for search ball sub-states."""

    UNDEFINED = -1
    TURN_DISCRETE = 0
    ALIGN_BASKET = 1
    MOVE_FORWARD = 2

    @staticmethod
    def get_name(state_value: int) -> str:
        state2name = {v: k for k, v in vars(SearchSubState).items() if isinstance(v, int)}
        return state2name.get(state_value, "UNKNOWN")


class GameLogicController(BaseGameLogicController):
    def __init__(self) -> None:
        # Initialize the Game Logic Controller node
        super().__init__("game_logic_controller_node", dev_mode=DEV_MODE)

        # subscriptions to sensors, offering convenient access/control sensors & actuators
        self.periph_manager = PeripheralManager(self)
        self.base_handler = BaseHandler(self.periph_manager)
        self.manipulation_handler = ManpulationHandler(self.periph_manager)

        self.game_timer = self.create_timer(1 / SAMPLING_RATE, self.game_logic_loop)

        self.pre_state = GameState.UNDEFINED
        self.cur_state = GameState.INIT
        self.pre_sub_state = SearchSubState.UNDEFINED
        self.cur_sub_state = SearchSubState.TURN_DISCRETE

        self.get_logger().info(f"Dev mode: {DEV_MODE}")

    def game_logic_loop(self) -> None:
        """Main game logic loop, called periodically by a timer."""
        self.print_current_state()

        # If game is not active (referee hasn't started or has stopped), don't execute game logic
        if not self.is_game_started and not DEV_MODE:
            # if we were playing and referee stopped us, ensure robot is stopped
            self.periph_manager.stop_robot()
            self.cur_state = GameState.INIT
            self.periph_manager.reset()
            return

        match self.cur_state:
            case GameState.INIT:
                self.periph_manager.stop_robot()  # trigger mainboard sends sensor data

                if self.periph_manager.is_ready():
                    self.periph_manager.set_target_basket_color(self.get_target_basket_color())
                    if self.periph_manager.is_ball_detected():
                        is_blocked, direction = self.periph_manager.is_ball_blocked_by_basket()
                        if is_blocked:
                            self.transition_to(
                                GameState.TURN_AROUND_BASKET, turning_basket_direction=direction
                            )
                        else:
                            self.transition_to(GameState.ALIGN_BALL)

                    if self.periph_manager.is_balls_not_detected_in_nframes(5):
                        # NOTE: SEARCH_BALL state mostly uses basic handlers in sub-state machine
                        self.transition_to(GameState.SEARCH_BALL)

            case GameState.SEARCH_BALL:
                # TODO: implement search ball sub state-machine
                if self.periph_manager.is_ball_grabbed():
                    self.transition_to(GameState.CLEAR_STUCK_BALL)

                if self.periph_manager.is_ball_detected():
                    is_blocked, direction = self.periph_manager.is_ball_blocked_by_basket()
                    if is_blocked:
                        self.transition_to(
                            GameState.TURN_AROUND_BASKET, turning_basket_direction=direction
                        )
                    else:
                        self.transition_to(GameState.ALIGN_BALL)
                    return  # exit early to avoid executing sub-state logic

                match self.cur_sub_state:
                    case SearchSubState.TURN_DISCRETE:
                        ret = self.base_handler.turn_robot_disc()
                        if ret == RetCode.SUCCESS or ret == RetCode.TIMEOUT:
                            self.sub_transition_to(
                                SearchSubState.ALIGN_BASKET, is_opponent_basket=False
                            )

                    case SearchSubState.ALIGN_BASKET:
                        ret = self.manipulation_handler.align_to_basket()
                        if ret == RetCode.SUCCESS or ret == RetCode.TIMEOUT:
                            basket_dis_mm = self.periph_manager.get_basket_distance()
                            if basket_dis_mm is not None and basket_dis_mm > 2500:
                                self.sub_transition_to(
                                    SearchSubState.MOVE_FORWARD, basket_dis_mm=basket_dis_mm
                                )

                            elif basket_dis_mm is not None and basket_dis_mm <= 2500:
                                self.sub_transition_to(
                                    SearchSubState.ALIGN_BASKET, is_opponent_basket=True
                                )

                    case SearchSubState.MOVE_FORWARD:
                        ret = self.base_handler.move_robot_forward()
                        if ret == RetCode.SUCCESS or ret == RetCode.TIMEOUT:
                            self.sub_transition_to(SearchSubState.TURN_DISCRETE)

            case GameState.ALIGN_BALL:
                ret = self.manipulation_handler.align_to_ball()
                if ret == RetCode.SUCCESS or ret == RetCode.TIMEOUT:
                    self.transition_to(GameState.GRAB_BALL)

                if ret == RetCode.FAILED_BALL_LOST:
                    self.transition_to(GameState.SEARCH_BALL)

            case GameState.GRAB_BALL:
                ret = self.manipulation_handler.move_forward_to_grab()
                if ret == RetCode.SUCCESS:
                    if ENABLE_ADVANCED_BASKET_ALIGNMENT:
                        self.transition_to(GameState.ALIGN_BASKET_ADVANCED)
                    else:
                        self.transition_to(GameState.ALIGN_BASKET)

                if ret == RetCode.TIMEOUT:
                    self.transition_to(GameState.SEARCH_BALL)

            case GameState.ALIGN_BASKET:
                ret = self.manipulation_handler.align_to_basket()
                if ret == RetCode.SUCCESS or ret == RetCode.TIMEOUT:
                    self.transition_to(GameState.THROW_BALL)

            case GameState.ALIGN_BASKET_ADVANCED:
                ret = self.manipulation_handler.align_to_basket_advanced()
                if ret == RetCode.SUCCESS or ret == RetCode.TIMEOUT:
                    self.transition_to(GameState.THROW_BALL)

            case GameState.THROW_BALL:
                ret = self.manipulation_handler.throw_ball()
                if ret == RetCode.TIMEOUT:
                    heading_error_deg = self.periph_manager.get_turning_angle_to_candidate_ball()
                    if heading_error_deg is None:  # no candidate ball
                        self.transition_to(GameState.SEARCH_BALL)
                    else:
                        self.transition_to(
                            GameState.TURN_TO_CANDIDATE_BALL, heading_error_deg=heading_error_deg
                        )

                if ret == RetCode.FAILED_BASKET_LOST:
                    self.transition_to(GameState.ALIGN_BASKET)

            case GameState.CLEAR_STUCK_BALL:
                ret = self.manipulation_handler.clear_stuck_ball()
                if ret == RetCode.TIMEOUT:
                    self.transition_to(GameState.SEARCH_BALL)

            case GameState.TURN_TO_CANDIDATE_BALL:
                ret = self.base_handler.turn_robot_cont(
                    Parameters.MAIN_TURNING_ANGULAR_SPEED_TO_CANDIDATE_BALL
                )
                if ret == RetCode.SUCCESS or ret == RetCode.TIMEOUT:
                    if self.periph_manager.is_ball_detected():
                        is_blocked, direction = self.periph_manager.is_ball_blocked_by_basket()
                        if is_blocked:
                            self.transition_to(
                                GameState.TURN_AROUND_BASKET, turning_basket_direction=direction
                            )
                        else:
                            self.transition_to(GameState.ALIGN_BALL)
                    else:
                        self.transition_to(GameState.SEARCH_BALL)

            case GameState.TURN_AROUND_BASKET:
                ret = self.manipulation_handler.turn_around_basket()
                if ret == RetCode.SUCCESS or ret == RetCode.TIMEOUT:
                    if self.periph_manager.is_ball_detected():
                        self.transition_to(GameState.ALIGN_BALL)
                    else:
                        heading_error_deg = (
                            self.periph_manager.get_turning_angle_to_candidate_ball()
                        )
                        if heading_error_deg is None:  # no candidate ball
                            self.transition_to(GameState.SEARCH_BALL)
                        else:
                            self.transition_to(
                                GameState.TURN_TO_CANDIDATE_BALL,
                                heading_error_deg=heading_error_deg,
                            )

            case _:
                raise RuntimeError("Unknown game state!")

    def transition_to(self, new_state: int, **kwargs: Union[int, float]) -> None:
        """Handle transition to a new state."""
        cur_state_name = GameState.get_name(self.cur_state)
        new_state_name = GameState.get_name(new_state)
        if new_state == self.cur_state:
            self.get_logger().info(f"Already in state {cur_state_name}, no transition needed.")
            return

        match new_state:
            case GameState.SEARCH_BALL:
                self.base_handler.initialize(
                    BaseAction.TURN_DISCRETE,
                    angle_deg=Parameters.MAIN_TURNING_DEGREE,
                    timeout=Parameters.MAIN_TIMEOUT_SEARCH_BALL_TURN_DISC,
                )
                
            case GameState.ALIGN_BALL:
                self.manipulation_handler.initialize(
                    ManipulationAction.ALIGN_BALL,
                    timeout=Parameters.MAIN_TIMEOUT_ALIGN_BALL,
                )
                
            case GameState.GRAB_BALL:
                self.manipulation_handler.initialize(
                    ManipulationAction.GRAB_BALL, timeout=Parameters.MAIN_TIMEOUT_GRAB_BALL
                )
                
            case GameState.ALIGN_BASKET:
                self.manipulation_handler.initialize(
                    ManipulationAction.ALIGN_BASKET,
                    basket_color=self.get_target_basket_color(),
                    base_thrower_percent=Parameters.MAIN_BASE_THROWER_PERCENT,
                    timeout=Parameters.MAIN_TIMEOUT_ALIGN_BASKET,
                )
                
            case GameState.ALIGN_BASKET_ADVANCED:
                self.manipulation_handler.initialize(
                    ManipulationAction.ALIGN_BASKET,
                    basket_color=self.get_target_basket_color(),
                    base_thrower_percent=Parameters.MAIN_BASE_THROWER_PERCENT,
                    timeout=Parameters.MAIN_TIMEOUT_ALIGN_BASKET_ADVANCED_TOTAL,
                    timeout_refine_angle=Parameters.MAIN_TIMEOUT_ALIGN_BASKET_ADVANCED_REFINE_ANGLE,
                )
                
            case GameState.THROW_BALL:
                self.manipulation_handler.initialize(
                    ManipulationAction.THROW_BALL, timeout=Parameters.MAIN_TIMEOUT_THROW_BALL
                )
                
            case GameState.CLEAR_STUCK_BALL:
                self.manipulation_handler.initialize(
                    ManipulationAction.CLEAR_STUCK_BALL,
                    timeout=Parameters.MAIN_TIMEOUT_CLEAR_STUCK_BALL,
                )
                
            case GameState.TURN_TO_CANDIDATE_BALL:
                assert "heading_error_deg" in kwargs, (
                    "heading_error_deg is required for TURN_TO_CANDIDATE_BALL state."
                )

                heading_error_deg = kwargs["heading_error_deg"]
                self.base_handler.initialize(
                    BaseAction.TURN_CONTINUOUS,
                    angle_deg=heading_error_deg,
                    timeout=Parameters.MAIN_TIMEOUT_TURN_TO_CANDIDATE_BALL,
                )
                
            case GameState.TURN_AROUND_BASKET:
                assert "turning_basket_direction" in kwargs, (
                    "turning_basket_direction is required for TURN_AROUND_BASKET state."
                )

                turning_basket_direction = kwargs["turning_basket_direction"]
                self.manipulation_handler.initialize(
                    ManipulationAction.TURN_AROUND_BASKET,
                    turning_basket_direction=turning_basket_direction,
                    timeout=Parameters.MAIN_TIMEOUT_TURN_AROUND_BASKET,
                )

        if new_state == GameState.SEARCH_BALL:
            # reset sub-state machine when entering SEARCH_BALL state
            self.cur_sub_state = SearchSubState.TURN_DISCRETE
            self.pre_sub_state = SearchSubState.UNDEFINED

        self.pre_state = self.cur_state
        self.cur_state = new_state
        self.get_logger().info(f"Transitioning from {cur_state_name} to {new_state_name}.")

    def sub_transition_to(self, new_sub_state: int, **kwargs: Union[bool, float]) -> None:
        """Handle transition to a new sub-state."""
        cur_sub_state_name = SearchSubState.get_name(self.cur_sub_state)
        new_sub_state_name = SearchSubState.get_name(new_sub_state)

        match new_sub_state:
            case SearchSubState.TURN_DISCRETE:
                self.base_handler.initialize(
                    BaseAction.TURN_DISCRETE,
                    angle_deg=Parameters.MAIN_TURNING_DEGREE,
                    timeout=Parameters.MAIN_TIMEOUT_SEARCH_BALL_TURN_DISC,
                )
            case SearchSubState.ALIGN_BASKET:
                if "is_opponent_basket" in kwargs and kwargs["is_opponent_basket"]:
                    basket_color = self.get_opponent_basket_color()
                else:
                    basket_color = self.get_target_basket_color()

                self.manipulation_handler.initialize(
                    ManipulationAction.ALIGN_BASKET,
                    basket_color=basket_color,
                    timeout=Parameters.MAIN_TIMEOUT_ALIGN_BASKET,
                )
            case SearchSubState.MOVE_FORWARD:
                assert "basket_dis_mm" in kwargs, (
                    "basket_dis_mm is required for MOVE_FORWARD sub-state."
                )

                basket_dis_mm = kwargs["basket_dis_mm"]
                self.base_handler.initialize(
                    BaseAction.MOVE_FORWARD,
                    offset_y_mm=basket_dis_mm - 1000,
                    timeout=Parameters.MAIN_TIMEOUT_SEARCH_BALL_MOVE_FW,
                )

        self.pre_sub_state = self.cur_sub_state
        self.cur_sub_state = new_sub_state
        self.get_logger().info(f"Transitioning from {cur_sub_state_name} to {new_sub_state_name}.")

    def print_current_state(self) -> None:
        """Log the current state and relevant information."""
        state_name = GameState.get_name(self.cur_state)
        sub_state_name = SearchSubState.get_name(self.cur_sub_state)
        if not self.referee_client.is_connected():
            status = "DISCONNECTED"
        else:
            status = "STARTED" if self.is_game_started else "INACTIVE"
        self.get_logger().info(
            f"State: {state_name} | SubState: {sub_state_name} | Ref Status: {status}"
            + f" | Color: {self.get_target_basket_color()}"
        )


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
