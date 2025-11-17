import asyncio
import json
import threading
from typing import TYPE_CHECKING, Any, Callable, Literal, Optional, Union

import websockets
from rclpy.impl.rcutils_logger import RcutilsLogger
from websockets.exceptions import WebSocketException

if TYPE_CHECKING:
    # only import for type checking to avoid circular imports
    from rclpy.impl.rcutils_logger import RcutilsLogger


class RefereeClient:
    """WebSocket client for communicating with the robot basketball referee server."""

    def __init__(
        self,
        robot_id: str,
        referee_ip: str,
        referee_port: int,
        on_signal: Callable[[Literal["start", "stop"], Optional[str]], None],
        logger: "RcutilsLogger",
    ):
        """
        Initialize the referee client.

        Args:
            robot_id: Unique identifier for this robot
            referee_ip: IP address of the referee server
            referee_port: Port of the referee server
            on_signal: Callback function to handle 'start' and 'stop' signals
            logger: ROS2 logger instance
        """
        self.robot_id = robot_id
        self.referee_url = f"ws://{referee_ip}:{referee_port}"
        self.on_signal_fnc = on_signal
        self.logger = logger

        self.websocket: Optional[Any] = None
        self.is_running = False
        self.reconnect_delay = 1.0  # seconds
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None

    async def _connect_loop(self) -> None:
        """Connect to the referee server and handle reconnections."""
        reconnect_delay = self.reconnect_delay

        while self.is_running:
            try:
                self.logger.info(f"Attempting to connect to referee at {self.referee_url}...")
                async with websockets.connect(self.referee_url) as websocket:
                    self.websocket = websocket
                    self.logger.info("Successfully connected to referee server.")
                    reconnect_delay = self.reconnect_delay  # Reset delay on successful connection

                    # Listen for messages
                    await self._listen()

            except (WebSocketException, ConnectionRefusedError, OSError) as e:
                self.logger.error(f"Connection failed: {e}")
                self.websocket = None

                if self.is_running:
                    self.logger.info(f"Retrying connection in {reconnect_delay:.1f} seconds...")
                    await asyncio.sleep(reconnect_delay)

            except (asyncio.TimeoutError, ConnectionError, TimeoutError) as e:
                self.logger.info(f"Network error in connection loop: {e}")
                self.websocket = None
                if self.is_running:
                    self.logger.info(f"Retrying connection in {reconnect_delay:.1f} seconds...")
                    await asyncio.sleep(reconnect_delay)

    async def _listen(self) -> None:
        """Listen for messages from the referee server."""
        if not self.websocket:
            return

        async for message in self.websocket:
            await self._handle_message(message)

    async def _handle_message(self, message: Union[str, bytes]) -> None:
        """
        Handle incoming messages from the referee.

        Expected message formats:
        Start: {"signal": "start", "targets": ["ID1", "ID2"], "baskets": ["magenta", "blue"]}
        Stop:  {"signal": "stop", "targets": ["ID1", "ID2"]}
        """
        try:
            data = json.loads(message)
            self.logger.info(f"Received referee message: {data}")

            # Check if this message targets our robot
            targets = data.get("targets", [])
            if self.robot_id not in targets:
                self.logger.debug(f"Message not targeted to robot {self.robot_id}, ignoring.")
                return

            baskets = data.get("baskets", [])
            signal = data.get("signal", "").lower()
            if signal == "start" and len(baskets) != len(targets):
                self.logger.error("Mismatch between number of targets and baskets in start signal.")
                return

            # Handle start signal
            if signal in ["start", "stop"]:
                self.logger.info(f"Received {signal.upper()} signal from referee.")
                if signal == "start":
                    basket = baskets[targets.index(self.robot_id)]
                    self.on_signal_fnc(signal, basket)
                else:
                    self.on_signal_fnc(signal, None)
            else:
                self.logger.warn(f"Unknown signal type: {signal}")

        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse referee message: {e}")
        except KeyError as e:
            self.logger.error(f"Missing required field in message: {e}")
        except (TypeError, ValueError) as e:
            self.logger.error(f"Invalid message format: {e}")

    def _run_event_loop(self) -> None:
        """Run the asyncio event loop in a separate thread."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._connect_loop())
        except RuntimeError:
            # Event loop stopped before task completed (normal on shutdown)
            pass
        finally:
            # Cancel all pending tasks
            pending = asyncio.all_tasks(self._loop)
            for task in pending:
                task.cancel()
            # Wait for task cancellation to complete
            if pending:
                self._loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            self._loop.close()

    def start(self) -> None:
        """Start the referee client in a background thread."""
        if self.is_running:
            self.logger.warn("Referee client is already running.")
            return

        self.is_running = True
        self._thread = threading.Thread(target=self._run_event_loop, daemon=True)
        self._thread.start()
        self.logger.info("Started referee client background thread.")

    def stop(self) -> None:
        """Stop the referee client."""
        if not self.is_running:
            return

        self.is_running = False

        # Stop the event loop (this will break the connect loop)
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)

        # Wait for thread to finish
        if self._thread:
            self._thread.join(timeout=2.0)

        self.logger.info("Stopped referee client.")

    def is_connected(self) -> bool:
        """Check if currently connected to referee server."""
        if self.websocket is None:
            return False
        # Check if websocket is still open (not closed from either end)
        try:
            return not (self.websocket.close_sent or self.websocket.close_rcvd)
        except AttributeError:
            # Fallback for different websockets versions
            return True
