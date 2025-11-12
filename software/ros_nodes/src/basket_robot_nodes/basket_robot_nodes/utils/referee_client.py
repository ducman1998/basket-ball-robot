import asyncio
import json
import threading
from typing import Any, Callable, Literal, Optional, Union

import websockets
from websockets.exceptions import WebSocketException


class RefereeClient:
    """WebSocket client for communicating with the robot basketball referee server."""

    def __init__(
        self,
        robot_id: str,
        referee_ip: str,
        referee_port: int,
        on_signal: Optional[Callable[[Literal["start", "stop"], Optional[str]], None]] = None,
        logger: Optional[Any] = None,
    ):
        """
        Initialize the referee client.

        Args:
            robot_id: Unique identifier for this robot
            referee_ip: IP address of the referee server
            referee_port: Port of the referee server
            on_start: Callback function when start signal received
            on_stop: Callback function when stop signal received
            logger: Optional logger instance (ROS2 logger)
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

    def log_info(self, message: str) -> None:
        """Log info message if logger is available."""
        if self.logger:
            self.logger.info(message)

    def log_error(self, message: str) -> None:
        """Log error message if logger is available."""
        if self.logger:
            self.logger.error(message)

    def log_warning(self, message: str) -> None:
        """Log warning message if logger is available."""
        if self.logger:
            self.logger.warning(message)

    def log_debug(self, message: str) -> None:
        """Log debug message if logger is available."""
        if self.logger:
            self.logger.debug(message)

    async def _connect_loop(self) -> None:
        """Connect to the referee server and handle reconnections."""
        reconnect_delay = self.reconnect_delay

        while self.is_running:
            try:
                self.log_info(f"Attempting to connect to referee at {self.referee_url}...")
                async with websockets.connect(self.referee_url) as websocket:
                    self.websocket = websocket
                    self.log_info("Successfully connected to referee server.")
                    reconnect_delay = self.reconnect_delay  # Reset delay on successful connection

                    # Listen for messages
                    await self._listen()

            except (WebSocketException, ConnectionRefusedError, OSError) as e:
                self.log_error(f"Connection failed: {e}")
                self.websocket = None

                if self.is_running:
                    self.log_info(f"Retrying connection in {reconnect_delay:.1f} seconds...")
                    await asyncio.sleep(reconnect_delay)

            except (asyncio.TimeoutError, ConnectionError, TimeoutError) as e:
                self.log_error(f"Network error in connection loop: {e}")
                self.websocket = None
                if self.is_running:
                    await asyncio.sleep(reconnect_delay)

    async def _listen(self) -> None:
        """Listen for messages from the referee server."""
        if not self.websocket:
            return

        try:
            async for message in self.websocket:
                await self._handle_message(message)
        except WebSocketException as e:
            self.log_warning(f"WebSocket connection closed: {e}")
        except (ConnectionError, asyncio.CancelledError) as e:
            self.log_warning(f"Connection interrupted: {e}")

    async def _handle_message(self, message: Union[str, bytes]) -> None:
        """
        Handle incoming messages from the referee.

        Expected message formats:
        Start: {"signal": "start", "targets": ["ID1", "ID2"], "baskets": ["magenta", "blue"]}
        Stop:  {"signal": "stop", "targets": ["ID1", "ID2"]}
        """
        try:
            data = json.loads(message)
            self.log_info(f"Received referee message: {data}")

            # Check if this message targets our robot
            targets = data.get("targets", [])
            if not targets or self.robot_id not in targets:
                self.log_debug(f"Message not targeted to robot {self.robot_id}, ignoring.")
                return

            baskets = data.get("baskets", [])
            signal = data.get("signal", "").lower()
            if signal == "start" and len(baskets) != len(targets):
                self.log_error("Mismatch between number of targets and baskets in start signal.")
                return

            # Handle start signal
            if signal in ["start", "stop"]:
                self.log_info(f"Received {signal.upper()} signal from referee.")
                if self.on_signal_fnc:
                    if signal == "start":
                        basket = baskets[targets.index(self.robot_id)]
                        self.on_signal_fnc(signal, basket)
                    else:
                        self.on_signal_fnc(signal, None)
            else:
                self.log_warning(f"Unknown signal type: {signal}")

        except json.JSONDecodeError as e:
            self.log_error(f"Failed to parse referee message: {e}")
        except KeyError as e:
            self.log_error(f"Missing required field in message: {e}")
        except (TypeError, ValueError) as e:
            self.log_error(f"Invalid message format: {e}")

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
            self.log_warning("Referee client is already running.")
            return

        self.is_running = True
        self._thread = threading.Thread(target=self._run_event_loop, daemon=True)
        self._thread.start()
        self.log_info("Started referee client background thread.")

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

        self.log_info("Stopped referee client.")

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
