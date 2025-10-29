import os
import threading
from time import time
from typing import List, Tuple

import cv2
import numpy as np
import pyrealsense2 as rs
import rclpy
from ament_index_python.packages import get_package_share_directory
from basket_robot_nodes.utils.constants import QOS_DEPTH
from basket_robot_nodes.utils.image_info import ImageInfo
from basket_robot_nodes.utils.image_processing import ImageProcessing
from basket_robot_nodes.utils.ros_utils import (
    bool_descriptor,
    fint_array_descriptor,
    float_descriptor,
    int_descriptor,
    log_initialized_parameters,
    parse_log_level,
    str_descriptor,
)
from cv_bridge import CvBridge
from numpy.typing import NDArray
from rclpy.node import Node
from rclpy.qos import QoSProfile
from sensor_msgs.msg import Image
from std_msgs.msg import String


class ImageProcessor(Node):
    def __init__(self) -> None:
        super().__init__("image_processor_node")
        self.shared_dir = get_package_share_directory("basket_robot_nodes")
        # declare params (no defaults, must be set in launch)
        self._declare_node_parameters()
        # read params
        self._read_node_parameters()

        # for checking: log all initialized parameters
        log_initialized_parameters(self)

        # setup camera
        ret = self._setup_realsense_cameras()
        if not ret:
            self.get_logger().error("Camera initialization failed. Exiting node.")
            raise RuntimeError("Camera initialization failed.")

        self.get_logger().info(
            f"Camera initialized with resolution {self.resolution} at {self.fps} FPS."
        )

        # setup ros publishers
        self.viz_im_pub = self.create_publisher(
            Image, "/image/visualized", QoSProfile(depth=QOS_DEPTH)
        )
        self.processed_info_pub = self.create_publisher(
            String, "/image/info", QoSProfile(depth=QOS_DEPTH)
        )

        self.handler = self.create_timer(1.0 / float(self.fps), self.process_frame)
        self.timestamp = self.get_clock().now().nanoseconds * 1e-9
        self.last_pub_viz_time = self.timestamp
        # internal queue to monitor fps
        self.fps_queue: List[float] = []

        _robot_mask = cv2.imread(
            os.path.join(self.shared_dir, "images/robot_base_mask.png"), cv2.IMREAD_GRAYSCALE
        )
        if _robot_mask is None:
            raise RuntimeError("Failed to load robot base mask image.")
        else:
            _robot_mask = _robot_mask.astype(bool).astype(np.uint8) * 255
            self.robot_base_mask: NDArray[np.uint8] = cv2.resize(
                _robot_mask,
                (self.resolution[0], self.resolution[1]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(np.uint8, copy=False)
        self.get_logger().info("Loaded robot base mask image.")

        self.image_processor = ImageProcessing(
            robot_base_mask=self.robot_base_mask,
            num_ignored_rows=20,
            depth_scale=self._get_depth_scale(),
            ball_morth_kernel_size=3,
        )
        self.bridge = CvBridge()

    def process_frame(self) -> None:
        """Capture and process a single frame from the camera."""
        self.timestamp = self.get_clock().now().nanoseconds * 1e-9
        is_visualized = self.pub_viz_image and (
            self.timestamp - self.last_pub_viz_time
        ) >= 1.0 / float(self.pub_viz_fps)

        t1 = time()
        frames = self.pipeline.wait_for_frames()
        if self.enable_depth:
            aligned_frames = self.align.process(frames)
            color_frame_bgr = aligned_frames.get_color_frame()  # HxW in bgr8 format
            depth_frame = frames.get_depth_frame()  # HxW (16 bits)
        else:
            color_frame_bgr = frames.get_color_frame()  # in bgr8 format
            depth_frame = None

        if not color_frame_bgr or (self.enable_depth and depth_frame is None):
            self.get_logger().error("No color or depth frame available.")
            return None

        color_frame_rgb = cv2.cvtColor(np.asanyarray(color_frame_bgr.get_data()), cv2.COLOR_BGR2RGB)
        depth_frame_f32 = None
        if depth_frame is not None:
            depth_frame_f32 = np.asanyarray(depth_frame.get_data()).astype(np.float32)

        t2 = time()
        detected_balls, detected_basket, viz_image = self.image_processor.process(
            im_rgb=color_frame_rgb, depth=depth_frame_f32, visualize=is_visualized
        )

        if (
            not np.isclose(self.pub_viz_resize, 1.0, rtol=1e-09, atol=1e-09)
            and viz_image is not None
        ):
            # resize and convert to uint8 (not necessary, but avoids mypy warnings)
            viz_image = cv2.resize(
                viz_image,
                (
                    int(viz_image.shape[1] * self.pub_viz_resize),
                    int(viz_image.shape[0] * self.pub_viz_resize),
                ),
                interpolation=cv2.INTER_AREA,
            ).astype(np.uint8, copy=False)
        t3 = time()

        # publish detected ball info
        img_info = ImageInfo(balls=detected_balls, basket=detected_basket)
        info_msg = String()
        info_msg.data = img_info.to_json()
        self.processed_info_pub.publish(info_msg)

        t4 = time()
        if is_visualized:
            self.get_logger().info(
                f"Timings (s): read={t2-t1:.3f}, detect={t3-t2:.3f}, "
                + f"pub_info={t4-t3:.3f}, detected={len(detected_balls)} balls"
            )

        elapsed_time = self.get_clock().now().nanoseconds * 1e-9 - self.timestamp
        avg_fps = self._monitor_fps(elapsed_time)
        if elapsed_time >= 1.0 / float(self.fps):
            self.get_logger().warn(
                f"Processing is too slow! FPS={1.0/elapsed_time:.2f} < {self.fps}"
                + f" (avg over last {len(self.fps_queue)} frames: {avg_fps:.2f})"
            )
        if is_visualized and viz_image is not None:
            return self.publish_viz_async(viz_image)
        else:
            return None

    def _declare_node_parameters(self) -> None:
        """Declare parameters with descriptors."""
        self.declare_parameter("resolution", descriptor=fint_array_descriptor)
        self.declare_parameter("fps", descriptor=int_descriptor)
        self.declare_parameter("exposure_auto", descriptor=bool_descriptor)
        self.declare_parameter("exposure_time", descriptor=int_descriptor)  # in microseconds
        self.declare_parameter("white_balance_auto", descriptor=bool_descriptor)
        self.declare_parameter("enable_depth", descriptor=bool_descriptor)
        self.declare_parameter("publish_viz_image", descriptor=bool_descriptor)
        self.declare_parameter("publish_viz_fps", descriptor=int_descriptor)
        self.declare_parameter("publish_viz_resize", descriptor=float_descriptor)
        self.declare_parameter("log_level", descriptor=str_descriptor)

    def _read_node_parameters(self) -> None:
        """Read and validate parameters."""
        res = self.get_parameter("resolution").get_parameter_value().integer_array_value.tolist()
        if len(res) != 2:
            self.get_logger().error(
                "Parameter 'resolution' must be a list of two integers \
                [width, height]."
            )
            raise ValueError("Invalid resolution parameter.")
        else:
            self.resolution: Tuple[int, int] = (res[0], res[1])

        self.fps: int = self.get_parameter("fps").get_parameter_value().integer_value
        self.exposure_auto: bool = (
            self.get_parameter("exposure_auto").get_parameter_value().bool_value
        )
        self.exposure_time: int = (
            self.get_parameter("exposure_time").get_parameter_value().integer_value
        )
        self.white_balance_auto: bool = (
            self.get_parameter("white_balance_auto").get_parameter_value().bool_value
        )
        self.enable_depth: bool = (
            self.get_parameter("enable_depth").get_parameter_value().bool_value
        )
        self.pub_viz_image: bool = (
            self.get_parameter("publish_viz_image").get_parameter_value().bool_value
        )
        self.pub_viz_fps: int = (
            self.get_parameter("publish_viz_fps").get_parameter_value().integer_value
        )
        self.pub_viz_resize: float = (
            self.get_parameter("publish_viz_resize").get_parameter_value().double_value
        )
        if self.pub_viz_resize <= 0.0 or self.pub_viz_resize > 1.0:
            self.get_logger().error("Parameter 'publish_viz_resize' must be in (0.0, 1.0].")
            raise ValueError("Invalid publish_viz_resize parameter.")

        if self.pub_viz_fps <= 0 or self.pub_viz_fps > self.fps:
            self.get_logger().error("Parameter 'publish_viz_fps' must be in (0, fps].")
            raise ValueError("Invalid publish_viz_fps parameter.")

        # read and set logging level
        log_level = self.get_parameter("log_level").get_parameter_value().string_value
        self.get_logger().set_level(parse_log_level(log_level))
        self.get_logger().info(f"Set node {self.get_name()} log level to {log_level}.")
        return None

    def _setup_realsense_cameras(self) -> bool:
        """Initialize RealSense camera pipeline."""
        self.pipeline = rs.pipeline()
        cfg = rs.config()
        # Enable color and depth streams
        cfg.enable_stream(
            rs.stream.color, self.resolution[0], self.resolution[1], rs.format.bgr8, self.fps
        )
        if self.enable_depth:
            cfg.enable_stream(
                rs.stream.depth, self.resolution[0], self.resolution[1], rs.format.z16, self.fps
            )
            self.align = rs.align(rs.stream.color)
        try:
            self.get_logger().info("Starting RealSense pipeline...")
            self.profile = self.pipeline.start(cfg)
            color_sensor = self.profile.get_device().first_color_sensor()
            # set exposure time if auto-exposure is off
            if self.exposure_auto:
                color_sensor.set_option(rs.option.enable_auto_exposure, 1)  # turn on auto-exposure
                self.get_logger().info("Auto-exposure is enabled.")
            else:
                color_sensor.set_option(rs.option.enable_auto_exposure, 0)  # turn off auto-exposure
                color_sensor.set_option(rs.option.exposure, self.exposure_time)
                self.get_logger().info(
                    f"Set exposure time of color camera to {self.exposure_time} microseconds."
                )

            if self.white_balance_auto:
                color_sensor.set_option(rs.option.enable_auto_white_balance, 1)  # turn on auto WB
                self.get_logger().info("Auto white balance is enabled.")
            else:
                color_sensor.set_option(rs.option.enable_auto_white_balance, 0)  # turn off auto WB
                self.get_logger().info("Set white balance of color camera to manual mode.")

        except RuntimeError as e:
            self.profile = None
            self.get_logger().error(f"Failed to start RealSense pipeline: {e}")
            return False  # for testing without camera

        # Warm up for auto-exposure to settle and wait for the first good frame set
        self.get_logger().info("Warming up camera (10 frames)...")
        for _ in range(10):
            self.pipeline.wait_for_frames()
        return True

    def _get_depth_scale(self) -> float:
        """Get depth scale from the RealSense device."""
        if self.profile is None:
            raise RuntimeError("RealSense camera is not initialized.")
        depth_sensor = self.profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        return float(depth_scale)

    def _monitor_fps(self, elapsed_time: float) -> float:
        """Monitor and log the average FPS over the last 10 frames."""
        self.fps_queue.append(1.0 / elapsed_time)
        if len(self.fps_queue) >= 10:
            self.fps_queue.pop(0)
        avg_fps = sum(self.fps_queue) / len(self.fps_queue)
        return avg_fps

    def publish_viz_async(self, viz_image: NDArray[np.uint8]) -> None:
        self.last_pub_viz_time = self.timestamp

        def worker() -> None:
            viz_msg = self.bridge.cv2_to_imgmsg(viz_image, encoding="rgb8")
            viz_msg.header.stamp = self.get_clock().now().to_msg()
            self.viz_im_pub.publish(viz_msg)

        threading.Thread(target=worker, daemon=True).start()


def main() -> None:
    rclpy.init()
    node = ImageProcessor()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
