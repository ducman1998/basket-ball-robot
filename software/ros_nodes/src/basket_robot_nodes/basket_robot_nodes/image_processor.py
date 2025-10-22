import os
from time import time
from typing import List, Tuple

import cv2
import numpy as np
import pyrealsense2 as rs
import rclpy
from ament_index_python.packages import get_package_share_directory
from basket_robot_nodes.utils.image_info import ImageInfo
from basket_robot_nodes.utils.image_utils import (
    detect_green_ball_centers,
    get_cur_working_court_center,
    segment_color_hsv,
)
from basket_robot_nodes.utils.ros_utils import (
    log_initialized_parameters,
    parse_log_level,
)
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
from rclpy.node import Node
from rclpy.qos import QoSProfile
from sensor_msgs.msg import Image
from std_msgs.msg import String

VIZ_TOPIC = "/image/visualized"
IMAGE_INFO_TOPIC = "/image/info"


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
        ret = self._init_camera()
        if not ret:
            self.get_logger().error("Camera initialization failed. Exiting node.")
            raise RuntimeError("Camera initialization failed.")

        self.get_logger().info(
            f"Camera initialized with resolution {self.resolution} at {self.fps} FPS."
        )

        # setup ros publishers
        self.viz_im_pub = self.create_publisher(Image, VIZ_TOPIC, QoSProfile(depth=3))
        self.processed_info_pub = self.create_publisher(
            String, IMAGE_INFO_TOPIC, QoSProfile(depth=3)
        )

        self.handler = self.create_timer(1.0 / float(self.fps), self.process_frame)
        self.timestamp = self.get_clock().now().nanoseconds * 1e-9
        self.last_pub_viz_time = self.timestamp
        # internal queue to monitor fps
        self.fps_queue: List[float] = []

        # load robot base's mask
        # find image in share folder for this package

        self.robot_base_mask = cv2.imread(
            os.path.join(self.shared_dir, "images/robot_base_mask.png"), cv2.IMREAD_GRAYSCALE
        )
        if self.robot_base_mask is None:
            self.get_logger().error("Failed to load robot base mask image.")
            raise RuntimeError("Failed to load robot base mask image.")
        else:
            self.robot_base_mask = self.robot_base_mask.astype(bool).astype(np.uint8) * 255
            self.robot_base_mask = cv2.resize(
                self.robot_base_mask,
                (self.resolution[0], self.resolution[1]),
                interpolation=cv2.INTER_NEAREST,
            )
        self.get_logger().info("Loaded robot base mask image.")

    def _declare_node_parameters(self) -> None:
        """Declare parameters with descriptors."""
        fint_array_descriptor = ParameterDescriptor(
            type=ParameterType.PARAMETER_INTEGER_ARRAY,
            description="An array of integer numbers.",
        )
        int_descriptor = ParameterDescriptor(
            type=ParameterType.PARAMETER_INTEGER, description="An integer parameter."
        )
        bool_descriptor = ParameterDescriptor(
            type=ParameterType.PARAMETER_BOOL, description="A boolean parameter."
        )
        float_descriptor = ParameterDescriptor(
            type=ParameterType.PARAMETER_DOUBLE, description="A floating point parameter."
        )
        str_descriptor = ParameterDescriptor(
            type=ParameterType.PARAMETER_STRING, description="A string parameter."
        )
        self.declare_parameter("ref_colors_flat", descriptor=fint_array_descriptor)
        self.declare_parameter("ref_court_color", descriptor=fint_array_descriptor)
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
        self.ref_colors_flat: List[int] = (
            self.get_parameter("ref_colors_flat").get_parameter_value().integer_array_value.tolist()
        )
        if len(self.ref_colors_flat) % 3 != 0 or any(
            (c < 0 or c > 255) for c in self.ref_colors_flat
        ):
            self.get_logger().error(
                "Parameter 'ref_colors_flat' must be a list of RGB integers "
                + "[R1, G1, B1, R2, G2, B2,..] "
                + "where all values are in [0, 255] and the total length is a multiple of 3."
            )
            raise ValueError("Invalid ref_colors_flat parameter.")
        self.ref_ball_color: List[List[int]] = [
            self.ref_colors_flat[i : i + 3]
            for i in range(0, len(self.ref_colors_flat), 3)  # noqa: E203
        ]
        self.ref_court_color: List[int] = (
            self.get_parameter("ref_court_color").get_parameter_value().integer_array_value.tolist()
        )
        if len(self.ref_court_color) != 3 or any((c < 0 or c > 255) for c in self.ref_court_color):
            self.get_logger().error(
                "Parameter 'ref_court_color' must be a list of three RGB integers [R, G, B] "
                + "where all values are in [0, 255]."
            )
            raise ValueError("Invalid ref_court_color parameter.")
        res: List[int] = (
            self.get_parameter("resolution").get_parameter_value().integer_array_value.tolist()
        )
        if len(res) != 2:
            self.get_logger().error(
                "Parameter 'resolution' must be a list of two integers \
                [width, height]."
            )
            raise ValueError("Invalid resolution parameter.")
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
        log_level = self.get_parameter("log_level").get_parameter_value().string_value
        # set logging level
        self.get_logger().set_level(parse_log_level(log_level))
        self.get_logger().info(f"Set node {self.get_name()} log level to {log_level}.")
        return None

    def _init_camera(self) -> bool:
        """Initialize RealSense camera pipeline."""
        self.pipeline = rs.pipeline()
        cfg = rs.config()
        # Enable color and depthn streams
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
            profile = self.pipeline.start(cfg)
            color_sensor = profile.get_device().first_color_sensor()
            # set exposure time if auto-exposure is off
            if self.exposure_auto:
                self.get_logger().info("Auto-exposure is enabled.")
            else:
                color_sensor.set_option(rs.option.enable_auto_exposure, 0)  # turn off auto-exposure
                color_sensor.set_option(rs.option.exposure, self.exposure_time)
                self.get_logger().info(
                    f"Set exposure time of color camera to {self.exposure_time} microseconds."
                )
            if self.white_balance_auto:
                self.get_logger().info("Auto white balance is enabled.")
            else:
                color_sensor.set_option(rs.option.enable_auto_white_balance, 0)  # turn off auto WB
                self.get_logger().info("Set white balance of color camera to manual mode.")
        except RuntimeError as e:
            self.get_logger().error(f"Failed to start RealSense pipeline: {e}")
            # raise RuntimeError("Camera initialization failed.")
            return False  # for testing without camera

        # Warm up for auto-exposure to settle and wait for the first good frame set
        self.get_logger().info("Warming up camera (10 frames)...")
        for _ in range(10):
            self.pipeline.wait_for_frames()
        return True

    def process_frame(self) -> None:
        """Capture and process a single frame from the camera."""
        self.timestamp = self.get_clock().now().nanoseconds * 1e-9
        t1 = time()
        frames = self.pipeline.wait_for_frames()
        if self.enable_depth:
            aligned_frames = self.align.process(frames)
            color_frame_bgr = aligned_frames.get_color_frame()  # in bgr8 format
        else:
            color_frame_bgr = frames.get_color_frame()  # in bgr8 format

        if not color_frame_bgr:
            self.get_logger().error("No color or depth frame available.")
            return None

        color_frame_rgb = cv2.cvtColor(
            np.asanyarray(color_frame_bgr.get_data()), cv2.COLOR_BGR2RGB
        )  # convert to rgb
        t2 = time()
        try:
            # segment working area
            roi_mask, _ = segment_color_hsv(
                color_frame_rgb,
                ref_rgb=self.ref_court_color,  # color of working area
                h_tol=40,
                s_tol=45,
                v_tol=55,
                resize=0.3,
                morph_kernel=11,
                close=True,
                close_iter=3,
                dilate=True,
                dilate_iter=3,
                min_component_area=2000,
            )
            cur_court_center_2d, cur_court_center_px, court_area = get_cur_working_court_center(
                roi_mask, self.resolution[1], self.resolution[0]
            )
            # apply robot base mask
            if self.robot_base_mask is not None:
                roi_mask = cv2.bitwise_and(roi_mask, self.robot_base_mask)
            # detect balls
            detected_balls, viz_image = detect_green_ball_centers(
                color_frame_rgb,
                ref_ball_rgb=self.ref_ball_color,
                h_tol=5,
                s_tol=20,
                v_tol=20,
                mask_open_iter=1,
                mask_open_kernel_size=3,
                min_component_area=5,
                roi_mask=roi_mask,
                visualize=True,
            )
            # draw current working area center
            if cur_court_center_px is not None:
                cv2.drawMarker(
                    viz_image,
                    tuple(cur_court_center_px),
                    color=(0, 255, 0),
                    markerType=cv2.MARKER_CROSS,
                    markerSize=50,
                    thickness=5,
                )

            if not np.isclose(self.pub_viz_resize, 1.0, rtol=1e-09, atol=1e-09):
                viz_image = cv2.resize(
                    viz_image,
                    (
                        int(viz_image.shape[1] * self.pub_viz_resize),
                        int(viz_image.shape[0] * self.pub_viz_resize),
                    ),
                    interpolation=cv2.INTER_AREA,
                ).astype(
                    np.uint8, copy=False
                )  # resize and convert to uint8 (not necessary, but avoids mypy warnings)
        except ValueError as e:
            self.get_logger().error(f"Error in processing image: {e}")
            return None  # skip this frame

        t3 = time()
        pub_viz_cond = self.pub_viz_image and (
            self.timestamp - self.last_pub_viz_time
        ) >= 1.0 / float(self.pub_viz_fps)
        if pub_viz_cond:
            # publish visualized image
            viz_msg = Image()
            viz_msg.header.stamp = self.get_clock().now().to_msg()
            viz_msg.height = viz_image.shape[0]
            viz_msg.width = viz_image.shape[1]
            viz_msg.encoding = "rgb8"
            viz_msg.is_bigendian = False
            viz_msg.step = viz_image.shape[1] * 3
            viz_msg.data = viz_image.tobytes()
            self.viz_im_pub.publish(viz_msg)
            self.last_pub_viz_time = self.timestamp
        t4 = time()

        # publish detected ball info
        img_info = ImageInfo(
            balls=detected_balls,
            court_center=cur_court_center_2d,
            court_area=court_area,
        )
        info_msg = String()
        info_msg.data = img_info.to_json()
        self.processed_info_pub.publish(info_msg)

        elapsed_time = self.get_clock().now().nanoseconds * 1e-9 - self.timestamp
        self.fps_queue.append(1.0 / elapsed_time)
        if len(self.fps_queue) >= 10:
            self.fps_queue.pop(0)
        avg_fps = sum(self.fps_queue) / len(self.fps_queue)
        if elapsed_time >= 1.0 / float(self.fps):
            self.get_logger().warn(
                f"Processing is too slow! FPS={1.0/elapsed_time:.2f} < {self.fps}"
                + f" (avg over last {len(self.fps_queue)} frames: {avg_fps:.2f})"
            )

        t5 = time()
        if pub_viz_cond:
            self.get_logger().info(
                f"Timings (s): read={t2-t1:.3f}, detect={t3-t2:.3f}, pub_img={t4-t3:.3f}, "
                + f"pub_info={t5-t4:.3f}, detected={len(detected_balls)} balls"
            )

        return None


def main() -> None:
    rclpy.init()
    node = ImageProcessor()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
