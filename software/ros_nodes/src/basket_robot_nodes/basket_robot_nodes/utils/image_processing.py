from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
from basket_robot_nodes.utils.color_segmention import ColorSegmenter
from basket_robot_nodes.utils.image_info import Basket, GreenBall
from cv2.typing import MatLike
from numpy.typing import NDArray

from .constants import COLOR_REFERENCE_RGB

# original calibration image sizes (1280x720, used to compute homography)
CALIB_SCALE = 1.0
# homography matrix and its inverse obtained from camera calibration
H = np.array(
    [
        [-7.91132827e02, 5.58534931e02, 3.48701120e05],
        [-9.35251977e00, 2.42440144e01, 1.87787741e05],
        [1.91409035e-02, 8.26195098e-01, 3.51145140e02],
    ]
)
H_INV = np.linalg.inv(H)
# transformation from world to robot base_footprint frame
T_BW = np.array(
    [
        [-9.99445054e-01, 3.11668140e-02, -1.17564753e-02, 1.51709101e02],
        [3.11689681e-02, 9.99514130e-01, -1.60270640e-18, 2.31365296e02],
        [-4.45704372e-17, -6.20777250e-16, -1.00000000e00, -5.68434189e-14],
        [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
    ]
)
T_BW_INV = np.linalg.inv(T_BW)
# transformation from camera to robot base_footprint frame
T_BC = np.array(
    [
        [9.99982722e-01, 4.85780235e-03, 3.31007976e-03, 0.00000000e00],
        [9.22953605e-19, -5.63097781e-01, 8.26390276e-01, -5.00000000e01],
        [5.87833920e-03, -8.26375998e-01, -5.63088052e-01, 2.08871495e02],
        [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
    ]
)
# camera intrinsic matrix, determined from calibration
K = np.array(
    [
        [803.938356905409, 0.0, 645.705518859391],
        [0.0, 696.727389723765, 503.716447214066],
        [0.0, 0.0, 1.0],
    ]
)


class ImageProcessing:
    def __init__(
        self,
        robot_base_mask: NDArray[np.uint8],
        num_ignored_rows: int = 20,
        depth_scale: float = 0.001,
        ball_morth_kernel_size: int = 3,
    ) -> None:
        self.image_segmenter = ColorSegmenter(COLOR_REFERENCE_RGB)
        self.robot_base_mask = robot_base_mask
        self.num_ignored_rows = num_ignored_rows
        self.depth_scale_mm = depth_scale * 1000  # to mm

        self.ball_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (ball_morth_kernel_size, ball_morth_kernel_size)
        )

    def process(
        self,
        im_rgb: NDArray[np.uint8],
        depth: Optional[NDArray[np.float32]] = None,
        visualize: bool = False,
    ) -> Tuple[List[GreenBall], Optional[Basket], Optional[NDArray[np.uint8]]]:
        """
        Process the input RGB and depth images to detect green balls and baskets.
        Inputs:
            im_rgb: input RGB image of shape (H, W, 3) with dtype np.uint8
            depth: input depth image of shape (H, W) with dtype np.float32
            visualize: whether to generate a visualization image
        Outputs:
            detected_balls: list of detected green balls
            detected_basket: detected basket (can be None if no basket detected)
            viz: visualization image (can be None if visualize is False)
        """
        image_hsv = cv2.cvtColor(im_rgb, cv2.COLOR_RGB2HSV)

        # segment all colors (court, green, blue, magenta, white, black)
        seg_mask = self.image_segmenter.segment_image(image_hsv)

        viz: Optional[NDArray[np.uint8]] = None
        if visualize:
            viz = im_rgb.copy()
        # detect green balls
        detected_balls = self.detect_green_balls(
            seg_mask=seg_mask,
            viz_rgb=viz if visualize else None,
            min_component_area=15,
        )

        # detect baskets
        detected_basket = self.detect_baskets(
            seg_mask=seg_mask,
            depth=depth,
            viz_rgb=viz if visualize else None,
            min_component_area=1000,
        )

        return detected_balls, detected_basket, viz if visualize else None

    def detect_green_balls(
        self,
        seg_mask: NDArray[np.uint8],
        viz_rgb: Optional[NDArray[np.uint8]] = None,
        min_component_area: int = 15,
    ) -> List[GreenBall]:
        """
        Detect green balls from the segmented image mask.
        Inputs:
            seg_mask: segmented image mask of shape (H, W) with dtype np.uint8
            viz_rgb: optional RGB image for visualization (can be None)
            min_component_area: minimum area threshold to filter small components
        Outputs:
            detected_balls: list of detected green balls
        """
        green_idx = self.image_segmenter.get_color_index("green")
        # get processed court and background masks, scale down to speed up processing
        court_mask, filled_court_mask = self._get_processed_court_masks(seg_mask, scale=0.15)
        # mask of presented green balls
        green_mask: NDArray[np.uint8] = ((seg_mask == green_idx).astype(np.uint8)) * 255
        # top rows are always outside the court area, remove them to reduce false positives
        green_mask[: self.num_ignored_rows, :] = 0
        mask_open = cv2.morphologyEx(
            green_mask, cv2.MORPH_OPEN, self.ball_kernel, iterations=1
        ).astype(np.uint8, copy=False)
        mask_dilate = cv2.morphologyEx(
            mask_open, cv2.MORPH_DILATE, self.ball_kernel, iterations=3
        ).astype(np.uint8, copy=False)
        # after dilation, remove areas within court area which are overlapped with
        # court/white color
        clean_mask = cv2.bitwise_and(mask_dilate, cv2.bitwise_not(court_mask))
        # then mask out the robot base + outside court area. Now the detected balls will
        # be ensured to be within the court area only
        clean_mask = cv2.bitwise_and(clean_mask, filled_court_mask)

        # connected component analysis
        n_labels, _, stats, centroids = cv2.connectedComponentsWithStats(clean_mask, connectivity=8)

        balls = []
        for i in range(1, n_labels):  # skip background (0)
            area = stats[i, cv2.CC_STAT_AREA]
            if area < min_component_area:
                continue
            cx, cy = centroids[i]
            r, (pos_x, pos_y) = self._get_ball_radius((cx, cy))
            if self._is_valid_ball(area, (pos_x, pos_y), min_component_area):
                balls.append(
                    GreenBall(
                        center=(int(cx), int(cy)),
                        radius=r,
                        area=float(area),
                        position_2d=(pos_x, pos_y),
                    )
                )

        # filter close detections to form one
        balls = sorted(balls, key=lambda b: b.radius * b.area, reverse=True)
        filtered_balls: List[GreenBall] = []
        for ball in balls:
            if all(
                np.linalg.norm(np.array(ball.center) - np.array(b.center)) > ball.radius * 2
                for b in filtered_balls
            ):
                filtered_balls.append(ball)

        if viz_rgb is not None:
            for ball in filtered_balls:
                cv2.circle(viz_rgb, ball.center, int(ball.radius), (255, 0, 255), 2)
                text_pos = (
                    (ball.center[0] - 20, ball.center[1] - 10)
                    if ball.center[0] < viz_rgb.shape[1] - 100
                    else (ball.center[0] - 150, ball.center[1] - 10)
                )
                cv2.putText(
                    viz_rgb,
                    f"a={ball.area:.0f}, ({ball.position_2d[0]:.0f}, {ball.position_2d[1]:.0f})",
                    text_pos,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 0, 255),
                    1,
                )
        return filtered_balls

    def detect_baskets(
        self,
        seg_mask: NDArray[np.uint8],
        depth: Optional[NDArray[np.float32]] = None,
        viz_rgb: Optional[NDArray[np.uint8]] = None,
        min_component_area: int = 1000,
    ) -> Optional[Basket]:
        """
        Detect baskets from the segmented image mask.
        Inputs:
            seg_mask: segmented image mask of shape (H, W) with dtype np.uint8
            depth: input depth image of shape (H, W) with dtype np.float32
            viz_rgb: optional RGB image for visualization (can be None)
            min_component_area: minimum area threshold to filter small components
        Outputs:
            detected_basket: detected basket (can be None if no basket detected)
        """
        basket_color: Optional[str] = None
        max_area: int = 0
        # bbox: (x_start, y_start, x_end, y_end)
        bbox: Optional[Tuple[int, int, int, int]] = None
        basket_mask: Optional[NDArray[np.uint8]] = None

        blue_idx, magenta_idx = self.image_segmenter.get_color_indices(["blue", "magenta"])
        mask: NDArray[np.uint8] = np.isin(seg_mask, [blue_idx, magenta_idx]).astype(np.uint8) * 255
        if np.count_nonzero(mask) < min_component_area:
            return None

        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for i in range(1, n_labels):  # skip background (label 0)
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area > max_area and area >= min_component_area:
                max_area = area
                x_start = int(stats[i, cv2.CC_STAT_LEFT])
                y_start = int(stats[i, cv2.CC_STAT_TOP])
                x_end = x_start + int(stats[i, cv2.CC_STAT_WIDTH])
                y_end = y_start + int(stats[i, cv2.CC_STAT_HEIGHT])
                bbox = (x_start, y_start, x_end, y_end)
                # determine color
                seg_region = seg_mask[y_start:y_end, x_start:x_end]
                num_blue = np.count_nonzero(seg_region == blue_idx)
                num_magenta = np.count_nonzero(seg_region == magenta_idx)
                basket_color = "blue" if num_blue >= num_magenta else "magenta"
                basket_mask = (labels[y_start:y_end, x_start:x_end] == i).astype(np.uint8) * 255

        if basket_color and bbox is not None and basket_mask is not None:
            center = (int((bbox[0] + bbox[2]) // 2), int((bbox[1] + bbox[3]) // 2))
            x_start, y_start, x_end, y_end = bbox
            if depth is not None:
                pos_2d = self._calculate_basket_2d_pos_from_depth(depth, basket_mask, bbox)
            else:
                pos_2d = None

            if viz_rgb is not None:
                cv2.rectangle(
                    viz_rgb,
                    (x_start, y_start),
                    (x_end, y_end),
                    (0, 0, 255) if basket_color == "blue" else (255, 0, 255),
                    2,
                )
                if pos_2d:
                    text = f"{basket_color}, a={max_area}, p=({pos_2d[0]:.1f}, {pos_2d[1]:.1f})"
                else:
                    text = f"{basket_color}, a={max_area}, p=(N/A)"
                cv2.putText(
                    viz_rgb,
                    text,
                    center,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 0, 0),
                    1,
                )

            return Basket(color=basket_color, center=center, position_2d=pos_2d, area=max_area)
        else:
            return None

    def _is_valid_ball(
        self, area: int, position_2d: Tuple[float, float], min_component_area: int
    ) -> bool:
        """
        Check if the detected ball is valid based on area and position.
        Inputs:
            ball: GreenBall object
            min_component_area: minimum area threshold to filter small components
        Outputs:
            is_valid: True if the ball is valid, False otherwise
        """
        if area < min_component_area:
            return False
        # below thesholds are likely noise, based on empirical observations
        if np.linalg.norm(np.array(position_2d)) < 500 and area < 100 * CALIB_SCALE**2:
            return False
        return True

    def _get_processed_court_masks(
        self, seg_mask: NDArray[np.uint8], scale: float = 0.15
    ) -> Tuple[Union[NDArray[np.uint8], MatLike], Union[NDArray[np.uint8], MatLike]]:
        """
        Get processed court and background masks from the segmented image mask.
        Inputs:
            seg_mask: segmented image mask of shape (H, W) with dtype np.uint8
            scale: scaling factor to downscale the mask for processing
        Outputs:
            court_mask: processed court mask of shape (H, W) with dtype np.uint8
            filled_court_mask: processed filled court mask of shape (H, W) with dtype np.uint8
        """

        court_idx, white_idx, black_idx = self.image_segmenter.get_color_indices(
            ["court", "white", "black"]
        )
        # Create base masks
        court_mask: NDArray[np.uint8] = np.uint8(seg_mask == court_idx) * 255
        bg_mask: NDArray[np.uint8] = (
            np.uint8((seg_mask == white_idx) | (seg_mask == black_idx) | (seg_mask == court_idx))
            * 255
        )

        # Downscale to speed up contour detection
        small_mask = cv2.resize(
            court_mask,
            (0, 0),
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_NEAREST,
        )

        filled_court_mask_small = np.zeros_like(small_mask, dtype=np.uint8)

        # Find contours on the smaller mask
        contours, _ = cv2.findContours(small_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            all_points = np.vstack(contours)
            hull = cv2.convexHull(all_points)
            cv2.fillPoly(filled_court_mask_small, [hull], color=(255,))

        # Upscale mask back to original size
        filled_court_mask = cv2.resize(  # ignore: type
            filled_court_mask_small,
            (court_mask.shape[1], court_mask.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

        # Mask out robot base and background
        filled_court_mask_ret = cv2.bitwise_and(filled_court_mask, self.robot_base_mask)
        court_mask_ret = cv2.bitwise_and(filled_court_mask, bg_mask)

        return court_mask_ret, filled_court_mask_ret

    def _calculate_basket_2d_pos_from_depth(
        self,
        depth: NDArray[np.float32],
        basket_mask: NDArray[np.uint8],
        bbox: Tuple[int, int, int, int],
    ) -> Optional[Tuple[float, float]]:
        """
        Calculate the 2D position of the basket in robot base_footprint frame using depth data.
        Inputs:
            depth: input depth image of shape (H, W) with dtype np.float32
            basket_mask: binary mask of the basket component of shape (h, w) with dtype np.uint8
            bbox: bounding box of the basket in the format (x_start, y_start, x_end, y_end)
        Outputs:
            (x, y) coordinates of the basket in robot base_footprint frame in mm,
            or None if no valid depth points.
        """
        h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
        vs, us = np.arange(h), np.arange(w)
        uu, vv = np.meshgrid(us, vs)
        uu, vv = uu.ravel(), vv.ravel()
        depth_f32 = depth[bbox[1] : bbox[3], bbox[0] : bbox[2]]
        # basket_mask is already cropped to the component size, no need to slice again
        Z = depth_f32[vv, uu] * self.depth_scale_mm  # in mm
        mask_values = basket_mask[vv, uu]  # get mask values for the cropped basket area
        # valid depth points within the basket mask
        valid_mask = (Z > 0) & (mask_values > 0)
        uu, vv, Z = uu[valid_mask], vv[valid_mask], Z[valid_mask]
        if len(Z) == 0:
            return None

        # Convert relative bbox coordinates to absolute image coordinates
        u_abs = uu + bbox[0]  # Add x_start offset
        v_abs = vv + bbox[1]  # Add y_start offset
        X = (u_abs - K[0, 2]) * Z / K[0, 0]
        Y = (v_abs - K[1, 2]) * Z / K[1, 1]
        # convert 3d points to robot's base_footprint frame
        pts_c = np.vstack((X, Y, Z, np.ones_like(Z)))  # 4xN
        pts_b = T_BC @ pts_c  # 4xN
        pts_b = pts_b / pts_b[3, :]
        xs_b, ys_b = pts_b[0, :], pts_b[1, :]
        # TODO: consider using median to be more robust to outliers
        return float(xs_b.mean()), float(np.min(ys_b))

    def _pixel_to_robot_coords(self, ball_center: Tuple[float, float]) -> Tuple[float, float]:
        """
        Convert ball pixel coordinates to robot base_footprint frame (Y forward, X right, Z up).
        Inputs:
            ball_center: (x, y) pixel coordinates of the ball center in the image.
            h_inv: 3x3 inverse homography matrix (image to world plane).
            tbw: 4x4 transformation matrix from world plane to robot base_footprint frame.
        Returns:
            (x, y) coordinates of the ball in the robot base_footprint frame in mm.
        Note:
            - The homography maps image pixels to a flat ground plane (Z=0).
            - The transformation tbw should be pre-calibrated.
            - The output coordinates are in mm.
        """
        # adjust pixel coordinates if image size differs from calibration size
        x_px_calib = ball_center[0] / CALIB_SCALE
        y_px_calib = ball_center[1] / CALIB_SCALE
        ball_pos = np.array([x_px_calib, y_px_calib])  # in pixels
        # map image pixels to world plane using inverse homography
        ball_pos_h = np.hstack([ball_pos, 1]).reshape(-1, 1)  # homogeneous coordinates
        wd_point = H_INV @ ball_pos_h
        wd_point = (wd_point / wd_point[-1]).ravel()
        # convert to robot base_footprint frame
        b_point = T_BW @ np.array([wd_point[0], wd_point[1], 0, 1]).reshape(4, 1)
        b_point = b_point / b_point[3]
        b_point_xy = np.round(b_point.ravel(), 3)[:2]
        return float(b_point_xy[0]), float(b_point_xy[1])  # in mm

    def _robot_to_pixel_coords(self, robot_xy: Tuple[float, float]) -> Tuple[float, float]:
        """Convert robot base_footprint (x,y) in mm to image pixel coordinates.
        Inputs:
            robot_xy: (x, y) coordinates in robot base_footprint frame in mm.
            h: 3x3 homography matrix (world plane to image).
            tbw_inv: 4x4 inverse transformation from robot base_footprint to world plane.
        Returns:
            (x, y) pixel coordinates in the image.
        Note:
            - The homography maps a flat ground plane (Z=0) to image pixels.
            - The transformation tbw_inv should be pre-calibrated.
        """
        # convert robot coords to homogeneous world plane
        robot_point = np.array([robot_xy[0], robot_xy[1], 0, 1]).reshape(4, 1)
        world_point = T_BW_INV @ robot_point
        world_point = world_point / world_point[3]
        x_w, y_w = world_point[0, 0], world_point[1, 0]
        # map world plane to image pixels using homography
        world_h = np.array([x_w, y_w, 1]).reshape(3, 1)
        img_h = H @ world_h
        img_h = img_h / img_h[2]  # normalize
        x_px_calib, y_px_calib = img_h[0, 0], img_h[1, 0]
        # adjust if image size differs from calibration size
        x_px = x_px_calib * CALIB_SCALE
        y_px = y_px_calib * CALIB_SCALE
        return float(x_px), float(y_px)

    def _get_ball_radius(
        self, ball_center: Tuple[float, float]
    ) -> Tuple[float, Tuple[float, float]]:
        """
        Estimate ball radius in pixels and its 2D position in robot base_footprint frame.
        Inputs:
            ball_center: (x, y) pixel coordinates of the ball center in the image.
        Returns:
            radius: estimated ball radius in pixels.
            position_2d: (x, y) coordinates of the ball in robot base_footprint frame in mm.
        Note:
            - The radius is estimated based on a fixed offset in robot coordinates (20mm).
            - The output position_2d is in mm.
        """
        xy_pos_center = self._pixel_to_robot_coords(ball_center)  # in mm
        xy_pos_edge = (xy_pos_center[0] - 20, xy_pos_center[1])  # 20mm to the left
        ball_edge = self._robot_to_pixel_coords(xy_pos_edge)  # in pixels
        rad = np.linalg.norm(np.array(ball_center) - np.array(ball_edge))
        return float(rad), (float(xy_pos_center[0]), float(xy_pos_center[1]))
