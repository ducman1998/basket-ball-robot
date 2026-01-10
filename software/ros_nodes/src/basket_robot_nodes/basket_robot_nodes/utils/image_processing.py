from typing import List, Optional, Tuple, Union, cast

import cv2
import numba as nb
import numpy as np
from basket_robot_nodes.utils.color_segmention import ColorSegmenter
from basket_robot_nodes.utils.image_info import Basket, GreenBall, Marker
from basket_robot_nodes.utils.number_utils import get_rotation_matrix
from basket_robot_nodes.utils.constants import CALIB_THROWER_BASKET_OFFSETS
from numpy.typing import NDArray

from .constants import COLOR_REFERENCE_RGB

# original calibration image sizes (1280x720, used to compute homography)
CALIB_SCALE = 1.0
MARKER_OFFSET_X_MM = 230
BASKET_RADIUS_MM = 80
VALID_MARKER_IDS = [11, 12, 21, 22]
# homography matrix and its inverse obtained from camera calibration
H = np.array(
    [
        [-625.985857, 362.722537, 341273.811],
        [-12.587899, 0.298031, 163005.627],
        [-0.05002945, 0.914011, 607.478011],
    ]
)
H_INV = np.linalg.inv(H)
# transformation from world to robot base_footprint frame
T_BW = np.array(
    [
        [-9.98747463e-01, -4.60865665e-02, -1.94816165e-02, 1.34886895e02],
        [-4.60953146e-02, 9.98937046e-01, 3.48390069e-18, 4.52773087e02],
        [-3.29925840e-17, 1.37587664e-16, -1.00000000e00, -1.42108547e-13],
        [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
    ]
)
T_BW_INV = np.linalg.inv(T_BW)
# transformation from camera to robot base_footprint frame
T_BC = np.array(
    [
        [1.00000000e00, 0.00000000e00, 0.00000000e00, 0.00000000e00],
        [0.00000000e00, -4.02668379e-01, 9.15345933e-01, -75],
        [1.94816165e-02, -9.15172214e-01, -4.02591958e-01, 2.49391356e02],
        [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
    ]
)
# camera intrinsic matrix, determined from calibration
K = np.array(
    [[605.362976923247, 0, 427.370659545530], [0, 589.816167677644, 260.424249730625], [0, 0, 1]]
)
DIST_COEFFS = np.array([0.105048424954046, -0.201651710545795, 0.0, 0.0, 0.0], dtype=np.float32)
MARKER_SIZE = 0.16  # ArUco marker size (160mm = 0.16 meters)


@nb.njit(parallel=True)
def _check_balls_parallel(
    ball_centers: NDArray[np.int32],
    origin_pix: Tuple[int, int],
    seg_mask: NDArray[np.uint8],
    black_idx: int,
    white_idx: int,
    black_thresh: int,
    white_thresh: int,
    court_thresh: int,
    im_h: int,
    im_w: int,
) -> NDArray[np.bool_]:
    """
    Numba-compiled function to check ball status in parallel.
    Returns array of boolean values indicating if each ball is inside.
    Line is traced from ball center to origin.
    """
    n_balls = ball_centers.shape[0]
    inside_status = np.empty(n_balls, dtype=np.bool_)

    ox = origin_pix[0]
    oy = origin_pix[1]

    for idx in nb.prange(n_balls):
        cx = ball_centers[idx, 0]
        cy = ball_centers[idx, 1]

        # Bresenham's line algorithm - from ball center to origin
        dx = abs(ox - cx)
        dy = abs(oy - cy)

        if cx < ox:
            sx = 1
        else:
            sx = -1

        if cy < oy:
            sy = 1
        else:
            sy = -1

        err = dx - dy

        x = cx
        y = cy
        b_count = 0
        w_count = 0
        c_count = 0
        inside = True
        consecutive_black = 0
        consecutive_white = 0
        # Line tracing loop
        while True:
            # Boundary check
            if x < 0 or x >= im_w or y < 0 or y >= im_h:
                break

            if consecutive_white >= court_thresh:
                break

            # Check pixel label
            label = seg_mask[y, x]

            if label == black_idx:
                b_count += 1
                consecutive_black += 1
                consecutive_white = 0
            elif label == white_idx:
                consecutive_black = 0
                # only count white after black
                if b_count >= black_thresh:
                    w_count += 1
                    c_count = 0  # reset court count after white
                consecutive_white += 1

            else:  # reset counts if other colors encountered
                consecutive_black = 0
                consecutive_white = 0
                c_count += 1
                if c_count >= court_thresh:
                    c_count = b_count = w_count = (
                        0  # reset counts if enough court pixels encountered
                    )

            if consecutive_black > black_thresh:
                c_count = w_count = 0

            # Early termination if thresholds reached
            if b_count >= black_thresh and w_count >= white_thresh:
                inside = False
                break

            # Check if we reached destination
            if x == ox and y == oy:
                break

            # Bresenham step
            e2 = 2 * err
            if e2 > -dy:
                err = err - dy
                x = x + sx
            if e2 < dx:
                err = err + dx
                y = y + sy

        inside_status[idx] = inside

    return inside_status


class ImageProcessing:
    def __init__(
        self,
        robot_base_mask: NDArray[np.uint8],
        num_ignored_rows: int = 20,
        depth_scale: float = 0.001,
        ball_morth_kernel_size: int = 3,
        marker_morth_kernel_size: int = 7,
    ) -> None:
        self.image_segmenter = ColorSegmenter(COLOR_REFERENCE_RGB)
        self.robot_base_mask = robot_base_mask
        self.num_ignored_rows = num_ignored_rows
        self.depth_scale_mm = depth_scale * 1000  # to mm

        self.ball_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (ball_morth_kernel_size, ball_morth_kernel_size)
        )
        self.im_h, self.im_w = robot_base_mask.shape

        # ArUco marker detection setup
        # Use ORIGINAL ArUco dictionary (OpenCV 4.6 syntax)
        self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_ARUCO_ORIGINAL)
        self.aruco_params = cv2.aruco.DetectorParameters_create()
        self.marker_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (marker_morth_kernel_size, marker_morth_kernel_size)
        )

        self.basket_offsets = CALIB_THROWER_BASKET_OFFSETS

    def process(
        self,
        im_rgb: NDArray[np.uint8],
        depth: Optional[NDArray[np.float32]] = None,
        visualize: bool = False,
    ) -> Tuple[List[GreenBall], Optional[Basket], List[Marker], Optional[NDArray[np.uint8]]]:
        """
        Process the input RGB (and optional depth) image to detect green balls, baskets, and
        ArUco markers.
        Inputs:
            im_rgb: input RGB image of shape (H, W, 3) with dtype np.uint8
            depth: optional input depth image of shape (H, W) with dtype np.float32
            visualize: whether to generate visualization image
        Outputs:
            detected_balls: list of detected green balls
            detected_basket: detected basket (can be None if no basket detected)
            detected_markers: list of detected ArUco markers
            viz: visualization RGB image (can be None if visualize is False)
        """
        im_h, im_w = im_rgb.shape[:2]
        image_hsv = cv2.cvtColor(im_rgb, cv2.COLOR_RGB2HSV)
        image_gray = cv2.cvtColor(im_rgb, cv2.COLOR_RGB2GRAY)

        # segment all colors (court, green, blue, magenta, white, black)
        seg_mask = self.image_segmenter.segment_image(image_hsv, use_numba=True)
        seg_mask[self.robot_base_mask == 0] = 0  # ignore robot base area

        viz: Optional[NDArray[np.uint8]] = None
        if visualize:
            viz = im_rgb
        # detect ArUco markers
        basket_2d_pos: Optional[Tuple[float, float]] = None
        detected_markers = self.detect_aruco_markers(image_gray, seg_mask, viz)
        if len(detected_markers) > 0:
            # average position from two markers
            basket_2d_posisions = []
            for marker in detected_markers:
                t_r_marker = np.eye(3)
                t_r_marker[:2, :2] = get_rotation_matrix(np.deg2rad(marker.theta))
                t_r_marker[:2, 2] = marker.position_2d
                t_marker_basket = np.eye(3)
                dis_to_basket = np.linalg.norm(marker.position_2d) - BASKET_RADIUS_MM
                offsets_based_on_calib = self._get_basket_offset(dis_to_basket)
                if marker.id % 2 == 0:  # right side marker
                    t_marker_basket[0, 2] = -(MARKER_OFFSET_X_MM + offsets_based_on_calib)
                else:
                    t_marker_basket[0, 2] = MARKER_OFFSET_X_MM - offsets_based_on_calib
                t_marker_basket[1, 2] = -BASKET_RADIUS_MM
                t_r_basket = t_r_marker @ t_marker_basket
                basket_2d_posisions.append(t_r_basket[:2, 2])

            basket_2d_pos = np.mean(basket_2d_posisions, axis=0).tolist()

        # detect green balls
        detected_balls = self.detect_green_balls(
            seg_mask=seg_mask,
            viz_rgb=viz if visualize else None,
            min_area_ratio=0.2,  # (0, 1]
            min_component_area_ratio=15 / (im_h * im_w),
        )

        # detect baskets
        detected_basket = self.detect_baskets(
            seg_mask=seg_mask,
            depth=depth,
            viz_rgb=viz if visualize else None,
            min_component_area_ratio=250 / (im_h * im_w),
            position_2d_from_markers=basket_2d_pos,
        )
        return detected_balls, detected_basket, detected_markers, viz if visualize else None

    def detect_green_balls(
        self,
        seg_mask: NDArray[np.uint8],
        viz_rgb: Optional[NDArray[np.uint8]] = None,
        min_area_ratio: float = 0.000,  # (0, 1]
        min_component_area_ratio: float = 15 / (1280 * 720),
    ) -> List[GreenBall]:
        """
        Detect green balls from the segmented image mask.
        Inputs:
            seg_mask: segmented image mask of shape (H, W) with dtype np.uint8
            viz_rgb: optional RGB image for visualization (can be None)
            min_area_ratio: minimum area ratio threshold to validate ball detection
            min_component_area_ratio: minimum area threshold to filter small components
        Outputs:
            detected_balls: list of detected green balls
        """
        min_component_area = int(min_component_area_ratio * self.im_h * self.im_w)
        green_idx = self.image_segmenter.get_color_index("green")
        # get processed court and background masks, scale down to speed up processing
        # mask of presented green balls
        green_mask: NDArray[np.uint8] = ((seg_mask == green_idx).astype(np.uint8)) * 255
        # top rows are always outside the court area, remove them to reduce false positives
        green_mask[: self.num_ignored_rows, :] = 0
        filled_court_mask = self._get_processed_court_masks(seg_mask, scale=0.1)
        green_mask = cv2.bitwise_and(green_mask, filled_court_mask)
        # connected component analysis
        n_labels, _, stats, centroids = cv2.connectedComponentsWithStats(green_mask, connectivity=8)

        balls = []
        areas = []
        centers = []
        for i in range(1, n_labels):  # skip background (0)
            area = stats[i, cv2.CC_STAT_AREA]
            cx, cy = centroids[i]
            areas.append(area)
            centers.append([cx, cy])

        if len(centers) > 0:
            # Batch compute radii and positions in one call
            radii, positions = self._get_ball_radius(np.array(centers, dtype=np.float32))

            # Validate and create ball objects
            for _, (area, center, r, pos) in enumerate(zip(areas, centers, radii, positions)):  # type: ignore
                if self._is_valid_ball(area, tuple(pos), r, min_area_ratio, min_component_area):
                    balls.append(
                        GreenBall(
                            center=(int(center[0]), int(center[1])),
                            radius=float(r),
                            area=float(area),
                            position_2d=(float(pos[0]), float(pos[1])),
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

        filtered_balls = self._check_ball_status(
            filtered_balls, seg_mask, black_thresh=5, white_thresh=5, court_thresh=20
        )

        if viz_rgb is not None:
            for ball in filtered_balls:
                if ball.inside:
                    cv2.circle(viz_rgb, ball.center, int(ball.radius), (255, 0, 255), 2)
                else:
                    cv2.circle(viz_rgb, ball.center, int(ball.radius), (0, 255, 255), 2)
                text_pos = (
                    (ball.center[0] - 20, ball.center[1] - 10)
                    if ball.center[0] < viz_rgb.shape[1] - 100
                    else (ball.center[0] - 150, ball.center[1] - 10)
                )
                cv2.putText(
                    viz_rgb,
                    f"a={ball.area:.0f}, ({ball.position_2d[0]:.0f}, {ball.position_2d[1]:.0f})",
                    # f"a={ball.area:.0f}, ({100*ball.area/(np.pi*ball.radius**2):.2f})",
                    text_pos,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    1,
                )

        return filtered_balls

    def detect_baskets(
        self,
        seg_mask: NDArray[np.uint8],
        depth: Optional[NDArray[np.float32]] = None,
        viz_rgb: Optional[NDArray[np.uint8]] = None,
        min_component_area_ratio: float = 1000 / (1280 * 720),
        position_2d_from_markers: Optional[Tuple[float, float]] = None,
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
        min_component_area = int(min_component_area_ratio * self.im_h * self.im_w)
        basket_color: Optional[str] = None
        max_area: int = 0
        # bbox: (x_start, y_start, x_end, y_end)
        bbox: Optional[Tuple[int, int, int, int]] = None
        basket_mask: Optional[NDArray[np.uint8]] = None

        blue_idx, magenta_idx = self.image_segmenter.get_color_indices(["blue", "magenta"])
        mask: NDArray[np.uint8] = ((seg_mask == blue_idx) | (seg_mask == magenta_idx)).astype(
            np.uint8
        ) * 255

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
            if depth is not None and position_2d_from_markers is None:
                pos_2d = self._calculate_basket_2d_pos_from_depth(depth, basket_mask, bbox)
            else:
                pos_2d = position_2d_from_markers

            if viz_rgb is not None:
                cv2.rectangle(
                    viz_rgb,
                    (x_start, y_start),
                    (x_end, y_end),
                    (0, 0, 255) if basket_color == "blue" else (255, 0, 255),
                    2,
                )
                if pos_2d:
                    dis = np.linalg.norm(np.array(pos_2d))
                    if position_2d_from_markers is not None:
                        text = (
                            f"{basket_color}, p=({pos_2d[0]:.1f}, {pos_2d[1]:.1f}), d={dis:.1f} [M]"
                        )
                    else:
                        text = f"{basket_color}, p=({pos_2d[0]:.1f}, {pos_2d[1]:.1f}), d={dis:.1f}"
                else:
                    text = f"{basket_color}, p=(N/A), d=(N/A)"
                cv2.putText(
                    viz_rgb,
                    text,
                    (self.im_w // 2, center[1]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 0, 0),
                    1,
                )

            return Basket(color=basket_color, center=center, position_2d=pos_2d, area=max_area)
        else:
            return None

    def detect_aruco_markers(
        self,
        im_gray: NDArray[np.uint8],
        seg_mask: NDArray[np.uint8],
        viz_rgb: Optional[NDArray[np.uint8]] = None,
    ) -> List[Marker]:
        black_idx, white_idx = self.image_segmenter.get_color_indices(["black", "white"])
        roi_mask = (seg_mask == white_idx) | (seg_mask == black_idx)
        roi_mask = roi_mask.astype(np.uint8) * 255
        roi_mask = cv2.morphologyEx(roi_mask, cv2.MORPH_DILATE, self.marker_kernel)
        input_image = cv2.bitwise_and(im_gray, roi_mask)
        corners, ids, _ = cv2.aruco.detectMarkers(
            input_image, self.aruco_dict, parameters=self.aruco_params
        )

        detected_markers: List[Marker] = []
        if ids is not None:
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, MARKER_SIZE, K, DIST_COEFFS
            )
        else:
            return detected_markers

        for i in range(len(ids)):
            if int(ids[i][0]) not in VALID_MARKER_IDS:
                continue
            if viz_rgb is not None:
                # Draw detected markers
                cv2.aruco.drawDetectedMarkers(viz_rgb, corners, ids)

                # Draw axis (OpenCV 4.6 syntax)
                cv2.drawFrameAxes(viz_rgb, K, DIST_COEFFS, rvecs[i], tvecs[i], 0.05)

            # Get rotation matrix
            r_cm, _ = cv2.Rodrigues(rvecs[i])  # transformation from marker to camera
            t_cm = np.eye(4)
            t_cm[0:3, 0:3] = r_cm
            t_cm[0:3, 3] = tvecs[i].flatten() * 1000  # convert to mm
            t_bm = T_BC @ t_cm  # transformation from marker to robot base
            r_bm = t_bm[0:3, 0:3]
            t_bm = t_bm[0:2, 3]
            xvec = r_bm[:, 0]
            theta = np.degrees(np.arctan2(xvec[1], xvec[0]))
            detected_markers.append(
                Marker(id=int(ids[i][0]), position_2d=(float(t_bm[0]), float(t_bm[1])), theta=theta)
            )
        return detected_markers

    def _is_valid_ball(
        self,
        area: int,
        position_2d: Tuple[float, float],
        r: int,
        min_area_ratio: float,
        min_component_area: float = 15,
    ) -> bool:
        """
        Validate if the detected component is a valid ball based on area ratio and position.
        # area ratio check
        """
        if area < min_component_area:
            return False
        if area / (np.pi * r**2) < min_area_ratio:
            return False
        # below thesholds are likely noise, based on empirical observations
        if np.linalg.norm(np.array(position_2d)) < 500 and area < 100 * CALIB_SCALE**2:
            return False
        return True

    def _get_processed_court_masks(
        self, seg_mask: NDArray[np.uint8], scale: float = 0.15
    ) -> NDArray[np.uint8]:
        court_idx = self.image_segmenter.get_color_index("court")
        # Create base masks
        court_mask: NDArray[np.uint8] = ((seg_mask == court_idx).astype(np.uint8)) * 255

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
        ).astype(np.uint8)
        return filled_court_mask

    def _check_ball_status(
        self,
        balls: List[GreenBall],
        seg_mask: NDArray[np.uint8],
        black_thresh: int = 8,
        white_thresh: int = 10,
        court_thresh: int = 20,
    ) -> List[GreenBall]:
        """
        Check if the detected balls are inside the court area.
        Inputs:
            balls: list of detected green balls
            seg_mask: segmented image mask of shape (H, W) with dtype np.uint8
        Outputs:
            filtered_balls: list of balls with updated 'inside' attribute
        """
        if not balls:
            return balls

        black_idx, white_idx = self.image_segmenter.get_color_indices(["black", "white"])
        origin_pix = (self.im_w // 2, self.im_h - 1)

        # Prepare data for Numba function
        ball_centers = np.array([ball.center for ball in balls], dtype=np.int32)

        # Call Numba-compiled parallel function
        inside_status = _check_balls_parallel(
            ball_centers,
            origin_pix,
            seg_mask,
            black_idx,
            white_idx,
            black_thresh,
            white_thresh,
            court_thresh,
            self.im_h,
            self.im_w,
        )

        # Update ball status
        for i, ball in enumerate(balls):
            ball.inside = bool(inside_status[i])

        return balls

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
        return float(xs_b.mean()), float(ys_b.min())

    def _pixel_to_robot_coords(
        self, ball_centers: Union[Tuple[float, float], NDArray[np.float32]]
    ) -> Union[Tuple[float, float], NDArray[np.float32]]:
        """
        Convert pixel coordinates to robot base_footprint frame (Y forward, X right, Z up).
        Inputs:
            ball_centers: single point (x, y) or array of points with shape (N, 2)
        Returns:
            (x, y) coordinates in robot base_footprint frame in mm.
            If input is single point, returns tuple; if array, returns Nx2 array.
        """
        is_single_point = isinstance(ball_centers, tuple)

        # Convert to array format for batch processing
        if is_single_point:
            points = np.array([ball_centers], dtype=np.float32)
        else:
            points = np.asarray(ball_centers, dtype=np.float32)

        # Adjust pixel coordinates if image size differs from calibration size
        points_calib = points / CALIB_SCALE

        # Map image pixels to world plane using inverse homography (vectorized)
        # Add homogeneous coordinate
        ones = np.ones((points_calib.shape[0], 1), dtype=np.float32)
        points_h = np.hstack([points_calib, ones])  # Nx3
        wd_points = (H_INV @ points_h.T).T  # Nx3
        wd_points = wd_points / wd_points[:, 2:3]  # normalize by last column

        # Convert to robot base_footprint frame
        # Create 4D points with z=0
        points_4d = np.hstack(
            [wd_points[:, :2], np.zeros((wd_points.shape[0], 1)), np.ones((wd_points.shape[0], 1))]
        )  # Nx4
        b_points = (T_BW @ points_4d.T).T  # Nx4
        b_points = b_points / b_points[:, 3:4]
        result = np.round(b_points[:, :2], 3).astype(np.float32)

        # Return in original format
        if is_single_point:
            return tuple(map(float, result[0]))
        else:
            return result

    def _robot_to_pixel_coords(
        self, robot_xy: Union[Tuple[float, float], NDArray[np.float32]]
    ) -> Union[Tuple[float, float], NDArray[np.float32]]:
        """Convert robot base_footprint (x,y) in mm to image pixel coordinates.
        Inputs:
            robot_xy: single point (x, y) or array of points with shape (N, 2)
        Returns:
            (x, y) pixel coordinates in the image.
            If input is single point, returns tuple; if array, returns Nx2 array.
        """
        is_single_point = isinstance(robot_xy, tuple)

        # Convert to array format for batch processing
        if is_single_point:
            points = np.array([robot_xy], dtype=np.float32)
        else:
            points = np.asarray(robot_xy, dtype=np.float32)

        # Convert robot coords to homogeneous world plane (Nx4)
        zeros = np.zeros((points.shape[0], 1), dtype=np.float32)
        ones = np.ones((points.shape[0], 1), dtype=np.float32)
        robot_points = np.hstack([points, zeros, ones])  # Nx4
        world_points = (T_BW_INV @ robot_points.T).T  # Nx4
        world_points = world_points / world_points[:, 3:4]

        # Map world plane to image pixels using homography (Nx3)
        world_h = np.hstack([world_points[:, :2], np.ones((world_points.shape[0], 1))])
        img_h = (H @ world_h.T).T  # Nx3
        img_h = img_h / img_h[:, 2:3]  # normalize by last column

        # Adjust if image size differs from calibration size
        result = (img_h[:, :2] * CALIB_SCALE).astype(np.float32)

        # Return in original format
        if is_single_point:
            return tuple(map(float, result[0]))
        else:
            return result

    def _get_ball_radius(
        self, ball_centers: Union[Tuple[float, float], NDArray[np.float32]]
    ) -> Union[Tuple[float, Tuple[float, float]], Tuple[NDArray[np.float32], NDArray[np.float32]]]:
        """
        Estimate ball radius and 2D position in robot base_footprint frame.
        Inputs:
            ball_centers: single point (x, y) or array of points with shape (N, 2)
        Returns:
            If single point: (radius, (x, y)) tuple
            If array: (radii_array, positions_array) where positions_array is (N, 2)
        """
        is_single_point = isinstance(ball_centers, tuple)

        # Get robot coordinates for all centers
        xy_pos_centers = self._pixel_to_robot_coords(ball_centers)

        if is_single_point:
            # Single point case
            xy_pos_edge = (xy_pos_centers[0] - 20, xy_pos_centers[1])  # 20mm to the left
            ball_edge = self._robot_to_pixel_coords(xy_pos_edge)
            rad = np.linalg.norm(np.array(ball_centers) - np.array(ball_edge))
            return float(rad), (float(xy_pos_centers[0]), float(xy_pos_centers[1]))
        else:
            # Multiple points case
            # Create edge points: shift each point 20mm to the left in robot coords
            edge_points = xy_pos_centers.copy()  # type: ignore[union-attr]
            edge_points[:, 0] -= 20  # 20mm to the left
            ball_edges = cast(NDArray[np.float32], self._robot_to_pixel_coords(edge_points))

            # Calculate radii (distance between center and edge)
            radii = np.linalg.norm(ball_centers - ball_edges, axis=1)

            return radii, xy_pos_centers

    def _get_basket_offset(self, distance_mm: float) -> float:
        # Get basket offset based on distance using linear interpolation (use np.interp)
        distances = [offset[0] for offset in self.basket_offsets]
        offsets = [offset[1] for offset in self.basket_offsets]
        return float(np.interp(distance_mm, distances, offsets))
