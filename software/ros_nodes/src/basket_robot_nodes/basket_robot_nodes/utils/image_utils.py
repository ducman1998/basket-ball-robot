from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
from basket_robot_nodes.utils.image_info import GreenBall
from numpy.typing import NDArray

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


def detect_green_ball_centers(
    rgb_img: np.ndarray,
    ref_ball_rgb: Union[List[List[int]], List[Tuple[int, ...]]],
    h_tol: int = 10,
    s_tol: int = 10,
    v_tol: int = 10,
    mask_open_iter: int = 1,
    mask_open_kernel_size: int = 3,
    min_component_area: int = 10,
    roi_mask: Optional[np.ndarray] = None,
    visualize: bool = True,
) -> Tuple[List[GreenBall], NDArray[np.uint8]]:
    """
    Detect centers of green/teal regions (segmented clusters) using strict HSV filtering.

    Parameters
    ----------
    rgb_img : np.ndarray
        RGB image (H, W, 3), dtype uint8.
    ref_ball_rgb : list of RGB colors (list/tuple of ints)
        Reference greenish colors, e.g. [(32, 111, 109), (2, 94, 94), (6, 106, 99)].
    h_tol, s_tol, v_tol : int
        HSV tolerances for color segmentation.
    visualize : bool
        Whether to visualize the detected centers.

    Returns
    -------
    centers : list of (x, y)
        Pixel coordinates of detected ball centers.
    vis_img : np.ndarray
        Visualization image (RGB) with detected centers.
    """
    if rgb_img is None or rgb_img.ndim != 3:
        raise ValueError("Input must be an RGB image.")

    if roi_mask is not None:
        rgb_img = cv2.bitwise_and(rgb_img, rgb_img, mask=roi_mask)

    hsv = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
    total_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)

    # Combine masks from all reference colors
    for color_rgb in ref_ball_rgb:
        ref_rgb_array = np.array([[color_rgb]], dtype=np.uint8)
        ref_hsv = cv2.cvtColor(ref_rgb_array, cv2.COLOR_RGB2HSV)[0, 0]
        h_min, h_max = (ref_hsv[0] - h_tol) % 180, (ref_hsv[0] + h_tol) % 180
        s_min, s_max = max(0, ref_hsv[1] - s_tol), min(255, ref_hsv[1] + s_tol)
        v_min, v_max = max(0, ref_hsv[2] - v_tol), min(255, ref_hsv[2] + v_tol)
        lower = np.array([h_min, s_min, v_min])
        upper = np.array([h_max, s_max, v_max])
        current_mask = cv2.inRange(hsv, lower, upper)
        total_mask = cv2.bitwise_or(total_mask, current_mask).astype(np.uint8, copy=False)

    # Clean up the mask
    mask = _morph_clean(total_mask, num_iter=mask_open_iter, ksize=mask_open_kernel_size)

    # Find connected components or contours
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    balls = []
    vis = rgb_img.copy()

    for cnt in cnts:
        M = cv2.moments(cnt)
        if M["m00"] > 0:
            area = cv2.contourArea(cnt)  # number of pixels inside the contour
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            r, (pos_x, pos_y) = _get_ball_radius((cx, cy))
            balls.append(
                GreenBall(center=(cx, cy), radius=r, area=area, position_2d=(pos_x, pos_y))
            )

    # filter close detections to form one
    balls = sorted(balls, key=lambda b: b.radius * b.area, reverse=True)
    filtered_balls: List[GreenBall] = []
    for ball in balls:
        if ball.area < min_component_area:
            continue
        # below thesholds are likely noise, based on empirical observations
        if np.linalg.norm(np.array(ball.position_2d)) < 500 and ball.area < 250:
            continue
        if np.linalg.norm(np.array(ball.position_2d)) < 1500 and ball.area < 50:
            continue
        if all(
            np.linalg.norm(np.array(ball.center) - np.array(b.center)) > ball.radius * 2
            for b in filtered_balls
        ):
            filtered_balls.append(ball)

    if visualize:
        for ball in filtered_balls:
            cv2.circle(vis, ball.center, int(ball.radius), (0, 255, 0), 2)
            cv2.circle(vis, ball.center, 3, (255, 0, 0), -1)
            cv2.putText(
                vis,
                f"({ball.position_2d[0]:.0f}, {ball.position_2d[1]:.0f}), a={ball.area:.0f}",
                (ball.center[0] + 10, ball.center[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

    return filtered_balls, vis


def segment_color_hsv(
    img_rgb: np.ndarray,
    ref_rgb: Union[Tuple[int, int, int], List[int]],
    h_tol: int = 15,
    s_tol: int = 40,
    v_tol: int = 40,
    resize: float = 1.0,
    min_component_area: int = 100,
    morph_kernel: int = 5,  # size of kernel
    close: bool = False,
    close_iter: int = 3,
    dilate: bool = True,
    dilate_iter: int = 1,  # number of iterations
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Segment pixels close to a reference color in HSV space,
    merge valid regions into one mask, and optionally fill holes using dilation or closing.

    Parameters
    ----------
    morph_method: "dilate" or "close" - type of morphological operation to fill holes/edges

    Returns
    -------
    mask_filtered : np.ndarray
        Final mask covering all valid pixels.
    seg : np.ndarray
        Segmented image.
    """
    if img_rgb.ndim != 3 or img_rgb.shape[2] != 3:
        raise ValueError("img_rgb must be HxWx3 RGB image.")
    if img_rgb.dtype != np.uint8:
        raise ValueError("img_rgb must be uint8.")

    h, w = img_rgb.shape[:2]

    # 1. Resize if needed
    if resize != 1.0:
        proc_img = cv2.resize(img_rgb, (int(w * resize), int(h * resize)))
    else:
        proc_img = img_rgb

    # 2. Convert to HSV
    hsv = cv2.cvtColor(proc_img, cv2.COLOR_RGB2HSV)

    # 3. Compute HSV bounds
    ref_hsv = cv2.cvtColor(np.array([[ref_rgb]], dtype=np.uint8), cv2.COLOR_RGB2HSV)[0, 0]
    h0, s0, v0 = map(int, ref_hsv)
    h_min, h_max = (h0 - h_tol) % 180, (h0 + h_tol) % 180
    s_min, s_max = max(0, s0 - s_tol), min(255, s0 + s_tol)
    v_min, v_max = max(0, v0 - v_tol), min(255, v0 + v_tol)

    # 4. Create mask with hue wrap-around
    if h_min <= h_max:
        mask = cv2.inRange(hsv, np.array([h_min, s_min, v_min]), np.array([h_max, s_max, v_max]))
    else:
        mask1 = cv2.inRange(hsv, np.array([0, s_min, v_min]), np.array([h_max, s_max, v_max]))
        mask2 = cv2.inRange(hsv, np.array([h_min, s_min, v_min]), np.array([179, s_max, v_max]))
        mask = cv2.bitwise_or(mask1, mask2)

    # 5. Keep only large components
    if resize != 1.0:
        min_area_small = int(min_component_area * resize * resize)
    else:
        min_area_small = min_component_area

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_filtered = np.zeros_like(mask)

    for cnt in cnts:
        if cv2.contourArea(cnt) >= min_area_small:
            cv2.drawContours(mask_filtered, [cnt], -1, (255,), -1)

    # 6. Morphological operation to fill holes/edges
    krn = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel))
    if close and close_iter > 0:
        mask_filtered = cv2.morphologyEx(mask_filtered, cv2.MORPH_CLOSE, krn, iterations=close_iter)

    if dilate and dilate_iter > 0:
        mask_filtered = cv2.dilate(mask_filtered, krn, iterations=dilate_iter)

    # 7. Resize mask back
    if resize != 1.0:
        mask_filtered = cv2.resize(mask_filtered, (w, h), interpolation=cv2.INTER_NEAREST)

    # 8. Apply mask to original image
    seg = cv2.bitwise_and(img_rgb, img_rgb, mask=mask_filtered)
    return mask_filtered, seg


def get_cur_working_area_center(
    court_mask: NDArray[np.uint8],
) -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[int, int]]]:
    """
    Calculate the center of the current working area (court) in pixel coordinates.
    Parameters
    ----------
    court_mask : np.ndarray
        Binary mask of the court area.
    Returns
    -------
    court_center_2d : Tuple[float, float] or None
        (x, y) coordinates of the court center in robot base_footprint frame (mm),
        or None if not found.
    court_center_px : Tuple[int, int] or None
        (x, y) pixel coordinates of the court center in the image, or None if not found.
    """
    moments = cv2.moments(court_mask)
    if moments["m00"] == 0:
        return None, None

    cx = int(moments["m10"] / moments["m00"])
    cy = int(moments["m01"] / moments["m00"])
    court_center_2d = _pixel_to_robot_coords((cx, cy))
    return court_center_2d, (cx, cy)


def _pixel_to_robot_coords(ball_center: Tuple[float, float]) -> Tuple[float, float]:
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
    ball_pos = np.array(ball_center, dtype=np.float32)
    ball_pos_h = np.hstack([ball_pos, 1]).reshape(-1, 1)  # homogeneous coordinates
    wd_point = H_INV @ ball_pos_h
    wd_point = (wd_point / wd_point[-1]).ravel()
    # convert to robot base_footprint frame
    b_point = T_BW @ np.array([wd_point[0], wd_point[1], 0, 1]).reshape(4, 1)
    b_point = b_point / b_point[3]
    b_point_xy = np.round(b_point.ravel(), 3)[:2]
    return float(b_point_xy[0]), float(b_point_xy[1])  # in mm


def _robot_to_pixel_coords(robot_xy: Tuple[float, float]) -> Tuple[float, float]:
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
    x_px, y_px = img_h[0, 0], img_h[1, 0]
    return float(x_px), float(y_px)


def _get_ball_radius(ball_center: Tuple[float, float]) -> Tuple[float, Tuple[float, float]]:
    xy_pos_center = _pixel_to_robot_coords(ball_center)  # in mm
    xy_pos_edge = (xy_pos_center[0] - 20, xy_pos_center[1])  # 20mm to the left
    ball_edge = _robot_to_pixel_coords(xy_pos_edge)  # in pixels
    rad = np.linalg.norm(np.array(ball_center) - np.array(ball_edge))
    return float(rad), (float(xy_pos_center[0]), float(xy_pos_center[1]))


def _morph_clean(mask: NDArray[np.uint8], num_iter: int = 1, ksize: int = 3) -> NDArray[np.uint8]:
    _kernel: NDArray[np.uint8] = np.ones((ksize, ksize), dtype=np.uint8)
    """Morphological closing to clean up binary mask."""
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _kernel, iterations=num_iter).astype(
        np.uint8, copy=False
    )
    return mask
