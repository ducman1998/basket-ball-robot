from typing import List, Tuple, Union

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


def detect_green_balls(
    rgb_img: np.ndarray,
    center_rgb: Union[Tuple[int, ...], List[int]] = (34, 95, 7),  # reference ball color in RGB
    h_tol: int = 20,  # +/- hue tolerance (OpenCV H in [0..179])
    s_min: int = 50,  # minimum saturation to avoid grays
    min_area: float = 40.0,  # contour area filter
    min_radius: int = 8,  # min radius in pixels
    circularity_thresh: float = 0.6,  # 0..1, higher = more circular
    visualize: bool = True,  # whether to generate visualization image
) -> Tuple[List[GreenBall], NDArray[np.uint8]]:
    """
    Detect teal/green balls using HS-only masking (brightness ignored).

    Parameters
    ----------
    rgb_img : np.ndarray
        RGB image (H,W,3), dtype uint8.
    center_rgb : tuple
        Reference RGB color used to compute target hue (e.g., your ball color).
    h_tol : int
        +/- tolerance around the target hue (OpenCV hue in [0..179]).
    s_min : int
        Minimum saturation threshold (0..255).
    min_area : float
        Minimum contour area (pixels^2).
    min_radius : int
        Minimum enclosing circle radius (pixels).
    circularity_thresh : float
        0..1, higher = more circular.

    Returns
    -------
    List[GreenBall] : list of detected balls with attributes:
        - center: (x, y) in pixels
        - radius: in pixels
        - area: contour area in pixels^2
        - circularity: float in [0..1]
        - bbox: (x, y, w, h) bounding rectangle
    vis_img : np.ndarray
        BGR visualization image with contours, circles, and labels.
    """
    # Runtime check for unsigned [0, 255]
    if any((c < 0 or c > 255) for c in center_rgb):
        raise ValueError(f"center_rgb must contain unsigned 8-bit values, got {center_rgb}")

    if rgb_img is None or rgb_img.ndim != 3 or rgb_img.shape[2] != 3:
        raise ValueError("rgb_img must be an RGB image with shape (H, W, 3).")

    # --- derive target hue from the provided RGB color
    ref_rgb = np.array([[center_rgb]], dtype=np.uint8)
    ref_hsv = cv2.cvtColor(ref_rgb, cv2.COLOR_RGB2HSV)[0, 0]
    h0 = int(ref_hsv[0])  # OpenCV H in [0..179]

    # --- convert image to HSV
    hsv = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
    mask: NDArray[np.uint8] = _hs_mask(hsv, h0, h_tol, s_min)
    # --- clean up mask
    mask = _morph_clean(mask)

    # --- find contours
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    vis = rgb_img.copy()
    balls: List[GreenBall] = []

    for cnt in cnts:
        ball = _contour_to_ball(cnt, min_area, min_radius, circularity_thresh)
        if ball is None:
            continue

        balls.append(ball)
        if visualize:
            # visualization
            _draw_ball(vis, cnt, balls[-1])

    return balls, vis  # vis is RGB image


def _pixel_to_robot_coords(
    ball_center: Tuple[float, float], h_inv: np.ndarray, tbw: np.ndarray
) -> Tuple[float, float]:
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
    wd_point = h_inv @ ball_pos_h
    wd_point = (wd_point / wd_point[-1]).ravel()
    # convert to robot base_footprint frame
    b_point = tbw @ np.array([wd_point[0], wd_point[1], 0, 1]).reshape(4, 1)
    b_point = b_point / b_point[3]
    b_point_xy = np.round(b_point[:, :3].ravel(), 3)[:2]
    return float(b_point_xy[0]), float(b_point_xy[1])  # in mm


def _hs_mask(hsv: np.ndarray, h0: int, h_tol: int, s_min: int) -> NDArray[np.uint8]:
    """Create binary mask based on hue and saturation thresholds."""
    h, s = hsv[..., 0], hsv[..., 1]
    lower, upper = (h0 - h_tol) % 180, (h0 + h_tol) % 180
    hue_mask = ((h >= lower) & (h <= upper)) if lower <= upper else ((h >= lower) | (h <= upper))
    sat_mask: NDArray[np.bool_] = s >= s_min
    return (hue_mask & sat_mask).astype(np.uint8) * np.uint8(255)


def _draw_ball(vis: np.ndarray, c: np.ndarray, ball: GreenBall) -> None:
    """Draw detected ball info on visualization image."""
    cx, cy, r = int(ball.center[0]), int(ball.center[1]), int(ball.radius)
    cv2.circle(vis, (cx, cy), r, (0, 255, 0), 2)
    cv2.circle(vis, (cx, cy), 3, (255, 0, 0), -1)
    cv2.drawContours(vis, [c], -1, (255, 0, 0), 2)
    if ball.position_2d is None:
        label = f"r={r}px, pos=N/A, A={int(ball.area)}"
    else:
        x, y = ball.position_2d
        x_cm, y_cm = round(x / 10, 1), round(y / 10, 1)  # convert mm to cm
        label = f"r={r}px, pos=({x_cm},{y_cm})cm, A={int(ball.area)}"
    org = (max(0, cx - r), max(15, cy - r - 5))
    cv2.putText(vis, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.75, (10, 10, 10), 3, cv2.LINE_AA)
    cv2.putText(vis, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 1, cv2.LINE_AA)


def _contour_to_ball(
    cnt: np.ndarray,
    min_area: float,
    min_radius: int,
    circularity_thresh: float,
) -> "GreenBall | None":
    """Convert contour to GreenBall if it passes all filters, else return None."""
    area = float(cv2.contourArea(cnt))
    if area < min_area:
        return None

    (cx_f, cy_f), r_f = cv2.minEnclosingCircle(cnt)
    (cx_mm, cy_mm) = _pixel_to_robot_coords((cx_f, cy_f), H_INV, T_BW)
    cx, cy, r = int(round(cx_f)), int(round(cy_f)), int(round(r_f))
    if r < min_radius:
        return None

    perim = cv2.arcLength(cnt, True)
    if perim <= 0:
        return None

    circ = float(4.0 * np.pi * (area / (perim * perim)))
    if circ < circularity_thresh:
        return None

    x, y, w, h = cv2.boundingRect(cnt)
    return GreenBall(
        center=(cx, cy),
        radius=r,
        area=area,
        circularity=circ,
        bbox=(x, y, w, h),
        position_2d=(cx_mm, cy_mm),
    )


_kernel: NDArray[np.uint8] = np.ones((5, 5), dtype=np.uint8)


def _morph_clean(mask: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Morphological opening followed by closing to clean up binary mask."""
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _kernel, iterations=1).astype(
        np.uint8, copy=False
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _kernel, iterations=2).astype(
        np.uint8, copy=False
    )
    return mask
