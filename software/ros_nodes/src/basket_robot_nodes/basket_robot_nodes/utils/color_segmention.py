from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from numpy.typing import NDArray
from numba import jit, prange

from .constants import ENABLED_SEGMENTED_COLORS, COLOR_TOLERANCES, COLOR_VIZ_RGB_MAP


@jit(nopython=True, parallel=True, cache=True, fastmath=True, nogil=True)
def segment_image_numba(
    ref_colors_mtx: NDArray[np.uint8], hsv_img: NDArray[np.uint8]
) -> NDArray[np.uint8]:
    """
    Numba-optimized color segmentation with parallelization.

    Args:
        ref_colors_mtx: Reference color lookup table (180, 128, 128)
        hsv_img: Input HSV image (H, W, 3)

    Returns:
        segmented_img: Color-indexed segmentation mask (H, W)
    """
    h = hsv_img.shape[0]
    w = hsv_img.shape[1]
    segmented_img = np.empty((h, w), dtype=np.uint8)

    # Parallelize across rows
    for y in prange(h):
        for x in range(w):
            h_idx = hsv_img[y, x, 0]
            s_idx = hsv_img[y, x, 1] >> 1
            v_idx = hsv_img[y, x, 2] >> 1
            segmented_img[y, x] = ref_colors_mtx[h_idx, s_idx, v_idx]

    return segmented_img


class ColorSegmenter:
    def __init__(self, ref_color_dict: Dict[str, List[List[int]]]) -> None:
        self.color_to_index: Dict[str, int] = {}
        for idx, color in enumerate(sorted(ref_color_dict.keys())):
            # zero index is reserved for non-defined colors
            self.color_to_index[color] = idx + 1
            if color not in COLOR_TOLERANCES:
                raise ValueError(f"Color {color} not found in reference color map.")
        self.ref_colors_mtx: NDArray[np.uint8] = self._process_reference_colors(ref_color_dict)

    def get_color_index(self, color_name: str) -> int:
        """Get the index of a specific color."""
        if color_name not in self.color_to_index:
            raise ValueError(f"Color {color_name} not found in reference colors.")
        return self.color_to_index[color_name]

    def get_color_indices(self, color_names: List[str]) -> List[int]:
        """Get the indices of a list of colors."""
        indices = []
        for color_name in color_names:
            if color_name not in self.color_to_index:
                raise ValueError(f"Color {color_name} not found in reference colors.")
            indices.append(self.color_to_index[color_name])
        return indices

    def _process_reference_colors(
        self, ref_color_dict: Dict[str, List[List[int]]]
    ) -> NDArray[np.uint8]:
        """
        Process the reference RGB colors into a look-up table in HSV space.
        Inputs:
            ref_color_dict: dictionary mapping color names to list of reference RGB colors
        Outputs:
            ref_lut: look-up table for reference colors in HSV space

        Note: The HSV look-up table has reduced resolution for S and V channels to save memory.
        """
        ref_lut: NDArray[np.uint8] = np.zeros((180, 128, 128), dtype=np.uint8)
        # the order of processing colors matters, for example, court will override green,
        # helping to reduce false positives
        for color in ENABLED_SEGMENTED_COLORS:
            if color not in ref_color_dict:
                raise ValueError(
                    f"Color {color} not found in reference color map. Please update it."
                )
            ref_rgb_colors = ref_color_dict[color]
            for ref_rgb in ref_rgb_colors:
                ref_hsv = cv2.cvtColor(np.array([[ref_rgb]], dtype=np.uint8), cv2.COLOR_RGB2HSV)
                ref_hsv = ref_hsv[0][0]
                color_idx = self.color_to_index[color]

                h0, s0, v0 = int(ref_hsv[0]), int(ref_hsv[1]), int(ref_hsv[2])
                h_tol, s_tol, v_tol = COLOR_TOLERANCES[color]
                h_range = np.arange(h0 - h_tol, h0 + h_tol + 1, dtype=np.int32)
                h_indices = np.unique(h_range % 180)
                s_range = np.arange(s0 - s_tol, s0 + s_tol + 1, dtype=np.int32)
                s_indices = np.unique(np.clip(s_range >> 1, 0, 127))
                v_range = np.arange(v0 - v_tol, v0 + v_tol + 1, dtype=np.int32)
                v_indices = np.unique(np.clip(v_range >> 1, 0, 127))
                ref_lut[np.ix_(h_indices, s_indices, v_indices)] = color_idx
        return ref_lut

    def segment_image(
        self, hsv_img: NDArray[np.uint8], use_numba: bool = False
    ) -> NDArray[np.uint8]:
        """
        Segment the input HSV image into color indices.
        Inputs:
            hsv_img: Input HSV image of shape (H, W, 3) with dtype
            use_numba: Whether to use numba-optimized segmentation
        Outputs:
            segmented_img: Segmented image of shape (H, W) with dtype np.uint8
        """
        if not use_numba:
            h_channel = hsv_img[:, :, 0]  # no resolution reduction for H channel
            s_channel = hsv_img[:, :, 1] >> 1  # reduce S channel resolution by half
            v_channel = hsv_img[:, :, 2] >> 1  # reduce V channel resolution by half
            segmented_img = np.asanyarray(
                self.ref_colors_mtx[h_channel, s_channel, v_channel], dtype=np.uint8
            )
            return segmented_img
        else:
            return segment_image_numba(self.ref_colors_mtx, hsv_img)  # type: ignore[arg-type]

    def visualize_segmentation(
        self,
        rgb_img: NDArray[np.uint8],
        seg_mask: NDArray[np.uint8],
        viz_color_name: Optional[str] = None,
        viz_color: Optional[Union[List[int], Tuple[int, ...]]] = None,
        alpha: float = 0.5,
    ) -> NDArray[np.uint8]:
        """
        Visualize the segmentation mask on the original RGB image.
        Inputs:
            rgb_img: Original RGB image of shape (H, W, 3) with dtype np.uint8
            seg_mask: Segmentation mask of shape (H, W) with dtype np.uint8
            viz_color_name: Specific color name to visualize. If None, visualize all colors.
            viz_color: Specific RGB color to visualize. If None, use default colors.
            alpha: Blending factor between original image and segmentation mask.
        Outputs:
            vis_img: Visualized image of shape (H, W, 3) with dtype np.uint8

        Note: this function will take around 40-50ms for a 1280x720 image. Please
        only use it for testing purposes, not for real-time processing.
        """
        vis_img = rgb_img.copy()
        for color, idx in self.color_to_index.items():
            if viz_color_name is not None and color != viz_color_name:
                continue
            if viz_color is not None:
                color_rgb = np.array(viz_color, dtype=np.uint8)
            else:
                color_rgb = np.array(COLOR_VIZ_RGB_MAP[color], dtype=np.uint8)
            mask = seg_mask == idx
            vis_img[mask] = (alpha * color_rgb + (1 - alpha) * vis_img[mask]).astype(np.uint8)
        return vis_img
