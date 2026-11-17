import cv2
import numpy as np
import math

# Four transformation matrices defined in the base RGb color-space model
FILTER_MATRICES = [
    np.array([-1.0, 0.5, 0.5], dtype=np.float32),  # Matrix a
    np.array([0.5, -1.0, 0.5], dtype=np.float32),  # Matrix b
    np.array([0.0, -1.0, 1.0], dtype=np.float32),  # Matrix c
    np.array([-1.0, 0.0, 1.0], dtype=np.float32),  # Matrix d
]


def apply_base_filter(image_rgb: np.ndarray, delta_e_user: float):
    """
    Global blue light filter (Baseline method).

    Args:
        image_rgb (np.ndarray): Input RGB image (H, W, 3) in uint8 format.
        delta_e_user (float): User-defined color difference control parameter
                              (e.g., -5 for limited, 0 for optimal, +10/+40/+100 for enhanced).

    Returns:
        tuple: (filtered_img, best_T, fc)
    """
    img_float = image_rgb.astype(np.float32)
    best_sum = -1.0
    best_T = None

    for matrix in FILTER_MATRICES:
        T = (
            img_float[:, :, 0] * matrix[0]
            + img_float[:, :, 1] * matrix[1]
            + img_float[:, :, 2] * matrix[2]
        )
        T[T < 0] = 0.0
        current_sum = np.sum(T)
        if current_sum > best_sum:
            best_sum = current_sum
            best_T = T

    delta_e_image = np.mean(best_T)
    if delta_e_image == 0:
        delta_e_image = 1e-5

    fc = 1.0
    if delta_e_user < 0 and delta_e_image > abs(delta_e_user):
        fc = abs(delta_e_user) / delta_e_image
    elif delta_e_user > 0:
        fc = 1.0 + (abs(delta_e_user) / delta_e_image)

    T_final = best_T * fc
    filtered_img = img_float.copy()
    filtered_img[:, :, 2] -= T_final

    return np.clip(filtered_img, 0, 255).astype(np.uint8), best_T, fc


def apply_smooth_local_filter(
    image_rgb: np.ndarray,
    delta_e_user: float,
    block_size: int,
    interp_method: int = cv2.INTER_LINEAR,
) -> np.ndarray:
    """
    Proposed BLISS Method: Blue Light Interpolated Spatial Smoothing.
    Computes optimal transformation matrices per spatial grid block and applies
    smooth spatial interpolation across the matrix field before filtering.

    Args:
        image_rgb (np.ndarray): Input RGB image (H, W, 3) in uint8 format.
        delta_e_user (float): User-defined Delta E control parameter.
        block_size (int): Spatial grid block size (e.g., 32 or 64).
        interp_method (int): OpenCV interpolation flag (default: cv2.INTER_LINEAR).

    Returns:
        np.ndarray: Filtered RGB image in uint8 format.
    """
    h, w, _ = image_rgb.shape
    img_float = image_rgb.astype(np.float32)

    grid_h = math.ceil(h / block_size)
    grid_w = math.ceil(w / block_size)
    fc_grid = np.zeros((grid_h, grid_w, 3), dtype=np.float32)

    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            block = image_rgb[y : y + block_size, x : x + block_size]

            best_sum = -1.0
            best_matrix = FILTER_MATRICES[0]

            for matrix in FILTER_MATRICES:
                T = (
                    block[:, :, 0] * matrix[0]
                    + block[:, :, 1] * matrix[1]
                    + block[:, :, 2] * matrix[2]
                )
                current_sum = np.sum(T[T > 0])
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_matrix = matrix

            fc_grid[y // block_size, x // block_size] = best_matrix

    # Smooth spatial interpolation of the matrix decision grid to eliminate block artifacts
    smoothed_matrices = cv2.resize(fc_grid, (w, h), interpolation=interp_method)

    T_map = (
        img_float[:, :, 0] * smoothed_matrices[:, :, 0]
        + img_float[:, :, 1] * smoothed_matrices[:, :, 1]
        + img_float[:, :, 2] * smoothed_matrices[:, :, 2]
    )
    T_map[T_map < 0] = 0.0

    delta_e_image = np.mean(T_map)
    if delta_e_image == 0:
        delta_e_image = 1e-5

    fc = 1.0
    if delta_e_user < 0 and delta_e_image > abs(delta_e_user):
        fc = abs(delta_e_user) / delta_e_image
    elif delta_e_user > 0:
        fc = 1.0 + (abs(delta_e_user) / delta_e_image)

    filtered_img = img_float.copy()
    filtered_img[:, :, 2] -= T_map * fc

    return np.clip(filtered_img, 0, 255).astype(np.uint8)