import cv2
import numpy as np
import torch
from scipy.spatial import ConvexHull
from skimage.color import deltaE_ciede2000, rgb2lab
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

try:
    import piq
except ImportError:
    piq = None


def get_gamut_volume(image_lab: np.ndarray) -> float:
    """Computes the 3D perceptual color gamut convex hull volume in CIELAB space."""
    small_img = cv2.resize(image_lab, (64, 64))
    pixels = small_img.reshape(-1, 3)
    unique_pixels = np.unique(np.round(pixels, 2), axis=0)

    if len(unique_pixels) < 4:
        return 0.0
    try:
        hull = ConvexHull(unique_pixels)
        return float(hull.volume)
    except Exception:
        return 0.0


def evaluate_metrics(orig: np.ndarray, filtered: np.ndarray, process_time: float) -> dict:
    """
    Evaluates full-reference, perceptual color, and no-reference quality metrics.

    Args:
        orig (np.ndarray): Original RGB image (uint8).
        filtered (np.ndarray): Filtered RGB image (uint8).
        process_time (float): Execution time in seconds.

    Returns:
        dict: Dictionary containing all computed metrics.
    """
    # 1. Structural Similarity (Overall RGB and Blue-channel specific)
    ssim_overall = ssim(orig, filtered, channel_axis=-1, data_range=255)
    ssim_blue = ssim(orig[:, :, 2], filtered[:, :, 2], data_range=255)
    psnr_val = psnr(orig, filtered, data_range=255)

    # 2. Blue Light Absorption Ratio
    b_orig_float = orig[:, :, 2].astype(np.float32)
    b_filt_float = filtered[:, :, 2].astype(np.float32)
    blue_absorption = 1.0 - (np.mean(b_filt_float) / (np.mean(b_orig_float) + 1e-5))

    # 3. Perceptual Color Difference (CIE Delta E 2000)
    orig_lab = rgb2lab(orig)
    filt_lab = rgb2lab(filtered)
    delta_e_2000 = float(np.mean(deltaE_ciede2000(orig_lab, filt_lab)))

    # 4. 3D Perceptual Gamut Volume Difference in CIELAB space
    vol_orig = get_gamut_volume(orig_lab)
    vol_filt = get_gamut_volume(filt_lab)
    volume_diff = abs(1.0 - (vol_filt / (vol_orig + 1e-5))) if vol_orig > 0 else 0.0

    # 5. Gradient & No-Reference IQA Metrics (GMSD, BRISQUE, NIQE, PIQE)
    gmsd_val, brisque_val, niqe_val, piqe_val = 0.0, 0.0, 0.0, 0.0
    if piq is not None:
        orig_t = torch.tensor(orig).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        filt_t = torch.tensor(filtered).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        try:
            gmsd_val = piq.gmsd(filt_t, orig_t, data_range=1.0).item()
            brisque_val = piq.brisque(filt_t, data_range=1.0).item()
            niqe_val = piq.niqe(filt_t, data_range=1.0).item() if hasattr(piq, "niqe") else 0.0
            piqe_val = piq.piqe(filt_t, data_range=1.0).item() if hasattr(piq, "piqe") else 0.0
        except Exception:
            pass

    return {
        "PSNR (Higher is Better)": float(psnr_val),
        "SSIM Overall (Higher is Better)": float(ssim_overall),
        "SSIM Blue Channel (Higher is Better)": float(ssim_blue),
        "GMSD (Lower is Better)": float(gmsd_val),
        "Blue Absorption (Higher is Better)": float(blue_absorption),
        "Volume Difference (Lower is Better)": float(volume_diff),
        "Delta E 2000 (Lower is Better)": delta_e_2000,
        "BRISQUE (Lower is Better)": float(brisque_val),
        "NIQE (Lower is Better)": float(niqe_val),
        "PIQE (Lower is Better)": float(piqe_val),
        "Time (s) (Lower is Better)": float(process_time),
    }