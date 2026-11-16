# src/metrics.py
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

try:
    import pyiqa
except ImportError:
    pyiqa = None

_NR_IQA_CACHE = {}


def _get_nr_iqa_metrics(device: torch.device):
    if pyiqa is None:
        return None, None
    if "niqe" not in _NR_IQA_CACHE:
        _NR_IQA_CACHE["niqe"] = pyiqa.create_metric("niqe", device=device)
    if "piqe" not in _NR_IQA_CACHE:
        _NR_IQA_CACHE["piqe"] = pyiqa.create_metric("piqe", device=device)
    return _NR_IQA_CACHE["niqe"], _NR_IQA_CACHE["piqe"]


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
    """Computes full-reference, perceptual color, and no-reference IQA metrics."""
    # 1. Structural Similarity (Overall RGB & Blue Channel) & PSNR
    ssim_overall, _ = ssim(orig, filtered, channel_axis=-1, data_range=255, full=True)
    ssim_blue, _ = ssim(orig[:, :, 2], filtered[:, :, 2], data_range=255, full=True)
    psnr_val = float(psnr(orig, filtered, data_range=255))

    # 2. Blue Light Absorption Ratio
    b_orig_float = orig[:, :, 2].astype(np.float32)
    b_filt_float = filtered[:, :, 2].astype(np.float32)
    blue_absorption = float(1.0 - (np.mean(b_filt_float) / (np.mean(b_orig_float) + 1e-5)))

    # 3. Perceptual Color Difference (CIE Delta E 2000)
    orig_lab = rgb2lab(orig)
    filt_lab = rgb2lab(filtered)
    delta_e = float(np.mean(deltaE_ciede2000(orig_lab, filt_lab)))

    # 4. 3D Perceptual Gamut Volume Difference in CIELAB space
    vol_orig = get_gamut_volume(orig_lab)
    vol_filt = get_gamut_volume(filt_lab)
    volume_diff = abs(1.0 - (vol_filt / (vol_orig + 1e-5))) if vol_orig > 0 else 0.0

    # 5. Gradient Distortion & No-Reference IQA (GMSD, BRISQUE, NIQE, PIQE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    orig_t = torch.tensor(orig, device=device).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    filt_t = torch.tensor(filtered, device=device).permute(2, 0, 1).unsqueeze(0).float() / 255.0

    gmsd_val = float(piq.gmsd(filt_t, orig_t).item()) if piq is not None else 0.0
    brisque_val = float(piq.brisque(filt_t).item()) if piq is not None else 0.0

    niqe_metric, piqe_metric = _get_nr_iqa_metrics(device)
    if niqe_metric is not None and piqe_metric is not None:
        with torch.no_grad():
            niqe_val = float(niqe_metric(filt_t).item())
            piqe_val = float(piqe_metric(filt_t).item())
    else:
        niqe_val, piqe_val = 0.0, 0.0

    return {
        "PSNR (Higher is Better)": psnr_val,
        "SSIM Overall (Higher is Better)": float(ssim_overall),
        "SSIM Blue Channel (Higher is Better)": float(ssim_blue),
        "GMSD (Lower is Better)": gmsd_val,
        "Blue Absorption (Higher is Better)": blue_absorption,
        "Volume Difference (Lower is Better)": volume_diff,
        "Delta E 2000 (Lower is Better)": delta_e,
        "BRISQUE (Lower is Better)": brisque_val,
        "NIQE (Lower is Better)": niqe_val,
        "PIQE (Lower is Better)": piqe_val,
        "Time (s) (Lower is Better)": float(process_time),
    }