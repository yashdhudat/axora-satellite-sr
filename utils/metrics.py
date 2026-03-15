"""
Image Quality Metrics
PSNR (Peak Signal-to-Noise Ratio) and SSIM (Structural Similarity Index)
Team AXORA | GSA Pan India Hackathon
"""

import numpy as np
from typing import Union
from PIL import Image


def calculate_psnr(img1: np.ndarray, img2: np.ndarray, max_val: float = 255.0) -> float:
    """
    Calculate PSNR between two images.
    Higher is better (typical satellite SR target: > 30 dB)

    Args:
        img1: Image array (H, W, C) or (H, W), uint8 or float
        img2: Reference image array, same shape
        max_val: Maximum pixel value (255 for uint8)

    Returns:
        PSNR value in dB
    """
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    if img1.shape != img2.shape:
        raise ValueError(f"Image shapes must match: {img1.shape} vs {img2.shape}")

    mse = np.mean((img1 - img2) ** 2)
    if mse < 1e-10:
        return float('inf')

    return 20 * np.log10(max_val / np.sqrt(mse))


def calculate_ssim(img1: np.ndarray, img2: np.ndarray,
                   window_size: int = 11, sigma: float = 1.5,
                   k1: float = 0.01, k2: float = 0.03,
                   max_val: float = 255.0) -> float:
    """
    Calculate SSIM between two images.
    Range: [-1, 1], higher is better (target: > 0.85 for satellite SR)

    Args:
        img1, img2: Image arrays (H, W, C) or (H, W)
        window_size: Gaussian window size
        sigma: Gaussian standard deviation
        k1, k2: SSIM stability constants
        max_val: Maximum pixel value

    Returns:
        Mean SSIM value
    """
    # Convert to grayscale for multi-channel
    if img1.ndim == 3:
        img1 = _rgb_to_y(img1)
        img2 = _rgb_to_y(img2)

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    C1 = (k1 * max_val) ** 2
    C2 = (k2 * max_val) ** 2

    kernel = _gaussian_kernel(window_size, sigma)

    # Local statistics
    mu1 = _convolve(img1, kernel)
    mu2 = _convolve(img2, kernel)
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = _convolve(img1 ** 2, kernel) - mu1_sq
    sigma2_sq = _convolve(img2 ** 2, kernel) - mu2_sq
    sigma12 = _convolve(img1 * img2, kernel) - mu1_mu2

    ssim_map = (
        (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    ) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    return float(np.mean(ssim_map))


def _rgb_to_y(img: np.ndarray) -> np.ndarray:
    """Convert RGB to Y channel (luminance) for SSIM."""
    return (0.299 * img[:, :, 0] +
            0.587 * img[:, :, 1] +
            0.114 * img[:, :, 2])


def _gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    """Create 2D Gaussian kernel."""
    coords = np.arange(size) - size // 2
    g = np.exp(-coords ** 2 / (2 * sigma ** 2))
    kernel = np.outer(g, g)
    return kernel / kernel.sum()


def _convolve(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Simple valid convolution using stride tricks."""
    from scipy.ndimage import convolve
    return convolve(img, kernel, mode='reflect')


def calculate_metrics_report(sr_img: Union[np.ndarray, Image.Image],
                              hr_img: Union[np.ndarray, Image.Image],
                              bicubic_img: Union[np.ndarray, Image.Image] = None) -> dict:
    """
    Full metrics report comparing SR vs HR and optionally bicubic baseline.

    Returns:
        dict with PSNR/SSIM for both SR and bicubic (if provided)
    """
    def to_array(img):
        if isinstance(img, Image.Image):
            return np.array(img.convert('RGB'))
        return img

    sr = to_array(sr_img)
    hr = to_array(hr_img)

    report = {
        'sr_psnr': calculate_psnr(sr, hr),
        'sr_ssim': calculate_ssim(sr, hr),
    }

    if bicubic_img is not None:
        bic = to_array(bicubic_img)
        if bic.shape == hr.shape:
            report['bicubic_psnr'] = calculate_psnr(bic, hr)
            report['bicubic_ssim'] = calculate_ssim(bic, hr)
            report['psnr_gain'] = report['sr_psnr'] - report['bicubic_psnr']
            report['ssim_gain'] = report['sr_ssim'] - report['bicubic_ssim']

    return report


def print_metrics_table(metrics: dict):
    """Pretty print metrics comparison table."""
    print(f"\n{'='*45}")
    print(f"{'METRIC':<15} {'SR (OURS)':<15} {'BICUBIC':<15}")
    print(f"{'-'*45}")

    psnr_sr = metrics.get('sr_psnr', 0)
    psnr_bic = metrics.get('bicubic_psnr', 0)
    ssim_sr = metrics.get('sr_ssim', 0)
    ssim_bic = metrics.get('bicubic_ssim', 0)

    print(f"{'PSNR (dB)':<15} {psnr_sr:<15.2f} {psnr_bic:<15.2f}")
    print(f"{'SSIM':<15} {ssim_sr:<15.4f} {ssim_bic:<15.4f}")

    if 'psnr_gain' in metrics:
        print(f"{'-'*45}")
        print(f"{'PSNR gain':<15} +{metrics['psnr_gain']:.2f} dB")
        print(f"{'SSIM gain':<15} +{metrics['ssim_gain']:.4f}")

    print(f"{'='*45}")
