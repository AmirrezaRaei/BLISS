import cv2
import matplotlib.pyplot as plt
import numpy as np


def load_image_rgb(image_path: str) -> np.ndarray:
    """Loads an image from disk and converts it from BGR to RGB."""
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not load image at: {image_path}")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def save_image_rgb(image_rgb: np.ndarray, save_path: str) -> None:
    """Saves an RGB image to disk."""
    img_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, img_bgr)


def plot_visual_comparison(
    images: list,
    titles: list,
    suptitle: str = "",
    save_path: str = None,
) -> None:
    """Plots multiple images side-by-side for visual comparison."""
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5.5))
    if n == 1:
        axes = [axes]

    if suptitle:
        fig.suptitle(suptitle, fontsize=15, fontweight="bold")

    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()