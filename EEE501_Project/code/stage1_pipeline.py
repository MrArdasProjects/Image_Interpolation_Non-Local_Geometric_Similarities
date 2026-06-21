import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


def load_image(path: str) -> np.ndarray:
    """
    Load image from disk, convert to grayscale if needed,
    return float32 array normalized to [0, 1].

    Note: To make the 2x downsample/upsample pipeline consistent, we trim
    HR dimensions to even numbers (if needed).
    """
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")

    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = img.astype(np.float32)

    # Normalize to [0, 1]
    mn, mx = float(img.min()), float(img.max())
    if mx <= 1.0 and mn >= 0.0:
        hr = img
    else:
        # Common cases: 8-bit or 16-bit; otherwise use min-max.
        if mx <= 255.0:
            hr = img / 255.0
        elif mx <= 65535.0:
            hr = img / 65535.0
        else:
            denom = (mx - mn) if (mx - mn) != 0 else 1.0
            hr = (img - mn) / denom

    hr = hr.astype(np.float32)

    # Ensure even H/W so that lr = hr[::2, ::2] and bicubic_upsample matches hr shape.
    h, w = hr.shape
    h2, w2 = h - (h % 2), w - (w % 2)
    if (h2, w2) != (h, w):
        print(f"Warning: trimming HR from ({h}, {w}) to even shape ({h2}, {w2}) for factor-2 pipeline.")
        hr = hr[:h2, :w2]

    return hr


def create_lr(hr: np.ndarray) -> np.ndarray:
    """
    Downsample by factor 2 without filtering:
      lr = hr[::2, ::2]
    """
    return hr[::2, ::2].astype(np.float32)


def bicubic_upsample(lr: np.ndarray) -> np.ndarray:
    """
    Upsample LR back to original size using bicubic interpolation.

    Assumes the original HR size is exactly 2x the LR size.
    """
    h_lr, w_lr = lr.shape
    h_hr, w_hr = h_lr * 2, w_lr * 2
    up = cv2.resize(lr, (w_hr, h_hr), interpolation=cv2.INTER_CUBIC)
    return up.astype(np.float32)


def visualize_images(hr: np.ndarray, lr: np.ndarray, bicubic: np.ndarray) -> None:
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(np.clip(hr, 0, 1), cmap="gray")
    plt.title("Original HR")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(np.clip(lr, 0, 1), cmap="gray")
    plt.title("LR (no-filter, 2x downsample)")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(np.clip(bicubic, 0, 1), cmap="gray")
    plt.title("Bicubic (coarse HR)")
    plt.axis("off")

    plt.suptitle("Aşama 1: Image Pipeline")
    plt.tight_layout()
    plt.show()

    print("Shapes:")
    print(f"  HR      : {hr.shape}")
    print(f"  LR      : {lr.shape}")
    print(f"  Bicubic : {bicubic.shape}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, type=str, help="Path to input image")
    args = parser.parse_args()

    image_path = Path(args.image)
    hr = load_image(str(image_path))
    lr = create_lr(hr)
    bicubic = bicubic_upsample(lr)
    visualize_images(hr, lr, bicubic)


if __name__ == "__main__":
    main()

