import argparse
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from stage1_pipeline import load_image


Coord = Tuple[int, int]


def classify_pixels(hr_shape: Tuple[int, int]) -> Tuple[List[Coord], List[Coord], List[Coord]]:
    """
    Classify HR pixel coordinates into 3 groups for 2x upsampling grid:

    Known: both indices even
      i % 2 == 0 and j % 2 == 0

    Step1 (diagonal): both indices odd
      i % 2 == 1 and j % 2 == 1

    Step2 (plus positions): one odd and one even
      (i % 2 == 1 and j % 2 == 0) OR (i % 2 == 0 and j % 2 == 1)
    """
    h, w = hr_shape

    known_pixels: List[Coord] = []
    step1_pixels: List[Coord] = []
    step2_pixels: List[Coord] = []

    # Simple explicit iteration (no interpolation yet).
    for i in range(h):
        for j in range(w):
            if (i % 2 == 0) and (j % 2 == 0):
                known_pixels.append((i, j))
            elif (i % 2 == 1) and (j % 2 == 1):
                step1_pixels.append((i, j))
            else:
                step2_pixels.append((i, j))

    return known_pixels, step1_pixels, step2_pixels


def build_visualization_map(hr_shape: Tuple[int, int]) -> np.ndarray:
    """
    Create a grayscale visualization map:
      known  -> 1.0
      step1  -> 0.6
      step2  -> 0.3
    """
    h, w = hr_shape
    vis = np.zeros((h, w), dtype=np.float32)

    for i in range(h):
        for j in range(w):
            if (i % 2 == 0) and (j % 2 == 0):
                vis[i, j] = 1.0
            elif (i % 2 == 1) and (j % 2 == 1):
                vis[i, j] = 0.6
            else:
                vis[i, j] = 0.3
    return vis


def visualize_grid(hr_shape: Tuple[int, int], known_pixels: List[Coord], step1_pixels: List[Coord], step2_pixels: List[Coord]) -> None:
    vis = build_visualization_map(hr_shape)

    plt.figure(figsize=(6, 6))
    plt.imshow(vis, cmap="gray", vmin=0.0, vmax=1.0)
    plt.title("Stage 2: 2x Grid / Mask System")
    plt.axis("off")

    # Legend-like text for clarity (kept minimal).
    plt.text(1, 1, "known=1.0", color="white", fontsize=8, va="top", ha="left")
    plt.text(1, 15, "step1=0.6", color="white", fontsize=8, va="top", ha="left")
    plt.text(1, 29, "step2=0.3", color="white", fontsize=8, va="top", ha="left")

    plt.show()

    print(f"Known pixels: {len(known_pixels)}")
    print(f"Step1 pixels: {len(step1_pixels)}")
    print(f"Step2 pixels: {len(step2_pixels)}")

    print("Samples (i, j):")
    print("  known:", known_pixels[:10])
    print("  step1:", step1_pixels[:10])
    print("  step2:", step2_pixels[:10])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default=None, help="Optional image path to infer HR shape")
    parser.add_argument("--H", type=int, default=None, help="HR height (used if --image is not provided)")
    parser.add_argument("--W", type=int, default=None, help="HR width (used if --image is not provided)")
    args = parser.parse_args()

    if args.image is not None:
        hr = load_image(args.image)
        hr_shape = hr.shape
    else:
        if args.H is None or args.W is None:
            raise ValueError("Provide either --image or both --H and --W.")
        hr_shape = (args.H, args.W)

    known_pixels, step1_pixels, step2_pixels = classify_pixels(hr_shape)
    visualize_grid(hr_shape, known_pixels, step1_pixels, step2_pixels)


if __name__ == "__main__":
    main()

