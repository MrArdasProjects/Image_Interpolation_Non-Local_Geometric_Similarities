import argparse
import random
from typing import Optional, Tuple, List

import numpy as np

from stage1_pipeline import load_image
from stage2_grid_mask import classify_pixels
from stage3_neighbors import get_cross_neighbors, get_plus_neighbors


Coord = Tuple[int, int]


def extract_patch(i: int, j: int, image: np.ndarray, mode: str) -> Optional[np.ndarray]:
    """
    Build a 5-element patch vector:
      [center, n1, n2, n3, n4]

    mode:
      "cross" -> diagonal neighbors (i±1, j±1) for Step1 pixels
      "plus"  -> plus neighbors (i±1, j) and (i, j±1) for Step2 pixels

    Returns:
      patch: np.ndarray shape (5,) if 4 neighbors exist
      None  : if any neighbor is out of bounds
    """
    if mode == "cross":
        neighbors = get_cross_neighbors(i, j, image)  # list of ((ni,nj), val) in fixed order
    elif mode == "plus":
        neighbors = get_plus_neighbors(i, j, image)  # list of ((ni,nj), val) in fixed order
    else:
        raise ValueError("mode must be either 'cross' or 'plus'")

    if len(neighbors) != 4:
        return None

    center = float(image[i, j])
    vals = [float(v) for (_, v) in neighbors]
    return np.array([center, *vals], dtype=np.float32)


def _eligible_pixels(
    pixels: List[Coord],
    image: np.ndarray,
    mode: str,
) -> List[Coord]:
    eligible: List[Coord] = []
    for (i, j) in pixels:
        if extract_patch(i, j, image, mode) is not None:
            eligible.append((i, j))
    return eligible


def print_patch(pixel: Coord, patch: Optional[np.ndarray]) -> None:
    i, j = pixel
    if patch is None:
        print(f"Pixel: ({i}, {j})")
        print("Patch: None (boundary / incomplete neighbors)")
        return

    vals = [float(x) for x in patch.tolist()]
    vals_str = ", ".join(f"{v:.6f}" for v in vals)
    print(f"Pixel: ({i}, {j})")
    print(f"Patch: [{vals_str}]")


def run_tests(hr: np.ndarray, seed: int = 0) -> None:
    random.seed(seed)

    _, step1_pixels, step2_pixels = classify_pixels(hr.shape)

    step1_eligible = _eligible_pixels(step1_pixels, hr, mode="cross")
    step2_eligible = _eligible_pixels(step2_pixels, hr, mode="plus")

    step1_samples = random.sample(step1_eligible, k=min(5, len(step1_eligible)))
    step2_samples = random.sample(step2_eligible, k=min(5, len(step2_eligible)))

    print("=== Stage 4: Patch Extraction Tests ===")
    print(f"HR shape: {hr.shape}")
    print(f"Eligible Step1 (cross) count: {len(step1_eligible)}")
    print(f"Eligible Step2 (plus) count : {len(step2_eligible)}")
    print("")

    print("--- Step1 samples (mode='cross') ---")
    for pix in step1_samples:
        patch = extract_patch(pix[0], pix[1], hr, mode="cross")
        print_patch(pix, patch)
        print("")

    print("--- Step2 samples (mode='plus') ---")
    for pix in step2_samples:
        patch = extract_patch(pix[0], pix[1], hr, mode="plus")
        print_patch(pix, patch)
        print("")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to HR image")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    hr = load_image(args.image)
    run_tests(hr, seed=args.seed)


if __name__ == "__main__":
    main()

