import argparse
import random
from typing import List, Tuple, Optional

import numpy as np

from stage1_pipeline import load_image
from stage2_grid_mask import classify_pixels


Coord = Tuple[int, int]


def _in_bounds(i: int, j: int, h: int, w: int) -> bool:
    return 0 <= i < h and 0 <= j < w


def get_cross_neighbors(i: int, j: int, image: np.ndarray) -> List[Tuple[Coord, float]]:
    """
    Cross (diagonal) neighbors for Step1 pixels at (i, j):
      (i-1, j-1), (i-1, j+1), (i+1, j-1), (i+1, j+1)
    Boundary neighbors outside the image are skipped.
    """
    h, w = image.shape
    candidates = [(i - 1, j - 1), (i - 1, j + 1), (i + 1, j - 1), (i + 1, j + 1)]
    out: List[Tuple[Coord, float]] = []
    for ni, nj in candidates:
        if _in_bounds(ni, nj, h, w):
            out.append(((ni, nj), float(image[ni, nj])))
    return out


def get_plus_neighbors(i: int, j: int, image: np.ndarray) -> List[Tuple[Coord, float]]:
    """
    Plus (horizontal/vertical) neighbors for Step2 pixels at (i, j):
      (i-1, j), (i+1, j), (i, j-1), (i, j+1)
    Boundary neighbors outside the image are skipped.
    """
    h, w = image.shape
    candidates = [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]
    out: List[Tuple[Coord, float]] = []
    for ni, nj in candidates:
        if _in_bounds(ni, nj, h, w):
            out.append(((ni, nj), float(image[ni, nj])))
    return out


def _has_full_neighbors(neighbors: List[Tuple[Coord, float]], expected: int = 4) -> bool:
    return len(neighbors) == expected


def _sample_eligible_pixels(
    coords: List[Coord],
    sample_k: int,
    neighbor_fn,
    image: np.ndarray,
) -> List[Coord]:
    """
    Sample k coordinates where neighbor_fn(i,j,image) returns exactly 4 neighbors.
    """
    eligible: List[Coord] = []
    for (i, j) in coords:
        neigh = neighbor_fn(i, j, image)
        if _has_full_neighbors(neigh, expected=4):
            eligible.append((i, j))

    if not eligible:
        return []

    k = min(sample_k, len(eligible))
    return random.sample(eligible, k)


def print_neighbors(step_name: str, pixel: Coord, neighbors: List[Tuple[Coord, float]]) -> None:
    i, j = pixel
    print(f"{step_name} pixel: ({i}, {j})")
    print("Neighbors:")
    if not neighbors:
        print("  (none - boundary)")
        return
    for (ni, nj), val in neighbors:
        print(f"  ({ni}, {nj}) -> {val:.6f}")


def run_tests(hr: np.ndarray, seed: Optional[int] = 0) -> None:
    if seed is not None:
        random.seed(seed)

    known_pixels, step1_pixels, step2_pixels = classify_pixels(hr.shape)

    # We only test step1 / step2 because known pixels are not interpolated.
    step1_samples = _sample_eligible_pixels(step1_pixels, sample_k=5, neighbor_fn=get_cross_neighbors, image=hr)
    step2_samples = _sample_eligible_pixels(step2_pixels, sample_k=5, neighbor_fn=get_plus_neighbors, image=hr)

    print("=== Stage 3: Neighbor Extraction Tests ===")
    print(f"HR shape: {hr.shape}")
    print(f"Eligible Step1 samples: {len(step1_samples)} / requested 5")
    print(f"Eligible Step2 samples: {len(step2_samples)} / requested 5")
    print("")

    for pix in step1_samples:
        neigh = get_cross_neighbors(pix[0], pix[1], hr)
        print_neighbors("Step1", pix, neigh)
        print("")

    for pix in step2_samples:
        neigh = get_plus_neighbors(pix[0], pix[1], hr)
        print_neighbors("Step2", pix, neigh)
        print("")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to HR image")
    args = parser.parse_args()

    hr = load_image(args.image)
    run_tests(hr, seed=0)


if __name__ == "__main__":
    main()

