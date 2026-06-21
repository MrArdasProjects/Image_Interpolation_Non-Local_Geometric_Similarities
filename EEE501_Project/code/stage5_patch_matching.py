import argparse
import random
from typing import List, Tuple, Optional

import numpy as np

from stage1_pipeline import load_image, create_lr, bicubic_upsample
from stage4_patch_extraction import extract_patch


Coord = Tuple[int, int]
Match = Tuple[Coord, np.ndarray, float]  # (coord, patch_vector, score)


def _candidates_for_mode(hr_shape: Tuple[int, int], mode: str) -> List[Coord]:
    """
    Generate candidate patch centers with full 4-neighbor support (boundary-safe),
    based on the 2x grid parity rules:
      - "cross" corresponds to Step1: (odd, odd)
      - "plus"  corresponds to Step2: (odd, even) or (even, odd)
    """
    h, w = hr_shape
    if h < 3 or w < 3:
        return []

    # Neighbors use i±1 and/or j±1, so i must be in [1, h-2], j in [1, w-2].
    i_min, i_max = 1, h - 1
    j_min, j_max = 1, w - 1  # exclusive upper bound

    out: List[Coord] = []
    if mode == "cross":
        for i in range(i_min, i_max, 2):  # odd i
            for j in range(j_min, j_max, 2):  # odd j
                out.append((i, j))
    elif mode == "plus":
        for i in range(i_min, i_max):
            if i % 2 == 1:  # odd i -> j must be even
                for j in range(2, j_max, 2):  # even j
                    out.append((i, j))
            else:  # even i -> j must be odd
                for j in range(1, j_max, 2):  # odd j
                    out.append((i, j))
    else:
        raise ValueError("mode must be either 'cross' or 'plus'")

    return out


def find_similar_patches(
    i: int,
    j: int,
    image: np.ndarray,
    mode: str,
    N: int = 60,
) -> List[Match]:
    """
    Brute-force search for N most similar patches to the target patch at (i, j).

    Scoring (as requested):
      E1 = Euclidean distance between 5D patch vectors
      E2 = Euclidean distance between coordinates
      score E = E1 * sqrt(E2)

    Returns:
      list of (coord, patch, score), sorted ascending by score.
    """
    hr_shape = image.shape
    target_patch = extract_patch(i, j, image, mode=mode)
    if target_patch is None:
        return []

    candidates = _candidates_for_mode(hr_shape, mode=mode)

    target_coord = np.array([i, j], dtype=np.float32)
    matches: List[Match] = []

    for (ci, cj) in candidates:
        if ci == i and cj == j:
            continue

        ref_patch = extract_patch(ci, cj, image, mode=mode)
        if ref_patch is None:
            continue

        e1 = float(np.linalg.norm(target_patch - ref_patch))  # L2 on 5 elements

        dxy = np.array([ci, cj], dtype=np.float32) - target_coord
        e2 = float(np.linalg.norm(dxy))  # Euclidean distance on coordinates
        if e2 <= 0.0:
            continue

        score = e1 * float(np.sqrt(e2))
        matches.append(((ci, cj), ref_patch, score))

    matches.sort(key=lambda x: x[2])
    return matches[:N]


def _pick_random_target(hr: np.ndarray, mode: str, seed: int) -> Optional[Coord]:
    random.seed(seed)
    candidates = _candidates_for_mode(hr.shape, mode=mode)
    if not candidates:
        return None

    # Keep sampling until extract_patch works (should always work due to boundary-safe candidates).
    for _ in range(50):
        (i, j) = random.choice(candidates)
        if extract_patch(i, j, hr, mode=mode) is not None:
            return (i, j)
    return None


def _print_top_matches(target: Coord, matches: List[Match], top_k: int = 5) -> None:
    i, j = target
    print(f"Target pixel: ({i}, {j})")
    print("")
    print("Top matches:")
    for (coord, _patch, score) in matches[:top_k]:
        x, y = coord
        print(f"({x}, {y}) -> {score:.6f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--N", type=int, default=60, help="Number of similar patches to return")
    parser.add_argument("--top_k", type=int, default=5, help="How many matches to print")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    hr = load_image(args.image)
    lr = create_lr(hr)
    coarse_hr = bicubic_upsample(lr)

    step1_target = _pick_random_target(hr, mode="cross", seed=args.seed + 1)
    step2_target = _pick_random_target(hr, mode="plus", seed=args.seed + 2)

    if step1_target is None or step2_target is None:
        raise RuntimeError("Could not find valid target pixels (boundary too small?).")

    print("=== Stage 5: Non-local Patch Matching (Brute Force, no weights) ===")
    print(f"HR shape: {hr.shape}")
    print(f"Coarse HR shape: {coarse_hr.shape}")
    print("")

    # Use coarse HR for patch matching, as suggested by the paper workflow.
    print("--- Step1 (mode='cross') ---")
    matches1 = find_similar_patches(step1_target[0], step1_target[1], coarse_hr, mode="cross", N=args.N)
    _print_top_matches(step1_target, matches1, top_k=args.top_k)
    print("")

    print("--- Step2 (mode='plus') ---")
    matches2 = find_similar_patches(step2_target[0], step2_target[1], coarse_hr, mode="plus", N=args.N)
    _print_top_matches(step2_target, matches2, top_k=args.top_k)


if __name__ == "__main__":
    main()

