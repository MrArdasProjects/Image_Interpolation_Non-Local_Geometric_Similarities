import argparse
import random
from typing import List, Tuple

import numpy as np

from stage1_pipeline import load_image, create_lr, bicubic_upsample
from stage2_grid_mask import classify_pixels
from stage4_patch_extraction import extract_patch
from stage5_patch_matching import find_similar_patches


Coord = Tuple[int, int]
Match = Tuple[Coord, np.ndarray, float]  # (coord, patch_vector, score)


def build_linear_system(matches: List[Match]) -> Tuple[np.ndarray, np.ndarray]:
    """
    matches: list of (coord, patch, score)
      patch = [center, n1, n2, n3, n4] (shape: (5,))

    Returns:
      phi: (N, 4) where each row is [n1, n2, n3, n4]
      b  : (N,)   where each value is center
    """
    if not matches:
        raise ValueError("matches is empty; cannot build linear system.")

    phi_rows = []
    b_vals = []

    for _coord, patch, _score in matches:
        patch = np.asarray(patch)
        if patch.shape[0] < 5:
            raise ValueError(f"patch must have at least 5 elements, got shape={patch.shape}")
        phi_rows.append(patch[1:5])
        b_vals.append(patch[0])

    phi = np.stack(phi_rows, axis=0).astype(np.float32)  # (N,4)
    b = np.asarray(b_vals, dtype=np.float32)  # (N,)
    return phi, b


def _pick_one_target(hr_shape: Tuple[int, int], coarse_hr: np.ndarray, mode: str, seed: int) -> Coord:
    """
    Pick one eligible target coordinate for the given mode:
      - mode='cross' -> Step1 pixels (odd,odd)
      - mode='plus'  -> Step2 pixels (odd,even or even,odd)
    Eligibility here means extract_patch(...) returns not None.
    """
    random.seed(seed)
    _known, step1_pixels, step2_pixels = classify_pixels(hr_shape)

    candidates = step1_pixels if mode == "cross" else step2_pixels
    random.shuffle(candidates)

    for (i, j) in candidates:
        if extract_patch(i, j, coarse_hr, mode=mode) is not None:
            return (i, j)

    raise RuntimeError("Could not find an eligible target pixel (boundary too tight).")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to HR image")
    parser.add_argument("--N", type=int, default=10, help="Number of matched patches for linear system")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    hr = load_image(args.image)
    lr = create_lr(hr)
    coarse_hr = bicubic_upsample(lr)

    # Test both Step1 and Step2 to verify parity+ordering consistency.
    for mode, label, seed_offset in [("cross", "Step1(cross)", 1), ("plus", "Step2(plus)", 2)]:
        target = _pick_one_target(hr.shape, coarse_hr, mode=mode, seed=args.seed + seed_offset)
        i, j = target

        matches = find_similar_patches(i, j, coarse_hr, mode=mode, N=args.N)
        if len(matches) == 0:
            raise RuntimeError(f"No matches returned for target={target}, mode={mode}")

        phi, b = build_linear_system(matches)

        print("==================================================")
        print(f"{label}: target pixel = ({i}, {j})")
        print(f"matches used: {len(matches)} (requested N={args.N})")

        print(f"phi shape: {phi.shape}")
        print(f"b shape  : {b.shape}")

        print("phi (first 3 rows):")
        for r in range(min(3, phi.shape[0])):
            row = phi[r].tolist()
            row_str = ", ".join(f"{x:.6f}" for x in row)
            print(f"  [{row_str}]")

        print("b (first 3 values):")
        for r in range(min(3, b.shape[0])):
            print(f"  b[{r}] = {b[r]:.6f}")


if __name__ == "__main__":
    main()

