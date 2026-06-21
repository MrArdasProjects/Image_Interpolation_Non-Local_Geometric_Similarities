import argparse
import random
from typing import List, Tuple, Optional

import numpy as np

from stage1_pipeline import load_image, create_lr, bicubic_upsample
from stage2_grid_mask import classify_pixels
from stage3_neighbors import get_cross_neighbors, get_plus_neighbors
from stage4_patch_extraction import extract_patch
from stage5_patch_matching import find_similar_patches
from stage7_weight_solve import (
    build_linear_system_from_matches,
    solve_weights_directional_regularization,
)


Coord = Tuple[int, int]


def reconstruct_pixel(W: np.ndarray, neighbor_vals: List[float]) -> float:
    """
    x = W^T * Omega, where neighbor_vals are in the same order as patch[1:].
    """
    W = np.asarray(W, dtype=np.float64).reshape(-1)
    omega = np.asarray(neighbor_vals, dtype=np.float64).reshape(-1)
    if W.shape[0] != 4 or omega.shape[0] != 4:
        raise ValueError(f"Expected W and omega to be length 4. Got W={W.shape}, omega={omega.shape}")
    return float(W @ omega)


def _pick_one_eligible_target(hr: np.ndarray, coarse_hr: np.ndarray, mode: str, seed: int) -> Coord:
    random.seed(seed)
    _known, step1_pixels, step2_pixels = classify_pixels(hr.shape)
    candidates = step1_pixels if mode == "cross" else step2_pixels

    candidates = list(candidates)
    random.shuffle(candidates)
    for (i, j) in candidates:
        if extract_patch(i, j, coarse_hr, mode=mode) is not None:
            return (i, j)
    raise RuntimeError(f"Could not find eligible target for mode={mode}.")


def _get_omega_from_hr(i: int, j: int, hr: np.ndarray, mode: str) -> Optional[List[float]]:
    """
    Omega = the 4 known neighbor samples around (i,j).
    Use HR values at neighbor positions (these are exactly the same as LR samples because LR was hr[::2,::2]).
    """
    if mode == "cross":
        neighbors = get_cross_neighbors(i, j, hr)
    elif mode == "plus":
        neighbors = get_plus_neighbors(i, j, hr)
    else:
        raise ValueError("mode must be 'cross' or 'plus'")

    if len(neighbors) != 4:
        return None
    # neighbor values in fixed order
    return [float(v) for (_coord, v) in neighbors]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to HR image")
    parser.add_argument("--N", type=int, default=60, help="Number of similar patches")
    parser.add_argument("--lam", type=float, default=0.01, help="Regularization lambda")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    hr = load_image(args.image)
    lr = create_lr(hr)
    coarse_hr = bicubic_upsample(lr)

    print("=== Stage 8: Pixel Reconstruction ===")
    print(f"HR shape    : {hr.shape}")
    print(f"coarse HR   : {coarse_hr.shape}")
    print(f"N           : {args.N}")
    print(f"lambda      : {args.lam}")
    print("")

    for mode, label, seed_offset in [("cross", "Step1(cross)", 10), ("plus", "Step2(plus)", 20)]:
        target = _pick_one_eligible_target(hr, coarse_hr, mode=mode, seed=args.seed + seed_offset)
        i, j = target

        # matches + linear system
        target_patch = extract_patch(i, j, coarse_hr, mode=mode)
        if target_patch is None:
            raise RuntimeError("Unexpected: target_patch is None for eligible target.")

        matches = find_similar_patches(i, j, coarse_hr, mode=mode, N=args.N)
        if not matches:
            raise RuntimeError("No matches returned.")

        phi, b = build_linear_system_from_matches(matches)
        W = solve_weights_directional_regularization(
            phi=phi,
            b=b,
            matches=matches,
            target_patch=target_patch,
            lam=args.lam,
        )

        omega_vals = _get_omega_from_hr(i, j, hr, mode=mode)
        if omega_vals is None:
            raise RuntimeError("Unexpected: omega neighbors incomplete on HR for eligible target.")

        xhat = reconstruct_pixel(W, omega_vals)
        x_true = float(hr[i, j])

        print("----------------------------------------------")
        print(f"{label}: target pixel = ({i}, {j})")
        print(f"Original     : {x_true:.6f}")
        print(f"Reconstructed : {xhat:.6f}")
        print(f"Difference   : {abs(x_true - xhat):.6f}")


if __name__ == "__main__":
    main()

