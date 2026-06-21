import argparse
import random
from typing import List, Tuple, Optional

import numpy as np

from stage1_pipeline import load_image, create_lr, bicubic_upsample
from stage2_grid_mask import classify_pixels
from stage4_patch_extraction import extract_patch
from stage5_patch_matching import find_similar_patches


Coord = Tuple[int, int]
Match = Tuple[Coord, np.ndarray, float]  # (coord, patch_vector(5,), score)


def _pick_one_target(hr_shape: Tuple[int, int], coarse_hr: np.ndarray, mode: str, seed: int) -> Coord:
    """
    Pick one eligible target coordinate for the given mode.
    Eligibility means extract_patch(...) is not None on coarse_hr.
    """
    random.seed(seed)
    _known, step1_pixels, step2_pixels = classify_pixels(hr_shape)
    candidates = step1_pixels if mode == "cross" else step2_pixels

    # shuffle a copy so we don't mutate classify_pixels output order
    candidates = list(candidates)
    random.shuffle(candidates)

    for (i, j) in candidates:
        if extract_patch(i, j, coarse_hr, mode=mode) is not None:
            return (i, j)

    raise RuntimeError(f"Could not find an eligible target pixel for mode={mode}.")


def build_linear_system_from_matches(matches: List[Match]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build phi and b using patch vector definition:
      patch = [center, n1, n2, n3, n4]

    Returns:
      phi: (N,4) rows = [n1,n2,n3,n4]
      b  : (N,)  values = center
    """
    if not matches:
        raise ValueError("matches is empty; cannot build phi/b.")

    phi_rows = []
    b_vals = []
    for _coord, patch_vec, _score in matches:
        patch_vec = np.asarray(patch_vec, dtype=np.float64).reshape(-1)
        if patch_vec.shape[0] < 5:
            raise ValueError(f"patch must have 5 elements, got shape {patch_vec.shape}")
        phi_rows.append(patch_vec[1:5])  # (4,)
        b_vals.append(patch_vec[0])

    phi = np.stack(phi_rows, axis=0).astype(np.float64)  # (N,4)
    b = np.asarray(b_vals, dtype=np.float64)  # (N,)
    return phi, b


def solve_weights_directional_regularization(
    phi: np.ndarray,              # (N,4)
    b: np.ndarray,                # (N,)
    matches: List[Match],        # provides Gi terms
    target_patch: np.ndarray,    # (5,)
    lam: float = 0.01,
    verbose: bool = False,
) -> np.ndarray:
    """
    Solve:
      W = argmin ||Phi W - b||^2 + (lam/N) * sum_i ||Gi - g(W)||^2

    Using the implementable closed form consistent with the paper's Eq.(10)-style:
      X = target_patch[1:]            (4,)
      A = tile(X, (4,1))             (4,4)
      Gi = patch_i[1:] - patch_i[0]  (4,)
      sum_term = sum_i (Gi - X)     (4,)

      M   = phi^T phi + lam * (A^T A)
      rhs = phi^T b   - (lam/N) * (A^T sum_term)
      W = solve(M, rhs)
    """
    phi = np.asarray(phi, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    target_patch = np.asarray(target_patch, dtype=np.float64).reshape(-1)

    if phi.ndim != 2 or phi.shape[1] != 4:
        raise ValueError(f"phi must be (N,4). Got {phi.shape}")
    if b.ndim != 1 or b.shape[0] != phi.shape[0]:
        raise ValueError(f"b must be (N,). Got {b.shape} with phi={phi.shape}")
    if target_patch.shape[0] != 5:
        raise ValueError(f"target_patch must be (5,), got {target_patch.shape}")
    if not matches:
        raise ValueError("matches is empty; cannot compute Gi.")

    N = phi.shape[0]

    # 1) X = target_patch[1:]
    X = target_patch[1:].reshape(4,)  # (4,)

    # 2-3) sum_term = sum_i (Gi - X)
    sum_term = np.zeros(4, dtype=np.float64)
    for _coord, patch_vec, _score in matches[:N]:
        patch_vec = np.asarray(patch_vec, dtype=np.float64).reshape(-1)
        Gi = patch_vec[1:5] - patch_vec[0]  # (4,)
        sum_term += (Gi - X)

    # 4) A = tile(X, (4,1))
    A = np.tile(X, (4, 1)).astype(np.float64)  # (4,4)

    # 5) M = phi^T phi + lam * (A^T A)
    M = (phi.T @ phi) + lam * (A.T @ A)

    # 6) rhs = phi^T b - (lam/N) * (A^T sum_term)
    rhs = (phi.T @ b) - (lam / float(N)) * (A.T @ sum_term)

    # 7) Solve (with fallback for numerical issues)
    try:
        W = np.linalg.solve(M, rhs)
    except np.linalg.LinAlgError:
        # If singular or ill-conditioned: least-squares solution.
        W = np.linalg.lstsq(M, rhs, rcond=None)[0]

    if verbose:
        residual = phi @ W - b
        print("W:", W)
        print("sum(W):", float(np.sum(W)))
        print("norm(M):", float(np.linalg.norm(M)))
        print("norm(rhs):", float(np.linalg.norm(rhs)))
        print("||phi W - b||:", float(np.linalg.norm(residual)))

    return W


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to input HR image")
    parser.add_argument("--N", type=int, default=60, help="Number of similar patches")
    parser.add_argument("--lam", type=float, default=0.01, help="Regularization lambda")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    hr = load_image(args.image)
    lr = create_lr(hr)
    coarse_hr = bicubic_upsample(lr)

    print("=== Stage 7: Weight Solving (Directional Gradient Regularization) ===")
    print(f"HR shape     : {hr.shape}")
    print(f"Coarse HR shp: {coarse_hr.shape}")
    print(f"N            : {args.N}")
    print(f"lambda       : {args.lam}")
    print("")

    for mode, label, seed_offset in [("cross", "Step1(cross)", 1), ("plus", "Step2(plus)", 2)]:
        target = _pick_one_target(hr.shape, coarse_hr, mode=mode, seed=args.seed + seed_offset)
        i, j = target

        target_patch = extract_patch(i, j, coarse_hr, mode=mode)
        if target_patch is None:
            raise RuntimeError(f"Unexpected: target_patch is None for target={target}, mode={mode}")

        matches = find_similar_patches(i, j, coarse_hr, mode=mode, N=args.N)
        if not matches:
            raise RuntimeError(f"No matches returned for target={target}, mode={mode}")

        phi, b = build_linear_system_from_matches(matches)

        print("----------------------------------------------")
        print(f"{label}: target pixel = ({i}, {j})")
        print(f"effective matches: {len(matches)}")
        print("phi shape:", phi.shape, "| b shape:", b.shape)
        print("Solving W...")

        W = solve_weights_directional_regularization(
            phi=phi,
            b=b,
            matches=matches,
            target_patch=target_patch,
            lam=args.lam,
            verbose=True,
        )


if __name__ == "__main__":
    main()

