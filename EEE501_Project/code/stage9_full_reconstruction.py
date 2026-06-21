import argparse
from dataclasses import dataclass
from typing import List, Tuple, Optional
from pathlib import Path

import numpy as np
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

from stage1_pipeline import load_image, create_lr, bicubic_upsample
from stage2_grid_mask import classify_pixels
from stage4_patch_extraction import extract_patch
from stage7_weight_solve import (
    build_linear_system_from_matches,
    solve_weights_directional_regularization,
)


Coord = Tuple[int, int]
PatchVec = np.ndarray  # (5,)


def psnr(x: np.ndarray, y: np.ndarray, data_range: float = 1.0) -> float:
    mse = float(np.mean((x - y) ** 2))
    if mse == 0.0:
        return float("inf")
    return 10.0 * float(np.log10((data_range ** 2) / mse))


def mae(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(np.abs(x - y)))


def omega_cross_from_output(output: np.ndarray, i: int, j: int) -> np.ndarray:
    # Fixed order must match stage4_patch_extraction "cross" neighbor order.
    # (i-1, j-1), (i-1, j+1), (i+1, j-1), (i+1, j+1)
    return np.array(
        [
            output[i - 1, j - 1],
            output[i - 1, j + 1],
            output[i + 1, j - 1],
            output[i + 1, j + 1],
        ],
        dtype=np.float64,
    )


def omega_plus_from_output(output: np.ndarray, i: int, j: int) -> np.ndarray:
    # Fixed order must match stage4_patch_extraction "plus" neighbor order.
    # (i-1, j), (i+1, j), (i, j-1), (i, j+1)
    return np.array(
        [
            output[i - 1, j],
            output[i + 1, j],
            output[i, j - 1],
            output[i, j + 1],
        ],
        dtype=np.float64,
    )


@dataclass
class CandidateSet:
    coords: List[Coord]          # length M
    patches: np.ndarray          # (M,5) float64

    def find_top_matches(
        self,
        target_coord: Coord,
        target_patch: np.ndarray,
        N: int,
    ) -> List[Tuple[Coord, np.ndarray, float]]:
        """
        Brute-force (vectorized) matching:
          E1 = L2 distance between 5D patches
          E2 = Euclidean distance between coords
          score = E1 * sqrt(E2)
        Returns top-N matches sorted by increasing score.
        """
        tc = np.array(target_coord, dtype=np.float64)  # (2,)
        patch_t = np.asarray(target_patch, dtype=np.float64).reshape(5,)  # (5,)

        coords_arr = np.array(self.coords, dtype=np.float64)  # (M,2)

        diffs = self.patches - patch_t[None, :]               # (M,5)
        e1 = np.linalg.norm(diffs, axis=1)                   # (M,)

        dxy = coords_arr - tc[None, :]                       # (M,2)
        e2 = np.linalg.norm(dxy, axis=1)                     # (M,)

        # score = E1 * sqrt(E2)
        score = e1 * np.sqrt(e2)

        # Exclude same coord (if present in candidate set)
        same = (coords_arr[:, 0] == tc[0]) & (coords_arr[:, 1] == tc[1])
        if np.any(same):
            score[same] = np.inf

        k = min(N, score.shape[0])
        idx_part = np.argpartition(score, k - 1)[:k]
        idx_sorted = idx_part[np.argsort(score[idx_part])]

        matches: List[Tuple[Coord, np.ndarray, float]] = []
        for idx in idx_sorted:
            matches.append((self.coords[idx], self.patches[idx].astype(np.float32), float(score[idx])))
        return matches


def build_candidate_set(coarse_hr: np.ndarray, mode: str) -> CandidateSet:
    """
    Build candidate patch vectors for all centers compatible with the mode,
    using boundary-safe patch extraction on coarse_hr.
    """
    h, w = coarse_hr.shape
    candidates: List[Coord] = []
    patches: List[np.ndarray] = []

    # For both cross (odd,odd) and plus (odd,even)/(even,odd), patch needs +/-1 neighbors.
    # So we only accept i in [1, h-2], j in [1, w-2].
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            if mode == "cross":
                if (i % 2 == 1) and (j % 2 == 1):
                    pass
                else:
                    continue
            elif mode == "plus":
                # Step2: one odd and one even
                if (i % 2 == 1 and j % 2 == 0) or (i % 2 == 0 and j % 2 == 1):
                    pass
                else:
                    continue
            else:
                raise ValueError("mode must be 'cross' or 'plus'")

            patch = extract_patch(i, j, coarse_hr, mode=mode)
            if patch is None:
                continue
            candidates.append((i, j))
            patches.append(patch.astype(np.float64))

    if not patches:
        raise RuntimeError(f"No candidates found for mode={mode}. Check image size/boundaries.")

    patch_mat = np.stack(patches, axis=0)  # (M,5)
    return CandidateSet(coords=candidates, patches=patch_mat)


def visualize_triplet(hr: np.ndarray, bicubic: np.ndarray, output: np.ndarray) -> None:
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(np.clip(hr, 0, 1), cmap="gray")
    plt.title("Original HR")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(np.clip(bicubic, 0, 1), cmap="gray")
    plt.title("Bicubic (coarse)")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(np.clip(output, 0, 1), cmap="gray")
    plt.title("Reconstruction (Proposed)")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--N", type=int, default=60)
    parser.add_argument("--lam", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no_vis", action="store_true", help="Disable matplotlib visualization.")
    parser.add_argument("--save_output", type=str, default="", help="Optional path to save reconstructed image.")

    # Safety limits for runtime; set to 0 or negative for "no limit".
    parser.add_argument("--max_step1", type=int, default=0, help="Max number of Step1 pixels to reconstruct (0=all).")
    parser.add_argument("--max_step2", type=int, default=0, help="Max number of Step2 pixels to reconstruct (0=all).")

    args = parser.parse_args()

    hr = load_image(args.image)
    lr = create_lr(hr)
    bicubic = bicubic_upsample(lr)

    h, w = hr.shape
    output = bicubic.astype(np.float64).copy()  # fill with bicubic as fallback for boundaries

    # Known pixels: even-even from LR
    known_mask = (np.arange(h)[:, None] % 2 == 0) & (np.arange(w)[None, :] % 2 == 0)
    # Upsample LR by repeating pixels (no filtering) to align with HR grid.
    lr_up = lr.repeat(2, axis=0).repeat(2, axis=1)  # (h,w)
    output[known_mask] = lr_up[known_mask]

    # Candidate sets (on coarse_hr) for brute-force matching
    print("Building candidate patch sets (coarse HR)...")
    coarse_hr = bicubic  # already coarse HR (from bicubic upsample)
    cand_cross = build_candidate_set(coarse_hr, mode="cross")
    cand_plus = build_candidate_set(coarse_hr, mode="plus")
    print(f"Candidates cross: {len(cand_cross.coords)} | plus: {len(cand_plus.coords)}")

    _known, step1_pixels, step2_pixels = classify_pixels(hr.shape)

    # Step 1 (cross / diagonal) reconstruction
    print("Reconstructing Step1 (cross) pixels...")
    step1_count = 0
    step1_limit = args.max_step1 if args.max_step1 > 0 else None

    for (i, j) in step1_pixels:
        # Skip boundaries where neighbors/patch can't be extracted.
        if i <= 0 or i >= h - 1 or j <= 0 or j >= w - 1:
            continue

        target_patch = extract_patch(i, j, coarse_hr, mode="cross")
        if target_patch is None:
            continue

        matches = cand_cross.find_top_matches(target_coord=(i, j), target_patch=target_patch, N=args.N)
        phi, b = build_linear_system_from_matches(matches)
        W = solve_weights_directional_regularization(
            phi=phi,
            b=b,
            matches=matches,
            target_patch=target_patch,
            lam=args.lam,
            verbose=False,
        )

        omega = omega_cross_from_output(output, i, j)
        output[i, j] = float(W @ omega)

        step1_count += 1
        if step1_limit is not None and step1_count >= step1_limit:
            break

    # Step 2 (plus) reconstruction
    print("Reconstructing Step2 (plus) pixels...")
    step2_count = 0
    step2_limit = args.max_step2 if args.max_step2 > 0 else None

    for (i, j) in step2_pixels:
        if i <= 0 or i >= h - 1 or j <= 0 or j >= w - 1:
            continue

        target_patch = extract_patch(i, j, coarse_hr, mode="plus")
        if target_patch is None:
            continue

        matches = cand_plus.find_top_matches(target_coord=(i, j), target_patch=target_patch, N=args.N)
        phi, b = build_linear_system_from_matches(matches)
        W = solve_weights_directional_regularization(
            phi=phi,
            b=b,
            matches=matches,
            target_patch=target_patch,
            lam=args.lam,
            verbose=False,
        )

        omega = omega_plus_from_output(output, i, j)
        output[i, j] = float(W @ omega)

        step2_count += 1
        if step2_limit is not None and step2_count >= step2_limit:
            break

    output = np.clip(output, 0.0, 1.0).astype(np.float32)

    # Metrics
    mae_out = mae(output, hr)
    mae_bic = mae(bicubic, hr)
    psnr_out = psnr(output, hr, data_range=1.0)
    psnr_bic = psnr(bicubic, hr, data_range=1.0)
    ssim_out = float(ssim(hr, output, data_range=1.0))
    ssim_bic = float(ssim(hr, bicubic, data_range=1.0))

    print("==================================================")
    print(f"Reconstructed Step1 pixels: {step1_count} / {len(step1_pixels)}")
    print(f"Reconstructed Step2 pixels: {step2_count} / {len(step2_pixels)}")
    print(f"MAE  (output vs HR) : {mae_out:.6f}")
    print(f"MAE  (bicubic vs HR): {mae_bic:.6f}")
    print(f"PSNR (output vs HR) : {psnr_out:.3f} dB")
    print(f"PSNR (bicubic vs HR): {psnr_bic:.3f} dB")
    print(f"SSIM (output vs HR)  : {ssim_out:.6f}")
    print(f"SSIM (bicubic vs HR) : {ssim_bic:.6f}")

    if args.save_output:
        out_path = Path(args.save_output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.imsave(out_path, output, cmap="gray", vmin=0.0, vmax=1.0)
        print(f"Saved output image: {out_path}")

    if not args.no_vis:
        visualize_triplet(hr, bicubic, output)


if __name__ == "__main__":
    main()

