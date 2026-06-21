"""
Benchmark: Non-Local Geometric Similarity based Image Interpolation (Zhu et al.)

Paper-faithful pipeline (Section IV):

    Original HR
        |
        V  (direct 2x decimation, no pre-filter)
    Low Resolution Image (LR)
        |
        V  Proposed Method (N=60, lambda=0.01)
    Reconstructed HR
        |
        V  Compare vs Original HR  ->  PSNR / SSIM

Usage (single image):
    python benchmark.py --image ../test_images/hr/butterfly.png

Paper test set (whatever is available in ../test_images/hr):
    python benchmark.py --paper_set

Full options:
    --N 60          number of similar patches (paper: 60)
    --lam 0.01      regularization lambda    (paper: 0.01)
    --outdir results  where to save outputs and figures
    --no_vis        disable interactive matplotlib window
"""

import argparse
import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

from stage1_pipeline import load_image, create_lr, bicubic_upsample
from stage2_grid_mask import classify_pixels
from stage4_patch_extraction import extract_patch
from stage7_weight_solve import (
    build_linear_system_from_matches,
    solve_weights_directional_regularization,
)
from stage9_full_reconstruction import (
    CandidateSet,
    build_candidate_set,
    omega_cross_from_output,
    omega_plus_from_output,
    psnr,
    mae,
)


PAPER_N = 60
PAPER_LAM = 0.01

DEFAULT_PAPER_IMAGES = [
    "../test_images/hr/einstein.png",
    "../test_images/hr/butterfly.png",
    "../test_images/hr/leaves.png",
    "../test_images/hr/Bike.png",
    "../test_images/hr/lena.png",
    "../test_images/hr/Lighthouse.png",
    "../test_images/hr/f16.png",
    "../test_images/hr/goldhill.png",
]


@dataclass
class BenchmarkResult:
    name: str
    shape: Tuple[int, int]
    psnr_bicubic: float
    psnr_proposed: float
    ssim_bicubic: float
    ssim_proposed: float
    mae_bicubic: float
    mae_proposed: float
    seconds: float


def reconstruct_proposed(
    hr: np.ndarray,
    N: int = PAPER_N,
    lam: float = PAPER_LAM,
    progress_every: int = 2000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run the full paper pipeline on a single HR image.

    Returns:
        lr       : (h/2, w/2) LR image (HR[::2, ::2])
        bicubic  : (h, w) bicubic upsampled HR (baseline)
        output   : (h, w) reconstructed HR (proposed method)
    """
    lr = create_lr(hr)
    bicubic = bicubic_upsample(lr)
    h, w = hr.shape

    output = bicubic.astype(np.float64).copy()

    known_mask = (np.arange(h)[:, None] % 2 == 0) & (np.arange(w)[None, :] % 2 == 0)
    lr_up = lr.repeat(2, axis=0).repeat(2, axis=1)
    output[known_mask] = lr_up[known_mask]

    coarse_hr = bicubic
    print("  Building candidate sets on coarse HR (bicubic)...")
    cand_cross = build_candidate_set(coarse_hr, mode="cross")
    cand_plus = build_candidate_set(coarse_hr, mode="plus")
    print(f"    Candidates: cross={len(cand_cross.coords)} | plus={len(cand_plus.coords)}")

    _known, step1_pixels, step2_pixels = classify_pixels(hr.shape)

    # Step 1 (diagonal / cross): pixels at (odd, odd). Neighbors (even, even) are known.
    print(f"  Step 1 (cross): reconstructing {len(step1_pixels)} pixels...")
    t0 = time.time()
    done = 0
    for (i, j) in step1_pixels:
        if i <= 0 or i >= h - 1 or j <= 0 or j >= w - 1:
            continue
        target_patch = extract_patch(i, j, coarse_hr, mode="cross")
        if target_patch is None:
            continue
        matches = cand_cross.find_top_matches(target_coord=(i, j), target_patch=target_patch, N=N)
        phi, b = build_linear_system_from_matches(matches)
        W = solve_weights_directional_regularization(
            phi=phi, b=b, matches=matches, target_patch=target_patch, lam=lam,
        )
        omega = omega_cross_from_output(output, i, j)
        output[i, j] = float(W @ omega)
        done += 1
        if progress_every and done % progress_every == 0:
            elapsed = time.time() - t0
            rate = done / max(elapsed, 1e-6)
            remaining = (len(step1_pixels) - done) / max(rate, 1e-6)
            print(f"    step1 {done}/{len(step1_pixels)}  "
                  f"({rate:.1f} px/s, ETA ~{remaining:.0f}s)")

    # Step 2 (plus): pixels at (odd, even) or (even, odd). Uses step1 + known pixels.
    print(f"  Step 2 (plus):  reconstructing {len(step2_pixels)} pixels...")
    t1 = time.time()
    done = 0
    for (i, j) in step2_pixels:
        if i <= 0 or i >= h - 1 or j <= 0 or j >= w - 1:
            continue
        target_patch = extract_patch(i, j, coarse_hr, mode="plus")
        if target_patch is None:
            continue
        matches = cand_plus.find_top_matches(target_coord=(i, j), target_patch=target_patch, N=N)
        phi, b = build_linear_system_from_matches(matches)
        W = solve_weights_directional_regularization(
            phi=phi, b=b, matches=matches, target_patch=target_patch, lam=lam,
        )
        omega = omega_plus_from_output(output, i, j)
        output[i, j] = float(W @ omega)
        done += 1
        if progress_every and done % progress_every == 0:
            elapsed = time.time() - t1
            rate = done / max(elapsed, 1e-6)
            remaining = (len(step2_pixels) - done) / max(rate, 1e-6)
            print(f"    step2 {done}/{len(step2_pixels)}  "
                  f"({rate:.1f} px/s, ETA ~{remaining:.0f}s)")

    output = np.clip(output, 0.0, 1.0).astype(np.float32)
    return lr, bicubic, output


def compute_metrics(hr: np.ndarray, bicubic: np.ndarray, output: np.ndarray) -> dict:
    return {
        "psnr_bicubic": psnr(bicubic, hr, data_range=1.0),
        "psnr_proposed": psnr(output, hr, data_range=1.0),
        "ssim_bicubic": float(ssim(hr, bicubic, data_range=1.0)),
        "ssim_proposed": float(ssim(hr, output, data_range=1.0)),
        "mae_bicubic": mae(bicubic, hr),
        "mae_proposed": mae(output, hr),
    }


def save_outputs(
    name: str,
    hr: np.ndarray,
    lr: np.ndarray,
    bicubic: np.ndarray,
    output: np.ndarray,
    metrics: dict,
    outdir: Path,
    show: bool,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    plt.imsave(outdir / f"{name}_hr.png", hr, cmap="gray", vmin=0.0, vmax=1.0)
    plt.imsave(outdir / f"{name}_lr.png", lr, cmap="gray", vmin=0.0, vmax=1.0)
    plt.imsave(outdir / f"{name}_bicubic.png", bicubic, cmap="gray", vmin=0.0, vmax=1.0)
    plt.imsave(outdir / f"{name}_proposed.png", output, cmap="gray", vmin=0.0, vmax=1.0)

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    axes[0, 0].imshow(np.clip(hr, 0, 1), cmap="gray", vmin=0, vmax=1)
    axes[0, 0].set_title(f"Original HR  ({hr.shape[0]}x{hr.shape[1]})")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(np.clip(lr, 0, 1), cmap="gray", vmin=0, vmax=1)
    axes[0, 1].set_title(f"LR (2x decimated, no filter)  ({lr.shape[0]}x{lr.shape[1]})")
    axes[0, 1].axis("off")

    axes[1, 0].imshow(np.clip(bicubic, 0, 1), cmap="gray", vmin=0, vmax=1)
    axes[1, 0].set_title(
        f"Bicubic\nPSNR={metrics['psnr_bicubic']:.3f} dB | SSIM={metrics['ssim_bicubic']:.4f}"
    )
    axes[1, 0].axis("off")

    axes[1, 1].imshow(np.clip(output, 0, 1), cmap="gray", vmin=0, vmax=1)
    axes[1, 1].set_title(
        f"Proposed (N={PAPER_N}, lam={PAPER_LAM})\n"
        f"PSNR={metrics['psnr_proposed']:.3f} dB | SSIM={metrics['ssim_proposed']:.4f}"
    )
    axes[1, 1].axis("off")

    fig.suptitle(f"{name}: Zhu et al. Non-Local Geometric Similarity Interpolation", fontsize=12)
    fig.tight_layout()
    fig.savefig(outdir / f"{name}_compare.png", dpi=120)

    if show:
        plt.show()
    else:
        plt.close(fig)


def run_single(image_path: Path, N: int, lam: float, outdir: Path, show: bool) -> Optional[BenchmarkResult]:
    if not image_path.exists():
        print(f"[SKIP] {image_path} not found.")
        return None

    name = image_path.stem
    print(f"\n=== {name}: {image_path} ===")

    hr = load_image(str(image_path))
    print(f"  HR shape: {hr.shape}  range=[{hr.min():.3f}, {hr.max():.3f}]")

    t0 = time.time()
    lr, bicubic, output = reconstruct_proposed(hr, N=N, lam=lam)
    elapsed = time.time() - t0

    metrics = compute_metrics(hr, bicubic, output)

    print(f"  Time: {elapsed:.1f}s")
    print(f"  PSNR  bicubic = {metrics['psnr_bicubic']:.3f} dB   "
          f"proposed = {metrics['psnr_proposed']:.3f} dB   "
          f"(gain = {metrics['psnr_proposed'] - metrics['psnr_bicubic']:+.3f} dB)")
    print(f"  SSIM  bicubic = {metrics['ssim_bicubic']:.4f}     "
          f"proposed = {metrics['ssim_proposed']:.4f}")

    save_outputs(name, hr, lr, bicubic, output, metrics, outdir, show=show)

    return BenchmarkResult(
        name=name,
        shape=hr.shape,
        psnr_bicubic=metrics["psnr_bicubic"],
        psnr_proposed=metrics["psnr_proposed"],
        ssim_bicubic=metrics["ssim_bicubic"],
        ssim_proposed=metrics["ssim_proposed"],
        mae_bicubic=metrics["mae_bicubic"],
        mae_proposed=metrics["mae_proposed"],
        seconds=elapsed,
    )


def write_summary(results: List[BenchmarkResult], outdir: Path) -> None:
    if not results:
        return
    csv_path = outdir / "benchmark_summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "image", "HxW", "PSNR_bicubic(dB)", "PSNR_proposed(dB)", "dPSNR(dB)",
            "SSIM_bicubic", "SSIM_proposed", "MAE_bicubic", "MAE_proposed", "time_s",
        ])
        for r in results:
            writer.writerow([
                r.name, f"{r.shape[0]}x{r.shape[1]}",
                f"{r.psnr_bicubic:.3f}", f"{r.psnr_proposed:.3f}",
                f"{r.psnr_proposed - r.psnr_bicubic:+.3f}",
                f"{r.ssim_bicubic:.4f}", f"{r.ssim_proposed:.4f}",
                f"{r.mae_bicubic:.6f}", f"{r.mae_proposed:.6f}",
                f"{r.seconds:.1f}",
            ])

    print("\n==================== SUMMARY ====================")
    header = f"{'image':<12} {'HxW':>10} {'PSNR_bic':>9} {'PSNR_prop':>10} {'dPSNR':>7}  {'SSIM_bic':>9} {'SSIM_prop':>10}  {'time(s)':>8}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r.name:<12} {r.shape[0]:>4}x{r.shape[1]:<4}"
            f"  {r.psnr_bicubic:>8.3f} {r.psnr_proposed:>10.3f} "
            f"{r.psnr_proposed - r.psnr_bicubic:>+7.3f}  "
            f"{r.ssim_bicubic:>9.4f} {r.ssim_proposed:>10.4f}  "
            f"{r.seconds:>8.1f}"
        )
    avg_bic_p = np.mean([r.psnr_bicubic for r in results])
    avg_prop_p = np.mean([r.psnr_proposed for r in results])
    avg_bic_s = np.mean([r.ssim_bicubic for r in results])
    avg_prop_s = np.mean([r.ssim_proposed for r in results])
    print("-" * len(header))
    print(f"{'AVERAGE':<12} {'':>10}"
          f"  {avg_bic_p:>8.3f} {avg_prop_p:>10.3f} "
          f"{avg_prop_p - avg_bic_p:>+7.3f}  "
          f"{avg_bic_s:>9.4f} {avg_prop_s:>10.4f}")
    print(f"\nSaved CSV: {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", type=str, default=None,
                        help="Path to a single HR image. Ignored if --paper_set is given.")
    parser.add_argument("--paper_set", action="store_true",
                        help="Run the full 8-image paper test set from ../test_images/hr/.")
    parser.add_argument("--N", type=int, default=PAPER_N, help=f"Similar patches (paper={PAPER_N}).")
    parser.add_argument("--lam", type=float, default=PAPER_LAM, help=f"Lambda (paper={PAPER_LAM}).")
    parser.add_argument("--outdir", type=str, default="../results", help="Output directory (default: project-root results/).")
    parser.add_argument("--no_vis", action="store_true", help="Disable interactive matplotlib window.")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("Paper parameters: upsample_factor=2  N={}  lambda={}".format(args.N, args.lam))

    if args.paper_set:
        image_list = [Path(p) for p in DEFAULT_PAPER_IMAGES]
    elif args.image:
        image_list = [Path(args.image)]
    else:
        parser.error("Provide either --image PATH or --paper_set.")

    results: List[BenchmarkResult] = []
    for p in image_list:
        r = run_single(p, N=args.N, lam=args.lam, outdir=outdir, show=(not args.no_vis))
        if r is not None:
            results.append(r)

    write_summary(results, outdir)


if __name__ == "__main__":
    main()
