"""
stage10_parallel_benchmark.py

High-performance parallel benchmark for the Zhu et al. Non-Local Geometric
Similarity based image interpolation scheme. Numerically equivalent to
stage9_full_reconstruction.py, but:

  * Uses multiprocessing (one process per logical core) to parallelize the two
    reconstruction phases.
  * Vectorizes the inner loop (patch matching, linear-system build, and
    regularized normal equations) with NumPy only.
  * Shares the large read-only arrays (coarse HR, candidate patches, candidate
    coordinates) once per worker via a Pool initializer -- no per-pixel or
    per-chunk pickling of big arrays.
  * Chunk scheduling with imap_unordered for load balancing and progress.

Algorithmic logic is identical to stage9:
  Pipeline:   HR --2x decimate--> LR --bicubic up--> coarse HR
              --proposed method--> Reconstructed HR  vs HR: PSNR/SSIM.

Paper parameters (defaults): upsample_factor=2, N=60, lambda=0.01, no pre-filter.

CLI:
    python stage10_parallel_benchmark.py --image lena_gray.bmp --N 60 --lam 0.01 --workers auto
"""

from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

from stage1_pipeline import load_image, create_lr, bicubic_upsample
from stage2_grid_mask import classify_pixels
from stage9_full_reconstruction import build_candidate_set, psnr, mae


PAPER_N = 60
PAPER_LAM = 0.01

Coord = Tuple[int, int]


# ---------------------------------------------------------------------------
# Worker-side globals. These are populated exactly once per worker process by
# _init_worker via the Pool initializer. They are never mutated afterwards.
# On Windows (spawn) each worker gets its own copy; on Linux (fork) they are
# copy-on-write.
# ---------------------------------------------------------------------------
_coarse_hr: Optional[np.ndarray] = None      # (H, W) float32
_output: Optional[np.ndarray] = None         # (H, W) float64 snapshot
_cross_coords: Optional[np.ndarray] = None   # (Mc, 2) float64
_cross_patches: Optional[np.ndarray] = None  # (Mc, 5) float64
_plus_coords: Optional[np.ndarray] = None    # (Mp, 2) float64
_plus_patches: Optional[np.ndarray] = None   # (Mp, 5) float64
_N: int = 60
_lam: float = 0.01


def _init_worker(
    coarse_hr: np.ndarray,
    output_snapshot: np.ndarray,
    cross_coords: np.ndarray,
    cross_patches: np.ndarray,
    plus_coords: np.ndarray,
    plus_patches: np.ndarray,
    N: int,
    lam: float,
) -> None:
    """Populate worker-side globals (runs once per worker process)."""
    global _coarse_hr, _output
    global _cross_coords, _cross_patches, _plus_coords, _plus_patches
    global _N, _lam

    _coarse_hr = coarse_hr
    _output = output_snapshot
    _cross_coords = cross_coords
    _cross_patches = cross_patches
    _plus_coords = plus_coords
    _plus_patches = plus_patches
    _N = int(N)
    _lam = float(lam)

    # Guard against uncontrolled oversubscription if MKL/OpenBLAS spin up
    # their own thread pool in every worker.
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, "1")


# ---------------------------------------------------------------------------
# Inline, vectorized primitives (equivalent math to stages 4, 5, 7, 9).
# ---------------------------------------------------------------------------
def _extract_patch_inline(image: np.ndarray, i: int, j: int, mode: str) -> Optional[np.ndarray]:
    """
    Build the 5-element patch vector [center, n1, n2, n3, n4] for the given
    mode. Returns None if any neighbor is out of bounds.

    Neighbor order matches stage3.get_cross_neighbors / get_plus_neighbors,
    so results are identical to stage4.extract_patch.
    """
    h, w = image.shape
    if mode == "cross":
        if i - 1 < 0 or i + 1 >= h or j - 1 < 0 or j + 1 >= w:
            return None
        return np.array(
            [
                image[i, j],
                image[i - 1, j - 1], image[i - 1, j + 1],
                image[i + 1, j - 1], image[i + 1, j + 1],
            ],
            dtype=np.float64,
        )
    elif mode == "plus":
        if i - 1 < 0 or i + 1 >= h or j - 1 < 0 or j + 1 >= w:
            return None
        return np.array(
            [
                image[i, j],
                image[i - 1, j], image[i + 1, j],
                image[i, j - 1], image[i, j + 1],
            ],
            dtype=np.float64,
        )
    else:
        raise ValueError(f"mode must be 'cross' or 'plus', got {mode}")


def _reconstruct_one(
    i: int,
    j: int,
    mode: str,
    cand_coords: np.ndarray,
    cand_patches: np.ndarray,
) -> Optional[float]:
    """
    Reconstruct a single HR pixel value at (i, j) using the same math as
    stage7.solve_weights_directional_regularization, but fully vectorized
    over matches (no Python loop over N).
    """
    coarse_hr = _coarse_hr
    output = _output
    N = _N
    lam = _lam

    h, w = coarse_hr.shape
    if i <= 0 or i >= h - 1 or j <= 0 or j >= w - 1:
        return None

    target_patch = _extract_patch_inline(coarse_hr, i, j, mode)
    if target_patch is None:
        return None

    # Matching: score = E1 * sqrt(E2) where
    #   E1 = L2 distance between 5D patches
    #   E2 = Euclidean distance between coordinates
    diffs = cand_patches - target_patch[None, :]                # (M, 5)
    e1 = np.sqrt(np.einsum("ij,ij->i", diffs, diffs))          # (M,)
    dxy = cand_coords - np.array([i, j], dtype=np.float64)[None, :]  # (M, 2)
    e2 = np.sqrt(np.einsum("ij,ij->i", dxy, dxy))              # (M,)
    score = e1 * np.sqrt(e2)                                   # (M,)

    # Exclude the target itself if it is present as a candidate.
    same = (cand_coords[:, 0] == i) & (cand_coords[:, 1] == j)
    if same.any():
        score[same] = np.inf

    k = min(N, score.shape[0])
    idx = np.argpartition(score, k - 1)[:k]
    sel = cand_patches[idx]                                    # (k, 5)

    # Linear system: phi (k,4), b (k,)
    phi = sel[:, 1:5]
    b_vec = sel[:, 0]

    # Regularization block (paper Eq. 7-10 style, same as stage7)
    X = target_patch[1:5]                                      # (4,)
    Gi = sel[:, 1:5] - sel[:, 0:1]                             # (k, 4)
    sum_term = (Gi - X[None, :]).sum(axis=0)                   # (4,)

    A = np.broadcast_to(X, (4, 4))                             # (4, 4), read-only
    M_mat = phi.T @ phi + lam * (A.T @ A)                      # (4, 4)
    rhs = phi.T @ b_vec - (lam / float(k)) * (A.T @ sum_term)  # (4,)

    try:
        W = np.linalg.solve(M_mat, rhs)
    except np.linalg.LinAlgError:
        W = np.linalg.lstsq(M_mat, rhs, rcond=None)[0]

    # Omega = the 4 actual neighbors in the (partially) reconstructed output.
    if mode == "cross":
        omega = np.array(
            [
                output[i - 1, j - 1], output[i - 1, j + 1],
                output[i + 1, j - 1], output[i + 1, j + 1],
            ],
            dtype=np.float64,
        )
    else:
        omega = np.array(
            [
                output[i - 1, j], output[i + 1, j],
                output[i, j - 1], output[i, j + 1],
            ],
            dtype=np.float64,
        )

    return float(W @ omega)


def _worker_step1(coords_chunk: List[Coord]) -> List[Tuple[int, int, float]]:
    assert _cross_coords is not None and _cross_patches is not None
    out: List[Tuple[int, int, float]] = []
    for (i, j) in coords_chunk:
        v = _reconstruct_one(i, j, "cross", _cross_coords, _cross_patches)
        if v is not None:
            out.append((i, j, v))
    return out


def _worker_step2(coords_chunk: List[Coord]) -> List[Tuple[int, int, float]]:
    assert _plus_coords is not None and _plus_patches is not None
    out: List[Tuple[int, int, float]] = []
    for (i, j) in coords_chunk:
        v = _reconstruct_one(i, j, "plus", _plus_coords, _plus_patches)
        if v is not None:
            out.append((i, j, v))
    return out


# ---------------------------------------------------------------------------
# Driver helpers
# ---------------------------------------------------------------------------
def _chunks(seq: List[Coord], size: int) -> List[List[Coord]]:
    return [seq[i : i + size] for i in range(0, len(seq), size)]


def _resolve_workers(arg: str) -> int:
    if arg == "auto":
        return max(1, (os.cpu_count() or 2) - 1)
    try:
        n = int(arg)
    except ValueError as exc:
        raise ValueError(f"--workers must be an int or 'auto', got {arg!r}") from exc
    return max(1, n)


def _run_phase(
    step_name: str,
    worker_fn,
    coords: List[Coord],
    chunk_size: int,
    workers: int,
    init_args: tuple,
) -> List[Tuple[int, int, float]]:
    """Run one parallel phase (step1 or step2). Returns list of (i, j, v)."""
    chunks = _chunks(coords, chunk_size)
    total_px = len(coords)
    n_chunks = len(chunks)
    results: List[Tuple[int, int, float]] = []

    t0 = time.time()
    report_every = max(1, n_chunks // 20)
    completed = 0
    got_px = 0

    ctx = mp.get_context("spawn")
    with ctx.Pool(
        processes=workers,
        initializer=_init_worker,
        initargs=init_args,
    ) as pool:
        for chunk_result in pool.imap_unordered(worker_fn, chunks):
            results.extend(chunk_result)
            completed += 1
            got_px += len(chunk_result)

            if completed % report_every == 0 or completed == n_chunks:
                elapsed = time.time() - t0
                frac = completed / n_chunks
                eta = elapsed * (1 - frac) / max(frac, 1e-6)
                rate = got_px / max(elapsed, 1e-6)
                print(
                    f"    {step_name}: {int(100 * frac):3d}%  "
                    f"({got_px}/{total_px} px, {rate:7.0f} px/s, ETA ~{eta:5.1f}s)"
                )

    return results


@dataclass
class BenchmarkResult:
    name: str
    shape: Tuple[int, int]
    workers: int
    seconds_total: float
    seconds_step1: float
    seconds_step2: float
    psnr_bicubic: float
    psnr_proposed: float
    ssim_bicubic: float
    ssim_proposed: float
    mae_bicubic: float
    mae_proposed: float


def run_benchmark(
    image_path: Path,
    N: int,
    lam: float,
    workers: int,
    chunk_size: int,
    outdir: Path,
    show: bool,
) -> BenchmarkResult:
    print(f"\n=== {image_path.stem}: {image_path} ===")
    print(f"Paper params: factor=2  N={N}  lambda={lam}  workers={workers}")

    # ---- Preprocessing ----
    hr = load_image(str(image_path))
    lr = create_lr(hr)
    bicubic = bicubic_upsample(lr)
    h, w = hr.shape
    print(f"  HR shape   : {hr.shape}  range=[{hr.min():.3f}, {hr.max():.3f}]")
    print(f"  LR shape   : {lr.shape}  (HR[::2, ::2], no pre-filter)")

    output = bicubic.astype(np.float64).copy()
    known_mask = (np.arange(h)[:, None] % 2 == 0) & (np.arange(w)[None, :] % 2 == 0)
    lr_up = lr.repeat(2, axis=0).repeat(2, axis=1)
    output[known_mask] = lr_up[known_mask]

    coarse_hr = bicubic.astype(np.float32)

    # ---- Candidate sets (built once, reused by all workers) ----
    print("  Building candidate sets on coarse HR...")
    cand_cross = build_candidate_set(coarse_hr, "cross")
    cand_plus = build_candidate_set(coarse_hr, "plus")
    cross_coords = np.asarray(cand_cross.coords, dtype=np.float64)
    cross_patches = np.asarray(cand_cross.patches, dtype=np.float64)
    plus_coords = np.asarray(cand_plus.coords, dtype=np.float64)
    plus_patches = np.asarray(cand_plus.patches, dtype=np.float64)
    print(f"    cross candidates: {cross_coords.shape[0]}")
    print(f"    plus  candidates: {plus_coords.shape[0]}")

    # ---- Work queues ----
    _known, step1_pixels, step2_pixels = classify_pixels(hr.shape)
    print(f"    step1 pixels    : {len(step1_pixels)}")
    print(f"    step2 pixels    : {len(step2_pixels)}")

    t_all = time.time()

    # ---------- Step 1 (cross) ----------
    print("  Step 1 (cross) parallel phase...")
    init_args_step1 = (
        coarse_hr, output,
        cross_coords, cross_patches,
        plus_coords, plus_patches,
        N, lam,
    )
    t1 = time.time()
    step1_results = _run_phase(
        step_name="Step1", worker_fn=_worker_step1,
        coords=step1_pixels, chunk_size=chunk_size,
        workers=workers, init_args=init_args_step1,
    )
    for (i, j, v) in step1_results:
        output[i, j] = v
    sec_step1 = time.time() - t1
    print(f"    -> Step 1 done in {sec_step1:.1f}s, {len(step1_results)} pixels reconstructed.")

    # ---------- Step 2 (plus), using updated output ----------
    print("  Step 2 (plus)  parallel phase...")
    init_args_step2 = (
        coarse_hr, output,
        cross_coords, cross_patches,
        plus_coords, plus_patches,
        N, lam,
    )
    t2 = time.time()
    step2_results = _run_phase(
        step_name="Step2", worker_fn=_worker_step2,
        coords=step2_pixels, chunk_size=chunk_size,
        workers=workers, init_args=init_args_step2,
    )
    for (i, j, v) in step2_results:
        output[i, j] = v
    sec_step2 = time.time() - t2
    print(f"    -> Step 2 done in {sec_step2:.1f}s, {len(step2_results)} pixels reconstructed.")

    sec_total = time.time() - t_all
    output = np.clip(output, 0.0, 1.0).astype(np.float32)

    # ---- Metrics ----
    r = BenchmarkResult(
        name=image_path.stem,
        shape=hr.shape,
        workers=workers,
        seconds_total=sec_total,
        seconds_step1=sec_step1,
        seconds_step2=sec_step2,
        psnr_bicubic=psnr(bicubic, hr, 1.0),
        psnr_proposed=psnr(output, hr, 1.0),
        ssim_bicubic=float(ssim(hr, bicubic, data_range=1.0)),
        ssim_proposed=float(ssim(hr, output, data_range=1.0)),
        mae_bicubic=mae(bicubic, hr),
        mae_proposed=mae(output, hr),
    )

    # ---- Persist outputs ----
    outdir.mkdir(parents=True, exist_ok=True)
    name = r.name
    plt.imsave(outdir / f"{name}_hr.png", hr, cmap="gray", vmin=0.0, vmax=1.0)
    plt.imsave(outdir / f"{name}_lr.png", lr, cmap="gray", vmin=0.0, vmax=1.0)
    plt.imsave(outdir / f"{name}_bicubic.png", bicubic, cmap="gray", vmin=0.0, vmax=1.0)
    plt.imsave(outdir / f"{name}_proposed.png", output, cmap="gray", vmin=0.0, vmax=1.0)

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes[0, 0].imshow(np.clip(hr, 0, 1), cmap="gray", vmin=0, vmax=1)
    axes[0, 0].set_title(f"Original HR ({h}x{w})"); axes[0, 0].axis("off")
    axes[0, 1].imshow(np.clip(lr, 0, 1), cmap="gray", vmin=0, vmax=1)
    axes[0, 1].set_title(f"LR ({lr.shape[0]}x{lr.shape[1]}, direct 2x decimation)"); axes[0, 1].axis("off")
    axes[1, 0].imshow(np.clip(bicubic, 0, 1), cmap="gray", vmin=0, vmax=1)
    axes[1, 0].set_title(f"Bicubic\nPSNR={r.psnr_bicubic:.3f} dB | SSIM={r.ssim_bicubic:.4f}")
    axes[1, 0].axis("off")
    axes[1, 1].imshow(np.clip(output, 0, 1), cmap="gray", vmin=0, vmax=1)
    axes[1, 1].set_title(
        f"Proposed (N={N}, lam={lam})\n"
        f"PSNR={r.psnr_proposed:.3f} dB | SSIM={r.ssim_proposed:.4f}"
    )
    axes[1, 1].axis("off")
    fig.suptitle(f"{name}: Non-local geometric similarity interpolation (parallel)")
    fig.tight_layout()
    fig.savefig(outdir / f"{name}_compare.png", dpi=120)
    if show:
        plt.show()
    else:
        plt.close(fig)

    # ---- CSV row for this run ----
    csv_path = outdir / "benchmark_summary.csv"
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                "image", "HxW", "workers", "time_total_s",
                "time_step1_s", "time_step2_s",
                "PSNR_bicubic_dB", "PSNR_proposed_dB", "dPSNR_dB",
                "SSIM_bicubic", "SSIM_proposed",
                "MAE_bicubic", "MAE_proposed",
            ])
        writer.writerow([
            r.name, f"{r.shape[0]}x{r.shape[1]}", r.workers,
            f"{r.seconds_total:.2f}", f"{r.seconds_step1:.2f}", f"{r.seconds_step2:.2f}",
            f"{r.psnr_bicubic:.3f}", f"{r.psnr_proposed:.3f}",
            f"{r.psnr_proposed - r.psnr_bicubic:+.3f}",
            f"{r.ssim_bicubic:.4f}", f"{r.ssim_proposed:.4f}",
            f"{r.mae_bicubic:.6f}", f"{r.mae_proposed:.6f}",
        ])

    return r


def _print_summary(r: BenchmarkResult) -> None:
    print("\n================ RESULTS ================")
    print(f"Image        : {r.name}  ({r.shape[0]}x{r.shape[1]})")
    print(f"CPU workers  : {r.workers}")
    print(f"Total time   : {r.seconds_total:7.2f} s")
    print(f"  Step 1     : {r.seconds_step1:7.2f} s")
    print(f"  Step 2     : {r.seconds_step2:7.2f} s")
    print("-------- Quality (vs original HR) --------")
    print(f"  MAE   bicubic  = {r.mae_bicubic:.6f}     proposed = {r.mae_proposed:.6f}")
    print(f"  PSNR  bicubic  = {r.psnr_bicubic:7.3f} dB  proposed = {r.psnr_proposed:7.3f} dB "
          f"(gain {r.psnr_proposed - r.psnr_bicubic:+.3f} dB)")
    print(f"  SSIM  bicubic  = {r.ssim_bicubic:7.4f}     proposed = {r.ssim_proposed:7.4f}")
    print("=========================================")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--image", type=str, required=True, help="Path to an HR image.")
    parser.add_argument("--N", type=int, default=PAPER_N,
                        help=f"Number of similar patches (paper={PAPER_N}).")
    parser.add_argument("--lam", type=float, default=PAPER_LAM,
                        help=f"Regularization lambda (paper={PAPER_LAM}).")
    parser.add_argument("--workers", type=str, default="auto",
                        help="Number of worker processes. 'auto' = cpu_count-1.")
    parser.add_argument("--chunk_size", type=int, default=500,
                        help="Pixels per work chunk (default 500).")
    parser.add_argument("--outdir", type=str, default="results",
                        help="Output directory for reconstructed images and CSV.")
    parser.add_argument("--no_vis", action="store_true",
                        help="Do not open the matplotlib window.")
    args = parser.parse_args()

    workers = _resolve_workers(args.workers)
    outdir = Path(args.outdir)
    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    r = run_benchmark(
        image_path=image_path,
        N=args.N,
        lam=args.lam,
        workers=workers,
        chunk_size=args.chunk_size,
        outdir=outdir,
        show=not args.no_vis,
    )
    _print_summary(r)


if __name__ == "__main__":
    mp.freeze_support()
    main()
