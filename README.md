# 🖼️ Image Interpolation Based on Non-Local Geometric Similarities

[![Python Version](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Course](https://img.shields.io/badge/MSc%20Project-EEE%20501-red)](https://www.ieu.edu.tr/)

This repository contains a from-scratch Python re-implementation of the advanced image super-resolution framework proposed by **Zhu et al.** in the paper: *"Image Interpolation Based on Non-Local Geometric Similarities"*. 

Developed as an **MSc Term Project** for the *Applied Digital Image Processing* course, this framework rejects strictly localized pixel boundaries, instead learning adaptive upscaling weights by mapping recurring structures across the entire image context.

---

## 🎓 Academic Context

| Course | Institution | Instructor | Developer |
| :--- | :--- | :--- | :--- |
| **EEE 501** — Applied Digital Image Processing | Izmir University of Economics | Doç. Dr. Mehmet TÜRKAN | Mehmet Arda UÇAR |

---

## 📌 Problem Definition & Motivation

Traditional upscaling techniques (like bilinear or bicubic interpolation) rely only on a tiny, immediate neighborhood (e.g., nearest 4 or 16 pixels). When dealing with high-frequency details, sharp transitions, or complex textures, this local focus inherently causes:
* **Blurred edges** along sharp intensity transitions.
* **Jagged artifacts (jaggies)** in diagonal structures.
* **Loss of fine texture details** in repeating patterns.

### 🔍 The Non-Local Philosophy
Natural images are packed with repeating patterns that reappear in entirely different, non-local regions. This algorithm maps the global image structure to find the top $N=60$ most geometrically similar patches. By solving a **Regularized Least Squares** optimization problem constrained by directional gradient stability ($\lambda = 0.01$), it reconstructs razor-sharp textures and clean geometric boundaries.

---

## 📊 Visual Comparison (Output Results)

Below is the visual parity and quality improvement achieved by this implementation compared to standard bicubic upscaling:

<img width="1200" height="1200" alt="f16_compare" src="https://github.com/user-attachments/assets/0a442da9-1ce5-4e37-9d7d-39e25b7100d6" />


---
---

## 📂 Folder Structure

Below is the layout of the project directories and files. [cite_start]All executable pipeline scripts are located inside the `code/` folder.

```text
EEE501_Project/
[cite_start]├── Project_Report.pdf            # Academic project report [cite: 495]
[cite_start]├── read_me.txt                   # Original raw readme file [cite: 495]
│
[cite_start]├── code/                         # Core Python implementation [cite: 496]
[cite_start]│   ├── requirements.txt          # Python library dependencies [cite: 497]
[cite_start]│   ├── benchmark.py              # Multi-image paper benchmark engine [cite: 498]
[cite_start]│   ├── prepare_paper_dataset.py  # Dataset preparation utility [cite: 499]
[cite_start]│   ├── stage1_pipeline.py        # HR -> LR -> Bicubic baseline triplet [cite: 499]
[cite_start]│   ├── stage2_grid_mask.py       # Lattice grid mask split [cite: 500]
[cite_start]│   ├── stage3_neighbors.py       # Cross/Plus mode neighbor extraction [cite: 501]
[cite_start]│   ├── stage4_patch_extraction.py# 5-D local patch vectorization [cite: 502]
[cite_start]│   ├── stage5_patch_matching.py  # Global brute-force similarity search [cite: 502]
[cite_start]│   ├── stage6_linear_system.py   # Phi and b matrix construction [cite: 503]
[cite_start]│   ├── stage7_weight_solve.py    # Regularized Least Squares optimizer [cite: 503]
[cite_start]│   ├── stage8_pixel_reconstruction.py # Single-pixel test demo [cite: 504]
[cite_start]│   ├── stage9_full_reconstruction.py  # Sequential full image processing [cite: 504]
[cite_start]│   └── stage10_parallel_benchmark.py # Multi-core accelerated framework [cite: 505]
│
[cite_start]├── test_images/                  # Source imagery root [cite: 506]
[cite_start]│   └── hr/                       # 256x256 and 512x512 ground-truth inputs [cite: 507, 510]
[cite_start]│       ├── Bike.png              # Case-sensitive testing files [cite: 508, 535]
│       ├── Lighthouse.png        
│       ├── einstein.png
│       ├── butterfly.png
│       ├── leaves.png
│       ├── f16.png
│       ├── goldhill.png
│       └── lena.png
│
[cite_start]└── results/                      # Output directory for generated PNGs and CSV summaries [cite: 512, 513]
## 🛠️ 10-Stage Modular Implementation Pipeline

The project is structured into 10 decoupled stages to maintain algorithmic transparency:
* **Stage 1:** Image Pipeline (Load $\rightarrow$ Grayscale $\rightarrow$ Normalize $\rightarrow$ 2x LR Decimation $\rightarrow$ Bicubic Coarse HR)
* **Stage 2:** Grid Classification (Even-Even, Odd-Odd, Odd-Even lattice split)
* **Stage 3:** Neighbor Extraction
* **Stage 4:** Patch Vector Representation
* **Stage 5:** Non-Local Patch Search (Brute-force joint similarity scoring)
* **Stage 6:** Linear System Build ($\Phi w \approx b$)
* **Stage 7:** Regularized Least Squares Weight Estimation (Tikhonov-like optimization)
* **Stage 8:** Pixel Reconstruction (Weighted matrix multiplication)
* **Stage 9:** Image Synthesis & Evaluation (PSNR, SSIM, MAE metrics calculation)
* **Stage 10:** Parallel Performance Benchmark (Multi-core acceleration)

---

## ⚡ Parallelization & Performance

Processing a $512 \times 512$ image requires estimating over **190,000 missing pixels**. Doing a global search and solving a matrix system for every single pixel is computationally heavy.

To reduce execution times from tens of minutes to seconds, this implementation features a **Parallel Framework using Python's `multiprocessing`**:
* Dynamically chunks image grids across available CPU cores.
* Upscales *Lena $512^2$* in just **~222 seconds** (using 19 workers) while maintaining 100% mathematical parity with sequential execution.

---

## 🚀 Getting Started & Execution

### 1️⃣ Required Software
* **OS:** Windows 10/11 (64-bit preferred)
* **Python:** 3.9 through 3.13 (Tested on Python 3.13.x)

### 2️⃣ Environment Setup
Open your terminal (e.g., PowerShell) and run the following commands:

```powershell
# 1. Navigate to the code directory
cd C:\Users\arda0\OneDrive\Masaustu\EEE501_Project\code

# 2. Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. Upgrade pip tools and install dependencies
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
