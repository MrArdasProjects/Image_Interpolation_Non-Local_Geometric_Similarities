# 🖼️ Image Interpolation Based on Non-Local Geometric Similarities

[![Python Version](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Course](https://img.shields.io/badge/MSc%20Project-EEE%20501-red)](https://www.ieu.edu.tr/)

[cite_start]This repository contains a high-performance, from-scratch Python framework for advanced image super-resolution based on Non-Local Geometric Similarities[cite: 1, 4, 5]. [cite_start]Developed as an **MSc Term Project** for the *Applied Digital Image Processing* course [cite: 5, 186][cite_start], this architecture rejects strictly localized pixel boundaries, instead learning adaptive upscaling weights by mapping recurring structures across the entire image context[cite: 25, 27].

---

## 🎓 Academic Context

| Course | Institution | Instructor | Developer |
| :--- | :--- | :--- | :--- |
| **EEE 501** — Applied Digital Image Processing | Izmir University of Economics | Doç. Dr. Mehmet TÜRKAN | Mehmet Arda UÇAR |

---

## 📌 Problem Definition & Motivation

[cite_start]Traditional upscaling techniques (like bilinear or bicubic interpolation) rely only on a tiny, immediate neighborhood (e.g., nearest 4 or 16 pixels)[cite: 26, 216]. [cite_start]When dealing with high-frequency details, sharp transitions, or complex textures, this local focus inherently causes[cite: 26, 231]:
* [cite_start]**Blurred edges** along sharp intensity transitions[cite: 26, 231].
* [cite_start]**Jagged artifacts (jaggies)** in diagonal structures[cite: 26, 231].
* [cite_start]**Loss of fine texture details** in repeating patterns[cite: 26, 231].

### 🔍 The Non-Local Philosophy
[cite_start]Natural images are packed with repeating patterns that reappear in entirely different, non-local regions[cite: 27, 239]. [cite_start]This algorithm maps the global image structure to find the top $N=60$ most geometrically similar patches[cite: 83, 85]. [cite_start]By solving a **Regularized Least Squares** optimization problem constrained by directional gradient stability ($\lambda = 0.01$) [cite: 45, 85, 86][cite_start], it reconstructs razor-sharp textures and clean geometric boundaries[cite: 84].

---

## 📊 Visual Comparison (Output Results)

Below is the visual parity and quality improvement achieved by this implementation compared to standard bicubic upscaling:

<img width="1200" height="1200" alt="f16_compare" src="https://github.com/user-attachments/assets/3ff8866b-8f3c-4c71-a35c-030caf2f8d7f" />

---

## 📂 Folder Structure

[cite_start]Below is the layout of the project directories and files[cite: 495]. [cite_start]All executable pipeline scripts are located inside the `code/` folder[cite: 496].

```text
EEE501_Project/
├── Project_Report.pdf            # Academic project report [cite: 495]
├── read_me.txt                   # Original raw readme file [cite: 495]
│
├── code/                         # Core Python implementation [cite: 496]
│   ├── requirements.txt          # Python library dependencies [cite: 497]
│   ├── benchmark.py              # Multi-image paper benchmark engine [cite: 498]
│   ├── prepare_paper_dataset.py  # Dataset preparation utility [cite: 499]
│   ├── stage1_pipeline.py        # HR -> LR -> Bicubic baseline triplet [cite: 499]
│   ├── stage2_grid_mask.py       # Lattice grid mask split [cite: 500]
│   ├── stage3_neighbors.py       # Cross/Plus mode neighbor extraction [cite: 501]
│   ├── stage4_patch_extraction.py# 5-D local patch vectorization [cite: 502]
│   ├── stage5_patch_matching.py  # Global brute-force similarity search [cite: 502]
│   ├── stage6_linear_system.py   # Phi and b matrix construction [cite: 503]
│   ├── stage7_weight_solve.py    # Regularized Least Squares optimizer [cite: 503]
│   ├── stage8_pixel_reconstruction.py # Single-pixel test demo [cite: 504]
│   ├── stage9_full_reconstruction.py  # Sequential full image processing [cite: 504]
│   └── stage10_parallel_benchmark.py # Multi-core accelerated framework [cite: 505]
│
├── test_images/                  # Source imagery root [cite: 506]
│   └── hr/                       # 256x256 and 512x512 ground-truth inputs [cite: 506]
│       ├── Bike.png              # Case-sensitive testing files [cite: 508]
│       ├── Lighthouse.png        # [cite: 509]
│       ├── einstein.png          # [cite: 507]
│       ├── butterfly.png         # [cite: 507]
│       ├── leaves.png            # [cite: 507]
│       ├── f16.png               # [cite: 510]
│       ├── goldhill.png          # [cite: 510]
│       └── lena.png              # [cite: 509]
│
└── results/                      # Output directory for generated PNGs and CSV summaries [cite: 512]
