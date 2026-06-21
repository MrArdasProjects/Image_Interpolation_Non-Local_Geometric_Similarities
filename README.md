## 🖼️ Image Interpolation Based on Non-Local Geometric Similarities

[![Python Version](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Computer Vision](https://img.shields.io/badge/Computer%20Vision-Image%20Processing-green)]()
[![Parallel Computing](https://img.shields.io/badge/Parallel%20Processing-Multiprocessing-orange)]()

---

## 📌 Overview

Image interpolation is the process of reconstructing a high-resolution image from a low-resolution input while preserving edges, textures, and structural details.

Traditional interpolation techniques such as nearest-neighbor, bilinear, and bicubic interpolation estimate missing pixels using only local neighboring information. While computationally efficient, these methods often introduce blurred edges, jagged structures, and loss of fine texture details when images are enlarged.

This project implements a non-local image interpolation framework in Python. Instead of relying solely on nearby pixels, the algorithm searches for structurally similar patches across the entire image and uses adaptive weight estimation to reconstruct missing pixels more accurately.

The implementation includes a complete reconstruction pipeline, quantitative evaluation metrics, and a multiprocessing-based acceleration framework for efficient full-image reconstruction.

---

<img width="1200" height="1200" alt="f16_compare" src="https://github.com/user-attachments/assets/b7d6740a-dbb9-4de9-830a-c1a203650b06" />


## 🎯 Problem Statement

When an image is downsampled, a large amount of visual information is lost.

The challenge is to estimate the missing pixels during upscaling while preserving:

* Sharp edges
* Fine textures
* Repeating patterns
* Geometric structures

Conventional interpolation methods only examine nearby pixels.

This project explores a non-local approach by searching for similar image structures across the entire image and using them to guide reconstruction.

---

## 🧠 Core Idea

Instead of estimating a missing pixel only from its immediate neighbors, the algorithm searches for similar patches in different regions of the image.

These similar patches are used to construct an adaptive reconstruction model.

The reconstruction weights are estimated using regularized least squares optimization and then used to reconstruct the missing pixel values.

This approach allows the algorithm to preserve image details more effectively than purely local interpolation techniques.

---

## ⚙️ Methodology

The reconstruction pipeline consists of ten modular stages:

1. Image loading and preprocessing
2. HR → LR image generation
3. Bicubic initialization
4. Grid classification
5. Neighbor extraction
6. Patch representation
7. Non-local patch matching
8. Linear system construction
9. Adaptive weight estimation
10. Full-image reconstruction and evaluation

The final reconstructed image is compared against the original high-resolution image using quantitative quality metrics.

---

## 📊 Results

Average performance across benchmark images:

| Metric            |  Bicubic | Proposed Method |
| ----------------- | -------: | --------------: |
| PSNR              | 26.50 dB |        29.39 dB |
| SSIM              |    0.858 |           0.907 |
| Average PSNR Gain |        - |        +2.90 dB |

Example result for the Lena image:

| Metric |  Bicubic | Proposed Method |
| ------ | -------: | --------------: |
| PSNR   | 30.86 dB |        35.00 dB |
| SSIM   |   0.8953 |          0.9347 |
| MAE    |   0.0163 |          0.0093 |

---

## ⚡ Parallel Processing

Full-image reconstruction is computationally expensive because the algorithm performs patch search and adaptive weight estimation for a large number of missing pixels.

For example, a 512×512 image contains more than 190,000 missing pixels after 2× downsampling.

To improve performance, the project includes a multiprocessing-based implementation that:

* Distributes reconstruction tasks across CPU workers
* Reduces total runtime significantly
* Produces identical reconstruction results
* Requires no GPU acceleration

---

## 🛠️ Technologies

* Python
* NumPy
* OpenCV
* SciPy
* scikit-image
* Matplotlib
* Pillow
* Multiprocessing

---

## 📂 Project Structure

```text
EEE501_Project/
│
├── code/
│   ├── benchmark.py
│   ├── prepare_paper_dataset.py
│   ├── requirements.txt
│   ├── stage1_pipeline.py
│   ├── stage2_grid_mask.py
│   ├── stage3_neighbors.py
│   ├── stage4_patch_extraction.py
│   ├── stage5_patch_matching.py
│   ├── stage6_linear_system.py
│   ├── stage7_weight_solve.py
│   ├── stage8_pixel_reconstruction.py
│   ├── stage9_full_reconstruction.py
│   └── stage10_parallel_benchmark.py
│
├── test_images/
│   └── hr/
│
└── results/
```

---

## 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/MrArdasProjects/Image_Interpolation_Non-Local_Geometric_Similarities.git
cd Image_Interpolation_Non-Local_Geometric_Similarities/EEE501_Project/code
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the environment:

### Windows

```powershell
.\.venv\Scripts\Activate.ps1
```

### Linux / macOS

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Quick Start

Run the parallel reconstruction pipeline:

```bash
python stage10_parallel_benchmark.py --image ../test_images/hr/lena.png --N 60 --lam 0.01 --workers auto --no_vis
```

Run the complete benchmark suite:

```bash
python benchmark.py --paper_set --N 60 --lam 0.01 --no_vis
```

---

## 📈 Evaluation Metrics

The reconstruction quality is evaluated using:

* PSNR (Peak Signal-to-Noise Ratio)
* SSIM (Structural Similarity Index)
* MAE (Mean Absolute Error)

These metrics compare the reconstructed image against the original high-resolution image.

---

## 👤 Author

**Mehmet Arda Uçar**

M.Sc. Computer Engineering Student

Areas of Interest:

* Artificial Intelligence
* Computer Vision
* Digital Image Processing
* Machine Learning

---

## 📄 License

This project is licensed under the MIT License.

<img width="2509" height="1338" alt="Ekran görüntüsü 2026-06-21 184055" src="https://github.com/user-attachments/assets/e16b42d8-b16f-44a3-bd59-b9b277098f2b" />
<img width="2512" height="1346" alt="Ekran görüntüsü 2026-06-21 184101" src="https://github.com/user-attachments/assets/ebc4dfea-d146-4fd7-9eb5-0c7332f70a1e" />
<img width="2505" height="1346" alt="Ekran görüntüsü 2026-06-21 184107" src="https://github.com/user-attachments/assets/e1b285b4-6187-4839-bd73-852587026d3d" />
<img width="2502" height="1333" alt="Ekran görüntüsü 2026-06-21 184112" src="https://github.com/user-attachments/assets/6dcf43f0-f732-4338-9de1-f71ba8466bbf" />
<img width="2500" height="1361" alt="Ekran görüntüsü 2026-06-21 184122" src="https://github.com/user-attachments/assets/2233a084-1be8-460c-9bde-1cf6148edb6f" />
<img width="2508" height="1360" alt="Ekran görüntüsü 2026-06-21 184130" src="https://github.com/user-attachments/assets/bb2fc224-72eb-4062-9908-aaad3fee8eb6" />








