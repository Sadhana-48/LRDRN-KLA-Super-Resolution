# LRDRN-KLA-Super-Resolution

Deep learning based image super-resolution using LRDRN for the KLA dataset.

## Project Overview

This project implements an LRDRN (Low-Resolution Deep Residual Network) model for restoring and enhancing low-resolution noisy images.

The model takes 128 × 128 grayscale low-resolution images as input and produces 256 × 256 restored images.

## Dataset

- Training samples: 2560
- Validation samples: 640
- Test images: 400
- Input resolution: 128 × 128
- Output resolution: 256 × 256
- Image type: Grayscale / Single-channel

The dataset contains NoisyLR and Ground Truth (GT) images in NumPy `.npy` format.

## Model

**Model:** LRDRN

- Parameters: 751,873
- Device: CUDA
- Best epoch: 38
- Best validation loss: 0.079789

## Results

| Metric | Previous LRDRN | Improved LRDRN |
|---|---:|---:|
| PSNR | 26.4881 dB | **26.7967 dB** |
| SSIM | 0.6750 | **0.6811** |

### Improvement

- PSNR improvement: **+0.3086 dB**
- SSIM improvement: **+0.0061**

The improved LRDRN model performs better than the previous model.

## Test Results

- Test images restored: 400
- Output resolution: 256 × 256
- Output range: 0.0 – 1.0
- Invalid values: 0

## Project Structure

```text
LRDRN-KLA-Super-Resolution/
│
├── Untitled1.ipynb
├── README.md
│
└── Results/
    └── Test_Improved/
