# LRDRN-KLA-Super-Resolution

Deep learning based image super-resolution using LRDRN for the KLA dataset.

## Model
- Model: LRDRN
- Input: 128 × 128 grayscale
- Output: 256 × 256 grayscale
- Parameters: 751,873

## Results

| Metric | Previous LRDRN | Improved LRDRN |
|---|---:|---:|
| PSNR | 26.4881 dB | **26.7967 dB** |
| SSIM | 0.6750 | **0.6811** |

## Dataset
- Training: 2560 images
- Validation: 640 images
- Test: 400 images

## Output
The improved LRDRN restored 400 test images at 256 × 256 resolution.
