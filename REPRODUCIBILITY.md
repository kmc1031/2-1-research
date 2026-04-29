# Reproducibility Guide

This document describes the exact steps to reproduce the experiments in this repository.

> **Key caveat:** Python dependencies alone are not sufficient.
> FFmpeg must include **libx264** for encoding. **libvmaf** is recommended for MS-SSIM extraction and supplementary VMAF logging.

---

## 1. System Requirements

| Item | Requirement |
|---|---|
| OS | Windows 10/11, Linux (Ubuntu 20.04+) |
| Python | ≥ 3.12 (see `.python-version`) |
| FFmpeg | ≥ 6.0, built with `--enable-libx264`; `--enable-libvmaf` recommended |
| CUDA *(optional)* | CUDA 12.6+, cuDNN compatible with PyTorch 2.x |
| RAM | ≥ 8 GB (≥ 16 GB recommended for full pipeline) |
| Disk | ≥ 2 GB free (video inputs + encoded outputs) |

---

## 2. FFmpeg Requirements

This project uses FFmpeg for encoding (libx264). MS-SSIM and supplementary VMAF logging use FFmpeg's libvmaf filter when available.

### Verify your FFmpeg has the required codecs

```bash
ffmpeg -codecs 2>/dev/null | grep -E "libx264"
ffmpeg -filters 2>/dev/null | grep -E "libvmaf"
```

Expected output:
```
DEV.LS h264    H.264 / AVC ... (encoders: ... libx264 ...)
 ... libvmaf         VV->V      Calculate the VMAF between two video streams.
```

### If libvmaf is missing

On Windows, the recommended option is to use a full-featured FFmpeg build:

```
https://github.com/BtbN/FFmpeg-Builds/releases
→ ffmpeg-master-latest-win64-gpl.zip  (includes libvmaf)
```

On Linux (Ubuntu):
```bash
sudo apt install ffmpeg libvmaf-dev
# or build from source with --enable-libvmaf
```

> **Without libvmaf:** MS-SSIM and supplementary VMAF scores will be `nan`. PSNR-Y, SSIM, EPSNR, PSNR-B, GBIM, and MEPR still work.

---

## 3. Python Environment Setup

This project uses [`uv`](https://github.com/astral-sh/uv) for dependency management.

```bash
# Install uv (if not already installed)
pip install uv

# Create virtual environment and install all dependencies
uv sync
```

This will install:
- `dtcwt` — CPU-based 3D/2D DT-CWT library
- `torch` (CUDA 12.6 build) — GPU-accelerated DT-CWT
- `ffmpeg-python`, `numpy`, `opencv-python`, `matplotlib`
- `yuvio`, `y4m` — YUV video I/O

### GPU (optional)

CUDA acceleration is used automatically if a compatible GPU is detected.
To verify:
```bash
uv run python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

---

## 4. Input Video Setup

Place `.y4m` format test sequences in the `videos/` directory.

**Recommended sequences** (publicly available from [Xiph.org Media](https://media.xiph.org/video/derf/)):

| Filename | Resolution | Frames | Content Type |
|---|---|---|---|
| `akiyo.y4m` | 352×288 (CIF) | 300 | Low motion (news anchor) |
| `foreman.y4m` | 352×288 (CIF) | 300 | Medium motion |
| `mobile.y4m` | 352×288 (CIF) | 300 | High motion |
| `stefan.y4m` | 352×288 (CIF) | 90 | Very high motion |

Direct download:
```bash
# Example (Linux/macOS)
wget https://media.xiph.org/video/derf/y4m/akiyo_cif.y4m -O videos/akiyo.y4m
wget https://media.xiph.org/video/derf/y4m/foreman_cif.y4m -O videos/foreman.y4m
```

---

## 5. Minimal Reproduction Example

This is the fastest way to verify the pipeline is working correctly.

```bash
# Step 1: Verify DT-CWT transform/inverse consistency
uv run pytest tests/test_transform.py -q

# Step 2: Run the core hypothesis experiment (clean vs noisy)
#   - 1 video, 2 noise levels (σ=0 clean, σ=10 noisy), 3 bitrates
#   - Takes ~5-10 minutes on CPU, ~2-3 minutes with GPU
uv run python run_noise_experiment.py \
    -v akiyo \
    --sigma 0 10 \
    -b 100 300 500 \
    -o outputs/repro_check
```

### Expected Output Files

After running the above, `outputs/repro_check/` should contain:

```
outputs/repro_check/
├── noisy_inputs/
│   └── akiyo_s10.y4m          # Gaussian-noised input (σ=10)
├── akiyo_s0_base_100k.mp4     # Baseline encoded (clean)
├── akiyo_s0_prop_100k.mp4     # Proposed DT-CWT encoded (clean)
├── akiyo_s10_base_100k.mp4    # Baseline encoded (noisy)
├── akiyo_s10_prop_100k.mp4    # Proposed DT-CWT encoded (noisy)
├── rd_psnr_akiyo_s0.png       # RD curve: clean condition
├── rd_psnr_akiyo_s10.png      # RD curve: noisy condition
├── rd_overlay_psnr_akiyo.png  # ★ Key result: both conditions overlaid
├── delta_psnr_akiyo.png       # ★ ΔPSNR trend (positive = Proposed wins)
├── raw_data_akiyo_s0.csv      # Per-bitrate metrics: clean
├── raw_data_akiyo_s10.csv     # Per-bitrate metrics: noisy
├── summary_bd_rates.csv       # BD-Rate / delta / win-rate summary table
└── summary_reliable_metrics.csv # PSNR/MS-SSIM/PSNR-B/EPSNR 중심 요약
```

### Expected Result Pattern

The key finding this experiment should reproduce:

| Condition | ΔPSNR (Proposed − Baseline) | Interpretation |
|---|---|---|
| Clean (σ=0) | Slightly negative (−0.3 ~ −0.6 dB) | Proposed slightly worse — expected |
| Noisy (σ=10) | **Clearly positive (+1.5 ~ +2.5 dB)** | Proposed significantly better ✅ |

If noisy conditions show positive ΔPSNR and clean conditions show near-zero or slightly negative ΔPSNR, the core hypothesis is confirmed.

---

## 6. Full Experiment

```bash
# All videos × all noise levels × 5 bitrates
# RTX 3090/server run: reuse bitrate-independent preprocessing and clean temp videos
uv run python run_noise_experiment.py \
    -v akiyo foreman mobile stefan \
    --sigma 0 5 10 15 \
    -b 100 200 300 400 500 \
    -o outputs/paper_full \
    --chunk_size 16 \
    --reuse_preprocessed \
    --cleanup_intermediates
```

`--reuse_preprocessed` avoids recomputing hqdn3d/DWT/Gaussian/Proposed preprocessing for every bitrate.
For the default `adaptive` mode, this reduces Proposed DT-CWT passes from one per bitrate to one per video/noise condition.
If you switch to `--threshold_mode rate_aware`, Proposed is bitrate-dependent and is not reused, but the other prefilters still are.
`--cleanup_intermediates` keeps CSV/PNG outputs and deletes regenerable `.mp4/.y4m` artifacts after each condition.

For the standard RD curve comparison (without noise injection):
```bash
uv run python run_rd_curve.py
```

---

## 7. Result Verification

### Qualitative check

Open `rd_overlay_psnr_akiyo.png` and verify:
- For **clean** sequences: Proposed (solid line) should be slightly **below** Baseline (dashed)
- For **noisy** sequences: Proposed (solid line) should be clearly **above** Baseline (dashed)

### Quantitative check

Check `summary_bd_rates.csv` and `summary_reliable_metrics.csv`:
- `post_delta_psnr`, `post_delta_msssim`, `post_delta_psnrb`, and `post_delta_epsnr` should be positive for noisy conditions
- clean conditions may be near-zero or negative
- `codec_gain_* = post_delta_* - pre_delta_*` separates codec interaction from pure denoising benefit

### Sanity check

```bash
# Verify transform round-trip error is near zero
uv run pytest tests/test_transform.py -q
# Expected: max round-trip error < 1e-5
```

---

## 8. Known Issues & Limitations

| Issue | Impact | Workaround |
|---|---|---|
| FFmpeg without libvmaf | MS-SSIM/supplementary VMAF = `nan` | Use GPL FFmpeg build, or rely on PSNR/SSIM/EPSNR/PSNR-B |
| CUDA OOM on <4 GB VRAM | Falls back to CPU | Reduce `chunk_size` in the pipeline/experiment CLI |
| BD-Rate requires ≥4 bitrate points | Returns `nan` with fewer points | Use ≥4 bitrates with `-b` flag |
| akiyo has low motion → small gains | Low-motion scenes are already efficient | Use `foreman` or `stefan` for stronger results |

---

## 9. Hardware Used in Development

| Component | Spec |
|---|---|
| CPU | — |
| GPU | NVIDIA (CUDA 12.6) |
| RAM | — |
| OS | Windows 11 |
| Python | 3.12 |
| FFmpeg | GPL build with libx264/libvmaf |
