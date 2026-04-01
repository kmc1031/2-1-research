# Reproducibility Guide

This document describes the exact steps to reproduce the experiments in this repository.

> **Key caveat:** Python dependencies alone are not sufficient.
> FFmpeg must be built with **libvmaf** support. See [FFmpeg Requirements](#2-ffmpeg-requirements) below.

---

## 1. System Requirements

| Item | Requirement |
|---|---|
| OS | Windows 10/11, Linux (Ubuntu 20.04+) |
| Python | ≥ 3.12 (see `.python-version`) |
| FFmpeg | ≥ 6.0, built with `--enable-libx264 --enable-libvmaf` |
| CUDA *(optional)* | CUDA 12.6+, cuDNN compatible with PyTorch 2.x |
| RAM | ≥ 8 GB (≥ 16 GB recommended for full pipeline) |
| Disk | ≥ 2 GB free (video inputs + encoded outputs) |

---

## 2. FFmpeg Requirements

This project uses FFmpeg for encoding (libx264) and quality measurement (libvmaf, MS-SSIM).

### Verify your FFmpeg has the required codecs

```bash
ffmpeg -codecs 2>/dev/null | grep -E "libx264|libvmaf"
```

Expected output (must include both):
```
DEV.LS h264    H.264 / AVC ... (encoders: ... libx264 ...)
 ...   vmaf   VMAF (Video Multi-Method Assessment Fusion) (decoders: libvmaf ...)
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

> **Without libvmaf:** VMAF and MS-SSIM scores will be `nan`. PSNR and SSIM will still work.

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
uv run python test.py

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
└── summary_bd_rates.csv       # BD-Rate summary table
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
# Takes ~1-2 hours on GPU, ~4-8 hours on CPU
uv run python run_noise_experiment.py \
    -v akiyo foreman mobile stefan \
    --sigma 0 5 10 15 \
    -b 100 200 300 400 500
```

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

Check `summary_bd_rates.csv`:
- `BD-Rate_PSNR_Prop(%)` should be **negative** for noisy conditions
- `BD-Rate_PSNR_Prop(%)` may be positive for clean conditions

### Sanity check

```bash
# Verify transform round-trip error is near zero
uv run python test.py
# Expected: max round-trip error < 1e-5
```

---

## 8. Known Issues & Limitations

| Issue | Impact | Workaround |
|---|---|---|
| FFmpeg without libvmaf | VMAF = `nan` | Use GPL FFmpeg build |
| CUDA OOM on <4 GB VRAM | Falls back to CPU | Reduce `chunk_size` in `main_pipeline.py` |
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
| FFmpeg | GPL build with libvmaf |
