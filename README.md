# 3D DT-CWT 기반 비디오 전처리 연구

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/)

**연구 주제**: Noisy 영상의 저비트레이트 x264 인코딩 환경에서 3D DT-CWT 전처리의 조건부 이점 분석

> ⚠️ **연구 포지션**: 이 연구는 3D DT-CWT 전처리가 *항상* 화질을 개선한다고 주장하지 않습니다.
> 실험 결과, **Gaussian 노이즈(σ≥5)가 포함된 영상 + 저비트레이트(≤300 kbps)** 조건에서 유의미한 개선이 관찰되며,
> clean 영상에서는 오히려 미세한 성능 저하가 나타납니다. 이를 투명하게 분석하는 것이 이 연구의 목적입니다.

---

## 연구 개요

3D Dual-Tree Complex Wavelet Transform (DT-CWT)의 핵심 특성인 **시점 불변성(shift-invariance)**과 **6방향 선택성(directional selectivity)**을 활용하여, x264 인코딩 전 비디오의 고주파 노이즈를 시간축까지 고려한 3D 방식으로 제거합니다.

### 핵심 가설

| 가설 | 예측 | 검증 방법 |
|---|---|---|
| H1 | Noisy 영상에서 Proposed의 ΔPSNR > 0 | σ≥5 조건에서 baseline 대비 PSNR 향상 |
| H2 | Clean 영상에서 Proposed의 ΔPSNR ≈ 0 또는 < 0 | 전처리 오버헤드로 인한 미세 손실 |
| H3 | 개선 폭은 bitrate ↓, σ ↑ 일수록 커짐 | 조건별 BD-Rate 비교 |

### 스모크 테스트 결과 (akiyo CIF, 2-bitrate 예비 실험)

| 조건 | PSNR Baseline | PSNR Proposed | ΔPSNR | 해석 |
|---|---|---|---|---|
| Clean (σ=0), 200k | 42.69 dB | 42.38 dB | **−0.31 dB** | Proposed 약간 열세 |
| Clean (σ=0), 400k | 45.44 dB | 44.85 dB | **−0.60 dB** | Proposed 약간 열세 |
| Noisy (σ=10), 200k | 35.37 dB | 36.94 dB | **+1.57 dB** ✅ | Proposed 우수 |
| Noisy (σ=10), 400k | 34.85 dB | 37.27 dB | **+2.42 dB** ✅ | Proposed 우수 |

> Clean 조건에서는 Baseline이 우세, Noisy 조건에서는 Proposed가 명확히 우세.
> 이 조건부 특성이 본 연구의 핵심 발견입니다.

---

## 프로젝트 구조

```
research/
├── src/dtcwt_video/
│   ├── dtcwt_cuda.py          # PyTorch 기반 CUDA GPU 가속 3D DT-CWT 구현
│   ├── dtcwt_processor.py     # 3D DT-CWT 핵심 모듈 (임계값, 캐싱, CPU/GPU 자동 폴백)
│   ├── evaluate_metrics.py    # PSNR/SSIM/VMAF/MS-SSIM/EPSNR/PSNR-B/GBIM/MEPR 측정
│   ├── advanced_evaluation.py # 고급 지표 평가 및 시각화
│   ├── edge_analysis.py       # Sobel 에지맵 기반 EPSNR 분석
│   ├── compare_frames.py      # 프레임 단위 시각적 비교
│   ├── config.py, encoders.py, data_loader.py 등
│   └── pipeline.py            # 읽기 → 전처리 → x264 인코딩 파이프라인
├── scripts/
│   ├── run_rd_curve.py        # 다중 비디오/비트레이트 RD Curve 실험 자동화
│   ├── run_noise_experiment.py# ⭐ Clean vs Noisy 조건 분리 실험 (핵심 가설 검증)
│   └── visualize_residuals.py # 잔차(Residual) 시각화 도구
├── tests/                     # 캐싱/변환 회귀 테스트
├── videos/                    # 입력 비디오 (.y4m)
├── outputs/                   # 실험 결과물
├── LICENSE
└── REPRODUCIBILITY.md         # 재현 가이드 (⭐ 필독)
```

---

## 빠른 시작

### 환경 설정

> **선행 조건**: FFmpeg with libvmaf이 필요합니다. 자세한 내용은 [REPRODUCIBILITY.md](REPRODUCIBILITY.md)를 참조하세요.

```bash
# 의존성 설치
uv sync

# GPU 가속 확인 (선택사항)
uv run python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

### 최소 실행 예제 (권장 첫 번째 실행)

```bash
# DT-CWT 동작 검증
uv run python test.py

# 핵심 실험: clean vs noisy 비교 (약 5-10분)
uv run python run_noise_experiment.py -v akiyo --sigma 0 10 -b 100 300 500
```

기대 결과: `outputs/noise_experiment/` 아래 RD Curve, ΔPSNR 차트, CSV 파일 생성.
자세한 기대 출력은 [REPRODUCIBILITY.md §5](REPRODUCIBILITY.md#5-minimal-reproduction-example) 참조.

---

## 실험 실행

### 🔬 Clean vs Noisy 조건 분리 실험 (핵심)

```bash
# 기본 실행 (akiyo, foreman / σ=0,5,10,15 / 100~500kbps)
uv run python run_noise_experiment.py

# 전체 비디오
uv run python run_noise_experiment.py -v akiyo foreman mobile stefan

# 빠른 검증용
uv run python run_noise_experiment.py -v akiyo --sigma 0 10 -b 100 300 500
```

출력물:
- `rd_psnr_*_s{σ}.png` — 조건별 RD Curve
- `rd_overlay_psnr_*.png` — 전체 조건 오버레이 비교
- `delta_psnr_*.png` — ΔPSNR 트렌드 (양수 = Proposed 우수)
- `bd_rate_heatmap.png` — BD-Rate 히트맵
- `summary_bd_rates.csv` — 요약 테이블

### 📈 RD Curve 실험 (전체 자동화)

```bash
uv run python run_rd_curve.py
```

### 🔍 시각화 도구

```bash
# 에지맵 기반 품질 분석
uv run python edge_analysis.py

# 잔차 증폭 시각화 (전처리 효과 육안 확인)
uv run python visualize_residuals.py \
    -o videos/akiyo.y4m \
    -p outputs/noise_experiment/akiyo_s10_prop_200k.mp4 \
    -f 10 -c 10
```

---

## 비교군 구성

| 방법 | 설명 | 사용 스크립트 |
|---|---|---|
| **Baseline** | x264 직접 인코딩 (전처리 없음) | `run_baseline_encoding()` |
| **Spatial (Gaussian)** | 5×5 Gaussian Blur + x264 | `run_spatial_encoding()` |
| **3D DWT** | PyWavelets Haar 3D DWT + x264 | `run_dwt3d_encoding()` |
| **Proposed** | 3D DT-CWT + BayesShrink 임계값 + x264 | `run_proposed_encoding()` |
| *x264 --nr* | x264 내장 노이즈 저감 + x264 | `run_nr_encoding()` *(planned)* |
| *hqdn3d* | FFmpeg hqdn3d 표준 디노이저 + x264 | `run_hqdn3d_encoding()` *(planned)* |

---

## 주요 결과 요약 (업데이트 예정)

> 아래 표는 전체 실험 완료 후 채워집니다. 현재는 예비 실험 결과입니다.

### BD-Rate Summary (Proposed vs Baseline, akiyo CIF)

| 조건 | BD-Rate (PSNR) | BD-Rate (VMAF) | 해석 |
|---|---|---|---|
| Clean (σ=0) | TBD | TBD | Proposed 불리 예상 |
| Noisy (σ=5) | TBD | TBD | 소폭 개선 예상 |
| Noisy (σ=10) | **TBD** | **TBD** | 명확한 개선 예상 |
| Noisy (σ=15) | **TBD** | **TBD** | 최대 개선 예상 |

> BD-Rate < 0: Proposed가 동일 품질에서 더 적은 비트레이트 사용 (= 개선)

---

## 의존성

| 패키지 | 버전 | 역할 |
|---|---|---|
| Python | ≥ 3.12 | — |
| `torch` | ≥ 2.0 (CUDA 12.6) | GPU 가속 DT-CWT |
| `dtcwt` | ≥ 0.14 | CPU 기준 DT-CWT |
| `ffmpeg-python` | ≥ 0.2 | FFmpeg 파이프라인 제어 |
| `numpy` | ≥ 1.26 | 수치 연산 |
| `opencv-python` | ≥ 4.11 | 고급 지표 계산 |
| `matplotlib` | ≥ 3.10 | 시각화 |
| FFmpeg (system) | ≥ 6.0 | 필수: libx264 + **libvmaf** 포함 빌드 |

---

## 라이선스

MIT License. 자세한 내용은 [LICENSE](LICENSE)를 참조하세요.

---

## 참고 문헌

- Kingsbury, N. G. (1999). "The dual-tree complex wavelet transform: a new technique for shift invariance and directional image filters."
- Selesnick, I. W., et al. (2005). "The dual-tree complex wavelet transform." *IEEE Signal Processing Magazine*.
- Selesnick, I., & Li, K. Y. "Video denoising using 2D and 3D dual-tree complex wavelet transforms."
- Bjontegaard, G. (2001). "Calculation of average PSNR differences between RD-curves." *ITU-T SG16 Doc. VCEG-M33*.
