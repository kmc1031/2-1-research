# 3D DT-CWT 기반 저비트레이트 비디오 전처리 연구

3D Dual-Tree Complex Wavelet Transform (DT-CWT)을 활용하여, x264 인코딩 전에 비디오의 고주파 노이즈를 제거함으로써, 특히 노이즈가 포함된(Noisy) 영상 조건과 저비트레이트 환경에서의 시각적 품질(PSNR, VMAF 등)을 개선하고 그 원인을 규명하는 연구 프로젝트입니다.

## 프로젝트 구조

```
research/
├── dtcwt_cuda.py             # PyTorch 기반 CUDA GPU 가속 3D DT-CWT 구현
├── dtcwt_processor.py       # 3D DT-CWT 변환 핵심 모듈 (적응형 임계값, CUDA 자동 폴백, Chroma 처리 포함)
├── data_loader.py            # Y4M 비디오 청크 로더 (프로토타입, Y 채널만)
├── main_pipeline.py          # 읽기 → 전처리 → x264 인코딩 통합 파이프라인
├── run_rd_curve.py           # 다중 비디오/비트레이트 RD Curve 실험 자동화 (병렬 처리, Spatial 비교군 지원)
├── run_noise_experiment.py   # ⭐ Clean vs Noisy 조건 분리 실험 (핵심 가설 검증)
├── evaluate_metrics.py       # FFmpeg/OpenCV 기반 PSNR/SSIM/VMAF/MS-SSIM/EPSNR/PSNR-B/GBIM/MEPR 통합 측정
├── advanced_evaluation.py    # EPSNR, PSNR-B, GBIM, MEPR 고급 지표 평가 및 시각화
├── edge_analysis.py          # Sobel 에지맵 기반 EPSNR 분석 및 시각화
├── compare_frames.py         # 프레임 단위 시각적 비교 이미지 생성
├── visualize_residuals.py    # 원본과 전처리본의 잔차 이미지(픽셀 차이) 증폭 및 시각화 도구
├── test.py                   # DT-CWT 변환-역변환 기본 동작 테스트
├── videos/                   # 입력 비디오 파일 (.y4m)
└── outputs/                  # 인코딩 결과물 및 분석 그래프 출력 디렉토리
```

## 실행 방법

### 환경 설정

```bash
uv sync
```

### 기본 동작 테스트

```bash
uv run python test.py
```

### 단일 비디오 전처리 및 인코딩

```bash
uv run python main_pipeline.py
```

### RD Curve 실험 (전체 자동화)

`videos/` 디렉토리에 테스트 비디오 (.y4m)를 배치한 후:

```bash
uv run python run_rd_curve.py
```

### ⭐ Clean vs Noisy 조건 분리 실험 (핵심 가설 검증)

Clean/Noisy(σ=5,10,15) 영상에 대해 Baseline vs Proposed의 RD 성능을 체계적으로 비교합니다.
참조 영상은 항상 clean 원본이므로, 전체 파이프라인의 실질적 품질이 측정됩니다.

```bash
# 기본 실행 (akiyo, foreman / σ=0,5,10,15 / 100~500kbps)
uv run python run_noise_experiment.py

# 빠른 테스트 (비디오 1개, 노이즈 2개, 비트레이트 3개)
uv run python run_noise_experiment.py -v akiyo --sigma 0 10 -b 100 300 500

# 전체 비디오 실험
uv run python run_noise_experiment.py -v akiyo foreman mobile stefan
```

출력물: 조건별 RD Curve, BD-Rate 히트맵, ΔPSNR 트렌드, 오버레이 비교 차트

### 고급 평가 및 시각화 (에지 분석, 잔차 시각화, 프레임 비교)

```bash
uv run python advanced_evaluation.py
uv run python edge_analysis.py
uv run python compare_frames.py
# 잔차(Residual) 이미지 증폭 시각화 (-o 원본영상, -p 처리영상, -f 프레임, -c 증폭배수)
uv run python visualize_residuals.py -o videos/akiyo.y4m -p outputs/akiyo_prop_100k.mp4 -f 10 -c 10
```

## 의존성

- Python ≥ 3.12
- `dtcwt` — 3D/2D DT-CWT 변환 라이브러리 (CPU 기준)
- `torch` — PyTorch (CUDA GPU 가속 3D DT-CWT 사용 시 필요)
- `ffmpeg-python` — FFmpeg 파이프라인 제어
- `numpy`, `opencv-python`, `matplotlib` — 수치 연산 및 시각화
- `yuvio`, `y4m` — YUV 비디오 I/O
- FFmpeg (시스템에 설치 필요)
