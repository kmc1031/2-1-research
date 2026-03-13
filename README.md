# 3D DT-CWT 기반 저비트레이트 비디오 전처리 연구

3D Dual-Tree Complex Wavelet Transform (DT-CWT)을 활용하여, x264 인코딩 전에 비디오의 고주파 노이즈를 제거함으로써 저비트레이트 환경에서의 시각적 품질(PSNR, VMAF 등)을 개선하는 연구 프로젝트입니다.

## 프로젝트 구조

```
research/
├── dtcwt_processor.py       # 3D DT-CWT 변환 + Soft Shrinkage 전처리 핵심 모듈
├── data_loader.py            # Y4M 비디오 청크 로더 (프로토타입, Y 채널만)
├── main_pipeline.py          # 읽기 → 전처리 → x264 인코딩 통합 파이프라인
├── run_rd_curve.py           # 다중 비디오/비트레이트 RD Curve 실험 자동화
├── evaluate_metrics.py       # FFmpeg 기반 PSNR/SSIM/VMAF 측정
├── advanced_evaluation.py    # EPSNR, PSNR-B, GBIM, STRRED 고급 지표 평가
├── edge_analysis.py          # Sobel 에지맵 기반 EPSNR 분석 및 시각화
├── compare_frames.py         # 프레임 단위 시각적 비교 이미지 생성
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

### 고급 평가 (에지 분석, 시각적 비교)

```bash
uv run python advanced_evaluation.py
uv run python edge_analysis.py
uv run python compare_frames.py
```

## 의존성

- Python ≥ 3.12
- `dtcwt` — 3D DT-CWT 변환 라이브러리
- `ffmpeg-python` — FFmpeg 파이프라인 제어
- `numpy`, `opencv-python`, `matplotlib` — 수치 연산 및 시각화
- `yuvio`, `y4m` — YUV 비디오 I/O
- FFmpeg (시스템에 설치 필요)
