"""
dtcwt_video 전역 설정 상수.

이 파일에서 기본값을 중앙 관리합니다.
개별 스크립트에서는 argparse 기본값으로 이 상수를 사용하세요.
"""

# ─── 3D DT-CWT 처리 ───────────────────────────────────────
DEFAULT_CHUNK_SIZE: int = 8          # 청크당 프레임 수
DEFAULT_OVERLAP: int = 4             # 청크 간 오버랩 프레임 수
DEFAULT_THRESHOLD: float = 0.03      # BayesShrink 임계값 multiplier
DEFAULT_NLEVELS: int = 1             # DT-CWT 레벨 수

# ─── 인코딩 ───────────────────────────────────────────────
DEFAULT_BITRATES: list[int] = [100, 200, 300, 400, 500]   # kbps
DEFAULT_PRESET: str = "fast"
DEFAULT_TUNE: str = "zerolatency"

# ─── 실험 ─────────────────────────────────────────────────
DEFAULT_SIGMA_LIST: list[int] = [0, 5, 10, 15]   # Gaussian 노이즈 σ
DEFAULT_VIDEO_NAMES: list[str] = ["akiyo", "foreman"]
DEFAULT_NUM_EVAL_FRAMES: int = 60    # 품질 평가 시 사용할 최대 프레임 수
DEFAULT_NOISE_SEED: int = 42

# ─── 경로 ─────────────────────────────────────────────────
DEFAULT_VIDEO_DIR: str = "./videos"
DEFAULT_OUTPUT_DIR: str = "./outputs"
DEFAULT_NOISE_OUTPUT_DIR: str = "./outputs/noise_experiment"
