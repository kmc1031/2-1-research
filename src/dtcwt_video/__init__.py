"""
dtcwt_video: 3D DT-CWT 기반 비디오 전처리 라이브러리.

핵심 공개 API:
    DTCWT3DProcessor    — 3D DT-CWT 변환 및 임계값 기반 전처리 (CPU/GPU)
    evaluate_video_quality — PSNR/SSIM/VMAF 등 비디오 품질 측정
    get_video_metadata  — 비디오 메타데이터 추출
"""

from dtcwt_video.dtcwt_processor import DTCWT3DProcessor
from dtcwt_video.evaluate_metrics import evaluate_video_quality
from dtcwt_video.pipeline import get_video_metadata, read_y4m_and_split, create_x264_encoder

__version__ = "0.1.0"
__all__ = [
    "DTCWT3DProcessor",
    "evaluate_video_quality",
    "get_video_metadata",
    "read_y4m_and_split",
    "create_x264_encoder",
]
