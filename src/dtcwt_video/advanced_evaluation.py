"""
고급 비디오 품질 평가 모듈: MS-SSIM, EPSNR, PSNR-B, GBIM, Temporal MEPR.

MS-SSIM은 FFmpeg libvmaf를 통해, 나머지 지표는 OpenCV로 프레임 단위 계산합니다.
결과를 Baseline vs Proposed 막대 그래프로 시각화합니다.
"""

import json
import os
import subprocess

import cv2
import numpy as np
import matplotlib.pyplot as plt

from dtcwt_video.evaluate_metrics import compute_custom_metrics

# 상수 정의
BLOCK_SIZE = 8          # 매크로블록 크기 (H.264 기본)
EDGE_PERCENTILE = 80    # 상위 20% 에지를 마스크로 사용
MAX_PIXEL_VALUE = 255.0


def extract_ms_ssim_vmaf(ref_video, dist_video):
    """FFmpeg libvmaf를 통해 MS-SSIM을 측정합니다.

    Args:
        ref_video: 원본 비디오 경로.
        dist_video: 왜곡 비디오 경로.

    Returns:
        MS-SSIM 평균값 (float). 실패 시 float('nan').
    """
    temp_json = f"temp_metrics_{os.path.basename(dist_video)}.json"
    cmd = [
        "ffmpeg", "-y", "-i", dist_video, "-i", ref_video,
        "-lavfi", f"libvmaf=feature=name=float_ms_ssim:log_path={temp_json}:log_fmt=json",
        "-f", "null", "-"
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    ms_ssim_val = float('nan')
    if os.path.exists(temp_json):
        try:
            with open(temp_json, 'r', encoding='utf-8') as f:
                data = json.load(f)
                ms_ssim_val = data["pooled_metrics"]["float_ms_ssim"]["mean"]
        except (json.JSONDecodeError, KeyError):
            pass
        finally:
            os.remove(temp_json)
    return ms_ssim_val



def evaluate_and_plot_advanced(video_name, bitrate):
    """Baseline vs Proposed 고급 지표를 측정하고 막대 그래프로 시각화합니다.

    Args:
        video_name: 비디오 이름 (확장자 제외).
        bitrate: 비트레이트 문자열 (예: '200k').
    """
    base_dir = "./outputs"
    os.makedirs(base_dir, exist_ok=True)

    ref_vid = f"./videos/{video_name}.y4m"
    base_vid = os.path.join(base_dir, f"{video_name}_base_{bitrate}.mp4")
    prop_vid = os.path.join(base_dir, f"{video_name}_prop_{bitrate}.mp4")

    if not os.path.exists(base_vid) or not os.path.exists(prop_vid):
        print(f"🚨 오류: {bitrate} 비트레이트의 인코딩된 결과물이 없습니다.")
        return

    print(f"📊 [{video_name.upper()} - {bitrate}] 고급 지표 평가 중 (약 10~20초 소요)...")

    # 1. MS-SSIM 측정
    ms_ssim_base = extract_ms_ssim_vmaf(ref_vid, base_vid)
    ms_ssim_prop = extract_ms_ssim_vmaf(ref_vid, prop_vid)

    # 2. Pixel & Block 기반 지표 측정
    cap_ref1 = cv2.VideoCapture(ref_vid)
    cap_base = cv2.VideoCapture(base_vid)
    epsnr_b, psnrb_b, gbim_b, mepr_b = compute_custom_metrics(
        cap_ref1, cap_base, num_frames=60
    )
    cap_ref1.release()
    cap_base.release()

    cap_ref2 = cv2.VideoCapture(ref_vid)
    cap_prop = cv2.VideoCapture(prop_vid)
    epsnr_p, psnrb_p, gbim_p, mepr_p = compute_custom_metrics(
        cap_ref2, cap_prop, num_frames=60
    )
    cap_ref2.release()
    cap_prop.release()

    # 결과 출력
    print("-" * 50)
    print(f"  [지표 결과 요약 - {video_name.upper()}]")
    print(f"  1. MS-SSIM (높을수록 좋음) : Base {ms_ssim_base:.4f} vs Prop {ms_ssim_prop:.4f}")
    print(f"  2. EPSNR   (높을수록 좋음) : Base {epsnr_b:.2f} vs Prop {epsnr_p:.2f}")
    print(f"  3. PSNR-B  (높을수록 좋음) : Base {psnrb_b:.2f} vs Prop {psnrb_p:.2f}")
    print(f"  4. GBIM    (낮을수록 좋음) : Base {gbim_b:.2f} vs Prop {gbim_p:.2f}")
    print(f"  5. MEPR    (1에 가까울수록) : Base {mepr_b:.4f} vs Prop {mepr_p:.4f}")
    print("-" * 50)

    # --- 시각화 ---
    _plot_advanced_chart(
        video_name, bitrate,
        ms_ssim_base, ms_ssim_prop,
        epsnr_b, epsnr_p,
        psnrb_b, psnrb_p,
        gbim_b, gbim_p,
        mepr_b, mepr_p,
        base_dir,
    )


def _plot_advanced_chart(video_name, bitrate,
                         ms_ssim_base, ms_ssim_prop,
                         epsnr_b, epsnr_p,
                         psnrb_b, psnrb_p,
                         gbim_b, gbim_p,
                         mepr_b, mepr_p,
                         base_dir):
    """고급 지표 비교 막대 그래프를 생성합니다."""
    metrics_higher = ['MS-SSIM', 'EPSNR (dB)', 'PSNR-B (dB)', 'MEPR']
    base_high = [ms_ssim_base, epsnr_b, psnrb_b, mepr_b]
    prop_high = [ms_ssim_prop, epsnr_p, psnrb_p, mepr_p]

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [3, 1]}
    )

    # 높을수록 좋은 지표: Baseline 대비 퍼센트 비율로 정규화
    x = np.arange(len(metrics_higher))
    width = 0.35

    base_normalized = [100] * len(metrics_higher)
    prop_normalized = [
        (p / b) * 100 if (b is not None and not np.isnan(b) and b != 0) else float('nan')
        for p, b in zip(prop_high, base_high)
    ]

    ax1.bar(x - width / 2, base_normalized, width,
            label='Baseline (x264)', color='#4c72b0')
    ax1.bar(x + width / 2, prop_normalized, width,
            label='Proposed (DT-CWT)', color='#c44e52')

    ax1.set_ylabel('Relative Score (%)\n(Baseline = 100%)', fontsize=12)
    ax1.set_title(
        f'Performance Metrics (Higher is Better) - {video_name.upper()} {bitrate}',
        fontsize=14
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics_higher, fontsize=11)
    ax1.legend()
    ax1.grid(axis='y', linestyle='--', alpha=0.7)

    for i in range(len(x)):
        ax1.text(x[i] - width / 2, 101, f"{base_high[i]:.2f}",
                 ha='center', va='bottom', fontsize=10)
        ax1.text(x[i] + width / 2, prop_normalized[i] + 1, f"{prop_high[i]:.2f}",
                 ha='center', va='bottom', fontsize=10, fontweight='bold')

    # 낮을수록 좋은 지표 (GBIM)
    x_low = np.arange(1)
    ax2.bar(x_low - width / 2, [gbim_b], width, label='Baseline', color='#4c72b0')
    ax2.bar(x_low + width / 2, [gbim_p], width, label='Proposed', color='#c44e52')

    ax2.set_ylabel('Impairment Score', fontsize=12)
    ax2.set_title('Blocking Artifacts\n(Lower is Better)', fontsize=14)
    ax2.set_xticks(x_low)
    ax2.set_xticklabels(['GBIM (Blocking)'], fontsize=11)
    ax2.grid(axis='y', linestyle='--', alpha=0.7)

    ax2.text(x_low[0] - width / 2, gbim_b + 0.1, f"{gbim_b:.2f}",
             ha='center', va='bottom', fontsize=10)
    ax2.text(x_low[0] + width / 2, gbim_p + 0.1, f"{gbim_p:.2f}",
             ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    result_img = os.path.join(base_dir, f"advanced_metrics_{video_name}_{bitrate}.png")
    plt.savefig(result_img, dpi=300)
    plt.close()
    print(f"💾 종합 지표 차트 저장 완료: {result_img}")


if __name__ == "__main__":
    evaluate_and_plot_advanced("stefan", "200k")
