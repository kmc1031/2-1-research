"""
고급 비디오 품질 평가 모듈: MS-SSIM, EPSNR, PSNR-B, GBIM, Temporal STRRED.

MS-SSIM은 FFmpeg libvmaf를 통해, 나머지 지표는 OpenCV로 프레임 단위 계산합니다.
결과를 Baseline vs Proposed 막대 그래프로 시각화합니다.
"""

import json
import os
import subprocess

import cv2
import numpy as np
import matplotlib.pyplot as plt

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
        MS-SSIM 평균값 (float). 실패 시 0.0.
    """
    temp_json = f"temp_metrics_{os.path.basename(dist_video)}.json"
    cmd = [
        "ffmpeg", "-y", "-i", dist_video, "-i", ref_video,
        "-lavfi", f"libvmaf=feature=name=float_ms_ssim:log_path={temp_json}:log_fmt=json",
        "-f", "null", "-"
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    ms_ssim_val = 0.0
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


def compute_custom_metrics(ref_cap, dist_cap, num_frames=30):
    """프레임 단위로 EPSNR, PSNR-B, GBIM, Temporal STRRED를 계산합니다.

    Args:
        ref_cap: 원본 비디오의 cv2.VideoCapture 객체.
        dist_cap: 왜곡 비디오의 cv2.VideoCapture 객체.
        num_frames: 분석할 프레임 수.

    Returns:
        (epsnr_mean, psnrb_mean, gbim_mean, strred_mean) 튜플.
    """
    epsnr_list, psnrb_list, gbim_list, strred_proxy_list = [], [], [], []
    prev_ref_gray = None
    prev_dist_gray = None

    for _ in range(num_frames):
        ret1, f_ref = ref_cap.read()
        ret2, f_dist = dist_cap.read()
        if not ret1 or not ret2:
            break

        ref_gray = cv2.cvtColor(f_ref, cv2.COLOR_BGR2GRAY).astype(np.float64)
        dist_gray = cv2.cvtColor(f_dist, cv2.COLOR_BGR2GRAY).astype(np.float64)

        # --- 1. EPSNR (Edge-PSNR) ---
        sobelx = cv2.Sobel(ref_gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(ref_gray, cv2.CV_64F, 0, 1, ksize=3)
        mag = np.sqrt(sobelx**2 + sobely**2)
        edge_mask = mag > np.percentile(mag, EDGE_PERCENTILE)

        mse_edge = np.mean((ref_gray[edge_mask] - dist_gray[edge_mask])**2)
        epsnr = 10 * np.log10((MAX_PIXEL_VALUE**2) / (mse_edge + 1e-10))
        epsnr_list.append(epsnr)

        # --- 2. GBIM & PSNR-B ---
        h, w = ref_gray.shape
        col_edges = np.arange(BLOCK_SIZE - 1, w - 1, BLOCK_SIZE)
        row_edges = np.arange(BLOCK_SIZE - 1, h - 1, BLOCK_SIZE)

        # GBIM: 블록 경계에서의 픽셀 단절 측정
        diff_h = np.abs(dist_gray[:, col_edges] - dist_gray[:, col_edges + 1])
        diff_v = np.abs(dist_gray[row_edges, :] - dist_gray[row_edges + 1, :])
        gbim_val = (np.mean(diff_h) + np.mean(diff_v)) / 2.0
        gbim_list.append(gbim_val)

        # PSNR-B: 경계 블로킹 에러를 포함한 보정 MSE
        mse_total = np.mean((ref_gray - dist_gray)**2)
        mse_boundary = (
            np.mean((ref_gray[:, col_edges] - dist_gray[:, col_edges])**2)
            + np.mean((ref_gray[row_edges, :] - dist_gray[row_edges, :])**2)
        ) / 2.0
        mse_b = mse_total + max(0, mse_boundary - mse_total)
        psnrb = 10 * np.log10((MAX_PIXEL_VALUE**2) / (mse_b + 1e-10))
        psnrb_list.append(psnrb)

        # --- 3. STRRED (Spatio-Temporal Proxy) ---
        if prev_ref_gray is not None:
            mot_ref = np.mean(np.abs(ref_gray - prev_ref_gray))
            mot_dist = np.mean(np.abs(dist_gray - prev_dist_gray))
            # 모션 에너지 보존율 (1에 가까울수록 원본 모션 완벽 보존)
            strred_proxy = min(mot_dist, mot_ref) / (max(mot_dist, mot_ref) + 1e-10)
            strred_proxy_list.append(strred_proxy)

        prev_ref_gray = ref_gray
        prev_dist_gray = dist_gray

    return (
        np.mean(epsnr_list),
        np.mean(psnrb_list),
        np.mean(gbim_list),
        np.mean(strred_proxy_list) if strred_proxy_list else 0
    )


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
    epsnr_b, psnrb_b, gbim_b, strred_b = compute_custom_metrics(
        cap_ref1, cap_base, num_frames=60
    )
    cap_ref1.release()
    cap_base.release()

    cap_ref2 = cv2.VideoCapture(ref_vid)
    cap_prop = cv2.VideoCapture(prop_vid)
    epsnr_p, psnrb_p, gbim_p, strred_p = compute_custom_metrics(
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
    print(f"  5. STRRED* (1에 가까울수록) : Base {strred_b:.4f} vs Prop {strred_p:.4f}")
    print("-" * 50)

    # --- 시각화 ---
    _plot_advanced_chart(
        video_name, bitrate,
        ms_ssim_base, ms_ssim_prop,
        epsnr_b, epsnr_p,
        psnrb_b, psnrb_p,
        gbim_b, gbim_p,
        strred_b, strred_p,
        base_dir,
    )


def _plot_advanced_chart(video_name, bitrate,
                         ms_ssim_base, ms_ssim_prop,
                         epsnr_b, epsnr_p,
                         psnrb_b, psnrb_p,
                         gbim_b, gbim_p,
                         strred_b, strred_p,
                         base_dir):
    """고급 지표 비교 막대 그래프를 생성합니다."""
    metrics_higher = ['MS-SSIM', 'EPSNR (dB)', 'PSNR-B (dB)', 'Temporal STRRED']
    base_high = [ms_ssim_base, epsnr_b, psnrb_b, strred_b]
    prop_high = [ms_ssim_prop, epsnr_p, psnrb_p, strred_p]

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [3, 1]}
    )

    # 높을수록 좋은 지표: Baseline 대비 퍼센트 비율로 정규화
    x = np.arange(len(metrics_higher))
    width = 0.35

    base_normalized = [100] * len(metrics_higher)
    prop_normalized = [
        (p / b) * 100 if b != 0 else 0
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