"""
비디오 품질 평가 모듈: PSNR, SSIM, VMAF, MS-SSIM, EPSNR, PSNR-B, GBIM, MEPR, STRRED.

FFmpeg의 내장 필터 및 OpenCV를 활용하여 참조(reference) 영상 대비
왜곡(distorted) 영상의 품질 지표를 한 번에 측정합니다.
"""

import json
import os
import re
import subprocess
import tempfile
import cv2
import numpy as np

# 상수 정의 (compute_custom_metrics용)
BLOCK_SIZE = 8
EDGE_PERCENTILE = 80
MAX_PIXEL_VALUE = 255.0
STRRED_EPSILON = 1e-10


def _entropy_from_coefficients(values):
    """Gaussian coefficient entropy proxy used by the lightweight STRRED score."""
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return float("nan")
    var = float(np.var(values))
    return 0.5 * np.log2(2.0 * np.pi * np.e * max(var, STRRED_EPSILON))


def _strred_frame_features(gray):
    """Return spatial and temporal-ready bandpass representation for STRRED."""
    gray = gray.astype(np.float64) / MAX_PIXEL_VALUE
    low = cv2.GaussianBlur(gray, (5, 5), 1.2)
    bandpass = gray - low
    return bandpass


def compute_strred(ref_cap, dist_cap, num_frames=60):
    """Compute a lightweight STRRED-style spatio-temporal quality score.

    This is an implementation-friendly approximation of the reduced-reference
    STRRED idea: compare entropy statistics of spatial bandpass coefficients and
    temporal bandpass frame differences. Lower is better, and 0 means identical
    statistics under this model.
    """
    spatial_terms = []
    temporal_terms = []
    prev_ref_band = None
    prev_dist_band = None

    for _ in range(num_frames):
        ret1, f_ref = ref_cap.read()
        ret2, f_dist = dist_cap.read()
        if not ret1 or not ret2:
            break

        ref_gray = cv2.cvtColor(f_ref, cv2.COLOR_BGR2GRAY)
        dist_gray = cv2.cvtColor(f_dist, cv2.COLOR_BGR2GRAY)
        ref_band = _strred_frame_features(ref_gray)
        dist_band = _strred_frame_features(dist_gray)

        spatial_terms.append(
            abs(_entropy_from_coefficients(ref_band) -
                _entropy_from_coefficients(dist_band))
        )

        if prev_ref_band is not None:
            ref_temporal = ref_band - prev_ref_band
            dist_temporal = dist_band - prev_dist_band
            temporal_terms.append(
                abs(_entropy_from_coefficients(ref_temporal) -
                    _entropy_from_coefficients(dist_temporal))
            )

        prev_ref_band = ref_band
        prev_dist_band = dist_band

    if not spatial_terms:
        return float("nan")
    spatial_score = float(np.mean(spatial_terms))
    temporal_score = float(np.mean(temporal_terms)) if temporal_terms else 0.0
    return spatial_score + temporal_score

def compute_custom_metrics(ref_cap, dist_cap, num_frames=30):
    """프레임 단위로 EPSNR, PSNR-B, GBIM, MEPR을 계산합니다.

    Args:
        ref_cap: 원본 비디오의 cv2.VideoCapture 객체.
        dist_cap: 왜곡 비디오의 cv2.VideoCapture 객체.
        num_frames: 분석할 프레임 수.

    Returns:
        (epsnr_mean, psnrb_mean, gbim_mean, mepr_mean, strred) 튜플.
    """
    epsnr_list, psnrb_list, gbim_list, mepr_list = [], [], [], []
    strred_spatial_terms, strred_temporal_terms = [], []
    prev_ref_gray = None
    prev_dist_gray = None
    prev_ref_band = None
    prev_dist_band = None

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

        # PSNR-B (Yim & Bovik, 2011): 블록 경계의 blocking effect를 분리하여 보정
        mse_total = np.mean((ref_gray - dist_gray)**2)
        
        # 블록 경계에서의 불연속(blocking artifact) 측정
        block_diff_h = np.mean((dist_gray[:, col_edges] - dist_gray[:, col_edges + 1])**2)
        block_diff_v = np.mean((dist_gray[row_edges, :] - dist_gray[row_edges + 1, :])**2)
        blocking_effect = (block_diff_h + block_diff_v) / 2.0
        
        # 원본에서의 자연스러운 경계 불연속 측정 (기저선)
        ref_diff_h = np.mean((ref_gray[:, col_edges] - ref_gray[:, col_edges + 1])**2)
        ref_diff_v = np.mean((ref_gray[row_edges, :] - ref_gray[row_edges + 1, :])**2)
        natural_edge = (ref_diff_h + ref_diff_v) / 2.0
        
        # 인코딩으로 추가된 블로킹 아티팩트만 분리
        added_blocking = max(0, blocking_effect - natural_edge)
        mse_b = mse_total + added_blocking
        psnrb = 10 * np.log10((MAX_PIXEL_VALUE**2) / (mse_b + 1e-10))
        psnrb_list.append(psnrb)

        # --- 3. MEPR (Motion Energy Preservation Ratio) ---
        # 원본과 왜곡 영상 간 프레임 차이의 에너지 보존율 측정
        # (1에 가까울수록 원본 모션 특성 완벽 보존)
        if prev_ref_gray is not None:
            mot_ref = np.mean(np.abs(ref_gray - prev_ref_gray))
            mot_dist = np.mean(np.abs(dist_gray - prev_dist_gray))
            mepr = min(mot_dist, mot_ref) / (max(mot_dist, mot_ref) + 1e-10)
            mepr_list.append(mepr)

        # --- 4. STRRED-style entropy distance (lower is better) ---
        ref_band = _strred_frame_features(ref_gray)
        dist_band = _strred_frame_features(dist_gray)
        strred_spatial_terms.append(
            abs(_entropy_from_coefficients(ref_band) -
                _entropy_from_coefficients(dist_band))
        )
        if prev_ref_band is not None:
            ref_temporal = ref_band - prev_ref_band
            dist_temporal = dist_band - prev_dist_band
            strred_temporal_terms.append(
                abs(_entropy_from_coefficients(ref_temporal) -
                    _entropy_from_coefficients(dist_temporal))
            )

        prev_ref_gray = ref_gray
        prev_dist_gray = dist_gray
        prev_ref_band = ref_band
        prev_dist_band = dist_band

    strred_spatial = np.mean(strred_spatial_terms) if strred_spatial_terms else float('nan')
    strred_temporal = np.mean(strred_temporal_terms) if strred_temporal_terms else 0.0
    strred_val = (
        float(strred_spatial + strred_temporal)
        if not np.isnan(strred_spatial) else float('nan')
    )

    return (
        np.mean(epsnr_list) if epsnr_list else float('nan'),
        np.mean(psnrb_list) if psnrb_list else float('nan'),
        np.mean(gbim_list) if gbim_list else float('nan'),
        np.mean(mepr_list) if mepr_list else float('nan'),
        strred_val,
    )


def _read_vmaf_metric(log_path, metric_name):
    """Read a metric from libvmaf JSON, accepting pooled or per-frame format."""
    with open(log_path, 'r', encoding='utf-8') as f:
        vmaf_data = json.load(f)
    pooled = vmaf_data.get("pooled_metrics", {})
    if metric_name in pooled and "mean" in pooled[metric_name]:
        return float(pooled[metric_name]["mean"])

    frame_values = [
        frame.get("metrics", {}).get(metric_name)
        for frame in vmaf_data.get("frames", [])
    ]
    frame_values = [float(v) for v in frame_values if v is not None]
    return float(np.mean(frame_values)) if frame_values else float('nan')


def evaluate_video_quality(ref_video, dist_video, num_frames_custom=60):
    """참조 영상 대비 왜곡 영상의 모든 품질 지표를 측정합니다.

    Args:
        ref_video: 원본(참조) 비디오 파일 경로.
        dist_video: 인코딩된(왜곡) 비디오 파일 경로.
        num_frames_custom: OpenCV 기반 고급 지표를 계산할 프레임 수.

    Returns:
        (psnr, ssim, vmaf, ms_ssim, epsnr, psnrb, gbim, mepr, strred) 튜플.
        측정 실패 시 해당 값은 float('nan')으로 반환됩니다.
    """
    if not os.path.exists(ref_video) or not os.path.exists(dist_video):
        return (float('nan'),) * 9

    # 1. PSNR 및 SSIM 측정
    cmd_psnr_ssim = [
        "ffmpeg", "-y", "-i", dist_video, "-i", ref_video,
        "-filter_complex",
        "[0:v]split=2[dist1][dist2];"
        "[1:v]split=2[ref1][ref2];"
        "[dist1][ref1]psnr;"
        "[dist2][ref2]ssim",
        "-f", "null", "-"
    ]
    result = subprocess.run(
        cmd_psnr_ssim, stderr=subprocess.PIPE, text=True, encoding='utf-8'
    )

    psnr_match = re.search(r'PSNR.*average:([0-9.]+|inf)', result.stderr)
    ssim_match = re.search(r'SSIM.*All:([0-9.]+)', result.stderr)

    psnr_val = float(psnr_match.group(1)) if psnr_match else float('nan')
    ssim_val = float(ssim_match.group(1)) if ssim_match else float('nan')

    # 2. VMAF & MS-SSIM 측정 (JSON 로그 생성 후 파싱)
    fd, temp_vmaf_abs = tempfile.mkstemp(
        prefix=f"temp_vmaf_{os.path.basename(dist_video)}_",
        suffix=".json",
        dir=os.getcwd(),
    )
    os.close(fd)
    os.remove(temp_vmaf_abs)
    # libvmaf filter options treat ':' and '\' specially, so pass only a
    # cwd-local basename instead of a Windows absolute path.
    temp_vmaf_log = os.path.basename(temp_vmaf_abs)
    cmd_vmaf = [
        "ffmpeg", "-y", "-i", dist_video, "-i", ref_video,
        "-lavfi", f"libvmaf=log_path={temp_vmaf_log}:log_fmt=json:feature=name=float_ms_ssim",
        "-f", "null", "-"
    ]
    subprocess.run(cmd_vmaf, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    vmaf_val = float('nan')
    ms_ssim_val = float('nan')
    if os.path.exists(temp_vmaf_log):
        try:
            vmaf_val = _read_vmaf_metric(temp_vmaf_log, "vmaf")
            ms_ssim_val = _read_vmaf_metric(temp_vmaf_log, "float_ms_ssim")
        except (json.JSONDecodeError, KeyError):
            pass
        finally:
            os.remove(temp_vmaf_log)

    # 3. OpenCV를 사용한 커스텀 고급 지표(EPSNR, PSNR-B, GBIM, MEPR) 측정
    cap_ref = cv2.VideoCapture(ref_video)
    cap_dist = cv2.VideoCapture(dist_video)
    
    epsnr_val, psnrb_val, gbim_val, mepr_val, strred_val = compute_custom_metrics(
        cap_ref, cap_dist, num_frames=num_frames_custom
    )
    
    cap_ref.release()
    cap_dist.release()

    return (
        psnr_val, ssim_val, vmaf_val, ms_ssim_val,
        epsnr_val, psnrb_val, gbim_val, mepr_val, strred_val,
    )
