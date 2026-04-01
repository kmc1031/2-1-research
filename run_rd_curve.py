"""
RD Curve 실험 자동화: 다중 비디오 × 다중 비트레이트 × Baseline vs Proposed 비교.

Baseline(x264 직접 인코딩)과 Proposed(3D DT-CWT 전처리 + x264)를 여러 비트레이트에서
비교하고, PSNR/VMAF 기반 RD Curve 및 BD-Rate 지표를 산출합니다.

성능 최적화:
- 비디오별 독립 실험을 multiprocessing.Pool로 병렬 처리
- DT-CWT 청크 프리페칭으로 I/O 대기 최소화
"""

import csv
import os
import cv2
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 멀티프로세싱 환경에서 GUI 없이 플롯 생성
import matplotlib.pyplot as plt
import subprocess

from main_pipeline import get_video_metadata, read_y4m_and_split, create_x264_encoder
from dtcwt_processor import DTCWT3DProcessor
from evaluate_metrics import evaluate_video_quality
from advanced_evaluation import _plot_advanced_chart
from compare_frames import plot_comparison
from edge_analysis import analyze_edges


def run_baseline_encoding(input_video, output_video, bitrate):
    """FFmpeg를 사용해 전처리 없이 바로 x264로 압축하는 베이스라인을 생성합니다."""
    print(f"  [Baseline] {bitrate} 인코딩 중...")
    cmd = [
        "ffmpeg", "-y", "-i", input_video,
        "-vcodec", "libx264",
        "-b:v", bitrate,
        "-preset", "fast",
        "-tune", "zerolatency",
        output_video,
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def run_nr_encoding(input_video, output_video, bitrate, nr_strength=100):
    """x264 내장 --nr (Noise Reduction) 옵션을 사용하는 비교군을 생성합니다.

    x264의 --nr 옵션은 DCT 계수에 직접 노이즈 저감을 적용합니다.
    이는 frequency-domain 방식으로, 외부 전처리 없이 코덱 내부에서 처리합니다.

    Args:
        input_video: 입력 비디오 경로.
        output_video: 출력 비디오 경로.
        bitrate: 목표 비트레이트 (예: "200k").
        nr_strength: --nr 강도. 0~65536, 기본 100. 값이 클수록 더 강한 노이즈 저감.
    """
    print(f"  [NR]      {bitrate} 인코딩 중 (x264 --nr={nr_strength})...")
    cmd = [
        "ffmpeg", "-y", "-i", input_video,
        "-vcodec", "libx264",
        "-b:v", bitrate,
        "-preset", "fast",
        "-tune", "zerolatency",
        # x264의 DCT 도메인 노이즈 제거 옵션
        "-x264opts", f"nr={nr_strength}",
        output_video,
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def run_hqdn3d_encoding(input_video, output_video, bitrate,
                         luma_spatial=4.0, luma_temporal=3.0,
                         chroma_spatial=3.0, chroma_temporal=2.5):
    """FFmpeg hqdn3d 시공간 디노이저를 전처리로 사용하는 비교군을 생성합니다.

    hqdn3d(High Quality 3D Denoiser)는 FFmpeg의 표준 시공간 저역통과 필터입니다.
    NLMeans와 달리 매우 빠르며, 방송/스트리밍에서 표준적으로 사용됩니다.

    파라미터 가이드:
        luma_spatial: 공간 축 루마 노이즈 제거 강도 (권장: 2~6)
        luma_temporal: 시간 축 루마 노이즈 제거 강도 (권장: 2~4)
        chroma_*: 크로마 채널 강도 (보통 루마보다 약하게 설정)

    Args:
        input_video: 입력 비디오 경로.
        output_video: 출력 비디오 경로.
        bitrate: 목표 비트레이트.
        luma_spatial: Luma 공간 강도.
        luma_temporal: Luma 시간 강도.
        chroma_spatial: Chroma 공간 강도.
        chroma_temporal: Chroma 시간 강도.
    """
    print(f"  [hqdn3d]  {bitrate} 인코딩 중 "
          f"(ls={luma_spatial},lt={luma_temporal},cs={chroma_spatial},ct={chroma_temporal})...")
    hqdn3d_filter = (
        f"hqdn3d={luma_spatial}:{chroma_spatial}:{luma_temporal}:{chroma_temporal}"
    )
    cmd = [
        "ffmpeg", "-y", "-i", input_video,
        "-vf", hqdn3d_filter,
        "-vcodec", "libx264",
        "-b:v", bitrate,
        "-preset", "fast",
        "-tune", "zerolatency",
        output_video,
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)



def run_spatial_encoding(input_video, output_video, bitrate, max_frames=float('inf')):
    """단순 2D 공간 필터(Gaussian Blur)를 적용한 후 x264로 압축하는 비교군을 생성합니다."""
    print(f"  [Spatial] {bitrate} 전처리 및 인코딩 중 (Gaussian Blur)...")
    w, h, fps = get_video_metadata(input_video)
    encoder_process = create_x264_encoder(output_video, w, h, fps, bitrate)

    total_processed_frames = 0
    # overlap=0 으로 청크를 읽어옴
    for y_array, u_np, v_np, frames, _ in read_y4m_and_split(
        input_video, w, h, fps=fps, chunk_size=8, overlap=0, scene_threshold=100.0
    ):
        if total_processed_frames >= max_frames:
            break

        y_uint8 = (y_array * 255.0).clip(0, 255).astype(np.uint8)
        
        for f in range(frames):
            blurred_y = cv2.GaussianBlur(y_uint8[f], (5, 5), 1.5).flatten()
            encoder_process.stdin.write(blurred_y.tobytes())
            encoder_process.stdin.write(u_np[f].tobytes())
            encoder_process.stdin.write(v_np[f].tobytes())

        total_processed_frames += frames

    encoder_process.stdin.close()
    encoder_process.wait()


def run_dwt3d_encoding(input_video, output_video, bitrate, threshold=0.03, max_frames=float('inf')):
    """일반 3D DWT(Discrete Wavelet Transform, PyWavelets) 기반 전처리 후 인코딩하는 비교군을 생성합니다."""
    print(f"  [DWT3D]   {bitrate} 전처리 및 인코딩 중 (General 3D DWT, T={threshold})...")
    w, h, fps = get_video_metadata(input_video)
    encoder_process = create_x264_encoder(output_video, w, h, fps, bitrate)
    
    try:
        import pywt
    except ImportError:
        print("  [에러] PyWavelets 패키지가 없습니다. 원본 그대로 인코딩합니다.")
        pywt = None

    total_processed_frames = 0
    # DT-CWT와 동일한 조건(오버랩)을 부여하여 공정한 비교를 수행
    for y_array, u_np, v_np, frames, overlap_len in read_y4m_and_split(
        input_video, w, h, fps=fps, chunk_size=8, overlap=4
    ):
        if total_processed_frames >= max_frames:
            break

        if pywt is not None:
            # 3D DWT 변환 (Haar 파장이 3D 비디오 연산에 흔히 쓰임)
            coeffs = pywt.dwtn(y_array, 'haar')
            
            # 고주파 부분(Details)에만 Soft Thresholding 적용
            shrunk_coeffs = {}
            for k, v in coeffs.items():
                if k == 'aaa': # Approximation (Lowpass)
                    shrunk_coeffs[k] = v
                else:          # Details (Highpass)
                    shrunk_coeffs[k] = pywt.threshold(v, threshold, mode='soft')
            
            processed_y = pywt.idwtn(shrunk_coeffs, 'haar')
        else:
            processed_y = y_array

        processed_y_valid = processed_y[overlap_len:]
        processed_y_uint8 = (processed_y_valid * 255.0).clip(0, 255).astype(np.uint8)
        processed_y_flat = processed_y_uint8.reshape((frames, -1))
        
        # 크로마는 원본 유지
        for f in range(frames):
            encoder_process.stdin.write(processed_y_flat[f].tobytes())
            encoder_process.stdin.write(u_np[f].tobytes())
            encoder_process.stdin.write(v_np[f].tobytes())

        total_processed_frames += frames

    encoder_process.stdin.close()
    encoder_process.wait()


def run_proposed_encoding(input_video, output_video, bitrate, threshold,
                          max_frames=float('inf'), disable_overlap=False, disable_adaptive=False):
    """3D DT-CWT 전처리를 거친 후 x264로 압축하는 제안 기법을 생성합니다."""
    print(f"  [Proposed] {bitrate} 전처리 및 인코딩 중 (T={threshold})...")
    w, h, fps = get_video_metadata(input_video)
    encoder_process = create_x264_encoder(output_video, w, h, fps, bitrate)
    
    adaptive = not disable_adaptive
    processor = DTCWT3DProcessor(threshold=threshold, adaptive_threshold=adaptive)

    overlap_frames = 0 if disable_overlap else 4
    total_processed_frames = 0
    for y_array, u_np, v_np, frames, overlap_len in read_y4m_and_split(
        input_video, w, h, fps=fps, chunk_size=8, overlap=overlap_frames
    ):
        if total_processed_frames >= max_frames:
            break

        processed_y = processor.process_chunk(y_array, overlap_len=overlap_len)

        # 겹쳤던 부분(앞쪽 overlap_len 프레임)을 잘라내어 순수 새로운 프레임만 추출
        processed_y_valid = processed_y[overlap_len:overlap_len + frames]

        valid_frames = processed_y_valid.shape[0]
        processed_y_uint8 = (processed_y_valid * 255.0).clip(0, 255).astype(np.uint8)
        processed_y_flat = processed_y_uint8.reshape((valid_frames, -1))
        
        # U/V 채널 전처리 (크로마 노이즈 제거)
        u_proc, v_proc = processor.process_chroma(u_np, v_np, w, h)

        for f in range(valid_frames):
            encoder_process.stdin.write(processed_y_flat[f].tobytes())
            encoder_process.stdin.write(u_proc[f].tobytes())
            encoder_process.stdin.write(v_proc[f].tobytes())

        total_processed_frames += valid_frames

    encoder_process.stdin.close()
    encoder_process.wait()


def plot_rd_curve(bitrates_kbps, baseline_scores, proposed_scores, dwt_scores, title, filename,
                  ylabel="PSNR (dB)", spatial_scores=None):
    """RD Curve 그래프를 생성 및 저장합니다."""
    plt.figure(figsize=(8, 6))
    plt.plot(bitrates_kbps, baseline_scores,
             marker="o", linestyle="-", label="Baseline (x264 only)", color="blue")
    plt.plot(bitrates_kbps, dwt_scores,
             marker="s", linestyle="-.", label="Ablation (General 3D DWT)", color="orange")
    plt.plot(bitrates_kbps, proposed_scores,
             marker="D", linestyle="-", label="Proposed (3D DT-CWT + x264)", color="red")
    
    if spatial_scores:
        plt.plot(bitrates_kbps, spatial_scores,
                 marker="^", linestyle="--", label="Spatial (Gaussian)", color="green")

    plt.title(title, fontsize=14)
    plt.xlabel("Bitrate (kbps)", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"\n=> 그래프가 저장되었습니다: {filename}")


def calculate_bd_rate(R1, PSNR1, R2, PSNR2):
    """Bjontegaard Delta Rate (BD-Rate): 동일 화질 대비 비트레이트 절감률(%)."""
    valid1 = [(r, p) for r, p in zip(R1, PSNR1) if p is not None and not np.isnan(p)]
    valid2 = [(r, p) for r, p in zip(R2, PSNR2) if p is not None and not np.isnan(p)]
    if len(valid1) < 4 or len(valid2) < 4:
        return float("nan")

    R1, PSNR1 = zip(*valid1)
    R2, PSNR2 = zip(*valid2)

    lR1, lR2 = np.log10(R1), np.log10(R2)

    p1 = np.polyfit(PSNR1, lR1, 3)
    p2 = np.polyfit(PSNR2, lR2, 3)

    min_psnr = max(min(PSNR1), min(PSNR2))
    max_psnr = min(max(PSNR1), max(PSNR2))

    if max_psnr <= min_psnr:
        return float("nan")

    int1 = np.polyint(p1)
    int2 = np.polyint(p2)

    avg_diff = (
        (np.polyval(int2, max_psnr) - np.polyval(int2, min_psnr))
        - (np.polyval(int1, max_psnr) - np.polyval(int1, min_psnr))
    ) / (max_psnr - min_psnr)

    return (10**avg_diff - 1) * 100


def calculate_bd_psnr(R1, PSNR1, R2, PSNR2):
    """Bjontegaard Delta PSNR: 동일 비트레이트 대비 화질 향상도(dB)."""
    valid1 = [(r, p) for r, p in zip(R1, PSNR1) if p is not None and not np.isnan(p)]
    valid2 = [(r, p) for r, p in zip(R2, PSNR2) if p is not None and not np.isnan(p)]
    if len(valid1) < 4 or len(valid2) < 4:
        return float("nan")

    R1, PSNR1 = zip(*valid1)
    R2, PSNR2 = zip(*valid2)

    lR1, lR2 = np.log10(R1), np.log10(R2)

    p1 = np.polyfit(lR1, PSNR1, 3)
    p2 = np.polyfit(lR2, PSNR2, 3)

    min_logR = max(min(lR1), min(lR2))
    max_logR = min(max(lR1), max(lR2))

    if max_logR <= min_logR:
        return float("nan")

    int1 = np.polyint(p1)
    int2 = np.polyint(p2)

    avg_diff = (
        (np.polyval(int2, max_logR) - np.polyval(int2, min_logR))
        - (np.polyval(int1, max_logR) - np.polyval(int1, min_logR))
    ) / (max_logR - min_logR)

    return avg_diff


def process_single_video(video_name, input_dir, output_dir, bitrates, threshold, disable_overlap, disable_adaptive, include_spatial=False, visualize_frame=None):
    """단일 비디오에 대해 모든 비트레이트의 인코딩 + 평가를 수행합니다.

    이 함수는 ProcessPoolExecutor의 워커에서 호출되므로,
    독립적으로 동작하며 결과를 딕셔너리로 반환합니다.

    Returns:
        결과 딕셔너리 또는 None (비디오 파일이 없는 경우).
    """
    input_video = os.path.join(input_dir, f"{video_name}.y4m")
    if not os.path.exists(input_video):
        return None

    print(f"\n{'=' * 50}")
    print(f"  🎬 타겟 비디오: {video_name.upper()}")
    print(f"{'=' * 50}")

    base_psnrs, prop_psnrs, spat_psnrs, dwt_psnrs = [], [], [], []
    base_ssims, prop_ssims, spat_ssims, dwt_ssims = [], [], [], []
    base_vmafs, prop_vmafs, spat_vmafs, dwt_vmafs = [], [], [], []
    base_msssims, prop_msssims, spat_msssims, dwt_msssims = [], [], [], []
    base_epsnrs, prop_epsnrs, spat_epsnrs, dwt_epsnrs = [], [], [], []
    base_psnrbs, prop_psnrbs, spat_psnrbs, dwt_psnrbs = [], [], [], []
    base_gbims, prop_gbims, spat_gbims, dwt_gbims = [], [], [], []
    base_meprs, prop_meprs, spat_meprs, dwt_meprs = [], [], [], []

    for br in bitrates:
        br_str = f"{br}k"
        base_out = os.path.join(output_dir, f"{video_name}_base_{br_str}.mp4")
        prop_out = os.path.join(output_dir, f"{video_name}_prop_{br_str}.mp4")
        spat_out = os.path.join(output_dir, f"{video_name}_spat_{br_str}.mp4")
        dwt_out = os.path.join(output_dir, f"{video_name}_dwt3d_{br_str}.mp4")

        run_baseline_encoding(input_video, base_out, br_str)
        run_dwt3d_encoding(input_video, dwt_out, br_str, threshold)
        run_proposed_encoding(input_video, prop_out, br_str, threshold, disable_overlap=disable_overlap, disable_adaptive=disable_adaptive)
        
        if include_spatial:
            run_spatial_encoding(input_video, spat_out, br_str)

        print(f"  [평가] {video_name} - {br_str} 결과 측정 중 (고급 지표 포함)...")
        b_p, b_s, b_v, b_ms, b_ep, b_pb, b_gb, b_me = evaluate_video_quality(input_video, base_out, num_frames_custom=60)
        d_p, d_s, d_v, d_ms, d_ep, d_pb, d_gb, d_me = evaluate_video_quality(input_video, dwt_out, num_frames_custom=60)
        p_p, p_s, p_v, p_ms, p_ep, p_pb, p_gb, p_me = evaluate_video_quality(input_video, prop_out, num_frames_custom=60)
        
        if include_spatial:
            s_p, s_s, s_v, s_ms, s_ep, s_pb, s_gb, s_me = evaluate_video_quality(input_video, spat_out, num_frames_custom=60)
            spat_psnrs.append(s_p); spat_ssims.append(s_s); spat_vmafs.append(s_v); spat_msssims.append(s_ms)
            spat_epsnrs.append(s_ep); spat_psnrbs.append(s_pb); spat_gbims.append(s_gb); spat_meprs.append(s_me)
        else:
            s_p, s_v = float('nan'), float('nan')

        base_psnrs.append(b_p); base_ssims.append(b_s); base_vmafs.append(b_v); base_msssims.append(b_ms)
        base_epsnrs.append(b_ep); base_psnrbs.append(b_pb); base_gbims.append(b_gb); base_meprs.append(b_me)
        
        dwt_psnrs.append(d_p); dwt_ssims.append(d_s); dwt_vmafs.append(d_v); dwt_msssims.append(d_ms)
        dwt_epsnrs.append(d_ep); dwt_psnrbs.append(d_pb); dwt_gbims.append(d_gb); dwt_meprs.append(d_me)
        
        prop_psnrs.append(p_p); prop_ssims.append(p_s); prop_vmafs.append(p_v); prop_msssims.append(p_ms)
        prop_epsnrs.append(p_ep); prop_psnrbs.append(p_pb); prop_gbims.append(p_gb); prop_meprs.append(p_me)

        # Plot advanced chart for this bitrate
        _plot_advanced_chart(
            video_name, br_str,
            b_ms, p_ms,
            b_ep, p_ep,
            b_pb, p_pb,
            b_gb, p_gb,
            b_me, p_me,
            output_dir
        )

        if visualize_frame is not None:
            import sys
            plot_comparison(video_name, br_str, visualize_frame)
            analyze_edges(video_name, br_str, visualize_frame)
            subprocess.run([sys.executable, "visualize_residuals.py", "-o", input_video, "-p", prop_out, "-f", str(visualize_frame), 
                            "--out", os.path.join(output_dir, f"residual_prop_{video_name}_{br_str}_f{visualize_frame}.png")], stdout=subprocess.DEVNULL)
            subprocess.run([sys.executable, "visualize_residuals.py", "-o", input_video, "-p", base_out, "-f", str(visualize_frame), 
                            "--out", os.path.join(output_dir, f"residual_base_{video_name}_{br_str}_f{visualize_frame}.png")], stdout=subprocess.DEVNULL)

        if include_spatial:
            print(f"      -> PSNR: B({b_p:.2f}) vs DWT({d_p:.2f}) vs DT({p_p:.2f}) vs S({s_p:.2f}) | VMAF: B({b_v:.2f}) vs DWT({d_v:.2f}) vs DT({p_v:.2f}) vs S({s_v:.2f})\n")
        else:
            print(f"      -> PSNR: B({b_p:.2f}) vs DWT({d_p:.2f}) vs DT({p_p:.2f}) | VMAF: B({b_v:.2f}) vs DWT({d_v:.2f}) vs DT({p_v:.2f})\n")

    return {
        "video_name": video_name, "bitrates": bitrates,
        "base_psnrs": base_psnrs, "dwt_psnrs": dwt_psnrs, "prop_psnrs": prop_psnrs, "spat_psnrs": spat_psnrs if include_spatial else None,
        "base_ssims": base_ssims, "dwt_ssims": dwt_ssims, "prop_ssims": prop_ssims, "spat_ssims": spat_ssims if include_spatial else None,
        "base_vmafs": base_vmafs, "dwt_vmafs": dwt_vmafs, "prop_vmafs": prop_vmafs, "spat_vmafs": spat_vmafs if include_spatial else None,
        "base_msssims": base_msssims, "dwt_msssims": dwt_msssims, "prop_msssims": prop_msssims, "spat_msssims": spat_msssims if include_spatial else None,
        "base_epsnrs": base_epsnrs, "dwt_epsnrs": dwt_epsnrs, "prop_epsnrs": prop_epsnrs, "spat_epsnrs": spat_epsnrs if include_spatial else None,
        "base_psnrbs": base_psnrbs, "dwt_psnrbs": dwt_psnrbs, "prop_psnrbs": prop_psnrbs, "spat_psnrbs": spat_psnrbs if include_spatial else None,
        "base_gbims": base_gbims, "dwt_gbims": dwt_gbims, "prop_gbims": prop_gbims, "spat_gbims": spat_gbims if include_spatial else None,
        "base_meprs": base_meprs, "dwt_meprs": dwt_meprs, "prop_meprs": prop_meprs, "spat_meprs": spat_meprs if include_spatial else None,
    }


def _safe(values):
    """None 값을 float('nan')으로 치환하여 분석 왜곡을 방지합니다."""
    return [v if v is not None else float('nan') for v in values]


def report_and_save(result, output_dir):
    """단일 비디오의 결과를 출력하고, CSV 및 RD Curve를 저장합니다."""
    video_name = result["video_name"]
    bitrates = result["bitrates"]
    base_psnrs = _safe(result["base_psnrs"])
    dwt_psnrs = _safe(result["dwt_psnrs"])
    prop_psnrs = _safe(result["prop_psnrs"])
    base_ssims = _safe(result["base_ssims"]); dwt_ssims = _safe(result["dwt_ssims"]); prop_ssims = _safe(result["prop_ssims"])
    base_vmafs = _safe(result["base_vmafs"])
    dwt_vmafs = _safe(result["dwt_vmafs"])
    prop_vmafs = _safe(result["prop_vmafs"])
    
    base_msssims = _safe(result["base_msssims"]); dwt_msssims = _safe(result["dwt_msssims"]); prop_msssims = _safe(result["prop_msssims"])
    base_epsnrs = _safe(result["base_epsnrs"]); dwt_epsnrs = _safe(result["dwt_epsnrs"]); prop_epsnrs = _safe(result["prop_epsnrs"])
    base_psnrbs = _safe(result["base_psnrbs"]); dwt_psnrbs = _safe(result["dwt_psnrbs"]); prop_psnrbs = _safe(result["prop_psnrbs"])
    base_gbims = _safe(result["base_gbims"]); dwt_gbims = _safe(result["dwt_gbims"]); prop_gbims = _safe(result["prop_gbims"])
    base_meprs = _safe(result["base_meprs"]); dwt_meprs = _safe(result["dwt_meprs"]); prop_meprs = _safe(result["prop_meprs"])

    spat_psnrs = _safe(result["spat_psnrs"]) if result.get("spat_psnrs") else None
    spat_ssims = _safe(result["spat_ssims"]) if result.get("spat_ssims") else None
    spat_vmafs = _safe(result["spat_vmafs"]) if result.get("spat_vmafs") else None
    spat_msssims = _safe(result["spat_msssims"]) if result.get("spat_msssims") else None
    spat_epsnrs = _safe(result["spat_epsnrs"]) if result.get("spat_epsnrs") else None
    spat_psnrbs = _safe(result["spat_psnrbs"]) if result.get("spat_psnrbs") else None
    spat_gbims = _safe(result["spat_gbims"]) if result.get("spat_gbims") else None
    spat_meprs = _safe(result["spat_meprs"]) if result.get("spat_meprs") else None

    # BD-Rate 계산
    bd_rate_psnr = calculate_bd_rate(bitrates, base_psnrs, bitrates, prop_psnrs)
    bd_rate_vmaf = calculate_bd_rate(bitrates, base_vmafs, bitrates, prop_vmafs)

    bd_str_psnr = f"{bd_rate_psnr:.3f} %" if not np.isnan(bd_rate_psnr) else "N/A (데이터 부족)"
    bd_str_vmaf = f"{bd_rate_vmaf:.3f} %" if not np.isnan(bd_rate_vmaf) else "N/A (데이터 부족)"

    print("-" * 50)
    print(f"  📈 [{video_name.upper()}] 최종 성능 지표")
    print(f"  * BD-Rate (PSNR 기준): {bd_str_psnr}")
    print(f"  * BD-Rate (VMAF 기준): {bd_str_vmaf}")
    print("-" * 50)

    # Raw Data CSV 저장
    csv_filename = os.path.join(output_dir, f"raw_data_{video_name}.csv")
    with open(csv_filename, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if spat_psnrs:
            writer.writerow([
                "Bitrate(kbps)", "Base_PSNR", "DWT_PSNR", "Prop_PSNR", "Spat_PSNR",
                "Base_SSIM", "DWT_SSIM", "Prop_SSIM", "Spat_SSIM",
                "Base_VMAF", "DWT_VMAF", "Prop_VMAF", "Spat_VMAF",
                "Base_MSSSIM", "DWT_MSSSIM", "Prop_MSSSIM", "Spat_MSSSIM",
                "Base_EPSNR", "DWT_EPSNR", "Prop_EPSNR", "Spat_EPSNR",
                "Base_PSNRB", "DWT_PSNRB", "Prop_PSNRB", "Spat_PSNRB",
                "Base_GBIM", "DWT_GBIM", "Prop_GBIM", "Spat_GBIM",
                "Base_MEPR", "DWT_MEPR", "Prop_MEPR", "Spat_MEPR"
            ])
            for (
                br, bp, dp, pp, sp, bss, dss, pss, sss, bv, dv, pv, sv,
                bms, dms, pms, sms, bep, dep, pep, sep,
                bpb, dpb, ppb, spb, bgb, dgb, pgb, sgb, bme, dme, pme, sme
            ) in zip(
                bitrates, base_psnrs, dwt_psnrs, prop_psnrs, spat_psnrs,
                base_ssims, dwt_ssims, prop_ssims, spat_ssims,
                base_vmafs, dwt_vmafs, prop_vmafs, spat_vmafs,
                base_msssims, dwt_msssims, prop_msssims, spat_msssims,
                base_epsnrs, dwt_epsnrs, prop_epsnrs, spat_epsnrs,
                base_psnrbs, dwt_psnrbs, prop_psnrbs, spat_psnrbs,
                base_gbims, dwt_gbims, prop_gbims, spat_gbims,
                base_meprs, dwt_meprs, prop_meprs, spat_meprs
            ):
                writer.writerow([
                    br, bp, dp, pp, sp, bss, dss, pss, sss, bv, dv, pv, sv,
                    bms, dms, pms, sms, bep, dep, pep, sep,
                    bpb, dpb, ppb, spb, bgb, dgb, pgb, sgb, bme, dme, pme, sme
                ])
        else:
            writer.writerow([
                "Bitrate(kbps)", "Base_PSNR", "DWT_PSNR", "Prop_PSNR",
                "Base_SSIM", "DWT_SSIM", "Prop_SSIM",
                "Base_VMAF", "DWT_VMAF", "Prop_VMAF",
                "Base_MSSSIM", "DWT_MSSSIM", "Prop_MSSSIM",
                "Base_EPSNR", "DWT_EPSNR", "Prop_EPSNR",
                "Base_PSNRB", "DWT_PSNRB", "Prop_PSNRB",
                "Base_GBIM", "DWT_GBIM", "Prop_GBIM",
                "Base_MEPR", "DWT_MEPR", "Prop_MEPR"
            ])
            for (
                br, bp, dp, pp, bss, dss, pss, bv, dv, pv,
                bms, dms, pms, bep, dep, pep,
                bpb, dpb, ppb, bgb, dgb, pgb, bme, dme, pme
            ) in zip(
                bitrates, base_psnrs, dwt_psnrs, prop_psnrs,
                base_ssims, dwt_ssims, prop_ssims,
                base_vmafs, dwt_vmafs, prop_vmafs,
                base_msssims, dwt_msssims, prop_msssims,
                base_epsnrs, dwt_epsnrs, prop_epsnrs,
                base_psnrbs, dwt_psnrbs, prop_psnrbs,
                base_gbims, dwt_gbims, prop_gbims,
                base_meprs, dwt_meprs, prop_meprs
            ):
                writer.writerow([
                    br, bp, dp, pp, bss, dss, pss, bv, dv, pv,
                    bms, dms, pms, bep, dep, pep,
                    bpb, dpb, ppb, bgb, dgb, pgb, bme, dme, pme
                ])

    bd_title_psnr = f"{bd_rate_psnr:.2f}%" if not np.isnan(bd_rate_psnr) else "N/A"
    bd_title_vmaf = f"{bd_rate_vmaf:.2f}%" if not np.isnan(bd_rate_vmaf) else "N/A"
    # RD Curve 생성
    plot_rd_curve(
        bitrates, base_psnrs, prop_psnrs, dwt_psnrs,
        title=f"PSNR RD Curve ({video_name.capitalize()}) | BD-Rate: {bd_title_psnr}",
        filename=os.path.join(output_dir, f"rd_curve_psnr_{video_name}.png"),
        spatial_scores=spat_psnrs
    )
    plot_rd_curve(
        bitrates, base_vmafs, prop_vmafs, dwt_vmafs,
        title=f"VMAF RD Curve ({video_name.capitalize()}) | BD-Rate: {bd_title_vmaf}",
        filename=os.path.join(output_dir, f"rd_curve_vmaf_{video_name}.png"),
        ylabel="VMAF Score",
        spatial_scores=spat_vmafs
    )


# --- 메인 실험 루프 ---
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RD Curve 실험을 위한 자동화 파이프라인")
    parser.add_argument("-v", "--video_names", nargs='+', default=["akiyo", "foreman", "mobile", "stefan"], help="처리할 비디오 이름 목록 (예: akiyo foreman)")
    parser.add_argument("-i", "--input_dir", default="./videos", help="입력 비디오 디렉토리")
    parser.add_argument("-o", "--output_dir", default="./outputs", help="출력 디렉토리 (자동 생성)")
    parser.add_argument("-b", "--bitrates", nargs='+', type=int, default=[100, 200, 300, 400, 500], help="테스트할 비트레이트 목록(kbps)")
    parser.add_argument("-t", "--threshold", type=float, default=0.03, help="DT-CWT 임계값")
    parser.add_argument("--max_workers", type=int, default=None, help="병렬 처리 워커 수 (기본값: 코어 수에 맞게 자동 설정)")
    parser.add_argument("--disable_overlap", action="store_true", help="오버랩 방식 블록 기반 처리 비활성화")
    parser.add_argument("--disable_adaptive_threshold", action="store_true", help="적응형 임계값 산출 로직 비활성화")
    parser.add_argument("--include_spatial", action="store_true", help="단순 2D 공간 필터(Gaussian) 비교군 포함")
    parser.add_argument("--visualize_frame", type=int, default=None, help="프레임 비교/에지/잔차 시각화를 수행할 특정 프레임 번호")
    
    args = parser.parse_args()

    VIDEO_NAMES = args.video_names
    INPUT_DIR = args.input_dir
    OUTPUT_DIR = args.output_dir
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    BITRATES = args.bitrates
    THRESHOLD = args.threshold
    MAX_WORKERS = min(len(VIDEO_NAMES), args.max_workers if args.max_workers else (os.cpu_count() or 1))

    print(f"=== 🚀 다중 비디오 자동화 시작 (병렬: {MAX_WORKERS}개 워커) ===")

    # 비디오별로 병렬 처리
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(
                process_single_video, name, INPUT_DIR, OUTPUT_DIR, BITRATES, THRESHOLD, args.disable_overlap, args.disable_adaptive_threshold, args.include_spatial, args.visualize_frame
            ): name
            for name in VIDEO_NAMES
        }

        for future in as_completed(futures):
            video_name = futures[future]
            try:
                result = future.result()
                if result is not None:
                    report_and_save(result, OUTPUT_DIR)
            except Exception as e:
                print(f"🚨 [{video_name}] 처리 중 오류 발생: {e}")

    print("\n=== ✅ 전체 실험 완료 ===")