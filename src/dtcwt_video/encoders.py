"""
dtcwt_video.encoders: 비교군 인코딩 함수 및 BD-Rate 계산 유틸리티.

이 모듈은 scripts/run_rd_curve.py와 scripts/run_noise_experiment.py에서
공통으로 사용하는 인코딩 함수와 Bjontegaard Delta 계산 함수를 제공합니다.

비교군:
    run_baseline_encoding  — x264 직접 인코딩 (전처리 없음)
    run_nr_encoding        — x264 --nr 옵션 (DCT 도메인 노이즈 저감)
    run_hqdn3d_encoding    — FFmpeg hqdn3d 시공간 디노이저 + x264
    run_spatial_encoding   — 2D Gaussian Blur + x264
    run_dwt3d_encoding     — 3D DWT (PyWavelets Haar) + x264
    run_proposed_encoding  — 3D DT-CWT + BayesShrink 임계값 + x264 (Proposed)

Bjontegaard Delta:
    calculate_bd_rate      — BD-Rate (%) : 동일 화질 대비 비트레이트 절감률
    calculate_bd_psnr      — BD-PSNR (dB): 동일 비트레이트 대비 화질 향상
    _safe                  — None/nan 값 안전 처리 유틸리티
"""

import subprocess
import csv

import cv2
import numpy as np

from dtcwt_video.pipeline import get_video_metadata, read_y4m_and_split, create_x264_encoder, build_processing_context
from dtcwt_video.dtcwt_processor import DTCWT3DProcessor


# ─── 인코딩 함수 ───────────────────────────────────────────────────────────────

def run_baseline_encoding(input_video: str, output_video: str, bitrate: str) -> None:
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


def run_nr_encoding(input_video: str, output_video: str, bitrate: str,
                    nr_strength: int = 100) -> None:
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
        "-x264opts", f"nr={nr_strength}",
        output_video,
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def run_hqdn3d_encoding(input_video: str, output_video: str, bitrate: str,
                         luma_spatial: float = 4.0, luma_temporal: float = 3.0,
                         chroma_spatial: float = 3.0, chroma_temporal: float = 2.5) -> None:
    """FFmpeg hqdn3d 시공간 디노이저를 전처리로 사용하는 비교군을 생성합니다.

    hqdn3d(High Quality 3D Denoiser)는 FFmpeg의 표준 시공간 저역통과 필터입니다.
    NLMeans와 달리 매우 빠르며, 방송/스트리밍에서 표준적으로 사용됩니다.

    파라미터 가이드:
        luma_spatial:   공간 축 루마 노이즈 제거 강도 (권장: 2~6)
        luma_temporal:  시간 축 루마 노이즈 제거 강도 (권장: 2~4)
        chroma_*:       크로마 채널 강도 (보통 루마보다 약하게 설정)

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


def run_spatial_encoding(input_video: str, output_video: str, bitrate: str,
                          max_frames: float = float('inf')) -> None:
    """단순 2D 공간 필터(Gaussian Blur)를 적용한 후 x264로 압축하는 비교군을 생성합니다."""
    print(f"  [Spatial] {bitrate} 전처리 및 인코딩 중 (Gaussian Blur)...")
    w, h, fps = get_video_metadata(input_video)
    encoder_process = create_x264_encoder(output_video, w, h, fps, bitrate)

    total_processed_frames = 0
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


def run_dwt3d_encoding(input_video: str, output_video: str, bitrate: str,
                        threshold: float = 0.03,
                        max_frames: float = float('inf')) -> None:
    """일반 3D DWT(PyWavelets Haar) 기반 전처리 후 인코딩하는 비교군을 생성합니다.

    DT-CWT와 동일한 청크/오버랩 조건을 부여하여 공정한 abaltion 비교를 수행합니다.

    Args:
        input_video: 입력 비디오 경로.
        output_video: 출력 비디오 경로.
        bitrate: 목표 비트레이트.
        threshold: Soft-thresholding 임계값 (고주파 서브밴드에 적용).
        max_frames: 최대 처리 프레임 수.
    """
    print(f"  [DWT3D]   {bitrate} 전처리 및 인코딩 중 (General 3D DWT, T={threshold})...")
    w, h, fps = get_video_metadata(input_video)
    encoder_process = create_x264_encoder(output_video, w, h, fps, bitrate)

    try:
        import pywt
    except ImportError:
        print("  [에러] PyWavelets 패키지가 없습니다. 원본 그대로 인코딩합니다.")
        pywt = None

    total_processed_frames = 0
    for y_array, u_np, v_np, frames, overlap_len in read_y4m_and_split(
        input_video, w, h, fps=fps, chunk_size=8, overlap=4
    ):
        if total_processed_frames >= max_frames:
            break

        if pywt is not None:
            coeffs = pywt.dwtn(y_array, 'haar')
            shrunk_coeffs = {
                k: v if k == 'aaa' else pywt.threshold(v, threshold, mode='soft')
                for k, v in coeffs.items()
            }
            processed_y = pywt.idwtn(shrunk_coeffs, 'haar')
        else:
            processed_y = y_array

        processed_y_valid = processed_y[overlap_len:]
        processed_y_uint8 = (processed_y_valid * 255.0).clip(0, 255).astype(np.uint8)
        processed_y_flat = processed_y_uint8.reshape((frames, -1))

        for f in range(frames):
            encoder_process.stdin.write(processed_y_flat[f].tobytes())
            encoder_process.stdin.write(u_np[f].tobytes())
            encoder_process.stdin.write(v_np[f].tobytes())

        total_processed_frames += frames

    encoder_process.stdin.close()
    encoder_process.wait()


def run_proposed_encoding(input_video: str, output_video: str, bitrate: str,
                           threshold: float,
                           max_frames: float = float('inf'),
                           disable_overlap: bool = False,
                           disable_adaptive: bool = False,
                           threshold_mode: str = "adaptive",
                           controller_a: float = 0.35,
                           controller_b: float = 0.25,
                           controller_c: float = 0.25,
                           controller_d: float = 0.25,
                           min_multiplier: float = 0.5,
                           max_multiplier: float = 2.5,
                           disable_rate_aware_scene_reset: bool = False,
                           log_context_path: str | None = None) -> None:
    """3D DT-CWT 전처리를 거친 후 x264로 압축하는 제안 기법을 생성합니다.

    Args:
        input_video: 입력 비디오 경로.
        output_video: 출력 비디오 경로.
        bitrate: 목표 비트레이트.
        threshold: BayesShrink 기본 임계값.
        max_frames: 최대 처리 프레임 수.
        disable_overlap: True이면 청크 간 오버랩을 비활성화.
        disable_adaptive: True이면 적응형 임계값을 비활성화.
        threshold_mode: fixed / adaptive / rate_aware.
        controller_*: rate-aware 계수.
        log_context_path: 지정 시 청크별 컨텍스트 CSV 기록.
    """
    print(f"  [Proposed] {bitrate} 전처리 및 인코딩 중 (T={threshold}, mode={threshold_mode})...")
    w, h, fps = get_video_metadata(input_video)
    encoder_process = create_x264_encoder(output_video, w, h, fps, bitrate)

    adaptive = not disable_adaptive
    processor = DTCWT3DProcessor(
        threshold=threshold,
        adaptive_threshold=adaptive,
        threshold_mode=threshold_mode if not disable_adaptive else "fixed",
        controller_a=controller_a,
        controller_b=controller_b,
        controller_c=controller_c,
        controller_d=controller_d,
        min_multiplier=min_multiplier,
        max_multiplier=max_multiplier,
        disable_rate_aware_scene_reset=disable_rate_aware_scene_reset,
    )

    log_writer = None
    if log_context_path:
        log_fp = open(log_context_path, "w", newline="", encoding="utf-8")
        log_writer = csv.writer(log_fp)
        log_writer.writerow(["chunk", "overlap", "bitrate_kbps", "noise", "motion",
                             "edge_density", "scene_cut", "threshold_mode", "multiplier"])

    overlap_frames = 0 if disable_overlap else 4
    total_processed_frames = 0
    chunk_idx = 0
    for y_array, u_np, v_np, frames, overlap_len, scene_cut in read_y4m_and_split(
        input_video, w, h, fps=fps, chunk_size=8, overlap=overlap_frames,
        return_scene_change=True
    ):
        if total_processed_frames >= max_frames:
            break

        ctx_log = {
            "bitrate_kbps": np.nan,
            "noise": np.nan,
            "motion": np.nan,
            "edge_density": np.nan,
        }
        if threshold_mode == "rate_aware":
            ctx, ctx_log = build_processing_context(
                y_array, bitrate, chunk_idx, fps, scene_cut, threshold_mode
            )
            processor.set_context(ctx)
        else:
            processor.set_context(None)

        processed_y = processor.process_chunk(y_array, overlap_len=overlap_len)
        processed_y_valid = processed_y[overlap_len:overlap_len + frames]

        valid_frames = processed_y_valid.shape[0]
        processed_y_uint8 = (processed_y_valid * 255.0).clip(0, 255).astype(np.uint8)
        processed_y_flat = processed_y_uint8.reshape((valid_frames, -1))

        u_proc, v_proc = processor.process_chroma(u_np, v_np, w, h)

        for f in range(valid_frames):
            encoder_process.stdin.write(processed_y_flat[f].tobytes())
            encoder_process.stdin.write(u_proc[f].tobytes())
            encoder_process.stdin.write(v_proc[f].tobytes())

        total_processed_frames += valid_frames

        if log_writer:
            log_writer.writerow([
                chunk_idx,
                overlap_len,
                ctx_log["bitrate_kbps"] if threshold_mode == "rate_aware" else np.nan,
                ctx_log["noise"] if threshold_mode == "rate_aware" else np.nan,
                ctx_log["motion"] if threshold_mode == "rate_aware" else np.nan,
                ctx_log["edge_density"] if threshold_mode == "rate_aware" else np.nan,
                scene_cut,
                threshold_mode,
                processor.compute_controller_multiplier(),
            ])

        chunk_idx += 1

    encoder_process.stdin.close()
    encoder_process.wait()
    if log_writer:
        log_fp.close()


# ─── BD-Rate / BD-PSNR ────────────────────────────────────────────────────────

def _safe(values: list) -> list[float]:
    """None 값을 float('nan')으로 치환하여 분석 왜곡을 방지합니다."""
    return [v if v is not None else float('nan') for v in values]


def calculate_bd_rate(R1, PSNR1, R2, PSNR2) -> float:
    """Bjontegaard Delta Rate (BD-Rate): 동일 화질 대비 비트레이트 절감률(%).

    음수 값: 방법 2(R2)가 방법 1(R1) 대비 더 적은 비트레이트 사용 = 개선.
    양수 값: 방법 2가 더 많은 비트레이트 사용 = 열화.

    Args:
        R1, PSNR1: 참조(Baseline) 비트레이트/PSNR 목록.
        R2, PSNR2: 비교(Proposed) 비트레이트/PSNR 목록.

    Returns:
        BD-Rate (%). 데이터 포인트가 4개 미만이면 float('nan').
    """
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


def calculate_bd_psnr(R1, PSNR1, R2, PSNR2) -> float:
    """Bjontegaard Delta PSNR: 동일 비트레이트 대비 화질 향상도(dB).

    양수 값: 방법 2(R2)가 방법 1(R1) 대비 더 높은 PSNR = 개선.

    Args:
        R1, PSNR1: 참조(Baseline) 비트레이트/PSNR 목록.
        R2, PSNR2: 비교(Proposed) 비트레이트/PSNR 목록.

    Returns:
        BD-PSNR (dB). 데이터 포인트가 4개 미만이면 float('nan').
    """
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
