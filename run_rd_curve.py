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


def run_spatial_encoding(input_video, output_video, bitrate, max_frames=float('inf')):
    """단순 2D 공간 필터(Gaussian Blur)를 적용한 후 x264로 압축하는 비교군을 생성합니다."""
    print(f"  [Spatial] {bitrate} 전처리 및 인코딩 중 (Gaussian Blur)...")
    w, h, fps = get_video_metadata(input_video)
    encoder_process = create_x264_encoder(output_video, w, h, fps, bitrate)

    total_processed_frames = 0
    # overlap=0 으로 청크를 읽어옴
    for y_array, u_np, v_np, frames, _ in read_y4m_and_split(
        input_video, w, h, chunk_size=8, overlap=0, scene_threshold=1.0
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
        input_video, w, h, chunk_size=8, overlap=overlap_frames
    ):
        if total_processed_frames >= max_frames:
            break

        processed_y = processor.process_chunk(y_array)

        # 겹쳤던 부분(앞쪽 overlap_len 프레임)을 잘라내어 순수 새로운 프레임만 추출
        processed_y_valid = processed_y[overlap_len:]

        processed_y_uint8 = (processed_y_valid * 255.0).clip(0, 255).astype(np.uint8)
        processed_y_flat = processed_y_uint8.reshape((frames, -1))
        
        # U/V 채널 전처리 (크로마 노이즈 제거)
        u_proc, v_proc = processor.process_chroma(u_np, v_np, w, h)

        for f in range(frames):
            encoder_process.stdin.write(processed_y_flat[f].tobytes())
            encoder_process.stdin.write(u_proc[f].tobytes())
            encoder_process.stdin.write(v_proc[f].tobytes())

        total_processed_frames += frames

    encoder_process.stdin.close()
    encoder_process.wait()


def plot_rd_curve(bitrates_kbps, baseline_scores, proposed_scores, title, filename,
                  ylabel="PSNR (dB)", spatial_scores=None):
    """RD Curve 그래프를 생성 및 저장합니다."""
    plt.figure(figsize=(8, 6))
    plt.plot(bitrates_kbps, baseline_scores,
             marker="o", linestyle="-", label="Baseline (x264 only)", color="blue")
    plt.plot(bitrates_kbps, proposed_scores,
             marker="s", linestyle="-", label="Proposed (3D DT-CWT + x264)", color="red")
    
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


def process_single_video(video_name, input_dir, output_dir, bitrates, threshold, disable_overlap, disable_adaptive, include_spatial=False):
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

    base_psnrs, prop_psnrs, spat_psnrs = [], [], []
    base_vmafs, prop_vmafs, spat_vmafs = [], [], []

    for br in bitrates:
        br_str = f"{br}k"
        base_out = os.path.join(output_dir, f"{video_name}_base_{br_str}.mp4")
        prop_out = os.path.join(output_dir, f"{video_name}_prop_{br_str}.mp4")
        spat_out = os.path.join(output_dir, f"{video_name}_spat_{br_str}.mp4")

        run_baseline_encoding(input_video, base_out, br_str)
        run_proposed_encoding(input_video, prop_out, br_str, threshold, disable_overlap=disable_overlap, disable_adaptive=disable_adaptive)
        
        if include_spatial:
            run_spatial_encoding(input_video, spat_out, br_str)

        print(f"  [평가] {video_name} - {br_str} 결과 측정 중 (VMAF 포함)...")
        b_p, b_s, b_v = evaluate_video_quality(input_video, base_out)
        p_p, p_s, p_v = evaluate_video_quality(input_video, prop_out)
        
        if include_spatial:
            s_p, s_s, s_v = evaluate_video_quality(input_video, spat_out)
            spat_psnrs.append(s_p)
            spat_vmafs.append(s_v)
        else:
            s_p, s_v = 0.0, 0.0

        base_psnrs.append(b_p)
        prop_psnrs.append(p_p)
        base_vmafs.append(b_v)
        prop_vmafs.append(p_v)

        if include_spatial:
            print(f"      -> PSNR: B({b_p:.2f}) vs P({p_p:.2f}) vs S({s_p:.2f}) | VMAF: B({b_v:.2f}) vs P({p_v:.2f}) vs S({s_v:.2f})\n")
        else:
            print(f"      -> PSNR: {b_p:.2f} vs {p_p:.2f} | VMAF: {b_v:.2f} vs {p_v:.2f}\n")

    return {
        "video_name": video_name,
        "bitrates": bitrates,
        "base_psnrs": base_psnrs,
        "prop_psnrs": prop_psnrs,
        "spat_psnrs": spat_psnrs if include_spatial else None,
        "base_vmafs": base_vmafs,
        "prop_vmafs": prop_vmafs,
        "spat_vmafs": spat_vmafs if include_spatial else None,
    }


def report_and_save(result, output_dir):
    """단일 비디오의 결과를 출력하고, CSV 및 RD Curve를 저장합니다."""
    video_name = result["video_name"]
    bitrates = result["bitrates"]
    base_psnrs = result["base_psnrs"]
    prop_psnrs = result["prop_psnrs"]
    base_vmafs = result["base_vmafs"]
    prop_vmafs = result["prop_vmafs"]

    spat_psnrs = result.get("spat_psnrs")
    spat_vmafs = result.get("spat_vmafs")

    # BD-Rate 계산
    bd_rate_psnr = calculate_bd_rate(bitrates, base_psnrs, bitrates, prop_psnrs)
    bd_rate_vmaf = calculate_bd_rate(bitrates, base_vmafs, bitrates, prop_vmafs)

    print("-" * 50)
    print(f"  📈 [{video_name.upper()}] 최종 성능 지표")
    print(f"  * BD-Rate (PSNR 기준): {bd_rate_psnr:.3f} %")
    print(f"  * BD-Rate (VMAF 기준): {bd_rate_vmaf:.3f} %")
    print("-" * 50)

    # Raw Data CSV 저장
    csv_filename = os.path.join(output_dir, f"raw_data_{video_name}.csv")
    with open(csv_filename, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if spat_psnrs:
            writer.writerow(["Bitrate(kbps)", "Base_PSNR", "Prop_PSNR", "Spat_PSNR",
                             "Base_VMAF", "Prop_VMAF", "Spat_VMAF"])
            for br, bp, pp, sp, bv, pv, sv in zip(
                bitrates, base_psnrs, prop_psnrs, spat_psnrs, base_vmafs, prop_vmafs, spat_vmafs
            ):
                writer.writerow([br, bp, pp, sp, bv, pv, sv])
        else:
            writer.writerow(["Bitrate(kbps)", "Base_PSNR", "Prop_PSNR",
                             "Base_VMAF", "Prop_VMAF"])
            for br, bp, pp, bv, pv in zip(
                bitrates, base_psnrs, prop_psnrs, base_vmafs, prop_vmafs
            ):
                writer.writerow([br, bp, pp, bv, pv])

    # RD Curve 생성
    plot_rd_curve(
        bitrates, base_psnrs, prop_psnrs,
        title=f"PSNR RD Curve ({video_name.capitalize()}) | BD-Rate: {bd_rate_psnr:.2f}%",
        filename=os.path.join(output_dir, f"rd_curve_psnr_{video_name}.png"),
        spatial_scores=spat_psnrs
    )
    plot_rd_curve(
        bitrates, base_vmafs, prop_vmafs,
        title=f"VMAF RD Curve ({video_name.capitalize()}) | BD-Rate: {bd_rate_vmaf:.2f}%",
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
                process_single_video, name, INPUT_DIR, OUTPUT_DIR, BITRATES, THRESHOLD, args.disable_overlap, args.disable_adaptive_threshold, args.include_spatial
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