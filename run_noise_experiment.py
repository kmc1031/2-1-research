"""
Clean vs Noisy 조건 분리 RD 실험.

핵심 실험 설계:
- 입력: clean 영상에 Gaussian 노이즈(σ=0,5,10,15)를 추가한 영상
- 비교: Baseline(x264 직접) vs Proposed(3D DT-CWT + x264) vs DWT(3D DWT + x264)
- 참조: 항상 clean 원본 (노이즈 없는 원본으로 품질 평가)
- 출력: 조건별 RD Curve, BD-Rate 테이블, 조건 간 비교 히트맵

이 실험의 목적:
"3D DT-CWT 전처리가 noisy 영상의 저비트레이트 인코딩에서
 baseline보다 RD 성능을 유의미하게 개선하는가?"

Usage:
    uv run python run_noise_experiment.py
    uv run python run_noise_experiment.py -v akiyo foreman --sigma 0 5 10 15
    uv run python run_noise_experiment.py -v akiyo -b 100 200 300 400 500 --sigma 0 10
"""

import argparse
import csv
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 한글 폰트 설정 (Windows: Malgun Gothic)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

import ffmpeg

from main_pipeline import get_video_metadata, read_y4m_and_split, create_x264_encoder
from dtcwt_processor import DTCWT3DProcessor
from evaluate_metrics import evaluate_video_quality
from run_rd_curve import (
    run_baseline_encoding, run_proposed_encoding, run_dwt3d_encoding,
    calculate_bd_rate, calculate_bd_psnr, _safe,
)


# ============================================================================
#  노이즈 주입
# ============================================================================

def create_noisy_video(input_path, output_path, sigma, seed=42):
    """Clean Y4M 영상에 Gaussian 노이즈를 추가하여 Noisy Y4M을 생성합니다.

    Y(Luma) 채널에만 노이즈를 추가합니다 (표준 디노이징 실험 관례).
    동일한 seed를 사용하면 동일한 노이즈 패턴이 재현됩니다.

    Args:
        input_path: 원본 Y4M 파일 경로.
        output_path: 출력 Y4M 파일 경로.
        sigma: Gaussian 노이즈 표준편차 (0~255 스케일).
        seed: 재현성을 위한 랜덤 시드.

    Returns:
        output_path (생성된 노이즈 영상 경로).
    """
    rng = np.random.default_rng(seed)

    w, h, fps = get_video_metadata(input_path)

    # FPS를 분수 문자열로 가져오기 (Y4M 헤더 호환)
    probe = ffmpeg.probe(input_path)
    video_stream = next(s for s in probe['streams'] if s['codec_type'] == 'video')
    fps_str = video_stream.get('r_frame_rate', '30/1')

    y_size = w * h
    uv_size = y_size // 4
    frame_size = y_size + uv_size * 2

    # 전체 rawvideo 읽기
    out, _ = (
        ffmpeg.input(input_path)
        .output('pipe:', format='rawvideo', pix_fmt='yuv420p')
        .global_args('-loglevel', 'quiet')
        .run(capture_stdout=True)
    )

    num_frames = len(out) // frame_size
    raw = np.frombuffer(out, dtype=np.uint8).copy().reshape(num_frames, frame_size)

    # Y 채널에 Gaussian 노이즈 추가
    for i in range(num_frames):
        y = raw[i, :y_size].astype(np.float32)
        noise = rng.normal(0, sigma, y.shape).astype(np.float32)
        raw[i, :y_size] = np.clip(y + noise, 0, 255).astype(np.uint8)

    # FFmpeg로 Y4M 파일 쓰기
    process = (
        ffmpeg.input('pipe:', format='rawvideo', pix_fmt='yuv420p',
                     s=f'{w}x{h}', framerate=fps_str)
        .output(output_path, pix_fmt='yuv420p', format='yuv4mpegpipe')
        .overwrite_output()
        .global_args('-loglevel', 'quiet')
        .run_async(pipe_stdin=True)
    )
    process.stdin.write(raw.tobytes())
    process.stdin.close()
    process.wait()

    print(f"  ✅ 노이즈 영상 생성 완료: {output_path} (σ={sigma}, {num_frames} frames)")
    return output_path


# ============================================================================
#  단일 조건 실험
# ============================================================================

METRIC_NAMES = ['psnr', 'ssim', 'vmaf', 'msssim', 'epsnr', 'psnrb', 'gbim', 'mepr']


def run_single_condition(video_name, clean_video, input_video, sigma,
                         output_dir, bitrates, threshold):
    """단일 노이즈 조건에서 Baseline/Proposed/DWT 실험을 수행합니다.

    Args:
        video_name: 비디오 이름 (확장자 제외).
        clean_video: 원본 clean 비디오 경로 (품질 평가의 참조 영상).
        input_video: 인코딩 대상 비디오 경로 (clean 또는 noisy).
        sigma: 노이즈 레벨 (표시용).
        output_dir: 출력 디렉토리.
        bitrates: 비트레이트 목록 (kbps 정수).
        threshold: DT-CWT 임계값.

    Returns:
        결과 딕셔너리.
    """
    condition = "Clean" if sigma == 0 else f"σ={sigma}"
    print(f"\n{'=' * 60}")
    print(f"  🎬 {video_name.upper()} | 조건: {condition}")
    print(f"{'=' * 60}")

    results = {
        'video': video_name, 'sigma': sigma, 'bitrates': bitrates,
        'base': {m: [] for m in METRIC_NAMES},
        'prop': {m: [] for m in METRIC_NAMES},
        'dwt':  {m: [] for m in METRIC_NAMES},
    }

    for br in bitrates:
        br_str = f"{br}k"
        prefix = f"{video_name}_s{sigma}"

        base_out = os.path.join(output_dir, f"{prefix}_base_{br_str}.mp4")
        prop_out = os.path.join(output_dir, f"{prefix}_prop_{br_str}.mp4")
        dwt_out  = os.path.join(output_dir, f"{prefix}_dwt3d_{br_str}.mp4")

        # --- 인코딩 ---
        run_baseline_encoding(input_video, base_out, br_str)
        run_dwt3d_encoding(input_video, dwt_out, br_str, threshold)
        run_proposed_encoding(input_video, prop_out, br_str, threshold)

        # --- 품질 평가 (참조 = 항상 clean 원본!) ---
        print(f"  [평가] {condition} - {br_str} (참조: clean 원본)")
        b_metrics = evaluate_video_quality(clean_video, base_out, num_frames_custom=60)
        d_metrics = evaluate_video_quality(clean_video, dwt_out, num_frames_custom=60)
        p_metrics = evaluate_video_quality(clean_video, prop_out, num_frames_custom=60)

        for i, name in enumerate(METRIC_NAMES):
            results['base'][name].append(b_metrics[i])
            results['prop'][name].append(p_metrics[i])
            results['dwt'][name].append(d_metrics[i])

        b_p, p_p, d_p = b_metrics[0], p_metrics[0], d_metrics[0]
        b_v, p_v, d_v = b_metrics[2], p_metrics[2], d_metrics[2]
        print(f"    PSNR: B({_fmt(b_p)}) vs DWT({_fmt(d_p)}) vs DT({_fmt(p_p)}) "
              f"| VMAF: B({_fmt(b_v)}) vs DWT({_fmt(d_v)}) vs DT({_fmt(p_v)})")

    return results


def _fmt(v):
    """숫자를 안전하게 포맷합니다 (nan 처리)."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    return f"{v:.2f}"


# ============================================================================
#  결과 요약 및 저장
# ============================================================================

def compute_condition_summary(results):
    """단일 조건의 BD-Rate/BD-PSNR을 계산합니다."""
    bitrates = results['bitrates']

    base_psnrs = _safe(results['base']['psnr'])
    prop_psnrs = _safe(results['prop']['psnr'])
    dwt_psnrs  = _safe(results['dwt']['psnr'])

    base_vmafs = _safe(results['base']['vmaf'])
    prop_vmafs = _safe(results['prop']['vmaf'])
    dwt_vmafs  = _safe(results['dwt']['vmaf'])

    return {
        'bd_rate_psnr_prop': calculate_bd_rate(bitrates, base_psnrs, bitrates, prop_psnrs),
        'bd_rate_vmaf_prop': calculate_bd_rate(bitrates, base_vmafs, bitrates, prop_vmafs),
        'bd_rate_psnr_dwt':  calculate_bd_rate(bitrates, base_psnrs, bitrates, dwt_psnrs),
        'bd_rate_vmaf_dwt':  calculate_bd_rate(bitrates, base_vmafs, bitrates, dwt_vmafs),
        'bd_psnr_prop': calculate_bd_psnr(bitrates, base_psnrs, bitrates, prop_psnrs),
        'bd_psnr_dwt':  calculate_bd_psnr(bitrates, base_psnrs, bitrates, dwt_psnrs),
    }


def save_condition_csv(results, output_dir):
    """단일 조건의 raw data를 CSV로 저장합니다."""
    video = results['video']
    sigma = results['sigma']
    csv_path = os.path.join(output_dir, f"raw_data_{video}_s{sigma}.csv")

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        header = ['Bitrate(kbps)']
        for metric in METRIC_NAMES:
            header.extend([f'Base_{metric.upper()}', f'DWT_{metric.upper()}',
                           f'Prop_{metric.upper()}'])
        writer.writerow(header)

        for i, br in enumerate(results['bitrates']):
            row = [br]
            for metric in METRIC_NAMES:
                row.extend([
                    results['base'][metric][i],
                    results['dwt'][metric][i],
                    results['prop'][metric][i],
                ])
            writer.writerow(row)

    print(f"  💾 CSV 저장: {csv_path}")


# ============================================================================
#  시각화
# ============================================================================

def plot_condition_rd_curves(results, summary, output_dir):
    """단일 조건의 PSNR/VMAF RD Curve를 생성합니다."""
    video = results['video']
    sigma = results['sigma']
    bitrates = results['bitrates']
    condition_label = "Clean" if sigma == 0 else f"Noisy (σ={sigma})"

    for metric_key, ylabel, metric_name in [
        ('psnr', 'PSNR (dB)', 'PSNR'),
        ('vmaf', 'VMAF Score', 'VMAF'),
    ]:
        bd_key = f'bd_rate_{metric_key}_prop'
        bd_val = summary[bd_key]
        bd_str = f"{bd_val:.2f}%" if not np.isnan(bd_val) else "N/A"

        plt.figure(figsize=(8, 6))
        plt.plot(bitrates, _safe(results['base'][metric_key]),
                 marker='o', linestyle='-', label='Baseline (x264)', color='#4c72b0')
        plt.plot(bitrates, _safe(results['dwt'][metric_key]),
                 marker='s', linestyle='-.', label='3D DWT + x264', color='#dd8452')
        plt.plot(bitrates, _safe(results['prop'][metric_key]),
                 marker='D', linestyle='-', label='3D DT-CWT + x264', color='#c44e52',
                 linewidth=2)

        plt.title(f"{metric_name} RD Curve | {video.upper()} [{condition_label}]\n"
                  f"BD-Rate (Proposed vs Baseline): {bd_str}", fontsize=13)
        plt.xlabel("Bitrate (kbps)", fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=11)
        plt.tight_layout()

        filename = os.path.join(output_dir, f"rd_{metric_key}_{video}_s{sigma}.png")
        plt.savefig(filename, dpi=300)
        plt.close()
    print(f"  📊 RD Curve 저장 완료 (σ={sigma})")


def plot_overlay_rd_curves(all_results, output_dir):
    """같은 비디오의 모든 노이즈 조건을 하나의 RD Curve에 겹쳐 그립니다.

    이 차트가 핵심 결과입니다: 노이즈가 커질수록 Proposed가 Baseline보다
    유리해지는 경향을 한눈에 볼 수 있습니다.
    """
    sigma_colors = {0: '#4c72b0', 5: '#55a868', 10: '#dd8452', 15: '#c44e52'}
    sigma_labels = {0: 'Clean', 5: 'σ=5', 10: 'σ=10', 15: 'σ=15'}

    for video in all_results:
        for metric_key, ylabel, metric_name in [
            ('psnr', 'PSNR (dB)', 'PSNR'),
            ('vmaf', 'VMAF Score', 'VMAF'),
        ]:
            plt.figure(figsize=(10, 7))

            for sigma in sorted(all_results[video].keys()):
                results = all_results[video][sigma]
                bitrates = results['bitrates']
                color = sigma_colors.get(sigma, '#8c8c8c')
                label = sigma_labels.get(sigma, f'σ={sigma}')

                # Baseline: 점선, Proposed: 실선
                plt.plot(bitrates, _safe(results['base'][metric_key]),
                         marker='o', linestyle='--', color=color, alpha=0.5,
                         label=f'{label} Baseline', markersize=5)
                plt.plot(bitrates, _safe(results['prop'][metric_key]),
                         marker='D', linestyle='-', color=color, linewidth=2,
                         label=f'{label} Proposed', markersize=6)

            plt.title(f"{metric_name} RD Curves | {video.upper()} | All Noise Conditions\n"
                      f"(실선=Proposed, 점선=Baseline)", fontsize=13)
            plt.xlabel("Bitrate (kbps)", fontsize=12)
            plt.ylabel(ylabel, fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend(fontsize=9, ncol=2, loc='lower right')
            plt.tight_layout()

            filename = os.path.join(output_dir, f"rd_overlay_{metric_key}_{video}.png")
            plt.savefig(filename, dpi=300)
            plt.close()

        print(f"  📊 오버레이 RD Curve 저장 완료: {video}")


def plot_bd_rate_comparison(all_summaries, output_dir):
    """조건별 BD-Rate 비교: 바 차트 + 히트맵.

    핵심 가설 검증 차트:
    - Clean에서는 BD-Rate > 0 (Proposed 불리)
    - Noisy에서는 BD-Rate < 0 (Proposed 유리) → 가설 H1 입증
    """
    videos = list(all_summaries.keys())
    sigmas = sorted(set(s for v in all_summaries.values() for s in v.keys()))

    # ---- 1. BD-Rate 바 차트 ----
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for metric_idx, (metric_key, title) in enumerate([
        ('bd_rate_psnr_prop', 'BD-Rate (PSNR) — Proposed vs Baseline'),
        ('bd_rate_vmaf_prop', 'BD-Rate (VMAF) — Proposed vs Baseline'),
    ]):
        ax = axes[metric_idx]
        x = np.arange(len(sigmas))
        width = 0.8 / max(len(videos), 1)

        for i, video in enumerate(videos):
            values = []
            for s in sigmas:
                val = all_summaries[video].get(s, {}).get(metric_key, float('nan'))
                values.append(val)
            offset = (i - len(videos) / 2 + 0.5) * width
            bars = ax.bar(x + offset, values, width, label=video.capitalize())

            # 값 표시
            for j, v in enumerate(values):
                if not np.isnan(v):
                    ax.text(x[j] + offset, v + (0.3 if v >= 0 else -0.3),
                            f"{v:.1f}", ha='center',
                            va='bottom' if v >= 0 else 'top', fontsize=8)

        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.8, linewidth=1)
        ax.set_xlabel('Noise Level', fontsize=12)
        ax.set_ylabel('BD-Rate (%)\n(< 0 = Proposed 우수)', fontsize=11)
        ax.set_title(title, fontsize=13)
        ax.set_xticks(x)
        ax.set_xticklabels([f"σ={s}" if s > 0 else "Clean" for s in sigmas])
        ax.legend(fontsize=9)
        ax.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cross_condition_bd_rate.png"), dpi=300)
    plt.close()
    print(f"  📊 BD-Rate 바 차트 저장 완료")

    # ---- 2. BD-Rate 히트맵 ----
    fig, axes = plt.subplots(1, 2, figsize=(14, max(3, 1 + len(videos))))

    for idx, (metric_key, title, cmap) in enumerate([
        ('bd_rate_psnr_prop', 'BD-Rate PSNR (%) — Proposed vs Baseline', 'RdYlGn_r'),
        ('bd_rate_vmaf_prop', 'BD-Rate VMAF (%) — Proposed vs Baseline', 'RdYlGn_r'),
    ]):
        ax = axes[idx]
        data = []
        for video in videos:
            row = [all_summaries[video].get(s, {}).get(metric_key, float('nan'))
                   for s in sigmas]
            data.append(row)

        data_arr = np.array(data)

        # 0 중심으로 대칭 범위 설정 (BD-Rate: 음수=좋음, 양수=나쁨)
        vmax = np.nanmax(np.abs(data_arr)) if not np.all(np.isnan(data_arr)) else 10
        im = ax.imshow(data_arr, cmap=cmap, aspect='auto', vmin=-vmax, vmax=vmax)

        ax.set_xticks(range(len(sigmas)))
        ax.set_xticklabels([f"σ={s}" if s > 0 else "Clean" for s in sigmas])
        ax.set_yticks(range(len(videos)))
        ax.set_yticklabels([v.capitalize() for v in videos])
        ax.set_title(title, fontsize=12)

        # 셀 안에 값 표시
        for i in range(len(videos)):
            for j in range(len(sigmas)):
                val = data_arr[i, j]
                if not np.isnan(val):
                    text_color = 'white' if abs(val) > vmax * 0.5 else 'black'
                    ax.text(j, i, f"{val:.1f}%", ha='center', va='center',
                            fontsize=11, fontweight='bold', color=text_color)

        plt.colorbar(im, ax=ax, label='BD-Rate (%)', shrink=0.8)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "bd_rate_heatmap.png"), dpi=300)
    plt.close()
    print(f"  📊 BD-Rate 히트맵 저장 완료")


def plot_delta_psnr_trend(all_results, output_dir):
    """각 비트레이트에서 (Proposed PSNR - Baseline PSNR)의 Δ를 σ별로 추적합니다.

    이 차트는 "어떤 비트레이트와 노이즈 레벨에서 Proposed가 이기는가"를
    직관적으로 보여줍니다.
    """
    for video in all_results:
        sigmas = sorted(all_results[video].keys())
        bitrates = all_results[video][sigmas[0]]['bitrates']

        plt.figure(figsize=(10, 6))

        for sigma in sigmas:
            results = all_results[video][sigma]
            base_psnrs = _safe(results['base']['psnr'])
            prop_psnrs = _safe(results['prop']['psnr'])

            deltas = [p - b for p, b in zip(prop_psnrs, base_psnrs)]
            label = "Clean" if sigma == 0 else f"σ={sigma}"
            plt.plot(bitrates, deltas, marker='o', linewidth=2, label=label)

        plt.axhline(y=0, color='gray', linestyle='-', alpha=0.8, linewidth=1)

        # Y축 범위를 데이터 기반으로 설정 후 배경 영역 표시
        ax = plt.gca()
        ax.set_xlim(min(bitrates) - 10, max(bitrates) + 10)
        # 데이터에서 Y 범위 추정
        all_deltas = []
        for sigma in sigmas:
            results = all_results[video][sigma]
            bp = _safe(results['base']['psnr'])
            pp = _safe(results['prop']['psnr'])
            all_deltas.extend([p - b for p, b in zip(pp, bp)])
        if all_deltas:
            y_margin = max(abs(min(all_deltas)), abs(max(all_deltas))) * 1.3 + 0.5
            ax.set_ylim(-y_margin, y_margin)

        # 양수(Proposed 우수) / 음수(Baseline 우수) 배경
        ax.axhspan(0, y_margin if all_deltas else 1, alpha=0.05, color='green')
        ax.axhspan(-y_margin if all_deltas else -1, 0, alpha=0.05, color='red')

        plt.text(bitrates[-1], 0.05 * (y_margin if all_deltas else 1),
                 '← Proposed 우수', ha='right', fontsize=10,
                 color='green', alpha=0.7, va='bottom')
        plt.text(bitrates[-1], -0.05 * (y_margin if all_deltas else 1),
                 '← Baseline 우수', ha='right', fontsize=10,
                 color='red', alpha=0.7, va='top')

        plt.title(f"ΔPSNR (Proposed - Baseline) | {video.upper()}\n"
                  f"양수 = Proposed가 더 좋음", fontsize=13)
        plt.xlabel("Bitrate (kbps)", fontsize=12)
        plt.ylabel("ΔPSNR (dB)", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=11)
        plt.tight_layout()

        filename = os.path.join(output_dir, f"delta_psnr_{video}.png")
        plt.savefig(filename, dpi=300)
        plt.close()

    print(f"  📊 ΔPSNR 트렌드 차트 저장 완료")


def save_summary_csv(all_summaries, output_dir):
    """전체 실험 BD-Rate 요약을 CSV로 저장합니다."""
    csv_path = os.path.join(output_dir, "summary_bd_rates.csv")

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Video', 'Sigma',
            'BD-Rate_PSNR_Prop(%)', 'BD-Rate_VMAF_Prop(%)',
            'BD-Rate_PSNR_DWT(%)', 'BD-Rate_VMAF_DWT(%)',
            'BD-PSNR_Prop(dB)', 'BD-PSNR_DWT(dB)',
        ])
        for video in all_summaries:
            for sigma in sorted(all_summaries[video].keys()):
                s = all_summaries[video][sigma]
                writer.writerow([
                    video, sigma,
                    f"{s['bd_rate_psnr_prop']:.3f}",
                    f"{s['bd_rate_vmaf_prop']:.3f}",
                    f"{s['bd_rate_psnr_dwt']:.3f}",
                    f"{s['bd_rate_vmaf_dwt']:.3f}",
                    f"{s['bd_psnr_prop']:.3f}",
                    f"{s['bd_psnr_dwt']:.3f}",
                ])

    print(f"\n  💾 요약 CSV 저장: {csv_path}")
    return csv_path


def print_summary_table(all_summaries):
    """콘솔에 BD-Rate 요약 테이블을 출력합니다."""
    print(f"\n{'=' * 80}")
    print(f"  📊 전체 조건별 BD-Rate 요약 (Proposed vs Baseline)")
    print(f"  (음수 = Proposed 우수 | 양수 = Baseline 우수)")
    print(f"{'=' * 80}")
    print(f"  {'Video':<12} {'σ':>4}  {'BD-Rate(PSNR)':>16}  {'BD-Rate(VMAF)':>16}  {'BD-PSNR':>10}")
    print(f"  {'-' * 12} {'-' * 4}  {'-' * 16}  {'-' * 16}  {'-' * 10}")

    for video in all_summaries:
        for sigma in sorted(all_summaries[video].keys()):
            s = all_summaries[video][sigma]
            bd_psnr = s['bd_rate_psnr_prop']
            bd_vmaf = s['bd_rate_vmaf_prop']
            bd_p = s['bd_psnr_prop']

            # 색상 힌트 (더 좋으면 ✅, 더 나쁘면 ❌)
            emoji_p = "✅" if bd_psnr < 0 else "❌" if bd_psnr > 0 else "➖"
            emoji_v = "✅" if bd_vmaf < 0 else "❌" if bd_vmaf > 0 else "➖"

            sigma_label = "clean" if sigma == 0 else f"σ={sigma}"
            print(f"  {video:<12} {sigma_label:>4}  "
                  f"{emoji_p} {bd_psnr:>+10.2f} %    "
                  f"{emoji_v} {bd_vmaf:>+10.2f} %    "
                  f"{bd_p:>+7.3f} dB")
        print()

    print(f"{'=' * 80}")


# ============================================================================
#  메인 실험 루프
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Clean vs Noisy 조건 분리 RD 실험",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""실험 예시:
  기본 실행 (akiyo, foreman / σ=0,5,10,15 / 100~500k):
    uv run python run_noise_experiment.py

  특정 비디오만 빠르게 테스트:
    uv run python run_noise_experiment.py -v akiyo --sigma 0 10 -b 100 300 500

  전체 비디오에 대해 전체 실험:
    uv run python run_noise_experiment.py -v akiyo foreman mobile stefan
""",
    )
    parser.add_argument("-v", "--video_names", nargs='+',
                        default=["akiyo", "foreman"],
                        help="실험할 비디오 이름 목록 (기본: akiyo foreman)")
    parser.add_argument("-i", "--input_dir", default="./videos",
                        help="입력 비디오 디렉토리 (기본: ./videos)")
    parser.add_argument("-o", "--output_dir", default="./outputs/noise_experiment",
                        help="출력 디렉토리 (기본: ./outputs/noise_experiment)")
    parser.add_argument("-b", "--bitrates", nargs='+', type=int,
                        default=[100, 200, 300, 400, 500],
                        help="테스트할 비트레이트 목록 kbps (기본: 100 200 300 400 500)")
    parser.add_argument("--sigma", nargs='+', type=int,
                        default=[0, 5, 10, 15],
                        help="Gaussian 노이즈 σ 목록 (0=clean) (기본: 0 5 10 15)")
    parser.add_argument("-t", "--threshold", type=float, default=0.03,
                        help="DT-CWT 임계값 (기본: 0.03)")
    parser.add_argument("--seed", type=int, default=42,
                        help="노이즈 랜덤 시드 (기본: 42)")
    parser.add_argument("--skip_existing", action="store_true",
                        help="이미 인코딩된 파일이 있으면 건너뜀")

    args = parser.parse_args()

    # 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)
    noisy_dir = os.path.join(args.output_dir, "noisy_inputs")
    os.makedirs(noisy_dir, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"  🧪 Clean vs Noisy 조건 분리 실험")
    print(f"  비디오: {args.video_names}")
    print(f"  노이즈: σ = {args.sigma}")
    print(f"  비트레이트: {args.bitrates} kbps")
    print(f"  임계값: {args.threshold}")
    print(f"{'=' * 60}")

    all_results = {}
    all_summaries = {}

    for video_name in args.video_names:
        clean_video = os.path.join(args.input_dir, f"{video_name}.y4m")
        if not os.path.exists(clean_video):
            print(f"  ⚠️ {clean_video} 파일을 찾을 수 없습니다. 건너뜁니다.")
            continue

        all_results[video_name] = {}
        all_summaries[video_name] = {}

        for sigma in args.sigma:
            # 1. 노이즈 영상 생성 (σ=0이면 원본 그대로 사용)
            if sigma == 0:
                input_video = clean_video
            else:
                noisy_path = os.path.join(noisy_dir, f"{video_name}_s{sigma}.y4m")
                if os.path.exists(noisy_path) and args.skip_existing:
                    print(f"  ♻️ 기존 노이즈 영상 재사용: {noisy_path}")
                    input_video = noisy_path
                else:
                    # 비디오+시그마별 고유 시드로 재현성 확보
                    noise_seed = args.seed + sigma * 1000 + sum(ord(c) for c in video_name)
                    input_video = create_noisy_video(
                        clean_video, noisy_path, sigma, seed=noise_seed
                    )

            # 2. 실험 수행
            results = run_single_condition(
                video_name, clean_video, input_video, sigma,
                args.output_dir, args.bitrates, args.threshold
            )

            # 3. 결과 처리
            summary = compute_condition_summary(results)
            save_condition_csv(results, args.output_dir)
            plot_condition_rd_curves(results, summary, args.output_dir)

            all_results[video_name][sigma] = results
            all_summaries[video_name][sigma] = summary

            # BD-Rate 출력
            bd_p = summary['bd_rate_psnr_prop']
            bd_v = summary['bd_rate_vmaf_prop']
            bd_p_str = f"{bd_p:+.2f}%" if not np.isnan(bd_p) else "N/A"
            bd_v_str = f"{bd_v:+.2f}%" if not np.isnan(bd_v) else "N/A"
            print(f"\n  📊 [{video_name.upper()} σ={sigma}] "
                  f"BD-Rate: PSNR={bd_p_str} | VMAF={bd_v_str}")

    # 4. 전체 비교 시각화
    if all_results:
        print(f"\n{'=' * 60}")
        print(f"  📊 전체 비교 시각화 생성 중...")
        print(f"{'=' * 60}")

        plot_overlay_rd_curves(all_results, args.output_dir)
        plot_bd_rate_comparison(all_summaries, args.output_dir)
        plot_delta_psnr_trend(all_results, args.output_dir)
        csv_path = save_summary_csv(all_summaries, args.output_dir)

        # 콘솔 요약 테이블
        print_summary_table(all_summaries)

    print(f"\n{'=' * 60}")
    print(f"  ✅ 전체 실험 완료!")
    print(f"  📁 결과: {os.path.abspath(args.output_dir)}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
