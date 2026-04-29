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

# Windows cp949 터미널에서 이모지/한글 출력 시 UnicodeEncodeError 방지
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ('utf-8', 'utf8'):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 한글 폰트 설정 (Windows: Malgun Gothic)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

import ffmpeg

from dtcwt_video.pipeline import get_video_metadata, read_y4m_and_split, create_x264_encoder
from dtcwt_video.dtcwt_processor import DTCWT3DProcessor
from dtcwt_video.evaluate_metrics import evaluate_video_quality
from dtcwt_video.encoders import (
    run_baseline_encoding, run_proposed_encoding, run_dwt3d_encoding,
    run_nr_encoding, run_hqdn3d_encoding, run_spatial_encoding,
    run_lossless_copy, run_proposed_preprocess, run_dwt3d_preprocess,
    run_hqdn3d_preprocess, run_spatial_preprocess,
    calculate_bd_rate, calculate_bd_psnr, _safe,
)
from dtcwt_video.experiment_analysis import (
    RELIABLE_PRIMARY_METRICS,
    get_actual_bitrate_kbps, summarize_method_against_baseline,
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
PLOT_METRICS = [
    ('psnr', 'PSNR-Y (dB)', 'PSNR-Y'),
    ('msssim', 'MS-SSIM', 'MS-SSIM'),
    ('psnrb', 'PSNR-B (dB)', 'PSNR-B'),
    ('epsnr', 'Edge PSNR (dB)', 'EPSNR'),
]

# 사용 가능한 비교군 키 → (표시명, 인코더 함수 or None(=Proposed))
ALL_BASELINES = {
    'base':   ('Baseline (x264)',        run_baseline_encoding),
    'nr':     ('x264 --nr',              run_nr_encoding),
    'hqdn3d': ('FFmpeg hqdn3d',          run_hqdn3d_encoding),
    'dwt':    ('3D DWT + x264',          run_dwt3d_encoding),
    'gaussian': ('Gaussian blur + x264', run_spatial_encoding),
    'prop':   ('3D DT-CWT (Proposed)',   None),   # 항상 포함
}
DEFAULT_BASELINES = ['base', 'nr', 'hqdn3d', 'dwt', 'gaussian']   # prop은 항상 포함
PRECODEC_METHODS = ['noisy', 'hqdn3d', 'dwt', 'gaussian', 'prop']


def run_single_condition(video_name, clean_video, input_video, sigma,
                         output_dir, bitrates, threshold, baselines=None,
                         include_precodec_ablation: bool = True,
                         reuse_preprocessed: bool = True,
                         skip_existing_outputs: bool = False,
                         chunk_size: int = 16,
                         overlap: int = 4,
                         process_chroma: bool = False,
                         seed: int | None = None,
                         threshold_mode: str = "adaptive",
                         controller_a: float = 0.35,
                         controller_b: float = 0.25,
                         controller_c: float = 0.25,
                         controller_d: float = 0.25,
                         min_multiplier: float = 0.5,
                         max_multiplier: float = 2.5,
                         disable_rate_aware_scene_reset: bool = False,
                         log_context: bool = False):
    """단일 노이즈 조건에서 여러 비교군과 Proposed의 실험을 수행합니다.

    Args:
        video_name: 비디오 이름 (확장자 제외).
        clean_video: 원본 clean 비디오 경로 (품질 평가의 참조 영상).
        input_video: 인코딩 대상 비디오 경로 (clean 또는 noisy).
        sigma: 노이즈 레벨 (표시용).
        output_dir: 출력 디렉토리.
        bitrates: 비트레이트 목록 (kbps 정수).
        threshold: DT-CWT 임계값.
        baselines: 포함할 비교군 키 목록. None이면 DEFAULT_BASELINES 사용.
                   사용 가능: 'base', 'nr', 'hqdn3d', 'dwt'  (prop은 항상 포함)

    Returns:
        결과 딕셔너리.
    """
    if baselines is None:
        baselines = DEFAULT_BASELINES
    # base와 prop은 항상 포함, 중복 제거
    active_methods = list(dict.fromkeys(['base'] + baselines + ['prop']))

    condition = "Clean" if sigma == 0 else f"σ={sigma}"
    print(f"\n{'=' * 60}")
    print(f"  🎬 {video_name.upper()} | 조건: {condition}")
    print(f"  비교군: {[ALL_BASELINES[m][0] for m in active_methods]}")
    print(f"{'=' * 60}")

    results = {
        'video': video_name, 'sigma': sigma, 'bitrates': bitrates,
        'active_methods': active_methods,
        'actual_bitrates': {method_key: [] for method_key in active_methods},
        'rows': [],
        'pre_metrics': {},
        'preprocessed_paths': {},
        'artifacts': [],
    }
    for method_key in active_methods:
        results[method_key] = {metric: [] for metric in METRIC_NAMES}

    if include_precodec_ablation:
        pre_metrics, pre_rows, preprocessed_paths = run_precodec_ablation(
            video_name, clean_video, input_video, sigma, output_dir,
            threshold, seed,
            skip_existing_outputs=skip_existing_outputs,
            chunk_size=chunk_size,
            overlap=overlap,
            process_chroma=process_chroma,
            threshold_mode=threshold_mode,
            controller_a=controller_a,
            controller_b=controller_b,
            controller_c=controller_c,
            controller_d=controller_d,
            min_multiplier=min_multiplier,
            max_multiplier=max_multiplier,
            disable_rate_aware_scene_reset=disable_rate_aware_scene_reset,
        )
        results['pre_metrics'] = pre_metrics
        results['preprocessed_paths'] = preprocessed_paths
        results['artifacts'].extend(preprocessed_paths.values())
        results['rows'].extend(pre_rows)

    preprocessed_inputs = _select_reusable_preprocessed_inputs(
        results.get('preprocessed_paths', {}),
        threshold_mode=threshold_mode,
        reuse_preprocessed=reuse_preprocessed,
    )

    for br in bitrates:
        br_str = f"{br}k"
        prefix = f"{video_name}_s{sigma}"

        # 출력 파일 경로 결정
        output_paths = {
            method_key: os.path.join(output_dir, f"{prefix}_{method_key}_{br_str}.mp4")
            for method_key in active_methods
        }
        results['artifacts'].extend(output_paths.values())

        # --- 인코딩 ---
        for method_key in active_methods:
            out_path = output_paths[method_key]
            if skip_existing_outputs and os.path.exists(out_path):
                print(f"  [Skip]    기존 인코딩 재사용: {out_path}")
            elif method_key in preprocessed_inputs:
                run_baseline_encoding(preprocessed_inputs[method_key], out_path, br_str)
            elif method_key == 'base':
                run_baseline_encoding(input_video, out_path, br_str)
            elif method_key == 'nr':
                run_nr_encoding(input_video, out_path, br_str)
            elif method_key == 'hqdn3d':
                run_hqdn3d_encoding(input_video, out_path, br_str)
            elif method_key == 'dwt':
                run_dwt3d_encoding(
                    input_video, out_path, br_str, threshold,
                    chunk_size=chunk_size, overlap=overlap,
                )
            elif method_key == 'gaussian':
                run_spatial_encoding(input_video, out_path, br_str, chunk_size=chunk_size)
            elif method_key == 'prop':
                ctx_log = None
                if log_context:
                    os.makedirs(os.path.join(output_dir, "context_logs"), exist_ok=True)
                    ctx_log = os.path.join(
                        output_dir, "context_logs",
                        f"{prefix}_{br_str}_ctx.csv"
                    )
                run_proposed_encoding(
                    input_video, out_path, br_str, threshold,
                    chunk_size=chunk_size,
                    overlap=overlap,
                    process_chroma=process_chroma,
                    threshold_mode=threshold_mode,
                    controller_a=controller_a,
                    controller_b=controller_b,
                    controller_c=controller_c,
                    controller_d=controller_d,
                    min_multiplier=min_multiplier,
                    max_multiplier=max_multiplier,
                    disable_rate_aware_scene_reset=disable_rate_aware_scene_reset,
                    log_context_path=ctx_log,
                )
            results['actual_bitrates'][method_key].append(
                get_actual_bitrate_kbps(out_path)
            )

        # --- 품질 평가 (참조 = 항상 clean 원본!) ---
        print(f"  [평가] {condition} - {br_str} (참조: clean 원본)")
        for method_key in active_methods:
            m_result = evaluate_video_quality(clean_video, output_paths[method_key], num_frames_custom=60)
            metrics = _metrics_to_dict(m_result)
            for i, name in enumerate(METRIC_NAMES):
                results[method_key][name].append(m_result[i])
            _append_result_row(
                results['rows'],
                video=video_name,
                sigma=sigma,
                seed=seed,
                stage="post_x264",
                method=method_key,
                target_bitrate_kbps=br,
                actual_bitrate_kbps=results['actual_bitrates'][method_key][-1],
                metrics=metrics,
            )

        # 콘솔 PSNR 요약
        psnr_parts = [
            f"{ALL_BASELINES[m][0].split()[0]}({_fmt(results[m]['psnr'][-1])})"
            for m in active_methods
        ]
        print(f"    PSNR: {' | '.join(psnr_parts)}")

    return results


def _select_reusable_preprocessed_inputs(preprocessed_paths: dict[str, str], *,
                                         threshold_mode: str,
                                         reuse_preprocessed: bool) -> dict[str, str]:
    """Return preprocessing artifacts that can be reused for all bitrates.

    In fixed/adaptive modes, proposed preprocessing is bitrate-independent.
    In rate-aware mode, proposed depends on target bitrate, so only non-proposed
    baseline prefilters are safely reusable.
    """
    if not reuse_preprocessed:
        return {}

    reusable = {
        method: path
        for method, path in preprocessed_paths.items()
        if method in {"hqdn3d", "dwt", "gaussian"} and os.path.exists(path)
    }
    if threshold_mode != "rate_aware":
        prop_path = preprocessed_paths.get("prop")
        if prop_path and os.path.exists(prop_path):
            reusable["prop"] = prop_path
    return reusable



def _fmt(v):
    """숫자를 안전하게 포맷합니다 (nan 처리)."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    return f"{v:.2f}"


def _metrics_to_dict(metric_tuple):
    """evaluate_video_quality 튜플을 이름 기반 dict로 변환합니다."""
    return {name: metric_tuple[i] for i, name in enumerate(METRIC_NAMES)}


def _append_result_row(rows, *, video, sigma, seed, stage, method,
                       target_bitrate_kbps, actual_bitrate_kbps, metrics):
    row = {
        "stage": stage,
        "method": method,
        "target_bitrate_kbps": target_bitrate_kbps,
        "actual_bitrate_kbps": actual_bitrate_kbps,
        "sigma": sigma,
        "video": video,
        "seed": seed,
    }
    row.update(metrics)
    rows.append(row)


def run_precodec_ablation(video_name, clean_video, input_video, sigma,
                          output_dir, threshold, seed,
                          skip_existing_outputs: bool = False,
                          chunk_size: int = 16,
                          overlap: int = 4,
                          process_chroma: bool = False,
                          threshold_mode: str = "adaptive",
                          controller_a: float = 0.35,
                          controller_b: float = 0.25,
                          controller_c: float = 0.25,
                          controller_d: float = 0.25,
                          min_multiplier: float = 0.5,
                          max_multiplier: float = 2.5,
                          disable_rate_aware_scene_reset: bool = False):
    """압축 전 전처리 자체의 denoising 이득을 측정합니다."""
    pre_dir = os.path.join(output_dir, "pre_x264")
    os.makedirs(pre_dir, exist_ok=True)

    prefix = f"{video_name}_s{sigma}"
    output_paths = {
        "noisy": os.path.join(pre_dir, f"{prefix}_noisy.y4m"),
        "hqdn3d": os.path.join(pre_dir, f"{prefix}_hqdn3d.y4m"),
        "dwt": os.path.join(pre_dir, f"{prefix}_dwt.y4m"),
        "gaussian": os.path.join(pre_dir, f"{prefix}_gaussian.y4m"),
        "prop": os.path.join(pre_dir, f"{prefix}_prop.y4m"),
    }

    _run_or_reuse(skip_existing_outputs, output_paths["noisy"],
                  run_lossless_copy, input_video, output_paths["noisy"])
    _run_or_reuse(skip_existing_outputs, output_paths["hqdn3d"],
                  run_hqdn3d_preprocess, input_video, output_paths["hqdn3d"])
    _run_or_reuse(skip_existing_outputs, output_paths["dwt"],
                  run_dwt3d_preprocess, input_video, output_paths["dwt"], threshold,
                  chunk_size=chunk_size, overlap=overlap)
    _run_or_reuse(skip_existing_outputs, output_paths["gaussian"],
                  run_spatial_preprocess, input_video, output_paths["gaussian"],
                  chunk_size=chunk_size)
    _run_or_reuse(
        skip_existing_outputs, output_paths["prop"],
        run_proposed_preprocess,
        input_video, output_paths["prop"], threshold,
        chunk_size=chunk_size,
        overlap=overlap,
        process_chroma=process_chroma,
        threshold_mode=threshold_mode,
        controller_a=controller_a,
        controller_b=controller_b,
        controller_c=controller_c,
        controller_d=controller_d,
        min_multiplier=min_multiplier,
        max_multiplier=max_multiplier,
        disable_rate_aware_scene_reset=disable_rate_aware_scene_reset,
    )

    pre_metrics = {}
    rows = []
    print(f"  [Pre-x264 평가] {video_name} σ={sigma} (참조: clean 원본)")
    for method in PRECODEC_METHODS:
        metrics = _metrics_to_dict(
            evaluate_video_quality(clean_video, output_paths[method], num_frames_custom=60)
        )
        pre_metrics[method] = metrics
        _append_result_row(
            rows,
            video=video_name,
            sigma=sigma,
            seed=seed,
            stage="pre_x264",
            method=method,
            target_bitrate_kbps="",
            actual_bitrate_kbps="",
            metrics=metrics,
        )

    return pre_metrics, rows, output_paths


def _run_or_reuse(skip_existing: bool, output_path: str, func, *args, **kwargs):
    """Run a producer unless the requested artifact already exists."""
    if skip_existing and os.path.exists(output_path):
        print(f"  [Skip]    기존 파일 재사용: {output_path}")
        return
    func(*args, **kwargs)


def cleanup_condition_intermediates(results, output_dir, extra_paths=None):
    """Remove generated video artifacts after metrics/plots are materialized."""
    output_root = os.path.abspath(output_dir)
    paths = list(results.get('artifacts', []))
    if extra_paths:
        paths.extend(extra_paths)

    removed = 0
    for path in dict.fromkeys(paths):
        if not path:
            continue
        abs_path = os.path.abspath(path)
        if not abs_path.startswith(output_root):
            continue
        if os.path.splitext(abs_path)[1].lower() not in {'.mp4', '.y4m', '.mkv'}:
            continue
        if os.path.exists(abs_path):
            os.remove(abs_path)
            removed += 1

    if removed:
        print(f"  🧹 중간 비디오 파일 정리: {removed}개 삭제")


# ============================================================================
#  결과 요약 및 저장
# ============================================================================

def compute_condition_summary(results):
    """단일 조건의 pre/post delta, codec gain, BD-Rate를 계산합니다."""
    bitrates = results['bitrates']
    active = results.get('active_methods', ['base', 'dwt', 'prop'])
    pre_metrics = results.get('pre_metrics', {})
    pre_base = pre_metrics.get('noisy')
    base_metrics = results['base']
    base_actual = results.get('actual_bitrates', {}).get('base', bitrates)

    summary = {}
    for method_key in active:
        if method_key == 'base':
            continue
        method_summary = summarize_method_against_baseline(
            bitrates,
            base_actual,
            results.get('actual_bitrates', {}).get(method_key, bitrates),
            base_metrics,
            results[method_key],
            pre_base,
            pre_metrics.get(method_key),
        )
        summary[method_key] = method_summary

    # Backward-compatible aliases for existing plotting/console code.
    prop = summary.get('prop', {})
    summary['bd_rate_psnr_prop'] = prop.get('bd_rate_psnr', float('nan'))
    summary['bd_rate_msssim_prop'] = prop.get('bd_rate_msssim', float('nan'))
    summary['bd_rate_vmaf_prop'] = prop.get('bd_rate_vmaf', float('nan'))  # supplementary
    summary['bd_psnr_prop'] = calculate_bd_psnr(
        base_actual, _safe(base_metrics.get('psnr', [])),
        results.get('actual_bitrates', {}).get('prop', bitrates),
        _safe(results.get('prop', {}).get('psnr', [])),
    ) if 'prop' in active else float('nan')
    for method_key in ['dwt', 'nr', 'hqdn3d', 'gaussian']:
        method_summary = summary.get(method_key, {})
        summary[f'bd_rate_psnr_{method_key}'] = method_summary.get('bd_rate_psnr', float('nan'))
        summary[f'bd_rate_msssim_{method_key}'] = method_summary.get('bd_rate_msssim', float('nan'))
        summary[f'bd_rate_vmaf_{method_key}'] = method_summary.get('bd_rate_vmaf', float('nan'))  # supplementary
    return summary


def save_condition_csv(results, output_dir):
    """단일 조건의 pre/post long-form raw data를 CSV로 저장합니다."""
    video = results['video']
    sigma = results['sigma']
    csv_path = os.path.join(output_dir, f"raw_data_{video}_s{sigma}.csv")
    fieldnames = [
        "stage", "method", "target_bitrate_kbps", "actual_bitrate_kbps",
        "sigma", "video", "seed", *METRIC_NAMES,
    ]

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results.get('rows', []):
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    print(f"  💾 CSV 저장: {csv_path}")



# ============================================================================
#  시각화
# ============================================================================

def plot_condition_rd_curves(results, summary, output_dir):
    """단일 조건의 신뢰도 높은 full-reference RD Curve를 생성합니다."""
    video = results['video']
    sigma = results['sigma']
    active = results.get('active_methods', ['base', 'dwt', 'prop'])
    condition_label = "Clean" if sigma == 0 else f"Noisy (σ={sigma})"
    colors = {
        'base': '#4c72b0', 'nr': '#8172b3', 'hqdn3d': '#55a868',
        'dwt': '#dd8452', 'gaussian': '#937860', 'prop': '#c44e52',
    }
    markers = {
        'base': 'o', 'nr': 'v', 'hqdn3d': 'P',
        'dwt': 's', 'gaussian': '^', 'prop': 'D',
    }

    for metric_key, ylabel, metric_name in PLOT_METRICS:
        bd_key = f'bd_rate_{metric_key}_prop'
        bd_val = summary.get(bd_key, float('nan'))
        bd_str = f"{bd_val:.2f}%" if not np.isnan(bd_val) else "N/A"

        plt.figure(figsize=(8, 6))
        for method_key in active:
            x_values = results.get('actual_bitrates', {}).get(method_key, results['bitrates'])
            label = ALL_BASELINES[method_key][0]
            linewidth = 2.2 if method_key == 'prop' else 1.4
            linestyle = '--' if method_key == 'base' else '-'
            plt.plot(x_values, _safe(results[method_key][metric_key]),
                     marker=markers.get(method_key, 'o'), linestyle=linestyle,
                     label=label, color=colors.get(method_key),
                     linewidth=linewidth)

        plt.title(f"{metric_name} RD Curve | {video.upper()} [{condition_label}]\n"
                  f"BD-Rate (Proposed vs Baseline): {bd_str}", fontsize=13)
        plt.xlabel("Actual Bitrate (kbps)", fontsize=12)
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
            ('psnr', 'PSNR-Y (dB)', 'PSNR-Y'),
            ('msssim', 'MS-SSIM', 'MS-SSIM'),
        ]:
            plt.figure(figsize=(10, 7))

            for sigma in sorted(all_results[video].keys()):
                results = all_results[video][sigma]
                color = sigma_colors.get(sigma, '#8c8c8c')
                label = sigma_labels.get(sigma, f'σ={sigma}')

                # Baseline: 점선, Proposed: 실선
                plt.plot(results.get('actual_bitrates', {}).get('base', results['bitrates']),
                         _safe(results['base'][metric_key]),
                         marker='o', linestyle='--', color=color, alpha=0.5,
                         label=f'{label} Baseline', markersize=5)
                plt.plot(results.get('actual_bitrates', {}).get('prop', results['bitrates']),
                         _safe(results['prop'][metric_key]),
                         marker='D', linestyle='-', color=color, linewidth=2,
                         label=f'{label} Proposed', markersize=6)

            plt.title(f"{metric_name} RD Curves | {video.upper()} | All Noise Conditions\n"
                      f"(실선=Proposed, 점선=Baseline)", fontsize=13)
            plt.xlabel("Actual Bitrate (kbps)", fontsize=12)
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
        ('bd_rate_msssim_prop', 'BD-Rate (MS-SSIM) — Proposed vs Baseline'),
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
        ('bd_rate_msssim_prop', 'BD-Rate MS-SSIM (%) — Proposed vs Baseline', 'RdYlGn_r'),
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


def plot_pre_post_delta_bars(all_summaries, output_dir):
    """Proposed의 pre/post ΔPSNR을 조건별로 비교합니다."""
    labels, pre_vals, post_vals = [], [], []
    for video in all_summaries:
        for sigma in sorted(all_summaries[video].keys()):
            prop = all_summaries[video][sigma].get('prop', {})
            labels.append(f"{video}\n{'clean' if sigma == 0 else f's{sigma}'}")
            pre_vals.append(prop.get('pre_delta_psnr', float('nan')))
            post_vals.append(prop.get('post_delta_psnr', float('nan')))

    if not labels:
        return

    x = np.arange(len(labels))
    width = 0.38
    plt.figure(figsize=(max(8, len(labels) * 0.8), 6))
    plt.bar(x - width / 2, pre_vals, width, label='Pre-x264 ΔPSNR')
    plt.bar(x + width / 2, post_vals, width, label='Post-x264 ΔPSNR')
    plt.axhline(0, color='gray', linewidth=1)
    plt.ylabel('ΔPSNR vs baseline/noisy input (dB)')
    plt.title('Pre vs Post x264 Proposed Gain')
    plt.xticks(x, labels, rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pre_post_delta_psnr.png"), dpi=300)
    plt.close()
    print(f"  📊 Pre/Post ΔPSNR 막대그래프 저장 완료")


def plot_codec_gain_heatmap(all_summaries, output_dir):
    """Codec gain = post Δ - pre Δ 히트맵을 생성합니다."""
    videos = list(all_summaries.keys())
    sigmas = sorted(set(s for v in all_summaries.values() for s in v.keys()))
    if not videos or not sigmas:
        return

    data = np.array([
        [
            all_summaries[video].get(sigma, {}).get('prop', {}).get(
                'codec_gain_psnr', float('nan')
            )
            for sigma in sigmas
        ]
        for video in videos
    ])

    vmax = np.nanmax(np.abs(data)) if not np.all(np.isnan(data)) else 1.0
    vmax = max(vmax, 0.1)
    plt.figure(figsize=(max(7, len(sigmas) * 1.4), max(3, len(videos) * 0.7 + 2)))
    im = plt.imshow(data, cmap='RdYlGn', aspect='auto', vmin=-vmax, vmax=vmax)
    plt.xticks(range(len(sigmas)), ["Clean" if s == 0 else f"σ={s}" for s in sigmas])
    plt.yticks(range(len(videos)), [v.capitalize() for v in videos])
    plt.title("Codec Gain Heatmap (Post ΔPSNR - Pre ΔPSNR)")
    for i in range(len(videos)):
        for j in range(len(sigmas)):
            val = data[i, j]
            if not np.isnan(val):
                plt.text(j, i, f"{val:+.2f}", ha='center', va='center',
                         color='white' if abs(val) > vmax * 0.45 else 'black')
    plt.colorbar(im, label='Codec gain (dB)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "codec_gain_heatmap.png"), dpi=300)
    plt.close()
    print(f"  📊 Codec gain 히트맵 저장 완료")


def save_summary_csv(all_summaries, output_dir):
    """전체 실험 요약을 method별 long-form CSV로 저장합니다."""
    csv_path = os.path.join(output_dir, "summary_bd_rates.csv")
    fieldnames = ['Video', 'Sigma', 'Method']
    for metric in [*RELIABLE_PRIMARY_METRICS, 'ssim', 'gbim', 'mepr', 'vmaf']:
        fieldnames.extend([
            f'pre_delta_{metric}', f'post_delta_{metric}', f'codec_gain_{metric}',
            f'bd_rate_{metric}', f'mean_delta_{metric}',
            f'low_bitrate_delta_{metric}', f'win_rate_{metric}',
        ])

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for video in all_summaries:
            for sigma in sorted(all_summaries[video].keys()):
                for method, values in all_summaries[video][sigma].items():
                    if not isinstance(values, dict) or not values:
                        continue
                    row = {'Video': video, 'Sigma': sigma, 'Method': method}
                    row.update(values)
                    writer.writerow({key: row.get(key, "") for key in fieldnames})

    print(f"\n  💾 요약 CSV 저장: {csv_path}")
    return csv_path


def save_reliable_metrics_csv(all_summaries, output_dir):
    """VMAF를 제외한 headline 지표만 별도 CSV로 저장합니다."""
    csv_path = os.path.join(output_dir, "summary_reliable_metrics.csv")
    metrics = [*RELIABLE_PRIMARY_METRICS, 'ssim', 'gbim', 'mepr']
    fieldnames = ['Video', 'Sigma', 'Method']
    for metric in metrics:
        fieldnames.extend([
            f'pre_delta_{metric}', f'post_delta_{metric}',
            f'codec_gain_{metric}', f'win_rate_{metric}',
        ])

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for video in all_summaries:
            for sigma in sorted(all_summaries[video].keys()):
                for method, values in all_summaries[video][sigma].items():
                    if not isinstance(values, dict) or not values:
                        continue
                    row = {'Video': video, 'Sigma': sigma, 'Method': method}
                    row.update(values)
                    writer.writerow({key: row.get(key, "") for key in fieldnames})

    print(f"  💾 신뢰 지표 요약 CSV 저장: {csv_path}")
    return csv_path


def print_summary_table(all_summaries):
    """콘솔에 BD-Rate 요약 테이블을 출력합니다."""
    print(f"\n{'=' * 80}")
    print(f"  📊 전체 조건별 요약 (Proposed vs Baseline)")
    print(f"  (BD-Rate 음수/ΔPSNR 양수 = Proposed 우수)")
    print(f"{'=' * 80}")
    print(f"  {'Video':<12} {'σ':>4}  {'BD-Rate(PSNR)':>16}  {'Post ΔPSNR':>12}  {'CodecGain':>12}  {'WinRate':>8}")
    print(f"  {'-' * 12} {'-' * 4}  {'-' * 16}  {'-' * 12}  {'-' * 12}  {'-' * 8}")

    for video in all_summaries:
        for sigma in sorted(all_summaries[video].keys()):
            s = all_summaries[video][sigma].get('prop', {})
            bd_psnr = s.get('bd_rate_psnr', float('nan'))
            post_delta = s.get('post_delta_psnr', float('nan'))
            codec_gain = s.get('codec_gain_psnr', float('nan'))
            wr = s.get('win_rate_psnr', float('nan'))

            emoji_p = "✅" if bd_psnr < 0 else "❌" if bd_psnr > 0 else "➖"

            sigma_label = "clean" if sigma == 0 else f"σ={sigma}"
            print(f"  {video:<12} {sigma_label:>4}  "
                  f"{emoji_p} {bd_psnr:>+10.2f} %    "
                  f"{post_delta:>+9.3f} dB  "
                  f"{codec_gain:>+9.3f} dB  "
                  f"{wr:>7.2f}")
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

  강화 비교군 + pre-x264 ablation:
    uv run python run_noise_experiment.py -v akiyo --sigma 0 10 \
      --baselines base nr hqdn3d dwt gaussian --include_precodec_ablation
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
    parser.add_argument("--threshold_mode", choices=["fixed", "adaptive", "rate_aware"],
                        default="adaptive", help="DT-CWT 임계값 모드")
    parser.add_argument("--controller_a", type=float, default=0.35, help="rate-aware 노이즈 계수")
    parser.add_argument("--controller_b", type=float, default=0.25, help="rate-aware 비트레이트 계수")
    parser.add_argument("--controller_c", type=float, default=0.25, help="rate-aware 모션 계수")
    parser.add_argument("--controller_d", type=float, default=0.25, help="rate-aware 에지 계수")
    parser.add_argument("--min_multiplier", type=float, default=0.5, help="rate-aware 최소 배율")
    parser.add_argument("--max_multiplier", type=float, default=2.5, help="rate-aware 최대 배율")
    parser.add_argument("--disable_rate_aware_scene_reset", action="store_true",
                        help="장면 전환 시 배율 중립화 비활성화")
    parser.add_argument("--log_context", action="store_true",
                        help="Proposed 청크 컨텍스트 CSV 로깅")
    parser.add_argument("--seed", type=int, default=42,
                        help="노이즈 랜덤 시드 (기본: 42)")
    parser.add_argument("--skip_existing", action="store_true",
                        help="이미 인코딩된 파일이 있으면 건너뜀")
    parser.add_argument("--include_precodec_ablation", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="x264 전 전처리-only ablation을 수행 (기본: 켬)")
    parser.add_argument("--reuse_preprocessed", action=argparse.BooleanOptionalAction,
                        default=True,
                        help=("pre-x264 전처리 산출물을 post-x264 인코딩 입력으로 재사용 "
                              "(fixed/adaptive Proposed에서 큰 폭으로 빠름, 기본: 켬)"))
    parser.add_argument("--chunk_size", type=int, default=16,
                        help="전처리 청크 크기. RTX 3090 권장값: 16~32 (기본: 16)")
    parser.add_argument("--overlap", type=int, default=4,
                        help="DT-CWT/DWT 시간축 overlap 프레임 수 (기본: 4)")
    parser.add_argument("--process_chroma", action="store_true",
                        help=("Proposed에서 U/V chroma도 DT-CWT 처리. "
                              "기본은 luma-only 노이즈 실험에 맞춰 비활성화"))
    parser.add_argument("--cleanup_intermediates", action="store_true",
                        help=("조건별 평가가 끝난 뒤 재생성 가능한 mp4/y4m 중간 파일을 삭제. "
                              "장시간 서버 실행 시 디스크 사용량 절감"))
    parser.add_argument("--baselines", nargs='+',
                        default=DEFAULT_BASELINES,
                        choices=['base', 'nr', 'hqdn3d', 'dwt', 'gaussian'],
                        help=("포함할 비교군 목록 (prop은 항상 포함됨). "
                              "기본: base nr hqdn3d dwt gaussian."))

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
    print(f"  임계값: {args.threshold} | 모드: {args.threshold_mode}")
    print(f"  청크: {args.chunk_size} frames | overlap: {args.overlap} | chroma DT-CWT: {args.process_chroma}")
    print(f"  전처리 재사용: {args.reuse_preprocessed} | 기존 산출물 재사용: {args.skip_existing}")
    print(f"  비교군: {args.baselines} + prop (always)")
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
                noise_seed = args.seed
            else:
                noisy_path = os.path.join(noisy_dir, f"{video_name}_s{sigma}.y4m")
                noise_seed = args.seed + sigma * 1000 + sum(ord(c) for c in video_name)
                if os.path.exists(noisy_path) and args.skip_existing:
                    print(f"  ♻️ 기존 노이즈 영상 재사용: {noisy_path}")
                    input_video = noisy_path
                else:
                    # 비디오+시그마별 고유 시드로 재현성 확보
                    input_video = create_noisy_video(
                        clean_video, noisy_path, sigma, seed=noise_seed
                    )

            # 2. 실험 수행
            results = run_single_condition(
                video_name, clean_video, input_video, sigma,
                args.output_dir, args.bitrates, args.threshold,
                baselines=args.baselines,
                include_precodec_ablation=args.include_precodec_ablation,
                reuse_preprocessed=args.reuse_preprocessed,
                skip_existing_outputs=args.skip_existing,
                chunk_size=args.chunk_size,
                overlap=args.overlap,
                process_chroma=args.process_chroma,
                seed=noise_seed,
                threshold_mode=args.threshold_mode,
                controller_a=args.controller_a,
                controller_b=args.controller_b,
                controller_c=args.controller_c,
                controller_d=args.controller_d,
                min_multiplier=args.min_multiplier,
                max_multiplier=args.max_multiplier,
                disable_rate_aware_scene_reset=args.disable_rate_aware_scene_reset,
                log_context=args.log_context,
            )

            # 3. 결과 처리
            summary = compute_condition_summary(results)
            save_condition_csv(results, args.output_dir)
            plot_condition_rd_curves(results, summary, args.output_dir)

            all_results[video_name][sigma] = results
            all_summaries[video_name][sigma] = summary

            if args.cleanup_intermediates:
                cleanup_extra = []
                if sigma != 0:
                    cleanup_extra.append(input_video)
                cleanup_condition_intermediates(results, args.output_dir, cleanup_extra)

            # BD-Rate 출력
            bd_p = summary['bd_rate_psnr_prop']
            bd_v = summary['bd_rate_msssim_prop']
            bd_p_str = f"{bd_p:+.2f}%" if not np.isnan(bd_p) else "N/A"
            bd_v_str = f"{bd_v:+.2f}%" if not np.isnan(bd_v) else "N/A"
            print(f"\n  📊 [{video_name.upper()} σ={sigma}] "
                  f"BD-Rate: PSNR={bd_p_str} | MS-SSIM={bd_v_str}")

    # 4. 전체 비교 시각화
    if all_results:
        print(f"\n{'=' * 60}")
        print(f"  📊 전체 비교 시각화 생성 중...")
        print(f"{'=' * 60}")

        plot_overlay_rd_curves(all_results, args.output_dir)
        plot_bd_rate_comparison(all_summaries, args.output_dir)
        plot_delta_psnr_trend(all_results, args.output_dir)
        plot_pre_post_delta_bars(all_summaries, args.output_dir)
        plot_codec_gain_heatmap(all_summaries, args.output_dir)
        csv_path = save_summary_csv(all_summaries, args.output_dir)
        save_reliable_metrics_csv(all_summaries, args.output_dir)

        # 콘솔 요약 테이블
        print_summary_table(all_summaries)

    print(f"\n{'=' * 60}")
    print(f"  ✅ 전체 실험 완료!")
    print(f"  📁 결과: {os.path.abspath(args.output_dir)}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
