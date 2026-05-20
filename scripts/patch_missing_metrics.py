"""Patch missing MS-SSIM and STRRED values in existing raw_data CSVs.

The script preserves already-computed metrics and only fills requested missing
columns from existing video artifacts. It then regenerates summary CSVs and,
optionally, plots.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import tempfile
from collections import defaultdict

import cv2
import numpy as np

from dtcwt_video.evaluate_metrics import compute_strred
from dtcwt_video.experiment_analysis import get_actual_bitrate_kbps
from scripts.redraw_noise_plots import _read_all_raw
from scripts.run_noise_experiment import (
    METRIC_NAMES,
    _metrics_to_dict,
    compute_condition_summary,
    plot_bd_rate_comparison,
    plot_codec_gain_heatmap,
    plot_condition_rd_curves,
    plot_delta_psnr_trend,
    plot_overlay_rd_curves,
    plot_pre_post_delta_bars,
    save_reliable_metrics_csv,
    save_summary_csv,
)


def _is_missing(value: str | None) -> bool:
    if value is None:
        return True
    value = str(value).strip()
    if value == "":
        return True
    try:
        return math.isnan(float(value))
    except ValueError:
        return value.lower() in {"nan", "none", "null"}


def _read_vmaf_metric(log_path: str, metric_name: str) -> float:
    with open(log_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    pooled = data.get("pooled_metrics", {})
    if metric_name in pooled and "mean" in pooled[metric_name]:
        return float(pooled[metric_name]["mean"])
    values = [
        frame.get("metrics", {}).get(metric_name)
        for frame in data.get("frames", [])
    ]
    values = [float(v) for v in values if v is not None]
    return float(np.mean(values)) if values else float("nan")


def compute_msssim(ref_video: str, dist_video: str) -> float:
    fd, log_abs = tempfile.mkstemp(
        prefix=f"patch_vmaf_{os.path.basename(dist_video)}_",
        suffix=".json",
        dir=os.getcwd(),
    )
    os.close(fd)
    os.remove(log_abs)
    log_name = os.path.basename(log_abs)
    cmd = [
        "ffmpeg", "-y", "-i", dist_video, "-i", ref_video,
        "-lavfi", f"libvmaf=log_path={log_name}:log_fmt=json:feature=name=float_ms_ssim",
        "-f", "null", "-",
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if not os.path.exists(log_name):
        return float("nan")
    try:
        return _read_vmaf_metric(log_name, "float_ms_ssim")
    finally:
        try:
            os.remove(log_name)
        except OSError:
            pass


def compute_strred_value(ref_video: str, dist_video: str, num_frames: int) -> float:
    cap_ref = cv2.VideoCapture(ref_video)
    cap_dist = cv2.VideoCapture(dist_video)
    try:
        return compute_strred(cap_ref, cap_dist, num_frames=num_frames)
    finally:
        cap_ref.release()
        cap_dist.release()


def artifact_path(output_dir: str, row: dict[str, str]) -> str | None:
    video = row.get("video", "")
    sigma = row.get("sigma", "")
    method = row.get("method", "")
    if row.get("stage") == "pre_x264":
        return os.path.join(output_dir, "pre_x264", f"{video}_s{sigma}_{method}.y4m")
    if row.get("stage") == "post_x264":
        br = row.get("target_bitrate_kbps", "")
        if br == "":
            return None
        return os.path.join(output_dir, f"{video}_s{sigma}_{method}_{int(float(br))}k.mp4")
    return None


def patch_raw_csv(path: str, output_dir: str, input_dir: str,
                  num_frames: int, metrics: set[str]) -> int:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    for metric in ["msssim", "strred"]:
        if metric in metrics and metric not in fieldnames:
            fieldnames.append(metric)

    changed = 0
    for row in rows:
        video = row.get("video", "")
        ref_video = os.path.join(input_dir, f"{video}.y4m")
        dist_video = artifact_path(output_dir, row)
        if not dist_video or not os.path.exists(ref_video) or not os.path.exists(dist_video):
            continue

        label = f"{row.get('video')} s{row.get('sigma')} {row.get('stage')} {row.get('method')}"
        if row.get("target_bitrate_kbps"):
            label += f" {row.get('target_bitrate_kbps')}k"

        if "msssim" in metrics and _is_missing(row.get("msssim")):
            print(f"  [MS-SSIM] {label}")
            row["msssim"] = compute_msssim(ref_video, dist_video)
            changed += 1

        if "strred" in metrics and _is_missing(row.get("strred")):
            print(f"  [STRRED]  {label}")
            row["strred"] = compute_strred_value(ref_video, dist_video, num_frames)
            changed += 1

    if changed:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key, "") for key in fieldnames})
    return changed


def refresh_actual_bitrates(all_results: dict, output_dir: str) -> None:
    for video, sigmas in all_results.items():
        for sigma, results in sigmas.items():
            for method in results.get("active_methods", []):
                bitrates = results.get("bitrates", [])
                if not bitrates:
                    continue
                actual = []
                for br in bitrates:
                    path = os.path.join(output_dir, f"{video}_s{sigma}_{method}_{br}k.mp4")
                    actual.append(get_actual_bitrate_kbps(path) if os.path.exists(path) else float("nan"))
                results.setdefault("actual_bitrates", {})[method] = actual


def rebuild_summaries_and_plots(output_dir: str, plots: bool) -> None:
    all_results = _read_all_raw(output_dir)
    refresh_actual_bitrates(all_results, output_dir)
    all_summaries = defaultdict(dict)
    for video, sigmas in all_results.items():
        for sigma, results in sigmas.items():
            if "base" not in results or not results.get("bitrates"):
                continue
            summary = compute_condition_summary(results)
            all_summaries[video][sigma] = summary
            if plots:
                plot_condition_rd_curves(results, summary, output_dir)

    all_summaries = {video: dict(sigmas) for video, sigmas in all_summaries.items()}
    save_summary_csv(all_summaries, output_dir)
    save_reliable_metrics_csv(all_summaries, output_dir)
    if plots:
        plot_overlay_rd_curves(all_results, output_dir)
        plot_bd_rate_comparison(all_summaries, output_dir)
        plot_delta_psnr_trend(all_results, output_dir)
        plot_pre_post_delta_bars(all_summaries, output_dir)
        plot_codec_gain_heatmap(all_summaries, output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fill missing MS-SSIM/STRRED values in existing raw_data CSVs."
    )
    parser.add_argument("-i", "--input_dir", default="videos")
    parser.add_argument("-o", "--output_dir", required=True)
    parser.add_argument("--num_frames", type=int, default=60)
    parser.add_argument("--metrics", nargs="+", choices=["msssim", "strred"],
                        default=["msssim", "strred"])
    parser.add_argument("--plots", action="store_true")
    args = parser.parse_args()

    output_dir = os.path.abspath(args.output_dir)
    input_dir = os.path.abspath(args.input_dir)
    metrics = set(args.metrics)

    raw_files = sorted(
        os.path.join(output_dir, name)
        for name in os.listdir(output_dir)
        if name.startswith("raw_data_") and name.endswith(".csv")
    )
    if not raw_files:
        raise SystemExit(f"No raw_data_*.csv files found in {output_dir}")

    total_changed = 0
    for path in raw_files:
        print(f"\n=== {os.path.basename(path)} ===")
        total_changed += patch_raw_csv(path, output_dir, input_dir, args.num_frames, metrics)

    print(f"\n패치한 값 수: {total_changed}")
    print("summary CSV 및 그래프 재생성 중...")
    rebuild_summaries_and_plots(output_dir, plots=args.plots)
    print(f"완료: {output_dir}")


if __name__ == "__main__":
    main()
