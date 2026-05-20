"""Redraw noise-experiment plots from existing CSV outputs.

This script rebuilds the in-memory result structures expected by
``run_noise_experiment.py`` plotting helpers, without rerunning encoding or
metric evaluation.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
from collections import defaultdict

from scripts.run_noise_experiment import (
    METRIC_NAMES,
    plot_bd_rate_comparison,
    plot_codec_gain_heatmap,
    plot_condition_rd_curves,
    plot_delta_psnr_trend,
    plot_overlay_rd_curves,
    plot_pre_post_delta_bars,
)


RAW_RE = re.compile(r"raw_data_(?P<video>.+)_s(?P<sigma>-?\d+)\.csv$")


def _to_float(value):
    if value is None or value == "":
        return float("nan")
    try:
        return float(value)
    except ValueError:
        return float("nan")


def _to_sigma(value) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _read_raw_csv(path: str) -> dict:
    name = os.path.basename(path)
    match = RAW_RE.match(name)
    if not match:
        raise ValueError(f"Unexpected raw-data filename: {name}")

    video = match.group("video")
    sigma = int(match.group("sigma"))
    post_rows = []
    pre_rows = []

    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("stage") == "post_x264":
                post_rows.append(row)
            elif row.get("stage") == "pre_x264":
                pre_rows.append(row)

    active_methods = list(dict.fromkeys(row["method"] for row in post_rows))
    bitrates = sorted({
        int(float(row["target_bitrate_kbps"]))
        for row in post_rows
        if row.get("target_bitrate_kbps") not in ("", None)
    })

    results = {
        "video": video,
        "sigma": sigma,
        "bitrates": bitrates,
        "active_methods": active_methods,
        "actual_bitrates": {method: [] for method in active_methods},
        "pre_metrics": {},
    }
    for method in active_methods:
        results[method] = {metric: [] for metric in METRIC_NAMES}

    rows_by_method = defaultdict(list)
    for row in post_rows:
        rows_by_method[row["method"]].append(row)

    for method in active_methods:
        method_rows = sorted(
            rows_by_method[method],
            key=lambda r: _to_float(r.get("target_bitrate_kbps")),
        )
        for row in method_rows:
            results["actual_bitrates"][method].append(
                _to_float(row.get("actual_bitrate_kbps"))
            )
            for metric in METRIC_NAMES:
                results[method][metric].append(_to_float(row.get(metric)))

    for row in pre_rows:
        method = row.get("method")
        if method:
            results["pre_metrics"][method] = {
                metric: _to_float(row.get(metric))
                for metric in METRIC_NAMES
            }

    return results


def _read_all_raw(output_dir: str) -> dict:
    all_results = defaultdict(dict)
    for name in sorted(os.listdir(output_dir)):
        if not RAW_RE.match(name):
            continue
        result = _read_raw_csv(os.path.join(output_dir, name))
        all_results[result["video"]][result["sigma"]] = result
    return {video: dict(sigmas) for video, sigmas in all_results.items()}


def _read_summaries(output_dir: str) -> dict:
    path = os.path.join(output_dir, "summary_bd_rates.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing summary CSV: {path}")

    all_summaries = defaultdict(lambda: defaultdict(dict))
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            video = row.get("Video", "")
            if not video:
                continue
            sigma = _to_sigma(row.get("Sigma"))
            method = row.get("Method", "")
            values = {
                key: _to_float(value)
                for key, value in row.items()
                if key not in {"Video", "Sigma", "Method"}
            }
            all_summaries[video][sigma][method] = values

    nested = {video: dict(sigmas) for video, sigmas in all_summaries.items()}
    for video, sigmas in nested.items():
        for sigma, summary in sigmas.items():
            prop = summary.get("prop", {})
            summary["bd_rate_psnr_prop"] = prop.get("bd_rate_psnr", float("nan"))
            summary["bd_rate_msssim_prop"] = prop.get("bd_rate_msssim", float("nan"))
            summary["bd_rate_vmaf_prop"] = prop.get("bd_rate_vmaf", float("nan"))
            for method in ["dwt", "nr", "hqdn3d", "gaussian"]:
                values = summary.get(method, {})
                summary[f"bd_rate_psnr_{method}"] = values.get("bd_rate_psnr", float("nan"))
                summary[f"bd_rate_msssim_{method}"] = values.get("bd_rate_msssim", float("nan"))
                summary[f"bd_rate_vmaf_{method}"] = values.get("bd_rate_vmaf", float("nan"))
    return nested


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Redraw noise-experiment PNG plots from existing CSV files."
    )
    parser.add_argument(
        "-i", "--input_dir",
        default="outputs/noise_experiment",
        help="Directory containing raw_data_*_s*.csv and summary_bd_rates.csv.",
    )
    parser.add_argument(
        "-o", "--output_dir",
        default=None,
        help="Directory to write plots. Defaults to input_dir.",
    )
    args = parser.parse_args()

    input_dir = os.path.abspath(args.input_dir)
    output_dir = os.path.abspath(args.output_dir or args.input_dir)
    os.makedirs(output_dir, exist_ok=True)

    all_results = _read_all_raw(input_dir)
    all_summaries = _read_summaries(input_dir)
    if not all_results:
        raise SystemExit(f"No raw_data_*_s*.csv files found in {input_dir}")

    for video in all_results:
        for sigma in sorted(all_results[video]):
            summary = all_summaries.get(video, {}).get(sigma, {})
            plot_condition_rd_curves(all_results[video][sigma], summary, output_dir)

    plot_overlay_rd_curves(all_results, output_dir)
    plot_bd_rate_comparison(all_summaries, output_dir)
    plot_delta_psnr_trend(all_results, output_dir)
    plot_pre_post_delta_bars(all_summaries, output_dir)
    plot_codec_gain_heatmap(all_summaries, output_dir)

    print(f"그래프 재생성 완료: {output_dir}")


if __name__ == "__main__":
    main()
