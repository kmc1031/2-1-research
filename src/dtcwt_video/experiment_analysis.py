"""Experiment result helpers for codec-aware preprocessing studies."""

from __future__ import annotations

import math
import os
import subprocess
from typing import Iterable

import numpy as np

from dtcwt_video.encoders import calculate_bd_rate
from dtcwt_video.pipeline import get_video_metadata

HIGHER_IS_BETTER_METRICS = {"psnr", "ssim", "vmaf", "msssim", "epsnr", "psnrb", "mepr"}
LOWER_IS_BETTER_METRICS = {"gbim"}
RELIABLE_PRIMARY_METRICS = ("psnr", "msssim", "psnrb", "epsnr")


def estimate_bitrate_kbps(file_size_bytes: int, duration_seconds: float) -> float:
    """Estimate bitrate from file size and duration."""
    if file_size_bytes <= 0 or duration_seconds <= 0:
        return float("nan")
    return (file_size_bytes * 8.0) / duration_seconds / 1000.0


def get_video_duration_seconds(video_path: str) -> float:
    """Return container duration from ffprobe, or nan if unavailable."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    try:
        result = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, encoding="utf-8", check=False
        )
        return float(result.stdout.strip())
    except (ValueError, OSError):
        return float("nan")


def get_actual_bitrate_kbps(video_path: str) -> float:
    """Return actual encoded bitrate in kbps using ffprobe, then size fallback."""
    if not os.path.exists(video_path):
        return float("nan")

    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=bit_rate",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    try:
        result = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, encoding="utf-8", check=False
        )
        bit_rate = float(result.stdout.strip())
        if bit_rate > 0:
            return bit_rate / 1000.0
    except (ValueError, OSError):
        pass

    duration = get_video_duration_seconds(video_path)
    return estimate_bitrate_kbps(os.path.getsize(video_path), duration)


def y4m_duration_seconds(video_path: str) -> float:
    """Estimate Y4M duration from FFmpeg metadata."""
    try:
        width, height, fps = get_video_metadata(video_path)
    except Exception:
        return float("nan")
    if fps <= 0 or width <= 0 or height <= 0:
        return float("nan")

    y_size = width * height
    frame_bytes = y_size + (y_size // 4) * 2
    try:
        with open(video_path, "rb") as f:
            header = f.readline()
        frame_record_bytes = len(b"FRAME\n") + frame_bytes
        frames = max(0, (os.path.getsize(video_path) - len(header)) // frame_record_bytes)
        return frames / fps if frames else float("nan")
    except OSError:
        return float("nan")


def safe_mean(values: Iterable[float]) -> float:
    vals = [v for v in values if v is not None and not math.isnan(v)]
    return float(np.mean(vals)) if vals else float("nan")


def win_rate(deltas: Iterable[float]) -> float:
    vals = [v for v in deltas if v is not None and not math.isnan(v)]
    if not vals:
        return float("nan")
    return float(sum(v > 0 for v in vals) / len(vals))


def low_bitrate_average(bitrates: list[float], values: list[float],
                        cutoff_kbps: float = 300.0) -> float:
    selected = [v for br, v in zip(bitrates, values)
                if br <= cutoff_kbps and v is not None and not math.isnan(v)]
    return float(np.mean(selected)) if selected else float("nan")


def compute_codec_gain(pre_delta: float, post_delta: float) -> float:
    """Measure whether x264 amplifies the preprocessing benefit."""
    if pre_delta is None or post_delta is None:
        return float("nan")
    if math.isnan(pre_delta) or math.isnan(post_delta):
        return float("nan")
    return post_delta - pre_delta


def metric_delta(method_metrics: dict, reference_metrics: dict, metric: str) -> float:
    """Return a signed improvement delta for one metric key."""
    method_val = method_metrics.get(metric, float("nan"))
    ref_val = reference_metrics.get(metric, float("nan"))
    if method_val is None or ref_val is None:
        return float("nan")
    if math.isnan(method_val) or math.isnan(ref_val):
        return float("nan")
    if metric in LOWER_IS_BETTER_METRICS:
        return ref_val - method_val
    return method_val - ref_val


def summarize_method_against_baseline(
    target_bitrates: list[int],
    base_actual_bitrates: list[float],
    method_actual_bitrates: list[float],
    base_metrics: dict[str, list[float]],
    method_metrics: dict[str, list[float]],
    pre_base_metrics: dict[str, float] | None,
    pre_method_metrics: dict[str, float] | None,
) -> dict[str, float]:
    """Build robust post/pre summary metrics for one method."""
    metric_names = sorted(
        set(base_metrics.keys()) | set(method_metrics.keys()) | set(RELIABLE_PRIMARY_METRICS)
    )
    summary: dict[str, float] = {}

    for metric in metric_names:
        deltas = [
            metric_delta({metric: m}, {metric: b}, metric)
            for m, b in zip(method_metrics.get(metric, []), base_metrics.get(metric, []))
        ]
        pre_delta = (
            metric_delta(pre_method_metrics, pre_base_metrics, metric)
            if pre_method_metrics and pre_base_metrics else float("nan")
        )
        post_delta = safe_mean(deltas)

        summary[f"pre_delta_{metric}"] = pre_delta
        summary[f"post_delta_{metric}"] = post_delta
        summary[f"codec_gain_{metric}"] = compute_codec_gain(pre_delta, post_delta)
        summary[f"mean_delta_{metric}"] = post_delta
        summary[f"low_bitrate_delta_{metric}"] = low_bitrate_average(target_bitrates, deltas)
        summary[f"win_rate_{metric}"] = win_rate(deltas)

        if metric in HIGHER_IS_BETTER_METRICS:
            summary[f"bd_rate_{metric}"] = calculate_bd_rate(
                base_actual_bitrates, base_metrics.get(metric, []),
                method_actual_bitrates, method_metrics.get(metric, []),
            )

    return summary
