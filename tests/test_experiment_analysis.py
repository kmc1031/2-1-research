import math

import pytest

from dtcwt_video.experiment_analysis import (
    compute_codec_gain,
    estimate_bitrate_kbps,
    summarize_method_against_baseline,
)


def test_estimate_bitrate_kbps_from_size_and_duration():
    # 1,000 bytes over 2 seconds = 4 kbps.
    assert estimate_bitrate_kbps(1000, 2.0) == pytest.approx(4.0)
    assert math.isnan(estimate_bitrate_kbps(0, 2.0))
    assert math.isnan(estimate_bitrate_kbps(1000, 0.0))


def test_compute_codec_gain():
    assert compute_codec_gain(0.5, 0.8) == pytest.approx(0.3)
    assert math.isnan(compute_codec_gain(float("nan"), 0.8))


def test_summarize_method_against_baseline_codec_gain():
    target_bitrates = [100, 200, 300, 400]
    actual_base = [105, 205, 305, 405]
    actual_method = [103, 202, 301, 402]
    base_metrics = {
        "psnr": [30.0, 31.0, 32.0, 33.0],
        "vmaf": [70.0, 75.0, 80.0, 84.0],
    }
    method_metrics = {
        "psnr": [31.0, 32.0, 33.0, 34.0],
        "vmaf": [72.0, 77.0, 82.0, 86.0],
    }
    pre_base = {"psnr": 29.0, "vmaf": 68.0}
    pre_method = {"psnr": 29.4, "vmaf": 69.0}

    summary = summarize_method_against_baseline(
        target_bitrates,
        actual_base,
        actual_method,
        base_metrics,
        method_metrics,
        pre_base,
        pre_method,
    )

    assert summary["pre_delta_psnr"] == pytest.approx(0.4)
    assert summary["post_delta_psnr"] == pytest.approx(1.0)
    assert summary["codec_gain_psnr"] == pytest.approx(0.6)
    assert summary["low_bitrate_delta_psnr"] == pytest.approx(1.0)
    assert summary["win_rate_psnr"] == pytest.approx(1.0)
