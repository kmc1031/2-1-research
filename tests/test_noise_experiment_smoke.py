import os

import pytest

from scripts.run_noise_experiment import (
    METRIC_NAMES,
    compute_condition_summary,
    run_single_condition,
    save_condition_csv,
)


@pytest.mark.requires_ffmpeg
def test_noise_experiment_pre_post_smoke():
    sample = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "data", "sample", "test_5frames.y4m")
    )
    if not os.path.exists(sample):
        pytest.skip("sample Y4M not available")
    output_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "outputs", "pytest_noise_smoke")
    )
    os.makedirs(output_dir, exist_ok=True)

    results = run_single_condition(
        "test_5frames",
        sample,
        sample,
        sigma=0,
        output_dir=output_dir,
        bitrates=[100],
        threshold=0.01,
        baselines=["base", "gaussian"],
        include_precodec_ablation=True,
        seed=42,
    )
    summary = compute_condition_summary(results)
    save_condition_csv(results, output_dir)

    csv_path = os.path.join(output_dir, "raw_data_test_5frames_s0.csv")
    assert os.path.exists(csv_path)
    assert os.path.exists(os.path.join(output_dir, "pre_x264", "test_5frames_s0_prop.y4m"))
    assert os.path.exists(os.path.join(output_dir, "test_5frames_s0_base_100k.mp4"))
    assert any(row["stage"] == "pre_x264" for row in results["rows"])
    assert any(row["stage"] == "post_x264" for row in results["rows"])
    assert results["actual_bitrates"]["base"][0] > 0
    assert "prop" in summary
    for metric in METRIC_NAMES:
        assert metric in results["prop"]
