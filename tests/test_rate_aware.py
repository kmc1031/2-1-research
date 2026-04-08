"""
Minimal sanity tests for the new rate-aware controller and context plumbing.
"""

import numpy as np

from dtcwt_video.dtcwt_processor import DTCWT3DProcessor, ProcessingContext
from dtcwt_video.pipeline import build_processing_context


def test_threshold_mode_backward_compatibility():
    proc_adapt = DTCWT3DProcessor(adaptive_threshold=True)
    assert proc_adapt.threshold_mode == "adaptive"

    proc_fixed = DTCWT3DProcessor(adaptive_threshold=False)
    assert proc_fixed.threshold_mode == "fixed"


def test_rate_aware_multiplier_direction():
    proc = DTCWT3DProcessor(
        threshold_mode="rate_aware",
        controller_a=0.35,
        controller_b=0.3,
        controller_c=0.3,
        controller_d=0.25,
        min_multiplier=0.5,
        max_multiplier=2.5,
    )
    low_bitrate_ctx = ProcessingContext(
        target_bitrate_kbps=200,
        noise_level=0.03,
        motion_strength=0.01,
        edge_density=0.05,
    )
    high_motion_ctx = ProcessingContext(
        target_bitrate_kbps=1200,
        noise_level=0.01,
        motion_strength=0.35,
        edge_density=0.45,
    )

    mult_low = proc.compute_controller_multiplier(low_bitrate_ctx)
    mult_high = proc.compute_controller_multiplier(high_motion_ctx)

    assert proc.min_multiplier <= mult_low <= proc.max_multiplier
    assert proc.min_multiplier <= mult_high <= proc.max_multiplier
    assert mult_low > mult_high, "Rate-aware multiplier should decrease for high-motion/high-detail chunks"


def test_build_processing_context_outputs():
    # Simple synthetic chunk with small gradients
    y = np.linspace(0, 1, 27, dtype=np.float32).reshape(3, 3, 3)
    ctx, ctx_log = build_processing_context(
        y, target_bitrate="500k", chunk_index=0, fps=30.0, scene_cut=False, mode="rate_aware"
    )
    assert ctx.target_bitrate_kbps == 500.0
    for key in ["chunk", "bitrate_kbps", "noise", "motion", "edge_density", "scene_cut"]:
        assert key in ctx_log
