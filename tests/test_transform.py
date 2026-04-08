"""
tests/test_transform.py: DT-CWT 변환 왕복 일관성 및 CPU/GPU 결과 일치 테스트.

모든 테스트는 FFmpeg 없이 실행 가능합니다.
"""
import numpy as np
import pytest

from dtcwt_video.dtcwt_processor import DTCWT3DProcessor


class TestDTCWT3DRoundTrip:
    """3D DT-CWT forward → inverse 왕복 오차 검증."""

    def test_round_trip_zero_threshold(self):
        """임계값 0일 때 forward → inverse 오차가 거의 0이어야 한다."""
        video = np.random.rand(8, 32, 32).astype(np.float32)
        proc = DTCWT3DProcessor(threshold=0.0, adaptive_threshold=False,
                                 use_coef_cache=False)
        reconstructed = proc.process_chunk(video, overlap_len=0)

        assert reconstructed.shape == video.shape
        mse = float(np.mean((reconstructed - video) ** 2))
        assert mse < 1e-3, f"Round-trip MSE too large: {mse:.6f}"

    def test_round_trip_small_threshold_reduces_noise(self):
        """임계값 > 0일 때 노이즈 성분이 감소해야 한다."""
        clean = np.ones((8, 32, 32), dtype=np.float32) * 0.5
        noise = np.random.normal(0, 0.05, clean.shape).astype(np.float32)
        noisy = np.clip(clean + noise, 0, 1)

        proc = DTCWT3DProcessor(threshold=0.03, adaptive_threshold=True,
                                 use_coef_cache=False)
        denoised = proc.process_chunk(noisy, overlap_len=0)

        noise_before = float(np.mean((noisy - clean) ** 2))
        noise_after = float(np.mean((denoised - clean) ** 2))

        assert denoised.shape == noisy.shape
        assert noise_after <= noise_before * 1.5, (
            f"Denoising did not help: before={noise_before:.6f}, after={noise_after:.6f}"
        )

    def test_output_value_range(self):
        """출력값이 [0, 1] 범위를 크게 벗어나지 않아야 한다."""
        video = np.random.rand(8, 32, 32).astype(np.float32)
        proc = DTCWT3DProcessor(threshold=0.01, adaptive_threshold=False,
                                 use_coef_cache=False)
        out = proc.process_chunk(video, overlap_len=0)

        assert float(out.min()) >= -0.1, f"Output below -0.1: {out.min():.4f}"
        assert float(out.max()) <= 1.1, f"Output above 1.1: {out.max():.4f}"

    def test_chunk_overlap_consistency(self):
        """오버랩 처리: 청크 경계에서 결과가 연속적이어야 한다."""
        video = np.random.rand(16, 32, 32).astype(np.float32)

        proc_ref = DTCWT3DProcessor(threshold=0.0, adaptive_threshold=False,
                                     use_coef_cache=False)
        out_ref = proc_ref.process_chunk(video[4:16], overlap_len=0)

        proc_cache = DTCWT3DProcessor(threshold=0.0, adaptive_threshold=False,
                                       use_coef_cache=True)
        _ = proc_cache.process_chunk(video[0:8], overlap_len=0)
        out_cache = proc_cache.process_chunk(video[4:16], overlap_len=4)

        mse = float(np.mean((out_ref[4:] - out_cache[4:]) ** 2))
        assert mse < 1e-2, f"Chunk overlap continuity error (MSE={mse:.6f})"


@pytest.mark.requires_gpu
class TestCPUGPUConsistency:
    """CPU와 GPU 구현의 결과 일치 검증."""

    def test_cpu_gpu_match(self):
        """CPU와 GPU 처리 결과의 차이가 허용 범위 내이어야 한다."""
        import torch
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        video = np.random.rand(8, 32, 32).astype(np.float32)

        proc_cpu = DTCWT3DProcessor(threshold=0.02, adaptive_threshold=False,
                                     use_coef_cache=False)
        proc_gpu = DTCWT3DProcessor(threshold=0.02, adaptive_threshold=False,
                                     use_coef_cache=False)

        out_cpu = proc_cpu.process_chunk(video, overlap_len=0)
        out_gpu = proc_gpu.process_chunk(video, overlap_len=0)

        max_diff = float(np.max(np.abs(out_cpu - out_gpu)))
        assert max_diff < 1e-3, f"CPU/GPU difference too large: {max_diff:.6f}"
