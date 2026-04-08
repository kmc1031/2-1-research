"""
tests/test_caching.py: 계수 캐싱 및 청크 오버랩 수학적 일관성 테스트.

캐시를 사용한 청크 처리와 연속 처리의 결과 차이를 검증합니다.
"""
import numpy as np
import pytest

from dtcwt_video.dtcwt_processor import DTCWT3DProcessor


class TestCachingConsistency:
    """계수 캐싱 방식의 수학적 일관성 검증."""

    def test_cache_vs_continuous_psnr(self):
        """캐시 처리와 연속 처리의 MSE가 허용 범위 내이어야 한다."""
        video = np.random.rand(16, 16, 16).astype(np.float32)

        # Reference: 연속 처리
        proc_ref = DTCWT3DProcessor(threshold=0.0, use_coef_cache=False)
        out_ref = proc_ref.process_chunk(video[4:16], overlap_len=0)

        # Cached: 청크 2번에 걸쳐 처리
        proc_cache = DTCWT3DProcessor(threshold=0.0, use_coef_cache=True)
        _ = proc_cache.process_chunk(video[0:8], overlap_len=0)
        out_cache = proc_cache.process_chunk(video[4:16], overlap_len=4)

        # 유효 부분만 비교
        mse = np.mean((out_cache[4:] - out_ref[4:]) ** 2)
        assert mse < 1e-2, f"Cache vs continuous MSE too large: {mse:.6f}"

    def test_no_cache_overlap_zero(self):
        """오버랩 0, 캐시 없이 처리해도 기본 왕복이 작동해야 한다."""
        video = np.random.rand(8, 16, 16).astype(np.float32)
        proc = DTCWT3DProcessor(threshold=0.0, use_coef_cache=False)
        out = proc.process_chunk(video, overlap_len=0)
        assert out.shape == video.shape

    def test_caching_does_not_corrupt_output(self):
        """캐시 처리 후 출력값이 NaN 또는 Inf를 포함하지 않아야 한다."""
        video = np.random.rand(16, 16, 16).astype(np.float32)
        proc = DTCWT3DProcessor(threshold=0.02, use_coef_cache=True)

        _ = proc.process_chunk(video[0:8], overlap_len=0)
        out = proc.process_chunk(video[4:16], overlap_len=4)

        assert not np.any(np.isnan(out)), "Output contains NaN values"
        assert not np.any(np.isinf(out)), "Output contains Inf values"


@pytest.mark.requires_gpu
class TestCUDACaching:
    """CUDA GPU 캐싱 수학 일관성 테스트 (gpu 필요 시만 실행)."""

    def test_cuda_coefficient_concat(self):
        """CUDA forward → 계수 concat → inverse의 일관성을 검증한다."""
        import torch
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        try:
            from dtcwt_video.dtcwt_cuda import CudaDTCWT3DProcessor
        except ImportError:
            pytest.skip("CudaDTCWT3DProcessor not available")

        proc = CudaDTCWT3DProcessor(nlevels=1, adaptive_threshold=False, threshold=0.0)
        video = np.random.rand(12, 16, 16).astype(np.float32)

        # 연속 처리
        cube_ref = video.transpose(1, 2, 0)
        Yl_ref, Yh_ref = proc.forward(cube_ref)
        out_ref = proc.inverse(Yl_ref, Yh_ref)
        out_ref_valid = out_ref[:, :, 8:12]

        # 청크 처리
        cube_c1 = video[:8].transpose(1, 2, 0)
        Yl_1, Yh_1 = proc.forward(cube_c1)
        cached_Yl = Yl_1[:, :, -2:]
        cached_Yh = tuple([h[:, :, -2:, :] for h in Yh_1])

        cube_c2 = video[8:12].transpose(1, 2, 0)
        Yl_2, Yh_2 = proc.forward(cube_c2)

        import torch
        Yl_concat = torch.cat([cached_Yl, Yl_2], dim=2)
        Yh_concat = tuple([torch.cat([h_old, h_new], dim=2)
                           for h_old, h_new in zip(cached_Yh, Yh_2)])
        out_concat = proc.inverse(Yl_concat, Yh_concat)
        out_valid = out_concat[:, :, 4:8]

        mse = float(np.mean((out_valid - out_ref_valid) ** 2))
        # CUDA 경계 패딩 차이로 완전 0은 어려우므로, 실용 허용오차로 검증
        assert mse < 2e-3, f"CUDA coefficient concat MSE too large: {mse:.8f}"
