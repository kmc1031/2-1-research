"""
3D DT-CWT 기반 비디오 전처리 모듈.

CUDA GPU가 사용 가능하면 PyTorch CUDA 가속 버전을 자동으로 사용하고,
그렇지 않으면 CPU NumPy 버전으로 폴백합니다.
"""

import numpy as np

# CUDA 가용 여부 확인
_USE_CUDA = False
try:
    import torch
    if torch.cuda.is_available():
        from dtcwt_cuda import CudaDTCWT3DProcessor
        _USE_CUDA = True
        print("[DT-CWT] CUDA GPU 가속 활성화됨")
except ImportError:
    pass

if not _USE_CUDA:
    import dtcwt
    print("[DT-CWT] CPU 모드 (NumPy)")


class DTCWT3DProcessor:
    """3D DT-CWT를 사용한 비디오 청크 전처리기.

    CUDA 가용 시 자동으로 GPU 가속을 사용합니다.
    """

    def __init__(self, threshold=0.03, nlevels=1):
        self.threshold = threshold
        self.nlevels = nlevels

        if _USE_CUDA:
            self._cuda_proc = CudaDTCWT3DProcessor(
                threshold=threshold, nlevels=nlevels, device='cuda'
            )
            self._transform = None
        else:
            self._cuda_proc = None
            self._transform = dtcwt.Transform3d(
                biort='near_sym_a', qshift='qshift_a'
            )

    def _apply_shrinkage_cpu(self, highpasses):
        """CPU NumPy Soft Shrinkage."""
        shrunk = []
        for hp in highpasses:
            magnitude = np.abs(hp)
            with np.errstate(divide='ignore', invalid='ignore'):
                phase = hp / magnitude
                phase[np.isnan(phase)] = 0.0
            shrunk_mag = np.maximum(magnitude - self.threshold, 0.0)
            shrunk.append(phase * shrunk_mag)
        return tuple(shrunk)

    def process_chunk(self, chunk_numpy):
        """비디오 청크에 3D DT-CWT 전처리를 적용합니다.

        Args:
            chunk_numpy: (T, H, W) 형태의 NumPy 배열. 값 범위 [0, 1].

        Returns:
            전처리된 (T, H, W) NumPy 배열. 값 범위 [0, 1].
        """
        if _USE_CUDA:
            return self._cuda_proc.process_chunk(chunk_numpy)

        # CPU 경로
        cube = chunk_numpy.transpose(1, 2, 0)
        pyramid = self._transform.forward(cube, nlevels=self.nlevels)
        pyramid.highpasses = self._apply_shrinkage_cpu(pyramid.highpasses)
        reconstructed = self._transform.inverse(pyramid)
        return np.clip(reconstructed.transpose(2, 0, 1), 0.0, 1.0)