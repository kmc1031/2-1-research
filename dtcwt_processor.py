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

        홀수 크기의 차원이 입력으로 들어오면 짝수 크기로 패딩을 적용하여
        dtcwt 레벨 변환 시의 에러를 방지합니다.

        Args:
            chunk_numpy: (T, H, W) 형태의 NumPy 배열. 값 범위 [0, 1].

        Returns:
            전처리된 (T, H, W) NumPy 배열. 값 범위 [0, 1].
        """
        orig_shape = chunk_numpy.shape
        pad_t = orig_shape[0] % 2
        pad_h = orig_shape[1] % 2
        pad_w = orig_shape[2] % 2
        
        # 홀수 차원이 있을 경우 'Edge' 모드로 패딩하여 짝수 크기로 맞춤
        if pad_t > 0 or pad_h > 0 or pad_w > 0:
            # np.pad는 ((before_t, after_t), (before_h, after_h), (before_w, after_w)) 형식
            chunk_numpy = np.pad(
                chunk_numpy, 
                ((0, pad_t), (0, pad_h), (0, pad_w)), 
                mode='edge'
            )

        if _USE_CUDA:
            result = self._cuda_proc.process_chunk(chunk_numpy)
        else:
            # CPU 경로
            cube = chunk_numpy.transpose(1, 2, 0)
            pyramid = self._transform.forward(cube, nlevels=self.nlevels)
            pyramid.highpasses = self._apply_shrinkage_cpu(pyramid.highpasses)
            reconstructed = self._transform.inverse(pyramid)
            result = np.clip(reconstructed.transpose(2, 0, 1), 0.0, 1.0)
            
        # 원래 크기로 복원 (Crop)
        if pad_t > 0 or pad_h > 0 or pad_w > 0:
            # t, h, w 중 덧붙인 부분만 잘라내기
            t, h, w = orig_shape
            result = result[:t, :h, :w]
            
        return result