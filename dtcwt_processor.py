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

    def __init__(self, threshold=0.03, nlevels=1, adaptive_threshold=True):
        self.threshold = threshold
        self.nlevels = nlevels
        self.adaptive_threshold = adaptive_threshold

        if _USE_CUDA:
            self._cuda_proc = CudaDTCWT3DProcessor(
                threshold=threshold, nlevels=nlevels, device='cuda',
                adaptive_threshold=adaptive_threshold
            )
            self._transform = None
        else:
            self._cuda_proc = None
            self._transform = dtcwt.Transform3d(
                biort='near_sym_a', qshift='qshift_a'
            )

    def _apply_shrinkage_cpu(self, highpasses):
        """Spatio-Temporal Adaptive Thresholding (BayesShrink 기반) 또는 고정 임계값."""
        if not self.adaptive_threshold:
            shrunk = []
            for hp in highpasses:
                magnitude = np.abs(hp)
                with np.errstate(divide='ignore', invalid='ignore'):
                    phase = hp / magnitude
                    phase[np.isnan(phase)] = 0.0
                shrunk_mag = np.maximum(magnitude - self.threshold, 0.0)
                shrunk.append(phase * shrunk_mag)
            return tuple(shrunk)

        finest_hps = highpasses[0]
        
        # 1. 전역 노이즈 분산 추정
        finest_mag = np.abs(finest_hps)
        mad = np.median(finest_mag)
        sigma = mad / 0.6745
        sigma_sq = sigma ** 2
        
        shrunk_levels = []
        for level_idx, hp in enumerate(highpasses):
            mag = np.abs(hp)
            
            # 서브밴드별 분산 계산 (H, W, T 축)
            subband_var = np.var(mag, axis=(0, 1, 2), keepdims=True)
            
            signal_var = np.maximum(subband_var - sigma_sq, 1e-8)
            sigma_x = np.sqrt(signal_var)
            
            base_factor = self.threshold * (2.0 ** level_idx)
            T_adapt = (sigma_sq / sigma_x) * base_factor
            
            with np.errstate(divide='ignore', invalid='ignore'):
                phase = hp / mag
                phase[np.isnan(phase)] = 0.0
                
            shrunk_mag = np.maximum(mag - T_adapt, 0.0)
            shrunk_levels.append(phase * shrunk_mag)
            
        return tuple(shrunk_levels)

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
        
        # 홀수 차원이 있을 경우 'symmetric' 모드로 패딩하여 짝수 크기로 맞춤
        if pad_t > 0 or pad_h > 0 or pad_w > 0:
            # np.pad는 ((before_t, after_t), (before_h, after_h), (before_w, after_w)) 형식
            chunk_numpy = np.pad(
                chunk_numpy, 
                ((0, pad_t), (0, pad_h), (0, pad_w)), 
                mode='symmetric'
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

    def process_chroma(self, u_flat, v_flat, w, h):
        """U, V 채널 (평면 배열)에 대해 간소화된 2D DT-CWT 노이즈 제거 진행.
        
        Args:
            u_flat: (frames, w*h//4) uint8
            v_flat: (frames, w*h//4) uint8
            w: 원본 Y 해상도 너비
            h: 원본 Y 해상도 높이
        
        Returns:
            전처리된 u_flat, v_flat
        """
        frames = u_flat.shape[0]
        cw, ch = w // 2, h // 2
        
        if not hasattr(self, '_transform2d'):
            import dtcwt
            self._transform2d = dtcwt.Transform2d(biort='near_sym_a', qshift='qshift_a')
            
        u_out = np.zeros_like(u_flat)
        v_out = np.zeros_like(v_flat)
        
        chroma_thresh = self.threshold * 1.5 # 색상 채널은 좀 더 강하게 컷오프
        
        for f in range(frames):
            u_frame = u_flat[f].reshape((ch, cw)).astype(np.float32) / 255.0
            v_frame = v_flat[f].reshape((ch, cw)).astype(np.float32) / 255.0
            
            # 홀수 패딩
            pad_h, pad_w = ch % 2, cw % 2
            if pad_h > 0 or pad_w > 0:
                u_frame = np.pad(u_frame, ((0, pad_h), (0, pad_w)), mode='symmetric')
                v_frame = np.pad(v_frame, ((0, pad_h), (0, pad_w)), mode='symmetric')
                
            # U 채널 2D
            u_pyr = self._transform2d.forward(u_frame, nlevels=1)
            u_shrunk = []
            for hp in u_pyr.highpasses:
                mag = np.abs(hp)
                with np.errstate(divide='ignore', invalid='ignore'):
                    phase = hp / mag
                    phase[np.isnan(phase)] = 0.0
                u_shrunk.append(phase * np.maximum(mag - chroma_thresh, 0.0))
            u_pyr.highpasses = tuple(u_shrunk)
            u_rec = self._transform2d.inverse(u_pyr)
            
            # V 채널 2D
            v_pyr = self._transform2d.forward(v_frame, nlevels=1)
            v_shrunk = []
            for hp in v_pyr.highpasses:
                mag = np.abs(hp)
                with np.errstate(divide='ignore', invalid='ignore'):
                    phase = hp / mag
                    phase[np.isnan(phase)] = 0.0
                v_shrunk.append(phase * np.maximum(mag - chroma_thresh, 0.0))
            v_pyr.highpasses = tuple(v_shrunk)
            v_rec = self._transform2d.inverse(v_pyr)
            
            if pad_h > 0 or pad_w > 0:
                u_rec = u_rec[:ch, :cw]
                v_rec = v_rec[:ch, :cw]
                
            u_out[f] = (np.clip(u_rec, 0.0, 1.0) * 255.0).astype(np.uint8).flatten()
            v_out[f] = (np.clip(v_rec, 0.0, 1.0) * 255.0).astype(np.uint8).flatten()
            
        return u_out, v_out