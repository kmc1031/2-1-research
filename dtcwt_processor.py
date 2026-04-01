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

    def __init__(self, threshold=0.03, nlevels=1, adaptive_threshold=True, use_coef_cache=True):
        self.threshold = threshold
        self.nlevels = nlevels
        self.adaptive_threshold = adaptive_threshold
        self.use_coef_cache = use_coef_cache
        self.cached_Yl = None
        self.cached_Yh = None

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
        # 복소수 magnitude는 Rayleigh 분포를 따르므로, Gaussian 가정의 MAD/0.6745가 성립하지 않음.
        # 실수부(또는 허수부)만으로 MAD를 계산하여 정확한 σ_noise 추정.
        finest_real = np.real(finest_hps)
        mad = np.median(np.abs(finest_real))
        sigma = mad / 0.6745
        sigma_sq = sigma ** 2
        
        shrunk_levels = []
        for level_idx, hp in enumerate(highpasses):
            mag = np.abs(hp)
            
            # 서브밴드별 분산 계산 (H, W, T 축) (PyTorch의 unbiased=True와 일치하도록 ddof=1 설정)
            subband_var = np.var(mag, axis=(0, 1, 2), ddof=1, keepdims=True)
            
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

    def process_chunk(self, chunk_numpy, overlap_len=0):
        """비디오 청크에 3D DT-CWT 전처리를 적용합니다.

        홀수 크기의 차원이 입력으로 들어오면 짝수 크기로 패딩을 적용하여
        dtcwt 레벨 변환 시의 에러를 방지합니다.

        Args:
            chunk_numpy: (T, H, W) 형태의 NumPy 배열. 값 범위 [0, 1].
            overlap_len: 이전 청크와의 오버랩 프레임 수

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

        if self.use_coef_cache and overlap_len > 0 and self.cached_Yl is not None:
            # 시간축 오버랩 구간에 대해 공간 변환(Forward) 생략 (계수 레벨 캐시 활용)
            # 신규 프레임 추출
            new_chunk = chunk_numpy[overlap_len:]
            cube_new = new_chunk.transpose(1, 2, 0)
            
            if _USE_CUDA:
                import torch
                Yl_new, Yh_new = self._cuda_proc.forward(cube_new)
                
                # CPU RAM에서 VRAM으로 업로드
                cached_Yl_gpu = self.cached_Yl.to(self._cuda_proc.device)
                cached_Yh_gpu = tuple([h.to(self._cuda_proc.device) for h in self.cached_Yh])
                
                # 시간축 결합
                Yl_full = torch.cat([cached_Yl_gpu, Yl_new], dim=2)
                Yh_full = tuple([torch.cat([h_old, h_new], dim=2) for h_old, h_new in zip(cached_Yh_gpu, Yh_new)])
                
                Yh_shrunk = self._cuda_proc.apply_shrinkage(Yh_full)
                reconstructed = self._cuda_proc.inverse(Yl_full, Yh_shrunk)
                res_np = reconstructed
            else:
                Yl_new, Yh_new_tuple = self._transform.forward(cube_new, nlevels=self.nlevels)
                Yh_new = Yh_new_tuple.highpasses
                
                Yl_full = np.concatenate([self.cached_Yl, Yl_new], axis=2)
                Yh_full = tuple([np.concatenate([h_old, h_new], axis=2) for h_old, h_new in zip(self.cached_Yh, Yh_new)])
                
                Yh_shrunk = self._apply_shrinkage_cpu(Yh_full)
                
                import dtcwt
                class TempPyr:
                    def __init__(self, l, h):
                        self.lowpass = l
                        self.highpasses = h
                
                reconstructed = self._transform.inverse(TempPyr(Yl_full, Yh_shrunk))
                res_np = reconstructed

            Yl_for_cache = Yl_full
            Yh_for_cache = Yh_full
            result = np.clip(res_np.transpose(2, 0, 1), 0.0, 1.0)
            
        else:
            # 캐시가 없거나 미사용 시 전체 프레임 Forward 연산
            cube = chunk_numpy.transpose(1, 2, 0)
            if _USE_CUDA:
                Yl_for_cache, Yh_for_cache = self._cuda_proc.forward(cube)
                Yh_shrunk = self._cuda_proc.apply_shrinkage(Yh_for_cache)
                reconstructed = self._cuda_proc.inverse(Yl_for_cache, Yh_shrunk)
                res_np = reconstructed
            else:
                pyramid = self._transform.forward(cube, nlevels=self.nlevels)
                Yl_for_cache = pyramid.lowpass
                Yh_for_cache = pyramid.highpasses
                pyramid.highpasses = self._apply_shrinkage_cpu(pyramid.highpasses)
                reconstructed = self._transform.inverse(pyramid)
                res_np = reconstructed
                
            result = np.clip(res_np.transpose(2, 0, 1), 0.0, 1.0)
            
        # 다음 청크를 위해 OOM 방지용으로 VRAM에서 CPU RAM으로 Offload 및 캐싱 수행
        if self.use_coef_cache and overlap_len > 0:
            o_t = int(Yl_for_cache.shape[2] * (overlap_len / chunk_numpy.shape[0]))
            if o_t == 0: o_t = overlap_len
            
            if _USE_CUDA:
                import torch
                # VRAM Memory GC (Offload to CPU)
                self.cached_Yl = Yl_for_cache[:, :, -o_t:].detach().cpu()
                # Yh shape: (H, W, T_coef, 28) — T 축(dim=2)에서 o_t에 비례하는 프레임만 캐시
                self.cached_Yh = tuple([
                    h[:, :, -max(1, int(h.shape[2] * (o_t / Yl_for_cache.shape[2]))):, :].detach().cpu()
                    for h in Yh_for_cache
                ])
            else:
                self.cached_Yl = Yl_for_cache[:, :, -o_t:]
                # CPU Yh shape: (H, W, T_coef, 6) 또는 (H, W, T_coef, ...) 등 라이브러리 의존적
                self.cached_Yh = tuple([
                    h[:, :, -max(1, int(h.shape[2] * (o_t / Yl_for_cache.shape[2]))):]
                    for h in Yh_for_cache
                ])
        else:
            self.cached_Yl = None
            self.cached_Yh = None
            
        # 원래 크기로 복원 (Crop)
        if pad_t > 0 or pad_h > 0 or pad_w > 0:
            # t, h, w 중 덧붙인 부분만 잘라내기
            t, h, w = orig_shape
            result = result[:t, :h, :w]
            
        return result

    def process_chroma(self, u_flat, v_flat, w, h):
        """U, V 채널도 Y 채널과 동일하게 3D DT-CWT 공간-시간 필터링을 적용합니다.
        
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
        
        # [0, 1] float32 변환 및 3D 텐서 (T, H, W) 형태로 reshape
        u_cube = u_flat.reshape((frames, ch, cw)).astype(np.float32) / 255.0
        v_cube = v_flat.reshape((frames, ch, cw)).astype(np.float32) / 255.0
        
        # Y 채널 캐시를 백업 (Chroma 처리가 캐시를 오염시키는 것을 방지)
        saved_Yl = self.cached_Yl
        saved_Yh = self.cached_Yh
        
        # 색상 채널은 좀 더 강하게 컷오프하기 위해 threshold 임시 1.5배 증가
        orig_thresh = self.threshold
        self.threshold = orig_thresh * 1.5
        if _USE_CUDA:
            self._cuda_proc.threshold = orig_thresh * 1.5
            
        # 3D DT-CWT 적용 (overlap_len=0으로 호출하여 캐시 업데이트 방지)
        u_proc = self.process_chunk(u_cube, overlap_len=0)
        v_proc = self.process_chunk(v_cube, overlap_len=0)
        
        # threshold 복구 및 Y 채널 캐시 복원
        self.threshold = orig_thresh
        if _USE_CUDA:
            self._cuda_proc.threshold = orig_thresh
        self.cached_Yl = saved_Yl
        self.cached_Yh = saved_Yh
            
        # uint8 변환 및 1D 평면 포맷(flat)으로 복구
        u_out = (np.clip(u_proc, 0.0, 1.0) * 255.0).astype(np.uint8).reshape((frames, -1))
        v_out = (np.clip(v_proc, 0.0, 1.0) * 255.0).astype(np.uint8).reshape((frames, -1))
        
        return u_out, v_out