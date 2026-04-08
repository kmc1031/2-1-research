"""
PyTorch CUDA 기반 3D DT-CWT 구현.

dtcwt 라이브러리의 NumPy 3D 변환(colfilter, coldfilt, colifilt, cube2c/c2cube,
level1/level2 forward/inverse)을 PyTorch 배치 conv1d로 재구현하여
NVIDIA GPU에서 가속합니다.
"""

import torch
import torch.nn.functional as F
import numpy as np
from dtcwt.coeffs import biort as _biort, qshift as _qshift
from dtcwt.utils import reflect


# ============================================================================
#  Low-level helpers
# ============================================================================

def _to_kernel(h):
    """1D filter → conv1d 커널 (1, 1, m)."""
    return h.flip(0).reshape(1, 1, -1)


def _reflect_idx(length, pad):
    """dtcwt 호환 대칭 확장 인덱스 (NumPy → LongTensor)."""
    idx = np.arange(-pad, length + pad, dtype=np.int64)
    return torch.from_numpy(
        reflect(idx, -0.5, length - 0.5).astype(np.int64)
    )


def _sym_pad_1d(x, pad):
    """배치 텐서 x (B, 1, L)에 대칭 패딩 적용."""
    L = x.shape[-1]
    idx = _reflect_idx(L, pad).to(x.device)
    return x[:, :, idx]


def _column_convolve_batch(x, h_kern):
    """배치 conv1d: x (B,1,L_padded), h_kern (1,1,m) → (B,1,L_valid)."""
    return F.conv1d(x, h_kern)


# ============================================================================
#  Batched colfilter / coldfilt / colifilt along arbitrary axis
# ============================================================================

def _colfilter_axis(X, h, axis):
    """3D 텐서 X의 지정 축에 colfilter 적용 (비데시메이션)."""
    X = X.movedim(axis, -1).contiguous()
    shape = X.shape
    L = shape[-1]
    batch = X.numel() // L

    m = h.shape[0]
    m2 = int(np.fix(m * 0.5))

    x_flat = X.reshape(batch, 1, L)
    x_pad = _sym_pad_1d(x_flat, m2)
    y_flat = _column_convolve_batch(x_pad, _to_kernel(h))

    new_shape = list(shape)
    new_shape[-1] = y_flat.shape[-1]
    return y_flat.reshape(new_shape).movedim(-1, axis).contiguous()


def _coldfilt_axis(X, ha, hb, axis):
    """3D 텐서 X의 지정 축에 coldfilt 적용 (2:1 데시메이션)."""
    X = X.movedim(axis, -1).contiguous()
    shape = X.shape
    L = shape[-1]
    batch = X.numel() // L

    m = ha.shape[0]
    m2 = int(np.fix(m * 0.5))
    r2 = L // 2

    # 대칭 확장
    xe = _reflect_idx(L, m).to(X.device)
    x_flat = X.reshape(batch, 1, L)
    x_ext = x_flat[:, :, xe]  # (B, 1, L+2m)

    # 홀/짝 필터 분리
    hao_k = _to_kernel(ha[0::2])
    hae_k = _to_kernel(ha[1::2])
    hbo_k = _to_kernel(hb[0::2])
    hbe_k = _to_kernel(hb[1::2])

    t = torch.arange(5, L + 2 * m - 2, 4, device=X.device)

    Y = torch.zeros(batch, 1, r2, device=X.device, dtype=X.dtype)

    if torch.sum(ha * hb) > 0:
        s1, s2 = slice(0, r2, 2), slice(1, r2, 2)
    else:
        s2, s1 = slice(0, r2, 2), slice(1, r2, 2)

    Y[:, :, s1] = (
        _column_convolve_batch(x_ext[:, :, t - 1], hao_k)
        + _column_convolve_batch(x_ext[:, :, t - 3], hae_k)
    )
    Y[:, :, s2] = (
        _column_convolve_batch(x_ext[:, :, t], hbo_k)
        + _column_convolve_batch(x_ext[:, :, t - 2], hbe_k)
    )

    new_shape = list(shape)
    new_shape[-1] = r2
    return Y.reshape(new_shape).movedim(-1, axis).contiguous()


def _colifilt_axis(X, ha, hb, axis):
    """3D 텐서 X의 지정 축에 colifilt 적용 (1:2 보간)."""
    X = X.movedim(axis, -1).contiguous()
    shape = X.shape
    L = shape[-1]
    batch = X.numel() // L

    m = ha.shape[0]
    m2 = int(np.fix(m * 0.5))

    Y = torch.zeros(batch, 1, L * 2, device=X.device, dtype=X.dtype)
    x_flat = X.reshape(batch, 1, L)

    # 영벡터 체크
    if not torch.any(x_flat != 0):
        new_shape = list(shape)
        new_shape[-1] = L * 2
        return Y.reshape(new_shape).movedim(-1, axis).contiguous()

    xe = _reflect_idx(L, int(m2)).to(X.device)
    x_ext = x_flat[:, :, xe]

    hao_k = _to_kernel(ha[0::2])
    hae_k = _to_kernel(ha[1::2])
    hbo_k = _to_kernel(hb[0::2])
    hbe_k = _to_kernel(hb[1::2])

    if m2 % 2 == 0:
        t = torch.arange(3, L + m, 2, device=X.device)
        if torch.sum(ha * hb) > 0:
            ta, tb = t, t - 1
        else:
            ta, tb = t - 1, t

        s = torch.arange(0, L * 2, 4, device=X.device)
        Y[:, :, s] = _column_convolve_batch(x_ext[:, :, tb - 2], hae_k)
        Y[:, :, s + 1] = _column_convolve_batch(x_ext[:, :, ta - 2], hbe_k)
        Y[:, :, s + 2] = _column_convolve_batch(x_ext[:, :, tb], hao_k)
        Y[:, :, s + 3] = _column_convolve_batch(x_ext[:, :, ta], hbo_k)
    else:
        t = torch.arange(2, L + m - 1, 2, device=X.device)
        if torch.sum(ha * hb) > 0:
            ta, tb = t, t - 1
        else:
            ta, tb = t - 1, t

        s = torch.arange(0, L * 2, 4, device=X.device)
        Y[:, :, s] = _column_convolve_batch(x_ext[:, :, tb], hao_k)
        Y[:, :, s + 1] = _column_convolve_batch(x_ext[:, :, ta], hbo_k)
        Y[:, :, s + 2] = _column_convolve_batch(x_ext[:, :, tb], hae_k)
        Y[:, :, s + 3] = _column_convolve_batch(x_ext[:, :, ta], hbe_k)

    new_shape = list(shape)
    new_shape[-1] = L * 2
    return Y.reshape(new_shape).movedim(-1, axis).contiguous()


# ============================================================================
#  cube2c / c2cube — 옥탠트 ↔ 복소수 매핑
# ============================================================================

def _cube2c(y):
    """실수 옥탠트 → 4개 복소 서브밴드."""
    A = y[0::2, 0::2, 0::2]
    B = y[0::2, 1::2, 0::2]
    C = y[1::2, 0::2, 0::2]
    D = y[1::2, 1::2, 0::2]
    E = y[0::2, 0::2, 1::2]
    F_ = y[0::2, 1::2, 1::2]
    G = y[1::2, 0::2, 1::2]
    H = y[1::2, 1::2, 1::2]

    p = 0.5 * (A - G - D - F_) + 0.5j * (B - H + C + E)
    q = 0.5 * (A - G + D + F_) + 0.5j * (-B + H + C + E)
    r = 0.5 * (A + G + D - F_) + 0.5j * (B + H - C + E)
    s = 0.5 * (A + G - D + F_) + 0.5j * (-B - H - C + E)

    return torch.stack([p, q, r, s], dim=3)


def _c2cube(z):
    """4개 복소 서브밴드 → 실수 옥탠트."""
    p, q, r, s = z[:, :, :, 0], z[:, :, :, 1], z[:, :, :, 2], z[:, :, :, 3]

    pr, pi = p.real, p.imag
    qr, qi = q.real, q.imag
    rr, ri = r.real, r.imag
    sr, si = s.real, s.imag

    out_shape = [d * 2 for d in z.shape[:3]]
    y = torch.zeros(out_shape, dtype=z.real.dtype, device=z.device)

    y[0::2, 0::2, 0::2] = (pr + qr + rr + sr)
    y[1::2, 0::2, 1::2] = (-pr - qr + rr + sr)
    y[1::2, 1::2, 0::2] = (-pr + qr + rr - sr)
    y[0::2, 1::2, 1::2] = (-pr + qr - rr + sr)
    y[0::2, 1::2, 0::2] = (pi - qi + ri - si)
    y[1::2, 1::2, 1::2] = (-pi + qi + ri - si)
    y[1::2, 0::2, 0::2] = (pi + qi - ri - si)
    y[0::2, 0::2, 1::2] = (pi + qi + ri + si)

    return y * 0.5


# ============================================================================
#  Level 1 Forward / Inverse
# ============================================================================

def _level1_xfm(X, h0o, h1o, ext_mode):
    """Level 1 순방향 3D DT-CWT (GPU 배치 연산)."""
    if ext_mode == 4 and any(s % 2 != 0 for s in X.shape):
        raise ValueError("ext_mode=4: 각 차원은 2의 배수여야 합니다.")

    # 작업 배열 생성
    ws = [s * 2 for s in X.shape]
    even_h = (h0o.shape[0] % 2 == 0)
    if even_h:
        ws = [s + 2 for s in ws]
    work = torch.zeros(ws, dtype=X.dtype, device=X.device)

    s0a = slice(None, ws[0] // 2)
    s1a = slice(None, ws[1] // 2)
    s2a = slice(None, ws[2] // 2)
    s0b = slice(ws[0] // 2, None)
    s1b = slice(ws[1] // 2, None)
    s2b = slice(ws[2] // 2, None)

    if even_h:
        work[:X.shape[0], :X.shape[1], :X.shape[2]] = X
        work[X.shape[0], :X.shape[1], :X.shape[2]] = X[-1, :, :]
        work[:X.shape[0], X.shape[1], :X.shape[2]] = X[:, -1, :]
        work[:X.shape[0], :X.shape[1], X.shape[2]] = X[:, :, -1]
        work[X.shape[0], X.shape[1], X.shape[2]] = X[-1, -1, -1]
        x0a = slice(None, X.shape[0])
        x1a = slice(None, X.shape[1])
        x2a = slice(None, X.shape[2])
        x0b = slice(ws[0] // 2, ws[0] // 2 + X.shape[0])
        x1b = slice(ws[1] // 2, ws[1] // 2 + X.shape[1])
        x2b = slice(ws[2] // 2, ws[2] // 2 + X.shape[2])
    else:
        work[s0a, s1a, s2a] = X
        x0a, x1a, x2a = s0a, s1a, s2a
        x0b, x1b, x2b = s0b, s1b, s2b

    # Step 1: 축2 (T) 필터링 — for loop 대신 배치
    y = work[s0a, s1a, x2a].clone()
    work[s0a, s1a, s2b] = _colfilter_axis(y, h1o, axis=2)
    work[s0a, s1a, s2a] = _colfilter_axis(y, h0o, axis=2)

    # Step 2: 축1(W) + 축0(H) 필터링 — for loop 대신 배치
    y1 = work[x0a, x1a, :].clone()  # (H, W, 2T)
    lo_w = _colfilter_axis(y1, h0o, axis=1)
    hi_w = _colfilter_axis(y1, h1o, axis=1)
    y2 = torch.cat([lo_w, hi_w], dim=1)  # (H, 2W, 2T)

    work[s0a, :, :] = _colfilter_axis(y2, h0o, axis=0)
    work[s0b, :, :] = _colfilter_axis(y2, h1o, axis=0)

    # 옥탠트 → 복소 계수
    Yl = work[s0a, s1a, s2a]
    Yh = torch.cat([
        _cube2c(work[x0a, x1b, x2a]),
        _cube2c(work[x0b, x1a, x2a]),
        _cube2c(work[x0b, x1b, x2a]),
        _cube2c(work[x0a, x1a, x2b]),
        _cube2c(work[x0a, x1b, x2b]),
        _cube2c(work[x0b, x1a, x2b]),
        _cube2c(work[x0b, x1b, x2b]),
    ], dim=3)

    return Yl, Yh


def _level1_ifm(Yl, Yh, g0o, g1o):
    """Level 1 역방향 3D DT-CWT (GPU 배치 연산)."""
    ws = [s * 2 for s in Yl.shape]
    work = torch.zeros(ws, dtype=Yl.dtype, device=Yl.device)

    Xshape = [s // 2 for s in ws]
    even_g = (g0o.shape[0] % 2 == 0)
    if even_g:
        Xshape = [s - 1 for s in Xshape]

    s0a = slice(None, ws[0] // 2)
    s1a = slice(None, ws[1] // 2)
    s2a = slice(None, ws[2] // 2)
    s0b = slice(ws[0] // 2, None)
    s1b = slice(ws[1] // 2, None)
    s2b = slice(ws[2] // 2, None)

    x0a = slice(None, Xshape[0])
    x1a = slice(None, Xshape[1])
    x2a = slice(None, Xshape[2])
    x0b = slice(ws[0] // 2, ws[0] // 2 + Xshape[0])
    x1b = slice(ws[1] // 2, ws[1] // 2 + Xshape[1])
    x2b = slice(ws[2] // 2, ws[2] // 2 + Xshape[2])

    # 복소 계수 → 옥탠트
    work[s0a, s1a, s2a] = Yl
    work[x0a, x1b, x2a] = _c2cube(Yh[:, :, :, 0:4])
    work[x0b, x1a, x2a] = _c2cube(Yh[:, :, :, 4:8])
    work[x0b, x1b, x2a] = _c2cube(Yh[:, :, :, 8:12])
    work[x0a, x1a, x2b] = _c2cube(Yh[:, :, :, 12:16])
    work[x0a, x1b, x2b] = _c2cube(Yh[:, :, :, 16:20])
    work[x0b, x1a, x2b] = _c2cube(Yh[:, :, :, 20:24])
    work[x0b, x1b, x2b] = _c2cube(Yh[:, :, :, 24:28])

    # Step 1: 축0(H) + 축1(W) 역필터링
    y_lo = _colfilter_axis(work[:, x1a, :], g0o, axis=1)
    y_hi = _colfilter_axis(work[:, x1b, :], g1o, axis=1)
    y = y_lo + y_hi  # (2H, W, 2T)

    lo_h = _colfilter_axis(y[x0a, :, :], g0o, axis=0)
    hi_h = _colfilter_axis(y[x0b, :, :], g1o, axis=0)
    work[s0a, s1a, :] = lo_h + hi_h  # (H, W, 2T)

    # Step 2: 축2(T) 역필터링
    y_t = work[s0a, :ws[1] // 2, :]
    lo_t = _colfilter_axis(y_t[:, :, x2a], g0o, axis=2)
    hi_t = _colfilter_axis(y_t[:, :, x2b], g1o, axis=2)
    result = lo_t + hi_t

    if even_g:
        return result[1:, 1:, 1:]
    return result


# ============================================================================
#  Level 2 Forward / Inverse
# ============================================================================

def _level2_xfm(X, h0a, h0b, h1a, h1b, ext_mode):
    """Level 2+ 순방향 3D DT-CWT."""
    # 패딩 (ext_mode=4)
    if ext_mode == 4:
        if X.shape[0] % 4 != 0:
            X = torch.cat([X[[0]], X, X[[-1]]], dim=0)
        if X.shape[1] % 4 != 0:
            X = torch.cat([X[:, [0]], X, X[:, [-1]]], dim=1)
        if X.shape[2] % 4 != 0:
            X = torch.cat([X[:, :, [0]], X, X[:, :, [-1]]], dim=2)

    ws = list(X.shape)
    work = X.clone()

    s0a = slice(None, ws[0] // 2)
    s1a = slice(None, ws[1] // 2)
    s2a = slice(None, ws[2] // 2)
    s0b = slice(ws[0] // 2, None)
    s1b = slice(ws[1] // 2, None)
    s2b = slice(ws[2] // 2, None)

    # 축2 필터링
    y = work.clone()
    work_new = torch.zeros_like(work)
    work_new[:, :, s2a] = _coldfilt_axis(y, h0b, h0a, axis=2)
    work_new[:, :, s2b] = _coldfilt_axis(y, h1b, h1a, axis=2)
    work = work_new

    # 축1 필터링
    y1 = work.clone()
    lo_w = _coldfilt_axis(y1, h0b, h0a, axis=1)
    hi_w = _coldfilt_axis(y1, h1b, h1a, axis=1)
    y2 = torch.cat([lo_w, hi_w], dim=1)

    # 축0 필터링
    lo_h = _coldfilt_axis(y2, h0b, h0a, axis=0)
    hi_h = _coldfilt_axis(y2, h1b, h1a, axis=0)
    work2 = torch.cat([lo_h, hi_h], dim=0)

    Yl = work2[s0a, s1a, s2a]
    Yh = torch.cat([
        _cube2c(work2[s0a, s1b, s2a]),
        _cube2c(work2[s0b, s1a, s2a]),
        _cube2c(work2[s0b, s1b, s2a]),
        _cube2c(work2[s0a, s1a, s2b]),
        _cube2c(work2[s0a, s1b, s2b]),
        _cube2c(work2[s0b, s1a, s2b]),
        _cube2c(work2[s0b, s1b, s2b]),
    ], dim=3)

    return Yl, Yh


def _level2_ifm(Yl, Yh, g0a, g0b, g1a, g1b, ext_mode, prev_shape):
    """Level 2+ 역방향 3D DT-CWT."""
    ws = [s * 2 for s in Yl.shape]
    work = torch.zeros(ws, dtype=Yl.dtype, device=Yl.device)

    s0a = slice(None, ws[0] // 2)
    s1a = slice(None, ws[1] // 2)
    s2a = slice(None, ws[2] // 2)
    s0b = slice(ws[0] // 2, None)
    s1b = slice(ws[1] // 2, None)
    s2b = slice(ws[2] // 2, None)

    work[s0a, s1a, s2a] = Yl
    work[s0a, s1b, s2a] = _c2cube(Yh[:, :, :, 0:4])
    work[s0b, s1a, s2a] = _c2cube(Yh[:, :, :, 4:8])
    work[s0b, s1b, s2a] = _c2cube(Yh[:, :, :, 8:12])
    work[s0a, s1a, s2b] = _c2cube(Yh[:, :, :, 12:16])
    work[s0a, s1b, s2b] = _c2cube(Yh[:, :, :, 16:20])
    work[s0b, s1a, s2b] = _c2cube(Yh[:, :, :, 20:24])
    work[s0b, s1b, s2b] = _c2cube(Yh[:, :, :, 24:28])

    # 축0 + 축1 역필터링
    y_lo = _colifilt_axis(work[:, s1a, :], g0b, g0a, axis=1)
    y_hi = _colifilt_axis(work[:, s1b, :], g1b, g1a, axis=1)
    y = y_lo + y_hi

    lo_h = _colifilt_axis(y[s0a, :, :], g0b, g0a, axis=0)
    hi_h = _colifilt_axis(y[s0b, :, :], g1b, g1a, axis=0)
    work2 = lo_h + hi_h

    # 축2 역필터링
    lo_t = _colifilt_axis(work2[:, :, s2a], g0b, g0a, axis=2)
    hi_t = _colifilt_axis(work2[:, :, s2b], g1b, g1a, axis=2)
    result = lo_t + hi_t

    # ext_mode에 따른 크롭
    prev_shape = torch.tensor(prev_shape[:3] if len(prev_shape) > 3 else prev_shape)
    curr_shape = torch.tensor(Yh.shape[:3])
    if ext_mode == 4:
        if curr_shape[0] * 2 != prev_shape[0]:
            result = result[1:-1, :, :]
        if curr_shape[1] * 2 != prev_shape[1]:
            result = result[:, 1:-1, :]
        if curr_shape[2] * 2 != prev_shape[2]:
            result = result[:, :, 1:-1]

    return result


# ============================================================================
#  Main Processor Class
# ============================================================================

class CudaDTCWT3DProcessor:
    """CUDA 가속 3D DT-CWT 프로세서.

    dtcwt.Transform3d와 동일한 인터페이스를 제공하되,
    내부 연산은 PyTorch CUDA에서 수행됩니다.
    """

    def __init__(self, threshold=0.03, nlevels=1, device='cuda',
                 adaptive_threshold=True,
                 biort_name='near_sym_a', qshift_name='qshift_a'):
        self.threshold = threshold
        self.nlevels = nlevels
        self.device = torch.device(device)
        self.ext_mode = 4
        self.adaptive_threshold = adaptive_threshold

        # 필터 계수 로딩 → PyTorch 텐서
        biort_coeffs = _biort(biort_name)
        qshift_coeffs = _qshift(qshift_name)

        def _t(arr):
            return torch.from_numpy(arr.flatten().astype(np.float32)).to(self.device)

        self.h0o, self.g0o, self.h1o, self.g1o = [_t(c) for c in biort_coeffs]
        self.h0a, self.h0b, self.g0a, self.g0b = [_t(c) for c in qshift_coeffs[:4]]
        self.h1a, self.h1b, self.g1a, self.g1b = [_t(c) for c in qshift_coeffs[4:8]]

    def forward(self, X_np, nlevels=None):
        """순방향 3D DT-CWT.

        Args:
            X_np: (H, W, T) NumPy 배열.
            nlevels: 분해 레벨 수. None이면 self.nlevels 사용.

        Returns:
            (lowpass, highpasses_tuple)
        """
        nlevels = nlevels or self.nlevels
        X = torch.from_numpy(X_np.astype(np.float32)).to(self.device)

        Yl = X
        Yh_list = [None] * nlevels

        for level in range(nlevels):
            if level == 0:
                Yl, Yh_list[level] = _level1_xfm(
                    Yl, self.h0o, self.h1o, self.ext_mode)
            else:
                Yl, Yh_list[level] = _level2_xfm(
                    Yl, self.h0a, self.h0b, self.h1a, self.h1b, self.ext_mode)

        return Yl, tuple(Yh_list)

    def inverse(self, Yl, Yh_tuple):
        """역방향 3D DT-CWT.

        Args:
            Yl: 저주파 텐서.
            Yh_tuple: 각 레벨의 고주파 계수 튜플.

        Returns:
            복원된 (H, W, T) NumPy 배열.
        """
        X = Yl
        nlevels = len(Yh_tuple)

        for level in range(nlevels):
            if level == nlevels - 1:
                X = _level1_ifm(X, Yh_tuple[-level - 1], self.g0o, self.g1o)
            else:
                if Yh_tuple[-level - 2] is not None:
                    prev_shape = Yh_tuple[-level - 2].shape
                else:
                    prev_shape = [s * 2 for s in Yh_tuple[-level - 1].shape[:3]]
                X = _level2_ifm(
                    X, Yh_tuple[-level - 1],
                    self.g0a, self.g0b, self.g1a, self.g1b,
                    self.ext_mode, prev_shape)

        return X.cpu().numpy()

    def apply_shrinkage(self, highpasses):
        """Spatio-Temporal Adaptive Thresholding (BayesShrink 기반) 또는 고정 임계값 감산."""
        if not self.adaptive_threshold:
            shrunk_levels = []
            for Yh in highpasses:
                mag = torch.abs(Yh)
                phase = torch.where(mag > 0, Yh / mag, torch.zeros_like(Yh))
                shrunk_mag = torch.clamp(mag - self.threshold, min=0.0)
                shrunk_levels.append(phase * shrunk_mag)
            return tuple(shrunk_levels)
            
        # adaptive_threshold == True 인 경우
        # highpasses는 튜플. 각 요소는 (H, W, T, 28) 형태의 복소수 텐서입니다.
        finest_hps = highpasses[0]
        
        # 1. 전역 노이즈 분산 추정
        # 복소수 magnitude는 Rayleigh 분포를 따르므로, Gaussian 가정의 MAD/0.6745가 성립하지 않음.
        # 실수부만으로 MAD를 계산하여 정확한 σ_noise 추정.
        finest_real = torch.real(finest_hps)
        mad = torch.median(torch.abs(finest_real))
        sigma = mad / 0.6745
        sigma_sq = sigma ** 2
        
        shrunk_levels = []
        for level_idx, Yh in enumerate(highpasses):
            # Yh: (H, W, T, 28)
            mag = torch.abs(Yh)
            
            # 방향별(서브밴드별)로 분산을 계산해야 하므로, (H, W, T) 차원에 대해 분산 계산
            # CPU 버전(ddof=1)과 완벽 동일한 동작을 위해 unbiased=True 명시
            # var shape: (28,)
            subband_var = torch.var(mag, dim=(0, 1, 2), unbiased=True, keepdim=True)
            
            signal_var = torch.clamp(subband_var - sigma_sq, min=1e-8)
            sigma_x = torch.sqrt(signal_var)
            
            base_factor = self.threshold * (2.0 ** level_idx)
            # T_adapt shape: (1, 1, 1, 28)
            T_adapt = (sigma_sq / sigma_x) * base_factor
            
            phase = torch.where(mag > 0, Yh / mag, torch.zeros_like(Yh))
            shrunk_mag = torch.clamp(mag - T_adapt, min=0.0)
            
            shrunk_levels.append(phase * shrunk_mag)
            
        return tuple(shrunk_levels)

    def process_chunk(self, chunk_numpy):
        """비디오 청크 전처리 (기존 DTCWT3DProcessor 호환 인터페이스).

        Args:
            chunk_numpy: (T, H, W) NumPy 배열. [0, 1] 범위.

        Returns:
            전처리된 (T, H, W) NumPy 배열.
        """
        # (T, H, W) → (H, W, T)
        cube = chunk_numpy.transpose(1, 2, 0)

        Yl, Yh = self.forward(cube, nlevels=self.nlevels)
        Yh_shrunk = self.apply_shrinkage(Yh)
        reconstructed = self.inverse(Yl, Yh_shrunk)

        # (H, W, T) → (T, H, W)
        return np.clip(reconstructed.transpose(2, 0, 1), 0.0, 1.0)
