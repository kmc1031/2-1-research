"""
3D DT-CWT 변환-역변환 기본 동작 테스트.

가상의 3D 비디오 큐브를 생성하여 순방향/역방향 변환의
복원 정확도(Reconstruction Error)를 검증합니다.
"""

import numpy as np
import dtcwt


def test_roundtrip():
    """3D DT-CWT 순방향 → 역방향 변환의 복원 오차를 측정합니다."""
    # 가상 비디오 큐브: (Height=288, Width=352, Time=8)
    video_cube = np.random.rand(288, 352, 8)

    # 3D DT-CWT 변환 객체 생성
    transform3d = dtcwt.Transform3d(biort='near_sym_a', qshift='qshift_a')

    # 순방향 변환 (1-level 분해)
    transformed = transform3d.forward(video_cube, nlevels=1)

    print(f"저주파 성분 형태: {transformed.lowpass.shape}")
    print(f"고주파 서브밴드 형태: {transformed.highpasses[0].shape}")

    # 역방향 변환
    reconstructed_cube = transform3d.inverse(transformed)

    # 복원 오차 확인
    error = np.max(np.abs(video_cube - reconstructed_cube))
    print(f"복원 오차 (Reconstruction Error): {error:.5e}")

    if error < 1e-10:
        print("✅ Perfect reconstruction 확인!")
    else:
        print(f"⚠️ 복원 오차가 존재합니다. (허용 범위 내인지 확인 필요)")


if __name__ == "__main__":
    test_roundtrip()
