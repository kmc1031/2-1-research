"""
에지(윤곽선) 보존율 분석 모듈: EPSNR 및 에지맵 시각화.

원본 영상의 Sobel 에지맵을 기준으로, Baseline vs Proposed 인코딩 결과의
윤곽선 보존 정도를 EPSNR(Edge-PSNR)로 정량화하고 시각적으로 비교합니다.
"""

import os
import subprocess

import cv2
import numpy as np
import matplotlib.pyplot as plt

EDGE_PERCENTILE = 80  # 상위 20% 에지만 마스크로 사용


def extract_frame(video_path, frame_num, output_path):
    """FFmpeg를 사용하여 특정 프레임을 흑백(Grayscale) PNG로 추출합니다.

    Args:
        video_path: 비디오 파일 경로.
        frame_num: 추출할 프레임 번호 (0-indexed).
        output_path: 출력 PNG 파일 경로.
    """
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vf", f"select=eq(n\\,{frame_num}),format=gray",
        "-vframes", "1", output_path
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def compute_sobel_edges(image):
    """Sobel 필터를 사용하여 에지맵(Edge Map)을 추출합니다.

    Args:
        image: Grayscale 이미지 (2D NumPy 배열).

    Returns:
        [0, 255] 범위로 정규화된 에지 크기(magnitude) 맵.
    """
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobelx**2 + sobely**2)

    # 0~255로 정규화
    magnitude = np.uint8(255 * magnitude / np.max(magnitude))
    return magnitude


def calculate_epsnr(img_orig, img_distorted, edge_mask):
    """원본 영상의 에지 영역에 대해서만 PSNR을 계산합니다.

    EPSNR이 높을수록 모션 에지 보존율이 뛰어남을 의미합니다.

    Args:
        img_orig: 원본 Grayscale 이미지.
        img_distorted: 왜곡된 Grayscale 이미지.
        edge_mask: 에지 마스크 (uint8, 0 또는 1).

    Returns:
        EPSNR 값 (dB).
    """
    orig_edges = img_orig[edge_mask > 0].astype(np.float64)
    dist_edges = img_distorted[edge_mask > 0].astype(np.float64)

    if len(orig_edges) == 0:
        return 0.0

    mse = np.mean((orig_edges - dist_edges) ** 2)
    if mse == 0:
        return float('inf')

    return 10 * np.log10((255.0 ** 2) / mse)


def analyze_edges(video_name, bitrate, frame_num, crop_box=None):
    """특정 프레임의 에지맵을 추출하고 EPSNR을 비교 분석합니다.

    Args:
        video_name: 비디오 이름 (확장자 제외).
        bitrate: 비트레이트 문자열 (예: '200k').
        frame_num: 분석할 프레임 번호.
        crop_box: (x1, y1, x2, y2) 크롭 영역. None이면 전체 프레임 사용.
    """
    base_dir = "./outputs"
    os.makedirs(base_dir, exist_ok=True)

    orig_vid = f"./videos/{video_name}.y4m"
    base_vid = os.path.join(base_dir, f"{video_name}_base_{bitrate}.mp4")
    prop_vid = os.path.join(base_dir, f"{video_name}_prop_{bitrate}.mp4")

    # 임시 이미지 경로
    img_paths = {
        "Orig": os.path.join(base_dir, f"temp_{video_name}_orig.png"),
        "Base": os.path.join(base_dir, f"temp_{video_name}_base.png"),
        "Prop": os.path.join(base_dir, f"temp_{video_name}_prop.png"),
    }

    print(f"[{video_name.upper()}] {frame_num}번째 프레임 추출 중...")
    extract_frame(orig_vid, frame_num, img_paths["Orig"])
    extract_frame(base_vid, frame_num, img_paths["Base"])
    extract_frame(prop_vid, frame_num, img_paths["Prop"])

    # 이미지 로드 (Grayscale)
    imgs = {k: cv2.imread(v, cv2.IMREAD_GRAYSCALE) for k, v in img_paths.items()}

    # 크롭 적용
    if crop_box:
        x1, y1, x2, y2 = crop_box
        imgs = {k: img[y1:y2, x1:x2] for k, img in imgs.items()}

    # 에지맵 계산
    edges = {k: compute_sobel_edges(img) for k, img in imgs.items()}

    # EPSNR 계산 (원본 에지 중 상위 20%만 마스크로 사용)
    threshold = np.percentile(edges["Orig"], EDGE_PERCENTILE)
    edge_mask = (edges["Orig"] > threshold).astype(np.uint8)

    epsnr_base = calculate_epsnr(imgs["Orig"], imgs["Base"], edge_mask)
    epsnr_prop = calculate_epsnr(imgs["Orig"], imgs["Prop"], edge_mask)

    print("-" * 50)
    print(f"  📈 [{video_name.upper()} - {bitrate}] Edge-PSNR (EPSNR) 분석 결과")
    print(f"  * Baseline (x264) EPSNR : {epsnr_base:.2f} dB")
    print(f"  * Proposed (DT-CWT) EPSNR: {epsnr_prop:.2f} dB")
    print(f"  => EPSNR이 높을수록 주요 객체의 윤곽선이 원본에 가깝게 보존됨")
    print("-" * 50)

    # 시각화
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    titles_img = [
        "Original Frame",
        f"Baseline ({epsnr_base:.2f} dB)",
        f"Proposed ({epsnr_prop:.2f} dB)",
    ]
    titles_edge = [
        "Original Edge Map",
        "Baseline Edge Map\n(Notice blocking artifacts)",
        "Proposed Edge Map\n(Cleaner edges)",
    ]

    plot_imgs = [imgs["Orig"], imgs["Base"], imgs["Prop"]]
    plot_edges = [edges["Orig"], edges["Base"], edges["Prop"]]

    for i in range(3):
        axes[0, i].imshow(plot_imgs[i], cmap='gray')
        axes[0, i].set_title(titles_img[i], fontsize=14, fontweight='bold')
        axes[0, i].axis('off')

        axes[1, i].imshow(plot_edges[i], cmap='magma')
        axes[1, i].set_title(titles_edge[i], fontsize=13)
        axes[1, i].axis('off')

    plt.tight_layout()
    result_file = os.path.join(
        base_dir, f"edge_analysis_{video_name}_{bitrate}_f{frame_num}.png"
    )
    plt.savefig(result_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"💾 윤곽선 비교 결과 저장 완료: {result_file}")

    # 임시 파일 삭제
    for p in img_paths.values():
        os.remove(p)


if __name__ == "__main__":
    # Stefan 영상: 움직임이 빠르므로 100k~200k에서 블로킹이 극심합니다.
    # 해상도 352x288에서 테니스 선수와 관중석(네트 주변)을 크롭합니다.
    CROP_STEFAN = (50, 50, 300, 250)
    analyze_edges(video_name="stefan", bitrate="200k", frame_num=45, crop_box=CROP_STEFAN)
