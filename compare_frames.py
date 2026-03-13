"""
프레임 단위 시각적 비교 모듈.

원본, Baseline, Proposed 인코딩 결과를 같은 프레임에서 추출하여
나란히 비교하는 이미지를 생성합니다. 특정 영역 확대(crop)도 지원합니다.
"""

import os
import subprocess

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def extract_frame(video_path, frame_num, output_img):
    """FFmpeg를 사용하여 특정 프레임을 PNG 이미지로 추출합니다.

    Args:
        video_path: 비디오 파일 경로.
        frame_num: 추출할 프레임 번호 (0-indexed).
        output_img: 출력 PNG 파일 경로.
    """
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vf", f"select=eq(n\\,{frame_num})",
        "-vsync", "vfr", "-vframes", "1",
        output_img
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def plot_comparison(video_name, bitrate, frame_num, crop_box=None):
    """원본/베이스라인/제안기법 프레임을 추출하여 비교 이미지를 생성합니다.

    Args:
        video_name: 비디오 이름 (확장자 제외).
        bitrate: 비트레이트 문자열 (예: '100k').
        frame_num: 비교할 프레임 번호.
        crop_box: (x1, y1, x2, y2) 확대 영역. None이면 전체 프레임.
    """
    base_dir = "./outputs"

    orig_vid = f"./videos/{video_name}.y4m"
    base_vid = os.path.join(base_dir, f"{video_name}_base_{bitrate}.mp4")
    prop_vid = os.path.join(base_dir, f"{video_name}_prop_{bitrate}.mp4")

    if not os.path.exists(base_vid) or not os.path.exists(prop_vid):
        print(f"오류: {bitrate} 비트레이트의 결과 파일이 존재하지 않습니다.")
        return

    # 임시 이미지 경로
    orig_img = os.path.join(base_dir, f"temp_{video_name}_orig.png")
    base_img = os.path.join(base_dir, f"temp_{video_name}_base.png")
    prop_img = os.path.join(base_dir, f"temp_{video_name}_prop.png")

    print(f"[{video_name.upper()}] {frame_num}번째 프레임 추출 중...")
    extract_frame(orig_vid, frame_num, orig_img)
    extract_frame(base_vid, frame_num, base_img)
    extract_frame(prop_vid, frame_num, prop_img)

    # 이미지 로드
    img_orig = mpimg.imread(orig_img)
    img_base = mpimg.imread(base_img)
    img_prop = mpimg.imread(prop_img)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    titles = [
        "Original",
        f"Baseline (x264, {bitrate})",
        f"Proposed (DT-CWT, {bitrate})",
    ]
    imgs = [img_orig, img_base, img_prop]

    for ax, img, title in zip(axes, imgs, titles):
        if crop_box:
            x1, y1, x2, y2 = crop_box
            img = img[y1:y2, x1:x2]

        ax.imshow(img)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axis('off')

    plt.tight_layout()
    result_file = os.path.join(
        base_dir, f"visual_cmp_{video_name}_{bitrate}_f{frame_num}.png"
    )
    plt.savefig(result_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"=> 시각화 결과 저장 완료: {result_file}")

    # 임시 파일 삭제
    os.remove(orig_img)
    os.remove(base_img)
    os.remove(prop_img)


if __name__ == "__main__":
    # Akiyo 영상의 얼굴 부분을 확대하여 100kbps 극한 환경의 화질 비교
    CROP_FACE = (120, 50, 230, 180)
    plot_comparison(video_name="akiyo", bitrate="100k", frame_num=50, crop_box=CROP_FACE)