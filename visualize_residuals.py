"""
잔차 이미지(Residual Image) 시각화 도구.
원본 비디오와 전처리된 비디오 간의 픽셀 차이(절대값)를 계산하고,
가시성을 높여(Amplification) 새로운 비디오 또는 단일 프레임 이미지로 저장합니다.
"""

import argparse
import numpy as np
import ffmpeg
import cv2
import os

def get_metadata(file_path):
    probe = ffmpeg.probe(file_path)
    video_stream = next(stream for stream in probe['streams'] if stream['codec_type'] == 'video')
    return int(video_stream['width']), int(video_stream['height'])

def extract_frame_luma(file_path, width, height, frame_idx):
    """지정된 프레임 인덱스에서 Y 채널(Luma)만 추출 (uint8)."""
    y_size = width * height
    frame_bytes = y_size * 3 // 2
    
    out, _ = (
        ffmpeg
        .input(file_path)
        .filter('select', f'eq(n,{frame_idx})')
        .output('pipe:', format='rawvideo', pix_fmt='yuv420p', vframes=1)
        .global_args('-loglevel', 'quiet')
        .run(capture_stdout=True)
    )
    if not out:
        return None
        
    raw = np.frombuffer(out, dtype=np.uint8)
    return raw[:y_size].reshape((height, width))

def main():
    parser = argparse.ArgumentParser(description="원본과 전처리 영상 간 잔차 이미지 생성")
    parser.add_argument("-o", "--original", required=True, help="원본 비디오 (예: akiyo.y4m)")
    parser.add_argument("-p", "--processed", required=True, help="전처리(또는 인코딩) 비디오 (예: akiyo_processed.mp4)")
    parser.add_argument("-f", "--frame", type=int, default=10, help="비교할 프레임 번호")
    parser.add_argument("-c", "--contrast", type=float, default=10.0, help="잔차 증폭 배수")
    parser.add_argument("--out", default="residual.png", help="출력 이미지 경로")
    args = parser.parse_args()

    w, h = get_metadata(args.original)
    
    orig_y = extract_frame_luma(args.original, w, h, args.frame)
    proc_y = extract_frame_luma(args.processed, w, h, args.frame)
    
    if orig_y is None or proc_y is None:
        print("프레임을 추출할 수 없습니다. (비디오가 프레임보다 짧을 수 있습니다.)")
        return
        
    orig_float = orig_y.astype(np.float32)
    proc_float = proc_y.astype(np.float32)
    
    diff = np.abs(orig_float - proc_float)
    amplified = np.clip(diff * args.contrast, 0, 255).astype(np.uint8)
    
    # Residual Edge Analysis (구조적 에지 보존 검증을 위한 잔차의 윤곽선 추출)
    sobelx = cv2.Sobel(amplified, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(amplified, cv2.CV_64F, 0, 1, ksize=3)
    edge_mag = np.sqrt(sobelx**2 + sobely**2)
    edge_mag = np.uint8(255 * edge_mag / np.max(edge_mag)) if np.max(edge_mag) > 0 else np.zeros_like(amplified)
    
    cv2.imwrite(args.out, amplified)
    print(f"잔차 이미지가 저장되었습니다: {args.out} (증폭: x{args.contrast})")
    
    edge_out = os.path.splitext(args.out)[0] + "_edges.png"
    cv2.imwrite(edge_out, edge_mag)
    print(f"잔차의 에지(윤곽선) 이미지가 저장되었습니다: {edge_out} (구조적 에지 파괴 여부 검증용)")
    
    # 추가로 비교가 쉽도록 가로로 이어붙인 (원본 | 처리본 | 잔차 | 잔차 에지) 이미지 생성
    combined = np.hstack((orig_y, proc_y, amplified, edge_mag))
    combined_out = os.path.splitext(args.out)[0] + "_combined.png"
    cv2.imwrite(combined_out, combined)
    print(f"비교 이미지가 저장되었습니다: {combined_out} [원본, 처리본, 잔차, 잔차에지]")

if __name__ == "__main__":
    main()
