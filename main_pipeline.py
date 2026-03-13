"""
메인 파이프라인: Y4M 읽기 → 3D DT-CWT 전처리 → x264 인코딩.

Y/U/V 채널을 분리하여 Y 채널에만 DT-CWT 전처리를 적용하고,
원본 U/V와 함께 x264 인코더로 전달하는 완전한 파이프라인입니다.
"""

import os

import ffmpeg
import numpy as np

from dtcwt_processor import DTCWT3DProcessor


def get_video_metadata(file_path):
    """FFmpeg를 사용하여 비디오 파일의 해상도와 FPS를 추출합니다.

    Args:
        file_path: 비디오 파일 경로.

    Returns:
        (width, height, fps) 튜플.
    """
    probe = ffmpeg.probe(file_path)
    video_stream = next(
        stream for stream in probe['streams']
        if stream['codec_type'] == 'video'
    )

    width = int(video_stream['width'])
    height = int(video_stream['height'])

    # FPS 추출 (예: '30000/1001' → 29.97)
    fps_str = video_stream.get('r_frame_rate', '30/1')
    num, den = map(int, fps_str.split('/'))
    fps = num / den if den != 0 else 30.0

    return width, height, fps


def read_y4m_and_split(file_path, width, height, chunk_size=8):
    """Y4M 비디오를 청크 단위로 읽어 Y/U/V 채널을 분리하여 반환합니다.

    Y 채널은 [0, 1] float64로 정규화하고,
    U/V 채널은 원본 uint8 바이트 배열 그대로 유지합니다.

    Args:
        file_path: 비디오 파일 경로.
        width: 비디오 너비.
        height: 비디오 높이.
        chunk_size: 한 번에 읽을 프레임 수.

    Yields:
        (y_array, u_array, v_array, actual_frames) 튜플.
        - y_array: (T, H, W) float64 [0, 1]
        - u_array: (T, W*H/4) uint8
        - v_array: (T, W*H/4) uint8
        - actual_frames: 실제로 읽은 프레임 수
    """
    y_size = width * height
    uv_size = y_size // 4
    frame_bytes = y_size + (uv_size * 2)
    chunk_bytes = frame_bytes * chunk_size

    process = (
        ffmpeg
        .input(file_path)
        .output('pipe:', format='rawvideo', pix_fmt='yuv420p')
        .run_async(pipe_stdout=True, pipe_stderr=True)
    )

    try:
        while True:
            in_bytes = process.stdout.read(chunk_bytes)
            if not in_bytes:
                break

            actual_frames = len(in_bytes) // frame_bytes
            if actual_frames == 0:
                break

            # 1D 바이트 배열을 프레임 단위로 재배열
            raw_data = np.frombuffer(in_bytes, dtype=np.uint8).reshape(
                (actual_frames, frame_bytes)
            )

            # Y, U, V 분리
            y_channel = raw_data[:, :y_size].reshape((actual_frames, height, width))
            u_channel = raw_data[:, y_size:y_size + uv_size]
            v_channel = raw_data[:, y_size + uv_size:]

            # Y 채널만 [0, 1] float64로 정규화
            y_normalized = y_channel.astype(np.float64) / 255.0

            yield y_normalized, u_channel, v_channel, actual_frames
    finally:
        process.stdout.close()
        process.wait()


def create_x264_encoder(output_path, width, height, fps, bitrate='100k'):
    """FFmpeg를 사용해 rawvideo를 받아 x264로 인코딩하는 서브프로세스를 생성합니다.

    Args:
        output_path: 출력 파일 경로.
        width: 비디오 너비.
        height: 비디오 높이.
        fps: 프레임레이트.
        bitrate: 목표 비트레이트 (예: '100k', '500k').

    Returns:
        FFmpeg 서브프로세스 객체 (stdin으로 rawvideo 쓰기 가능).
    """
    process = (
        ffmpeg
        .input('pipe:', format='rawvideo', pix_fmt='yuv420p',
               s=f'{width}x{height}', framerate=fps)
        .output(output_path, vcodec='libx264', video_bitrate=bitrate,
                preset='fast', tune='zerolatency')
        .overwrite_output()
        .run_async(pipe_stdin=True)
    )
    return process


# --- 메인 실행 파이프라인 ---
if __name__ == "__main__":
    INPUT_VIDEO = "akiyo.y4m"
    TARGET_BITRATE = "100k"
    OUTPUT_VIDEO = f"akiyo_processed_{TARGET_BITRATE}.mp4"

    print(f"파이프라인을 시작합니다.")

    # 1. 메타데이터 추출
    w, h, fps = get_video_metadata(INPUT_VIDEO)
    print(f"해상도: {w}x{h}, FPS: {fps:.2f}, 목표 비트레이트: {TARGET_BITRATE}")

    # 2. 인코더 서브프로세스 열기
    encoder_process = create_x264_encoder(OUTPUT_VIDEO, w, h, fps, TARGET_BITRATE)

    # 3. 3D DT-CWT 프로세서 초기화
    processor = DTCWT3DProcessor(threshold=0.03)

    print("스트리밍 전처리 및 인코딩 진행 중...")

    # 4. 청크 단위 스트리밍 처리
    chunk_idx = 0
    for y_array, u_np, v_np, frames_in_chunk in read_y4m_and_split(
        INPUT_VIDEO, w, h, chunk_size=8
    ):
        # DT-CWT 전처리 (Y 채널만)
        processed_y = processor.process_chunk(y_array)

        # [0, 1] float → [0, 255] uint8 변환
        processed_y_uint8 = (processed_y * 255.0).clip(0, 255).astype(np.uint8)
        processed_y_flat = processed_y_uint8.reshape((frames_in_chunk, -1))

        # 인코더로 프레임 단위 쓰기 (Y + U + V)
        for f in range(frames_in_chunk):
            encoder_process.stdin.write(processed_y_flat[f].tobytes())
            encoder_process.stdin.write(u_np[f].tobytes())
            encoder_process.stdin.write(v_np[f].tobytes())

        chunk_idx += 1
        if chunk_idx % 5 == 0:
            print(f"  ... {chunk_idx * 8} 프레임 처리 완료")

    # 5. 파이프 종료 및 인코딩 마무리
    encoder_process.stdin.close()
    encoder_process.wait()

    print(f"완료! 인코딩된 파일이 저장되었습니다: {OUTPUT_VIDEO}")