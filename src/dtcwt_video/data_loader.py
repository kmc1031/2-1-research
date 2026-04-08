"""
비디오 데이터 로더 (프로토타입).

Y4M 비디오 파일을 청크 단위로 읽어 Y 채널만 추출하는 초기 버전입니다.
U/V 채널 분리까지 지원하는 완전한 버전은 main_pipeline.py를 참고하세요.
"""

import ffmpeg
import numpy as np

from dtcwt_processor import DTCWT3DProcessor


def read_exact(stdout, size):
    """지정된 크기만큼 파이프에서 정확히 바이트를 읽어옵니다."""
    buf = bytearray(size)
    view = memoryview(buf)
    bytes_read = 0
    while bytes_read < size:
        chunk = stdout.read(size - bytes_read)
        if not chunk:
            break
        view[bytes_read:bytes_read + len(chunk)] = chunk
        bytes_read += len(chunk)
    return bytes(buf[:bytes_read])


def get_video_metadata(file_path):
    """FFmpeg를 사용하여 비디오 파일의 해상도(width, height)를 추출합니다.

    Args:
        file_path: 비디오 파일 경로.

    Returns:
        (width, height) 튜플.
    """
    probe = ffmpeg.probe(file_path)
    video_stream = next(
        stream for stream in probe['streams']
        if stream['codec_type'] == 'video'
    )
    width = int(video_stream['width'])
    height = int(video_stream['height'])
    return width, height


def read_y4m_in_chunks(file_path, width, height, chunk_size=8, overlap=4):
    """비디오를 chunk_size 프레임 단위로 읽어 Y 채널 NumPy 배열로 반환하는 제너레이터.

    시간축 아티팩트를 방지하기 위해 이전 청크의 마지막 overlap 개 프레임을
    현재 청크 앞에 덧붙여 반환합니다 (Overlap-Save 방식).

    YUV420 포맷에서 Y(휘도) 채널만 추출합니다.
    3D DT-CWT는 시각적으로 가장 민감한 Y 채널에 우선 적용하는 것이 효율적입니다.

    Args:
        file_path: 비디오 파일 경로.
        width: 비디오 너비.
        height: 비디오 높이.
        chunk_size: 한 번에 반환할 순수 프레임 수.
        overlap: 겹칠 프레임 수.

    Yields:
        (y_array, actual_frames, overlap_len) 튜플
        - y_array: (T, H, W) 형태의 float32 NumPy 배열. 값 범위 [0, 1].
                   (T = overlap_len + actual_frames)
        - actual_frames: 실제로 읽은 순수 새로운 프레임 수
        - overlap_len: 이번 청크에 앞부분에 추가된 오버랩 프레임 수
    """
    frame_bytes = int(width * height * 1.5)  # YUV420: Y=w*h, U=w*h/4, V=w*h/4
    chunk_bytes = frame_bytes * chunk_size

    process = (
        ffmpeg
        .input(file_path)
        .output('pipe:', format='rawvideo', pix_fmt='yuv420p')
        .global_args('-loglevel', 'quiet')
        .run_async(pipe_stdout=True)
    )

    prev_y_overlap = None

    try:
        while True:
            in_bytes = read_exact(process.stdout, chunk_bytes)
            if not in_bytes:
                break

            actual_frames = len(in_bytes) // frame_bytes
            if actual_frames == 0:
                break

            # Y 채널만 추출: YUV420에서 각 프레임의 첫 w*h 바이트가 Y
            raw_data = np.frombuffer(in_bytes, dtype=np.uint8)
            y_channel = raw_data.reshape(
                (actual_frames, int(height * 1.5), width)
            )[:, :height, :]

            # [0, 255] uint8 → [0, 1] float32 정규화
            y_normalized = y_channel.astype(np.float32) / 255.0

            # 오버랩 윈도우 결합
            if prev_y_overlap is not None:
                y_with_overlap = np.concatenate([prev_y_overlap, y_normalized], axis=0)
                overlap_len = len(prev_y_overlap)
            else:
                y_with_overlap = y_normalized
                overlap_len = 0
                
            # 다음 번을 위해 현재 청크의 마지막 overlap 개 프레임 저장
            prev_y_overlap = y_normalized[-overlap:]
            if len(prev_y_overlap) == 0:
                prev_y_overlap = None

            yield y_with_overlap, actual_frames, overlap_len
    finally:
        process.stdout.close()
        process.wait()


if __name__ == "__main__":
    video_path = "./videos/akiyo.y4m"
    w, h = get_video_metadata(video_path)

    # 프로세서 초기화 (임계값은 0.01 ~ 0.1 사이에서 실험적으로 조정)
    processor = DTCWT3DProcessor(threshold=0.03)

    for i, (chunk, frames, overlap) in enumerate(read_y4m_in_chunks(video_path, w, h, chunk_size=8, overlap=4)):
        processed_chunk = processor.process_chunk(chunk)
        processed_valid = processed_chunk[overlap:]
        print(f"Chunk {i} processed. Input: {chunk.shape}, Output valid shape: {processed_valid.shape}")

        if i == 2:
            break
