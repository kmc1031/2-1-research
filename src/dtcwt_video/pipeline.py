"""
메인 파이프라인: Y4M 읽기 → 3D DT-CWT 전처리 → x264 인코딩.

Y/U/V 채널을 분리하여 Y 채널에만 DT-CWT 전처리를 적용하고,
원본 U/V와 함께 x264 인코더로 전달하는 완전한 파이프라인입니다.
"""

import os
import subprocess
import re
import csv
from typing import Optional, Tuple

import ffmpeg
import numpy as np

from dtcwt_video.dtcwt_processor import DTCWT3DProcessor, ProcessingContext


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


def get_scene_changes(file_path, fps, threshold=10.0):
    """FFmpeg scdet 필터를 사용하여 장면 전환 인덱스 세트를 반환합니다."""
    cmd = [
        "ffmpeg", "-y", "-i", file_path,
        "-vf", f"scdet=threshold={threshold}",
        "-f", "null", "-"
    ]
    process = subprocess.Popen(
        cmd, stderr=subprocess.PIPE, stdout=subprocess.DEVNULL, universal_newlines=True
    )
    
    scene_frames = set()
    pattern = re.compile(r"lavfi\.scdet\.time:\s*([\d\.]+)")
    
    for line in process.stderr:
        match = pattern.search(line)
        if match:
            time_sec = float(match.group(1))
            scene_frames.add(int(round(time_sec * fps)))
            
    process.wait()
    return scene_frames

def read_y4m_and_split(file_path, width, height, fps, chunk_size=8, overlap=4,
                       scene_threshold=10.0, return_scene_change: bool = False):
    """Y4M 비디오를 청크 단위로 읽어 Y/U/V 채널을 분리하여 반환합니다.

    시간축 아티팩트를 방지하기 위해 이전 청크의 마지막 overlap 개 프레임을
    현재 청크 앞에 덧붙여 반환합니다 (Overlap-Save 방식).
    만약 청크 경계에서 장면 전환(Scene Change)이 감지되면 오버랩을 무시합니다.
    
    Y 채널은 [0, 1] float32로 정규화하고,
    U/V 채널은 원본 uint8 바이트 배열 그대로 유지합니다.

    Args:
        file_path: 비디오 파일 경로.
        width: 비디오 너비.
        height: 비디오 높이.
        fps: 비디오 프레임레이트.
        chunk_size: 한 번에 반환할 순수 프레임 수.
        overlap: 겹칠 프레임 수.
        scene_threshold: 장면 전환 감지 임계값 (0~100, scdet 기본값 10).

    Yields:
        (y_array, u_array, v_array, actual_frames, overlap_len[, scene_cut]) 튜플.
        return_scene_change=True일 때 마지막 항목 scene_cut(bool)이 추가됩니다.
    """
    y_size = width * height
    uv_size = y_size // 4
    frame_bytes = y_size + (uv_size * 2)
    chunk_bytes = frame_bytes * chunk_size

    # 사전에 전체 장면 전환 프레임을 미리 파싱
    scene_frames = get_scene_changes(file_path, fps, threshold=scene_threshold)

    process = (
        ffmpeg
        .input(file_path)
        .output('pipe:', format='rawvideo', pix_fmt='yuv420p')
        .global_args('-loglevel', 'quiet')
        .run_async(pipe_stdout=True)
    )

    prev_y_overlap = None
    chunk_idx = 0
    current_frame_idx = 0

    try:
        while True:
            in_bytes = read_exact(process.stdout, chunk_bytes)
            if not in_bytes:
                break
            
            actual_frames = len(in_bytes) // frame_bytes
            if actual_frames == 0:
                break

            raw_data = np.frombuffer(in_bytes, dtype=np.uint8).reshape(
                (actual_frames, frame_bytes)
            )

            y_channel = raw_data[:, :y_size].reshape((actual_frames, height, width))
            u_channel = raw_data[:, y_size:y_size + uv_size]
            v_channel = raw_data[:, y_size + uv_size:]

            y_normalized = y_channel.astype(np.float32) / 255.0

            # 파싱된 FFmpeg 메타데이터를 기반으로 현재 청크 내에 장면 전환이 있는지 확인
            has_scene_change = False
            for f in range(current_frame_idx, current_frame_idx + actual_frames):
                if f in scene_frames:
                    has_scene_change = True
                    break

            if prev_y_overlap is not None:
                if has_scene_change:
                    print(f"  [장면 전환 감지 (Metadata)] Chunk {chunk_idx}. 오버랩 및 캐시를 초기화합니다.")
                    y_with_overlap = y_normalized
                    overlap_len = 0
                else:
                    y_with_overlap = np.concatenate([prev_y_overlap, y_normalized], axis=0)
                    overlap_len = len(prev_y_overlap)
            else:
                y_with_overlap = y_normalized
                overlap_len = 0

            # 다음 청크를 위한 오버랩 초기화 (chunk가 overlap보다 작아도 안전하게 자름)
            if overlap > 0:
                keep = min(overlap, len(y_normalized))
                prev_y_overlap = y_normalized[-keep:] if keep > 0 else None
            else:
                prev_y_overlap = None

            if return_scene_change:
                yield y_with_overlap, u_channel, v_channel, actual_frames, overlap_len, has_scene_change
            else:
                yield y_with_overlap, u_channel, v_channel, actual_frames, overlap_len
            current_frame_idx += actual_frames
            chunk_idx += 1
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


# --- 경량 컨텍스트 특성 계산 유틸리티 -----------------------------------------

def _parse_bitrate_to_kbps(bitrate: str) -> float:
    """'500k' 또는 '2M' 같은 문자열을 kbps float으로 변환합니다."""
    if isinstance(bitrate, (int, float)):
        return float(bitrate)
    s = bitrate.strip().lower()
    if s.endswith('k'):
        return float(s[:-1])
    if s.endswith('m'):
        return float(s[:-1]) * 1000.0
    if s.endswith('bps'):
        try:
            return float(s[:-3]) / 1000.0
        except ValueError:
            return 0.0
    try:
        return float(s)
    except ValueError:
        return 0.0


def estimate_noise_level(y_chunk: np.ndarray) -> float:
    """Laplacian 스타일의 고주파 잔차로 노이즈를 근사합니다."""
    # 간단한 3D Laplacian 근사: 공간/시간 1차 차분 사용
    gx = np.diff(y_chunk, axis=2, prepend=y_chunk[:, :, :1])
    gy = np.diff(y_chunk, axis=1, prepend=y_chunk[:, :1, :])
    gt = np.diff(y_chunk, axis=0, prepend=y_chunk[:1, :, :])
    residual = np.concatenate([gx, gy, gt], axis=None)
    mad = np.median(np.abs(residual))
    # Rayleigh 편차가 아닌 실수 채널 MAD 기준
    return float(mad / 0.6745)


def estimate_motion_strength(y_chunk: np.ndarray) -> float:
    """시간 축 평균 절대 차이로 모션 세기를 측정합니다."""
    if y_chunk.shape[0] < 2:
        return 0.0
    diff = np.abs(np.diff(y_chunk, axis=0))
    return float(np.mean(diff))


def estimate_edge_density(y_chunk: np.ndarray) -> float:
    """공간 그래디언트 크기 상위 퍼센타일 비율로 에지 밀도를 근사합니다."""
    gy, gx = np.gradient(y_chunk, axis=(1, 2))
    mag = np.sqrt(gx ** 2 + gy ** 2)
    thresh = np.percentile(mag, 90.0)
    if thresh <= 0:
        return 0.0
    return float((mag > thresh).mean())


def build_processing_context(y_chunk: np.ndarray,
                             target_bitrate: str,
                             chunk_index: int,
                             fps: float,
                             scene_cut: bool,
                             mode: str) -> Tuple[ProcessingContext, dict]:
    """청크 단위 컨텍스트와 로깅용 dict를 반환합니다."""
    bitrate_kbps = _parse_bitrate_to_kbps(target_bitrate)
    noise = estimate_noise_level(y_chunk)
    motion = estimate_motion_strength(y_chunk)
    edge = estimate_edge_density(y_chunk)

    ctx = ProcessingContext(
        target_bitrate_kbps=bitrate_kbps,
        noise_level=noise,
        motion_strength=motion,
        edge_density=edge,
        scene_cut=scene_cut,
        chunk_index=chunk_index,
        fps=fps,
        mode=mode,
    )
    return ctx, {
        "chunk": chunk_index,
        "bitrate_kbps": bitrate_kbps,
        "noise": noise,
        "motion": motion,
        "edge_density": edge,
        "scene_cut": scene_cut,
    }


# --- 메인 실행 파이프라인 ---
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="3D DT-CWT 비디오 전처리 및 인코딩 파이프라인")
    parser.add_argument("-i", "--input_video", default="./videos/akiyo.y4m", help="입력 Y4M 비디오 파일 경로")
    parser.add_argument("-b", "--bitrate", default="100k", help="목표 비트레이트 (예: '100k', '500k')")
    parser.add_argument("-o", "--output_video", help="출력 MP4 파일 경로 (기본값: 자동 생성)")
    parser.add_argument("-t", "--threshold", type=float, default=0.03, help="DT-CWT 임계값")
    parser.add_argument("-c", "--chunk_size", type=int, default=8, help="처리 청크 크기 (순수 프레임 수)")
    parser.add_argument("--overlap", type=int, default=4, help="시간축 아티팩트 방지를 위한 겹침 프레임 수")
    parser.add_argument("--threshold-mode", choices=["fixed", "adaptive", "rate_aware"],
                        default="adaptive", help="임계값 모드 (고정/적응형/비트레이트-인식)")
    parser.add_argument("--controller-a", type=float, default=0.35, help="rate-aware 노이즈 계수")
    parser.add_argument("--controller-b", type=float, default=0.25, help="rate-aware 비트레이트 계수")
    parser.add_argument("--controller-c", type=float, default=0.25, help="rate-aware 모션 억제 계수")
    parser.add_argument("--controller-d", type=float, default=0.25, help="rate-aware 에지 억제 계수")
    parser.add_argument("--min-multiplier", type=float, default=0.5, help="컨트롤러 최소 배율")
    parser.add_argument("--max-multiplier", type=float, default=2.5, help="컨트롤러 최대 배율")
    parser.add_argument("--log-context", action="store_true", help="청크별 컨텍스트/배율 CSV 로깅")
    parser.add_argument("--disable-rate-aware-scene-reset", action="store_true",
                        help="장면 전환 시 중립 배율로 초기화하지 않음")

    args = parser.parse_args()

    INPUT_VIDEO = args.input_video
    TARGET_BITRATE = args.bitrate
    OUTPUT_VIDEO = args.output_video if args.output_video else f"{os.path.splitext(os.path.basename(INPUT_VIDEO))[0]}_processed_{TARGET_BITRATE}.mp4"

    print(f"파이프라인을 시작합니다.")

    # 1. 메타데이터 추출
    w, h, fps = get_video_metadata(INPUT_VIDEO)
    print(f"해상도: {w}x{h}, FPS: {fps:.2f}, 목표 비트레이트: {TARGET_BITRATE}")

    # 2. 인코더 서브프로세스 열기
    encoder_process = create_x264_encoder(OUTPUT_VIDEO, w, h, fps, TARGET_BITRATE)

    # 3. 3D DT-CWT 프로세서 초기화
    processor = DTCWT3DProcessor(
        threshold=args.threshold,
        nlevels=1,
        adaptive_threshold=args.threshold_mode != "fixed",
        threshold_mode=args.threshold_mode,
        controller_a=args.controller_a,
        controller_b=args.controller_b,
        controller_c=args.controller_c,
        controller_d=args.controller_d,
        min_multiplier=args.min_multiplier,
        max_multiplier=args.max_multiplier,
        disable_rate_aware_scene_reset=args.disable_rate_aware_scene_reset,
    )

    # 3-1. 로깅 준비
    log_writer = None
    log_file = None
    if args.log_context:
        os.makedirs("logs", exist_ok=True)
        base = os.path.splitext(os.path.basename(INPUT_VIDEO))[0]
        log_file = os.path.join("logs", f"context_log_{base}_{TARGET_BITRATE}.csv")
        log_fp = open(log_file, "w", newline="", encoding="utf-8")
        log_writer = csv.writer(log_fp)
        log_writer.writerow(["chunk", "overlap", "bitrate_kbps", "noise",
                             "motion", "edge_density", "scene_cut",
                             "threshold_mode", "multiplier"])

    print("스트리밍 전처리 및 인코딩 진행 중...")

    # 4. 청크 단위 스트리밍 처리
    chunk_idx = 0
    for y_array, u_np, v_np, frames_in_chunk, overlap_len, scene_cut in read_y4m_and_split(
        INPUT_VIDEO, w, h, fps=fps, chunk_size=args.chunk_size,
        overlap=args.overlap, return_scene_change=True
    ):
        ctx, ctx_log = build_processing_context(
            y_array, TARGET_BITRATE, chunk_idx, fps, scene_cut, args.threshold_mode
        )
        processor.set_context(ctx)

        # DT-CWT 전처리 (Y 채널만, 오버랩 포함된 임시 T 프레임)
        processed_y = processor.process_chunk(y_array, overlap_len=overlap_len)

        # 겹쳤던 부분(앞쪽 overlap_len 프레임)을 잘라내어 순수 새로운 프레임만 추출
        processed_y_valid = processed_y[overlap_len:overlap_len + frames_in_chunk]

        # [0, 1] float → [0, 255] uint8 변환
        valid_frames = processed_y_valid.shape[0]
        processed_y_uint8 = (processed_y_valid * 255.0).clip(0, 255).astype(np.uint8)
        processed_y_flat = processed_y_uint8.reshape((valid_frames, -1))
        
        # U/V 채널은 전처리 생략: 단일 파이프라인에서는 속도/단순화 우선.
        # (run_rd_curve.py에서는 process_chroma로 U/V도 전처리하여 비교함)
        # 인코더로 프레임 단위 쓰기 (Y + U + V)
        for f in range(valid_frames):
            encoder_process.stdin.write(processed_y_flat[f].tobytes())
            encoder_process.stdin.write(u_np[f].tobytes())
            encoder_process.stdin.write(v_np[f].tobytes())

        if log_writer:
            log_writer.writerow([
                chunk_idx,
                overlap_len,
                ctx_log["bitrate_kbps"],
                ctx_log["noise"],
                ctx_log["motion"],
                ctx_log["edge_density"],
                ctx_log["scene_cut"],
                args.threshold_mode,
                processor.compute_controller_multiplier(),
            ])

        chunk_idx += 1
        if chunk_idx % 5 == 0:
            print(f"  ... {chunk_idx * args.chunk_size} 프레임 처리 완료")

    if log_writer:
        log_fp.close()
        print(f"컨텍스트 로그 저장: {os.path.abspath(log_file)}")

    # 5. 파이프 종료 및 인코딩 마무리
    encoder_process.stdin.close()
    encoder_process.wait()

    print(f"완료! 인코딩된 파일이 저장되었습니다: {OUTPUT_VIDEO}")
