import os
import time
import psutil
import torch
import numpy as np
from dtcwt_processor import DTCWT3DProcessor
from main_pipeline import read_y4m_and_split, get_video_metadata

def profile_pipeline(video_path, use_cache, title):
    print(f"\n==========================================")
    print(f" 프로파일링 시작: {title}")
    print(f" (Coefficient Caching: {use_cache})")
    print(f"==========================================")
    
    w, h, fps = get_video_metadata(video_path)
    
    proc = DTCWT3DProcessor(threshold=0.03, use_coef_cache=use_cache)
    
    process = psutil.Process()
    vram_peak = 0
    sys_ram_peak = 0
    
    start_time = time.time()
    total_frames = 0
    chunk_times = []
    
    # 40 프레임 정도만 테스트
    max_frames = 40
    
    for y_array, u_np, v_np, frames_in_chunk, overlap_len in read_y4m_and_split(
        video_path, w, h, fps=fps, chunk_size=8, overlap=4
    ):
        if total_frames >= max_frames:
            break
            
        chunk_start = time.time()
        
        # Processor
        _ = proc.process_chunk(y_array, overlap_len=overlap_len)
        
        chunk_end = time.time()
        chunk_times.append(chunk_end - chunk_start)
        
        # 메모리 측정
        sys_ram_peak = max(sys_ram_peak, process.memory_info().rss)
        if torch.cuda.is_available():
            vram_peak = max(vram_peak, torch.cuda.max_memory_allocated() + torch.cuda.max_memory_reserved())
            
        total_frames += frames_in_chunk

    total_time = time.time() - start_time
    avg_fps = total_frames / total_time
    avg_chunk = np.mean(chunk_times)
    
    print(f" -> 처리 완료 프레임: {total_frames}")
    # time
    print(f" -> 소요 시간: {total_time:.2f} 초")
    print(f" -> 평균 속도(FPS): {avg_fps:.2f} FPS")
    print(f" -> 평균 청크 연산 시간: {avg_chunk:.3f} 초")
    
    # mem
    print(f" -> 최대 System RAM: {sys_ram_peak / (1024**2):.2f} MB")
    if torch.cuda.is_available():
        print(f" -> 최대 VRAM (PyTorch): {vram_peak / (1024**2):.2f} MB")
        
    return {
        "fps": avg_fps,
        "sys_mb": sys_ram_peak / (1024**2),
        "vram_mb": vram_peak / (1024**2) if torch.cuda.is_available() else 0,
        "chunk_time": avg_chunk
    }

if __name__ == "__main__":
    test_video = "./videos/akiyo.y4m"
    if not os.path.exists(test_video):
        print(f"오류: 테스트 비디오가 없습니다 -> {test_video}")
        exit(1)
        
    res_no_cache = profile_pipeline(test_video, use_cache=False, title="Baseline (No Cache, 12 frames/chunk)")
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
    res_cache = profile_pipeline(test_video, use_cache=True, title="Proposed (Coefficient Cache, 8 frames/chunk)")
    
    print("\n\n======== [최종 프로파일링 리포트] ========")
    print(f"[성능 프로파일링]")
    print(f" - 청크 당 연산 속도 개선: {res_no_cache['chunk_time']:.3f}s -> {res_cache['chunk_time']:.3f}s")
    print(f" - 전체 처리 FPS 개선: {res_no_cache['fps']:.2f} -> {res_cache['fps']:.2f} FPS (+{((res_cache['fps']/res_no_cache['fps'])-1)*100:.1f}%)")
    print(f"\n[메모리 프로파일링 (VRAM GC 효과)]")
    print(f" - 최대 시스템 RAM 사용량: {res_no_cache['sys_mb']:.2f}MB vs {res_cache['sys_mb']:.2f}MB")
    print(f" - 최대 VRAM 사용량: {res_no_cache['vram_mb']:.2f}MB vs {res_cache['vram_mb']:.2f}MB (+{res_cache['vram_mb']-res_no_cache['vram_mb']:.1f} MB Overhead for GC Offload context)")
    print("==========================================")
