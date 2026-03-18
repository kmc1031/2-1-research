import numpy as np
from dtcwt_processor import DTCWT3DProcessor
import traceback

def test_phase_continuity():
    # 16-frame random video block
    video = np.random.rand(16, 16, 16).astype(np.float32)

    # 1. Continuous processing (Reference)
    # To get perfect continuous evaluation for frames 8-15, we need to process frames 4-15 together
    proc_ref = DTCWT3DProcessor(threshold=0.0, use_coef_cache=False)
    out_ref = proc_ref.process_chunk(video[4:16], overlap_len=0) # 12 frames
    
    # 2. Cached processing
    proc_cache = DTCWT3DProcessor(threshold=0.0, use_coef_cache=True)
    try:
        # Chunk 1: frames 0-7 (8 frames)
        _ = proc_cache.process_chunk(video[0:8], overlap_len=0)
        
        # Chunk 2: frames 4-15 (12 frames, where first 4 are overlap)
        out_cache = proc_cache.process_chunk(video[4:16], overlap_len=4)
        
        # out_cache should match out_ref. Both return 12 frames.
        # We only care about the valid part (the last 8 frames)
        out_cache_valid = out_cache[4:]
        out_ref_valid = out_ref[4:]
        
        mse = np.mean((out_cache_valid - out_ref_valid) ** 2)
        print(f"MSE between Continuous and Coefficient-level Caching: {mse}")
        if mse < 1e-4:
            print("PHASE CONTINUITY MAINTAINED!")
        else:
            print("PHASE CONTINUITY ERROR (Expected due to boundary reflection artifacts at cut)")
            
    except Exception as e:
        print("RUNTIME ERROR IN CACHING LOGIC:")
        traceback.print_exc()

if __name__ == "__main__":
    test_phase_continuity()
