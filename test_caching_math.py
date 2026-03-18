import torch
import numpy as np
from dtcwt_cuda import CudaDTCWT3DProcessor

def main():
    proc = CudaDTCWT3DProcessor(nlevels=1, adaptive_threshold=False, threshold=0.0)

    # 연속된 12 프레임 비디오 (임의)
    video = np.random.rand(12, 16, 16).astype(np.float32)

    # 1. Continuous (Reference)
    cube_ref = video.transpose(1, 2, 0)
    Yl_ref, Yh_ref = proc.forward(cube_ref)
    
    out_ref = proc.inverse(Yl_ref, Yh_ref)
    out_ref_valid = out_ref[:, :, 8:12]

    # 2. Coefficient Caching Method (Chunked)
    cube_c1 = video[:8].transpose(1, 2, 0)
    Yl_1, Yh_1 = proc.forward(cube_c1)
    
    # Cache overlap (last 4 frames of chunk 1 -> in wavelet domain it's last 2 frames)
    cached_Yl = Yl_1[:, :, -2:]
    cached_Yh = tuple([h[:, :, -2:, :] for h in Yh_1])

    cube_c2 = video[8:12].transpose(1, 2, 0)
    Yl_2, Yh_2 = proc.forward(cube_c2)

    # Concatenate cached (from frames 4 to 7) and new (frames 8 to 11)
    Yl_concat = torch.cat([cached_Yl, Yl_2], dim=2)
    Yh_concat = []
    for h_old, h_new in zip(cached_Yh, Yh_2):
        Yh_concat.append(torch.cat([h_old, h_new], dim=2))
    Yh_concat = tuple(Yh_concat)

    # Inverse
    out_concat = proc.inverse(Yl_concat, Yh_concat)
    
    # out_concat represents frames 4 to 11 (8 frames). We want 8 to 11 (the last 4 frames).
    out_valid = out_concat[:, :, 4:8]

    # Compare
    mse = np.mean((out_valid - out_ref_valid) ** 2)
    print(f"MSE between Continuous and Naive Coefficient Concat: {mse}")
    if mse < 1e-5:
        print("PHASE CONTINUITY MAINTAINED!")
    else:
        print("PHASE CONTINUITY BROKEN!")

if __name__ == "__main__":
    main()
