"""
비디오 품질 평가 모듈: PSNR, SSIM, VMAF.

FFmpeg의 내장 필터를 활용하여 참조(reference) 영상 대비
왜곡(distorted) 영상의 품질 지표를 측정합니다.
"""

import json
import os
import re
import subprocess


def evaluate_video_quality(ref_video, dist_video):
    """참조 영상 대비 왜곡 영상의 PSNR, SSIM, VMAF를 측정합니다.

    Args:
        ref_video: 원본(참조) 비디오 파일 경로.
        dist_video: 인코딩된(왜곡) 비디오 파일 경로.

    Returns:
        (psnr, ssim, vmaf) 튜플. 측정 실패 시 해당 값은 None.
    """
    if not os.path.exists(ref_video) or not os.path.exists(dist_video):
        return None, None, None

    # 1. PSNR 및 SSIM 측정
    cmd_psnr_ssim = [
        "ffmpeg", "-y", "-i", dist_video, "-i", ref_video,
        "-filter_complex",
        "[0:v]split=2[dist1][dist2];"
        "[1:v]split=2[ref1][ref2];"
        "[dist1][ref1]psnr;"
        "[dist2][ref2]ssim",
        "-f", "null", "-"
    ]
    result = subprocess.run(
        cmd_psnr_ssim, stderr=subprocess.PIPE, text=True, encoding='utf-8'
    )

    psnr_match = re.search(r'PSNR.*average:([0-9.]+)', result.stderr)
    ssim_match = re.search(r'SSIM.*All:([0-9.]+)', result.stderr)

    psnr_val = float(psnr_match.group(1)) if psnr_match else None
    ssim_val = float(ssim_match.group(1)) if ssim_match else None

    # 2. VMAF 측정 (JSON 로그 생성 후 파싱)
    temp_vmaf_log = f"temp_vmaf_{os.path.basename(dist_video)}.json"
    cmd_vmaf = [
        "ffmpeg", "-y", "-i", dist_video, "-i", ref_video,
        "-lavfi", f"libvmaf=log_path={temp_vmaf_log}:log_fmt=json",
        "-f", "null", "-"
    ]
    subprocess.run(cmd_vmaf, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    vmaf_val = None
    if os.path.exists(temp_vmaf_log):
        try:
            with open(temp_vmaf_log, 'r', encoding='utf-8') as f:
                vmaf_data = json.load(f)
                if "pooled_metrics" in vmaf_data:
                    vmaf_val = vmaf_data["pooled_metrics"]["vmaf"]["mean"]
        except (json.JSONDecodeError, KeyError):
            pass
        finally:
            os.remove(temp_vmaf_log)

    return psnr_val, ssim_val, vmaf_val