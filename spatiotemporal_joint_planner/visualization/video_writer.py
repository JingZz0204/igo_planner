from __future__ import annotations

import os
import shutil
import subprocess
from typing import Optional


def prepare_mp4_frame_dir(mp4_path: str) -> tuple[str, str]:
    output_path = os.path.abspath(mp4_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    root, _ = os.path.splitext(output_path)
    base_frame_dir = root + "_frames"
    frame_dir = base_frame_dir
    suffix = 1
    while os.path.isdir(frame_dir) and any(
        name.startswith("frame_") and name.endswith(".png") for name in os.listdir(frame_dir)
    ):
        frame_dir = f"{base_frame_dir}_{suffix:03d}"
        suffix += 1
    os.makedirs(frame_dir, exist_ok=True)
    return output_path, frame_dir


def mp4_frame_path(frame_dir: str, frame_index: int) -> str:
    return os.path.join(frame_dir, f"frame_{frame_index:05d}.png")


def find_ffmpeg_executable() -> Optional[str]:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        return ffmpeg
    try:
        import imageio_ffmpeg
    except ModuleNotFoundError:
        return None
    return imageio_ffmpeg.get_ffmpeg_exe()


def encode_mp4_from_frames(frame_dir: str, output_path: str, fps: float) -> bool:
    first_frame = mp4_frame_path(frame_dir, 0)
    if not os.path.exists(first_frame):
        print(f"MP4 skipped: no frames were written to {frame_dir}")
        return False

    ffmpeg = find_ffmpeg_executable()
    if ffmpeg is None:
        print(f"MP4 skipped: ffmpeg/imageio-ffmpeg unavailable. Frames are kept in {frame_dir}")
        return False

    command = [
        ffmpeg,
        "-y",
        "-framerate",
        f"{max(float(fps), 1.0):g}",
        "-i",
        os.path.join(frame_dir, "frame_%05d.png"),
        "-vf",
        "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-pix_fmt",
        "yuv420p",
        "-vcodec",
        "libx264",
        output_path,
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        tail = "\n".join(stderr.splitlines()[-8:])
        print(f"MP4 encoding failed. Frames are kept in {frame_dir}")
        if tail:
            print(tail)
        return False
    print(f"MP4 saved: {output_path}")
    return True
