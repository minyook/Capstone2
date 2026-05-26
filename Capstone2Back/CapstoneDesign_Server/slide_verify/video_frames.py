"""영상에서 샘플 프레임 추출."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def extract_frames(video_path: Path, out_dir: Path, fps: float = 0.5) -> list[Path]:
    """초당 fps장 프레임을 jpg로 저장하고 경로 리스트를 반환합니다."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("frame_*.jpg"):
        old.unlink(missing_ok=True)

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError(
            "FFmpeg가 PATH에 없습니다. https://ffmpeg.org 에서 설치 후 다시 시도하세요."
        )

    pattern = out_dir / "frame_%05d.jpg"
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-i",
            str(video_path.resolve()),
            "-vf",
            f"fps={fps}",
            "-q:v",
            "2",
            str(pattern),
        ],
        check=True,
        capture_output=True,
        timeout=600,
    )

    frames = sorted(out_dir.glob("frame_*.jpg"))
    if not frames:
        raise RuntimeError("영상에서 프레임을 추출하지 못했습니다.")
    return frames
