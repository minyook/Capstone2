"""슬라이드 일치 검증 오케스트레이션."""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path

from slide_verify.ppt_render import render_ppt_to_images
from slide_verify.verify import verify_ppt_video
from slide_verify.video_frames import extract_frames

BASE_DIR = Path(__file__).resolve().parent.parent
CACHE_DIR = BASE_DIR / "analysis_json" / "slide_verify_cache"
# 1fps는 10분 영상 기준 600프레임 → 매칭만 수 분. 0.5fps + 상한으로 체감 속도 개선.
FRAME_FPS = 0.5
MAX_VERIFY_FRAMES = 90


def run_slide_verify(
    ppt_path: Path,
    video_path: Path,
    video_type: str = "fullscreen",
) -> dict:
    job_id = uuid.uuid4().hex[:12]
    slide_dir = CACHE_DIR / job_id / "slides"
    frame_dir = CACHE_DIR / job_id / "frames"

    slide_images, render_mode = render_ppt_to_images(ppt_path, slide_dir)
    if not slide_images:
        raise RuntimeError("PPT 슬라이드 이미지 생성에 실패했습니다.")

    frame_paths = extract_frames(video_path, frame_dir, fps=FRAME_FPS)
    if not frame_paths:
        raise RuntimeError("영상 프레임 추출에 실패했습니다.")
    if len(frame_paths) > MAX_VERIFY_FRAMES:
        step = max(1, len(frame_paths) // MAX_VERIFY_FRAMES)
        head = frame_paths[: min(4, len(frame_paths))]
        sampled = frame_paths[::step]
        merged: list[Path] = []
        for p in head + sampled:
            if p not in merged:
                merged.append(p)
        frame_paths = merged[:MAX_VERIFY_FRAMES]

    result = verify_ppt_video(
        slide_images=slide_images,
        frame_paths=frame_paths,
        frame_interval_sec=1.0 / FRAME_FPS,
        video_type=video_type,
    )
    result["job_id"] = job_id
    result["video_type"] = video_type
    result["render_mode"] = render_mode
    # 캐시 정리 (디스크 누적 방지)
    try:
        shutil.rmtree(CACHE_DIR / job_id, ignore_errors=True)
    except OSError:
        pass
    return result
