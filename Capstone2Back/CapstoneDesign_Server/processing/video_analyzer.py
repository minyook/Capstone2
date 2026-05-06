import subprocess
from pathlib import Path

from processing.vision_dto import FrameVisionResult, MediaPipeFaceResult
from processing.face_analyzer import analyze_image
from processing.gesture_analyzer import analyze_frame_yolo_pose

"""
MediaPipe는 FaceLandmarker(Tasks) 방식으로 통일합니다.
- 구현: processing/face_analyzer.py
- video_analyzer.py에서는 해당 모듈을 호출해 DTO로만 변환
"""

# === 기존 extract_audio 로직 (수정 없음) ===
def extract_audio(video_path: Path, output_audio_path: Path) -> Path:
    print(f"   > [1/6] 오디오 트랙 추출 중...")
    try:
        subprocess.run(['ffmpeg', '-i', str(video_path), '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', str(output_audio_path)], check=True, capture_output=True, text=True)
        print(f"   > [1/6] ✅ 오디오 추출 완료: {output_audio_path.name}")
        return output_audio_path
    except subprocess.CalledProcessError as e:
        raise Exception("FFmpeg 오디오 추출 실패")

# === 기존 extract_all_frames 로직 (수정 없음) ===
def extract_all_frames(video_path: Path, output_dir: Path, fps: int) -> list[Path]:
    print(f"   > [2/6] 비디오 프레임 추출 중... (초당 {fps} 프레임)")
    output_pattern = output_dir / "frame-%04d.jpg"
    try:
        subprocess.run(['ffmpeg', '-i', str(video_path), '-vf', f'fps={fps}', str(output_pattern)], check=True, capture_output=True, text=True) 
    except subprocess.CalledProcessError as e:
        raise Exception("FFmpeg 프레임 추출 실패")
    frames = sorted([f for f in output_dir.glob('*.jpg')])
    print(f"   > [2/6] ✅ {len(frames)}개 프레임 추출 완료.")
    return frames

def analyze_frame_face(frame_path: str) -> MediaPipeFaceResult:
    f = analyze_image(frame_path)

    err = f.get("error")
    if err is not None:
        return MediaPipeFaceResult(has_face=False, error=str(err))

    return MediaPipeFaceResult(
        has_face=True,
        smile=float(f.get("smile", 0.0)),
        frown=float(f.get("frown", 0.0)),
        brow_up=float(f.get("brow_up", 0.0)),
        brow_down=float(f.get("brow_down", 0.0)),
        jaw_open=float(f.get("jaw_open", 0.0)),
        mouth_open=float(f.get("mouth_open", 0.0)),
        squint=float(f.get("squint", 0.0)),
        gaze_h=float(f.get("gaze_h", 0.0)),
        gaze_v=float(f.get("gaze_v", 0.0)),
        # 🌟 추가된 부분: MediaPipe에서 추출한 52개 상세 좌표를 DTO에 매핑
        all_blendshapes=f.get("all_blendshapes", {}),
        error=None,
    )


def analyze_frame_vision(frame_path: str, time_s: float) -> FrameVisionResult:
    """
    역할 분리된 프레임 분석:
    - YOLO: 사람/포즈 가시성 (gesture_analyzer.py 호출)
    - MediaPipe: 얼굴/표정/시선 (face_analyzer.py 호출)
    """
    yolo = analyze_frame_yolo_pose(frame_path)
    face = analyze_frame_face(frame_path)
    return FrameVisionResult(time=time_s, yolo=yolo, face=face)