import subprocess
from pathlib import Path

import cv2
import numpy as np
from processing.vision_dto import FrameVisionResult, MediaPipeFaceResult
from processing.face_analyzer import analyze_image
from processing.gesture_analyzer import analyze_frame_yolo_pose

# === 기존 extract_audio 로직 ===
def extract_audio(video_path: Path, output_audio_path: Path) -> Path:
    print(f"   > [1/6] 오디오 트랙 추출 중...")
    try:
        subprocess.run(['ffmpeg', '-i', str(video_path), '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', str(output_audio_path)], check=True, capture_output=True, text=True)
        print(f"   > [1/6] ✅ 오디오 추출 완료: {output_audio_path.name}")
        return output_audio_path
    except subprocess.CalledProcessError as e:
        raise Exception("FFmpeg 오디오 추출 실패")

# === 기존 extract_all_frames 로직 ===
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

def analyze_frame_face(image_input: str | np.ndarray) -> MediaPipeFaceResult:
    """
    image_input: 파일 경로(str) 또는 ROI 이미지(np.ndarray)
    """
    f = analyze_image(image_input)

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
        emotions=f.get("emotions", {}),
        all_blendshapes=f.get("all_blendshapes", {}),
        error=None,
    )


def analyze_frame_vision(frame_path: str, time_s: float) -> FrameVisionResult:
    """
    2단계 분석 전략 (ROI 적용):
    """
    # 1. YOLO로 사람 및 포즈 감지
    yolo = analyze_frame_yolo_pose(frame_path)

    face = None
    log_prefix = f"   > [Vision Debug {time_s:.1f}s]"

    if yolo.has_person and yolo.person_bbox:
        try:
            img = cv2.imread(frame_path)
            if img is not None:
                h, w = img.shape[:2]
                x1, y1, x2, y2 = map(int, yolo.person_bbox)

                # 상체/머리 영역 계산
                target_x1, target_y1, target_x2, target_y2 = x1, y1, x2, y2

                if yolo.keypoints and len(yolo.keypoints) > 5:
                    kp = np.array(yolo.keypoints)
                    head_kp = kp[0:5] # nose, eyes, ears
                    # 0,0 인 포인트 제외 (YOLOv8-pose는 미검출 시 0,0)
                    valid_kp = head_kp[np.all(head_kp > 0, axis=1)]

                    if len(valid_kp) >= 2: # 최소 2개의 포인트는 있어야 머리 방향 추정 가능
                        hx1, hy1 = np.min(valid_kp, axis=0)
                        hx2, hy2 = np.max(valid_kp, axis=0)

                        # 머리 크기에 비례한 넉넉한 여유 공간 (작은 인물일수록 더 넓게)
                        base_w = (hx2 - hx1) if (hx2 - hx1) > 5 else (x2 - x1) * 0.1
                        base_h = (hy2 - hy1) if (hy2 - hy1) > 5 else (y2 - y1) * 0.1

                        target_x1 = max(0, int(hx1 - base_w * 3.5))
                        target_y1 = max(0, int(hy1 - base_h * 4.5))
                        target_x2 = min(w, int(hx2 + base_w * 3.5))
                        target_y2 = min(h, int(hy2 + base_h * 4.5))
                    else:
                        # 키포인트 부족 시 상체 영역 사용 (박스 상단 50%)
                        target_y2 = y1 + int((y2 - y1) * 0.5)
                else:
                    target_y2 = y1 + int((y2 - y1) * 0.5)

                # ROI 크롭 및 분석
                if (target_x2 - target_x1) > 10 and (target_y2 - target_y1) > 10:
                    # 1. 정방형(Square) ROI로 보정 (MediaPipe 인식률 향상)
                    rw = target_x2 - target_x1
                    rh = target_y2 - target_y1
                    side = max(rw, rh)
                    
                    cx, cy = (target_x1 + target_x2) // 2, (target_y1 + target_y2) // 2
                    target_x1 = max(0, cx - side // 2)
                    target_y1 = max(0, cy - side // 2)
                    target_x2 = min(w, cx + side // 2)
                    target_y2 = min(h, cy + side // 2)

                    roi_img = img[target_y1:target_y2, target_x1:target_x2]

                    # 2. 업스케일링 및 이미지 개선 (Sharpening)
                    roi_h, roi_w = roi_img.shape[:2]
                    if roi_w < 480 or roi_h < 480: # 더 높은 해상도 확보
                        scale = max(480/roi_w, 480/roi_h)
                        roi_img = cv2.resize(roi_img, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)
                        
                        # 약간의 선명도 개선
                        kernel = np.array([[0, -0.5, 0], [-0.5, 3, -0.5], [0, -0.5, 0]])
                        roi_img = cv2.filter2D(roi_img, -1, kernel)

                    face = analyze_frame_face(roi_img)
                    
                    if not face.has_face:
                        # 1차 실패 시: 박스 상단 60% 전체를 정방형으로 크롭하여 재시도
                        alt_y2 = y1 + int((y2 - y1) * 0.6)
                        roi_img_alt = img[y1:alt_y2, x1:x2]
                        ah, aw = roi_img_alt.shape[:2]
                        
                        s = max(480/aw, 480/ah)
                        roi_img_alt = cv2.resize(roi_img_alt, (0,0), fx=s, fy=s, interpolation=cv2.INTER_LANCZOS4)
                        face_alt = analyze_frame_face(roi_img_alt)
                        if face_alt.has_face:
                            face = face_alt
        except Exception as e:
            print(f"{log_prefix} ROI 처리 중 예외 발생: {e}")

    # 최종적으로 얼굴이 안 나오면 원본 전체 시도
    if face is None or not face.has_face:
        face = analyze_frame_face(frame_path)

    # 10프레임마다 상태 출력 (너무 자주 출력하면 로그가 지저분하므로)
    if int(time_s * 5) % 10 == 0:
        status = "✅ 감지됨" if face.has_face else "❌ 미검출"
        method = "ROI" if (yolo.has_person and face.has_face) else "전체화면"
        print(f"{log_prefix} 사람:{'O' if yolo.has_person else 'X'} | 얼굴:{status} ({method})")

    return FrameVisionResult(time=time_s, yolo=yolo, face=face)
