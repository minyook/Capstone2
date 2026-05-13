from __future__ import annotations

import os
import json
from pathlib import Path
from datetime import datetime
import numpy as np

try:
    import cv2  # type: ignore
    import mediapipe as mp  # type: ignore
    from mediapipe.tasks import python  # type: ignore
    from mediapipe.tasks.python import vision  # type: ignore
    _MP_AVAILABLE = True
except Exception:
    cv2 = mp = python = vision = None
    _MP_AVAILABLE = False

if _MP_AVAILABLE:
    FaceLandmarker = vision.FaceLandmarker
    FaceLandmarkerOptions = vision.FaceLandmarkerOptions
    VisionRunningMode = vision.RunningMode
else:
    FaceLandmarker = FaceLandmarkerOptions = VisionRunningMode = None

face_landmarker_instance = None
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "face_landmarker.task"

def setup_face_landmarker():
    global face_landmarker_instance
    if not _MP_AVAILABLE: return None
    if face_landmarker_instance: return face_landmarker_instance
    model_path_str = os.path.abspath(str(MODEL_PATH))
    try:
        with open(model_path_str, 'rb') as f:
            model_data = f.read()
        base_options = python.BaseOptions(model_asset_buffer=model_data)
        options = FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=VisionRunningMode.IMAGE,
            num_faces=1,
            output_face_blendshapes=True
        )
        face_landmarker_instance = FaceLandmarker.create_from_options(options)
        return face_landmarker_instance
    except Exception: return None

def _process_face_data(results) -> dict:
    """코와 눈의 상대적 위치를 이용한 초정밀 시선 분석"""
    if not results.face_blendshapes or not results.face_landmarks:
        return {}
        
    # 1. Blendshapes (표정용)
    cats = {c.category_name: c.score for c in results.face_blendshapes[0]}
    def pick(n): return cats.get(n, 0)

    # 2. 초정밀 시선 분석 (Nose-to-Eye Center Ratio)
    # Landmark Indices: 코끝(4), 왼쪽 눈 외곽(33), 오른쪽 눈 외곽(263)
    lm = results.face_landmarks[0]
    nose = lm[4]
    l_eye = lm[33]
    r_eye = lm[263]
    
    # 양 눈의 중심점 X좌표
    eye_center_x = (l_eye.x + r_eye.x) / 2
    # 눈 사이의 수평 거리 (기준값)
    eye_dist = abs(r_eye.x - l_eye.x) + 1e-6
    
    # 코가 중심에서 벗어난 정도 (Positive = Right, Negative = Left)
    # 화면 기준: 코가 눈 중심보다 오른쪽에 있으면(x값이 크면) 우측 응시
    gaze_ratio = (nose.x - eye_center_x) / eye_dist
    
    # 보정 가중치 (고개를 돌릴 때 수치가 더 확실히 변하도록)
    combined_gaze_h = gaze_ratio * 5.0 

    # 3. 기타 감정 및 표정
    smile = (pick('mouthSmileLeft') + pick('mouthSmileRight')) / 2
    frown = (pick('mouthFrownLeft') + pick('mouthFrownRight')) / 2
    brow_up = (pick('browInnerUp') + pick('browOuterUpLeft') + pick('browOuterUpRight')) / 3
    jaw_open = pick('jawOpen')

    emotions = {
        "smile": float(smile),
        "angry": float(pick('browDownLeft') * 0.5 + pick('browDownRight') * 0.5 + frown * 0.5),
        "blank": float(max(0, 1.0 - (smile + frown + brow_up + jaw_open) * 2.0)),
        "anxious": float(pick('browInnerUp') * 0.5 + pick('eyeSquintLeft') * 0.5)
    }
    
    return {
        "gaze_h": float(combined_gaze_h),
        "gaze_v": float((pick('eyeLookUpLeft') + pick('eyeLookUpRight')) / 2 - (pick('eyeLookDownLeft') + pick('eyeLookDownRight')) / 2),
        "smile": float(smile),
        "emotions": emotions,
        "all_blendshapes": cats
    }

def analyze_image(image_input: str | np.ndarray) -> dict:
    landmarker = setup_face_landmarker()
    if not landmarker: return {"error": "모델 로드 실패"}
    try:
        if isinstance(image_input, str):
            image_bgr = cv2.imread(image_input)
            if image_bgr is None: return {"error": "이미지 읽기 실패"}
        else:
            image_bgr = image_input
            
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        results = landmarker.detect(mp_image)
        
        if results.face_blendshapes and results.face_landmarks:
            return _process_face_data(results)
        return {"error": "얼굴 미검출"}
    except Exception as e:
        return {"error": str(e)}

def save_face_data(all_vision_results: list, frame_rate: int, job_id: str = "default"):
    time_series_face = {}
    for i, res in enumerate(all_vision_results):
        seconds = i / frame_rate
        face = res.face
        state = "정면 응시함"
        if not face.has_face: state = "얼굴 미검출"
        elif face.gaze_h > 0.1: state = "우측 응시 (PPT)"
        elif face.gaze_h < -0.1: state = "좌측 응시 (PPT)"
        
        time_series_face[f"{seconds:.2f}"] = {
            "info": {"main_state": state},
            "gaze_h": face.gaze_h,
            "emotions": face.emotions if hasattr(face, 'emotions') else {}
        }
    
    out_dir = Path("analysis_json/MediaPipe_json")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{job_id}_face.json", 'w', encoding='utf-8') as f:
        json.dump(time_series_face, f, indent=4, ensure_ascii=False)
