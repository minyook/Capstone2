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
    cv2 = None  # type: ignore
    mp = None  # type: ignore
    python = None  # type: ignore
    vision = None  # type: ignore
    _MP_AVAILABLE = False

# MediaPipe 설정
if _MP_AVAILABLE:
    FaceLandmarker = vision.FaceLandmarker
    FaceLandmarkerOptions = vision.FaceLandmarkerOptions
    VisionRunningMode = vision.RunningMode
else:
    FaceLandmarker = None
    FaceLandmarkerOptions = None
    VisionRunningMode = None

face_landmarker_instance = None

# 현재 파일 위치를 기준으로 절대 경로 재구성
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "face_landmarker.task"

# 🌟 참고: 이제 개별 저장은 하지 않고 상위 모듈(task_manager)에서 통합 저장합니다.
# 하지만 폴더 경로는 구조 유지를 위해 남겨둡니다.
JSON_SAVE_DIR = Path(__file__).resolve().parent / "MediaPipe_json"

def setup_face_landmarker():
    """MediaPipe 모델을 로드합니다."""
    global face_landmarker_instance
    if not _MP_AVAILABLE:
        return None
    if face_landmarker_instance:
        return face_landmarker_instance

    model_path_str = os.path.abspath(str(MODEL_PATH))
    print(f"\n   > [1/5] AI 모델 로드 시도: {model_path_str}")
    
    if not os.path.exists(model_path_str):
        print(f"❌ 파일을 찾을 수 없습니다: {model_path_str}")
        raise FileNotFoundError(f"모델 파일이 없습니다.")

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
        print("   > [1/5] ✅ MediaPipe 모델 로드 완료.")
        return face_landmarker_instance
    except Exception as e:
        print(f"❌ 로드 실패: {e}")
        raise

def _process_blendshapes(blendshapes: list, image_path: str = None) -> dict:
    """얼굴 특징 수치를 계산하여 반환합니다. 원천 데이터는 결과 딕셔너리에 포함됩니다."""
    if not blendshapes:
        return {}
        
    # 52개의 모든 카테고리와 점수를 추출
    cats = {c.category_name: c.score for c in blendshapes[0]}
    
    def pick(n):
        return cats.get(n, 0)

    # 시선, 웃음, 찡그림 등 주요 지표 계산
    gaze_h = ((pick('eyeLookOutLeft') - pick('eyeLookInLeft')) + (pick('eyeLookInRight') - pick('eyeLookOutRight'))) / 2
    gaze_v = ((pick('eyeLookUpLeft') - pick('eyeLookDownLeft')) + (pick('eyeLookUpRight') - pick('eyeLookDownRight'))) / 2
    smile = (pick('mouthSmileLeft') + pick('mouthSmileRight')) / 2
    frown = (pick('mouthFrownLeft') + pick('mouthFrownRight')) / 2
    brow_down = (pick('browDownLeft') + pick('browDownRight')) / 2
    jaw_open = pick('jawOpen')
    
    brow_up = (pick('browInnerUp') + pick('browOuterUpLeft') + pick('browOuterUpRight')) / 3
    mouth_open = pick('mouthOpen')
    squint = (pick('eyeSquintLeft') + pick('eyeSquintRight')) / 2
    
    # 🌟 핵심: 모든 좌표 데이터(cats)를 'all_blendshapes'라는 키로 반환합니다.
    # 이를 통해 task_manager에서 이 데이터를 모아 하나의 통합 JSON을 만들 수 있습니다.
    return {
        "gaze_h": gaze_h,
        "gaze_v": gaze_v,
        "smile": smile,
        "frown": frown,
        "brow_down": brow_down,
        "jaw_open": jaw_open,
        "brow_up": brow_up,
        "mouth_open": mouth_open,
        "squint": squint,
        "all_blendshapes": cats  # 통합 저장을 위한 원천 데이터 전송
    }

def analyze_image(image_input: str | np.ndarray) -> dict:
    """프레임을 분석하여 표정 및 시선 데이터를 반환합니다.
    image_input: 이미지 파일 경로(str) 또는 로드된 이미지(np.ndarray)
    """
    landmarker = setup_face_landmarker()
    if not landmarker:
        if not _MP_AVAILABLE:
            return {"error": "mediapipe 미설치"}
        return {"error": "모델 미로드"}
    
    try:
        if cv2 is None or mp is None:
            return {"error": "라이브러리 초기화 실패"}
            
        if isinstance(image_input, str):
            image_bgr = cv2.imread(image_input)
            if image_bgr is None:
                return {"error": "이미지 읽기 실패"}
        else:
            image_bgr = image_input
        
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        results = landmarker.detect(mp_image)
        
        if results.face_blendshapes:
            # 이미지 경로가 없는 경우(ROI인 경우) None 전달
            img_path = image_input if isinstance(image_input, str) else None
            return _process_blendshapes(results.face_blendshapes, img_path)
        else:
            return {"error": "얼굴 미검출"}
            
    except Exception as e:
        return {"error": str(e)}

def save_face_data(all_vision_results: list, frame_rate: int, job_id: str = "default"):
    """MediaPipe 데이터를 AI 피드백에 최적화된 구간별 이벤트 형식으로 가공하여 저장합니다."""
    processed_events = []
    if not all_vision_results:
        return

    current_state = None
    start_time = 0.0

    for i, res in enumerate(all_vision_results):
        time_s = i / frame_rate
        face = res.face
        
        # 상태 판별 (기존 로직 유지하되 한글 상태명 사용)
        state = "정면 응시함"
        if not face.has_face:
            state = "얼굴 미검출"
        elif abs(face.gaze_h) > 0.35:
            state = "시선 분산 (좌우)"
        elif face.gaze_v < -0.2:
            state = "시선 분산 (바닥)"
        elif face.gaze_v > 0.3:
            state = "시선 분산 (천장)"
        elif face.brow_up > 0.45:
            state = "눈썹 강조 (열정적)"

        # 상태가 바뀌거나 마지막 프레임인 경우 저장
        if state != current_state:
            if current_state is not None:
                processed_events.append({
                    "start": round(start_time, 2),
                    "end": round(time_s, 2),
                    "duration": round(time_s - start_time, 2),
                    "state": current_state
                })
            current_state = state
            start_time = time_s

    # 마지막 구간 추가
    processed_events.append({
        "start": round(start_time, 2),
        "end": round(len(all_vision_results) / frame_rate, 2),
        "duration": round((len(all_vision_results) / frame_rate) - start_time, 2),
        "state": current_state
    })

    # AI 피드백용 핵심 요약 데이터 생성
    summary = {
        "job_id": job_id,
        "total_duration": round(len(all_vision_results) / frame_rate, 2),
        "face_events": processed_events,
        "stats": {
            "face_detected_ratio": round(len([r for r in all_vision_results if r.face.has_face]) / len(all_vision_results) * 100, 1)
        }
    }

    face_out_dir = Path("processing/MediaPipe_json")
    face_out_dir.mkdir(parents=True, exist_ok=True)
    file_name = f"face_results_{job_id}.json"
    with open(face_out_dir / file_name, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)
    
    print(f"   > [데이터 가공] 얼굴 분석 결과를 AI용 구간 데이터로 압축 저장 완료.")