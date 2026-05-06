import json
from pathlib import Path
import numpy as np
from processing.vision_dto import YoloPoseResult

# YOLO: 제스처(포즈) 전용
try:
    from ultralytics import YOLO  # type: ignore
    _YOLO_AVAILABLE = True
    pose_model = YOLO("yolov8n-pose.pt")
except Exception:
    _YOLO_AVAILABLE = False
    pose_model = None

def analyze_frame_gesture_yolo(frame_path: str) -> dict:
    """
    YOLO는 제스처(포즈) 담당:
    - 사람 존재 여부
    - 골반/발목 가시성
    - 상세 제스처 (손 높이, 팔짱 등)
    """
    data = {
        "has_person": False,
        "has_pelvis": False,
        "has_ankles": False,
        "gesture_name": "Stand",
        "left_hand_state": "Low",
        "right_hand_state": "Low",
        "is_arm_crossed": False,
        "body_tilt": 0.0,
        "keypoints": [],
        "person_bbox": []
    }

    if not _YOLO_AVAILABLE or pose_model is None:
        data["error"] = "ultralytics 미설치"
        return data

    results = pose_model(frame_path, verbose=False)

    if (
        results
        and len(results) > 0
        and getattr(results[0], "boxes", None) is not None
        and len(results[0].boxes) > 0
        and getattr(results[0], "keypoints", None) is not None
    ):
        data["has_person"] = True
        
        # 가장 큰 박스(보통 첫 번째)를 선택하거나 신뢰도 높은 것 선택
        box = results[0].boxes[0]
        data["person_bbox"] = box.xyxy[0].cpu().numpy().tolist()

        kp_xy = results[0].keypoints.xy[0].cpu().numpy()   # 좌표 (x, y)
        kp_conf = results[0].keypoints.conf[0].cpu().numpy() # 신뢰도 (0~1)

        data["keypoints"] = kp_xy.tolist()

        # 가시성 체크 (11: L_Hip, 12: R_Hip, 15: L_Ankle, 16: R_Ankle)
        # 신뢰도가 0.5 이상인 경우에만 감지된 것으로 인정
        data["has_pelvis"] = bool(np.any(kp_conf[11:13] > 0.5))
        data["has_ankles"] = bool(np.any(kp_conf[15:17] > 0.5))

        # --- 제스처 분석 로직 ---
        l_sh, r_sh = kp_xy[5], kp_xy[6]   # 어깨
        l_hip, r_hip = kp_xy[11], kp_xy[12] # 골반
        l_wr, r_wr = kp_xy[9], kp_xy[10]  # 손목
        l_el, r_el = kp_xy[7], kp_xy[8]   # 팔꿈치

        # 신뢰도 추출
        l_wr_conf, r_wr_conf = kp_conf[9], kp_conf[10]

        def get_hand_state_kr(wrist, shoulder, hip, confidence):
            if confidence < 0.5: 
                return "확인 불가"
            
            if wrist[1] < shoulder[1]: return "높음"
            if wrist[1] < hip[1]: return "중간"
            return "낮음"

        data["left_hand_state"] = get_hand_state_kr(l_wr, l_sh, l_hip, l_wr_conf)
        data["right_hand_state"] = get_hand_state_kr(r_wr, r_sh, r_hip, r_wr_conf)

        # 1. 상세 제스처 판별
        gesture_name = "기본 자세"

        # 팔짱 끼기
        if l_wr_conf > 0.5 and r_wr_conf > 0.5:
            dist_l_to_r_el = np.linalg.norm(l_wr - r_el)
            dist_r_to_l_el = np.linalg.norm(r_wr - l_el)
            if dist_l_to_r_el < 60 and dist_r_to_l_el < 60:
                data["is_arm_crossed"] = True
                gesture_name = "팔짱 끼기"

        # 양손 모으기 (경청/대기 자세)
        if not data["is_arm_crossed"] and l_wr_conf > 0.5 and r_wr_conf > 0.5:
            hand_dist = np.linalg.norm(l_wr - r_wr)
            if hand_dist < 50:
                gesture_name = "양손 모으기"

        # 가리키기 (Pointing)
        if gesture_name == "기본 자세":
            # 오른손으로 왼쪽 가리키기 (발표 스크린 방향)
            if r_wr_conf > 0.5 and r_wr[0] < l_sh[0] and r_wr[1] < r_hip[1]:
                gesture_name = "오른손으로 왼쪽 가리키기"
            # 왼손으로 오른쪽 가리키기
            elif l_wr_conf > 0.5 and l_wr[0] > r_sh[0] and l_wr[1] < l_hip[1]:
                gesture_name = "왼손으로 오른쪽 가리키기"
            # 강조 제스처 (손이 높을 때)
            elif data["left_hand_state"] == "높음" or data["right_hand_state"] == "높음":
                gesture_name = "손을 높여 강조"
            elif data["left_hand_state"] == "중간" or data["right_hand_state"] == "중간":
                gesture_name = "활발한 손동작"

        data["gesture_name"] = gesture_name

        # 몸의 기울기
        if l_sh[1] > 0 and r_sh[1] > 0:
            data["body_tilt"] = float(l_sh[1] - r_sh[1])

    return data

def analyze_frame_yolo_pose(frame_path: str) -> YoloPoseResult:
    y = analyze_frame_gesture_yolo(frame_path)
    return YoloPoseResult(
        has_person=bool(y.get("has_person", False)),
        has_pelvis=bool(y.get("has_pelvis", False)),
        has_ankles=bool(y.get("has_ankles", False)),
        gesture_name=str(y.get("gesture_name", "Stand")),
        left_hand_state=str(y.get("left_hand_state", "Low")),
        right_hand_state=str(y.get("right_hand_state", "Low")),
        is_arm_crossed=bool(y.get("is_arm_crossed", False)),
        body_tilt=float(y.get("body_tilt", 0.0)),
        keypoints=list(y.get("keypoints", [])),
        person_bbox=list(y.get("person_bbox", []))
    )

def save_gesture_data(all_vision_results: list, frame_rate: int, job_id: str = "default"):
    """YOLO 데이터를 AI 피드백에 최적화된 구간별 이벤트 형식으로 가공하여 저장합니다."""
    processed_events = []
    if not all_vision_results:
        return

    current_gesture = None
    start_time = 0.0

    for i, res in enumerate(all_vision_results):
        time_s = i / frame_rate
        gesture = res.yolo.gesture_name

        # 제스처가 바뀌거나 마지막 프레임인 경우 저장
        if gesture != current_gesture:
            if current_gesture is not None:
                processed_events.append({
                    "start": round(start_time, 2),
                    "end": round(time_s, 2),
                    "duration": round(time_s - start_time, 2),
                    "gesture": current_gesture
                })
            current_gesture = gesture
            start_time = time_s

    # 마지막 구간 추가
    processed_events.append({
        "start": round(start_time, 2),
        "end": round(len(all_vision_results) / frame_rate, 2),
        "duration": round((len(all_vision_results) / frame_rate) - start_time, 2),
        "gesture": current_gesture
    })

    # 제스처 통계 요약 (AI가 선호하는 데이터)
    gesture_counts = {}
    for event in processed_events:
        g = event["gesture"]
        gesture_counts[g] = gesture_counts.get(g, 0) + event["duration"]

    summary = {
        "job_id": job_id,
        "total_duration": round(len(all_vision_results) / frame_rate, 2),
        "gesture_events": processed_events,
        "gesture_stats": {k: round(v, 2) for k, v in gesture_counts.items()}
    }

    yolo_out_dir = Path("processing/Yolo_json")
    yolo_out_dir.mkdir(parents=True, exist_ok=True)
    file_name = f"gesture_results_{job_id}.json"
    with open(yolo_out_dir / file_name, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)
    
    print(f"   > [데이터 가공] 제스처 분석 결과를 AI용 구간 데이터로 압축 저장 완료.")

