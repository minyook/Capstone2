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

        def get_hand_state(wrist, shoulder, hip, confidence):
            # 🌟 핵심: 신뢰도가 낮으면 "안 보임"으로 반환
            if confidence < 0.5: 
                return "Not Visible"
            
            # 신뢰도가 높을 때만 위치 판단
            if wrist[1] < shoulder[1]: return "High"
            if wrist[1] < hip[1]: return "Middle"
            return "Low"

        data["left_hand_state"] = get_hand_state(l_wr, l_sh, l_hip, l_wr_conf)
        data["right_hand_state"] = get_hand_state(r_wr, r_sh, r_hip, r_wr_conf)

        # 팔짱 끼기 체크 (두 손이 모두 보이고 신뢰도가 높을 때만)
        if l_wr_conf > 0.5 and r_wr_conf > 0.5:
            dist_l = np.linalg.norm(l_wr - r_el)
            dist_r = np.linalg.norm(r_wr - l_el)
            if dist_l < 50 and dist_r < 50: 
                data["is_arm_crossed"] = True

            # 몸의 기울기
            if l_sh[1] > 0 and r_sh[1] > 0:
                data["body_tilt"] = float(l_sh[1] - r_sh[1])

            # 대표 제스처 이름 결정
            if data["is_arm_crossed"]: data["gesture_name"] = "Arms Crossed"
            elif data["left_hand_state"] == "High" or data["right_hand_state"] == "High": data["gesture_name"] = "Emphasizing"
            elif data["left_hand_state"] == "Middle" or data["right_hand_state"] == "Middle": data["gesture_name"] = "Active"
            else: data["gesture_name"] = "Normal Stand"

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
    """YOLO 제스처 데이터를 시계열 JSON으로 저장합니다."""
    time_series_gesture = {}

    for i, res in enumerate(all_vision_results):
        seconds = i / frame_rate
        timestamp_key = f"{seconds:.2f}" # 시각화 편의를 위해 초 단위 키 사용
        
        yolo_data = res.yolo.to_dict()
        time_series_gesture[timestamp_key] = {
            "gesture_name": yolo_data["gesture_name"],
            "left_hand": yolo_data["left_hand_state"],
            "right_hand": yolo_data["right_hand_state"],
            "is_arm_crossed": yolo_data["is_arm_crossed"],
            "body_tilt": yolo_data["body_tilt"],
            "keypoints": yolo_data["keypoints"]
        }

    yolo_out_dir = Path("processing/Yolo_json")
    yolo_out_dir.mkdir(parents=True, exist_ok=True)
    file_name = f"gesture_results_{job_id}.json"
    with open(yolo_out_dir / file_name, 'w', encoding='utf-8') as f:
        json.dump(time_series_gesture, f, indent=4, ensure_ascii=False)
    
    print(f"   > YOLO JSON 저장 완료: {yolo_out_dir / file_name}")

