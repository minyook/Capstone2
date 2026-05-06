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
        "keypoints": []
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
        xy = results[0].keypoints.xy
        if len(xy) > 0 and len(xy[0]) > 0:
            kp = xy[0].cpu().numpy() # COCO format (17 points)
            data["keypoints"] = kp.tolist()

            # 가시성 체크 (11: L_Hip, 12: R_Hip, 15: L_Ankle, 16: R_Ankle)
            data["has_pelvis"] = bool(np.any(kp[11:13] > 0))
            data["has_ankles"] = bool(np.any(kp[15:17] > 0))

            # --- 제스처 분석 로직 ---
            # 어깨(5, 6), 팔꿈치(7, 8), 손목(9, 10), 골반(11, 12)
            l_sh, r_sh = kp[5], kp[6]
            l_el, r_el = kp[7], kp[8]
            l_wr, r_wr = kp[9], kp[10]
            l_hip, r_hip = kp[11], kp[12]

            def get_hand_state(wrist, shoulder, hip):
                if wrist[1] == 0: return "Unknown"
                if wrist[1] < shoulder[1]: return "High"
                if wrist[1] < hip[1]: return "Middle"
                return "Low"

            data["left_hand_state"] = get_hand_state(l_wr, l_sh, l_hip)
            data["right_hand_state"] = get_hand_state(r_wr, r_sh, r_hip)

            # 팔짱 끼기 체크 (손목이 반대편 팔꿈치 근처에 있는지)
            if l_wr[0] > 0 and r_wr[0] > 0 and l_el[0] > 0 and r_el[0] > 0:
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
        keypoints=list(y.get("keypoints", []))
    )

def save_gesture_data(all_vision_results: list, frame_rate: int):
    """YOLO 제스처 데이터를 시계열 JSON으로 저장합니다."""
    time_series_gesture = {}

    for i, res in enumerate(all_vision_results):
        seconds = i / frame_rate
        mins, secs = divmod(seconds, 60)
        hours, mins = divmod(mins, 60)
        timestamp_key = f"{int(hours):02d}:{int(mins):02d}:{int(secs):02d}.{int((seconds % 1) * 100):02d}"
        
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
    with open(yolo_out_dir / "gesture_time_series.json", 'w', encoding='utf-8') as f:
        json.dump(time_series_gesture, f, indent=4, ensure_ascii=False)
    
    print(f"   > YOLO JSON 저장 완료: {yolo_out_dir / 'gesture_time_series.json'}")
