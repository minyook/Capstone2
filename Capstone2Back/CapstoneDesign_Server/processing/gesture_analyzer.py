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
    YOLO 제스처 분석 로직 개선:
    - PPT 위치 자동 판별 (인물 위치 반대편)
    - 좌우 손 판별 로직 정규화
    - 양손 모으기 우선순위 강화
    """
    data = {
        "has_person": False,
        "has_pelvis": False,
        "has_ankles": False,
        "gesture_name": "기본 자세",
        "left_hand_state": "Low",
        "right_hand_state": "Low",
        "is_arm_crossed": False,
        "body_tilt": 0.0,
        "keypoints": [],
        "person_bbox": []
    }

    if not _YOLO_AVAILABLE or pose_model is None:
        return data

    import cv2
    img = cv2.imread(frame_path)
    if img is None: return data
    img_h, img_w = img.shape[:2]

    results = pose_model(frame_path, verbose=False)

    if results and len(results) > 0 and len(results[0].boxes) > 0:
        data["has_person"] = True
        box = results[0].boxes[0]
        data["person_bbox"] = box.xyxy[0].cpu().numpy().tolist()
        
        # PPT 위치 추정: 인물이 왼쪽이면 PPT는 오른쪽(Right), 인물이 오른쪽이면 PPT는 왼쪽(Left)
        person_center_x = (data["person_bbox"][0] + data["person_bbox"][2]) / 2
        ppt_side = "Right" if person_center_x < img_w / 2 else "Left"

        kp_xy = results[0].keypoints.xy[0].cpu().numpy()
        kp_conf = results[0].keypoints.conf[0].cpu().numpy()
        data["keypoints"] = kp_xy.tolist()

        # 주요 포인트 추출 (5:L_Sh, 6:R_Sh, 9:L_Wr, 10:R_Wr, 11:L_Hip, 12:R_Hip)
        l_sh, r_sh = kp_xy[5], kp_xy[6]
        l_wr, r_wr = kp_xy[9], kp_xy[10]
        l_hip, r_hip = kp_xy[11], kp_xy[12]
        l_wr_conf, r_wr_conf = kp_conf[9], kp_conf[10]
        shoulder_width = np.linalg.norm(l_sh - r_sh)

        # 손 높이 계산
        def get_hand_state(wrist, shoulder, hip, conf):
            if conf < 0.5: return "확인 불가"
            if wrist[1] < shoulder[1]: return "높음"
            if wrist[1] < hip[1]: return "중간"
            return "낮음"

        data["left_hand_state"] = get_hand_state(l_wr, l_sh, l_hip, l_wr_conf)
        data["right_hand_state"] = get_hand_state(r_wr, r_sh, r_hip, r_wr_conf)

        gesture_name = "기본 자세"

        # [우선순위 1] 팔짱 끼기
        if l_wr_conf > 0.5 and r_wr_conf > 0.5:
            if np.linalg.norm(l_wr - kp_xy[8]) < 40 and np.linalg.norm(r_wr - kp_xy[7]) < 40:
                data["is_arm_crossed"] = True
                gesture_name = "팔짱 끼기"

        # [우선순위 2] 양손 모으기 (활발한 손동작보다 우선)
        if gesture_name == "기본 자세" and l_wr_conf > 0.5 and r_wr_conf > 0.5:
            if np.linalg.norm(l_wr - r_wr) < (shoulder_width * 0.4):
                gesture_name = "양손 모으기"

        # [우선순위 3] PPT 가리키기 (확실한 방향성)
        if gesture_name == "기본 자세":
            # 오른손(화면상 좌측)이 바깥쪽(화면 좌측)으로 뻗어질 때
            if r_wr_conf > 0.6 and (r_sh[0] - r_wr[0]) > (shoulder_width * 0.5):
                if ppt_side == "Left":
                    gesture_name = "PPT 가리키기 (오른손)"
                else:
                    gesture_name = "손 뻗기 (바깥쪽)"
            
            # 왼손(화면상 우측)이 바깥쪽(화면 우측)으로 뻗어질 때
            elif l_wr_conf > 0.6 and (l_wr[0] - l_sh[0]) > (shoulder_width * 0.5):
                if ppt_side == "Right":
                    gesture_name = "PPT 가리키기 (왼손)"
                else:
                    gesture_name = "손 뻗기 (바깥쪽)"

            # [우선순위 4] 강조 및 활발한 동작
            if gesture_name == "기본 자세":
                if data["left_hand_state"] == "높음" or data["right_hand_state"] == "높음":
                    gesture_name = "손을 높여 강조"
                elif (l_wr_conf > 0.5 and abs(l_wr[0] - l_sh[0]) > shoulder_width * 0.6) or \
                     (r_wr_conf > 0.5 and abs(r_wr[0] - r_sh[0]) > shoulder_width * 0.6):
                    gesture_name = "활발한 손동작"

        data["gesture_name"] = gesture_name
        data["ppt_side"] = ppt_side
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
        person_bbox=list(y.get("person_bbox", [])),
        left_hand_visible=bool(y.get("left_hand_visible", True)),
        right_hand_visible=bool(y.get("right_hand_visible", True)),
        l_hand_hip_dist=float(y.get("l_hand_hip_dist", 0.0)),
        r_hand_hip_dist=float(y.get("r_hand_hip_dist", 0.0)),
        ppt_side=str(y.get("ppt_side", "Unknown"))
    )

def save_gesture_data(all_vision_results: list, frame_rate: int, job_id: str = "default"):
    """UI와 AI 피드백 모두에 최적화된 시계열 데이터를 저장합니다."""
    time_series_gesture = {}
    processed_events = []
    
    if not all_vision_results:
        return

    current_gesture = None
    start_time = 0.0

    for i, res in enumerate(all_vision_results):
        seconds = i / frame_rate
        timestamp_key = f"{seconds:.2f}"
        yolo = res.yolo
        
        # 1. UI용 데이터 구성
        time_series_gesture[timestamp_key] = {
            "gesture_name": yolo.gesture_name,
            "left_hand": yolo.left_hand_state,
            "right_hand": yolo.right_hand_state,
            "is_arm_crossed": yolo.is_arm_crossed
        }

        # 2. AI 피드백용 이벤트 압축 로직
        if yolo.gesture_name != current_gesture:
            if current_gesture is not None:
                processed_events.append({"start": round(start_time, 2), "end": round(seconds, 2), "gesture": current_gesture})
            current_gesture = yolo.gesture_name
            start_time = seconds

    # AI 요약 정보 추가
    time_series_gesture["__AI_SUMMARY__"] = {
        "events": processed_events
    }

    yolo_out_dir = Path("analysis_json/Yolo_json")
    yolo_out_dir.mkdir(parents=True, exist_ok=True)
    file_name = f"{job_id}_gesture.json"
    with open(yolo_out_dir / file_name, 'w', encoding='utf-8') as f:
        json.dump(time_series_gesture, f, indent=4, ensure_ascii=False)
    
    print(f"   > [UI/AI 통합] 제스처 리포트 저장 완료: {yolo_out_dir / file_name}")

