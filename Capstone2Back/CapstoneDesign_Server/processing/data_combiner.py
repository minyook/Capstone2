# [신규 파일] processing/data_combiner.py
import numpy as np
import re

from processing.vision_dto import FrameVisionResult

def align_data(vision_data: list, audio_segments: list) -> list:
    """
    문장(audio_segments)별로 해당 시간대의 평균 시선/표정(vision_data), 
    YOLO 제스처 패턴, 필러워드 빈도 및 운율(prosody) 데이터를 계산하고 정렬합니다.
    """
    print(f"   > [6/6] 데이터 정렬 시작...")
    aligned_results = []
    
    for segment in audio_segments:
        start_time = segment['start']
        end_time = segment['end']
        duration = end_time - start_time
        text = segment.get('text', '')
        
        # 1. 발표 속도 (Speech Rate) 계산 (초당 글자 수)
        speech_rate_cps = len(text) / duration if duration > 0 else 0

        # 2. 운율(Prosody) 데이터 추출
        prosody = {
            "jitter": round(segment.get('jitter', 0), 3),
            "shimmer": round(segment.get('shimmer', 0), 3)
        }
        
        if np.isnan(prosody['jitter']): prosody['jitter'] = 0
        if np.isnan(prosody['shimmer']): prosody['shimmer'] = 0
        
        # 3. 비전 데이터 추출 (해당 구간 내 프레임들)
        frames_in_segment = [
            frame for frame in vision_data 
            if getattr(frame, 'time', -1) >= start_time and getattr(frame, 'time', -1) <= end_time
        ]

        # 얼굴 데이터 평균 계산
        face_frames = [f for f in frames_in_segment if f.face and f.face.has_face]
        if not face_frames:
            avg_vision = {"error": "얼굴 미검출"}
        else:
            avg_vision = {
                "smile": round(sum(f.face.smile for f in face_frames) / len(face_frames), 3),
                "frown": round(sum(f.face.frown for f in face_frames) / len(face_frames), 3),
                "brow_up": round(sum(f.face.brow_up for f in face_frames) / len(face_frames), 3),
                "brow_down": round(sum(f.face.brow_down for f in face_frames) / len(face_frames), 3),
                "jaw_open": round(sum(f.face.jaw_open for f in face_frames) / len(face_frames), 3),
                "mouth_open": round(sum(f.face.mouth_open for f in face_frames) / len(face_frames), 3),
                "squint": round(sum(f.face.squint for f in face_frames) / len(face_frames), 3),
                "gaze_h": round(sum(f.face.gaze_h for f in face_frames) / len(face_frames), 3),
                "gaze_v": round(sum(f.face.gaze_v for f in face_frames) / len(face_frames), 3),
            }

        # 4. YOLO 제스처 패턴 추출 (프론트엔드 동적 자막용)
        is_arm_crossed = False
        gesture_counts = {}
        for f in frames_in_segment:
            if hasattr(f, 'yolo'):
                if getattr(f.yolo, 'is_arm_crossed', False):
                    is_arm_crossed = True
                g_name = getattr(f.yolo, 'gesture_name', "Unknown")
                gesture_counts[g_name] = gesture_counts.get(g_name, 0) + 1
                
        dominant_gesture = "Unknown"
        if gesture_counts:
            dominant_gesture = max(gesture_counts, key=gesture_counts.get)
            
        # 5. 필러워드 추출 (간격/문장부호 유연성 부여)
        # re.compile 구조화로 어..., 음... 등 단독/특이치 바인딩 필터링
        fillers_patterns = [
            r"\b어+\b", r"\b음+\b", r"\b그+\b", r"\b저+\b", r"\b아+\b",
            r"\b그냥\b", r"\b그니까\b", r"\b그러니까\b", r"\b이제\b", r"\b자\b",
            r"\b뭐\b", r"\b좀\b", r"\b으음\b"
        ]
        fillers_count = 0
        for pat in fillers_patterns:
            fillers_count += len(re.findall(pat, text))

        aligned_results.append({
            "start": start_time,
            "end": end_time,
            "text": text,
            "speech_rate_cps": round(speech_rate_cps, 2),
            "fillers_count": fillers_count,
            "is_arm_crossed": is_arm_crossed,
            "dominant_gesture": dominant_gesture,
            "vision_avg": avg_vision,
            "prosody": prosody
        })
        
    print(f"   > [6/6] ✅ 데이터 정렬 및 메타 주입 완료.")
    return aligned_results