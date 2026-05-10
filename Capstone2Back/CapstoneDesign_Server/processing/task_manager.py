from pathlib import Path
import time as timer 
import traceback
import json

# 모든 처리 모듈 임포트
from processing.video_analyzer import extract_all_frames, extract_audio, analyze_frame_vision
from processing.audio_analyzer import transcribe_audio_with_timestamps, analyze_prosody_for_segments
from processing.face_analyzer import save_face_data
from processing.gesture_analyzer import save_gesture_data
from processing.data_combiner import align_data
from utils.helpers import cleanup_dirs

# 품질 검사 및 유틸리티 임포트
from utils.quality_checker import check_video_quality, check_audio_quality
from schemas.video_type import VideoType
from core.llama_client import get_feedback_from_coach
from core.exceptions import QualityException

FRAME_RATE = 5
job_status = {} 

def run_analysis_task(job_id: str, video_path: Path, frame_dir: Path, video_dir: Path, custom_criteria: list = None, video_filename: str = None, persona: str = "soft"):
    all_vision_results = []
    audio_path = frame_dir / "audio.wav" 
    
    # 파일명 결정 (전달받은게 없으면 job_id 사용)
    file_id = video_filename if video_filename else job_id

    # 채점 기준 통합 텍스트
    unified_rubric = ""
    if custom_criteria:
        for item in custom_criteria:
            if isinstance(item, str) and not item.endswith(('.pdf', '.docx', '.hwp', '.pptx', '.txt')):
                unified_rubric += f"- {item}\n"
            else:
                unified_rubric += f"[파일 기반 기준]: {item} (파일 내용 분석 대기 중)\n"

    # 4단계 분류를 위한 영상 전체의 최대 가시성 추적
    max_visibility = {"face": False, "pelvis": False, "ankles": False}
    
    try:
        print(f"\n{'='*60}\n🚀 분석 시작 (Job ID: {job_id}, File: {file_id})\n{'='*60}")

        # 0. 품질 검증
        job_status[job_id] = {"status": "Checking", "message": "0/6: 품질 검사 중..."}
        if not check_video_quality(video_path): raise QualityException("영상 화질이 너무 낮거나 손상되었습니다.")
        if not check_audio_quality(video_path): raise QualityException("오디오 트랙을 찾을 수 없습니다.")

        # 1 & 2. 오디오/프레임 추출
        extract_audio(video_path, audio_path)
        frame_paths = extract_all_frames(video_path, frame_dir, FRAME_RATE)
        if not frame_paths: raise Exception("비디오 프레임 추출 실패.")
        
        # 3. YOLO(제스처) + MediaPipe(표정/시선) 실시간 분석
        print(f"\n[3/6] 👀 시각 데이터(YOLO & MediaPipe) 추출 중... (터미널 출력 생략)")
        for i, path in enumerate(frame_paths):
            current_time = i / FRAME_RATE
            frame = analyze_frame_vision(str(path), current_time)
            all_vision_results.append(frame)
            
            # 가시성 업데이트
            if frame.face.has_face:
                max_visibility["face"] = True
            
            yolo_data = frame.yolo
            if hasattr(yolo_data, 'has_pelvis'):
                if yolo_data.has_pelvis: max_visibility["pelvis"] = True
                if yolo_data.has_ankles: max_visibility["ankles"] = True
            
        print(f"   > ✅ 시각 데이터 추출 완료.")

        # [시각화 체크용] 첫 번째 프레임의 분석 결과를 이미지로 저장
        if all_vision_results and frame_paths:
            import cv2
            debug_frame = cv2.imread(str(frame_paths[0]))
            y_res = all_vision_results[0].yolo
            cv2.putText(debug_frame, f"Gesture: {y_res.gesture_name}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(debug_frame, f"L-Hand: {y_res.left_hand_state}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(debug_frame, f"R-Hand: {y_res.right_hand_state}", (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            check_dir = Path("out/aa/testopen")
            check_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(check_dir / "yolo_check.jpg"), debug_frame)
            print(f"   > 🖼️ [시각화 체크] 분석 샘플 이미지가 저장되었습니다: {check_dir / 'yolo_check.jpg'}")

        # 4 & 5. Whisper 및 Praat 음성 분석
        job_status[job_id] = {"status": "Analyzing", "message": "4/6: 로컬 음성 인식 실행 중..."}
        # audio_analyzer.py 내부에서 파일 저장을 위해 file_id(video_filename)를 넘겨야 함
        from processing.audio_analyzer import transcribe_audio_with_timestamps
        audio_segments, whisper_error = transcribe_audio_with_timestamps(str(audio_path), video_filename=file_id)
        
        if not audio_segments: 
            print(f"\n[4/6] ⚠️ 목소리 텍스트가 추출되지 않았습니다. (음성 분석 스킵)")
            aligned_data = [] 
        else:
            print(f"\n[4/6] ✅ 로컬 음성 인식 완료.")
            from processing.audio_analyzer import analyze_prosody_for_segments
            audio_segments = analyze_prosody_for_segments(audio_path, audio_segments, video_filename=file_id)
            print(f"\n[5/6] ✅ 운율 분석 완료.")
            
            job_status[job_id] = {"status": "Analyzing", "message": "6/6: 데이터 정렬 중..."}
            aligned_data = align_data(all_vision_results, audio_segments)

        # 4단계 비디오 분류
        if max_visibility["ankles"]: video_type = VideoType.FULL_BODY
        elif max_visibility["pelvis"]: video_type = VideoType.UPPER_BODY
        elif max_visibility["face"]: video_type = VideoType.FACE_ONLY
        else: video_type = VideoType.VOICE_ONLY
        
        print(f"\n📊 [분석 결과] 영상 타입 판별: {video_type.value}")
        
        # 시각 데이터 집계
        total_frames = len(all_vision_results)
        face_stats = {"smile": 0, "gaze_h": 0, "gaze_v": 0, "detected_count": 0}
        pose_stats = {"detected_count": 0}

        for res in all_vision_results:
            if res.face.has_face:
                face_stats["detected_count"] += 1
                face_stats["smile"] += res.face.smile
                face_stats["gaze_h"] += abs(res.face.gaze_h)
                face_stats["gaze_v"] += abs(res.face.gaze_v)
            
            yolo = res.yolo
            if hasattr(yolo, 'has_pelvis') and yolo.has_pelvis:
                pose_stats["detected_count"] += 1

        if face_stats["detected_count"] > 0:
            for key in ["smile", "gaze_h", "gaze_v"]:
                face_stats[key] /= face_stats["detected_count"]

        # 음성 데이터 집계
        voice_summary = "음성 데이터 없음"
        avg_speed = 0 # 초기화
        if audio_segments:
            avg_pitch = sum(s.get('pitch', 0) for s in audio_segments) / len(audio_segments)
            avg_db = sum(s.get('db', 0) for s in audio_segments) / len(audio_segments)
            avg_speed = sum(s.get('speed', 1.0) for s in audio_segments) / len(audio_segments)
            voice_summary = (f"평균 주파수: {avg_pitch:.1f}Hz, 평균 음량: {avg_db:.1f}dB, "
                             f"말하기 속도: {avg_speed:.1f}x, 총 {len(audio_segments)}개 구간")

        # PPT 분석 결과 수신
        ppt_summary = "PPT 분석 데이터 없음"
        ppt_result_path = Path("ppt-analysis-engine/data/results/example.json")
        if ppt_result_path.exists():
            try:
                with open(ppt_result_path, 'r', encoding='utf-8') as f:
                    ppt_data = json.load(f)
                    ppt_summary = (f"슬라이드 수: {ppt_data.get('total_slides', 0)}, "
                                   f"주요 키워드: {', '.join(ppt_data.get('keywords', []))}")
            except Exception:
                ppt_summary = "PPT 결과 파일 읽기 실패"

        # [신규] 분석 요약 데이터 생성 (Feedback Engine 및 UI용)
        active_gestures = ["오른손으로 왼쪽 가리키기", "왼손으로 오른쪽 가리키기", "손을 높여 강조", "활발한 손동작"]
        active_count = sum(1 for res in all_vision_results if res.yolo.gesture_name in active_gestures)
        
        analysis_summary = {
            "face_detection_rate": (face_stats["detected_count"] / total_frames * 100) if total_frames > 0 else 0,
            "gaze_score": max(0, 1.0 - (face_stats["gaze_h"] + face_stats["gaze_v"])) if face_stats["detected_count"] > 0 else 0,
            "smile_score": face_stats["smile"],
            "gesture_status": "활발함" if (active_count / total_frames) > 0.1 else "정적임",
            "avg_speed": avg_speed,
            "ppt_summary": ppt_summary,
            "voice_summary": voice_summary,
            "video_type": video_type.value
        }

        # 7. AI 피드백 생성 (Fine-tuned EXAONE 모델 사용)
        from core.feedback_engine import feedback_engine
        
        # [신규] 생성 중임을 알림 (CPU 모드 대응)
        job_status[job_id] = {
            "status": "Analyzing", 
            "message": "7/7: AI 피드백 생성 중... (CPU 모드이므로 1~2분 정도 소요될 수 있습니다)"
        }
        
        # 프로젝트 이름(file_id)을 기반으로 모든 데이터를 취합하여 피드백 생성
        llama_feedback = feedback_engine.generate_feedback(file_id, unified_rubric, persona)
        
        print(f"\n{'='*20} 🤖 AI 발표 코치 피드백 (LoRA/RTX 5060 Ti) {'='*20}")
        print(llama_feedback)

        # 🌟 타임라인 피드백 생성 (실시간 자막용)
        timeline_feedback = feedback_engine.generate_timeline_feedback(aligned_data, file_id, persona)

        # 🌟 기존 저장 방식 유지
        save_face_data(all_vision_results, FRAME_RATE, file_id)
        save_gesture_data(all_vision_results, FRAME_RATE, file_id)

        # 🌟 [신규] 통합 Total JSON 생성 및 저장 (용량 최적화 버전)
        total_out_dir = Path("analysis_json/total_json")
        total_out_dir.mkdir(parents=True, exist_ok=True)
        
        # UI에 필요한 핵심 데이터만 필터링하여 용량 축소
        optimized_raw_data = []
        for f in all_vision_results:
            # 실시간 상태 판별 (face_analyzer 로직과 동일하게 유지)
            face = f.face
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
            elif face.jaw_open > 0.3 or face.mouth_open > 0.3:
                state = "말하는 중"

            optimized_raw_data.append({
                "time": f.time,
                "face": {
                    "has_face": face.has_face,
                    "smile": face.smile,
                    "gaze_h": face.gaze_h,
                    "info": {"main_state": state}
                },
                "yolo": {
                    "gesture_name": f.yolo.gesture_name,
                    "left_hand_state": f.yolo.left_hand_state,
                    "right_hand_state": f.yolo.right_hand_state,
                    "is_arm_crossed": f.yolo.is_arm_crossed
                }
            })
        
        total_result = {
            "metadata": {
                "job_id": job_id,
                "video_filename": file_id,
                "video_type": video_type.value,
                "total_time": total_frames / FRAME_RATE,
                "analysis_date": timer.strftime("%Y-%m-%d %H:%M:%S")
            },
            "summary": analysis_summary,
            "overall_feedback": llama_feedback,
            "timeline_feedback": timeline_feedback,
            "timeline_data": aligned_data,
            "raw_data": optimized_raw_data # 최적화된 데이터만 저장
        }
        
        total_json_path = total_out_dir / f"{file_id}_total.json"
        with open(total_json_path, 'w', encoding='utf-8') as f:
            json.dump(total_result, f, indent=4, ensure_ascii=False)
        print(f"✅ 통합 분석 결과 저장 완료: {total_json_path}")

        raw_data_json = [f.to_dict() for f in all_vision_results]
        final_result = {
            "job_id": job_id,
            "video_filename": file_id,
            "video_type": video_type.value,
            "analysis_summary": analysis_summary,
            "llama_feedback": llama_feedback,
            "timeline_feedback": timeline_feedback,
            "raw_data": raw_data_json,
            "aligned_transcript_data": aligned_data,
            "total_json_url": f"/results/total/{file_id}_total.json"
        }
        
        job_status[job_id] = {"status": "Complete", "result": final_result}
        print(f"✅ 모든 분석 작업 완료! (Job: {job_id}, File: {file_id})")

    except Exception as e:
        print(f"\n❌ 작업 실패 (Job: {job_id}) | 오류: {e}")
        traceback.print_exc()
        job_status[job_id] = {"status": "Error", "message": str(e)}
    finally:
        # 비디오는 보존하고 프레임(이미지들)만 삭제하도록 변경
        if frame_dir and frame_dir.exists():
            cleanup_dirs(frame_dir)
        if video_dir and video_dir.exists():
            cleanup_dirs(video_dir)
