from pathlib import Path
import time as timer 
import traceback
import json  # 🌟 추가: JSON 파일 저장을 위해 필요

# 모든 처리 모듈 임포트
from processing.video_analyzer import extract_all_frames, extract_audio, analyze_frame_vision
from processing.audio_analyzer import transcribe_audio_with_timestamps, analyze_prosody_for_segments
from processing.data_combiner import align_data
from utils.helpers import cleanup_dirs

# 🌟 신규 임포트
from utils.quality_checker import check_video_quality, check_audio_quality
from schemas.video_type import VideoType
from core.llama_client import get_feedback_from_coach
from core.exceptions import QualityException

FRAME_RATE = 5
job_status = {} 

def run_analysis_task(job_id: str, video_path: Path, frame_dir: Path, video_dir: Path, custom_criteria: list = None):
    # custom_criteria는 텍스트 리스트일 수도 있고, 파일 경로 리스트일 수도 있음
    all_vision_results = []
    audio_path = frame_dir / "audio.wav" 
    
    # [추가] 채점 기준 통합 텍스트
    unified_rubric = ""
    if custom_criteria:
        for item in custom_criteria:
            # 1. 텍스트인 경우 그대로 추가
            if isinstance(item, str) and not item.endswith(('.pdf', '.docx', '.hwp', '.pptx', '.txt')):
                unified_rubric += f"- {item}\n"
            # 2. 파일 경로인 경우 (다른 친구가 구현할 파일 파서 호출부)
            else:
                # 여기에 파일 파싱 로직을 넣거나, 파싱된 텍스트가 이미 들어있다고 가정
                unified_rubric += f"[파일 기반 기준]: {item} (파일 내용 분석 대기 중)\n"

    audio_path = frame_dir / "audio.wav" 
    
    # 🌟 4단계 분류를 위한 영상 전체의 최대 가시성 추적
    max_visibility = {"face": False, "pelvis": False, "ankles": False}
    
    try:
        print(f"\n{'='*60}\n🚀 분석 시작 (Job ID: {job_id})\n{'='*60}")

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
            
            # YOLO 감지 데이터 업데이트 (객체 속성 확인 방식 대응)
            yolo_data = frame.yolo
            if hasattr(yolo_data, 'has_pelvis'):
                if yolo_data.has_pelvis: max_visibility["pelvis"] = True
                if yolo_data.has_ankles: max_visibility["ankles"] = True
            
        print(f"   > ✅ 시각 데이터 추출 완료.")

        # 4 & 5. Whisper 및 Praat 음성 분석
        job_status[job_id] = {"status": "Analyzing", "message": "4/6: 로컬 음성 인식 실행 중..."}
        audio_segments, whisper_error = transcribe_audio_with_timestamps(str(audio_path))
        
        # 🌟 신규: 목소리가 아예 없는 경우 방어 로직
        if not audio_segments: 
            print(f"\n[4/6] ⚠️ 목소리 텍스트가 추출되지 않았습니다. (음성 분석 스킵)")
            aligned_data = [] # 데이터 정렬 건너뛰기
        else:
            print(f"\n[4/6] ✅ 로컬 음성 인식 완료.")
            audio_segments = analyze_prosody_for_segments(audio_path, audio_segments)
            print(f"\n[5/6] ✅ 운율 분석 완료.")
            
            job_status[job_id] = {"status": "Analyzing", "message": "6/6: 데이터 정렬 중..."}
            aligned_data = align_data(all_vision_results, audio_segments)

        # ==========================================
        # 🌟 4단계 비디오 분류 및 데이터 수집 (통합 수신부)
        # ==========================================
        if max_visibility["ankles"]: video_type = VideoType.FULL_BODY
        elif max_visibility["pelvis"]: video_type = VideoType.UPPER_BODY
        elif max_visibility["face"]: video_type = VideoType.FACE_ONLY
        else: video_type = VideoType.VOICE_ONLY
        
        print(f"\n📊 [분석 결과] 영상 타입 판별: {video_type.value}")
        
        # 1. 시각 데이터(표정, 시선, 제스처) 집계
        total_frames = len(all_vision_results)
        face_stats = {"smile": 0, "gaze_h": 0, "gaze_v": 0, "detected_count": 0}
        pose_stats = {"detected_count": 0} # YOLO/Pose 데이터용

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

        # 2. 음성 데이터 집계
        voice_summary = "음성 데이터 없음"
        if audio_segments:
            avg_pitch = sum(s.get('pitch', 0) for s in audio_segments) / len(audio_segments)
            avg_db = sum(s.get('db', 0) for s in audio_segments) / len(audio_segments)
            avg_speed = sum(s.get('speed', 1.0) for s in audio_segments) / len(audio_segments)
            voice_summary = (f"평균 주파수: {avg_pitch:.1f}Hz, 평균 음량: {avg_db:.1f}dB, "
                             f"말하기 속도: {avg_speed:.1f}x, 총 {len(audio_segments)}개 구간")

        # 3. PPT 분석 결과 수신 (파일 기반 연동)
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

        # ==========================================
        # 🚀 모든 데이터를 취합하여 EXAONE 3.5에게 전달
        # ==========================================
        llama_prompt = f"""
[사용자 제공 채점 기준]
{unified_rubric if unified_rubric else "표준 발표 채점 기준에 따라 평가하세요."}

[발표 종합 분석 데이터]
1. 시각 지표:
   - 영상 타입: {video_type.value}
   - 얼굴 검출률: {(face_stats['detected_count']/total_frames)*100:.1f}%
   - 시선 집중도: {1 - face_stats['gaze_h']:.2f} (1.0에 가까울수록 정면 응시 양호)
   - 미소 빈도: {face_stats['smile']:.2f}
   - 제스처 감지: {"활발함" if pose_stats['detected_count'] > 0 else "제한적임"}

2. 음성 지표:
   - {voice_summary}

3. PPT 콘텐츠:
   - {ppt_summary}

위 지표들과 [사용자 제공 채점 기준]을 바탕으로 발표자의 태도, 전달력, 전문성을 종합 평가하여 항목별 점수와 상세 피드백을 작성해줘.
"""
        llama_feedback = get_feedback_from_coach(llama_prompt)
        
        print(f"\n{'='*20} 🤖 EXAONE 3.5 발표 코치 피드백 {'='*20}")
        print(llama_feedback)

        # ==========================================
        # 🌟 [신규 추가] 시간대별 상세 좌표 통합 JSON 저장 (요청 포맷 반영)
        # ==========================================
        time_series_detail = {}
        for i, res in enumerate(all_vision_results):
            # 1. 00:00:00.00 형식의 타임스탬프 계산
            seconds = i / FRAME_RATE
            mins, secs = divmod(seconds, 60)
            hours, mins = divmod(mins, 60)
            timestamp_key = f"{int(hours):02d}:{int(mins):02d}:{int(secs):02d}.{int((seconds % 1) * 100):02d}"
            
            # 2. 메인 상태(main_state) 판별 로직
            main_state = "정면 응시 / 진지함"
            if res.face.has_face:
                if res.face.smile > 0.5: main_state = "미소 감지"
                elif res.face.squint > 0.5: main_state = "집중 / 찌푸림"
                elif abs(res.face.gaze_h) > 0.3: main_state = "시선 분산 (좌우)"
                elif res.face.brow_up > 0.5: main_state = "눈썹 치켜뜸 (강조)"
            else:
                main_state = "얼굴 미검출"
            
            # 3. MediaPipe 52개 상세 좌표 추출
            face_data = res.face.to_dict()
            blendshapes = face_data.get("all_blendshapes", {})

            time_series_detail[timestamp_key] = {
                "info": {
                    "frame_index": i,
                    "main_state": main_state
                },
                "blendshapes": blendshapes
            }

        # 파일 저장 (MediaPipe_json 폴더 내)
        json_out_dir = Path("processing/MediaPipe_json")
        json_out_dir.mkdir(parents=True, exist_ok=True)
        final_json_path = json_out_dir / "final_time_series.json"
        
        with open(final_json_path, 'w', encoding='utf-8') as f:
            json.dump(time_series_detail, f, indent=4, ensure_ascii=False)
        
        print(f"\n✨ [분석 완료] 시계열 통합 데이터 저장됨: {final_json_path}")
        # ==========================================

        raw_data_json = [f.to_dict() for f in all_vision_results]
        final_result = {
            "video_type": video_type.value,
            "llama_feedback": llama_feedback,
            "raw_data": raw_data_json,
            "aligned_transcript_data": aligned_data
        }
        
        job_status[job_id] = {"status": "Complete", "result": final_result}
        print(f"✅ 모든 분석 작업 완료! (Job: {job_id})")

    except Exception as e:
        print(f"\n❌ 작업 실패 (Job: {job_id}) | 오류: {e}")
        traceback.print_exc()
        job_status[job_id] = {"status": "Error", "message": str(e)}
    finally:
        cleanup_dirs(video_dir, frame_dir)