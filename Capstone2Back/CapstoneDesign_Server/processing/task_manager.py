from pathlib import Path
import time as timer 
import traceback
import json
import sys
import io

# 터미널 출력 한글 및 유니코드 깨짐/에러 방지
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


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

FRAME_RATE = 0.2
job_status = {} 

def run_analysis_task(job_id: str, video_path: Path, frame_dir: Path, video_dir: Path, custom_criteria: list = None, video_filename: str = None, persona: str = "soft", ppt_filename: str = None):
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
        timing_log = {}
        
        job_status[job_id] = {"status": "Analyzing", "message": "1/6: 🎬 오디오 추출 중..."}
        t_start = timer.time()
        extract_audio(video_path, audio_path)
        timing_log["audio_extraction_time"] = round(timer.time() - t_start, 2)
        
        job_status[job_id] = {"status": "Analyzing", "message": "2/6: 🎬 비디오 프레임 추출 중..."}
        t_start = timer.time()
        frame_paths = extract_all_frames(video_path, frame_dir, FRAME_RATE)
        timing_log["frame_extraction_time"] = round(timer.time() - t_start, 2)
        if not frame_paths: raise Exception("비디오 프레임 추출 실패.")
        
        # 3. YOLO(제스처) + MediaPipe(표정/시선) 실시간 분석
        print(f"\n[3/6] 👀 시각 데이터(YOLO & MediaPipe) 추출 중... (터미널 출력 생략)")
        total_frames_cnt = len(frame_paths)
        t_start = timer.time()
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
            
            # 🌟 실시간 프레임 분석 진행률(%) 피드백 연동하여 로딩바 싱크 완벽 조절!
            # 5프레임 단위로 상태 메시지 갱신하여 API 오버헤드 억제 및 프론트와 싱크 매핑
            if i % 5 == 0 or i == total_frames_cnt - 1:
                pct = int((i + 1) / total_frames_cnt * 100)
                job_status[job_id] = {
                    "status": "Analyzing", 
                    "message": f"3/6: 🤸 시각 데이터(YOLO & MediaPipe) 분석 중... ({pct}%)"
                }
        timing_log["vision_analysis_time"] = round(timer.time() - t_start, 2)
        print(f"   > ✅ 시각 데이터 추출 완료.")

        # [시각화 체크용] 첫 번째 프레임의 분석 결과를 이미지로 저장
        if all_vision_results and frame_paths:
            import cv2
            debug_frame = cv2.imread(str(frame_paths[0]))
            if debug_frame is not None:
                y_res = all_vision_results[0].yolo
                cv2.putText(debug_frame, f"Gesture: {y_res.gesture_name}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(debug_frame, f"L-Hand: {y_res.left_hand_state}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(debug_frame, f"R-Hand: {y_res.right_hand_state}", (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                check_dir = Path("out/aa/testopen")
                check_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(check_dir / "yolo_check.jpg"), debug_frame)
                print(f"   > 🖼️ [시각화 체크] 분석 샘플 이미지가 저장되었습니다: {check_dir / 'yolo_check.jpg'}")
            else:
                print(f"   > ⚠️ [시각화 체크 실패] 첫 번째 프레임을 읽을 수 없습니다.")

        # 4 & 5. Whisper 및 Praat 음성 분석
        job_status[job_id] = {"status": "Analyzing", "message": "4/6: 🎙️ 로컬 음성(Whisper STT) 인식 중..."}
        # audio_analyzer.py 내부에서 파일 저장을 위해 file_id(video_filename)를 넘겨야 함
        from processing.audio_analyzer import transcribe_audio_with_timestamps
        t_start = timer.time()
        audio_segments, whisper_error = transcribe_audio_with_timestamps(str(audio_path), video_filename=file_id)
        timing_log["speech_recognition_time"] = round(timer.time() - t_start, 2)
        
        voice_metrics: dict = {}
        voice_scores: dict = {}
        aligned_data: list = []

        if not audio_segments:
            print(f"\n[4/6] ⚠️ 목소리 텍스트가 추출되지 않았습니다. (음성 분석 스킵)")
        else:
            print(f"\n[4/6] ✅ 로컬 음성 인식 완료.")
            job_status[job_id] = {"status": "Analyzing", "message": "5/6: 🗣️ 음성 운율(Praat) 분석 중..."}
            from processing.audio_analyzer import analyze_prosody_for_segments
            t_start = timer.time()
            audio_segments = analyze_prosody_for_segments(audio_path, audio_segments, video_filename=file_id)
            timing_log["prosody_analysis_time"] = round(timer.time() - t_start, 2)
            print(f"\n[5/6] ✅ 운율 분석 완료.")

            job_status[job_id] = {"status": "Analyzing", "message": "6/6: 🧩 음성-시각 멀티모달 데이터 정렬 중..."}
            aligned_data = align_data(all_vision_results, audio_segments)

            print(f"\n[5b/6] 발화 습관·항목기준표(발표 음성) 분석 중...")
            from processing.voice_patterns import analyze_voice_behavior, voice_scores_from_metrics
            voice_metrics = analyze_voice_behavior(audio_segments, aligned_data, audio_path)
            voice_scores = voice_scores_from_metrics(voice_metrics)

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
        avg_speed = 0.0
        if audio_segments and voice_metrics:
            vm = voice_metrics
            avg_speed = float(vm.get("avg_speech_rate_cps", 0) or 0)
            vs = voice_scores
            items = vs.get("items_100") or []
            voice_summary = (
                f"발표 음성 영역 {vs.get('category_100', 0)}점 "
                f"(속도 {items[0] if len(items) > 0 else 0}, "
                f"안정 {items[1] if len(items) > 1 else 0}, "
                f"말버릇 {items[2] if len(items) > 2 else 0}, "
                f"반복 {items[3] if len(items) > 3 else 0}), "
                f"필러 분당 {vm.get('fillers_per_minute', 0):.1f}회, "
                f"말 속도 {avg_speed:.2f} 글자/초"
            )
        elif audio_segments:
            voice_summary = f"STT 구간 {len(audio_segments)}개 (습관 분석 없음)"

        # PPT 분석 결과 수신
        ppt_summary = "PPT 분석 데이터 없음"
        ppt_metrics = {}
        
        # Fallback if ppt_filename not provided: look for recently created JSON in ppt_json
        if not ppt_filename:
            ppt_json_dir = Path("analysis_json/ppt_json")
            if ppt_json_dir.exists():
                json_files = list(ppt_json_dir.glob("*.json"))
                # Filter out files that already contain a job_id (e.g. have "_total" or "_ppt" or end with "_ppt.json")
                candidate_files = [f for f in json_files if not f.name.endswith("_ppt.json") and "total" not in f.name]
                if candidate_files:
                    # Get the most recently modified one
                    candidate_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                    # If it was created within the last 5 minutes (300 seconds)
                    if timer.time() - candidate_files[0].stat().st_mtime < 300:
                        ppt_filename = candidate_files[0].name
                        print(f"   > 🔎 [PPT 자동 매칭] 최근 분석된 PPT 결과 매칭: {ppt_filename}")

        if ppt_filename:
            ppt_stem = Path(ppt_filename).stem
            possible_paths = [
                Path("analysis_json/ppt_json") / f"{ppt_stem}.json",
                Path("ppt-analysis-engine/data/results") / f"{ppt_stem}.json"
            ]
            for ppt_result_path in possible_paths:
                if ppt_result_path.exists():
                    try:
                        with open(ppt_result_path, 'r', encoding='utf-8') as f:
                            ppt_data = json.load(f)
                            
                        slide_count = ppt_data.get("metadata", {}).get("slide_count", 0) or ppt_data.get("quantitative_stats", {}).get("slide_count", 0)
                        
                        norm_metrics = ppt_data.get("normalized_metrics", {}) or {}
                        readability = norm_metrics.get("readability", 0.8)
                        visual_balance = norm_metrics.get("visual_balance", 0.7)
                        consistency = norm_metrics.get("consistency", 0.8)
                        
                        ppt_summary = (
                            f"슬라이드 수: {slide_count}, "
                            f"가독성 점수: {readability * 100:.1f}/100, "
                            f"레이아웃 균형 점수: {visual_balance * 100:.1f}/100, "
                            f"슬라이드 일관성 점수: {consistency * 100:.1f}/100"
                        )
                        ppt_metrics = {
                            "readability": readability,
                            "visual_balance": visual_balance,
                            "consistency": consistency
                        }
                        
                        # feedback_engine이 이 JSON을 검색할 수 있도록 복사본을 `{file_id}_ppt.json` 형태로 저장해줍니다.
                        ppt_dest_dir = Path("analysis_json/ppt_json")
                        ppt_dest_dir.mkdir(parents=True, exist_ok=True)
                        ppt_dest_path = ppt_dest_dir / f"{file_id}_ppt.json"
                        with open(ppt_dest_path, 'w', encoding='utf-8') as df:
                            json.dump(ppt_data, df, indent=4, ensure_ascii=False)
                        print(f"   > 📊 [PPT 분석 매핑 완료] {ppt_dest_path} 복사 완료")
                        break
                    except Exception as e:
                        ppt_summary = f"PPT 결과 파일 읽기 실패: {e}"

        # [신규] 분석 요약 데이터 생성 (Feedback Engine 및 UI용)
        active_gestures = ["오른손으로 왼쪽 가리키기", "왼손으로 오른쪽 가리키기", "손을 높여 강조", "활발한 손동작"]
        active_count = sum(1 for res in all_vision_results if res.yolo.gesture_name in active_gestures)
        
        # [신규] 전체 대본 추출
        full_transcript = " ".join([seg.get('text', '').strip() for seg in audio_segments]) if audio_segments else "STT 대본 없음"

        analysis_summary = {
            "face_detection_rate": (face_stats["detected_count"] / total_frames * 100) if total_frames > 0 else 0,
            "gaze_score": max(0, 1.0 - (face_stats["gaze_h"] + face_stats["gaze_v"])) if face_stats["detected_count"] > 0 else 0,
            "smile_score": face_stats["smile"],
            "gesture_status": "활발함" if (active_count / total_frames) > 0.1 else "정적임",
            "avg_speed": avg_speed,
            "ppt_summary": ppt_summary,
            "ppt_metrics": ppt_metrics,  # 🌟 ppt_metrics 추가
            "voice_summary": voice_summary,
            "video_type": video_type.value,
            "voice_metrics": voice_metrics,
            "voice_scores": voice_scores,
            "full_transcript": full_transcript,
        }

        # 7. AI 피드백 생성 (Fine-tuned EXAONE 모델 사용)
        from core.feedback_engine import feedback_engine
        
        # [신규] 생성 중임을 알림 (CPU 모드 대응)
        job_status[job_id] = {
            "status": "Analyzing", 
            "message": "7/7: AI 피드백 생성 중... (CPU 모드이므로 1~2분 정도 소요될 수 있습니다)"
        }
        
        # 프로젝트 이름(file_id)을 기반으로 모든 데이터를 취합하여 피드백 생성
        t_start = timer.time()
        llama_feedback = feedback_engine.generate_feedback(file_id, unified_rubric, persona, analysis_summary=analysis_summary)
        timing_log["ai_feedback_generation_time"] = round(timer.time() - t_start, 2)
        
        # 🌟 AI가 직접 채점한 [SCORES_START] ... [SCORES_END] JSON 점수 블록 파싱 및 정제
        ai_scores = None
        import re
        try:
            scores_pattern = re.compile(r'\[SCORES_START\](.*?)\[SCORES_END\]', re.DOTALL)
            match = scores_pattern.search(llama_feedback)
            if match:
                ai_scores = json.loads(match.group(1).strip())
                llama_feedback = scores_pattern.sub("", llama_feedback).strip()
                # 🌟 [교정] AI 채점 JSON 블록이 제거된 후, 리포트 끝에 남게 되는 "4. AI 초정밀 정량 채점표" 마크다운 헤더 라인과 수평선(---)을 말끔히 지워주어 잘림 현상 제거!
                llama_feedback = re.sub(r'(?m)^#+\s*.*?AI\s*초정밀\s*정량\s*채점표.*$', '', llama_feedback).strip()
                llama_feedback = re.sub(r'(?m)^---+$', '', llama_feedback).strip()
                print("   > 🎯 [AI 정량 채점] AI가 직접 매긴 점수 파싱 성공!")
        except Exception as e:
            print(f"   > ⚠️ [AI 정량 채점 파싱 실패] {e}")

        # 🌟 초정밀 예외 방어용 파이썬 채점 폴백 알고리즘 작동!
        if not ai_scores:
            print("   > 🛡️ [정량 채점 폴백 가동] 파이썬 초정밀 알고리즘으로 동적 채점을 적용합니다.")
            try:
                gaze_score_val = round(analysis_summary.get("gaze_score", 0.8) * 100)
                smile_score_val = round(analysis_summary.get("smile_score", 0.5) * 100)
                gesture_score_val = 90 if analysis_summary.get("gesture_status") == "활발함" else 70

                att_items = [
                    round(gaze_score_val * 0.1),
                    round((smile_score_val * 0.5 + gesture_score_val * 0.5) * 0.1)
                ]
                att_total = sum(att_items)

                v_scores = voice_scores or {}
                v_items = v_scores.get("items_100") or [80, 80, 80, 80]
                stab_score = round((v_scores.get("voice_stability_item") or v_items[1] or 80) / 10)
                fillers_per_min = voice_metrics.get("fillers_per_minute", 0.0)
                body_stab = max(1, min(10, round((gaze_score_val * 0.5 + (100 - fillers_per_min * 15)) / 20)))
                fluency = round((v_scores.get("filler_control") or v_items[2] or 80) / 10)
                
                voice_items = [stab_score, body_stab, fluency]
                voice_total = max(0, min(30, sum(voice_items)))

                content_items = [8, 5, 8, 4]
                has_ppt = ppt_summary != "PPT 분석 데이터 없음"
                speech_error_cnt = voice_metrics.get("repeated_phrase_hits", 0)

                if ppt_metrics:
                    readability = ppt_metrics.get("readability", 0.8)
                    balance = ppt_metrics.get("visual_balance", 0.7)
                    consistency = ppt_metrics.get("consistency", 0.8)
                    
                    ppt_rel = 12 if has_ppt else 0
                    del_rel = max(0, 10 - speech_error_cnt * 3)
                    read_bal = round((readability * 0.5 + balance * 0.5) * 15)
                    const_val = round(consistency * 10)
                    
                    content_items = [ppt_rel, del_rel, read_bal, const_val]
                
                content_total = sum(content_items)
                if speech_error_cnt > 0 or not has_ppt:
                    penalty = 0.45 if not has_ppt else 0.70
                    content_total = round(content_total * penalty)
                    content_items = [round(item * penalty) for item in content_items]

                ai_scores = {
                    "attitude": { "category": att_total, "items": att_items },
                    "voice": { "category": voice_total, "items": voice_items },
                    "content": { "category": content_total, "items": content_items }
                }
            except Exception as fe:
                print(f"   > ❌ [폴백 알고리즘 실행 오류] {fe}")
                ai_scores = {
                    "attitude": { "category": 15, "items": [8, 7] },
                    "voice": { "category": 22, "items": [7, 8, 7] },
                    "content": { "category": 30, "items": [10, 5, 10, 5] }
                }
        
        # 🌟 AI 초정밀 정량 채점표 (Scorecard Dashboard) 동적 마크다운 대시보드 주입!
        if ai_scores:
            try:
                def make_progress_bar(score, max_score, bar_length=10):
                    filled = int(round((score / max_score) * bar_length)) if max_score > 0 else 0
                    filled = max(0, min(bar_length, filled))
                    empty = bar_length - filled
                    return "■" * filled + "□" * empty

                def get_status_label(score, max_score):
                    if max_score <= 0: return "🔴 미흡"
                    ratio = score / max_score
                    if ratio >= 0.85: return "⭐ 최상"
                    elif ratio >= 0.70: return "🟢 우수"
                    elif ratio >= 0.50: return "🟡 보통"
                    else: return "🔴 미흡"

                att_cat = ai_scores.get("attitude", {}).get("category", 0)
                att_items = ai_scores.get("attitude", {}).get("items", [0, 0])
                att_gaze = att_items[0] if len(att_items) > 0 else 0
                att_motion = att_items[1] if len(att_items) > 1 else 0

                voc_cat = ai_scores.get("voice", {}).get("category", 0)
                voc_items = ai_scores.get("voice", {}).get("items", [0, 0, 0])
                voc_stab = voc_items[0] if len(voc_items) > 0 else 0
                voc_calm = voc_items[1] if len(voc_items) > 1 else 0
                voc_control = voc_items[2] if len(voc_items) > 2 else 0

                con_cat = ai_scores.get("content", {}).get("category", 0)
                con_items = ai_scores.get("content", {}).get("items", [0, 0, 0, 0])
                con_sync = con_items[0] if len(con_items) > 0 else 0
                con_delivery = con_items[1] if len(con_items) > 1 else 0
                con_layout = con_items[2] if len(con_items) > 2 else 0
                con_theme = con_items[3] if len(con_items) > 3 else 0

                total_score = att_cat + voc_cat + con_cat

                gaze_bar = make_progress_bar(att_gaze, 10)
                gaze_lbl = get_status_label(att_gaze, 10)
                motion_bar = make_progress_bar(att_motion, 10)
                motion_lbl = get_status_label(att_motion, 10)
                att_bar = make_progress_bar(att_cat, 20)
                att_lbl = get_status_label(att_cat, 20)

                stab_bar = make_progress_bar(voc_stab, 10)
                stab_lbl = get_status_label(voc_stab, 10)
                calm_bar = make_progress_bar(voc_calm, 10)
                calm_lbl = get_status_label(voc_calm, 10)
                control_bar = make_progress_bar(voc_control, 10)
                control_lbl = get_status_label(voc_control, 10)
                voc_bar = make_progress_bar(voc_cat, 30)
                voc_lbl = get_status_label(voc_cat, 30)

                sync_bar = make_progress_bar(con_sync, 15)
                sync_lbl = get_status_label(con_sync, 15)
                delivery_bar = make_progress_bar(con_delivery, 10)
                delivery_lbl = get_status_label(con_delivery, 10)
                layout_bar = make_progress_bar(con_layout, 15)
                layout_lbl = get_status_label(con_layout, 15)
                theme_bar = make_progress_bar(con_theme, 10)
                theme_lbl = get_status_label(con_theme, 10)
                con_bar = make_progress_bar(con_cat, 50)
                con_lbl = get_status_label(con_cat, 50)

                total_bar = make_progress_bar(total_score, 100, bar_length=20)
                total_lbl = get_status_label(total_score, 100)

                scorecard_md = f"""
---

## 📊 AI 초정밀 정량 채점표 (Scorecard Dashboard)

이 표는 AI 멀티모달 분석 결과와 채점 기준표를 매핑하여 도출된 객관적인 정량 평가 대시보드입니다.

| 평가 부문 | 세부 분석 지표 | 배점 | 정량 점수 | 수준 및 시각적 달성도 |
| :--- | :--- | :---: | :---: | :--- |
| **🟢 발표 태도 (Attitude)** | 👁️ 카메라 정면 응시율 (시선) | 10점 | **{att_gaze}점** / 10점 | `{gaze_bar}` {gaze_lbl} |
| (부문 합계: **{att_cat}점** / 20점) | 🤸 제스처 역동성 및 자세 (모션) | 10점 | **{att_motion}점** / 10점 | `{motion_bar}` {motion_lbl} |
| **🔵 음성 유창성 (Voice)** | 🔊 음성 데시벨/피치 (안정도) | 10점 | **{voc_stab}점** / 10점 | `{stab_bar}` {stab_lbl} |
| (부문 합계: **{voc_cat}점** / 30점) | 🧘 정면 시선 & 필러 밀집 (평정심) | 10점 | **{voc_calm}점** / 10점 | `{calm_bar}` {calm_lbl} |
| | 🎙️ 불필요한 필러워드 제어 (유창성) | 10점 | **{voc_control}점** / 10점 | `{control_bar}` {control_lbl} |
| **🟡 자료 및 내용 (Content)** | 📂 발화-PPT 싱크로율 (연관성) | 15점 | **{con_sync}점** / 15점 | `{sync_bar}` {sync_lbl} |
| (부문 합계: **{con_cat}점** / 50점) | 📢 대본상 발화 완성도 (전달력) | 10점 | **{con_delivery}점** / 10점 | `{delivery_bar}` {delivery_lbl} |
| | 🎨 슬라이드 균형/가독성 (디자인) | 15점 | **{con_layout}점** / 15점 | `{layout_bar}` {layout_lbl} |
| | 📏 슬라이드 테마 통일 (일관성) | 10점 | **{con_theme}점** / 10점 | `{theme_bar}` {theme_lbl} |
| **🏆 종합 점수 (Total Score)** | **종합 발표 스피치 스코어** | **100점** | **{total_score}점** / 100점 | `{total_bar}` **({total_score}%)** {total_lbl} |
"""
                llama_feedback = llama_feedback.strip() + "\n" + scorecard_md.strip()
            except Exception as se:
                print(f"   > ⚠️ [시각 채점표 대시보드 생성 실패] {se}")
        
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
            # 실시간 상태 판별 (민감도 상향 및 PPT 연동)
            face = f.face
            yolo = f.yolo
            state = "정면 응시함"
            
            if not face.has_face:
                state = "얼굴 미검출"
            else:
                # Nose-Eye Ratio 기반 초정밀 시선 분석 (임계값 0.05)
                gh = face.gaze_h
                if gh > 0.05: # 화면상 우측 응시
                    state = "PPT 응시 중" if yolo.ppt_side == "Right" else "시선 분산 (우측)"
                elif gh < -0.05: # 화면상 좌측 응시
                    state = "PPT 응시 중" if yolo.ppt_side == "Left" else "시선 분산 (좌측)"
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
                    "gaze_v": face.gaze_v,
                    "emotions": getattr(face, 'emotions', {}), # 신규 추가된 감정 데이터
                    "info": {"main_state": state}
                },
                "yolo": {
                    "gesture_name": f.yolo.gesture_name,
                    "left_hand_state": f.yolo.left_hand_state,
                    "right_hand_state": f.yolo.right_hand_state,
                    "is_arm_crossed": f.yolo.is_arm_crossed,
                    "left_hand_visible": f.yolo.left_hand_visible,
                    "right_hand_visible": f.yolo.right_hand_visible,
                    "l_hand_hip_dist": f.yolo.l_hand_hip_dist,
                    "r_hand_hip_dist": f.yolo.r_hand_hip_dist,
                    "person_bbox": f.yolo.person_bbox, # 시각화용
                    "has_person": f.yolo.has_person
                }
            })
        
        total_result = {
            "metadata": {
                "job_id": job_id,
                "video_filename": file_id,
                "video_type": video_type.value,
                "total_time": total_frames / FRAME_RATE,
                "analysis_date": timer.strftime("%Y-%m-%d %H:%M:%S"),
                "timing_log": timing_log
            },
            "summary": analysis_summary,
            "overall_feedback": llama_feedback,
            "timeline_feedback": timeline_feedback,
            "timeline_data": aligned_data,
            "raw_data": optimized_raw_data, # 최적화된 데이터만 저장
            "ai_scores": ai_scores # 🌟 AI 직접 채점 데이터 저장
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
            "total_json_url": f"/results/total/{file_id}_total.json",
            "ai_scores": ai_scores # 🌟 AI 직접 채점 데이터 주입
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
