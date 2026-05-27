import os
import json
import sys
import io
from pathlib import Path
from typing import Dict, Any, Optional
import ollama
from dotenv import load_dotenv

# 🌟 .env 파일의 환경 변수를 가동 전에 시스템 환경 변수로 명시적 로드!
load_dotenv()

# 터미널 출력 한글 깨짐 방지
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

class FeedbackEngine:
    def __init__(self, provider: str = "gemma"):
        # AI 제공자 설정 (gemma 또는 gemini)
        self.provider = provider.lower()
        # RAM 다이어트: torch, transformers, peft 완전 제거. 부팅 속도 극대화.
        print(f"🎯 [FeedbackEngine] {self.provider.upper()} 엔진 가동 완료 (최적화 모드)")

    def generate_feedback(self, project_name: str, rubric: str = "", persona: str = "soft", analysis_summary: dict = None) -> str:
        # 데이터 취합 (전달받은 analysis_summary가 없으면 디스크에서 JSON을 읽어옴)
        if not analysis_summary:
            json_paths = self._find_project_json_files(project_name)
            detailed_data = self._load_json_data(json_paths)
            analysis_summary = detailed_data.get("summary", {})
        
        # 1. 시각적 비언어 메트릭 추출
        face_rate = analysis_summary.get('face_detection_rate', 50.0)
        smile = analysis_summary.get('smile_score', 0.0)
        gaze = analysis_summary.get('gaze_score', 0.5) * 100
        gesture_status = analysis_summary.get('gesture_status', '정적임')

        # 2. 전달의 안정성(음성) 메트릭 추출
        speed = analysis_summary.get('avg_speed', 5.0)
        voice_scores = analysis_summary.get('voice_scores', {})
        voice_metrics = analysis_summary.get('voice_metrics', {})
        
        filler_rate = voice_metrics.get('fillers_per_minute', 0.0)
        filler_total = voice_metrics.get('filler_total', 0)
        rep_phrase = voice_metrics.get('repeated_phrase_hits', 0)
        silence_count = voice_metrics.get('silence_pause_count', 0)

        # 3. 내용 및 시각화 메트릭 추출
        ppt_summary = analysis_summary.get('ppt_summary', 'PPT 분석 데이터 없음')
        ppt_metrics = analysis_summary.get('ppt_metrics', {})
        ppt_readability = ppt_metrics.get('readability', 0.0) if ppt_metrics else 0.8
        ppt_balance = ppt_metrics.get('visual_balance', 0.0) if ppt_metrics else 0.7
        ppt_consistency = ppt_metrics.get('consistency', 0.0) if ppt_metrics else 0.8

        # 4. 페르소나 매핑
        persona_guide = ""
        if persona.lower() == "sharp":
            persona_guide = "당신은 '🔥 냉철한 전문가' 스피치 컨설턴트입니다. 오버가 없고, 차가우며 대단히 날카롭고 직설적인 지적과 정교한 솔루션을 제공하는 톤앤매너로 작성하십시오."
        else:
            persona_guide = "당신은 '🌸 부드러운 조언자' 스피치 컨설턴트입니다. 발표자를 따뜻하게 격려하고, 긍정적인 면을 칭찬하며 친근하게 단계적인 발전을 유도하는 톤앤매너로 작성하십시오."

        # 시스템 지시문 (수사학적 신뢰도 Ethos 및 인지부하 이론 탑재)
        system_prompt = f"""당신은 대한민국 최고의 스피치 컨설턴트이자 의사소통 학술 전문가입니다. 
{persona_guide}

본 평가는 인지 부하 이론(Cognitive Load Theory)과 수사학적 신뢰도(Ethos)를 기반으로 작성됩니다.
입력된 스코어카드 요약 데이터를 심도 깊게 해독하여, 아래의 출력 규칙에 맞추어 전문적이고 학술적인 발표 코칭 리포트를 마크다운으로 작성해 주십시오."""
        
        user_content = f"""
[발표 평가 핵심 스코어카드]
- 프로젝트 명: {project_name} 

1. 내용 및 시각화 영역 (PPT 디자인 & 구성)
  - PPT 연동 정보: {ppt_summary}
  - 일관성 점수(논리 구조): {ppt_consistency * 100:.1f}/100
  - 가독성 점수(헤드라인/가독성): {ppt_readability * 100:.1f}/100
  - 레이아웃 균형 점수(데이터 신뢰성): {ppt_balance * 100:.1f}/100

2. 전달의 안정성 영역 (음성 평정심 & 발화 유창성)
  - 평균 발화 속도: {speed:.2f}cps
  - 필러 워드(어, 음 등): 분당 {filler_rate:.1f}회 (총 {filler_total}회 감지)
  - 중복 표현 / 단어 반복: {rep_phrase}회 감지
  - 3초 이상 장기 침묵(무음): {silence_count}회 감지
  - 자세 안정성: {'안정적' if gesture_status == '정적임' else '역동적'}

3. 시각적 비언어 영역 (시선 교감 & 제스처)
  - 청중 정면 응시율: {gaze:.1f}%
  - 미소/표정 적절성: {smile * 100:.1f}/100
  - 상체/손 움직임 상태: {gesture_status}

[출력 규칙 - 필독]
1. 리포트는 정확히 아래의 3개 대영역 제목을 순서대로 포함해야 합니다:
  ## I. 내용 및 시각화
  ## II. 전달의 안정성
  ## III. 시각적 비언어

2. 각 대영역(I, II, III) 아래에는 반드시 다음 **4가지 태그**로 문단을 시작하여 분석을 완성해야 합니다. 절대 누락하지 마십시오:
  - **[현상 파악]**: 메트릭 수치와 분석 정황에 근거하여 사용자의 현재 상태를 사실적으로 해독합니다.
  - **[원인/이론적 분석]**: 이러한 현상이 발생한 심리학적, 스피치 공학적 원인을 수사학(Ethos) 또는 인지 심리학(인지 부하 이론)적 관점에서 깊이 분석합니다.
  - **[인지적 영향]**: 발표자의 이 행동/자료 구성이 청중의 정보 수용성과 신뢰도, 그리고 뇌 자원(Cognitive Load) 소모에 어떤 영향을 미치는지 밝힙니다.
  - **[구체적 개선 솔루션]**: 이를 당장 교정할 수 있는 훈련법, PPT 템플릿 정리법 등 명확한 단계별 처방을 내려줍니다.

3. 마지막 부분은 아래의 대제목으로 끝맺음하십시오:
  ## IV. 총평 및 교수님 제출용 근거
  - 발표의 강점과 약점을 종합 요약하고, 교수님이 이 발표자를 채점할 때 참고할 객관적 근거(Justification) 및 가산점 조항(Q&A 대응법 등) 조언을 포함하십시오.

4. 반드시 한국어로 답변하십시오. 전문적이고 설득력 있는 단어들을 구사하십시오.
5. 페르소나(🌸 부드러운 조언자 또는 🔥 냉철한 전문가)의 성격에 어울리도록, 시작 부분에 발표자를 반기는 따뜻하고 세련된 머리말/인사말을 적절히 포함하고, 마지막 마무리 부분에도 발표자를 진심으로 격려하고 성장을 기원하는 맺음말을 조화롭게 작성해 주십시오.
"""

        # 1. Gemma (Ollama 로컬) 모드 구동
        if self.provider == "gemma":
            print(f"   > [AI] Gemma 3:4B (Ollama)를 사용하여 피드백 생성 중...")
            try:
                response = ollama.chat(
                    model='gemma3:4b',
                    messages=[
                        {'role': 'system', 'content': system_prompt},
                        {'role': 'user', 'content': user_content}
                    ],
                    options={
                        'num_predict': 1600,  # 🌟 한글 답변 Truncation(잘림) 방지 및 감성적 격려 멘트의 풍부한 서술 보장
                        'temperature': 0.7,
                        'top_p': 0.95
                    }
                )
                return response['message']['content']
            except Exception as e:
                print(f"❌ Gemma API 오류: {e} (Gemini API 폴백 가동)")
                self.provider = "gemini" # 오류 발생 시 Gemini API로 강제 폴백

        # 2. Gemini API 모드 구동 (시연/발표용)
        if self.provider == "gemini":
            print(f"   > [AI] Gemini API를 사용하여 초고속 피드백 생성 중 (3초 소요)...")
            try:
                from core.gemini_client import chat_with_gemini
                chat_res = chat_with_gemini(f"{system_prompt}\n\n{user_content}", [])
                if chat_res and len(chat_res) > 1:
                    return chat_res[-1]["content"]
            except Exception as e:
                print(f"❌ Gemini API 오류: {e}")
            
            return "피드백 생성 엔진 작동 중 오류가 발생했습니다. API 키 및 로컬 Ollama 구동 여부를 확인해 주세요."

    def _find_project_json_files(self, project_name: str) -> Dict[str, Path]:
        base_dir = Path(__file__).resolve().parent.parent / "analysis_json"
        paths = {}
        mapping = {"total": "total_json", "face": "MediaPipe_json", "gesture": "Yolo_json", "voice": "Voice_json", "ppt": "ppt_json"}
        for key, folder in mapping.items():
            search_pattern = str(base_dir / folder / f"*{project_name}*.json")
            import glob
            files = glob.glob(search_pattern)
            if files: paths[key] = Path(files[0])
        return paths

    def _load_json_data(self, paths: Dict[str, Path]) -> Dict[str, Any]:
        detailed = {}
        if "total" in paths:
            try:
                with open(paths["total"], 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    detailed["summary"] = data.get("summary", {})
            except: pass
        return detailed

    def generate_timeline_feedback(self, aligned_data: list, project_name: str, persona: str = "soft") -> Dict[str, str]:
        feedback_map = {"0.0": "학습된 AI 코치가 실시간 분석을 시작합니다. 화면을 정면으로 바라보며 자신감 있게 출발하세요! 💡"}
        
        if not aligned_data:
            return feedback_map

        for i, segment in enumerate(aligned_data):
            # 세그먼트 시작 시간 결정
            start_time = float(segment.get("start", 0.0))
            if start_time < 0.5 and i == 0:
                continue # 0초 기본 멘트 보호
                
            text = segment.get("text", "")
            
            # 🌟 [개선] 시각/자세 데이터 추출
            vision_avg = segment.get("vision_avg", {})
            if isinstance(vision_avg, dict) and "smile" in vision_avg:
                smile_score = float(vision_avg.get("smile", 0.0))
                gaze_h = float(vision_avg.get("gaze_h", 0.0))
                gaze_v = float(vision_avg.get("gaze_v", 0.0))
                gaze_score = max(0.0, 1.0 - (abs(gaze_h) + abs(gaze_v)))
            else:
                smile_score = 0.0
                gaze_score = 1.0

            gesture = segment.get("gesture_name", segment.get("dominant_gesture", "기본 자세"))
            
            # 🌟 [개선] 음성 속도 및 필러 분석
            speed = float(segment.get("speech_rate_cps", segment.get("speech_rate", 5.0)))
            has_fillers = int(segment.get("fillers_count", 0)) > 0
            
            tip = ""
            
            # 1. 제스처 피드백 (팔짱 감지 우선순위)
            if gesture == "팔짱 감지" or segment.get("is_arm_crossed", False):
                tip = "⚠️ [제스처 피드백] 발표 중 팔짱을 끼면 청중에게 방어적이며 Ethos(신뢰성)가 저하되는 인상을 줍니다. 양손을 자연스럽게 펴 보세요."
            elif gesture in ["오른손으로 왼쪽 가리키기", "왼손으로 오른쪽 가리키기", "손을 높여 강조", "활발한 손동작"]:
                tip = "👍 [제스처 피드백] 발표 흐름에 걸맞은 자연스러운 손동작 제스처를 아주 훌륭하게 구사하고 계십니다! 발표의 흡입력을 더해 줍니다."
            
            # 2. 시선 처리 팁 생성 (시선 분산 시)
            elif gaze_score < 0.4:
                tip = "👁️ [시선 피드백] 현재 화면이나 PPT 쪽으로 시선이 과도하게 쏠리고 있습니다. 청중을 향해 정면을 바라보며 시선을 다시 응집시키세요!"
            
            # 3. 발화 및 필러 팁 생성
            elif has_fillers:
                tip = "🎙️ [발화 유창성] 단어 사이에 '어...', '음...' 같은 불필요한 말버릇(필러워드)이 포착되었습니다. 침착하게 정적(Pause)을 이용해보세요."
            elif speed > 6.0:
                tip = "⏱️ [말 속도 조절] 발화 속도가 초당 6글자를 넘어가 지나치게 조급해 보입니다. 인지 부하(Cognitive Load) 완화를 위해 살짝 템포를 늦추세요."
            elif speed < 3.0 and len(text) > 2:
                tip = "⏱️ [말 속도 조절] 스피치 템포가 다소 느려 발표가 다소 지루해질 위험이 있습니다. 텍스트에 강세와 리듬감을 가미해 또박또박 뱉어주세요."
            
            # 4. 밝은 표정 팁 생성
            elif smile_score > 0.35:
                tip = "🌸 [표정 및 태도] 부드럽고 호감 있는 밝은 표정을 지어 전반적인 스피치 분위기를 아주 편안하게 리드하고 있습니다. 좋습니다!"
                
            # 기본 멘트 (특이치가 없을 때)
            else:
                if len(text) > 0:
                    tip = f"📝 [발표 음성 융합] '{text[:15]}...' 구간을 침착하고 안정적으로 서술하고 있습니다. 정면 응시와 자세 평정심을 조화롭게 유지하세요."
                else:
                    tip = "💡 [실시간 분석] 안정된 시선 집중과 척추 밸런스를 훌륭하게 준수하며 발표를 차분히 이어가고 있습니다. 훌륭합니다."

            # 소수점 한자리 시간 키로 매핑
            time_key = f"{start_time:.1f}"
            feedback_map[time_key] = tip

        return feedback_map

# .env 환경 변수에서 AI_PROVIDER 읽어오기 (기본값 gemma)
env_provider = os.getenv("AI_PROVIDER", "gemma").lower()
feedback_engine = FeedbackEngine(provider=env_provider)
