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
if sys.stdout.encoding != 'utf-8':class FeedbackEngine:
    def __init__(self, provider: str = "gemma"):
        # AI 제공자 설정 (gemma 또는 gemini)
        self.provider = provider.lower()
        self.custom_model = "overnight-coach"
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
        full_transcript = analysis_summary.get('full_transcript', 'STT 대본 없음')
        ppt_metrics = analysis_summary.get('ppt_metrics', {})
        ppt_readability = ppt_metrics.get('readability', 0.0) if ppt_metrics else 0.8
        ppt_balance = ppt_metrics.get('visual_balance', 0.0) if ppt_metrics else 0.7
        ppt_consistency = ppt_metrics.get('consistency', 0.0) if ppt_metrics else 0.8

        # 4. 페르소나 매핑
        persona_guide = ""
        if persona.lower() == "sharp":
            persona_guide = "말투는 오버가 없고, 차가우며 대단히 날카롭고 직설적이다. 데이터에 없는 칭찬은 절대 생성하지 않는다."
        else:
            persona_guide = "말투는 따뜻하고 부드럽지만, 데이터 기반 사실 지적에서는 단호하고 명확하다. 없는 사실을 지어내어 칭찬하는 행위는 절대 금지한다."

        # === 시스템 프롬프트: 냉철하고 정밀한 발표 데이터 분석가 ===
        system_prompt = f"""# Role
너는 발표자의 영상 및 음성 멀티모달 파이프라인(물리 dB VAD + Whisper Word-level STT, YOLO-Pose 어깨 Yaw 각도 추정 & 3D Keypoint 변위 분석)을 통해 계측된 초정밀 정량 데이터(JSON Log)를 실전적인 피드백 리포트로 변환하는 '냉철하고 정밀한 발표 데이터 분석가'이다.
{persona_guide}

# Operational Principles (작동 원칙 - 필수 준수)
1. STRICT TRUTH-TELLING: 오직 제공된 [Input Data JSON]의 수치와 타임스탬프(VAD 무음 구간, Whisper 워드 타임스탬프, YOLO 몸 방향/Yaw Yaw 변동성)만을 기반으로 작성하라. 데이터에 없는 내용을 상상하거나, 임의로 수치를 지어내어 칭찬하거나 비판하는 '환각(Hallucination)' 발생 시 시스템 오류로 간주된다.
2. EVIDENCE-BASED ANALYSIS: 모든 현상 파악 및 단점 지적 세션에는 반드시 해당 현상이 발생한 물리적 '타임스탬프(MM:SS)' 및 '정확한 수치'를 근거로 제시해야 한다. 특히 무음 구간의 시작과 끝, 어깨 Yaw 각도에 따른 측면 응시 시점, 필러 워드가 감지된 개별 단어의 타임라인을 매핑하라.
3. LATERAL GESTURE & ORIENTATION INTERPRETATION: 발표자가 정면이 아닌 측면(스크린 방향)을 향해 어깨 Yaw 각도가 돌아가 서 있는 경우(`looking_at_screen` 등), 단순 얼굴 미검출로 넘기지 말고 "어깨 각도 Yaw 벡터 분석 결과 측면(스크린)을 보고 서 있는 자세 불일치 및 시선 차단"으로 정확하게 지적하라.
4. NO EMPTY BUZZWORDS: '에토스', '파토스', '인지 부하 이론', '아레테' 같은 거창하고 추상적인 학술 수사학 용어는 전면 금지한다. 청중이 직관적으로 이해할 수 있는 실전 개발/교정 용어(예: 시선 이탈, 발화 공백, 정렬 불일치, 측면 차단 등)만 사용하라.
5. ACTIONABLE SOLUTIONS: 조언은 "Z자로 보세요" 같은 뻔한 교과서적 문구가 아니라, 데이터 로그에 나타난 구체적인 문제(예: 특정 구간에서 발생한 3초 이상의 장기 침묵, 꼬인 발화 자책 구간, 측면 서 있기 지점)를 타겟팅하는 단계별 '행동 지침(Action Item)'을 제공하라."""

        user_content = f"""
[Input Data JSON - 초정밀 멀티모달 분석 결과]
- 프로젝트 명: {project_name}

[audio_analysis - 발표 대본 (Whisper STT 전문)]
{full_transcript}

[audio_analysis - VAD & STT 정량 로그]
- 평균 발화 속도: {speed:.2f} cps (글자/초)
- 필러 워드(어, 음, 그, 아 등): 분당 {filler_rate:.1f}회 (총 {filler_total}회 감지)
- 3초 이상 장기 침묵(VAD 무음 구간): {silence_count}회 감지
- 발화 실수 및 꼬인 문맥 감지(Speech Errors): {rep_phrase}회 감지

[vision_analysis - 3D YOLO Pose & Gaze 메트릭]
- 카메라(청중) 정면 응시율: {gaze:.1f}%
- 측면 서 있기 상태(어깨 Yaw 변위): {gesture_status}
- 양손 손목 3D 변위 분산(제스처 역동성): {smile * 100:.1f}/100
- 얼굴 검출률: {face_rate:.1f}%

[slide_analysis - PPT 시각 자료 메트릭]
- PPT 연동 정보: {ppt_summary}
- 가독성 점수: {ppt_readability * 100:.1f}/100
- 레이아웃 균형 점수: {ppt_balance * 100:.1f}/100
- 일관성 점수: {ppt_consistency * 100:.1f}/100

---
---
# Output Report Format (출력 양식)
반드시 다음 구조와 마크다운(Markdown) 포맷을 칼같이 유지하여 답변을 출력하라. 표(Table)를 생성할 때는 반드시 유효한 마크다운 문법(| 열1 | 열2 | 형태)을 사용하여 개행과 파이프라인이 어긋나지 않는 깔끔한 격자형 표를 출력하라.

## 📊 1. 멀티모달 정량 분석 요약
백엔드 JSON의 핵심 수치를 청중이 보기 쉽게 다음 마크다운 테이블(격자 표) 포맷으로 정확히 출력하라.

| 카테고리 | 지표 | 측정 수치 및 상태 |
| :--- | :--- | :--- |
| **발화 유창성** | 3초 이상 무음 감지 횟수 | [VAD 감지 총 침묵 횟수 및 누적 시간] |
| | 필러 워드 감지 횟수 | [Whisper 감지 총 필러워드 횟수 (분당 fpm)] |
| **시각적 역동성** | 평균 청중 응시율 | [평균 청중 응시율 %] |
| | 어깨 Yaw 변위 상태 | [어깨 Yaw 각도 기준 정면 vs 측면 유지 비율 및 상태] |
| **시각 자료 완성도** | 가독성 점수 | [가독성 점수 평균] |
| | 슬라이드 일관성 점수 | [슬라이드 일관성 수치] |

- **발화 유창성 요약**: [VAD 감지 총 침묵 횟수 및 누적 시간], [Whisper 감지 총 필러워드 횟수]
- **시각적 역동성 요약**: [평균 청중 응시율 %], [어깨 Yaw 각도 기준 정면 vs 측면 유지 비율]
- **시각 자료 완성도 요약**: [가독성 점수 평균], [슬라이드 일관성 수치]

## 🎯 2. AI 초정밀 100점 만점 발표 채점표
(전체 점수를 시각적으로 대단히 미려하고 직관적인 대시보드 포맷으로 구조화하여 출력하라.)

### 🏆 종합 스피치 스코어: [종합 점수] / 100점
> **[종합 한줄 총평]** (예: "안정적인 스펙트럼의 발화 유창성을 지녔으나, 시각적 시선 분산 극복이 필요한 발표입니다.")

#### 🟢 [20점] 발표 태도 및 모션 스펙트럼 (Attitude) — **[Attitude 총점]점 / 20점**
- 👁️ **시선 집중도** (10점 만점): `[items[0] 점수]점` — [간략한 데이터 평: 예: 정면 유지 우수]
- 🤸 **제스처 및 포스쳐** (10점 만점): `[items[1] 점수]점` — [간략한 데이터 평: 예: 양손 3D 분산 안정]

#### 🔵 [30점] 음성 유창성 및 전달 밸런스 (Voice) — **[Voice 총점]점 / 30점**
- 🔊 **음성 피치 안정도** (10점 만점): `[items[0] 점수]점` — [간략 평]
- 🧘 **스피치 평정심 지수** (10점 만점): `[items[1] 점수]점` — [간략 평]
- 🎙️ **불필요한 필러워드 조절** (10점 만점): `[items[2] 점수]점` — [간략 평: 분당 fpm 지적]

#### 🟡 [50점] 컨텐츠 가독성 및 맥락 동기화 (Content) — **[Content 총점]점 / 50점**
- 📂 **발화-PPT 싱크로율** (15점 만점): `[items[0] 점수]점` — [간략 평]
- 📢 **대본 가치 전달력** (10점 만점): `[items[1] 점수]점` — [간략 평]
- 🎨 **슬라이드 레이아웃 균형** (15점 만점): `[items[2] 점수]점` — [간략 평]
- 📏 **슬라이드 테마 일관성** (10점 만점): `[items[3] 점수]점` — [간략 평]

---

## 🔍 3. 타임라인 기반 정밀 팩트 체크 (현상 및 원인)
(JSON 로그에서 수치가 낮거나 이벤트가 발생한 타임스탬프를 매핑하여 날카롭게 분석)

### 🔴 오디오 파이프라인 감지 항목
- **정밀 무음(VAD Silence)**: VAD가 감지한 물리적 무음 구간의 시작과 끝 타임스탬프(소수점 표기)를 정확하게 기록하고 발표 공백 원인을 대본과 대조하여 서술하라.
- **필러 워드 밀집도**: Whisper의 `word_timestamps=True`를 통해 감지된 '어...', '음...' 등이 몇 초 대에 다량 분포해 있는지 정확한 초 단위 타임스탬프와 발생 텍스트를 인용하여 지적하라.
- **발화 실수 및 자책**: 발표 중간에 대사가 꼬여 자책하는 멘트(예: "아이 뭐야", "과ㅈ...")가 터져 나온 정확한 타임스탬프와 해당 텍스트를 칼같이 집어내라.

### 🔵 비전 파이프라인 감지 항목
- **측면 서 있기 및 시선 이탈**: 어깨 Yaw 각도 벡터의 사잇각 연산에 따라 몸이 완전히 측면으로 돌아가 스크린을 보았던 타임라인 구간을 정확히 기재하고, 청중에게 등을 돌려 소통을 차단한 현상을 지적하라.
- **제스처 활성도 미흡/과도**: 양손 손목 3D 변위 분산 분석 결과를 바탕으로, 너무 정적이거나 반대로 불필요하게 움직임이 심했던 구간을 구체적인 변위 데이터 변동폭과 함께 설명하라.

### 🟡 슬라이드 파이프라인 감지 항목
- **PPT 내용과 발화의 의미적 일치도**: 발표자가 PPT 장표를 넘기는 시점과 STT 발화 내용의 문맥이 일치하는지 분석하여 서술하라.

## 🛠️ 4. 실전 커스텀 교정 솔루션 (Action Items)
(위의 팩트 체크에서 발견된 문제점을 1:1로 치료하는 실전 피드백 처방전)
- 각 Action Item은 **[Action N] 제목** 형식으로 시작하고, 해당 문제의 타임스탬프를 다시 인용하며, 구체적인 단계별 실천 가이드를 제시하라.
- "열린 자세를 취하세요", "Z자로 보세요"와 같은 뻔한 교과서적인 문구는 즉시 반려된다. 발표자의 특정 타임라인 문제(예: 특정 구간 측면 서 있기, 필러 워드 폭발 지점)를 확실하게 고칠 수 있는 행동 지침을 적어라.

---
# 📊 [CRITICAL] 5. AI 초정밀 정량 채점표 (JSON)
피드백 보고서 작성이 완전히 끝난 후, 맨 마지막 줄에 아래 스펙 및 채점 기준에 따라 한치의 뻥튀기 없는 차갑고 객관적인 정량 점수 JSON 블록을 반드시 `[SCORES_START]`와 `[SCORES_END]` 태그로 감싸서 출력하라. 절대 임의로 키 이름을 바꾸거나 주석을 넣지 말라.

## 채점 기준 가이드:
1. **attitude (태도 - 20점 만점)**:
   - items[0] (시선 점수 - 10점 만점): 카메라 정면 응시율이 80% 이상이면 9~10점, 50% 미만이면 5점 이하로 냉정하게 매긴다.
   - items[1] (모션 점수 - 10점 만점): 제스처 역동성이 너무 미흡(50 미만)하거나 측면 서 있기 비율이 높다면 5~7점 이하로 깎는다.
   - category: items[0] + items[1] (20점 만점)
2. **voice (목소리 - 30점 만점)**:
   - items[0] (음성 안정도 - 10점 만점): 데시벨 및 주파수 변동도가 불균형하거나 안정도가 낮다면 6~7점 이하로 깎는다.
   - items[1] (평정심 - 10점 만점): 정면 시선과 필러 밀집도를 종합하여 불안 요소가 있으면 감점한다.
   - items[2] (유창성 - 10점 만점): 분당 필러 워드가 3회 이상이면 7점 이하, 5회 이상이면 5점 이하로 감점한다.
   - category: items[0] + items[1] + items[2] (30점 만점)
3. **content (내용 - 50점 만점)**:
   - items[0] (PPT 연관성 - 15점 만점): PPT 연동이 없거나 'PPT 분석 데이터 없음' 이면 0점, 발화와 매칭율이 낮으면 8점 이하.
   - items[1] (발화 전달력 - 10점 만점): 대본상 발화 실수가 적발되거나 꼬인 문맥이 있으면 감점한다.
   - items[2] (슬라이드 가독성/균형 - 15점 만점): PPT 가독성 및 균형 점수가 80점 미만이면 10점 이하.
   - items[3] (슬라이드 일관성 - 10점 만점): 일관성 점수가 낮거나 과부하 장표가 많으면 감점한다.
   - category: items[0] + items[1] + items[2] + items[3] (50점 만점)

## JSON 출력 스펙 예시 (반드시 이 포맷을 준수):
[SCORES_START]
{
  "attitude": { "category": 15, "items": [8, 7] },
  "voice": { "category": 22, "items": [7, 8, 7] },
  "content": { "category": 34, "items": [10, 8, 9, 7] }
}
[SCORES_END]

반드시 한국어로 답변하십시오.
"""

        # 1. Gemma (Ollama 로컬) 모드 구동
        if self.provider == "gemma":
            # Ollama 모델 목록을 확인하여 전이학습 모델(overnight-coach) 존재 여부 검사
            target_model = self.custom_model
            try:
                models_list = ollama.list()
                available_models = [m.get('name') or m.get('model') for m in models_list.get('models', [])]
                if not any(target_model in str(m) for m in available_models if m):
                    print(f"   > [AI] '{target_model}' 모델이 Ollama에 등록되어 있지 않습니다. 기본 'gemma3:4b' 모델로 진행합니다.")
                    target_model = "gemma3:4b"
            except Exception as le:
                print(f"   > [AI] Ollama 모델 리스트 확인 실패 ({le}). 기본 'gemma3:4b'로 진행합니다.")
                target_model = "gemma3:4b"

            print(f"   > [AI] {target_model} (Ollama)를 사용하여 피드백 생성 중...")
            try:
                response = ollama.chat(
                    model=target_model,
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
            
            return "피드백 생성 엔진 작동 중 오류가 발생했습니다. API 키 및 로컬 Ollama 구동 여부를 확인해 주세요."�� 타임스탬프를 매핑하여 날카롭게 분석)

### 🔴 오디오 파이프라인 감지 항목
- **정밀 무음(VAD Silence)**: VAD가 감지한 물리적 무음 구간의 시작과 끝 타임스탬프(소수점 표기)를 정확하게 기록하고 발표 공백 원인을 대본과 대조하여 서술하라.
- **필러 워드 밀집도**: Whisper의 `word_timestamps=True`를 통해 감지된 '어...', '음...' 등이 몇 초 대에 다량 분포해 있는지 정확한 초 단위 타임스탬프와 발생 텍스트를 인용하여 지적하라.
- **발화 실수 및 자책**: 발표 중간에 대사가 꼬여 자책하는 멘트(예: "아이 뭐야", "과ㅈ...")가 터져 나온 정확한 타임스탬프와 해당 텍스트를 칼같이 집어내라.

### 🔵 비전 파이프라인 감지 항목
- **측면 서 있기 및 시선 이탈**: 어깨 Yaw 각도 벡터의 사잇각 연산에 따라 몸이 완전히 측면으로 돌아가 스크린을 보았던 타임라인 구간을 정확히 기재하고, 청중에게 등을 돌려 소통을 차단한 현상을 지적하라.
- **제스처 활성도 미흡/과도**: 양손 손목 3D 변위 분산 분석 결과를 바탕으로, 너무 정적이거나 반대로 불필요하게 움직임이 심했던 구간을 구체적인 변위 데이터 변동폭과 함께 설명하라.

### 🟡 슬라이드 파이프라인 감지 항목
- **PPT 내용과 발화의 의미적 일치도**: 발표자가 PPT 장표를 넘기는 시점과 STT 발화 내용의 문맥이 일치하는지 분석하여 서술하라.

## 🛠️ 4. 실전 커스텀 교정 솔루션 (Action Items)
(위의 팩트 체크에서 발견된 문제점을 1:1로 치료하는 실전 피드백 처방전)
- 각 Action Item은 **[Action N] 제목** 형식으로 시작하고, 해당 문제의 타임스탬프를 다시 인용하며, 구체적인 단계별 실천 가이드를 제시하라.
- "열린 자세를 취하세요", "Z자로 보세요"와 같은 뻔한 교과서적인 문구는 즉시 반려된다. 발표자의 특정 타임라인 문제(예: 특정 구간 측면 서 있기, 필러 워드 폭발 지점)를 확실하게 고칠 수 있는 행동 지침을 적어라.

---
# 📊 [CRITICAL] 5. AI 초정밀 정량 채점표 (JSON)
피드백 보고서 작성이 완전히 끝난 후, 맨 마지막 줄에 아래 스펙 및 채점 기준에 따라 한치의 뻥튀기 없는 차갑고 객관적인 정량 점수 JSON 블록을 반드시 `[SCORES_START]`와 `[SCORES_END]` 태그로 감싸서 출력하라. 절대 임의로 키 이름을 바꾸거나 주석을 넣지 말라.

## 채점 기준 가이드:
1. **attitude (태도 - 20점 만점)**:
   - items[0] (시선 점수 - 10점 만점): 카메라 정면 응시율이 80% 이상이면 9~10점, 50% 미만이면 5점 이하로 냉정하게 매긴다.
   - items[1] (모션 점수 - 10점 만점): 제스처 역동성이 너무 미흡(50 미만)하거나 측면 서 있기 비율이 높다면 5~7점 이하로 깎는다.
   - category: items[0] + items[1] (20점 만점)
2. **voice (목소리 - 30점 만점)**:
   - items[0] (음성 안정도 - 10점 만점): 데시벨 및 주파수 변동도가 불균형하거나 안정도가 낮다면 6~7점 이하로 깎는다.
   - items[1] (평정심 - 10점 만점): 정면 시선과 필러 밀집도를 종합하여 불안 요소가 있으면 감점한다.
   - items[2] (유창성 - 10점 만점): 분당 필러 워드가 3회 이상이면 7점 이하, 5회 이상이면 5점 이하로 감점한다.
   - category: items[0] + items[1] + items[2] (30점 만점)
3. **content (내용 - 50점 만점)**:
   - items[0] (PPT 연관성 - 15점 만점): PPT 연동이 없거나 'PPT 분석 데이터 없음' 이면 0점, 발화와 매칭율이 낮으면 8점 이하.
   - items[1] (발화 전달력 - 10점 만점): 대본상 발화 실수가 적발되거나 꼬인 문맥이 있으면 감점한다.
   - items[2] (슬라이드 가독성/균형 - 15점 만점): PPT 가독성 및 균형 점수가 80점 미만이면 10점 이하.
   - items[3] (슬라이드 일관성 - 10점 만점): 일관성 점수가 낮거나 과부하 장표가 많으면 감점한다.
   - category: items[0] + items[1] + items[2] + items[3] (50점 만점)

## JSON 출력 스펙 예시 (반드시 이 포맷을 준수):
[SCORES_START]
{{
  "attitude": {{ "category": 15, "items": [8, 7] }},
  "voice": {{ "category": 22, "items": [7, 8, 7] }},
  "content": {{ "category": 34, "items": [10, 8, 9, 7] }}
}}
[SCORES_END]

반드시 한국어로 답변하십시오.
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
                tip = "⚠️ [제스처 피드백] 발표 중 팔짱을 끼면 청중에게 방어적이고 닫힌 인상을 줍니다. 양손을 자연스럽게 펴서 개방적인 자세를 유지하세요."
            elif gesture in ["오른손으로 왼쪽 가리키기", "왼손으로 오른쪽 가리키기", "손을 높여 강조", "활발한 손동작"]:
                tip = "👍 [제스처 피드백] 발표 흐름에 걸맞은 자연스러운 손동작 제스처를 아주 훌륭하게 구사하고 계십니다! 발표의 흡입력을 더해 줍니다."
            
            # 2. 시선 처리 팁 생성 (시선 분산 시)
            elif gaze_score < 0.4:
                tip = "👁️ [시선 피드백] 현재 화면이나 PPT 쪽으로 시선이 과도하게 쏠리고 있습니다. 청중을 향해 정면을 바라보며 시선을 다시 응집시키세요!"
            
            # 3. 발화 및 필러 팁 생성
            elif has_fillers:
                tip = "🎙️ [발화 유창성] 단어 사이에 '어...', '음...' 같은 불필요한 말버릇(필러워드)이 포착되었습니다. 침착하게 정적(Pause)을 이용해보세요."
            elif speed > 6.0:
                tip = "⏱️ [말 속도 조절] 발화 속도가 초당 6글자를 넘어가 청중이 따라가기 어렵습니다. 살짝 템포를 늦추고 호흡을 넣으세요."
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
