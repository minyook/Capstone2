import os
import json
from pathlib import Path
from typing import Dict, Any, Optional
from core.llama_client import get_feedback_from_coach
from core.gemini_client import model as gemini_model

class FeedbackEngine:
    """
    발표 분석 데이터를 기반으로 전문적인 피드백을 생성하는 엔진입니다.
    로컬 모델(Ollama/EXAONE)과 클라우드 모델(Gemini)을 모두 지원하며, 
    구조화된 피드백을 생성하는 데 최적화되어 있습니다.
    """
    
    def __init__(self, provider: str = "llama"):
        self.provider = provider.lower()

    def generate_feedback(self, analysis_data: Dict[str, Any], rubric: str = "", json_paths: Dict[str, Path] = None) -> str:
        """
        데이터를 기반으로 종합 피드백 리포트를 생성합니다.
        json_paths가 제공되면 파일에서 더 상세한 데이터를 읽어옵니다.
        """
        detailed_data = self._load_json_data(json_paths) if json_paths else {}
        prompt = self._build_evaluation_prompt(analysis_data, rubric, detailed_data)
        
        if self.provider == "gemini":
            return self._get_gemini_feedback(prompt)
        else:
            feedback = get_feedback_from_coach(prompt)
            if "실패" in feedback and os.getenv("GEMINI_API_KEY"):
                print("⚠️ Ollama 응답 실패로 인해 Gemini로 폴백합니다.")
                return self._get_gemini_feedback(prompt)
            return feedback

    def generate_timeline_feedback(self, aligned_data: list, video_filename: str = "default") -> Dict[float, str]:
        """
        시간대별로 구체적인 피드백 팁을 생성합니다.
        aligned_data: [ {start: 0.0, vision_avg: {...}, text: "..."}, ... ]
        """
        if not aligned_data:
            return {5.0: "발표가 시작되었습니다! 적극적인 제스처를 사용해보세요."}

        # 데이터 샘플링
        sampled_data = aligned_data[::2] # 문장 단위 데이터이므로 적절히 조절
        data_str = json.dumps([{
            "t": d['start'],
            "v": d.get('vision_avg', {}),
            "text": d.get('text', "")
        } for d in sampled_data], ensure_ascii=False)

        prompt = f"""
당신은 발표 코칭 전문가입니다. 제공된 문장별 분석 데이터를 바탕으로, 발표 영상 아래에 실시간으로 띄워줄 '한 줄 피드백'들을 작성하세요.

[데이터 요약]
{data_str}

[요구사항]
1. 특정 시간대(초)에 맞춰 발표자에게 도움이 되는 짧고 명확한 조언을 작성하십시오.
2. 모든 시간에 다 넣을 필요는 없으며, 중요한 변화가 있는 지점 위주로 5~8개 정도의 핵심 팁을 추출하십시오.
3. JSON 형식으로만 답변하십시오. (키: 시간(float), 값: 피드백 문자열)
"""
        
        try:
            raw_response = ""
            if self.provider == "gemini" and os.getenv("GEMINI_API_KEY") and "YOUR_GEMINI" not in os.getenv("GEMINI_API_KEY"):
                raw_response = self._get_gemini_feedback(prompt)
            else:
                # LLM을 사용할 수 없는 경우 (API 키 없음 등) 더미 데이터 생성
                print("⚠️ API 키가 없거나 유효하지 않아 샘플 피드백을 생성합니다.")
                return {
                    3.0: "발표를 시작했습니다. 시선을 정면으로 유지하세요!",
                    10.0: "목소리 톤이 일정합니다. 강조할 부분에서 변화를 주세요.",
                    20.0: "제스처가 아주 자연스럽습니다. 계속 유지하세요!",
                    30.0: "마무리까지 자신감 있는 목소리를 유지하세요."
                }
            
            # JSON 파싱
            clean_json = raw_response.strip().replace("```json", "").replace("```", "")
            return json.loads(clean_json)
        except Exception as e:
            print(f"⚠️ 타임라인 피드백 생성 실패: {e}")
            return {5.0: "자신감 있게 발표를 이어가세요!"}

    def _load_json_data(self, paths: Dict[str, Path]) -> Dict[str, Any]:
        """
        MediaPipe, YOLO, Voice, PPT 결과 파일에서 상세 데이터를 로드합니다.
        사용자 요청에 따라 analysis_json 하위 폴더들을 참조합니다.
        """
        detailed = {}
        
        # 1. PPT 데이터 로드 (analysis_json/ppt_json)
        ppt_path = paths.get("ppt")
        if ppt_path and ppt_path.exists():
            try:
                with open(ppt_path, 'r', encoding='utf-8') as f:
                    ppt_data = json.load(f)
                    detailed["ppt"] = {
                        "slide_count": ppt_data.get("metadata", {}).get("slide_count", 0),
                        "metrics": ppt_data.get("normalized_metrics", {}),
                        "top_slides": ppt_data.get("slides", [])[:3]
                    }
            except Exception as e:
                print(f"⚠️ PPT JSON 로드 실패: {e}")

        # 2. MediaPipe 구간 데이터 로드 (정제됨)
        face_path = paths.get("face")
        if face_path and face_path.exists():
            try:
                with open(face_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    summary = data.get("__AI_SUMMARY__", {})
                    detailed["face_analysis"] = {
                        "events": summary.get("events", []),
                        "stats": {"face_detected_ratio": summary.get("detection_ratio", 0)}
                    }
            except Exception as e:
                print(f"⚠️ Face JSON 로드 실패: {e}")

        # 3. YOLO 구간 데이터 로드 (정제됨)
        gesture_path = paths.get("gesture")
        if gesture_path and gesture_path.exists():
            try:
                with open(gesture_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    summary = data.get("__AI_SUMMARY__", {})
                    detailed["gesture_analysis"] = {
                        "events": summary.get("events", []),
                        "stats": {} # 필요시 추가
                    }
            except Exception as e:
                print(f"⚠️ Gesture JSON 로드 실패: {e}")

        # 4. 음성 데이터 로드 (analysis_json/voice_json)
        voice_path = paths.get("voice")
        if voice_path and voice_path.exists():
            try:
                with open(voice_path, 'r', encoding='utf-8') as f:
                    voice_data = json.load(f)
                    segments = voice_data.get("segments", {})
                    detailed["voice_analysis"] = {
                        "segment_count": voice_data.get("segment_count", 0),
                        "sample_segments": list(segments.values())[:5]
                    }
            except Exception as e:
                print(f"⚠️ Voice JSON 로드 실패: {e}")

        return detailed

    def _build_evaluation_prompt(self, data: Dict[str, Any], rubric: str, detailed: Dict[str, Any] = None) -> str:
        """
        분석 지표를 바탕으로 정교한 평가 프롬프트를 구성합니다.
        """
        unified_rubric = rubric if rubric else "전문 발표자로서의 일반적인 스피치 및 태도 기준"
        detailed_str = json.dumps(detailed, ensure_ascii=False, indent=2) if detailed else "상세 데이터 없음"
        
        prompt = f"""
당신은 'Overnight AI'의 수석 발표 코치입니다. 아래의 [핵심 지표]와 [상세 분석 데이터], 그리고 [채점 기준]을 바탕으로 정밀 리포트를 작성하십시오.

[채점 기준]
{unified_rubric}

[핵심 지표 요약]
- 영상 타입: {data.get('video_type', '알 수 없음')}
- 얼굴 검출률: {data.get('face_detection_rate', 0):.1f}%
- 시선/표정 점수: 시선 집중도({data.get('gaze_score', 0):.2f} / 1.0)
- 음성 지표: Pitch({data.get('avg_pitch', 0):.1f}Hz), Vol({data.get('avg_db', 0):.1f}dB), Speed({data.get('avg_speed', 1.0):.1f}x)

[상세 분석 데이터 (JSON)]
{detailed_str}

[출력 요구사항]
1. **데이터 통합 진단**: [핵심 지표]와 [상세 분석 데이터]를 결합하여, 특정 시간대나 특정 슬라이드에서 발생한 문제점을 짚어내십시오.
2. **Action Plan**: 개선을 위해 바로 실천할 수 있는 구체적인 팁을 3가지 이상 제시하십시오.
3. **형식**: 마크다운(Markdown)을 사용하되, 가독성을 위해 섹션을 명확히 나누십시오.

### [1] 종합 평가 및 점수 (100점 만점)
### [2] 시각 및 태도 분석 (Visual & Gesture)
### [3] 음성 및 전달력 분석 (Vocal)
### [4] PPT 콘텐츠 및 조화도 (Content)
### [5] 핵심 개선 Action Plan (Top 3)
### [6] 코치의 최종 한마디
"""
        return prompt

    def _get_gemini_feedback(self, prompt: str) -> str:
        try:
            response = gemini_model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Gemini 피드백 생성 중 오류 발생: {str(e)}"

feedback_engine = FeedbackEngine(provider=os.getenv("FEEDBACK_PROVIDER", "llama"))
