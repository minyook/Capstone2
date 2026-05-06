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

    def _load_json_data(self, paths: Dict[str, Path]) -> Dict[str, Any]:
        """
        MediaPipe, YOLO, PPT 결과 파일에서 상세 데이터를 로드합니다.
        """
        detailed = {}
        
        # 1. PPT 데이터 로드
        ppt_path = paths.get("ppt")
        if ppt_path and ppt_path.exists():
            try:
                with open(ppt_path, 'r', encoding='utf-8') as f:
                    ppt_data = json.load(f)
                    detailed["ppt"] = {
                        "slide_count": ppt_data.get("metadata", {}).get("slide_count", 0),
                        "metrics": ppt_data.get("normalized_metrics", {}),
                        "top_slides": ppt_data.get("slides", [])[:3] # 상위 3개 슬라이드 요약만 포함 (토큰 절약)
                    }
            except Exception as e:
                print(f"⚠️ PPT JSON 로드 실패: {e}")

        # 2. MediaPipe 구간 데이터 로드 (가공됨)
        face_path = paths.get("face")
        if face_path and face_path.exists():
            try:
                with open(face_path, 'r', encoding='utf-8') as f:
                    face_data = json.load(f)
                    # 전체 구간 데이터와 통계 포함
                    detailed["face_analysis"] = {
                        "events": face_data.get("face_events", []),
                        "stats": face_data.get("stats", {})
                    }
            except Exception as e:
                print(f"⚠️ Face JSON 로드 실패: {e}")

        # 3. YOLO 구간 데이터 로드 (가공됨)
        gesture_path = paths.get("gesture")
        if gesture_path and gesture_path.exists():
            try:
                with open(gesture_path, 'r', encoding='utf-8') as f:
                    gesture_data = json.load(f)
                    detailed["gesture_analysis"] = {
                        "events": gesture_data.get("gesture_events", []),
                        "stats": gesture_data.get("gesture_stats", {})
                    }
            except Exception as e:
                print(f"⚠️ Gesture JSON 로드 실패: {e}")

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
1. **데이터 통합 진단**: [핵심 지표]와 [상세 분석 데이터]를 결합하여, 특정 시간대나 특정 슬라이드에서 발생한 문제점을 짚어내십시오. (예: "15번 슬라이드 설명 시 시선이 불안정했습니다.")
2. **Action Plan**: 개선을 위해 바로 실천할 수 있는 구체적인 팁을 3가지 이상 제시하십시오.
3. **PPT와 발표의 조화**: PPT의 가독성/균형 데이터와 실제 발표(음성/태도)가 얼마나 조화로웠는지 평가하십시오.
4. **형식**: 마크다운(Markdown)을 사용하되, 가독성을 위해 섹션을 명확히 나누십시오.

### [1] 종합 평가 및 점수 (100점 만점)
### [2] 시각 및 태도 분석 (Visual & Gesture)
### [3] 음성 및 전달력 분석 (Vocal)
### [4] PPT 콘텐츠 및 조화도 (Content)
### [5] 핵심 개선 Action Plan (Top 3)
### [6] 코치의 최종 한마디
"""
        return prompt

    def _get_gemini_feedback(self, prompt: str) -> str:
        """
        Gemini를 사용한 고품질 피드백 생성
        """
        try:
            # 코치 전용 시스템 인스트럭션 적용 (필요시 별도 모델 인스턴스 생성 가능)
            response = gemini_model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Gemini 피드백 생성 중 오류 발생: {str(e)}"

feedback_engine = FeedbackEngine(provider=os.getenv("FEEDBACK_PROVIDER", "llama"))
