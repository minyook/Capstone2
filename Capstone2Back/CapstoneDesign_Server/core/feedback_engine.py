import os
import json
import sys
import io
import glob
from pathlib import Path
from typing import Dict, Any, Optional

# 터미널 출력 한글 깨짐 방지 (Windows 환경)
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Triton 캐시 경로 설정
os.environ["TRITON_CACHE_DIR"] = "C:/temp/triton_cache"

from core.llama_client import get_feedback_from_coach
from core.gemini_client import model as gemini_model

# llama-cpp-python (GGUF 모델 로드용)
try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    print("ℹ️ llama-cpp-python 라이브러리가 없습니다. CPU 추론이 제한될 수 있습니다.")

# 기존 PyTorch 환경 확인 (HAS_CUDA 판별용)
try:
    import torch
    HAS_CUDA = torch.cuda.is_available()
except ImportError:
    HAS_CUDA = False

class FeedbackEngine:
    """
    발표 분석 데이터를 기반으로 전문적인 피드백을 생성하는 엔진입니다.
    GGUF 모델(llama-cpp-python)을 사용하여 노트북 환경에서 최적의 성능을 냅니다.
    """
    
    def __init__(self, provider: str = "exaone"):
        self.provider = provider.lower()
        self.local_model = None
        self.device = "cuda" if HAS_CUDA else "cpu"
        
        # 로컬 모델(EXAONE GGUF) 초기화
        if self.provider == "exaone":
            if LLAMA_CPP_AVAILABLE:
                self._init_gguf_model()
            else:
                print("⚠️ llama-cpp-python이 없어 Gemini로 전환합니다.")
                self.provider = "gemini"

    def _init_gguf_model(self):
        """
        GGUF 파일을 찾아 llama-cpp-python으로 로드합니다.
        """
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        # 학습 스크립트에서 저장한 GGUF 경로
        gguf_dir = os.path.join(os.path.dirname(curr_dir), "training", "exaone_presenter_gguf")
        
        # .gguf 확장자를 가진 파일 찾기
        gguf_files = glob.glob(os.path.join(gguf_dir, "*.gguf"))
        
        if not gguf_files:
            print(f"⚠️ GGUF 파일을 찾을 수 없습니다: {gguf_dir}")
            # 기본 모델 폴더에서도 확인
            gguf_files = glob.glob(os.path.join(os.path.dirname(curr_dir), "*.gguf"))

        if gguf_files:
            model_path = gguf_files[0]
            print(f"\n--- [FeedbackEngine] GGUF 모델 로드 중 ({os.path.basename(model_path)}) ---")
            try:
                # n_gpu_layers: -1이면 모든 레이어를 GPU로 (GPU가 있는 경우), 0이면 CPU로
                self.local_model = Llama(
                    model_path=model_path,
                    n_ctx=2048,
                    n_gpu_layers=-1 if HAS_CUDA else 0,
                    verbose=False
                )
                print(f"✅ GGUF 모델 로드 완료 ({self.device} 모드)")
            except Exception as e:
                print(f"❌ GGUF 모델 로드 실패: {e}")
                self.provider = "gemini"
        else:
            print("⚠️ 학습된 GGUF 모델이 없습니다. Gemini를 사용합니다.")
            self.provider = "gemini"

    def generate_feedback(self, project_name: str, rubric: str = "", persona: str = "soft", 
                          existing_summary: Optional[Dict] = None, 
                          existing_detailed: Optional[Dict] = None) -> str:
        """
        발표 분석 데이터를 기반으로 최종 피드백을 생성합니다.
        데이터가 직접 전달되면 파일 읽기를 건너뜁니다.
        """
        # 1. 데이터 취합
        if existing_summary and existing_detailed:
            analysis_summary = existing_summary
            detailed_data = existing_detailed
        else:
            json_paths = self._find_project_json_files(project_name)
            detailed_data = self._load_json_data(json_paths)
            analysis_summary = detailed_data.get("summary", {})
        
        # 3. 프롬프트 구성
        prompt = self._build_evaluation_prompt(analysis_summary, rubric, detailed_data, persona)
        
        # 4. 모델 공급자에 따른 피드백 생성
        if self.provider == "exaone" and self.local_model:
            return self._get_gguf_feedback(prompt)
        elif self.provider == "gemini":
            return self._get_gemini_feedback(prompt)
        else:
            return get_feedback_from_coach(prompt)

    def _find_project_json_files(self, project_name: str) -> Dict[str, Path]:
        """
        사용자 요청에 따라 analysis_json 폴더 내에서 프로젝트 이름이 포함된 파일들을 찾습니다.
        """
        base_dir = Path("Capstone2Back/CapstoneDesign_Server/analysis_json")
        paths = {}
        
        # 각 폴더별 매칭 규칙
        mapping = {
            "total": "total_json",
            "face": "MediaPipe_json",
            "gesture": "Yolo_json",
            "voice": "Voice_json",
            "ppt": "ppt_json"
        }
        
        for key, folder in mapping.items():
            search_pattern = str(base_dir / folder / f"*{project_name}*.json")
            files = glob.glob(search_pattern)
            if files:
                paths[key] = Path(files[0])
                
        return paths

    def _load_json_data(self, paths: Dict[str, Path]) -> Dict[str, Any]:
        """각 분석 폴더의 JSON에서 모델이 참고할 핵심 디테일을 추출합니다."""
        detailed = {}
        
        # 1. Total JSON (기본 요약 및 타임라인 데이터)
        if "total" in paths:
            try:
                with open(paths["total"], 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    detailed["summary"] = data.get("summary", {})
                    # 너무 긴 데이터는 제외하고 핵심 타임라인만 추출
                    detailed["timeline_samples"] = data.get("timeline_data", [])[:10]
            except: pass

        # 2. PPT 분석 결과
        if "ppt" in paths:
            try:
                with open(paths["ppt"], 'r', encoding='utf-8') as f:
                    ppt = json.load(f)
                    detailed["ppt"] = {
                        "slides": ppt.get("total_slides", 0),
                        "keywords": ppt.get("keywords", []),
                        "warnings": ppt.get("warnings", [])
                    }
            except: pass

        # 3. 시각 데이터 디테일 (Face/Gesture)
        if "face" in paths:
            try:
                with open(paths["face"], 'r', encoding='utf-8') as f:
                    face = json.load(f)
                    detailed["face_events"] = face.get("__AI_SUMMARY__", {}).get("events", [])[:5]
            except: pass

        return detailed

    def _build_evaluation_prompt(self, data: Dict[str, Any], rubric: str, detailed: Dict[str, Any], persona: str = "soft") -> str:
        unified_rubric = rubric if rubric else "전문 발표자로서의 일반적인 스피치 및 태도 기준"
        
        # 페르소나별 지침 추가
        persona_guide = ""
        if persona == "sharp":
            persona_guide = "당신은 냉철하고 분석적인 전문가입니다. 칭찬보다는 구체적인 문제점과 개선 방안을 날카롭게 지적해 주세요."
        else:
            persona_guide = "당신은 따뜻하고 부드러운 발표 코치입니다. 사용자의 장점을 먼저 칭찬하고, 격려하는 말투로 개선 점을 조언해 주세요."

        # 상세 데이터를 텍스트로 변환
        detailed_context = f"""
        [상세 구간 정보]
        - 시각적 주요 이벤트: {detailed.get('face_events', '없음')}
        - PPT 분석: {detailed.get('ppt', '데이터 없음')}
        - 음성 문장 샘플: {[t.get('text') for t in detailed.get('timeline_samples', [])]}
        """

        prompt = f"""[|system|]
발표 자료 구성 및 시각화 전문가로서 사용자의 발표 분석 데이터를 바탕으로 개선을 위한 전문 피드백을 제공합니다.
{persona_guide}
[|user|]
[분석 데이터 요약]
- 얼굴 검출률: {data.get('face_detection_rate', 0):.1f}%
- 시선 집중도: {data.get('gaze_score', 0):.2f}
- 발화 속도: {data.get('avg_speed', 1.0):.2f} cps
- PPT 상태: {data.get('ppt_summary', '정보 없음')}

{detailed_context}

[채점 기준]
{unified_rubric}

위의 구체적인 데이터를 바탕으로 시간대별 특징을 포함한 종합 피드백 리포트를 작성해 주세요.
[|assistant|]
"""
        return prompt

    def _get_gguf_feedback(self, prompt: str) -> str:
        """llama-cpp-python을 이용한 GGUF 추론"""
        try:
            if self.device == "cpu":
                print("\n   > [AI] 피드백 생성 중... (GGUF CPU 모드)")
            
            response = self.local_model(
                prompt,
                max_tokens=1024,
                temperature=0.7,
                top_p=0.9,
                stop=["[|user|]", "[|system|]", "</s>"],
                echo=False
            )
            
            text = response["choices"][0]["text"].strip()
            # assistant 답변 부분만 추출
            if "[|assistant|]" in text:
                text = text.split("[|assistant|]")[1].strip()
            return text
        except Exception as e:
            print(f"⚠️ GGUF 추론 중 오류 발생: {e}")
            return "로컬 AI 모델(GGUF) 응답 실패."

    def _get_gemini_feedback(self, prompt: str) -> str:
        try:
            response = gemini_model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Gemini 피드백 생성 중 오류 발생: {str(e)}"

    def generate_timeline_feedback(self, aligned_data: list, project_name: str, persona: str = "soft") -> Dict[str, str]:
        """
        영상 전체의 타임라인 데이터를 분석하여 주요 시점별로 코칭 팁을 생성합니다.
        GGUF 모델을 사용하여 실시간에 가까운 속도로 생성합니다.
        """
        timeline_tips = {}
        if not aligned_data: return {}
        
        # 시점 샘플링 (CPU 환경에서는 5개, GPU 환경에서는 8개 정도로 제한)
        sample_count = 5 if self.device == "cpu" else 8
        step = max(1, len(aligned_data) // sample_count)
        sample_points = aligned_data[::step]
        
        persona_guide = "냉철하게 지적해줘." if persona == "sharp" else "다정하게 격려하며 조언해줘."

        for point in sample_points:
            time_sec = round(point.get('start', 0), 1)
            text = point.get('text', '')
            gaze = "정면 응시" if abs(point.get('vision_avg', {}).get('gaze_h', 0)) < 0.2 else "시선 분산"
            
            prompt = f"""[|system|]
발표 전문가로서 짧고 명확한 한 문장 조언을 제공합니다. {persona_guide}
[|user|]
시간: {time_sec}초, 상황: {gaze}, 자막: "{text}"
이 시점에 필요한 짧은 피드백 한 문장을 생성해줘.
[|assistant|]
"""
            if self.provider == "exaone" and self.local_model:
                tip = self._get_gguf_feedback(prompt)
                timeline_tips[str(time_sec)] = tip.split('\n')[0]
            else:
                timeline_tips[str(time_sec)] = f"{time_sec}초 지점: 시선 처리에 유의하세요."

        return timeline_tips

# 싱글톤 인스턴스 생성 (서버 시작 시 모델 로드)
feedback_engine = FeedbackEngine(provider="exaone")
