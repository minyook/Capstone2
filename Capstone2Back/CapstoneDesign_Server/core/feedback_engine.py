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

# Unsloth 및 PyTorch (로컬 모델용)
try:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    HAS_CUDA = torch.cuda.is_available()
    
    if HAS_CUDA:
        try:
            from unsloth import FastLanguageModel
            UNSLOTH_AVAILABLE = True
        except ImportError:
            UNSLOTH_AVAILABLE = False
            print("ℹ️ Unsloth 라이브러리가 없습니다. 일반 Transformers 모드로 실행합니다.")
    else:
        UNSLOTH_AVAILABLE = False
        print("ℹ️ GPU가 감지되지 않았습니다. CPU 모드로 전환합니다.")
    LOCAL_MODEL_SUPPORT = True
except ImportError:
    LOCAL_MODEL_SUPPORT = False
    print("⚠️ 필수 라이브러리(transformers, peft, torch)를 불러올 수 없습니다.")

class FeedbackEngine:
    """
    발표 분석 데이터를 기반으로 전문적인 피드백을 생성하는 엔진입니다.
    로컬 모델(Fine-tuned EXAONE 3.5)과 클라우드 모델(Gemini)을 지원합니다.
    """
    
    def __init__(self, provider: str = "exaone"):
        self.provider = provider.lower()
        self.local_model = None
        self.local_tokenizer = None
        self.device = "cuda" if HAS_CUDA else "cpu"
        
        # 로컬 모델(EXAONE LoRA) 초기화
        if self.provider == "exaone" and LOCAL_MODEL_SUPPORT:
            self._init_local_model()

    def _init_local_model(self):
        """
        RTX 5060 Ti GPU(Unsloth) 또는 노트북 CPU(Transformers) 환경에 맞춰 모델을 로드합니다.
        """
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        lora_path = os.path.join(os.path.dirname(curr_dir), "training", "exaone_presenter_lora")
        base_model_name = "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"

        try:
            if HAS_CUDA and UNSLOTH_AVAILABLE:
                print("\n--- [FeedbackEngine] GPU 모드로 모델 로드 (RTX 5060 Ti) ---")
                model, tokenizer = FastLanguageModel.from_pretrained(
                    model_name = base_model_name,
                    max_seq_length = 2048,
                    load_in_4bit = True,
                    trust_remote_code = True,
                )
                model.get_input_embeddings = lambda: model.transformer.wte
                model.transformer.get_input_embeddings = lambda: model.transformer.wte
                self.local_model = PeftModel.from_pretrained(model, lora_path)
                self.local_tokenizer = tokenizer
                FastLanguageModel.for_inference(self.local_model)
            else:
                print("\n--- [FeedbackEngine] CPU 모드로 모델 로드 (노트북) ---")
                print("※ 주의: CPU 모드는 속도가 느리며 많은 램(16GB+)이 필요합니다.")
                self.local_tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
                base_model = AutoModelForCausalLM.from_pretrained(
                    base_model_name,
                    torch_dtype=torch.float32, # CPU 호환성을 위해 float32 사용
                    device_map="cpu",
                    trust_remote_code=True
                )
                # EXAONE 임베딩 패치
                if not hasattr(base_model, "get_input_embeddings"):
                    base_model.get_input_embeddings = lambda: base_model.transformer.wte
                
                self.local_model = PeftModel.from_pretrained(base_model, lora_path)
                self.local_model.eval()

            print(f"✅ 로컬 모델 로드 완료 ({self.device} 모드)")
        except Exception as e:
            print(f"❌ 로컬 모델 로드 실패: {e}")
            self.provider = "gemini" # 실패 시 폴백

    def generate_feedback(self, project_name: str, rubric: str = "", persona: str = "soft") -> str:
        """
        프로젝트 이름을 기반으로 모든 JSON 데이터를 취합하여 최종 피드백을 생성합니다.
        """
        # 1. 자동 데이터 취합 (analysis_json 하위 폴더 검색)
        json_paths = self._find_project_json_files(project_name)
        detailed_data = self._load_json_data(json_paths)
        
        # 2. 핵심 지표 요약 (total_json 기준)
        analysis_summary = detailed_data.get("summary", {})
        
        # 3. 프롬프트 구성
        prompt = self._build_evaluation_prompt(analysis_summary, rubric, detailed_data, persona)
        
        # 4. 모델 공급자에 따른 피드백 생성
        if self.provider == "exaone" and self.local_model:
            return self._get_local_exaone_feedback(prompt)
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

    def _get_local_exaone_feedback(self, prompt: str) -> str:
        """로컬 RTX 5060 Ti에서 추론 실행"""
        inputs = self.local_tokenizer([prompt], return_tensors = "pt").to("cuda")
        outputs = self.local_model.generate(
            **inputs, 
            max_new_tokens = 1024, 
            use_cache = True,
            temperature = 0.7,
            top_p = 0.9
        )
        response = self.local_tokenizer.batch_decode(outputs)
        # assistant 답변만 추출
        try:
            return response[0].split("[|assistant|]")[1].replace(self.local_tokenizer.eos_token, "").strip()
        except:
            return response[0]

    def _get_gemini_feedback(self, prompt: str) -> str:
        try:
            response = gemini_model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Gemini 피드백 생성 중 오류 발생: {str(e)}"

    def generate_timeline_feedback(self, aligned_data: list, project_name: str, persona: str = "soft") -> Dict[str, str]:
        """
        영상 전체의 타임라인 데이터를 분석하여 주요 시점별로 짧은 AI 코칭 팁을 생성합니다.
        """
        timeline_tips = {}

        # 페르소나 가이드
        persona_guide = "냉철하게 지적해줘." if persona == "sharp" else "다정하게 격려하며 조언해줘."

        # 30초 단위 혹은 주요 이벤트(시선 이탈 등)가 있는 지점 추출 (샘플링)
        # 여기서는 간단하게 5개 핵심 지점만 추려 모델에게 요청
        sample_points = aligned_data[::len(aligned_data)//5] if len(aligned_data) > 5 else aligned_data

        for point in sample_points:
            time_sec = point.get('time', 0)
            text = point.get('text', '')
            gaze = "정면 응시" if abs(point.get('face', {}).get('gaze_h', 0)) < 0.2 else "시선 분산"

            # 짧은 팁 생성을 위한 프롬프트
            prompt = f"""[|system|]
    발표 전문가로서 짧고 명확한 한 문장 조언을 제공합니다. {persona_guide}
    [|user|]
    시간: {time_sec}초, 상황: {gaze}, 자막: "{text}"
    이 시점에 필요한 짧은 피드백 한 문장을 생성해줘.
    [|assistant|]
    """
            if self.provider == "exaone" and self.local_model:
                tip = self._get_local_exaone_feedback(prompt)
            else:
                tip = f"{time_sec}초 지점: 시선 처리에 유의하세요."

            timeline_tips[str(round(time_sec, 1))] = tip.split('\n')[0] # 첫 줄만 사용

        return timeline_tips

# 싱글톤 인스턴스 생성 (서버 시작 시 모델 로드)
# VRAM 절약을 위해 필요할 때만 생성하도록 설정할 수 있음
feedback_engine = FeedbackEngine(provider="exaone")
