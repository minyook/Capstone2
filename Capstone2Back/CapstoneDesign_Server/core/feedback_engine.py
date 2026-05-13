import os
import json
import sys
import io
import glob
import torch
from pathlib import Path
from typing import Dict, Any, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 터미널 출력 한글 깨짐 방지
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

class FeedbackEngine:
    def __init__(self, provider: str = "exaone"):
        self.provider = provider.lower()
        self.local_model = None
        self.local_tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if self.provider == "exaone":
            self._init_local_model()

    def _init_local_model(self):
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        base_model_name = "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct"
        lora_path = os.path.join(os.path.dirname(curr_dir), "training", "exaone_presenter_lora")

        print(f"\n--- [FeedbackEngine] 모델 로드 중 ({self.device} 모드) ---")
        try:
            # 1. 토크나이저 로드
            self.local_tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
            
            # 2. 베이스 모델 로드
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            ).to(self.device)
            
            # 3. LoRA 어댑터 강제 로드 (학습한 내용 적용)
            if os.path.exists(lora_path):
                print(f"   > [LoRA] 학습된 가중치를 적용합니다: {lora_path}")
                self.local_model = PeftModel.from_pretrained(base_model, lora_path)
            else:
                print("   > [Warn] LoRA 폴더를 찾을 수 없어 기본 모델로 로드합니다.")
                self.local_model = base_model
                
            self.local_model.eval()
            print(f"✅ 모델 및 학습 데이터 로드 완료!")
        except Exception as e:
            print(f"❌ 모델 로드 실패: {e}")
            self.provider = "gemini"

    def generate_feedback(self, project_name: str, rubric: str = "", persona: str = "soft") -> str:
        # 데이터 취합 (기존 로직)
        json_paths = self._find_project_json_files(project_name)
        detailed_data = self._load_json_data(json_paths)
        analysis_summary = detailed_data.get("summary", {})
        
        # 프롬프트 구성 (학습 때와 동일한 포맷)
        prompt_style = """[|system|]
발표 자료 구성 및 시각화 전문가로서 사용자의 요청에 대해 전문적이고 구체적인 피드백을 제공합니다.
[|user|]
제시된 기술 분석 데이터(PPT, Whisper, YOLO, MediaPipe)를 기반으로, 영역별 5줄 이상의 상세 피드백을 포함한 분석 보고서를 작성해줘.
[{project_name}] PPT: 텍스트 면적 {face_rate:.1f}%, 이미지 있음 / Whisper: 필러워드 분당 {speed:.1f}회 / YOLO: 상체 흔들림 높음 / MediaPipe: 시선 응시율 {gaze:.1f}%
[|assistant|]
"""
        # 데이터 매핑 (안전한 기본값 설정)
        prompt = prompt_style.format(
            project_name=project_name,
            face_rate=analysis_summary.get('face_detection_rate', 50.0),
            speed=analysis_summary.get('avg_speed', 5.0),
            gaze=analysis_summary.get('gaze_score', 0.5) * 100
        )

        if self.provider == "exaone" and self.local_model:
            print(f"   > [AI] 학습된 지식을 바탕으로 심층 리포트 생성 중... (CPU 모드)")
            inputs = self.local_tokenizer([prompt], return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.local_model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    temperature=0.7,
                    repetition_penalty=1.2,
                    eos_token_id=self.local_tokenizer.eos_token_id
                )
            response = self.local_tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
            return response.split("[|assistant|]")[1].strip() if "[|assistant|]" in response else response
        else:
            return "모델 로드 오류로 피드백을 생성할 수 없습니다."

    def _find_project_json_files(self, project_name: str) -> Dict[str, Path]:
        base_dir = Path("Capstone2Back/CapstoneDesign_Server/analysis_json")
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
        return {"0.0": "학습된 AI 코치가 실시간 분석을 시작합니다."}

feedback_engine = FeedbackEngine(provider="exaone")
