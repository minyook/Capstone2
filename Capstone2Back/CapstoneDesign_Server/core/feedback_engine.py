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

            # --- [Patch] EXAONE 모델의 Peft 호환성 이슈 해결 ---
            def patch_exaone_embeddings(model):
                model.get_input_embeddings = lambda: model.transformer.wte
                model.set_input_embeddings = lambda value: setattr(model.transformer, 'wte', value)
                model.transformer.get_input_embeddings = lambda: model.transformer.wte
                model.transformer.set_input_embeddings = lambda value: setattr(model.transformer, 'wte', value)
            
            patch_exaone_embeddings(base_model)
            # ------------------------------------------------
            
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
        
        # 프롬프트 구성 (마크다운 및 줄바꿈 강조)
        prompt_style = """[|system|]
당신은 대한민국 최고의 발표 전문가이자 스피치 컨설턴트입니다. 
입력된 기술 분석 데이터를 바탕으로 사용자의 발표에 대해 전문적인 피드백을 제공하십시오.

[출력 규칙]
1. 반드시 한국어로 답변하십시오.
2. 가독성을 위해 마크다운(Markdown) 형식을 사용하십시오.
3. 각 분석 항목별로 소제목(##)을 사용하고, 상세 내용은 글머리 기호(-)를 사용하여 5줄 이상 작성하십시오.
4. 문장 사이에 적절한 줄바꿈을 적용하여 읽기 편하게 만드십시오.
5. 마지막에는 종합적인 개선 방향을 '총평'으로 요약하십시오.

[|user|]
제시된 기술 분석 데이터(PPT, Whisper, YOLO, MediaPipe)를 기반으로 분석 보고서를 작성해줘.
[{project_name}] 
- PPT: 텍스트 면적 {face_rate:.1f}%, 이미지 포함 여부(있음)
- Whisper(음성): 필러워드 분당 {speed:.1f}회
- YOLO(자세): 상체 흔들림 분석 결과(높음)
- MediaPipe(시선): 정면 응시율 {gaze:.1f}%

[|assistant|]
"""
        # 데이터 매핑
        prompt = prompt_style.format(
            project_name=project_name,
            face_rate=analysis_summary.get('face_detection_rate', 50.0),
            speed=analysis_summary.get('avg_speed', 5.0),
            gaze=analysis_summary.get('gaze_score', 0.5) * 100
        )

        if self.provider == "exaone" and self.local_model:
            print(f"   > [AI] 학습된 지식을 바탕으로 심층 리포트 생성 중...")
            inputs = self.local_tokenizer([prompt], return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.local_model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    temperature=0.7,
                    repetition_penalty=1.2,
                    do_sample=True,
                    eos_token_id=self.local_tokenizer.eos_token_id
                )
            response = self.local_tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
            
            # 답변 추출 및 줄바꿈 정리
            if "[|assistant|]" in response:
                final_text = response.split("[|assistant|]")[1].strip()
            else:
                final_text = response.strip()
            
            # 소제목 앞뒤 줄바꿈 보강 (가독성 향상)
            final_text = final_text.replace("##", "\n\n##").replace("###", "\n###")
            return final_text
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
