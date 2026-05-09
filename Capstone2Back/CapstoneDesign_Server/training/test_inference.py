import os
# Triton 캐시 경로
os.environ["TRITON_CACHE_DIR"] = "C:/temp/triton_cache"

from unsloth import FastLanguageModel
import torch
from peft import PeftModel

# 현재 파일의 디렉토리 기준으로 LoRA 폴더 경로 설정
base_path = os.path.dirname(__file__)
lora_path = os.path.join(base_path, "exaone_presenter_lora")

# 1. 베이스 모델 및 토크나이저 먼저 로드
print("\n--- 베이스 모델 로드 중 ---")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct",
    max_seq_length = 2048,
    load_in_4bit = True,
    trust_remote_code = True,
)

# 2. 로드 직후 패치 적용 (Peft 연결 전)
def patch_exaone_embeddings(model):
    model.get_input_embeddings = lambda: model.transformer.wte
    model.set_input_embeddings = lambda value: setattr(model.transformer, 'wte', value)
    model.transformer.get_input_embeddings = lambda: model.transformer.wte
    model.transformer.set_input_embeddings = lambda value: setattr(model.transformer, 'wte', value)

patch_exaone_embeddings(model)

# 3. 학습된 LoRA 어댑터 연결
print("\n--- LoRA 어댑터 연결 중 ---")
model = PeftModel.from_pretrained(model, lora_path)
FastLanguageModel.for_inference(model)

# 4. 테스트용 질문 (프롬프트) 설정
prompt_style = """[|system|]
발표 자료 구성 및 시각화 전문가로서 사용자의 요청에 대해 전문적이고 구체적인 피드백을 제공합니다.
[|user|]
{}
{}
[|assistant|]
{}"""

instruction = "발표 슬라이드 디자인에 대한 피드백을 주세요."
input_text = "슬라이드 15장, 텍스트가 많고 이미지는 거의 없음, 전체 10분 발표"

# 5. 입력 데이터 생성
inputs = tokenizer(
    [
        prompt_style.format(
            instruction,
            input_text,
            "", 
        )
    ], return_tensors = "pt").to("cuda")

# 6. 답변 생성
print("\n--- EXAONE 모델 답변 생성 중 ---")
outputs = model.generate(
    **inputs, 
    max_new_tokens = 512, 
    use_cache = True,
    temperature = 0.7,
    top_p = 0.9,
)
response = tokenizer.batch_decode(outputs)

# 7. 결과 출력
final_response = response[0].split("[|assistant|]")[1].replace(tokenizer.eos_token, "").strip()
print("\n" + "="*50)
print("[발표 전문가 피드백]:")
print(final_response)
print("="*50)
