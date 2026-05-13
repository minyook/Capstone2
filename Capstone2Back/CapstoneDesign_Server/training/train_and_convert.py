import os
import sys
import subprocess

# 1. 윈도우 한글 경로 및 인코딩 문제 해결 (최우선)
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONUTF8"] = "1"

# 외부 명령 실행 시 인코딩 충돌 방지 패치
_original_popen = subprocess.Popen
class PatchedPopen(_original_popen):
    def __init__(self, *args, **kwargs):
        if kwargs.get('text') or kwargs.get('universal_newlines'):
            kwargs['encoding'] = 'utf-8'
            kwargs['errors'] = 'replace'
        super().__init__(*args, **kwargs)
subprocess.Popen = PatchedPopen

# Triton 캐시 경로 설정
triton_cache_dir = "C:/temp/triton_cache"
if not os.path.exists(triton_cache_dir):
    os.makedirs(triton_cache_dir, exist_ok=True)
os.environ["TRITON_CACHE_DIR"] = triton_cache_dir

from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset

# 2. 모델 및 토크나이저 로드 (EXAONE-3.5-2.4B)
model_name = "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct"
max_seq_length = 2048

print(f"🚀 모델 로드 중: {model_name}")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    load_in_4bit = True,
    trust_remote_code = True,
)

# [디버깅] EXAONE 모델의 임베딩 레이어 인식 문제 수정
if hasattr(model, "transformer"):
    model.transformer.get_input_embeddings = lambda: model.transformer.wte
    model.transformer.set_input_embeddings = lambda v: setattr(model.transformer, "wte", v)

# 3. LoRA 설정
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, 
    target_modules = ["q_proj", "k_proj", "v_proj", "out_proj", "c_fc_0", "c_fc_1", "c_proj"],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
)

# 4. 데이터셋 로드 및 포맷팅
prompt_style = """[|system|]
발표 자료 구성 및 시각화 전문가로서 사용자의 요청에 대해 전문적이고 구체적인 피드백을 제공합니다.
[|user|]
{}
{}
[|assistant|]
{}"""

EOS_TOKEN = tokenizer.eos_token

def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input_text, output in zip(instructions, inputs, outputs):
        text = prompt_style.format(instruction, input_text, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }

data_path = os.path.join(os.path.dirname(__file__), "dataset.json")
dataset = load_dataset("json", data_files=data_path, split="train")
dataset = dataset.map(formatting_prompts_func, batched = True,)

# 5. 학습 설정
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 10,
        max_steps = 350, # 충분한 학습을 위해 350스텝 설정
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
)

# 6. 학습 시작
print("🚀 LoRA 파인튜닝을 시작합니다...")
trainer.train()

# 7. LoRA 어댑터 우선 저장 (안전장치)
lora_output_dir = "exaone_presenter_lora"
print(f"💾 LoRA 어댑터를 {lora_output_dir} 폴더에 저장합니다...")
model.save_pretrained(lora_output_dir)
tokenizer.save_pretrained(lora_output_dir)

# 8. GGUF 변환 시도
output_dir = "exaone_presenter_gguf"
print(f"📦 모델을 GGUF(q4_k_m) 형식으로 변환 시도 중...")

try:
    model.save_pretrained_gguf(
        output_dir, 
        tokenizer, 
        quantization_method = "q4_k_m"
    )
    print(f"✅ GGUF 변환 완료! {output_dir} 폴더를 확인하세요.")
except Exception as e:
    print(f"❌ GGUF 자동 변환 실패: {e}")
    print(f"ℹ️ 하지만 LoRA 어댑터는 '{lora_output_dir}'에 성공적으로 저장되었습니다.")

print("🏁 모든 과정이 완료되었습니다.")
