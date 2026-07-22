# -*- coding: utf-8 -*-
import os

# Triton 캐시 경로 오류 해결 (한글 사용자명 대응)
triton_cache_dir = "C:/temp/triton_cache"
if not os.path.exists(triton_cache_dir):
    os.makedirs(triton_cache_dir, exist_ok=True)
os.environ["TRITON_CACHE_DIR"] = triton_cache_dir

from unsloth import FastLanguageModel
import torch
from trl import SFTConfig, SFTTrainer
from datasets import load_dataset

# 1. 모델 및 토크나이저 로드 (Gemma 3 4B Instruct 4-bit)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/gemma-3-4b-it",
    max_seq_length = 2048,
    load_in_4bit = True,
    trust_remote_code = True,
)

# 2. LoRA 설정
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, 
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
)

# 3. 데이터셋 로드 및 포맷팅 (Gemma 3 대화 포맷)
prompt_style = """[|system|]
발표 자료 구성 및 시각화 전문가로서 사용자의 발표 분석 데이터를 바탕으로 개선을 위한 전문 피드백을 제공합니다.
[|user|]
{}
[|assistant|]
{}"""

EOS_TOKEN = tokenizer.eos_token

def formatting_prompts_func(examples):
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for input_text, output in zip(inputs, outputs):
        text = prompt_style.format(input_text, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }

data_path = os.path.join(os.path.dirname(__file__), "dataset.json")
dataset = load_dataset("json", data_files=data_path, split="train")
dataset = dataset.map(formatting_prompts_func, batched = True,)

# 4. 학습 설정 (TRL 버전별 타입 힌트 불일치로 인한 IDE 빨간 줄을 방지하기 위해 type: ignore 적용)
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,            # type: ignore
    train_dataset = dataset,
    dataset_text_field = "text",      # type: ignore
    max_seq_length = 2048,            # type: ignore
    args = SFTConfig(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 10,
        max_steps = 150,
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

# 5. 학습 시작
print("Gemma 3 4B LoRA 학습을 시작합니다...")
trainer.train()

# 6. 학습된 LoRA 어댑터 저장
output_dir = "gemma3_presenter_lora"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"학습 완료! {output_dir} 폴더에 어댑터가 저장되었습니다.")
