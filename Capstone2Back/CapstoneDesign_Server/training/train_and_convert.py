import os

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

# 1. 모델 및 토크나이저 로드 (2.4B 경량 모델 사용)
model_name = "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct"
max_seq_length = 2048

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    load_in_4bit = True,
    trust_remote_code = True,
)

# 2. LoRA 설정
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, 
    target_modules = ["q_proj", "k_proj", "v_proj", "out_proj", "c_fc_0", "c_fc_1", "c_proj"],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
)

# 3. 데이터셋 로드 및 포맷팅 (EXAONE 프롬프트 스타일)
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

# 4. 학습 설정
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 60, # 테스트를 위해 60스텝으로 설정 (데이터셋 크기에 따라 조절 가능)
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
print("🚀 LoRA 파인튜닝을 시작합니다...")
trainer.train()

# 6. GGUF 변환 및 저장 (q4_k_m 양자화 방식)
# Unsloth의 save_pretrained_gguf 함수를 사용하여 Merge와 GGUF 변환을 한 번에 진행
output_dir = "exaone_presenter_gguf"

print(f"📦 모델을 GGUF(q4_k_m) 형식으로 변환하여 {output_dir} 폴더에 저장합니다...")
model.save_pretrained_gguf(
    output_dir, 
    tokenizer, 
    quantization_method = "q4_k_m"
)

print(f"✅ 학습 및 GGUF 변환 완료! {output_dir} 폴더를 확인하세요.")
