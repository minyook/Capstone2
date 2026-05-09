import os
import sys
import io
import json
import glob

# 터미널 출력 한글 깨짐 방지
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Triton 캐시 경로
os.environ["TRITON_CACHE_DIR"] = "C:/temp/triton_cache"

from unsloth import FastLanguageModel
import torch
from peft import PeftModel

def load_analysis_data(project_name, base_dir="Capstone2Back/CapstoneDesign_Server/analysis_json"):
    """
    프로젝트 이름을 기반으로 각 폴더에서 JSON 데이터를 취합합니다.
    """
    data_summary = {}
    
    # 1. Total JSON 찾기 (가장 기본 데이터)
    total_files = glob.glob(os.path.join(base_dir, "total_json", f"*{project_name}*.json"))
    if total_files:
        with open(total_files[0], 'r', encoding='utf-8') as f:
            total_data = json.load(f)
            data_summary['summary'] = total_data.get('summary', {})
            data_summary['metadata'] = total_data.get('metadata', {})
    
    # 2. 개별 데이터 보완 (필요한 경우)
    # 여기서는 total_json에 이미 요약이 잘 되어 있으므로 summary 데이터를 주로 사용합니다.
    
    return data_summary

def format_input_from_data(data):
    """
    취합된 JSON 데이터를 모델 입력용 텍스트로 변환합니다.
    """
    summary = data.get('summary', {})
    metadata = data.get('metadata', {})
    
    input_text = f"""
    발표 모드: {metadata.get('video_type', 'N/A')}
    전체 시간: {metadata.get('total_time', 0)}초
    얼굴 검출률: {summary.get('face_detection_rate', 0)}%
    시선 집중도 점수: {summary.get('gaze_score', 0):.2f}
    미소 점수: {summary.get('smile_score', 0):.2f}
    제스처 상태: {summary.get('gesture_status', 'N/A')}
    평균 발화 속도: {summary.get('avg_speed', 0):.2f} cps
    PPT 분석 결과: {summary.get('ppt_summary', 'N/A')}
    """
    return input_text.strip()

def run_expert_feedback(project_name):
    # 1. 데이터 로드
    print(f"\n--- '{project_name}' 프로젝트 데이터 분석 중 ---")
    analysis_data = load_analysis_data(project_name)
    if not analysis_data:
        print(f"오류: '{project_name}'에 해당하는 데이터를 찾을 수 없습니다.")
        return

    formatted_input = format_input_from_data(analysis_data)
    print("\n[추출된 분석 지표]:")
    print(formatted_input)

    # 2. 모델 로드
    print("\n--- 베이스 모델 및 LoRA 어댑터 로드 중 (RTX 5060 Ti) ---")
    base_model_path = "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"
    # 현재 파일 위치 기준으로 lora_path 설정
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    lora_path = os.path.join(curr_dir, "exaone_presenter_lora")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = base_model_path,
        max_seq_length = 2048,
        load_in_4bit = True,
        trust_remote_code = True,
    )

    # Exaone 임베딩 패치
    model.get_input_embeddings = lambda: model.transformer.wte
    model.transformer.get_input_embeddings = lambda: model.transformer.wte

    model = PeftModel.from_pretrained(model, lora_path)
    FastLanguageModel.for_inference(model)

    # 3. 프롬프트 생성
    prompt_style = """[|system|]
발표 자료 구성 및 시각화 전문가로서 사용자의 발표 분석 데이터를 바탕으로 개선을 위한 전문 피드백을 제공합니다.
[|user|]
다음 발표 분석 데이터를 바탕으로 종합적인 피드백을 주세요:
{}
[|assistant|]
{}"""

    inputs = tokenizer(
        [
            prompt_style.format(formatted_input, "")
        ], return_tensors = "pt").to("cuda")

    # 4. 답변 생성
    print("\n--- 전문가 피드백 생성 중 ---")
    outputs = model.generate(
        **inputs, 
        max_new_tokens = 1024, 
        use_cache = True,
        temperature = 0.8,
        top_p = 0.9,
    )
    response = tokenizer.batch_decode(outputs)

    # 5. 결과 출력
    final_response = response[0].split("[|assistant|]")[1].replace(tokenizer.eos_token, "").strip()
    print("\n" + "="*60)
    print(f"[{project_name}] 발표에 대한 AI 전문가 상세 피드백")
    print("-" * 60)
    print(final_response)
    print("="*60)

if __name__ == "__main__":
    # 테스트할 프로젝트 이름 (사용자가 말한 adiotest 사용)
    test_project = "adiotest"
    run_expert_feedback(test_project)
