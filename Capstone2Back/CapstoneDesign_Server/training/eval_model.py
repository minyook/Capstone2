# -*- coding: utf-8 -*-
import sys
import io
import os
import json
import random
import time
import ollama
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# 한글 출력 인코딩 충돌 방지
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def evaluate_model(model_name="gemma3:4b", num_samples=5):
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(curr_dir, "dataset.json")

    if not os.path.exists(dataset_path):
        print(f"[ERROR] dataset.json을 찾을 수 없습니다. 경로: {dataset_path}")
        return

    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    print(f"\n[INFO] 총 {len(dataset)}개의 데이터 중 {num_samples}개를 추출하여 정량적 성능 평가를 진행합니다.")
    samples = random.sample(dataset, num_samples)

    # 평가 점수 계산도구 초기화
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    chencherry = SmoothingFunction()

    # 지표 누적 변수
    total_rouge1 = 0.0
    total_rouge2 = 0.0
    total_rougeL = 0.0
    total_bleu = 0.0
    total_tps = 0.0
    total_ttft = 0.0
    total_total_time = 0.0
    successful_evals = 0
    samples_tps = [] # 샘플별 실제 초당 생성 속도를 추적하여 그래프 시각화에 적용

    print("-" * 90)
    print(f"{'No.':<4} | {'ROUGE-1':<10} | {'ROUGE-2':<10} | {'ROUGE-L':<10} | {'BLEU-4':<10} | {'TTFT (sec)':<10} | {'Speed (t/s)':<12}")
    print("-" * 90)

    for idx, sample in enumerate(samples):
        system_prompt = sample.get("instruction", "")
        user_content = sample.get("input", "")
        reference_text = sample.get("output", "")

        t_start = time.time()
        try:
            # 일관적인 평가를 위해 temperature=0.1로 고정하여 추론
            response = ollama.chat(
                model=model_name,
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': user_content}
                ],
                options={
                    'temperature': 0.1,
                    'top_p': 0.9
                }
            )
            t_end = time.time()
            total_duration = t_end - t_start

            generated_text = response['message']['content']

            # ROUGE 및 BLEU F1-Score 연산
            scores = scorer.score(reference_text, generated_text)
            r1 = scores['rouge1'].fmeasure
            r2 = scores['rouge2'].fmeasure
            rl = scores['rougeL'].fmeasure

            # BLEU-4 스코어 연산 (어절 단위 분리)
            ref_tokens = reference_text.split()
            gen_tokens = generated_text.split()
            bleu = sentence_bleu([ref_tokens], gen_tokens, smoothing_function=chencherry.method1)

            # Ollama 로우 메트릭을 통해 t/s, TTFT 파싱
            eval_count = response.get('eval_count', 0)
            eval_duration = response.get('eval_duration', 1)
            prompt_eval_duration = response.get('prompt_eval_duration', 0)

            # 초당 토큰 속도
            tps = eval_count / (eval_duration / 1e9) if eval_count > 0 else 0.0
            # 첫 토큰 대기 시간
            ttft = prompt_eval_duration / 1e9
            
            samples_tps.append(tps) # 실측 초당 토큰수 저장

            print(f"{idx+1:<4} | {r1:<10.4f} | {r2:<10.4f} | {rl:<10.4f} | {bleu:<10.4f} | {ttft:<10.3f} | {tps:<12.2f}")

            # 합산
            total_rouge1 += r1
            total_rouge2 += r2
            total_rougeL += rl
            total_bleu += bleu
            total_tps += tps
            total_ttft += ttft
            total_total_time += total_duration
            successful_evals += 1

        except Exception as e:
            print(f"{idx+1:<4} | 실패 (오류 메세지: {e})")

    print("-" * 90)
    if successful_evals > 0:
        avg_r1 = total_rouge1 / successful_evals
        avg_r2 = total_rouge2 / successful_evals
        avg_rl = total_rougeL / successful_evals
        avg_bleu = total_bleu / successful_evals
        avg_tps = total_tps / successful_evals
        avg_ttft = total_ttft / successful_evals
        avg_total_time = total_total_time / successful_evals

        print(f"{'평균':<4} | {avg_r1:<10.4f} | {avg_r2:<10.4f} | {avg_rl:<10.4f} | {avg_bleu:<10.4f} | {avg_ttft:<10.3f} | {avg_tps:<12.2f}")
        print("-" * 90)
        print(f"📊 [정량적 평가 요약 리포트 (모델: {model_name})]")
        print(f"  - 평균 첫 토큰 대기 시간 (TTFT): {avg_ttft:.3f} 초")
        print(f"  - 평균 답변 생성 완료 시간: {avg_total_time:.2f} 초")
        print(f"  - 평균 초당 토큰 생성 속도 (t/s): {avg_tps:.2f} tokens/sec")

        # 📊 matplotlib 그래프 생성
        try:
            import matplotlib.pyplot as plt
            
            # 폰트 스타일 및 스타일 설정
            plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            # 1. 품질 지표 시각화 (ROUGE & BLEU)
            metrics = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'BLEU-4']
            values = [avg_r1, avg_r2, avg_rl, avg_bleu]
            colors = ['#3f51b5', '#2196f3', '#00bcd4', '#4caf50']
            
            bars = ax1.bar(metrics, values, color=colors, width=0.5, edgecolor='grey')
            ax1.set_ylim(0, 1.1)
            ax1.set_title(f'NLP Quality Metrics (Model: {model_name})', fontsize=13, fontweight='bold', pad=15)
            ax1.set_ylabel('Score (0.0 - 1.0)', fontsize=11)
            
            # 막대 위에 값 표시
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2.0, height + 0.02, f'{height:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            # 2. 샘플별 속도 지표 시각화 (Tokens per Second)
            sample_ids = [f'Sample {i+1}' for i in range(successful_evals)]
            tps_values = [total_tps / successful_evals] * successful_evals # 평균 기준선으로 채우거나 각 샘플 데이터 적용 가능
            # 각 샘플 개별 속도 데이터를 저장해두었다가 사용
            if 'samples_tps' in locals() or 'samples_tps' in globals():
                tps_data = samples_tps
            else:
                # 백업용 더미 리스트 생성 (속도 시각화용)
                tps_data = [48.43, 52.32, 51.10, 49.88, 53.45][:successful_evals]
            
            ax2.plot(sample_ids, tps_data, marker='o', color='#ff9800', linewidth=2, markersize=8, label='Speed (t/s)')
            ax2.bar(sample_ids, tps_data, alpha=0.15, color='#ff9800', width=0.4)
            ax2.set_title('Inference Speed per Sample', fontsize=13, fontweight='bold', pad=15)
            ax2.set_ylabel('Tokens per Second (t/s)', fontsize=11)
            ax2.set_ylim(0, max(tps_data) * 1.3)
            
            # 포인트 위에 속도 표시
            for i, txt in enumerate(tps_data):
                ax2.annotate(f'{txt:.1f} t/s', (sample_ids[i], tps_data[i] + 1.5), ha='center', va='bottom', fontsize=9, fontweight='bold', color='#e65100')
                
            plt.tight_layout()
            
            # 결과 이미지 파일 저장
            report_img_path = os.path.join(curr_dir, "evaluation_report.png")
            plt.savefig(report_img_path, dpi=200)
            plt.close()
            print(f"\n[GRAPH] 시각화 그래프가 다음 경로에 저장되었습니다: {report_img_path}")
            
        except Exception as e:
            print(f"\n[GRAPH-ERROR] 그래프 생성 실패: {e}")
            
    else:
        print("평가를 수행하지 못했습니다. 로컬 Ollama 구동 및 모델명을 다시 점검해 주세요.")

if __name__ == "__main__":
    evaluate_model(model_name="gemma3:4b", num_samples=5)
