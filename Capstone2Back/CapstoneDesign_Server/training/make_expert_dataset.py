# -*- coding: utf-8 -*-
import json
import random
import os

def generate_expert_dataset(count=1000):
    dataset = []
    
    scenarios = ["대학 전공 과제 발표", "기업 IR 투자 피칭", "기술 세미나 강연", "사내 분기 실적 보고", "신규 프로젝트 제안서 발표"]
    grades = ["S (최우수)", "B (보통)", "D (보완 필요)"]

    # 지식 베이스 정의 (영역별 피드백 조각)
    kb = {
        "visual": {
            "S": [
                "[현상 파악] PPT 슬라이드의 시각적 밸런스가 매우 뛰어나며, 핵심 키워드 중심의 레이아웃이 80% 이상 적용되었습니다. [원인/이론적 분석] '단순함의 미학'을 인지 심리학적으로 잘 녹여내어 정보의 위계 구조가 명확히 설계된 결과입니다. [인지적 영향] 청중은 복잡한 텍스트를 해독하는 대신 화자의 메시지를 즉각적으로 수용할 수 있는 최적의 인지 상태를 유지하게 됩니다. [구체적 개선 솔루션] 현재의 디자인 톤앤매너를 유지하되, 핵심 데이터 수치에 강조색(Point Color)을 하나 더 추가하여 시각적 정점을 형성하시길 권장합니다."
            ],
            "B": [
                "[현상 파악] 전체 슬라이드 중 약 50%의 면적이 텍스트로 채워져 있어 시각적 여백이 다소 부족한 상태입니다. [원인/이론적 분석] 이는 구두로 설명해야 할 내용을 화면에 모두 담으려는 '발표자의 불안감'에서 기인한 설계 오류입니다. [인지적 영향] 과도한 텍스트 밀도는 청중에게 인지 과부하를 유발하여 발표 내용의 핵심을 놓치게 만드는 장벽이 됩니다. [구체적 개선 솔루션] '6x6 법칙'을 적용하여 한 슬라이드에 6줄 이하, 한 줄에 6단어 이하로 텍스트를 과감히 정리하십시오."
            ],
            "D": [
                "[현상 파악] PPT 텍스트 면적이 {text_area}%에 달하며 이미지가 전무하여 시각 자료로서의 기능이 상실된 상태입니다. [원인/이론적 분석] 발표 자료를 시각 도구가 아닌 '읽는 스크립트'로 활용하고 있는 전형적인 텍스트 의존형 패턴입니다. [인지적 영향] 청중은 화자의 말에 집중하지 못하고 화면의 긴 글을 읽는 데 뇌 자원을 소모하게 되어 전달력이 급격히 하락합니다. [구체적 개선 솔루션] 전체 텍스트의 70%를 삭제하고 핵심 헤드라인만 남기십시오."
            ]
        },
        "stability": {
            "S": [
                "[현상 파악] 발화의 유창성이 매우 높으며 필러 워드가 분당 {filler}회 미만으로 거의 나타나지 않는 안정적인 상태입니다. [원인/이론적 분석] 충분한 리허설을 통해 스크립트가 체득되었으며, 호흡 조절이 심리학적으로 안정 궤도에 올랐음을 보여줍니다. [인지적 영향] 끊김 없는 유려한 발화는 청중에게 발표자에 대한 깊은 신뢰감과 권위를 심어주어 메시지의 수용도를 극대화합니다. [구체적 개선 솔루션] 현재의 안정감을 유지하면서 중요한 강조점 직전에 1~2초간의 '전략적 멈춤(Power Pause)'을 활용해 보십시오."
            ],
            "B": [
                "[현상 파악] 필러 워드가 분당 {filler}회 감지되며, 특정 문장 끝에서 음성이 미세하게 떨리는 지터(Jitter) 현상이 관찰됩니다. [원인/이론적 분석] 발표 중반부의 논리 구조가 복잡해지면서 뇌의 부하가 음성 근육의 조절력을 일시적으로 약화시킨 결과입니다. [인지적 영향] 미세한 음성 떨림은 청중에게 발표자가 긴장하고 있다는 신호를 주어 전문성을 의심하게 만드는 요인이 됩니다. [구체적 개선 솔루션] 문장과 문장 사이에서 의도적으로 깊은 복식 호흡을 수행하고 필러 워드가 나올 타이밍에 차라리 침묵하십시오."
            ],
            "D": [
                "[현상 파악] 필러 워드가 분당 {filler}회 이상으로 과도하며 상체 흔들림 빈도가 매우 높아 산만한 인상을 줍니다. [원인/이론적 분석] 준비 부족으로 인한 극도의 심리적 불안이 신체적 '흔들림'과 발화의 '막힘'으로 표출되는 전형적인 케이스입니다. [인지적 영향] 청중은 화자의 메시지보다 불안정한 태도와 불필요한 추임새에 신경이 분산되어 발표의 본질을 놓치게 됩니다. [구체적 개선 솔루션] 스크립트를 문장 단위가 아닌 키워드 단위로 다시 암기하고, 발표 시 발바닥 전체에 힘을 주어 지면을 지탱하십시오."
            ]
        },
        "nonverbal": {
            "S": [
                "[현상 파악] 청중 응시율(Gaze)이 {gaze}%로 매우 높으며, 제스처가 명확하게 수행됩니다. [원인/이론적 분석] 비언어적 소통의 중요성을 정확히 인지하고 있으며 청중과의 심리적 교감을 능동적으로 주도하고 있는 상태입니다. [인지적 영향] 적극적인 시선 맞춤과 개방적인 제스처는 화자의 자신감을 증명하며 청중이 발표 내용에 몰입하게 만드는 강력한 도구가 됩니다. [구체적 개선 솔루션] 현재의 우수한 태도에 더해, 제스처의 속도에 완급 조절을 주어 중요한 정보를 강조할 때 손끝의 힘을 실어 보십시오."
            ],
            "B": [
                "[현상 파악] 시선 응시율이 {gaze}% 수준으로 무난하지만 주로 특정 방향의 청중에게만 쏠려 있는 편향성이 보입니다. [원인/이론적 분석] 심리적으로 편안함을 느끼는 구역에만 시선을 고정하려는 '안전지대 편향' 현상이 나타나고 있습니다. [인지적 영향] 시선을 받지 못하는 나머지 구역의 청중들은 소외감을 느끼게 되어 전체적인 반응도가 떨어질 수 있습니다. [구체적 개선 솔루션] 발표 공간을 3등분하여 'M자' 혹은 'Z자' 형태로 시선을 골고루 분산시키는 연습을 하십시오."
            ],
            "D": [
                "[현상 파악] 시선 응시율이 {gaze}%로 극히 저조하며 대부분 바닥이나 슬라이드만을 응시하는 폐쇄적 자세를 보입니다. [원인/이론적 분석] 청중과의 시각적 접촉을 회피하려는 방어 기제가 작동하여 비언어적 채널이 완전히 닫혀 있는 상태입니다. [인지적 영향] 시선이 닿지 않는 발표는 일방적인 독백으로 전락하며 청중은 화자와의 신뢰 관계를 형성하는 데 실패하게 됩니다. [구체적 개선 솔루션] 슬라이드가 아닌 청중의 이마나 인중을 보는 것부터 시작하십시오."
            ]
        }
    }

    for i in range(count):
        scenario = random.choice(scenarios)
        text_area = random.randint(10, 85)
        filler = random.randint(1, 15)
        gaze = random.randint(5, 95)
        
        v_grade = "S" if text_area < 30 else ("B" if text_area < 60 else "D")
        s_grade = "S" if filler < 3 else ("B" if filler < 8 else "D")
        n_grade = "S" if gaze > 70 else ("B" if gaze > 30 else "D")

        # Input Data 구성
        input_data = {
            "project_name": f"test_project_{i}",
            "full_transcript": f"안녕하세요. 오늘 {scenario} 발표를 맡은 발표자입니다. 자료를 보시면서 말씀드리겠습니다.",
            "avg_speed": round(random.uniform(3.5, 7.5), 2),
            "fillers_per_minute": round(float(filler), 1),
            "filler_total": filler * 2,
            "silence_count": random.randint(0, 5),
            "repeated_phrase_hits": random.randint(0, 3),
            "gaze": float(gaze),
            "gesture_status": "활발함" if n_grade == "S" else "정적임",
            "smile": round(random.uniform(0.1, 0.9), 2),
            "face_rate": round(random.uniform(70.0, 100.0), 1),
            "ppt_summary": f"슬라이드 수: {random.randint(5, 20)}, 가독성: {random.uniform(0.6, 0.9):.2f}",
            "ppt_readability": round(random.uniform(0.5, 0.95), 2),
            "ppt_balance": round(random.uniform(0.5, 0.95), 2),
            "ppt_consistency": round(random.uniform(0.5, 0.95), 2)
        }

        user_content = f"""[Input Data JSON - 초정밀 멀티모달 분석 결과]
- 프로젝트 명: {input_data['project_name']}

[audio_analysis - 발표 대본 (Whisper STT 전문)]
{input_data['full_transcript']}

[audio_analysis - VAD & STT 정량 로그]
- 평균 발화 속도: {input_data['avg_speed']} cps (글자/초)
- 필러 워드(어, 음, 그, 아 등): 분당 {input_data['fillers_per_minute']}회 (총 {input_data['filler_total']}회 감지)
- 3초 이상 장기 침묵(VAD 무음 구간): {input_data['silence_count']}회 감지
- 발화 실수 및 꼬인 문맥 감지(Speech Errors): {input_data['repeated_phrase_hits']}회 감지

[vision_analysis - 3D YOLO Pose & Gaze 메트릭]
- 카메라(청중) 정면 응시율: {input_data['gaze']}%
- 측면 서 있기 상태(어깨 Yaw 변위): {input_data['gesture_status']}
- 양손 손목 3D 변위 분산(제스처 역동성): {input_data['smile'] * 100:.1f}/100
- 얼굴 검출률: {input_data['face_rate']}%

[slide_analysis - PPT 시각 자료 메트릭]
- PPT 연동 정보: {input_data['ppt_summary']}
- 가독성 점수: {input_data['ppt_readability'] * 100:.1f}/100
- 레이아웃 균형 점수: {input_data['ppt_balance'] * 100:.1f}/100
- 일관성 점수: {input_data['ppt_consistency'] * 100:.1f}/100"""

        # 점수 계산
        att_gaze = round(gaze / 10)
        att_motion = 9 if n_grade == "S" else (7 if n_grade == "B" else 4)
        att_total = att_gaze + att_motion

        voc_stab = random.randint(6, 10)
        voc_calm = random.randint(6, 10)
        voc_control = 9 if s_grade == "S" else (7 if s_grade == "B" else 4)
        voc_total = voc_stab + voc_calm + voc_control

        con_sync = random.randint(10, 15)
        con_delivery = max(2, 10 - input_data['repeated_phrase_hits'] * 2)
        con_layout = round(input_data['ppt_balance'] * 15)
        con_theme = round(input_data['ppt_consistency'] * 10)
        con_total = con_sync + con_delivery + con_layout + con_theme

        output_text = f"""## 📊 1. 멀티모달 정량 분석 요약
| 카테고리 | 지표 | 측정 수치 및 상태 |
| :--- | :--- | :--- |
| **발화 유창성** | 3초 이상 무음 감지 횟수 | {input_data['silence_count']}회 감지 |
| | 필러 워드 감지 횟수 | 총 {input_data['filler_total']}회 (분당 {input_data['fillers_per_minute']}회) |
| **시각적 역동성** | 평균 청중 응시율 | {input_data['gaze']}% |
| | 어깨 Yaw 변위 상태 | {input_data['gesture_status']} |
| **시각 자료 완성도** | 가독성 점수 | {input_data['ppt_readability'] * 100:.1f}점 |
| | 슬라이드 일관성 점수 | {input_data['ppt_consistency'] * 100:.1f}점 |

- **발화 유창성 요약**: VAD 감지 무음 {input_data['silence_count']}회, 필러 워드 분당 {input_data['fillers_per_minute']}회 감지.
- **시각적 역동성 요약**: 청중 정면 응시율 {input_data['gaze']}%, 몸체 상태는 {input_data['gesture_status']} 수준입니다.
- **시각 자료 완성도 요약**: PPT 가독성 {input_data['ppt_readability'] * 100:.1f}점, 일관성 {input_data['ppt_consistency'] * 100:.1f}점입니다.

## 🎯 2. AI 초정밀 100점 만점 발표 채점표
### 🏆 종합 스피치 스코어: {att_total + voc_total + con_total} / 100점
> **[종합 한줄 총평]** 정밀 분석 결과 전반적으로 양호한 발표 전달력을 보여주고 있습니다.

#### 🟢 [20점] 발표 태도 및 모션 스펙트럼 (Attitude) — {att_total}점 / 20점
- 👁️ **시선 집중도** (10점 만점): {att_gaze}점 — {random.choice(kb['nonverbal'][n_grade]).format(gaze=gaze)}
- 🤸 **제스처 및 포스쳐** (10점 만점): {att_motion}점 — 어깨 Yaw 움직임이 {input_data['gesture_status']} 상태를 보여주고 있습니다.

#### 🔵 [30점] 음성 유창성 및 전달 밸런스 (Voice) — {voc_total}점 / 30점
- 🔊 **음성 피치 안정도** (10점 만점): {voc_stab}점 — 음성 대역폭이 균형을 이루며 발성 안정성이 우수합니다.
- 🧘 **스피치 평정심 지수** (10점 만점): {voc_calm}점 — 무음 구간 {input_data['silence_count']}회로 적절한 호흡 템포를 조율하고 있습니다.
- 🎙️ **불필요한 필러워드 조절** (10점 만점): {voc_control}점 — {random.choice(kb['stability'][s_grade]).format(filler=filler)}

#### 🟡 [50점] 컨텐츠 가독성 및 맥락 동기화 (Content) — {con_total}점 / 50점
- 📂 **발화-PPT 싱크로율** (15점 만점): {con_sync}점 — 발화 타이밍과 PPT 슬라이드 전개 연동 상태가 양호합니다.
- 📢 **대본 가치 전달력** (10점 만점): {con_delivery}점 — 발화 실수 {input_data['repeated_phrase_hits']}회 감지로 문맥 전달 완성도가 높습니다.
- 🎨 **슬라이드 레이아웃 균형** (15점 만점): {con_layout}점 — {random.choice(kb['visual'][v_grade]).format(text_area=text_area)}
- 📏 **슬라이드 테마 일관성** (10점 만점): {con_theme}점 — 전체 장표의 템플릿과 테마 통일도가 균일합니다.

[SCORES_START]
{{
  "attitude": {{ "category": {att_total}, "items": [{att_gaze}, {att_motion}] }},
  "voice": {{ "category": {voc_total}, "items": [{voc_stab}, {voc_calm}, {voc_control}] }},
  "content": {{ "category": {con_total}, "items": [{con_sync}, {con_delivery}, {con_layout}, {con_theme}] }}
}}
[SCORES_END]"""

        dataset.append({
            "instruction": "제시된 기술 분석 데이터(PPT, Whisper, YOLO, MediaPipe)를 기반으로, 영역별 상세 피드백을 포함한 분석 보고서를 작성해줘.",
            "input": user_content,
            "output": output_text
        })

    # 저장 경로를 현재 디렉토리 기준 절대 경로로 명시하여 꼬임 방지
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    target_path = os.path.join(curr_dir, "dataset.json")
    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=4, ensure_ascii=False)
    
    return target_path, len(dataset)

if __name__ == "__main__":
    path, size = generate_expert_dataset(1000)
    print(f"✅ 성공: {size}개의 데이터셋이 '{path}'에 저장되었습니다.")
