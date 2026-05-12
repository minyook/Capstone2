import json
import random
import os

def generate_expert_dataset(count=1000):
    dataset = []
    
    # 지식 베이스 정의 (전달해주신 링크 내용 반영)
    body_language_tips = [
        "열린 자세(Open body language)를 유지하세요. 팔짱을 끼는 것은 청중과의 심리적 장벽을 만듭니다.",
        "청중과 60-70%의 시선을 맞추는 것이 신뢰도 향상의 핵심입니다. 한 사람만 보지 말고 골고루 응시하세요.",
        "손을 주머니에 넣거나 뒤로 숨기지 마세요. 가슴과 허리 사이 구역에서 제스처를 사용하는 것이 가장 자연스럽습니다.",
        "발표 시 '뿌리 내린 자세'보다는 중요한 포인트를 강조할 때 조금씩 이동하며 에너지를 전달하는 것이 좋습니다.",
        "미소는 청중의 긴장을 완화시킵니다. 분석된 미소 수치가 낮으니 조금 더 밝은 표정을 시도해 보세요."
    ]
    
    ppt_tips = [
        "6x6 법칙(한 슬라이드에 6줄 이하, 한 줄에 6단어 이하)을 적용하여 가독성을 높이세요.",
        "슬라이드에 적힌 텍스트를 그대로 읽는 '스크립트 읽기'는 지양해야 합니다. 핵심 키워드만 배치하세요.",
        "대비(Contrast)가 명확한 폰트와 크기를 선택하여 멀리 있는 청중도 배려해야 합니다.",
        "시각적 자료(이미지, 도표)는 텍스트보다 6만 배 빠르게 뇌에 전달됩니다. 이미지를 더 활용해 보세요.",
        "슬라이드 장수가 너무 많거나 적지 않은지, 전체 흐름과 시간 배분을 다시 점검해 보세요."
    ]

    # 상황별 시나리오 세분화
    scenarios = [
        {
            "type": "시선 부족 & 정적",
            "vision_pattern": lambda: f"gaze_score: {random.uniform(0.1, 0.4):.2f}, smile: {random.uniform(0.0, 0.2):.2f}, gesture: 정적임",
            "voice_pattern": lambda: f"speed: {random.uniform(0.8, 1.1):.1f}x",
            "ppt_pattern": lambda: random.choice(["텍스트 많음", "양호", "키워드 중심"]),
            "template": "시선 점수가 매우 낮고 자세가 경직되어 있습니다. Simply Amazing Training 지침에 따라 청중과 60% 이상 눈을 맞추고, 손을 활용한 제스처로 열린 자세를 보여주세요."
        },
        {
            "type": "팔짱 감지 & 방어적",
            "vision_pattern": lambda: f"is_arm_crossed: true, gaze_score: {random.uniform(0.5, 0.8):.2f}",
            "voice_pattern": lambda: f"speed: {random.uniform(1.0, 1.3):.1f}x",
            "ppt_pattern": lambda: "양호",
            "template": "현재 팔짱을 낀 자세(Closed body language)가 감지되었습니다. 이는 전문가 가이드에서 지양하는 방어적 자세입니다. 손을 앞쪽으로 내밀어 개방적인 태도를 보여주시는 것이 좋습니다."
        },
        {
            "type": "말 빠름 & 긴장",
            "vision_pattern": lambda: f"gaze_score: {random.uniform(0.3, 0.6):.2f}, smile: {random.uniform(0.0, 0.1):.2f}",
            "voice_pattern": lambda: f"speed: {random.uniform(1.5, 2.0):.1f}x, pitch: high",
            "ppt_pattern": lambda: "텍스트 밀도 높음",
            "template": "말하기 속도가 너무 빠르고 음조가 높습니다. 긴장하신 것으로 보입니다. 문장 끝에서 의도적으로 2초간 멈추는(Pause) 연습을 통해 여유를 가지시기 바랍니다."
        },
        {
            "type": "PPT 의존 & 스크립트 읽기",
            "vision_pattern": lambda: f"gaze_score: {random.uniform(0.1, 0.3):.2f}, gesture: 정적임",
            "voice_pattern": lambda: f"speed: {random.uniform(0.9, 1.1):.1f}x",
            "ppt_pattern": lambda: "텍스트 매우 많음 (불량)",
            "template": "슬라이드의 텍스트가 너무 많아 화면을 보고 그대로 읽는 경향이 있습니다. 6x6 법칙을 적용해 텍스트를 줄이고, 청중의 눈을 보며 핵심 내용을 전달하세요."
        },
        {
            "type": "최고의 발표",
            "vision_pattern": lambda: f"gaze_score: {random.uniform(0.7, 0.9):.2f}, smile: {random.uniform(0.4, 0.7):.2f}, gesture: 활발함",
            "voice_pattern": lambda: f"speed: {random.uniform(1.1, 1.3):.1f}x",
            "ppt_pattern": lambda: "시각화 우수",
            "template": "완벽한 발표입니다! 열린 자세와 적절한 시선 처리, 그리고 말하기 속도까지 매우 조화롭습니다. SlideShare 가이드에 맞는 훌륭한 슬라이드 구성도 인상적입니다."
        }
    ]

    for _ in range(count):
        s = random.choice(scenarios)
        
        # 입력 데이터 생성
        input_text = f"[실시간 분석 로그] 시각: {s['vision_pattern']()}, 음성: {s['voice_pattern']()}, PPT: {s['ppt_pattern']()}"
        
        # 지식 기반 피드백 다양화
        feedback = s['template']
        # 랜덤하게 전문 팁 하나 추가
        if random.random() > 0.5:
            extra_tip = random.choice(body_language_tips if "시각" in input_text else ppt_tips)
            feedback += " " + extra_tip

        dataset.append({
            "instruction": "발표 분석 수치를 기반으로 전문 코칭 지침(Simply Amazing Training 및 SlideShare)을 적용한 피드백을 작성하세요.",
            "input": input_text,
            "output": feedback
        })

    # 파일 저장
    target_dir = os.path.join("Capstone2Back", "CapstoneDesign_Server", "training")
    os.makedirs(target_dir, exist_ok=True)
    target_path = os.path.join(target_dir, "dataset.json")
    
    with open(target_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=4, ensure_ascii=False)
    
    return target_path, len(dataset)

if __name__ == "__main__":
    path, size = generate_expert_dataset(1000)
    print(f"COMPLETE: Generated {size} items at {path}")
