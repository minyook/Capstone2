import ollama

# EXAONE 3.5 기반 커스텀 코치 모델명
MODEL_NAME = "exaone-coach"

def get_feedback_from_coach(vision_audio_data: str) -> str:
    """
    [발표 코치] 분석 데이터를 기반으로 피드백 생성
    - 모델: exaone-coach (Ollama 커스텀 모델)
    """
    system_prompt = """
당신은 대한민국 최고의 발표 전문가이자 스피치 컨설턴트인 '오버나잇 AI 코치'입니다.
입력된 영상 및 음성 분석 데이터를 바탕으로 사용자의 발표에 대해 구체적이고 전문적인 피드백을 제공하십시오.

[피드백 규칙]
1. **전문성**: 시선 처리, 자세, 목소리 톤, 말하기 속도 등 다각도에서 분석하십시오.
2. **구체성**: 단순히 "잘했습니다"가 아니라, "어느 부분에서 시선이 불안정했습니다"와 같이 구체적으로 지적하십시오.
3. **긍정적 마무리**: 개선점뿐만 아니라 잘한 점도 언급하여 동기부여를 해주십시오.
4. **언어**: 반드시 한국어(KO-KR)로 답변하십시오.
5. **형식**: 가독성을 위해 마크다운 형식을 사용하십시오.
"""

    try:
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': f"다음 분석 데이터를 바탕으로 피드백을 작성해줘:\n\n{vision_audio_data}"}
            ]
        )
        return response['message']['content']
    except Exception as e:
        return f"코칭 AI(EXAONE 3.5) 응답 실패: {e}\n(모델이 설치되어 있는지 확인해주세요: ollama run exaone3.5)"

# 🧪 피드백 코치 테스트
if __name__ == "__main__":
    print(f"=== {MODEL_NAME} 피드백 코치 테스트 ===")
    mock_data = """
    - 시선 처리: 정면 응시율 40%, 좌측 40%, 우측 20%
    - 목소리: 평균 데시벨 65dB (안정적), 말하기 속도 약간 빠름
    - 자세: 어깨가 왼쪽으로 5도 정도 기울어짐
    """
    print(get_feedback_from_coach(mock_data))
