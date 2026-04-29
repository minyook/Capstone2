import ollama

def get_feedback_from_coach(vision_audio_data: str) -> str:
    """
    [피드백 코치 전용] 분석 파이프라인용 (영상 분석 후 피드백 받을 때 사용)
    - 모델: presenter-coach (Ollama 로컬 모델 사용)
    """
    try:
        response = ollama.chat(
            model='presenter-coach',
            messages=[{'role': 'user', 'content': vision_audio_data}]
        )
        return response['message']['content']
    except Exception as e:
        return f"코칭 AI 응답 실패: {e}"

# 🧪 피드백 코치 테스트
if __name__ == "__main__":
    print("=== 피드백 코치 테스트 ===")
    mock_data = "데이터: 어깨 기울어짐, 목소리 떨림 감지됨"
    print(get_feedback_from_coach(mock_data))
