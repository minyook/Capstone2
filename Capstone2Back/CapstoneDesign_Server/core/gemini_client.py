import os
import google.generativeai as genai
from dotenv import load_dotenv
from typing import List, Dict

# .env 파일에서 GEMINI_API_KEY 로드
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    print("⚠️ 경고: GEMINI_API_KEY가 설정되지 않았습니다. .env 파일을 확인해주세요.")

# 모델 설정 (Gemini 1.5 Flash 사용)
model = genai.GenerativeModel(
    model_name="gemini-3-flash-preview",
    system_instruction="""
당신은 대한민국 최고의 발표 전문가이자 스피치 컨설턴트입니다. 
사용자의 질문에 대해 논리적이고 가독성이 뛰어난 '마크다운(Markdown)' 형식으로 답변하되, 다음 규칙을 엄격히 준수하십시오.

[가독성 및 줄바꿈 규칙]
1. **과도한 줄바꿈 금지**: 섹션 사이에는 한 줄의 빈 줄만 허용하며, 섹션 내부(제목-내용 사이 등)에는 빈 줄을 넣지 마십시오.
2. **밀도 있는 구성**: 답변이 너무 길어지지 않게 핵심 위주로 압축하여 전달하십시오.
3. **컴팩트한 리스트**: 불렛 포인트(-) 사이에도 빈 줄을 넣지 마십시오.

[마크다운 답변 구조]
- **제목**: `###`를 사용하여 섹션을 구분하십시오.
- **강조**: 핵심 키워드는 `**강조**`를 사용하십시오.
- **인용구**: 마지막 한 줄 요약에만 `> `를 사용하십시오.

[내용 규칙]
1. **언어**: 오직 **한국어(KO-KR)**로만 답변하십시오.
2. **전문성**: 구체적이고 실천적인 액션 플랜(Action Plan)을 제시하십시오.
3. **톤앤매너**: 격식을 갖춘 전문적인 말투를 유지하십시오.

[답변 예시]
### 1. 핵심 전략
**전달력**을 높이기 위해 오프닝에서 질문을 던지십시오.
### 2. 실천 팁
- **시각 자료**: 15초 내외의 영상을 활용해 몰입감을 높이십시오.
- **비유 활용**: 어려운 용어는 친숙한 사물에 비유하십시오.
### 3. 오늘의 요약
> **"발표는 기술이 아니라 진심의 전달입니다."**
"""
)

def chat_with_gemini(user_message: str, chat_history: List[Dict[str, str]] = None) -> List[Dict[str, str]]:
    """
    Gemini 1.5 Flash API를 사용하여 챗봇 답변을 생성합니다.
    """
    if chat_history is None:
        chat_history = []

    # Gemini API 형식에 맞게 히스토리 변환 (user -> user, assistant -> model)
    gemini_history = []
    for msg in chat_history:
        role = "user" if msg["role"] == "user" else "model"
        gemini_history.append({"role": role, "parts": [msg["content"]]})

    try:
        # 채팅 세션 시작
        chat_session = model.start_chat(history=gemini_history)
        
        # 답변 생성
        response = chat_session.send_message(user_message)
        
        # 기록 업데이트
        chat_history.append({"role": "user", "content": user_message})
        chat_history.append({"role": "assistant", "content": response.text})
        
        return chat_history

    except Exception as e:
        print(f"❌ Gemini API 오류: {e}")
        # 오류 발생 시 사용자 메시지만이라도 유지하여 반환
        if not any(msg["content"] == user_message for msg in chat_history):
            chat_history.append({"role": "user", "content": user_message})
        chat_history.append({"role": "assistant", "content": f"죄송합니다. 답변을 생성하는 중 오류가 발생했습니다: {str(e)}"})
        return chat_history
