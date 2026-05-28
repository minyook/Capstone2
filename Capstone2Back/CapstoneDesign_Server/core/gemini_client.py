import os
import time
from google import genai
from google.genai import types
from dotenv import load_dotenv
from typing import List, Dict, Any

# .env 파일에서 GEMINI_API_KEY 로드
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

client = None
if GEMINI_API_KEY:
    client = genai.Client(api_key=GEMINI_API_KEY)
else:
    print("⚠️ 경고: GEMINI_API_KEY가 설정되지 않았습니다. .env 파일을 확인해주세요.")

# 모델 설정 (Files API 지원을 위해 1.5 Flash 또는 최신 모델 권장)
model_name = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash")

system_instruction = """
당신은 'Overnight AI'의 수석 컨설턴트입니다. 
사용자의 질문에 대해 전문적이고 논리적인 '마크다운(Markdown)' 형식으로 답변하십시오.
첨부된 파일(PPT, PDF 등)이 있다면, 그 내용을 실시간으로 분석하여 구체적인 피드백을 제공하십시오.

[답변 규칙]
1. **데이터 기반**: 파일의 텍스트와 이미지 컨텍스트를 정확히 인용하십시오.
2. **가독성**: 섹션을 명확히 나누고 불필요한 공백을 제거하십시오.
3. **전문성**: 발표 전략, 디자인, 논리 구조 측면에서 깊이 있는 분석을 제공하십시오.
4. **언어**: 한국어(KO-KR)로만 답변하십시오.
"""

def upload_to_gemini(path: str, mime_type: str = None):
    """
    Gemini Files API를 사용하여 파일을 업로드합니다.
    """
    if not client:
        print("❌ Gemini 클라이언트가 초기화되지 않았습니다.")
        return None

    try:
        config = {'mime_type': mime_type} if mime_type else None
        file = client.files.upload(file=path, config=config)
        print(f"   > [Files API] 파일 업로드 완료: {file.uri}")
        
        # 파일 처리가 완료될 때까지 대기 (상태 체크)
        while file.state.name == "PROCESSING":
            print(".", end="", flush=True)
            time.sleep(2)
            file = client.files.get(name=file.name)
            
        if file.state.name == "FAILED":
            raise Exception("Gemini 파일 처리 실패")
            
        return file
    except Exception as e:
        print(f"❌ Gemini 파일 업로드 오류: {e}")
        return None

def stream_chat_with_gemini(user_message: str, chat_history: List[Dict[str, str]] = None, attachments: List[Any] = None):
    """
    Gemini API를 사용하여 스트리밍 답변을 생성합니다. (파일 첨부 지원)
    """
    if not client:
        yield "오류: Gemini 클라이언트가 초기화되지 않았습니다."
        return

    if chat_history is None:
        chat_history = []

    # google-genai SDK 1.2+ 버전은 pydantic validation을 매우 엄격하게 합니다.
    # 안전하게 딕셔너리 포맷 {"role": ..., "parts": [{"text": ...}]} 로 변환하여 전달합니다.
    contents = []
    for msg in chat_history:
        role = "user" if msg.get("role") == "user" else "model"
        contents.append({
            "role": role,
            "parts": [{"text": msg.get("content", "")}]
        })

    # 메시지 구성 (텍스트 + 파일)
    parts = []
    if attachments:
        for a in attachments:
            parts.append(a)
    parts.append({"text": user_message})
    
    contents.append({
        "role": "user",
        "parts": parts
    })

    try:
        response = client.models.generate_content_stream(
            model=model_name,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction
            )
        )
        for chunk in response:
            if chunk.text:
                yield chunk.text

    except Exception as e:
        print(f"❌ Gemini Streaming API 오류: {e}")
        yield f"죄송합니다. 답변을 생성하는 중 오류가 발생했습니다: {str(e)}"

def chat_with_gemini(user_message: str, chat_history: List[Dict[str, str]] = None, attachments: List[Any] = None) -> List[Dict[str, str]]:
    """
    Gemini API를 사용하여 챗봇 답변을 생성합니다. (파일 첨부 지원)
    """
    if not client:
        return chat_history + [{"role": "assistant", "content": "오류: Gemini 클라이언트가 초기화되지 않았습니다."}]

    if chat_history is None:
        chat_history = []

    contents = []
    for msg in chat_history:
        role = "user" if msg.get("role") == "user" else "model"
        contents.append({
            "role": role,
            "parts": [{"text": msg.get("content", "")}]
        })

    # 메시지 구성 (텍스트 + 파일)
    parts = []
    if attachments:
        for a in attachments:
            parts.append(a)
    parts.append({"text": user_message})
    
    contents.append({
        "role": "user",
        "parts": parts
    })

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction
            )
        )
        
        chat_history.append({"role": "user", "content": user_message})
        chat_history.append({"role": "assistant", "content": response.text})
        
        return chat_history

    except Exception as e:
        print(f"❌ Gemini API 오류: {e}")
        if not any(msg.get("content") == user_message for msg in chat_history):
            chat_history.append({"role": "user", "content": user_message})
        chat_history.append({"role": "assistant", "content": f"오류 발생: {str(e)}"})
        return chat_history

