import os
import uvicorn
import uuid 
import asyncio
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import JSONResponse
import argparse
import sys


# 🌟 CORS 및 챗봇 데이터 처리를 위한 추가 임포트
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict

from utils.helpers import setup_temp_dirs, create_session_dirs, save_upload_file
from utils.json_helpers import setup_json_dirs
from processing.audio_analyzer import load_local_whisper_model
from processing.task_manager import run_analysis_task, job_status

# 🌟 신규 임포트
from core.exceptions import QualityException
# 🌟 챗봇 함수 임포트
from core.llama_client import chat_with_mentor

BASE_DIR = Path(__file__).resolve().parent

@asynccontextmanager
async def lifespan(app: FastAPI):
    if os.name == 'nt':
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    print("\n" + "="*50)
    print("🚀 Overnight.AI 서버 시작 완료")
    print("="*50)
    
    setup_temp_dirs()
    setup_json_dirs() 
    
    try:
        load_local_whisper_model()
        print("✅ AI 모델 로드 완료! 클라이언트(앱)의 요청을 대기 중입니다...\n")
        
        #아래 3줄포함 else까지 테스트용으로 추가한것
        parser = argparse.ArgumentParser()
        parser.add_argument("--test_video", type=str, help="자동 분석할 영상 경로")
        args, _ = parser.parse_known_args()
        
        if args.test_video:
            test_path = Path(args.test_video)
            if test_path.exists():
                print(f"🚀 [자동 분석 모드] '{test_path.name}' 분석을 즉시 시작합니다...")
                # 발표용이므로 별도 백그라운드 없이 직접 실행
                run_analysis_task("AUTO_DEMO", test_path, Path("frames"), Path("uploads"), [])
            else:
                print(f"❌ 자동 분석 실패: {test_path} 파일을 찾을 수 없습니다.")

    except Exception as e:
        print(f"❌ 초기화 오류: {e}")
        
    yield
    print("\n" + "="*50)
    print("서버가 종료됩니다.")
    print("="*50)

app = FastAPI(lifespan=lifespan)

# ==========================================
# 🌟 1. CORS 미들웨어 설정 (프론트엔드 연결 허용)
# ==========================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # 프론트엔드(리액트, 앱) 요청 허용
    allow_credentials=True,
    allow_methods=["*"], # POST, GET 등 모든 방식 허용
    allow_headers=["*"],
)

# ==========================================
# 🌟 2. 챗봇 API 엔드포인트 및 데이터 모델 세팅
# ==========================================
class ChatRequest(BaseModel):
    message: str
    chat_history: List[Dict[str, str]] = [] # 이전 대화 기록 보관용

@app.post("/api/chat")
def chat_with_ai(request: ChatRequest):
    """
    프론트엔드(React)에서 사용자의 채팅과 이전 대화 기록을 보내면,
    LLaMA 챗봇이 문맥을 파악해 답변을 돌려주는 API입니다.
    """
    print(f"\n[📱 프론트엔드에서 온 메시지]: {request.message}")
    
    # 챗봇 AI에게 메시지와 기록을 던져서 답변 생성
    updated_history = chat_with_mentor(request.message, request.chat_history)
    
    print(f"[🤖 챗봇 AI의 답변]: {updated_history[-1]['content']}\n")
    
    # 업데이트된 전체 대화 기록을 프론트엔드로 다시 반환
    return {"chat_history": updated_history}

# ==========================================
# 기존 코드 (예외 처리 및 서버 실행)
# ==========================================
# 🌟 신규: 커스텀 예외 발생 시 JSON 에러 반환
@app.exception_handler(QualityException)
async def quality_exception_handler(request, exc: QualityException):
    return JSONResponse(status_code=exc.status_code, content={"status": "error", "message": exc.detail})

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)

    # .\venv\Scripts\activate    빽에서python main.py  프론트에서 npm run dev  http://127.0.0.1:8000
    # pip install -r requirements.txt (라이브러리 설치)
    # winget install Gyan.FFmpeg
    # ollama 홈페이지가서 다운로드
    # exe 설치하고 vscode 껏다키기
    # 가상환경이나 터미널가서 ollama pull llama3 (라마 다운로드)
    # http://127.0.0.1:8000/chat

    # --------------핸드폰으로 실행 방법-----------------
    # .\venv\Scripts\activate
    # uvicorn main:app --host 0.0.0.0 --port 8000 (서버 키기)