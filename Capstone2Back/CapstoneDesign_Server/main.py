import os
import uvicorn
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
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
# 🌟 챗봇 함수 임포트 (Gemini로 교체)
from core.gemini_client import chat_with_gemini, stream_chat_with_gemini
from fastapi.responses import StreamingResponse

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
                # 백그라운드 없이 직접 실행
                run_analysis_task("AUTO_DEMO", test_path, Path("frames"), Path("uploads"), [])
            else:
                print(f"❌ 자동 분석 실패: {test_path} 파일을 찾을 수 없습니다.")

    except Exception as e:
        print(f"❌ 초기화 오류: {e}")
        
    yield
    print("\n" + "="*50)
    print("서버가 종료됩니다.")
    print("="*50)

from fastapi import FastAPI, BackgroundTasks, UploadFile, File, Form, Depends
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import json

app = FastAPI(lifespan=lifespan)

# 🌟 필수 디렉토리 확인 및 생성 (RuntimeError 방지)
for d in ["uploads", "processing/MediaPipe_json", "processing/Yolo_json"]:
    Path(d).mkdir(parents=True, exist_ok=True)

# 🌟 정적 파일 서버 설정
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
app.mount("/results/face", StaticFiles(directory="processing/MediaPipe_json"), name="face_results")
app.mount("/results/gesture", StaticFiles(directory="processing/Yolo_json"), name="gesture_results")

@app.get("/")
async def read_index():
    return FileResponse('analysis_viewer.html')

# ==========================================
# 🌟 1. CORS 미들웨어 설정 (프론트엔드 연결 허용)
# ==========================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"],
)

# ==========================================
# 🌟 3. 영상 업로드 및 분석 시작 API
# ==========================================
@app.post("/api/upload")
async def upload_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    job_id = str(uuid.uuid4())[:8]
    
    # 작업 전용 폴더 생성
    job_upload_dir = Path("uploads") / job_id
    job_upload_dir.mkdir(parents=True, exist_ok=True)
    
    file_extension = Path(file.filename).suffix
    save_filename = f"video{file_extension}" # 이름을 고정하거나 원본 유지
    upload_path = job_upload_dir / save_filename
    
    # 파일 저장
    content = await file.read()
    with open(upload_path, "wb") as f:
        f.write(content)
    
    # 분석 작업용 프레임 폴더
    frame_dir = Path("frames") / job_id
    frame_dir.mkdir(parents=True, exist_ok=True)
    
    # 분석 작업 백그라운드 실행
    # video_dir로 job_upload_dir를 넘겨서 해당 폴더만 정리되게 함
    # 단, 결과 확인을 위해 비디오는 남겨두고 싶다면 cleanup_dirs 호출 방식을 조정해야 함
    # 현재 task_manager.py의 cleanup_dirs는 video_dir 전체를 지우므로, 
    # 비디오를 남기려면 task_manager.py를 수정하거나 빈 경로를 넘겨야 함.
    # 일단은 task_manager.py가 지우는 것을 방지하기 위해 빈 Path()를 넘기거나 
    # 비디오 재생용으로 복사본을 유지하는 방법이 있음.
    # 여기서는 비디오를 보존하기 위해 task_manager에 video_dir 대신 None을 넘기도록 수정하는게 안전함.
    
    background_tasks.add_task(
        run_analysis_task, 
        job_id, 
        upload_path, 
        frame_dir, 
        None # video_dir를 None으로 주어 비디오 삭제 방지 (task_manager 수정 필요)
    )
    
    return {"job_id": job_id, "video_url": f"/uploads/{job_id}/{save_filename}"}

@app.get("/api/status/{job_id}")
async def get_status(job_id: str):
    status = job_status.get(job_id, {"status": "Waiting", "message": "대기 중..."})
    return status

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
    updated_history = chat_with_gemini(request.message, request.chat_history)
    
    print(f"[🤖 챗봇 AI의 답변]: {updated_history[-1]['content']}\n")
    
    # 업데이트된 전체 대화 기록을 프론트엔드로 다시 반환
    return {"chat_history": updated_history}

@app.post("/api/chat/with-file")
async def chat_with_ai_file(
    message: str = Form(...),
    chat_history: str = Form(...),
    file: UploadFile = File(...)
):
    """
    파일과 메시지를 함께 받아 처리하는 챗봇 API입니다.
    """
    import json
    history = json.loads(chat_history)
    
    print(f"\n[📱 파일 첨부 메시지]: {message}")
    print(f"[📎 첨부 파일]: {file.filename}")

    # Gemini 답변 생성 (파일 정보 포함)
    full_message = f"[첨부 파일: {file.filename}]\n\n{message}"
    updated_history = chat_with_gemini(full_message, history)
    
    return {"chat_history": updated_history}

@app.post("/api/chat/stream")
async def chat_with_ai_stream(request: ChatRequest):
    """
    Gemini 스트리밍 답변을 반환하는 API입니다.
    """
    print(f"\n[📱 프론트엔드에서 온 메시지 (스트림)]: {request.message}")
    
    return StreamingResponse(
        stream_chat_with_gemini(request.message, request.chat_history),
        media_type="text-event-stream"
    )

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


    # python main.py --test_video "adiotest.mp4" 미디어파이프 테스트 명령어