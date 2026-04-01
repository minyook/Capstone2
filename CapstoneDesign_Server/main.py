import os
import uvicorn
import uuid 
import asyncio
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import JSONResponse

from utils.helpers import setup_temp_dirs, create_session_dirs, save_upload_file
from utils.json_helpers import setup_json_dirs
from processing.audio_analyzer import load_local_whisper_model
from processing.task_manager import run_analysis_task, job_status

# 🌟 신규 임포트
from core.exceptions import QualityException

BASE_DIR = Path(__file__).resolve().parent

@asynccontextmanager
async def lifespan(app: FastAPI):
    if os.name == 'nt':
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    print("\n" + "="*50)
    print("🚀 데이터 추출 서버 시작 (자동 분석 모드 / YOLO26 + LLaMA)")
    print("="*50)
    
    setup_temp_dirs()
    setup_json_dirs() 
    
    try:
        # setup_face_landmarker() <- 🗑️ 지움!
        load_local_whisper_model()
        
        # --- 🟢 자동 분석 하드코딩 구간 (유지) ---
        test_file = BASE_DIR / "adiotest.mp4"
        if test_file.exists():
            print(f"\n📦 테스트 파일 발견: {test_file.name}")
            print("⚙️ 즉시 분석을 시작합니다...\n")
            
            video_dir, frame_dir = create_session_dirs()
            job_id = "AUTO_TEST_001"
            job_status[job_id] = {"status": "Started"}
            
            asyncio.create_task(asyncio.to_thread(
                run_analysis_task, job_id, test_file, frame_dir, video_dir, []
            ))
        else:
            print(f"\n⚠️ 자동 분석 실패: {test_file.name} 파일이 없습니다.")

    except Exception as e:
        print(f"❌ 초기화 오류: {e}")
        
    yield
    print("\n" + "="*50)
    print("서버가 종료됩니다.")
    print("="*50)

app = FastAPI(lifespan=lifespan)

# 🌟 신규: 커스텀 예외 발생 시 JSON 에러 반환
@app.exception_handler(QualityException)
async def quality_exception_handler(request, exc: QualityException):
    return JSONResponse(status_code=exc.status_code, content={"status": "error", "message": exc.detail})

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)

    # .\venv\Scripts\activate    python main.py  http://127.0.0.1:8000
    # pip install -r requirements.txt (라이브러리 설치)
    # winget install Gyan.FFmpeg
    # ollama 홈페이지가서 다운로드
    # exe 설치하고 vscode 껏다키기
    # 가상환경이나 터미널가서 ollama pull llama3 (라마 다운로드)
    # http://127.0.0.1:8000/chat

    # --------------핸드폰으로 실행 방법-----------------
    # .\venv\Scripts\activate
    # uvicorn main:app --host 0.0.0.0 --port 8000 (서버 키기)