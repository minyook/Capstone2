import os
import uvicorn
import uuid 
import asyncio
import importlib.util
import json
import argparse
import sys
import shutil
from pathlib import Path
from contextlib import asynccontextmanager
from typing import List, Dict, Optional

from fastapi import FastAPI, BackgroundTasks, UploadFile, File, Form, HTTPException, Depends
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# 🌟 프로젝트 모듈 임포트
from utils.helpers import setup_temp_dirs, create_session_dirs, save_upload_file
from utils.json_helpers import setup_json_dirs
from processing.audio_analyzer import load_local_whisper_model
from processing.task_manager import run_analysis_task, job_status
from core.exceptions import QualityException
from core.gemini_client import chat_with_gemini, stream_chat_with_gemini, upload_to_gemini

# 환경 변수 설정
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
BASE_DIR = Path(__file__).resolve().parent

# PPT 엔진 관련 설정
PPT_ENGINE_DIR = BASE_DIR / "ppt-analysis-engine"
PPT_UPLOAD_DIR = PPT_ENGINE_DIR / "data" / "uploads"
PPT_JSON_DIR = BASE_DIR / "analysis_json" / "ppt_json"
_PPT_ANALYZE_FUNC = None

def _get_ppt_analyze_func():
    global _PPT_ANALYZE_FUNC
    if _PPT_ANALYZE_FUNC is not None:
        return _PPT_ANALYZE_FUNC

    engine_root = str(PPT_ENGINE_DIR.resolve())
    if engine_root not in sys.path:
        sys.path.insert(0, engine_root)

    module_path = PPT_ENGINE_DIR / "main.py"
    spec = importlib.util.spec_from_file_location("ppt_analysis_engine_main", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("PPT 분석 엔진 모듈 로드에 실패했습니다.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    analyze_func = getattr(module, "analyze_ppt_file", None)
    if analyze_func is None:
        raise RuntimeError("analyze_ppt_file 함수를 찾을 수 없습니다.")
    _PPT_ANALYZE_FUNC = analyze_func
    return _PPT_ANALYZE_FUNC

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
        
        parser = argparse.ArgumentParser()
        parser.add_argument("--test_video", type=str, help="자동 분석할 영상 경로")
        args, _ = parser.parse_known_args()
        
        if args.test_video:
            test_path = Path(args.test_video)
            if test_path.exists():
                print(f"🚀 [자동 분석 모드] '{test_path.name}' 분석을 즉시 시작합니다...")
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

# 🌟 필수 디렉토리 확인 및 생성
for d in ["uploads", "analysis_json/MediaPipe_json", "analysis_json/Yolo_json", "analysis_json/total_json", "analysis_json/Voice_json"]:
    Path(d).mkdir(parents=True, exist_ok=True)

# 🌟 정적 파일 서버 설정
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
app.mount("/results/face", StaticFiles(directory="analysis_json/MediaPipe_json"), name="face_results")
app.mount("/results/gesture", StaticFiles(directory="analysis_json/Yolo_json"), name="gesture_results")
app.mount("/results/total", StaticFiles(directory="analysis_json/total_json"), name="total_results")

# 🌟 CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def read_index():
    return FileResponse('analysis_viewer.html')

@app.get("/diagnostic")
async def read_diagnostic():
    return FileResponse('diagnostic_viewer.html')

# ==========================================
# 🌟 API 엔드포인트
# ==========================================

@app.post("/api/upload")
async def upload_video(
    background_tasks: BackgroundTasks, 
    file: UploadFile = File(...),
    persona: str = Form("soft")
):
    job_id = str(uuid.uuid4())[:8]
    filename = file.filename or "unknown_video"
    original_filename = Path(filename).stem
    
    # 작업 전용 폴더 생성
    job_upload_dir = Path("uploads") / job_id
    job_upload_dir.mkdir(parents=True, exist_ok=True)
    
    file_extension = Path(filename).suffix or ".mp4"
    save_filename = f"video{file_extension}"
    upload_path = job_upload_dir / save_filename
    
    # 파일 저장
    try:
        with open(upload_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
    finally:
        await file.close()
    
    # 분석 작업용 프레임 폴더
    frame_dir = Path("frames") / job_id
    frame_dir.mkdir(parents=True, exist_ok=True)
    
    # 파일 식별자에 job_id를 포함시켜 충돌 방지
    file_id = f"{job_id}_{original_filename}"
    
    background_tasks.add_task(
        run_analysis_task, 
        job_id, 
        upload_path, 
        frame_dir, 
        None, 
        None, 
        file_id,
        persona
    )
    
    return {"job_id": job_id, "video_url": f"/uploads/{job_id}/{save_filename}", "video_name": original_filename}

@app.get("/api/status/{job_id}")
async def get_status(job_id: str):
    status = job_status.get(job_id)
    if status:
        return status
    
    total_json_dir = Path("analysis_json/total_json")
    files = list(total_json_dir.glob(f"{job_id}_*_total.json"))
    
    if files:
        try:
            with open(files[0], 'r', encoding='utf-8') as f:
                data = json.load(f)
                return {
                    "status": "Complete",
                    "result": {
                        "video_filename": data.get("metadata", {}).get("video_filename", job_id),
                        "llama_feedback": data.get("overall_feedback"),
                        "timeline_feedback": data.get("timeline_feedback"),
                        "analysis_summary": data.get("summary"),
                        "raw_data": data.get("raw_data")
                    }
                }
        except Exception as e:
            print(f"결과 파일 읽기 오류: {e}")
    
    return {"status": "Waiting", "message": "대기 중이거나 만료된 작업입니다."}

class ChatRequest(BaseModel):
    message: str
    chat_history: List[Dict[str, str]] = []

@app.post("/api/chat")
def chat_with_ai(request: ChatRequest):
    updated_history = chat_with_gemini(request.message, request.chat_history)
    return {"chat_history": updated_history}

@app.post("/api/chat/with-file")
async def chat_with_ai_file(
    message: str = Form(...),
    chat_history: str = Form(...),
    file: UploadFile = File(...)
):
    history = json.loads(chat_history)
    temp_path = BASE_DIR / "uploads" / (file.filename or "temp_file")
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    gemini_file = upload_to_gemini(str(temp_path), mime_type=file.content_type)
    if not gemini_file:
        return JSONResponse(status_code=500, content={"message": "Gemini 파일 업로드 실패"})

    updated_history = chat_with_gemini(message, history, attachments=[gemini_file])
    return {"chat_history": updated_history}

@app.post("/api/chat/stream")
async def chat_with_ai_stream(request: ChatRequest):
    return StreamingResponse(
        stream_chat_with_gemini(request.message, request.chat_history),
        media_type="text-event-stream"
    )

@app.post("/api/ppt/analyze")
async def analyze_ppt(file: UploadFile = File(...)):
    file_name = file.filename or "temp.pptx"
    ext = Path(file_name).suffix.lower()
    if ext not in {".ppt", ".pptx"}:
        raise HTTPException(status_code=400, detail="PPT 또는 PPTX 파일만 업로드할 수 있습니다.")

    PPT_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    saved_name = f"{uuid.uuid4().hex}_{Path(file_name).name}"
    saved_path = PPT_UPLOAD_DIR / saved_name

    try:
        with open(saved_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        analyze_ppt_file = _get_ppt_analyze_func()
        PPT_JSON_DIR.mkdir(parents=True, exist_ok=True)
        result_json_path = PPT_JSON_DIR / f"{saved_path.stem}.json"
        result = analyze_ppt_file(saved_path, result_path=result_json_path)
        return {
            "status": "ok",
            "uploaded_file": saved_name,
            "result_path": result.get("result_path"),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PPT 분석 실패: {e}") from e

@app.exception_handler(QualityException)
async def quality_exception_handler(request, exc: QualityException):
    return JSONResponse(status_code=exc.status_code, content={"status": "error", "message": exc.detail})

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
