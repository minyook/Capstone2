import os
import uvicorn
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import uuid 
import asyncio
import importlib.util
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, BackgroundTasks, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
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
from core.gemini_client import chat_with_gemini, stream_chat_with_gemini, upload_to_gemini

BASE_DIR = Path(__file__).resolve().parent
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
for d in ["uploads", "analysis_json/MediaPipe_json", "analysis_json/Yolo_json", "analysis_json/total_json", "analysis_json/Voice_json"]:
    Path(d).mkdir(parents=True, exist_ok=True)

# 🌟 정적 파일 서버 설정
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
app.mount("/results/face", StaticFiles(directory="analysis_json/MediaPipe_json"), name="face_results")
app.mount("/results/gesture", StaticFiles(directory="analysis_json/Yolo_json"), name="gesture_results")
app.mount("/results/total", StaticFiles(directory="analysis_json/total_json"), name="total_results")

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
async def upload_video(
    background_tasks: BackgroundTasks, 
    file: UploadFile = File(...),
    persona: str = Form("soft")
):
    job_id = str(uuid.uuid4())[:8]
    original_filename = Path(file.filename).stem # 확장자 제외 파일명 추출
    
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
    
    background_tasks.add_task(
        run_analysis_task, 
        job_id, 
        upload_path, 
        frame_dir, 
        None, # video_dir를 None으로 주어 비디오 삭제 방지
        None, # custom_criteria
        original_filename, # 원본 파일명 추가 전달
        persona # 페르소나 전달
    )
    
    return {"job_id": job_id, "video_url": f"/uploads/{job_id}/{save_filename}", "video_name": original_filename}

@app.get("/api/status/{job_id}")
async def get_status(job_id: str):
    # 1. 메모리에서 현재 작업 상태 확인
    status = job_status.get(job_id)
    if status:
        return status
    
    # 2. 메모리에 없으면(서버 재시작 등) 저장된 파일 확인
    total_json_dir = Path("analysis_json/total_json")
    # 파일명에 job_id가 포함된 total.json 찾기
    files = list(total_json_dir.glob(f"*{job_id}*_total.json"))
    
    if files:
        try:
            with open(files[0], 'r', encoding='utf-8') as f:
                data = json.load(f)
                return {
                    "status": "Complete",
                    "result": {
                        "llama_feedback": data.get("overall_feedback"),
                        "timeline_feedback": data.get("timeline_feedback"),
                        "analysis_summary": data.get("summary"),
                        "raw_data": data.get("raw_data")
                    }
                }
        except:
            pass

    return {"status": "Waiting", "message": "대기 중이거나 만료된 작업입니다."}

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
    파일과 메시지를 함께 받아 Gemini Files API를 통해 실시간으로 처리하는 챗봇 API입니다.
    """
    import json
    import shutil
    history = json.loads(chat_history)
    
    print(f"\n[📱 파일 첨부 메시지]: {message}")
    print(f"[📎 첨부 파일]: {file.filename}")

    # 1. 파일을 임시 폴더에 저장
    temp_path = BASE_DIR / "uploads" / file.filename
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 2. Gemini Files API로 업로드 및 분석
    gemini_file = upload_to_gemini(str(temp_path), mime_type=file.content_type)
    
    if not gemini_file:
        return JSONResponse(status_code=500, content={"message": "Gemini 파일 업로드 실패"})

    # 3. Gemini 답변 생성 (파일 객체 포함)
    updated_history = chat_with_gemini(message, history, attachments=[gemini_file])
    
    return {"chat_history": updated_history}
    
    return {"chat_history": updated_history}

@app.post("/api/ppt/analyze")
async def analyze_ppt(file: UploadFile = File(...)):
    file_name = file.filename or ""
    ext = Path(file_name).suffix.lower()
    if ext not in {".ppt", ".pptx"}:
        raise HTTPException(status_code=400, detail="PPT 또는 PPTX 파일만 업로드할 수 있습니다.")

    PPT_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    saved_name = f"{uuid.uuid4().hex}_{Path(file_name).name}"
    saved_path = PPT_UPLOAD_DIR / saved_name

    try:
        save_upload_file(file, saved_path)
        analyze_ppt_file = _get_ppt_analyze_func()
        PPT_JSON_DIR.mkdir(parents=True, exist_ok=True)
        result_json_path = PPT_JSON_DIR / f"{saved_path.stem}.json"
        result = analyze_ppt_file(saved_path, result_path=result_json_path)
        return {
            "status": "ok",
            "uploaded_file": saved_name,
            "result_path": result.get("result_path"),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PPT 분석 실패: {e}") from e

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
    #2. 노트북으로 옮겨야 할 필수 파일/폴더

    # 💻 [노트북 시연 가이드] - 모델 실행 및 환경 설정
    # 1. 필수 파일/폴더 이동 (데스크탑 -> 노트북)
     #    - LoRA 어댑터: training/exaone_presenter_lora (전체 폴더)
     #    - 데이터셋: training/dataset.json
     #    - 프론트엔드 설정: .env (Firebase Key)
     #    - 백엔드 소스: core/, processing/, utils/, schemas/, main.py
     #
     # 2. 노트북 설치 및 실행 과정 (터미널)
     #    (1) 가상환경 생성 및 활성화
    #        python -m venv venv
    #        .\venv\Scripts\activate
    #    (2) 라이브러리 설치
    #        pip install -r requirements.txt
    #    (3) 허깅페이스 로그인 (최초 1회 필수)
    #        pip install huggingface_hub
    #        huggingface-cli login
    #        # 생성하신 토큰(Hugging Face Token) 입력