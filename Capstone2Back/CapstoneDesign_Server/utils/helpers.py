# [재구성 파일] utils/helpers.py
import os
import shutil
import uuid
from pathlib import Path
from fastapi import UploadFile

# 프로젝트 루트 디렉토리를 기준으로 경로 설정
BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
FRAME_DIR = BASE_DIR / "frames"
# ⭐️ JSON 관련 경로는 json_helpers.py로 이동

def setup_temp_dirs():
    """서버 시작 시 임시 폴더들이 있는지 확인하고 없으면 생성합니다."""
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(FRAME_DIR, exist_ok=True)
    # ⭐️ [수정] JSON 폴더 생성 로직은 json_helpers.py의 setup_json_dirs로 이동

def create_session_dirs():
    """
    각 요청마다 고유한 임시 폴더를 생성합니다.
    동시에 여러 요청이 들어와도 파일이 섞이지 않게 합니다.
    """
    session_id = str(uuid.uuid4())
    video_dir = UPLOAD_DIR / session_id
    frame_session_dir = FRAME_DIR / session_id

    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(frame_session_dir, exist_ok=True)
    
    return video_dir, frame_session_dir

def save_upload_file(upload_file: UploadFile, destination: Path) -> Path:
    """업로드된 파일을 지정된 경로에 저장합니다."""
    try:
        with destination.open("wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)
    finally:
        upload_file.file.close()
    return destination

def cleanup_dirs(*dirs: Path):
    """분석 완료 후 사용된 임시 폴더들을 재귀적으로 삭제합니다."""
    import time
    import gc
    import shutil

    # 가비지 컬렉션을 강제로 실행하여 열려있는 파일 핸들을 해제 시도
    gc.collect()

    for d in dirs:
        if d and os.path.exists(d):
            # Windows에서 가끔 파일 잠금으로 인해 삭제 실패하는 경우 대응
            for i in range(5): # 재시도 횟수 증가
                try:
                    # 읽기 전용 속성이 있으면 삭제가 안 될 수 있으므로 처리하는 내부 함수
                    def remove_readonly(func, path, excinfo):
                        import stat
                        os.chmod(path, stat.S_IWRITE)
                        func(path)

                    shutil.rmtree(d, onerror=remove_readonly)
                    print(f"   > 임시 폴더 삭제 성공: {d}")
                    break
                except Exception as e:
                    if i == 4:
                        print(f"   > 임시 폴더 삭제 최종 실패: {d}, 오류: {e}")
                    else:
                        print(f"   > 임시 폴더 삭제 대기 중... ({i+1}/5) - 사유: {e}")
                        time.sleep(2) # 대기 시간 증가