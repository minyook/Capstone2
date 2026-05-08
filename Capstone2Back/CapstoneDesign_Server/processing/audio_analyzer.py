import whisper
import parselmouth 
import os
import json
from dotenv import load_dotenv
from pathlib import Path
import numpy as np

# ❗️ 로컬 모델을 전역 변수로 관리하여 한번만 로드
model = None
VOICE_JSON_DIR = Path("analysis_json/Voice_json")

def _format_timestamp(seconds: float) -> str:
    mins, secs = divmod(seconds, 60)
    hours, mins = divmod(mins, 60)
    return f"{int(hours):02d}:{int(mins):02d}:{int(secs):02d}.{int((seconds % 1) * 100):02d}"


def save_voice_data(segments: list, error_message: str | None = None, video_filename: str = "default") -> None:
    """
    음성 분석 결과를 최종 채점 결합용 JSON으로 저장합니다.
    - key: 구간 시작 시각 타임스탬프
    - value: STT/운율/속도 정보
    """
    payload: dict[str, object] = {
        "source": "audio_analyzer",
        "segment_count": len(segments),
        "segments": {},
    }
    if error_message:
        payload["error"] = error_message

    segment_map: dict[str, object] = {}
    for idx, segment in enumerate(segments):
        start = float(segment.get("start", 0.0))
        end = float(segment.get("end", start))
        text = str(segment.get("text", "")).strip()
        duration = max(0.0, end - start)
        speech_rate_cps = (len(text) / duration) if duration > 0 else 0.0

        jitter = float(segment.get("jitter", 0.0))
        shimmer = float(segment.get("shimmer", 0.0))
        if np.isnan(jitter):
            jitter = 0.0
        if np.isnan(shimmer):
            shimmer = 0.0

        segment_map[_format_timestamp(start)] = {
            "segment_index": idx,
            "start": round(start, 3),
            "end": round(end, 3),
            "duration": round(duration, 3),
            "text": text,
            "speech_rate_cps": round(speech_rate_cps, 3),
            "prosody": {
                "jitter": round(jitter, 3),
                "shimmer": round(shimmer, 3),
            },
        }

    payload["segments"] = segment_map
    VOICE_JSON_DIR.mkdir(parents=True, exist_ok=True)
    voice_json_path = VOICE_JSON_DIR / f"{video_filename}_voice.json"
    with open(voice_json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4, ensure_ascii=False)
    print(f"   > Voice JSON 저장 완료: {voice_json_path}")

def load_local_whisper_model():
    """
    서버 시작 시 로컬 Whisper 모델을 로드합니다.
    """
    global model
    if model:
        return model

    print("   > [AI 1/3] ❗️ 로컬 음성인식 AI(Whisper 'small' 모델) 로드 중...")
    try:
        model = whisper.load_model("small") 
        print("   > [AI 1/3] ✅ 로컬 Whisper 모델 로드 완료.")
        return model
    except Exception as e:
        print(f"❌ 로컬 Whisper 모델 로드 중 심각한 오류 발생: {e}")
        raise

def transcribe_audio_with_timestamps(audio_path: str, video_filename: str = "default"):
    """
    로컬 Whisper 모델을 사용하여 타임스탬프가 찍힌 텍스트(대본)를 반환합니다.
    """
    global model
    if not model:
        err = "Whisper 모델이 서버에 로드되지 않았습니다. 서버 로그를 확인하세요."
        save_voice_data([], error_message=err, video_filename=video_filename)
        return [], err

    print(f"   > [4/6] ❗️ 로컬 음성 인식(Whisper) 실행 중... (시간 소요)")

    try:
        result = model.transcribe(audio_path, language="ko", fp16=False) 
        print("   > [4/6] ✅ 음성 인식 완료.")
        return result["segments"], None 

    except Exception as e:
        print(f"❌ 로컬 Whisper 실행 오류: {e}")
        err = str(e)
        save_voice_data([], error_message=err, video_filename=video_filename)
        return [], err 

# ⭐️ [수정] 음성 운율(목소리 떨림) 분석 함수 로직 수정
def analyze_prosody_for_segments(audio_path: Path, segments: list, video_filename: str = "default") -> list:
    """
    Whisper가 나눠놓은 'segments' 시간대별로 Jitter와 Shimmer를 계산합니다.
    (segments 리스트를 직접 수정하여 반환합니다)
    """
    print(f"   > [5/6] ❗️ 음성 운율(목소리 떨림) 분석 중... (Praat)")
    try:
        snd = parselmouth.Sound(str(audio_path))

        for segment in segments:
            start_time = segment['start']
            end_time = segment['end']

            part = snd.extract_part(from_time=start_time, to_time=end_time, preserve_times=True)

            pitch = part.to_pitch()

            point_process = parselmouth.praat.call(pitch, "To PointProcess")

            jitter_local = parselmouth.praat.call(point_process, "Get jitter (local)", 0.0, 0.0, 0.0001, 0.02, 1.3)

            shimmer_local = parselmouth.praat.call([part, point_process], "Get shimmer (local)", 0.0, 0.0, 0.0001, 0.02, 1.3, 1.6)

            segment['jitter'] = jitter_local * 100  
            segment['shimmer'] = shimmer_local * 100 

            if np.isnan(segment['jitter']): segment['jitter'] = 0
            if np.isnan(segment['shimmer']): segment['shimmer'] = 0

        print(f"   > [5/6] ✅ 음성 운율 분석 완료.")
        save_voice_data(segments, video_filename=video_filename)
        return segments 

    except Exception as e:
        print(f"   > [5/6] ⚠️  음성 운율 분석 경고: {e}")
        for segment in segments:
            if 'jitter' not in segment:
                segment['jitter'] = 0
            if 'shimmer' not in segment:
                segment['shimmer'] = 0
        save_voice_data(segments, error_message=str(e), video_filename=video_filename)
        return segments