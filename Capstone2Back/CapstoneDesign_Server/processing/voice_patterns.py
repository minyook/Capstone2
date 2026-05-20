"""
STT·운율·무음 기반 발화 습관 분석 및 채점기준표(전달의 안정성) 음성 항목 점수화.
신체 평정심은 태도(제스처) 영역에서 별도 평가합니다.
"""
from __future__ import annotations

import array
import re
import wave
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

_FILLER_RES = [
    re.compile(r"(?:^|[\s,.!?…])어+(?:[\s,.!?…]|$)"),
    re.compile(r"(?:^|[\s,.!?…])음+(?:[\s,.!?…]|$)"),
    re.compile(r"(?:^|[\s,.!?…])(?:그{2,}|그냥|그니까|그러니까)(?:[\s,.!?…]|$)"),
    re.compile(r"(?:^|[\s,.!?…])(?:뭐{1,2}|뭐야|뭐지)(?:[\s,.!?…]|$)"),
    re.compile(r"(?:^|[\s,.!?…])(?:저{2,}|쫌|좀)(?:[\s,.!?…]|$)"),
    re.compile(r"(?:^|[\s,.!?…])이제(?:[\s,.!?…]|$)"),
    re.compile(r"(?:^|[\s,.!?…])자(?:[,.\s]|$)"),
    re.compile(r"(?:^|[\s,.!?…])아+(?:[\s,.!?…]|$)"),
]

_NGRAM_LEN = 8
_MIN_REPEAT = 2
_SILENCE_MIN_SEC = 0.35
_SILENCE_MERGE_GAP = 0.08
_FRAME_MS = 20.0
_FRAME_SAMPLES = 320


def _read_wav_mono(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as w:
        if w.getsampwidth() != 2:
            raise ValueError("16-bit PCM WAV만 지원합니다.")
        sr = w.getframerate()
        nch = w.getnchannels()
        raw = w.readframes(w.getnframes())
    buf = array.array("h")
    buf.frombytes(raw)
    samples = np.asarray(buf, dtype=np.int64)
    if nch > 1:
        samples = samples.reshape(-1, nch).mean(axis=1).astype(np.int16)
    return samples.astype(np.float64), sr


def _detect_silence_intervals(samples: np.ndarray, sample_rate: int) -> list[dict[str, float]]:
    if samples.size == 0:
        return []
    hop = int(sample_rate * (_FRAME_MS / 1000.0))
    rms_vals: list[float] = []
    pos = 0
    n = len(samples)
    while pos + _FRAME_SAMPLES <= n:
        chunk = samples[pos : pos + _FRAME_SAMPLES]
        rms_vals.append(float(np.sqrt(np.mean(chunk * chunk)) + 1e-9))
        pos += hop
    if not rms_vals:
        return []
    med = float(np.median(np.array(rms_vals)))
    thresh = max(100.0, min(1200.0, med * 0.32))
    silent_flags = [v < thresh for v in rms_vals]
    frame_sec = _FRAME_MS / 1000.0
    intervals: list[tuple[float, float]] = []
    run_start: int | None = None
    for i, is_silent in enumerate(silent_flags):
        if is_silent and run_start is None:
            run_start = i
        elif not is_silent and run_start is not None:
            t0, t1 = run_start * frame_sec, i * frame_sec
            if t1 - t0 >= _SILENCE_MIN_SEC:
                intervals.append((t0, t1))
            run_start = None
    if run_start is not None:
        t0 = run_start * frame_sec
        t1 = len(silent_flags) * frame_sec
        if t1 - t0 >= _SILENCE_MIN_SEC:
            intervals.append((t0, t1))
    if not intervals:
        return []
    merged: list[tuple[float, float]] = [intervals[0]]
    for t0, t1 in intervals[1:]:
        p0, p1 = merged[-1]
        if t0 - p1 <= _SILENCE_MERGE_GAP:
            merged[-1] = (p0, max(p1, t1))
        else:
            merged.append((t0, t1))
    return [{"start": round(a, 2), "end": round(b, 2), "duration_sec": round(b - a, 2)} for a, b in merged]


def _count_fillers_in_text(text: str) -> tuple[int, list[str]]:
    hits: list[str] = []
    for rx in _FILLER_RES:
        for m in rx.finditer(text):
            hits.append(m.group(0).strip())
    return len(hits), hits


def _build_char_time_map(segments: list[dict[str, Any]]) -> tuple[str, list[tuple[int, int, float, float]]]:
    chunks: list[str] = []
    mapping: list[tuple[int, int, float, float]] = []
    cursor = 0
    for seg in segments:
        t = str(seg.get("text", ""))
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", start))
        stripped = re.sub(r"\s+", "", t)
        if not stripped:
            continue
        n = len(stripped)
        mapping.append((cursor, cursor + n - 1, start, end))
        chunks.append(stripped)
        cursor += n
    return "".join(chunks), mapping


def _char_index_to_time(idx: int, mapping: list[tuple[int, int, float, float]]) -> float | None:
    for lo, hi, ts, te in mapping:
        if lo <= idx <= hi:
            span = max(hi - lo, 1)
            return ts + (te - ts) * ((idx - lo) / span)
    return None


def _non_overlapping_hits(starts: list[int], length: int) -> int:
    starts = sorted(starts)
    cnt, last = 0, -10**9
    for s in starts:
        if s - last >= length:
            cnt += 1
            last = s
    return cnt


def _find_repeated_ngrams(norm_text: str, mapping: list[tuple[int, int, float, float]]) -> list[dict[str, Any]]:
    if len(norm_text) < _NGRAM_LEN * 2:
        return []
    pos: dict[str, list[int]] = defaultdict(list)
    for i in range(0, len(norm_text) - _NGRAM_LEN + 1):
        pos[norm_text[i : i + _NGRAM_LEN]].append(i)
    out: list[dict[str, Any]] = []
    for phrase, starts in pos.items():
        occ = _non_overlapping_hits(starts, _NGRAM_LEN)
        if occ < _MIN_REPEAT:
            continue
        t0 = _char_index_to_time(starts[0], mapping)
        if t0 is None:
            continue
        out.append(
            {
                "type": "repeated_phrase",
                "phrase": phrase,
                "count": occ,
                "time_sec": round(t0, 2),
                "detail": f"같은 {len(phrase)}글자 구절이 발화 중 약 {occ}회 반복",
            }
        )
    out.sort(key=lambda x: (-x["count"], x["time_sec"]))
    return out[:12]


def _prosody_stats(segments: list[dict[str, Any]]) -> dict[str, Any]:
    jitters = [float(s.get("jitter", 0) or 0) for s in segments]
    shimmers = [float(s.get("shimmer", 0) or 0) for s in segments]
    if not jitters:
        return {
            "jitter_mean": 0.0,
            "shimmer_mean": 0.0,
            "jitter_p90": 0.0,
            "shimmer_p90": 0.0,
            "high_variability_segments": [],
        }
    j_arr = np.array(jitters, dtype=np.float64)
    s_arr = np.array(shimmers, dtype=np.float64)
    high_j = float(np.percentile(j_arr, 75))
    high_s = float(np.percentile(s_arr, 75))
    spikes: list[dict[str, Any]] = []
    for seg in segments:
        j = float(seg.get("jitter", 0) or 0)
        sh = float(seg.get("shimmer", 0) or 0)
        if j >= high_j and j > 0.5 and sh >= high_s and sh > 0.5:
            spikes.append(
                {
                    "type": "voice_variability",
                    "time_sec": round(float(seg.get("start", 0)), 2),
                    "end_sec": round(float(seg.get("end", 0)), 2),
                    "jitter": round(j, 3),
                    "shimmer": round(sh, 3),
                    "detail": "목소리 변동(지터·시머)이 구간 평균보다 높음",
                }
            )
    spikes.sort(key=lambda x: -(x["jitter"] + x["shimmer"]))
    return {
        "jitter_mean": round(float(np.mean(j_arr)), 3),
        "shimmer_mean": round(float(np.mean(s_arr)), 3),
        "jitter_p90": round(float(np.percentile(j_arr, 90)), 3),
        "shimmer_p90": round(float(np.percentile(s_arr, 90)), 3),
        "high_variability_segments": spikes[:8],
    }


def analyze_voice_behavior(
    whisper_segments: list[dict[str, Any]],
    aligned_data: list[dict[str, Any]],
    audio_path: Path | None,
) -> dict[str, Any]:
    filler_total = 0
    segment_fillers: list[dict[str, Any]] = []
    for seg in whisper_segments:
        t = str(seg.get("text", ""))
        n, hits = _count_fillers_in_text(t)
        if n:
            filler_total += n
            segment_fillers.append(
                {
                    "type": "filler",
                    "time_sec": round(float(seg.get("start", 0)), 2),
                    "end_sec": round(float(seg.get("end", 0)), 2),
                    "count": n,
                    "samples": hits[:4],
                    "detail": "말버릇(필러) 구간",
                }
            )
    norm_text, char_map = _build_char_time_map(whisper_segments)
    repetitions = _find_repeated_ngrams(norm_text, char_map)
    prosody = _prosody_stats(whisper_segments)

    silences: list[dict[str, float]] = []
    if audio_path and audio_path.exists():
        try:
            samples, sr = _read_wav_mono(audio_path)
            silences = _detect_silence_intervals(samples, sr)
        except Exception:
            silences = []

    speech_rates = [
        float(x.get("speech_rate_cps", 0) or 0) for x in aligned_data if x.get("speech_rate_cps") is not None
    ]
    avg_cps = float(np.mean(speech_rates)) if speech_rates else 0.0

    evidence: list[dict[str, Any]] = []
    for s in segment_fillers[:6]:
        quote = ""
        for x in whisper_segments:
            if abs(float(x.get("start", 0)) - float(s["time_sec"])) < 0.05:
                quote = str(x.get("text", ""))[:160]
                break
        evidence.append({**s, "quote": quote})
    for r in repetitions[:5]:
        evidence.append(r)
    for p in prosody.get("high_variability_segments", [])[:4]:
        evidence.append(p)
    for sil in silences[:6]:
        evidence.append(
            {
                "type": "hesitation_pause",
                "time_sec": sil["start"],
                "end_sec": sil["end"],
                "duration_sec": sil["duration_sec"],
                "detail": "짧은 무음(망설임·호흡 후보)",
            }
        )
    evidence.sort(key=lambda x: float(x.get("time_sec", 0)))

    duration = max((float(s.get("end", 0) or 0) for s in whisper_segments), default=0.0)
    fillers_per_min = (filler_total / duration) * 60.0 if duration > 0.5 else float(filler_total)

    return {
        "filler_total": filler_total,
        "fillers_per_minute": round(fillers_per_min, 2),
        "repeated_phrase_hits": len(repetitions),
        "silence_pause_count": len(silences),
        "silence_total_sec": round(sum(s["duration_sec"] for s in silences), 2),
        "avg_speech_rate_cps": round(avg_cps, 3),
        "prosody": prosody,
        "repetitions": repetitions,
        "silences": silences[:20],
        "evidence": evidence[:22],
    }


def _score_voice_stability_10(jm: float, sm: float) -> int:
    """음성 안정도 (10점): 지터·시머 낮을수록 고득점."""
    variability = jm + sm
    if variability <= 1.0:
        return 10
    if variability <= 2.5:
        return 9
    if variability <= 4.0:
        return 8
    if variability <= 6.0:
        return 7
    if variability <= 8.0:
        return 6
    if variability <= 12.0:
        return 5
    return max(0, min(4, int(10 - variability * 0.5)))


def _score_linguistic_fluency_10(vm: dict[str, Any]) -> int:
    """언어적 유창성 (10점): 필러 분당 1회 미만 목표."""
    fpm = float(vm.get("fillers_per_minute", 0) or 0)
    rep = int(vm.get("repeated_phrase_hits", 0) or 0)
    pause_n = int(vm.get("silence_pause_count", 0) or 0)

    if fpm < 1.0:
        score = 10
    elif fpm < 2.0:
        score = 8
    elif fpm < 4.0:
        score = 6
    elif fpm < 6.0:
        score = 4
    else:
        score = 2

    score -= min(3, rep)
    score -= min(2, pause_n // 5)
    return int(max(0, min(10, score)))


def _score_speech_speed_100(cps: float) -> int:
    """말하는 속도 (0~100): 발표용 약 5~7 글자/초를 적정으로 봄."""
    if cps <= 0:
        return 0
    if 5.0 <= cps <= 7.5:
        return 95
    if 4.0 <= cps < 5.0 or 7.5 < cps <= 8.5:
        return 85
    if 3.0 <= cps < 4.0 or 8.5 < cps <= 10.0:
        return 70
    if 2.0 <= cps < 3.0 or 10.0 < cps <= 12.0:
        return 55
    return 40


def _score_voice_stability_100(jm: float, sm: float) -> int:
    """목소리 안정성 (0~100): jitter·shimmer % 스케일 분리."""
    # shimmer: Praat local % (대개 5~12), jitter: 대개 0.5~2
    shimmer_penalty = max(0.0, (sm - 5.0) * 4.5)
    jitter_penalty = max(0.0, (jm - 0.8) * 12.0)
    return int(max(0, min(100, round(100 - shimmer_penalty - jitter_penalty))))


def _score_filler_habit_100(fpm: float, pause_n: int) -> int:
    """말버릇·필러 (0~100)."""
    if fpm < 1.0:
        score = 95
    elif fpm < 2.0:
        score = 85
    elif fpm < 4.0:
        score = 70
    elif fpm < 6.0:
        score = 55
    else:
        score = 40
    score -= min(15, pause_n // 4 * 3)
    return int(max(0, min(100, score)))


def _score_repetition_100(rep_hits: int) -> int:
    """같은 구절 반복 (0~100)."""
    if rep_hits <= 0:
        return 90
    if rep_hits == 1:
        return 80
    if rep_hits <= 3:
        return 65
    return max(40, 90 - rep_hits * 12)


def voice_scores_from_metrics(vm: dict[str, Any]) -> dict[str, Any]:
    """항목기준표 「발표 음성」4항목 (각 0~100) + UI·하위 호환 필드."""
    if not vm:
        return {
            "items_100": [0, 0, 0, 0],
            "category_100": 0,
            "voice_stability": 0,
            "linguistic_fluency": 0,
            "total_voice_rubric_20": 0,
        }

    jm = float(vm.get("prosody", {}).get("jitter_mean", 0) or 0)
    sm = float(vm.get("prosody", {}).get("shimmer_mean", 0) or 0)
    cps = float(vm.get("avg_speech_rate_cps", 0) or 0)
    fpm = float(vm.get("fillers_per_minute", 0) or 0)
    rep = int(vm.get("repeated_phrase_hits", 0) or 0)
    pause_n = int(vm.get("silence_pause_count", 0) or 0)

    speed = _score_speech_speed_100(cps)
    stability = _score_voice_stability_100(jm, sm)
    fillers = _score_filler_habit_100(fpm, pause_n)
    repetition = _score_repetition_100(rep)
    items = [speed, stability, fillers, repetition]
    category = int(round(sum(items) / len(items)))

    voice_stability_10 = _score_voice_stability_10(jm, sm)
    linguistic_fluency_10 = _score_linguistic_fluency_10(vm)

    return {
        "items_100": items,
        "category_100": category,
        "speech_speed": speed,
        "voice_stability_item": stability,
        "filler_control": fillers,
        "repetition_control": repetition,
        "voice_stability": voice_stability_10,
        "linguistic_fluency": linguistic_fluency_10,
        "total_voice_rubric_20": voice_stability_10 + linguistic_fluency_10,
    }
