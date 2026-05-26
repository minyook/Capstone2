"""PPT 슬라이드 이미지 ↔ 영상 프레임 매칭 (전체화면 슬라이드)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import imagehash
import numpy as np
from PIL import Image

from slide_verify.audience_roi import AudienceScreenResult, crop_slide_content_bgr, load_audience_frame_bgr
from slide_verify.pip_roi import detect_largest_face, load_pip_frame_bgr

# 비교용 공통 해상도 (PPT PNG · 영상 프레임 동일 스케일)
COMPARE_W, COMPARE_H = 640, 360
HASH_SIZE = 12
HASH_BITS = HASH_SIZE * HASH_SIZE
# 해밍 거리 → 유사도 (PiP는 UI·압축 차이로 임계값 완화)
DIST_STRONG_FULL = int(HASH_BITS * 0.22)
DIST_WEAK_FULL = int(HASH_BITS * 0.38)
DIST_STRONG_PIP = int(HASH_BITS * 0.30)
DIST_WEAK_PIP = int(HASH_BITS * 0.50)
DIST_STRONG_AUDIENCE = int(HASH_BITS * 0.36)
DIST_WEAK_AUDIENCE = int(HASH_BITS * 0.62)


def _dist_thresholds(video_type: str) -> tuple[int, int]:
    if video_type == "pip":
        return DIST_STRONG_PIP, DIST_WEAK_PIP
    if video_type == "audience":
        return DIST_STRONG_AUDIENCE, DIST_WEAK_AUDIENCE
    return DIST_STRONG_FULL, DIST_WEAK_FULL


@dataclass
class FrameMatch:
    slide_index: int
    confidence: float
    best_dist: float
    margin: float


@dataclass
class Segment:
    start_sec: float
    end_sec: float
    ppt_slide_index: int
    confidence: float
    frame_count: int


def _load_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def _load_frame_bgr(path: Path) -> np.ndarray | None:
    return cv2.imread(str(path))


def _normalize_slide(img: Image.Image, video_type: str = "fullscreen") -> Image.Image:
    """PPT PNG: 비율 유지 레터박스 후 리사이즈."""
    img = img.convert("RGB")
    if video_type == "audience":
        bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        bgr = crop_slide_content_bgr(bgr)
        img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    src = np.array(img)
    h, w = src.shape[:2]
    scale = min(COMPARE_W / w, COMPARE_H / h)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(src, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((COMPARE_H, COMPARE_W, 3), 255, dtype=np.uint8)
    x0 = (COMPARE_W - nw) // 2
    y0 = (COMPARE_H - nh) // 2
    canvas[y0 : y0 + nh, x0 : x0 + nw] = resized
    return Image.fromarray(canvas)


def _normalize_frame(
    path: Path,
    video_type: str = "fullscreen",
    audience_meta: list[AudienceScreenResult | None] | None = None,
    frame_index: int = 0,
) -> Image.Image:
    """영상 프레임 전처리 후 슬라이드와 동일 크기로 맞춤."""
    if video_type == "pip":
        bgr, _ = load_pip_frame_bgr(path)
        if bgr is None:
            return _normalize_slide(_load_rgb(path))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return _normalize_slide(Image.fromarray(rgb))

    if video_type == "audience":
        bgr, meta = load_audience_frame_bgr(path)
        if audience_meta is not None:
            audience_meta[frame_index] = meta
        if bgr is None:
            return _normalize_slide(_load_rgb(path))
        bgr = crop_slide_content_bgr(bgr)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return _normalize_slide(Image.fromarray(rgb))

    bgr = _load_frame_bgr(path)
    if bgr is None:
        return _normalize_slide(_load_rgb(path))
    h, w = bgr.shape[:2]
    mx, my = int(w * 0.04), int(h * 0.04)
    crop = bgr[my : h - my, mx : w - mx]
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    return _normalize_slide(Image.fromarray(rgb))


def _combined_hash(
    img: Image.Image,
    video_type: str = "fullscreen",
) -> tuple[imagehash.ImageHash, imagehash.ImageHash]:
    if video_type == "audience":
        gray = np.array(img.convert("L"))
        edges = cv2.Canny(gray, 40, 120)
        ph = imagehash.phash(Image.fromarray(edges), hash_size=HASH_SIZE)
    else:
        ph = imagehash.phash(img, hash_size=HASH_SIZE)
    dh = imagehash.dhash(img, hash_size=HASH_SIZE)
    return ph, dh


def _hash_distance(
    a: tuple[imagehash.ImageHash, imagehash.ImageHash],
    b: tuple[imagehash.ImageHash, imagehash.ImageHash],
    video_type: str = "fullscreen",
) -> float:
    ph = int(a[0] - b[0])
    dh = int(a[1] - b[1])
    if video_type == "audience":
        return ph * 0.50 + dh * 0.50
    return (ph + dh) / 2.0


def _confidence_from_distances(
    best: float,
    second: float,
    dist_strong: int,
    dist_weak: int,
) -> float:
    """거리가 작을수록, 1·2위 차이가 클수록 신뢰도 상승."""
    if best <= dist_strong:
        sim = 1.0
    elif best >= dist_weak:
        sim = 0.0
    else:
        sim = 1.0 - (best - dist_strong) / max(1.0, dist_weak - dist_strong)

    margin = max(0.0, second - best)
    margin_boost = min(0.25, margin / 12.0)
    return round(min(1.0, sim + margin_boost), 3)


def _match_frame_to_slides(
    frame_hash: tuple[imagehash.ImageHash, imagehash.ImageHash],
    slide_hashes: list[tuple[imagehash.ImageHash, imagehash.ImageHash]],
    dist_strong: int,
    dist_weak: int,
    video_type: str = "fullscreen",
) -> FrameMatch:
    dists = [_hash_distance(frame_hash, sh, video_type) for sh in slide_hashes]
    order = np.argsort(dists)
    best_i = int(order[0])
    best_d = dists[best_i]
    second_d = dists[int(order[1])] if len(order) > 1 else best_d + HASH_BITS
    conf = _confidence_from_distances(best_d, second_d, dist_strong, dist_weak)
    return FrameMatch(
        slide_index=best_i,
        confidence=conf,
        best_dist=round(best_d, 2),
        margin=round(second_d - best_d, 2),
    )


def _viterbi_monotonic(
    dist_matrix: np.ndarray,
    max_skip: int = 2,
) -> list[int]:
    """
    프레임×슬라이드 거리 행렬에서 단조 비감소 경로(슬라이드 번호는 뒤로 안 감)를 찾습니다.
    같은 슬라이드에 머무르거나 앞으로만 진행 가능.
    """
    n_frames, n_slides = dist_matrix.shape
    if n_frames == 0 or n_slides == 0:
        return []

    inf = 1e18
    dp = np.full((n_frames, n_slides), inf)
    parent = np.full((n_frames, n_slides), -1, dtype=np.int32)

    dp[0, :] = dist_matrix[0, :]

    for t in range(1, n_frames):
        for s in range(n_slides):
            best_prev = inf
            best_p = -1
            s_min = max(0, s - max_skip)
            for sp in range(s_min, s + 1):
                cost = dp[t - 1, sp]
                if cost < best_prev:
                    best_prev = cost
                    best_p = sp
            if best_p >= 0:
                dp[t, s] = best_prev + dist_matrix[t, s]
                parent[t, s] = best_p

    path: list[int] = []
    s_last = int(np.argmin(dp[-1, :]))
    path.append(s_last)
    for t in range(n_frames - 1, 0, -1):
        s_last = int(parent[t, s_last])
        path.append(s_last)
    path.reverse()
    return path


def _segments_from_path(
    path: list[int],
    frame_confidences: list[float],
    frame_interval_sec: float,
) -> list[Segment]:
    if not path:
        return []

    segments: list[Segment] = []
    seg_start = 0
    seg_slide = path[0]
    confs = [frame_confidences[0]]

    for i in range(1, len(path)):
        if path[i] != seg_slide:
            segments.append(
                Segment(
                    start_sec=round(seg_start * frame_interval_sec, 2),
                    end_sec=round(i * frame_interval_sec, 2),
                    ppt_slide_index=seg_slide,
                    confidence=round(float(np.mean(confs)), 3),
                    frame_count=len(confs),
                )
            )
            seg_start = i
            seg_slide = path[i]
            confs = [frame_confidences[i]]
        else:
            confs.append(frame_confidences[i])

    segments.append(
        Segment(
            start_sec=round(seg_start * frame_interval_sec, 2),
            end_sec=round((len(path) - 1) * frame_interval_sec, 2),
            ppt_slide_index=seg_slide,
            confidence=round(float(np.mean(confs)), 3),
            frame_count=len(confs),
        )
    )
    return segments


def _collapsed_sequence(path: list[int]) -> list[int]:
    """연속 중복 제거한 슬라이드 순서."""
    if not path:
        return []
    out = [path[0]]
    for s in path[1:]:
        if s != out[-1]:
            out.append(s)
    return out


def _find_missing_slides(slides_seen: set[int], n_slides: int) -> list[int]:
    """영상에 한 번도 등장하지 않은 슬라이드 (1-based 페이지 번호)."""
    return [i + 1 for i in range(n_slides) if i not in slides_seen]


def _find_sequence_gaps(detected_seq: list[int]) -> list[dict[str, Any]]:
    """감지 순서에서 건너뛴 구간 (예: 17장 다음 19장 → 18장 누락)."""
    gaps: list[dict[str, Any]] = []
    for prev, cur in zip(detected_seq, detected_seq[1:]):
        if cur - prev > 1:
            missing = list(range(prev + 2, cur + 1))  # 1-based page numbers
            gaps.append(
                {
                    "from_page": prev + 1,
                    "to_page": cur + 1,
                    "missing_pages": missing,
                }
            )
    return gaps


def verify_ppt_video(
    slide_images: list[Path],
    frame_paths: list[Path],
    frame_interval_sec: float = 2.0,
    video_type: str = "fullscreen",
) -> dict[str, Any]:
    if not slide_images:
        raise ValueError("PPT 슬라이드 이미지가 없습니다.")
    if not frame_paths:
        raise ValueError("영상 프레임이 없습니다.")

    dist_strong, dist_weak = _dist_thresholds(video_type)

    slide_hashes = [
        _combined_hash(_normalize_slide(_load_rgb(p), video_type), video_type)
        for p in slide_images
    ]

    audience_meta: list[AudienceScreenResult | None] = [None] * len(frame_paths)
    frame_hashes = [
        _combined_hash(
            _normalize_frame(
                p,
                video_type,
                audience_meta=audience_meta if video_type == "audience" else None,
                frame_index=i,
            ),
            video_type,
        )
        for i, p in enumerate(frame_paths)
    ]

    pip_face_detected = 0
    if video_type == "pip":
        for fp in frame_paths:
            bgr = _load_frame_bgr(fp)
            if bgr is not None and detect_largest_face(bgr) is not None:
                pip_face_detected += 1

    audience_screen_detected = 0
    audience_perspective_ok = 0
    audience_slide_present = 0
    if video_type == "audience":
        for meta in audience_meta:
            if meta is None:
                continue
            if meta.slide_present:
                audience_slide_present += 1
            if meta.mode.startswith("screen"):
                audience_screen_detected += 1
            if meta.perspective_ok:
                audience_perspective_ok += 1

    n_slides = len(slide_hashes)
    n_frames = len(frame_hashes)

    raw_matches: list[FrameMatch] = [
        _match_frame_to_slides(fh, slide_hashes, dist_strong, dist_weak, video_type)
        for fh in frame_hashes
    ]

    dist_matrix = np.zeros((n_frames, n_slides), dtype=np.float64)
    for t, fh in enumerate(frame_hashes):
        for s, sh in enumerate(slide_hashes):
            dist_matrix[t, s] = _hash_distance(fh, sh, video_type)

    # 프레임당 슬라이드 인덱스는 최대 +1만 허용 (중간 슬라이드는 반드시 경로에 포함)
    path = _viterbi_monotonic(dist_matrix, max_skip=1)
    frame_confidences = [raw_matches[t].confidence for t in range(n_frames)]

    # Viterbi 경로 기준으로 프레임별 신뢰도 재계산 (할당된 슬라이드와의 거리)
    viterbi_confidences: list[float] = []
    for t, s in enumerate(path):
        d = dist_matrix[t, s]
        sorted_d = np.sort(dist_matrix[t, :])
        second = float(sorted_d[1]) if n_slides > 1 else d + HASH_BITS
        conf = _confidence_from_distances(float(d), second, dist_strong, dist_weak)
        if video_type == "audience":
            meta = audience_meta[t] if t < len(audience_meta) else None
            if meta is None:
                conf = min(conf, 0.15)
            elif not meta.slide_present:
                conf = min(conf, 0.25)
            elif not meta.perspective_ok:
                conf = min(conf, 0.45)
            elif meta.confidence >= 0.5:
                conf = min(1.0, conf + 0.08)
        viterbi_confidences.append(conf)

    segments = _segments_from_path(path, viterbi_confidences, frame_interval_sec)
    detected_seq = _collapsed_sequence(path)
    # 누락·커버리지는 "실제로 넘어간 슬라이드 순서" 기준 (1프레임 지나친 값은 제외)
    slides_in_sequence = set(detected_seq)
    missing_pages = _find_missing_slides(slides_in_sequence, n_slides)
    sequence_gaps = _find_sequence_gaps(detected_seq)

    # 순서: 뒤로 가지 않음 + 연속 전환(한 장씩)만 허용
    order_ok = True
    if detected_seq:
        for prev, s in zip(detected_seq, detected_seq[1:]):
            if s < prev or s - prev > 1:
                order_ok = False
                break
        if detected_seq[0] > 0 or detected_seq[-1] < n_slides - 1:
            order_ok = False

    coverage = len(slides_in_sequence) / n_slides
    sequence_complete = len(missing_pages) == 0 and not sequence_gaps

    conf_floor = 0.38 if video_type == "audience" else 0.55
    strong_frames = sum(
        1
        for c, d in zip(
            viterbi_confidences,
            [dist_matrix[t, path[t]] for t in range(n_frames)],
        )
        if c >= conf_floor and d <= dist_weak
    )
    frame_match_rate = strong_frames / max(1, n_frames)
    visual_percent = round(frame_match_rate * 100, 1)
    if video_type == "audience" and viterbi_confidences:
        conf_visual = round(float(np.mean(viterbi_confidences)) * 100, 1)
        visual_percent = round(0.45 * visual_percent + 0.55 * conf_visual, 1)
    coverage_percent = round(coverage * 100, 1)

    # 일치율 = 화면 유사도 × 슬라이드 전체 등장 여부 (한 장이라도 빠지면 100% 불가)
    overall = round(visual_percent * coverage, 1)
    if video_type == "audience" and coverage >= 0.75 and visual_percent >= 35:
        overall = round(min(100.0, overall + visual_percent * 0.08), 1)

    good_segments = sum(1 for seg in segments if seg.confidence >= 0.55)
    segment_match_rate = good_segments / max(1, len(segments))

    issues: list[dict[str, Any]] = []

    for page in missing_pages:
        issues.append(
            {
                "time_sec": 0,
                "message": f"PPT {page}장이 영상에 나타나지 않았습니다 (건너뜀·누락).",
                "ppt_slide_index": page - 1,
                "confidence": 0,
                "issue_type": "missing_slide",
            }
        )

    for gap in sequence_gaps:
        missing_str = ", ".join(f"{p}장" for p in gap["missing_pages"])
        issues.append(
            {
                "time_sec": 0,
                "message": (
                    f"슬라이드 {gap['from_page']}장 다음 {gap['to_page']}장으로 넘어감 "
                    f"({missing_str} 생략)"
                ),
                "ppt_slide_index": gap["missing_pages"][0] - 1 if gap["missing_pages"] else -1,
                "confidence": 0,
                "issue_type": "skipped_sequence",
            }
        )

    seg_conf_floor = 0.45 if video_type == "audience" else 0.55
    for seg in segments:
        if seg.confidence < seg_conf_floor:
            issues.append(
                {
                    "time_sec": seg.start_sec,
                    "message": (
                        f"{seg.start_sec:.0f}s~{seg.end_sec:.0f}s: "
                        f"PPT {seg.ppt_slide_index + 1}장 유사도 낮음 "
                        f"(confidence={seg.confidence})"
                    ),
                    "ppt_slide_index": seg.ppt_slide_index,
                    "confidence": seg.confidence,
                    "issue_type": "low_confidence",
                }
            )

    if not order_ok and not sequence_gaps:
        issues.append(
            {
                "time_sec": 0,
                "message": "슬라이드 진행 순서가 PPT 1장부터 끝까지 순차적이지 않습니다.",
                "ppt_slide_index": -1,
                "confidence": 0,
                "issue_type": "order",
            }
        )

    if video_type == "audience" and n_frames > 0:
        no_screen = n_frames - audience_slide_present
        if no_screen > 0:
            rate = no_screen / n_frames
            issues.append(
                {
                    "time_sec": 0,
                    "message": (
                        f"스크린 미검출 프레임 {no_screen}/{n_frames}개 "
                        f"({rate * 100:.0f}%) — 발표자·청중만 보이는 구간은 판단 불가."
                    ),
                    "ppt_slide_index": -1,
                    "confidence": 0,
                    "issue_type": "audience_no_screen",
                }
            )
        if audience_slide_present > 0 and audience_perspective_ok / audience_slide_present < 0.5:
            issues.append(
                {
                    "time_sec": 0,
                    "message": (
                        "스크린 원근 보정 신뢰도가 낮습니다. "
                        "각도·조명·가림이 크면 일치율이 떨어질 수 있습니다."
                    ),
                    "ppt_slide_index": -1,
                    "confidence": 0,
                    "issue_type": "audience_low_perspective",
                }
            )

    if sequence_complete and visual_percent >= 75 and order_ok:
        verdict = "match"
    elif visual_percent >= 40 or coverage >= 0.5:
        verdict = "partial"
    else:
        verdict = "mismatch"

    if missing_pages and verdict == "match":
        verdict = "partial"

    return {
        "verdict": verdict,
        "overall_match_percent": overall,
        "visual_match_percent": visual_percent,
        "frame_match_rate_percent": visual_percent,
        "segment_match_rate_percent": round(segment_match_rate * 100, 1),
        "slide_coverage_percent": coverage_percent,
        "sequence_complete": sequence_complete,
        "missing_slides": missing_pages,
        "sequence_gaps": sequence_gaps,
        "slide_count_ppt": n_slides,
        "segment_count": len(segments),
        "unique_transitions": len(detected_seq),
        "order_matches_ppt": order_ok and sequence_complete,
        "detected_slide_sequence": [i + 1 for i in detected_seq],
        "segments": [
            {
                "start_sec": s.start_sec,
                "end_sec": s.end_sec,
                "ppt_slide_index": s.ppt_slide_index,
                "ppt_slide_number": s.ppt_slide_index + 1,
                "confidence": s.confidence,
                "frame_count": s.frame_count,
            }
            for s in segments
        ],
        "issues": issues[:40],
        "diagnostics": {
            "hash_size": HASH_SIZE,
            "compare_resolution": f"{COMPARE_W}x{COMPARE_H}",
            "dist_strong": dist_strong,
            "dist_weak": dist_weak,
            "median_frame_confidence": round(float(np.median(viterbi_confidences)), 3),
            "mean_best_distance": round(float(np.mean([dist_matrix[t, path[t]] for t in range(n_frames)])), 2),
            "video_type": video_type,
            "pip_face_detection_rate": (
                round(pip_face_detected / max(1, n_frames), 3) if video_type == "pip" else None
            ),
            "screen_detection_rate": (
                round(audience_screen_detected / max(1, n_frames), 3)
                if video_type == "audience"
                else None
            ),
            "perspective_ok_rate": (
                round(audience_perspective_ok / max(1, n_frames), 3)
                if video_type == "audience"
                else None
            ),
            "slide_present_rate": (
                round(audience_slide_present / max(1, n_frames), 3)
                if video_type == "audience"
                else None
            ),
        },
        "note": (
            "일치율 = 화면 유사도 × 슬라이드 커버리지. "
            "PPT 중 한 장이라도 영상에 없으면 100%가 될 수 없습니다."
            + (" PiP: 얼굴 영역 제외 후 슬라이드만 비교." if video_type == "pip" else "")
            + (
                " Audience: 프로젝터/모니터 영역 검출·원근 보정 후 비교."
                if video_type == "audience"
                else ""
            )
        ),
    }
