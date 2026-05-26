"""PiP(얼굴 + 슬라이드) 영상에서 슬라이드 비교용 영역 추출."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

_FACE_CASCADE = None


def _face_cascade() -> cv2.CascadeClassifier | None:
    global _FACE_CASCADE
    if _FACE_CASCADE is None:
        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        cascade = cv2.CascadeClassifier(path)
        _FACE_CASCADE = cascade if not cascade.empty() else None
    return _FACE_CASCADE


def _detect_faces_in_gray(gray: np.ndarray, min_size: int) -> list[tuple[int, int, int, int]]:
    cascade = _face_cascade()
    if cascade is None:
        return []
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=3,
        minSize=(min_size, min_size),
    )
    return [(int(x), int(y), int(x + fw), int(y + fh)) for x, y, fw, fh in faces]


def detect_largest_face(bgr: np.ndarray) -> tuple[int, int, int, int] | None:
    """가장 큰 얼굴 bbox. 전체 + 우하단 영역 이중 검색."""
    h, w = bgr.shape[:2]
    gray = cv2.equalizeHist(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
    min_size = max(20, int(min(w, h) * 0.04))

    candidates: list[tuple[int, int, int, int]] = []
    candidates.extend(_detect_faces_in_gray(gray, min_size))

    # PiP는 보통 우하단 → 해당 영역만 추가 검색 (슬라이드 텍스트 오탐 감소)
    y0, x0 = int(h * 0.45), int(w * 0.45)
    roi = gray[y0:h, x0:w]
    if roi.size > 0:
        for x1, y1, x2, y2 in _detect_faces_in_gray(roi, max(16, min_size // 2)):
            candidates.append((x1 + x0, y1 + y0, x2 + x0, y2 + y0))

    if not candidates:
        return None
    return max(candidates, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))


def _slide_fill_color(bgr: np.ndarray) -> tuple[int, int, int]:
    """슬라이드 배경에 가까운 색 (상단·좌측 밝은 영역 중앙값)."""
    h, w = bgr.shape[:2]
    patch = bgr[0 : max(8, h // 8), 0 : max(8, w // 4)]
    if patch.size == 0:
        return 245, 245, 245
    med = np.median(patch.reshape(-1, 3), axis=0)
    return int(med[0]), int(med[1]), int(med[2])


def _crop_content_area(bgr: np.ndarray) -> np.ndarray:
    """녹화 테두리(회색) 제거 — 밝은 슬라이드 영역만 남김."""
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    mask = (gray > 200).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    coords = cv2.findNonZero(mask)
    if coords is None:
        return bgr
    x, y, bw, bh = cv2.boundingRect(coords)
    if bw < w * 0.3 or bh < h * 0.3:
        return bgr
    pad = 2
    return bgr[max(0, y - pad) : y + bh + pad, max(0, x - pad) : x + bw + pad]


def _mask_presenter_ui(bgr: np.ndarray, fill: tuple[int, int, int]) -> None:
    """PowerPoint 발표 화면 하단 아이콘·툴바 영역 마스킹."""
    h, w = bgr.shape[:2]
    bar_h = int(h * 0.11)
    cv2.rectangle(bgr, (0, h - bar_h), (w, h), fill, -1)


def _mask_pip_corner(bgr: np.ndarray, fill: tuple[int, int, int], corner: str = "br") -> None:
    """일반적인 PiP 위치(우하단 등) 고정 마스킹."""
    h, w = bgr.shape[:2]
    rw, rh = int(w * 0.28), int(h * 0.30)
    if corner == "br":
        cv2.rectangle(bgr, (w - rw, h - rh), (w, h), fill, -1)
    elif corner == "bl":
        cv2.rectangle(bgr, (0, h - rh), (rw, h), fill, -1)
    elif corner == "tr":
        cv2.rectangle(bgr, (w - rw, 0), (w, rh), fill, -1)
    else:
        cv2.rectangle(bgr, (0, 0), (rw, rh), fill, -1)


def _expand_bbox(x1: int, y1: int, x2: int, y2: int, w: int, h: int, pad: float = 0.45) -> tuple[int, int, int, int]:
    bw, bh = x2 - x1, y2 - y1
    px, py = int(bw * pad), int(bh * pad)
    return (
        max(0, x1 - px),
        max(0, y1 - py),
        min(w, x2 + px),
        min(h, y2 + py),
    )


def prepare_pip_frame(bgr: np.ndarray, face: tuple[int, int, int, int] | None) -> tuple[np.ndarray, str]:
    h, w = bgr.shape[:2]
    out = _crop_content_area(bgr)
    fill = _slide_fill_color(out)
    _mask_presenter_ui(out, fill)

    if face is not None:
        x1, y1, x2, y2 = _expand_bbox(*face, out.shape[1], out.shape[0], pad=0.5)
        cv2.rectangle(out, (x1, y1), (x2, y2), fill, -1)
        mode = "pip_mask_detected_face"
    else:
        # 얼굴 미검출: Zoom/Teams 기본 우하단 PiP + 하단 바 제거
        _mask_pip_corner(out, fill, "br")
        mode = "pip_mask_default_br_pip"

    # 슬라이드 본문 위주 — 상단·좌측 여백 소폭만 제거
    oh, ow = out.shape[:2]
    mx, my = int(ow * 0.02), int(oh * 0.02)
    bottom_trim = int(oh * 0.02)
    crop = out[my : oh - bottom_trim, mx : ow - mx]
    if crop.size == 0:
        return out, mode + "_fallback"
    return crop, mode


def load_pip_frame_bgr(path: Path) -> tuple[np.ndarray | None, str]:
    bgr = cv2.imread(str(path))
    if bgr is None:
        return None, "read_error"
    face = detect_largest_face(bgr)
    processed, mode = prepare_pip_frame(bgr, face)
    if face is not None:
        mode += "+face"
    return processed, mode
