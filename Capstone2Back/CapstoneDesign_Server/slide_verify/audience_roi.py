"""강의실·청중 시점 영상에서 프로젝터/모니터(스크린) ROI 추출 및 원근 보정."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

WARP_W, WARP_H = 640, 360
MIN_SCREEN_AREA_RATIO = 0.06
MAX_SCREEN_AREA_RATIO = 0.88
MIN_RECTANGULARITY = 0.55
MIN_PERSPECTIVE_CONF = 0.28


@dataclass
class AudienceScreenResult:
    bgr: np.ndarray | None
    slide_present: bool
    perspective_ok: bool
    confidence: float
    mode: str


def _order_quad(pts: np.ndarray) -> np.ndarray:
    pts = pts.reshape(4, 2).astype(np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def _quad_rectangularity(ordered: np.ndarray) -> float:
    tl, tr, br, bl = ordered
    w_top = np.linalg.norm(tr - tl)
    w_bot = np.linalg.norm(br - bl)
    h_left = np.linalg.norm(bl - tl)
    h_right = np.linalg.norm(br - tr)
    if min(w_top, w_bot, h_left, h_right) < 1:
        return 0.0
    w_ratio = min(w_top, w_bot) / max(w_top, w_bot)
    h_ratio = min(h_left, h_right) / max(h_left, h_right)
    return float(w_ratio * h_ratio)


def _aspect_score(ordered: np.ndarray) -> float:
    w = max(np.linalg.norm(ordered[1] - ordered[0]), np.linalg.norm(ordered[2] - ordered[3]))
    h = max(
        1.0,
        min(np.linalg.norm(ordered[3] - ordered[0]), np.linalg.norm(ordered[2] - ordered[1])),
    )
    aspect = w / h
    for ideal in (16 / 9, 4 / 3, 3 / 2):
        err = abs(aspect - ideal) / ideal
        if err < 0.35:
            return 1.0 - err
    return max(0.0, 1.0 - abs(aspect - 16 / 9) / (16 / 9))


def _score_quad(ordered: np.ndarray, frame_area: int, center_bonus: float = 0.0) -> float:
    area = cv2.contourArea(ordered.astype(np.float32))
    if area < frame_area * MIN_SCREEN_AREA_RATIO:
        return 0.0
    if area > frame_area * MAX_SCREEN_AREA_RATIO:
        return 0.0
    rect = _quad_rectangularity(ordered)
    if rect < MIN_RECTANGULARITY:
        return 0.0
    area_score = min(1.0, area / (frame_area * 0.22))
    return float(rect * 0.40 + _aspect_score(ordered) * 0.40 + area_score * 0.20 + center_bonus)


def _quad_from_contour(cnt: np.ndarray, frame_area: int) -> tuple[np.ndarray | None, float]:
    peri = cv2.arcLength(cnt, True)
    for eps_ratio in (0.015, 0.025, 0.04, 0.06, 0.09):
        approx = cv2.approxPolyDP(cnt, eps_ratio * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            ordered = _order_quad(approx)
            score = _score_quad(ordered, frame_area)
            if score > 0:
                return ordered, score
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    ordered = _order_quad(box)
    score = _score_quad(ordered, frame_area) * 0.9
    if score > 0:
        return ordered, score
    return None, 0.0


def _center_bonus(ordered: np.ndarray, w: int, h: int) -> float:
    cx, cy = w / 2, h / 2
    quad_cx = float(np.mean(ordered[:, 0]))
    quad_cy = float(np.mean(ordered[:, 1]))
    dist = np.hypot(quad_cx - cx, quad_cy - cy) / np.hypot(cx, cy)
    return max(0.0, 0.12 * (1.0 - dist * 1.8))


def _find_quad_in_mask(mask: np.ndarray, offset_x: int = 0, offset_y: int = 0) -> tuple[np.ndarray | None, float]:
    h, w = mask.shape[:2]
    frame_area = h * w
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_pts: np.ndarray | None = None
    best_score = 0.0

    for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:10]:
        if cv2.contourArea(cnt) < frame_area * MIN_SCREEN_AREA_RATIO:
            break
        ordered, score = _quad_from_contour(cnt, frame_area)
        if ordered is None:
            continue
        ordered[:, 0] += offset_x
        ordered[:, 1] += offset_y
        score += _center_bonus(ordered, w + offset_x * 2, h + offset_y * 2)
        if score > best_score:
            best_score = score
            best_pts = ordered.copy()

    return best_pts, best_score


def _find_screen_by_bezel(bgr: np.ndarray) -> tuple[np.ndarray | None, float]:
    """프로젝터 스크린 좌·우 검은 베젤을 이용해 사각형 추정."""
    h, w = bgr.shape[:2]
    gray = cv2.GaussianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (5, 5), 0)

    y1, y2 = int(h * 0.12), int(h * 0.88)
    x1, x2 = int(w * 0.08), int(w * 0.92)
    roi = gray[y1:y2, x1:x2]
    rh, rw = roi.shape[:2]

    # 여러 높이에서 밝은 영역의 좌·우 경계(어두운 베젤 직후) 탐색
    xs_left: list[float] = []
    xs_right: list[float] = []
    ys: list[float] = []
    for frac in (0.25, 0.40, 0.50, 0.60, 0.75):
        row = roi[int(rh * frac), :]
        bright = row > 175
        if bright.sum() < rw * 0.15:
            continue
        idx = np.where(bright)[0]
        left, right = int(idx[0]), int(idx[-1])
        # 베젤: 밝은 영역 바깥의 어두운 띠
        l_edge = left
        for i in range(left, max(0, left - 40), -1):
            if row[i] < 95:
                l_edge = i
                break
        r_edge = right
        for i in range(right, min(rw - 1, right + 40)):
            if row[i] < 95:
                r_edge = i
                break
        if right - left < rw * 0.2:
            continue
        xs_left.append(x1 + l_edge)
        xs_right.append(x1 + r_edge)
        ys.append(y1 + rh * frac)

    if len(xs_left) < 2:
        return None, 0.0

    xl = float(np.median(xs_left))
    xr = float(np.median(xs_right))
    yt = y1 + rh * 0.05
    yb = y1 + rh * 0.95

    # 상·하단: 밝은 영역의 수직 범위
    col_x = int((xl + xr) / 2) - x1
    col_x = max(0, min(rw - 1, col_x))
    col = roi[:, col_x]
    bright_col = col > 170
    if bright_col.sum() > rh * 0.2:
        v_idx = np.where(bright_col)[0]
        yt = y1 + max(0, int(v_idx[0]) - 5)
        yb = y1 + min(rh - 1, int(v_idx[-1]) + 5)

    quad = np.array(
        [[xl, yt], [xr, yt], [xr, yb], [xl, yb]],
        dtype=np.float32,
    )
    score = _score_quad(quad, h * w, center_bonus=_center_bonus(quad, w, h))
    return quad, score


def _find_screen_by_center_bright(bgr: np.ndarray) -> tuple[np.ndarray | None, float]:
    """중앙 영역만 사용해 밝은 스크린 사각형 검출 (화이트보드 합침 완화)."""
    h, w = bgr.shape[:2]
    mx1, mx2 = int(w * 0.06), int(w * 0.94)
    my1, my2 = int(h * 0.06), int(h * 0.94)
    crop = bgr[my1:my2, mx1:mx2]
    ch, cw = crop.shape[:2]

    gray = cv2.GaussianBlur(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    # Otsu: 스크린(밝음) vs 주변
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(otsu) > 127:
        otsu = 255 - otsu

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    otsu = cv2.morphologyEx(otsu, cv2.MORPH_OPEN, kernel, iterations=1)
    otsu = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel, iterations=2)

    pts, score = _find_quad_in_mask(otsu, offset_x=mx1, offset_y=my1)
    if pts is not None and score > 0:
        return pts, score

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    for v_th in (145, 155, 165):
        _, bright = cv2.threshold(hsv[:, :, 2], v_th, 255, cv2.THRESH_BINARY)
        bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, kernel, iterations=2)
        pts, score = _find_quad_in_mask(bright, offset_x=mx1, offset_y=my1)
        if score > 0:
            return pts, score

    return None, 0.0


def _find_screen_by_edges(bgr: np.ndarray) -> tuple[np.ndarray | None, float]:
    h, w = bgr.shape[:2]
    scale = 1.0
    work = bgr
    if max(w, h) > 960:
        scale = 960 / max(w, h)
        work = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    wh, ww = work.shape[:2]
    gray = cv2.GaussianBlur(cv2.cvtColor(work, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.dilate(edges, kernel, iterations=1)

    pts, score = _find_quad_in_mask(edges)
    if pts is not None:
        pts = pts / scale
        score += _center_bonus(pts, w, h)
    return pts, score


def _find_screen_quad(bgr: np.ndarray) -> tuple[np.ndarray | None, float]:
    candidates: list[tuple[np.ndarray, float, str]] = []

    for finder in (_find_screen_by_bezel, _find_screen_by_center_bright, _find_screen_by_edges):
        pts, score = finder(bgr)
        if pts is not None and score > 0:
            candidates.append((pts, score, finder.__name__))

    if not candidates:
        return None, 0.0

    best = max(candidates, key=lambda c: c[1])
    return best[0], best[1]


def _enhance_for_match(bgr: np.ndarray) -> np.ndarray:
    """프로젝터 과노출·저대비 보정 (색상 유지 — pHash 변별력)."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l = clahe.apply(l)
    out = cv2.cvtColor(cv2.merge([l, a, b_ch]), cv2.COLOR_LAB2BGR)
    # 옅은 회색 글자만 살짝 강조 (그레이스케일 단일화는 하지 않음)
    gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    if float(np.mean(gray)) > 185:
        _, text_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        text_mask = cv2.erode(text_mask, np.ones((2, 2), np.uint8), iterations=1)
        dark = text_mask > 0
        if dark.sum() > 50:
            out[dark] = np.clip(out[dark].astype(np.int16) * 0.55, 0, 255).astype(np.uint8)
    return out


def crop_slide_content_bgr(bgr: np.ndarray) -> np.ndarray:
    return _crop_to_slide_content(bgr)


def _crop_to_slide_content(bgr: np.ndarray) -> np.ndarray:
    """흰 여백 제거 후 슬라이드 본문만 남김."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    mask = gray < 245
    if mask.sum() < 100:
        return bgr
    coords = cv2.findNonZero(mask.astype(np.uint8))
    if coords is None:
        return bgr
    x, y, bw, bh = cv2.boundingRect(coords)
    h, w = bgr.shape[:2]
    pad_x, pad_y = int(bw * 0.04), int(bh * 0.04)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(w, x + bw + pad_x)
    y2 = min(h, y + bh + pad_y)
    crop = bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return bgr
    return crop


def _shrink_quad_inward(quad: np.ndarray, ratio: float = 0.02) -> np.ndarray:
    """검은 베젤·프레임 제외를 위해 안쪽으로 살짝 수축."""
    center = np.mean(quad, axis=0)
    return center + (quad - center) * (1.0 - ratio)


def _warp_screen(bgr: np.ndarray, quad: np.ndarray) -> np.ndarray:
    quad = _shrink_quad_inward(quad, 0.025)
    dst = np.array(
        [[0, 0], [WARP_W - 1, 0], [WARP_W - 1, WARP_H - 1], [0, WARP_H - 1]],
        dtype=np.float32,
    )
    m = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
    warped = cv2.warpPerspective(bgr, m, (WARP_W, WARP_H), flags=cv2.INTER_LINEAR)
    return _enhance_for_match(warped)


def _trim_letterbox(bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(th)
    if coords is None:
        return bgr
    x, y, bw, bh = cv2.boundingRect(coords)
    if bw < WARP_W * 0.35 or bh < WARP_H * 0.35:
        return bgr
    pad = 3
    return bgr[max(0, y - pad) : y + bh + pad, max(0, x - pad) : x + bw + pad]


def _mask_presenter_occlusion(bgr: np.ndarray) -> np.ndarray:
    """스크린 왼쪽 발표자 실루엣(어두운 덩어리)을 배경색으로 채움."""
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    left = gray[:, : int(w * 0.38)]
    dark = left < 85
    if dark.sum() < left.size * 0.04:
        return bgr
    # 배경 밝기 추정
    bright = gray[gray > 200]
    fill = int(np.median(bright)) if bright.size > 100 else 248
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[:, : int(w * 0.38)] = (gray[:, : int(w * 0.38)] < 90).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    out = bgr.copy()
    out[mask > 0] = (fill, fill, fill)
    return out


def _fallback_center_crop(bgr: np.ndarray) -> np.ndarray:
    h, w = bgr.shape[:2]
    mx, my = int(w * 0.10), int(h * 0.08)
    crop = bgr[my : h - my, mx : w - mx]
    if crop.size == 0:
        return bgr
    out = cv2.resize(crop, (WARP_W, WARP_H), interpolation=cv2.INTER_AREA)
    return _enhance_for_match(out)


def extract_audience_screen(bgr: np.ndarray) -> AudienceScreenResult:
    quad, score = _find_screen_quad(bgr)
    if quad is None or score < MIN_PERSPECTIVE_CONF:
        fallback = _fallback_center_crop(bgr)
        return AudienceScreenResult(
            bgr=fallback,
            slide_present=score >= 0.12,
            perspective_ok=False,
            confidence=round(score, 3),
            mode="no_screen_fallback",
        )

    try:
        warped = _warp_screen(bgr, quad)
        warped = _trim_letterbox(warped)
        warped = _crop_to_slide_content(warped)
        warped = _mask_presenter_occlusion(warped)
    except cv2.error:
        return AudienceScreenResult(
            bgr=_fallback_center_crop(bgr),
            slide_present=False,
            perspective_ok=False,
            confidence=round(score, 3),
            mode="warp_failed",
        )

    perspective_ok = score >= MIN_PERSPECTIVE_CONF + 0.06
    return AudienceScreenResult(
        bgr=warped,
        slide_present=True,
        perspective_ok=perspective_ok,
        confidence=round(min(1.0, score), 3),
        mode="screen_warp" if perspective_ok else "screen_low_conf",
    )


def load_audience_frame_bgr(path: Path) -> tuple[np.ndarray | None, AudienceScreenResult | None]:
    bgr = cv2.imread(str(path))
    if bgr is None:
        return None, None
    result = extract_audience_screen(bgr)
    return result.bgr, result
