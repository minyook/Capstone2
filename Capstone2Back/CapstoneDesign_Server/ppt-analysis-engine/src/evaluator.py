"""순수 Python 기반 PPT 특징 추출기."""

from __future__ import annotations

import math
import re
from collections import Counter
from datetime import datetime
from statistics import mean
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer

_TOKEN_RE = re.compile(r"[0-9A-Za-z가-힣]{2,}")
_SENT_SPLIT_RE = re.compile(r"[.!?]\s+|\n+")


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def _join(*parts: str) -> str:
    return "\n".join([p.strip() for p in parts if (p or "").strip()])


def _distance(ax: float, ay: float, bx: float, by: float) -> float:
    return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)


def _center(item: dict[str, Any]) -> tuple[float, float]:
    left = float(item.get("left") or 0.0)
    top = float(item.get("top") or 0.0)
    width = float(item.get("width") or 0.0)
    height = float(item.get("height") or 0.0)
    return (left + width / 2.0, top + height / 2.0)


def _keywords(text: str, top_k: int = 3) -> list[str]:
    tokens = [t.lower() for t in _TOKEN_RE.findall(text)]
    if not tokens:
        return []
    c = Counter(tokens)
    return [w for w, _ in c.most_common(top_k)]


def _top_sentences(text: str, max_points: int = 3) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    sentences = [s.strip() for s in _SENT_SPLIT_RE.split(text) if s.strip()]
    if not sentences:
        return [text[:120]]
    if len(sentences) <= max_points:
        return [s[:120] for s in sentences]
    try:
        vec = TfidfVectorizer(lowercase=True, token_pattern=r"(?u)[0-9A-Za-z가-힣]{2,}", max_features=4000)
        x = vec.fit_transform(sentences)
        scores = x.sum(axis=1).A1
        ranked = sorted(range(len(sentences)), key=lambda i: scores[i], reverse=True)[:max_points]
        ranked.sort()
        return [sentences[i][:120] for i in ranked]
    except Exception:
        ranked = sorted(sentences, key=len, reverse=True)[:max_points]
        return [s[:120] for s in ranked]


def _image_context_for_slide(slide: dict[str, Any]) -> list[dict[str, Any]]:
    images = slide.get("image_items") or []
    text_boxes = slide.get("text_boxes") or []
    title = (slide.get("title") or "").strip()
    contexts: list[dict[str, Any]] = []

    title_box = None
    if title:
        title_box = {"text": title, "left": 0, "top": 0, "width": 0, "height": 0}

    candidates = list(text_boxes)
    if title_box:
        candidates.append(title_box)

    for idx, img in enumerate(images):
        ix, iy = _center(img)
        best_text = ""
        best_dist = None
        for t in candidates:
            txt = (t.get("text") or "").strip()
            if not txt:
                continue
            tx, ty = _center(t)
            d = _distance(ix, iy, tx, ty)
            if best_dist is None or d < best_dist:
                best_dist = d
                best_text = txt[:120]
        contexts.append(
            {
                "name": img.get("name") or f"image_{idx + 1}",
                "context": best_text,
                "distance_emu": int(best_dist) if best_dist is not None else None,
            }
        )
    return contexts


def _slide_visual_balance(slide: dict[str, Any], slide_area: float) -> float:
    if slide_area <= 0:
        return 0.0
    img_area = 0.0
    for img in slide.get("image_items") or []:
        img_area += float((img.get("width") or 0) * (img.get("height") or 0))
    text_area = float(slide.get("text_shape_area_emu2") or 0)
    used_ratio = _clamp01((img_area + text_area) / slide_area)
    img_ratio = _clamp01(img_area / slide_area)
    text_ratio = _clamp01(text_area / slide_area)
    # 텍스트/이미지 비율이 극단적으로 치우치지 않을수록 점수↑
    balance = 1.0 - min(abs(text_ratio - img_ratio) / 0.5, 1.0)
    return round(_clamp01(0.5 * balance + 0.5 * used_ratio), 3)


def _slide_readability(slide: dict[str, Any]) -> float:
    sizes = [float(f["size_pt"]) for f in (slide.get("fonts") or []) if f.get("size_pt") is not None]
    if not sizes:
        return 0.0
    good = sum(1 for s in sizes if s >= 18.0)
    return round(good / len(sizes), 3)


def extract_ppt_features(parsed: dict[str, Any]) -> dict[str, Any]:
    slides = parsed.get("slides") or []
    slide_size = parsed.get("slide_size_emu") or {}
    slide_area = float((slide_size.get("width") or 0) * (slide_size.get("height") or 0))

    slide_outputs: list[dict[str, Any]] = []
    visual_scores: list[float] = []
    readability_scores: list[float] = []
    text_overload_slides: list[int] = []

    for s in slides:
        idx = int(s.get("slide_index") or 0)
        title = (s.get("title") or "").strip()
        body = (s.get("body_text") or "").strip()
        notes = (s.get("notes_text") or "").strip()

        summary_points = _top_sentences(_join(title, body, notes), max_points=3)
        image_context = _image_context_for_slide(s)
        visual_balance = _slide_visual_balance(s, slide_area)
        readability = _slide_readability(s)

        visual_scores.append(visual_balance)
        readability_scores.append(readability)

        char_len = len(_join(title, body, notes))
        flags: list[str] = []
        if readability >= 0.7:
            flags.append("font_size_ok")
        else:
            flags.append("font_size_small")
        if s.get("image_count", 0) > 0 and any((img.get("context") or "").strip() for img in image_context):
            flags.append("image_text_matched")
        elif s.get("image_count", 0) > 0:
            flags.append("image_text_unmatched")
        if char_len > 320:
            flags.append("text_overload")
            text_overload_slides.append(idx)

        slide_outputs.append(
            {
                "index": idx,
                "title": title,
                "summary_points": summary_points,
                "images": image_context,
                "visual_balance_score": visual_balance,
                "readability_score": readability,
                "flags": flags,
            }
        )

    metadata = {
        "file_name": parsed.get("file_name"),
        "slide_count": len(slides),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }

    normalized_metrics = {
        "readability": round(mean(readability_scores), 3) if readability_scores else 0.0,
        "visual_balance": round(mean(visual_scores), 3) if visual_scores else 0.0,
        "consistency": round(_clamp01(1.0 - (len(text_overload_slides) / max(1, len(slides)))), 3),
    }

    quantitative_stats = {
        "slide_count": len(slides),
        "image_count_total": int(sum(int(s.get("image_count") or 0) for s in slides)),
        "text_box_count_total": int(sum(int(s.get("textbox_count") or 0) for s in slides)),
        "text_overload": text_overload_slides,
        "readability_flags": {
            "text_overload": text_overload_slides,
        },
    }

    return {
        "metadata": metadata,
        "normalized_metrics": normalized_metrics,
        "quantitative_stats": quantitative_stats,
        "slides": slide_outputs,
    }
