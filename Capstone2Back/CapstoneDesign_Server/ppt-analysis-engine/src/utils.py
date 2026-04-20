"""JSON 입출력 및 경로 관련 유틸리티."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def project_root() -> Path:
    """저장소 루트(main.py가 있는 디렉터리)를 반환합니다."""
    return Path(__file__).resolve().parent.parent


def uploads_dir() -> Path:
    return project_root() / "data" / "uploads"


def results_dir() -> Path:
    return project_root() / "data" / "results"


def ppt_features_json_path(ppt_path: str | Path) -> Path:
    """입력 PPT stem과 동일한 결과 JSON 경로를 반환합니다."""
    ppt = Path(ppt_path)
    return results_dir() / f"{ppt.stem}.json"


def ensure_directory(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(path: str | Path, data: Any, *, indent: int | None = 2) -> Path:
    out = Path(path)
    ensure_directory(out.parent)
    if indent is None:
        body = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    else:
        body = json.dumps(data, ensure_ascii=False, indent=indent)
    out.write_text(body, encoding="utf-8")
    return out


def load_json(path: str | Path) -> Any:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"JSON 파일을 찾을 수 없습니다: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def safe_result_path_for_ppt(ppt_path: str | Path, results_base: str | Path | None = None) -> Path:
    """PPT 파일명과 동일한 stem의 결과 JSON 경로를 반환합니다."""
    ppt = Path(ppt_path)
    base = Path(results_base) if results_base else results_dir()
    return base / f"{ppt.stem}.json"
