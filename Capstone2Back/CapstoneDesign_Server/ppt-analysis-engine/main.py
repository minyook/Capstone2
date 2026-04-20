"""
PPT 분석 파이프라인 엔트리포인트.

`data/uploads`의 파일을 읽어 원천 데이터 + 정량 통계를 추출하고,
입력 파일명과 같은 `data/results/{stem}.json`으로 저장합니다.

웹 프레임워크에서도 `analyze_ppt_file`을 import 해 재사용할 수 있습니다.
"""

from __future__ import annotations

import argparse
import traceback
from pathlib import Path
from typing import Any

from src.evaluator import extract_ppt_features
from src.parser import parse_ppt_file
from src.utils import ppt_features_json_path, results_dir, save_json, uploads_dir


def analyze_ppt_file(
    ppt_path: str | Path,
    *,
    result_path: str | Path | None = None,
) -> dict[str, Any]:
    """
    PPT 경로를 입력받아 features JSON dict를 반환하고 저장합니다.

    Args:
        ppt_path: 분석할 .pptx 파일 경로
        result_path: 저장 경로(미지정 시 data/results/{stem}.json)
    """
    ppt_path = Path(ppt_path)
    parsed = parse_ppt_file(ppt_path)
    out = extract_ppt_features(parsed)

    dest = Path(result_path) if result_path else ppt_features_json_path(ppt_path)
    save_json(dest, out, indent=2)
    out["result_path"] = str(dest.resolve())
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="PPT 특징 추출 후 data/results/{stem}.json 저장")
    parser.add_argument(
        "filename",
        nargs="?",
        default="example.pptx",
        help="data/uploads 내 파일명 (기본: example.pptx)",
    )
    args = parser.parse_args()

    uploads = uploads_dir()
    ppt_full = uploads / args.filename

    print(f"업로드 경로: {ppt_full}")
    print(f"결과 디렉터리: {results_dir()}")

    try:
        result = analyze_ppt_file(ppt_full)
        print("분석 완료.")
        print(f"결과 파일: {result.get('result_path')}")
    except FileNotFoundError as exc:
        print(f"[오류] {exc}")
    except ValueError as exc:
        print(f"[설정 오류] {exc}")
    except Exception:
        print("[예기치 않은 오류]")
        traceback.print_exc()


if __name__ == "__main__":
    main()
