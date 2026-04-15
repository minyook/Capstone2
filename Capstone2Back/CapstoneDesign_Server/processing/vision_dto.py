from __future__ import annotations

from dataclasses import asdict, dataclass, field # field 추가
from typing import Any, Optional


@dataclass(slots=True)
class YoloPoseResult:
    """
    YOLO (pose) 전용 결과.
    - 사람 존재/가시성 신호만 포함
    """

    has_person: bool
    has_pelvis: bool
    has_ankles: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class MediaPipeFaceResult:
    """
    MediaPipe 전용 결과.
    - 얼굴 존재 및 표정/시선 스코어만 포함
    - 실패/미검출 사유는 error에만 담음
    """

    has_face: bool
    smile: float = 0.0
    frown: float = 0.0
    brow_up: float = 0.0
    brow_down: float = 0.0
    jaw_open: float = 0.0
    mouth_open: float = 0.0
    squint: float = 0.0
    gaze_h: float = 0.0
    gaze_v: float = 0.0
    error: Optional[str] = None
    # 🌟 추가: 52개 상세 좌표 수치를 저장할 필드 (기본값은 빈 딕셔너리)
    all_blendshapes: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class FrameVisionResult:
    """
    프레임 단위의 비전 결과 컨테이너.
    - time은 task_manager에서 프레임 인덱스로 주입
    - yolo/face는 각자 독립된 구조 유지
    """

    time: float
    yolo: YoloPoseResult
    face: MediaPipeFaceResult

    def to_dict(self) -> dict[str, Any]:
        return {
            "time": self.time,
            "yolo": self.yolo.to_dict(),
            "face": self.face.to_dict(),
        }