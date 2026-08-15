"""Ground Truth Labeling 결과 스키마."""

from dataclasses import dataclass
from typing import Any, List


@dataclass
class LabelResult:
    """
    Gemini Judge의 단일 논문 평가 결과.
    """

    score: int
    """
    0~100.
    """

    decision: str
    """
    정답 / 경계 / 오답
    """

    label: int
    """
    1 = 정답
    0 = 오답
    """

    tag: str
    """
    none
    excl_violation
    kw_only
    uncertain
    """

    reason: str

    analysis: str

    raw_response: str = ""

    parse_warnings: List[str] = None

    def __post_init__(self):
        if self.parse_warnings is None:
            self.parse_warnings = []