"""Ground Truth Labeler API 요청/응답 스키마 (내부 LabelResult 데이터클래스와는 별개)."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class LabelRequest(BaseModel):
    profile: str
    title: str
    abstract: str = ""


class BatchLabelRequest(BaseModel):
    profile: str
    papers: List[Dict[str, Any]] = Field(
        ..., description='예: [{"arxiv_id": "...", "title": "...", "abstract_clean": "..."}]'
    )
    title_key: str = "title"
    abstract_key: str = "abstract_clean"
    id_key: str = "arxiv_id"
    keep_raw: bool = False
    """True면 각 결과에 analysis/raw_response(Gemini 원문 전체)도 포함한다."""


class LabelResultOut(BaseModel):
    score: int
    decision: str
    label: int
    tag: str
    reason: str
    analysis: Optional[str] = None
    raw_response: Optional[str] = None
    parse_warnings: List[str]


class BatchLabelItemOut(BaseModel):
    arxiv_id: Optional[str] = None
    title: Optional[str] = None
    result: Optional[LabelResultOut] = None
    error: Optional[str] = None


class BatchLabelResponse(BaseModel):
    count: int
    success_count: int
    failed_count: int
    positive_count: int
    results: List[BatchLabelItemOut]
