"""
RecommendService 요청/응답 스키마. 요청하신 JSON 응답 형식과 1:1로 대응한다.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class RecommendRequest(BaseModel):
    profile: str
    category: Optional[str] = None


class ExtractedProfile(BaseModel):
    profile_text_en: str
    """Gemini가 번역/재구성한 영어 검색 쿼리 (임베딩/재랭커 입력으로 사용)."""
    keywords: List[str]
    exclude: List[str]


class RecommendedPaperOut(BaseModel):
    rank: int
    paper: Dict[str, Any]
    reason: str


class RecommendResponse(BaseModel):
    profile: str
    extracted_profile: ExtractedProfile
    count: int
    recommendations: List[RecommendedPaperOut]
