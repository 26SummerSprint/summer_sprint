"""
RecommendService 요청/응답 스키마.

입력 유효성(valid) 정보를 포함하여
KeywordExtractionService -> RecommendService -> API Response
전체 단계에서 동일하게 사용한다.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class RecommendRequest(BaseModel):
    profile: str

    category: Optional[str] = None

    diversity: float = 0.0
    # 다양성(MMR) 강도
    # 0 = 관련성 중심
    # 클수록 다양성 증가


class ExtractedProfile(BaseModel):
    """
    Stage 1a Gemini 프로필 분석 결과.

    valid
        사용자의 입력이 실제 연구 관심사인지 여부.

    profile_text_en
        영어 embedding 검색 쿼리.

    keywords
        BM25 검색용 키워드.

    exclude
        제외 조건.
    """

    valid: bool = True

    profile_text_en: str = ""

    keywords: List[str] = Field(
        default_factory=list
    )

    exclude: List[str] = Field(
        default_factory=list
    )


class RecommendedPaperOut(BaseModel):
    rank: int

    paper: Dict[str, Any]

    reason: str


class RecommendResponse(BaseModel):
    profile: str

    extracted_profile: ExtractedProfile

    count: int

    recommendations: List[RecommendedPaperOut]


class FeedbackRequest(BaseModel):
    """
    사용자 추천 피드백 1건.

    키워드 단위로 저장하여
    프로필이 달라도 같은 키워드 세션에 반영한다.
    """

    profile: str

    keywords: List[str] = Field(
        default_factory=list
    )

    arxiv_id: str

    title: Optional[str] = None

    category: Optional[str] = None

    vote: str
    # "up" | "down" | "save"


class SavedPaperRequest(BaseModel):
    """
    재열람용 보관 1건.

    추천 당시의 맥락을 함께 저장한다.
    """

    arxiv_id: str

    title: Optional[str] = None

    link: Optional[str] = None

    pdf_url: Optional[str] = None

    profile: Optional[str] = None

    keywords: List[str] = Field(
        default_factory=list
    )

    reason: Optional[str] = None

    abstract: Optional[str] = None

    category: Optional[str] = None


class SavedPaperOut(SavedPaperRequest):

    ts: str


class RecommendFromPaperRequest(BaseModel):
    """
    '이 논문으로 다시 추천받기' 요청.

    기존 프로필/키워드는 사용하지 않고,
    해당 논문의 abstract를 기반으로
    새로운 키워드를 추출하여 추천한다.

    선택한 논문 자신은 결과에서 제외한다.
    """

    arxiv_id: str

    title: Optional[str] = None

    abstract: str

    category: Optional[str] = None