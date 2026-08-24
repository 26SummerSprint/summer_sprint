"""
RecommendService 요청/응답 스키마. 요청하신 JSON 응답 형식과 1:1로 대응한다.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class RecommendRequest(BaseModel):
    profile: str
    category: Optional[str] = None
    diversity: float = 0.0  # 다양성(MMR) 강도. 0=관련성만, 클수록 다양성↑ (권장 0~0.7)


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


class FeedbackRequest(BaseModel):
    """사용자 추천 피드백 1건. 키워드 단위로 저장해 프로필이 달라도
    같은 키워드 세션에 반영한다(닫힌 학습 루프)."""
    profile: str
    keywords: List[str] = []   # 세션의 추출 키워드 → 키워드 단위 반영의 핵심
    arxiv_id: str
    title: Optional[str] = None
    category: Optional[str] = None
    vote: str  # "up" | "down" | "save"


class SavedPaperRequest(BaseModel):
    """재열람용 보관 1건. 추천 당시의 맥락(프로필·키워드·이유·초록)을 함께 저장해
    나중에 그대로 다시 볼 수 있게 한다(학습 신호와 분리)."""
    arxiv_id: str
    title: Optional[str] = None
    link: Optional[str] = None            # abs_url 등 논문 링크
    pdf_url: Optional[str] = None
    profile: Optional[str] = None         # 추천 시 입력한 프로필
    keywords: List[str] = []              # 추출된 키워드
    reason: Optional[str] = None          # 추천 이유(Gemini)
    abstract: Optional[str] = None        # 초록
    category: Optional[str] = None


class SavedPaperOut(SavedPaperRequest):
    ts: str  # 저장 시각(ISO)


class RecommendFromPaperRequest(BaseModel):
    """
    '이 논문으로 다시 추천받기' 요청. 기존 프로필/키워드는 전혀 쓰지 않고,
    이 논문의 abstract에서 새로 추출한 키워드만으로 추천 파이프라인을
    다시 실행한다. 선택한 논문 자신은 결과에서 제외된다.
    """
    arxiv_id: str
    title: Optional[str] = None
    abstract: str
    category: Optional[str] = None
