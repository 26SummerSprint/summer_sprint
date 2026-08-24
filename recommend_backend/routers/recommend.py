"""POST /recommend 엔드포인트. RecommendService를 그대로 호출만 한다."""

from fastapi import APIRouter, HTTPException

from ..schemas import (
    RecommendFromPaperRequest,
    RecommendRequest,
    RecommendResponse,
)
from ..services.paper_keyword_extraction_service import (
    PaperKeywordExtractionService,
    PaperKeywordExtractionServiceError,
)
from ..services.recommend_service import RecommendService

router = APIRouter()

# 프로세스당 1회만 생성 (각 클라이언트/서비스는 상태 없는 HTTP 래퍼라 재사용 가능).
_service = RecommendService()
_paper_keyword_service = PaperKeywordExtractionService()


@router.post("/recommend", response_model=RecommendResponse)
async def recommend(req: RecommendRequest) -> RecommendResponse:
    try:
        return await _service.recommend(
            profile=req.profile, category=req.category, diversity=req.diversity
        )
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=str(e)) from e


@router.post("/recommend/from_paper", response_model=RecommendResponse)
async def recommend_from_paper(req: RecommendFromPaperRequest) -> RecommendResponse:
    """
    '이 논문으로 다시 추천받기'.

    선택한 논문 한 편의 abstract에서 핵심 연구 키워드만 새로 추출하고,
    기존 프로필/기존 검색 키워드는 전혀 쓰지 않은 채 그 키워드만으로
    기존 추천 파이프라인(Hybrid Retrieval → Re-ranker → Gemini Final
    Selection)을 그대로 다시 실행한다. 선택한 논문 자신은 결과에서 제외된다.
    """
    try:
        keywords = await _paper_keyword_service.extract(req.abstract)
    except PaperKeywordExtractionServiceError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e

    if not keywords:
        raise HTTPException(
            status_code=422, detail="논문에서 핵심 키워드를 추출하지 못했습니다."
        )

    pseudo_profile = "; ".join(keywords)

    try:
        return await _service.recommend(
            profile=pseudo_profile,
            category=req.category,
            override_keywords=keywords,
            exclude_arxiv_ids=[req.arxiv_id],
        )
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=str(e)) from e
