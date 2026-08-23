"""POST /recommend 엔드포인트. RecommendService를 그대로 호출만 한다."""

from fastapi import APIRouter, HTTPException

from ..schemas import RecommendRequest, RecommendResponse
from ..services.recommend_service import RecommendService

router = APIRouter()

# 프로세스당 1회만 생성 (각 클라이언트/서비스는 상태 없는 HTTP 래퍼라 재사용 가능).
_service = RecommendService()


@router.post("/recommend", response_model=RecommendResponse)
async def recommend(req: RecommendRequest) -> RecommendResponse:
    try:
        return await _service.recommend(
            profile=req.profile, category=req.category, diversity=req.diversity
        )
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=str(e)) from e
