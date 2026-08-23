"""
Recommend Backend (오케스트레이션 서버) FastAPI 앱 진입점.

arxiv_pipeline / keyword_pipeline / rerank_pipeline은 전부 별도로 떠 있는
서비스라고 가정하고, 그 주소를 환경변수로 주입받아 HTTP로 호출한다.
(rerank_pipeline은 summerSprint 기준 아직 배포되어 있지 않아서, 주소가
비어 있으면 그 단계는 자동으로 건너뛴다. 자세한 내용은 README.md,
services/recommend_service.py 참고.)

실행:
    uvicorn recommend_backend.main:app --reload --port 8100
"""

from fastapi import FastAPI

from .routers.recommend import router as recommend_router
from .routers.feedback import router as feedback_router
from .routers.saved import router as saved_router

app = FastAPI(title="AI Paper Recommender - Orchestration Backend", version="0.1.0")

app.include_router(recommend_router, prefix="/api/v1", tags=["recommend"])
app.include_router(feedback_router, prefix="/api/v1", tags=["feedback"])
app.include_router(saved_router, prefix="/api/v1", tags=["saved"])


@app.get("/health")
def health():
    return {"status": "ok"}
