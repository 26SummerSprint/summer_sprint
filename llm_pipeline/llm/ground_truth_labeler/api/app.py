"""
Ground Truth Labeler - 독립 API 서비스.

라벨링 로직(GroundTruthLabeler)을 그대로 재사용하고 HTTP로 노출한다.
다른 서비스(recommend_backend, 평가/데이터셋 구축 파이프라인 등)가 이 API를
호출해서 Ground Truth 라벨을 받아갈 수 있다.

실행:
    uvicorn ground_truth_labeler.api.app:app --reload --port 8300
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routers.labeling import router as labeling_router

app = FastAPI(title="Ground Truth Labeler API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(labeling_router, tags=["labeling"])


@app.get("/health")
def health():
    return {"status": "ok"}
