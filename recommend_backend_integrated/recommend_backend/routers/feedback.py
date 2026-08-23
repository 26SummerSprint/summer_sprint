"""POST /feedback — 사용자 피드백(👍/👎/저장)을 JSONL로 로깅.

수집한 피드백은 골드셋 확장·재랭커 재학습(닫힌 학습 루프)의 재료로 쓴다.
경로: config.FEEDBACK_LOG_PATH (기본 recommend_backend/feedback_log.jsonl).
"""

import json
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException

from ..config import FEEDBACK_LOG_PATH
from ..schemas import FeedbackRequest

router = APIRouter()


@router.post("/feedback")
def feedback(req: FeedbackRequest):
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "profile": req.profile,
        "keywords": req.keywords,   # 키워드 단위 반영의 핵심
        "category": req.category,
        "arxiv_id": req.arxiv_id,
        "title": req.title,
        "vote": req.vote,  # up | down | save
    }
    try:
        with open(FEEDBACK_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"feedback 기록 실패: {e}") from e
    return {"ok": True}
