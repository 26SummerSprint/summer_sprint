"""보관함 API — 재열람용 논문 저장/조회/삭제.

- POST   /saved             : 논문 1건 보관(맥락 포함)
- GET    /saved             : 보관 목록(표시 순서)
- DELETE /saved/{id}        : 보관 1건 삭제
- POST   /saved/{id}/move   : 보관 순위 한 칸 이동(up/down)
"""

from typing import List

from fastapi import APIRouter, HTTPException, Query

from ..schemas import SavedPaperOut, SavedPaperRequest
from ..services import saved_store

router = APIRouter()


@router.post("/saved", response_model=SavedPaperOut)
def add_saved(req: SavedPaperRequest):
    try:
        saved = saved_store.add_saved(req.model_dump())
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"보관 실패: {e}") from e
    return saved


@router.get("/saved", response_model=List[SavedPaperOut])
def list_saved():
    return saved_store.list_saved()


@router.delete("/saved/{arxiv_id}")
def remove_saved(arxiv_id: str):
    removed = saved_store.remove_saved(arxiv_id)
    return {"ok": True, "removed": removed}


@router.post("/saved/{arxiv_id}/move")
def move_saved(
    arxiv_id: str,
    direction: str = Query(..., pattern="^(up|down)$"),
):
    moved = saved_store.move_saved(arxiv_id, direction)
    return {"ok": True, "moved": moved}
