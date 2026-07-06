"""
FastAPI 서버.

- AWS 인스턴스에서 이 앱을 띄우면, 팀원들은 chromadb/sentence-transformers를
  설치하지 않고도 REST API로 벡터 검색 결과를 받아볼 수 있음.
- 쿼리 텍스트(관심사 프로필)를 서버가 임베딩까지 계산해주므로, 클라이언트는
  순수 텍스트만 보내면 됨.

실행:
    uvicorn app:app --host 0.0.0.0 --port 8000

pip install fastapi "uvicorn[standard]"
"""

from typing import List, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from pydantic import BaseModel

from config import API_KEY
from db import count_papers, get_conn, get_paper_by_id, get_papers_by_ids, init_db
from embedder import search_similar

app = FastAPI(title="arXiv Paper Search API", version="1.0")

init_db()


# ── 인증 ────────────────────────────────────────────────
def verify_api_key(x_api_key: str = Header(..., alias="X-API-Key")):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return True


# ── 스키마 ───────────────────────────────────────────────
class SearchRequest(BaseModel):
    query: str
    top_k: int = 20
    category: Optional[str] = None  # 예: "cs.RO"


class SearchResultItem(BaseModel):
    arxiv_id: str
    score: float  # 거리값 (작을수록 유사, cosine distance)
    title: str
    primary_category: str
    submitted_date: str
    abs_url: Optional[str] = None
    pdf_url: Optional[str] = None
    abstract_clean: Optional[str] = None


class PaperDetail(BaseModel):
    arxiv_id: str
    version: int
    title: str
    authors: str
    abstract_raw: str
    abstract_clean: str
    categories: str
    primary_category: str
    comments: Optional[str]
    submitted_date: str
    updated_date: str
    abs_url: str
    pdf_url: str


# ── 엔드포인트 ───────────────────────────────────────────
@app.get("/health")
def health():
    with get_conn() as conn:
        total = count_papers(conn)
    return {"status": "ok", "paper_count": total}


@app.post("/search", response_model=List[SearchResultItem], dependencies=[Depends(verify_api_key)])
def search(req: SearchRequest):
    """
    관심사 프로필(또는 임의 쿼리 텍스트)로 유사 논문 top_k 검색.
    벤치마크 B단계의 '임베딩 검색 후보 추출'에 바로 사용 가능.
    """
    raw_results = search_similar(req.query, top_k=req.top_k, category=req.category)
    ids = [r[0] for r in raw_results]

    with get_conn() as conn:
        meta_map = get_papers_by_ids(conn, ids)

    items = []
    for arxiv_id, distance, chroma_meta in raw_results:
        full = meta_map.get(arxiv_id, {})
        items.append(
            SearchResultItem(
                arxiv_id=arxiv_id,
                score=distance,
                title=full.get("title") or chroma_meta.get("title", ""),
                primary_category=full.get("primary_category") or chroma_meta.get("primary_category", ""),
                submitted_date=full.get("submitted_date") or chroma_meta.get("submitted_date", ""),
                abs_url=full.get("abs_url"),
                pdf_url=full.get("pdf_url"),
                abstract_clean=full.get("abstract_clean"),
            )
        )
    return items


@app.get("/papers/{arxiv_id}", response_model=PaperDetail, dependencies=[Depends(verify_api_key)])
def get_paper(arxiv_id: str):
    with get_conn() as conn:
        record = get_paper_by_id(conn, arxiv_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Paper not found")
    return record


@app.get("/papers", response_model=List[PaperDetail], dependencies=[Depends(verify_api_key)])
def list_papers_by_ids(ids: str = Query(..., description="쉼표로 구분된 arxiv_id 목록")):
    """여러 건을 한 번에 조회 (라벨링 후보 리스트를 화면에 뿌릴 때 유용)."""
    id_list = [i.strip() for i in ids.split(",") if i.strip()]
    with get_conn() as conn:
        meta_map = get_papers_by_ids(conn, id_list)
    return list(meta_map.values())
