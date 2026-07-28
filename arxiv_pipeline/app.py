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
from db import (
    PaperRecord,
    count_papers,
    count_labels,
    delete_papers_by_ids,
    get_conn,
    get_ids_by_primary_category,
    get_labels,
    get_paper_by_id,
    get_papers_by_ids,
    init_db,
    keyword_df,
    upsert_label,
    upsert_paper,
)
from embedder import delete_vectors, embed_pending_papers, search_similar
from preprocess import clean_abstract
from retrieval import build_labeling_pool, hybrid_retrieve

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


class RawPaperIn(BaseModel):
    """팀원이 본인 노트북에서 arXiv로 수집한 원본 논문 1건 (abstract_clean은 서버가 계산)."""
    arxiv_id: str
    version: int = 1
    title: str
    authors: List[str]
    abstract_raw: str
    categories: List[str]
    primary_category: str
    comments: Optional[str] = None
    submitted_date: str  # ISO 8601
    updated_date: str  # ISO 8601
    abs_url: str
    pdf_url: str


class IngestRequest(BaseModel):
    papers: List[RawPaperIn]
    auto_embed: bool = True  # 저장 직후 바로 임베딩까지 계산할지 여부


class IngestResponse(BaseModel):
    ingested_or_updated: int
    newly_embedded: int


class DeleteRequest(BaseModel):
    """arxiv_ids 또는 category 중 정확히 하나만 지정.
    전체 삭제(--all)는 위험해서 API로 제공하지 않음 — 서버에서 reset_data.py 사용."""
    arxiv_ids: Optional[List[str]] = None
    category: Optional[str] = None  # primary_category 기준 (예: "cs.CL")
    confirm: bool = False  # 실수 방지: True를 명시해야 실제 삭제


class DeleteResponse(BaseModel):
    deleted: int
    remaining: int


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


# ── Stage1 하이브리드 검색 / 골드셋 후보 / 라벨 ──────────────
class RetrieveRequest(BaseModel):
    profile_text: str
    keywords: List[str] = []
    category: Optional[str] = None
    n_keyword: int = 30
    m_embedding: int = 70


class CandidateItem(BaseModel):
    arxiv_id: str
    source: str  # keyword | embedding | both | random
    title: str = ""
    primary_category: str = ""
    submitted_date: str = ""
    abs_url: Optional[str] = None
    abstract_clean: Optional[str] = None


class LabelingPoolRequest(BaseModel):
    profile_text: str
    keywords: List[str] = []
    category: Optional[str] = None
    total: int = 60           # 프로필당 후보 편수 (회의 확정 60)
    n_random: int = 0         # 그중 랜덤 샘플 (기본 0=하이브리드만)
    n_keyword: int = 30
    m_embedding: int = 70


class LabelIn(BaseModel):
    profile_id: str
    arxiv_id: str
    labeler: str            # 'judge' 또는 사람 이름
    label: int              # 0/1 (또는 0~3)
    source: Optional[str] = None
    tag: Optional[str] = None
    profile_version: str = "v1"


class LabelsUploadRequest(BaseModel):
    labels: List[LabelIn]


class LabelsUploadResponse(BaseModel):
    saved: int
    total: int


def _attach_meta(candidates: List[dict]) -> List[CandidateItem]:
    """후보 id 리스트에 SQLite 메타데이터를 붙여 CandidateItem으로 변환 (라벨링용)."""
    ids = [c["arxiv_id"] for c in candidates]
    with get_conn() as conn:
        meta = get_papers_by_ids(conn, ids)
    items = []
    for c in candidates:
        m = meta.get(c["arxiv_id"], {})
        items.append(
            CandidateItem(
                arxiv_id=c["arxiv_id"],
                source=c["source"],
                title=m.get("title", ""),
                primary_category=m.get("primary_category", ""),
                submitted_date=m.get("submitted_date", ""),
                abs_url=m.get("abs_url"),
                abstract_clean=m.get("abstract_clean"),
            )
        )
    return items


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


@app.post("/retrieve", response_model=List[CandidateItem], dependencies=[Depends(verify_api_key)])
def retrieve(req: RetrieveRequest):
    """
    Stage1 하이브리드 검색 (쿼터 union): 키워드 top N + 임베딩 top M, 중복 제거.
    데일리 추천의 후보 추출 단계. 최종 순위는 재랭커(Stage2)가 다시 매긴다.
    """
    candidates = hybrid_retrieve(
        req.profile_text, req.keywords, req.category, req.n_keyword, req.m_embedding
    )
    return _attach_meta(candidates)


@app.post("/labeling_pool", response_model=List[CandidateItem], dependencies=[Depends(verify_api_key)])
def labeling_pool(req: LabelingPoolRequest):
    """
    골드셋 라벨링 후보 풀 = 하이브리드 ∪ 랜덤 샘플.
    랜덤 샘플로 pool bias를 완화하고 0(무관) 라벨을 확보한다.
    """
    candidates = build_labeling_pool(
        req.profile_text, req.keywords, category=req.category,
        total=req.total, n_random=req.n_random,
        n_keyword=req.n_keyword, m_embedding=req.m_embedding,
    )
    return _attach_meta(candidates)


@app.get("/keyword_df", dependencies=[Depends(verify_api_key)])
def get_keyword_df(term: str = Query(...), category: Optional[str] = None):
    """키워드 DF(문서빈도) 조회 — 프로필 키워드의 50~500 검증용."""
    with get_conn() as conn:
        df_all = keyword_df(conn, term, category=None)
        df_cat = keyword_df(conn, term, category=category) if category else None
    return {"term": term, "df": df_all, "df_in_category": df_cat}


@app.post("/labels", response_model=LabelsUploadResponse, dependencies=[Depends(verify_api_key)])
def upload_labels(req: LabelsUploadRequest):
    """골드셋 라벨 업로드 (judge 채점 결과 또는 사람 검증 라벨)."""
    with get_conn() as conn:
        for lab in req.labels:
            upsert_label(
                conn, lab.profile_id, lab.arxiv_id, lab.labeler, lab.label,
                source=lab.source, tag=lab.tag, profile_version=lab.profile_version,
            )
        total = count_labels(conn)
    return LabelsUploadResponse(saved=len(req.labels), total=total)


@app.get("/labels", dependencies=[Depends(verify_api_key)])
def list_labels(profile_id: Optional[str] = None):
    """저장된 라벨 조회 (profile_id 지정 시 해당 프로필만)."""
    with get_conn() as conn:
        return get_labels(conn, profile_id)


@app.post("/papers/ingest", response_model=IngestResponse, dependencies=[Depends(verify_api_key)])
def ingest_papers(req: IngestRequest):
    """
    팀원이 본인 노트북에서 arXiv 수집(collect_and_push.py)한 결과를 공유 DB에 반영.
    - 초록 클린업(clean_abstract)과 버전 비교 upsert는 서버가 일괄 처리 (규칙 일원화).
    - auto_embed=True(기본값)면 저장 직후 이 서버에서 임베딩까지 계산해 바로 검색 가능해짐.
    """
    changed = 0
    with get_conn() as conn:
        for p in req.papers:
            record = PaperRecord(
                arxiv_id=p.arxiv_id,
                version=p.version,
                title=p.title,
                authors=p.authors,
                abstract_raw=p.abstract_raw,
                abstract_clean=clean_abstract(p.abstract_raw),
                categories=p.categories,
                primary_category=p.primary_category,
                comments=p.comments,
                submitted_date=p.submitted_date,
                updated_date=p.updated_date,
                abs_url=p.abs_url,
                pdf_url=p.pdf_url,
            )
            if upsert_paper(conn, record):
                changed += 1

    newly_embedded = embed_pending_papers() if req.auto_embed else 0
    return IngestResponse(ingested_or_updated=changed, newly_embedded=newly_embedded)


@app.post("/papers/delete", response_model=DeleteResponse, dependencies=[Depends(verify_api_key)])
def delete_papers(req: DeleteRequest):
    """
    논문 삭제 (SQLite + 모든 Chroma 컬렉션에서 함께 제거). 되돌릴 수 없음.
    - arxiv_ids 지정: 해당 논문들만 삭제
    - category 지정: primary_category가 일치하는 논문 일괄 삭제 (카테고리 개편용)
    """
    if not req.confirm:
        raise HTTPException(status_code=400, detail="confirm=true가 필요합니다 (실수 방지)")
    if bool(req.arxiv_ids) == bool(req.category):
        raise HTTPException(
            status_code=400, detail="arxiv_ids 또는 category 중 정확히 하나만 지정하세요"
        )

    with get_conn() as conn:
        if req.category:
            ids = get_ids_by_primary_category(conn, req.category)
        else:
            ids = req.arxiv_ids
        deleted = delete_papers_by_ids(conn, ids)
        remaining = count_papers(conn)

    delete_vectors(ids)
    return DeleteResponse(deleted=deleted, remaining=remaining)


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
