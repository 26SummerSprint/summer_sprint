"""
FastAPI 서버.

- AWS EC2에서 실행
- SQLite + ChromaDB + SentenceTransformer를 서버에서 관리
- recommend_backend는 이 서버의 REST API를 호출

주요 엔드포인트
    GET  /health

    POST /search
        Embedding 단독 검색

    POST /retrieve
        BM25 + Embedding Hybrid Retrieval

    POST /rerank
        CrossEncoder 재랭킹

    GET  /keyword_df
        키워드 Document Frequency 조회

    POST /labeling_pool
        골드셋 라벨링 후보 생성

    POST /labels
    GET  /labels

    POST /papers/ingest
    POST /papers/delete

    GET /papers/{arxiv_id}
    GET /papers

실행:
    uvicorn app:app --host 0.0.0.0 --port 8000
"""

from typing import List, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from pydantic import BaseModel

from config import API_KEY, RERANKER_PATH

from db import (
    PaperRecord,
    count_labels,
    count_papers,
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

from embedder import (
    delete_vectors,
    embed_pending_papers,
    search_similar,
)

from preprocess import clean_abstract

from retrieval import (
    build_labeling_pool,
    hybrid_retrieve,
)

# CrossEncoder 재랭커
from reranker import load_reranker, rerank as run_rerank


# ============================================================
# FastAPI
# ============================================================

app = FastAPI(
    title="arXiv Paper Search API",
    version="1.1",
)

init_db()


# ============================================================
# 인증
# ============================================================

def verify_api_key(
    x_api_key: str = Header(..., alias="X-API-Key"),
):
    if x_api_key != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key",
        )

    return True


# ============================================================
# Reranker 설정
# ============================================================

# EC2의 arxiv_pipeline 디렉터리 기준
#
# 예:
#
# arxiv_pipeline/
# ├── app.py
# ├── reranker.py
# └── reranker_model/
#
RERANK_MODEL_PATH = RERANKER_PATH  # config 단일 출처 (train_reranker_cli.py 저장 경로와 일치)

_reranker_model = None


def get_reranker_model():
    """
    CrossEncoder 모델을 최초 /rerank 요청 때 한 번만 로드한다.

    모델 로딩은 매우 무거울 수 있으므로 서버 시작 시 바로 로드하지 않고
    첫 번째 rerank 요청에서 lazy loading한다.
    """

    global _reranker_model

    if _reranker_model is None:

        print(
            f"[Reranker] 모델 로드 시작: {RERANK_MODEL_PATH}"
        )

        try:
            _reranker_model = load_reranker(
                RERANK_MODEL_PATH
            )

        except Exception as e:
            print(
                f"[Reranker] 모델 로드 실패: {e}"
            )

            raise HTTPException(
                status_code=500,
                detail=f"Reranker model load failed: {e}",
            )

        print(
            "[Reranker] 모델 로드 완료"
        )

    return _reranker_model


# ============================================================
# 스키마
# ============================================================


# ------------------------------------------------------------
# Search
# ------------------------------------------------------------

class SearchRequest(BaseModel):
    query: str
    top_k: int = 20
    category: Optional[str] = None


class SearchResultItem(BaseModel):
    arxiv_id: str
    score: float

    title: str
    primary_category: str
    submitted_date: str

    abs_url: Optional[str] = None
    pdf_url: Optional[str] = None
    abstract_clean: Optional[str] = None


# ------------------------------------------------------------
# Paper Ingest
# ------------------------------------------------------------

class RawPaperIn(BaseModel):
    """
    팀원이 본인 노트북에서 arXiv로 수집한 원본 논문 1건.
    abstract_clean은 서버에서 계산한다.
    """

    arxiv_id: str

    version: int = 1

    title: str

    authors: List[str]

    abstract_raw: str

    categories: List[str]

    primary_category: str

    comments: Optional[str] = None

    submitted_date: str

    updated_date: str

    abs_url: str

    pdf_url: str


class IngestRequest(BaseModel):
    papers: List[RawPaperIn]

    auto_embed: bool = True


class IngestResponse(BaseModel):
    ingested_or_updated: int
    newly_embedded: int


# ------------------------------------------------------------
# Paper Delete
# ------------------------------------------------------------

class DeleteRequest(BaseModel):
    """
    arxiv_ids 또는 category 중 정확히 하나만 지정.

    전체 삭제(--all)는 API로 제공하지 않는다.
    """

    arxiv_ids: Optional[List[str]] = None

    category: Optional[str] = None

    confirm: bool = False


class DeleteResponse(BaseModel):
    deleted: int
    remaining: int


# ------------------------------------------------------------
# Paper Detail
# ------------------------------------------------------------

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


# ============================================================
# Stage 1 Hybrid Retrieval
# ============================================================

class RetrieveRequest(BaseModel):
    profile_text: str

    keywords: List[str] = []

    category: Optional[str] = None

    n_keyword: int = 30

    m_embedding: int = 70


class CandidateItem(BaseModel):
    arxiv_id: str

    # keyword | embedding | both | random
    source: str

    title: str = ""

    primary_category: str = ""

    submitted_date: str = ""

    abs_url: Optional[str] = None

    abstract_clean: Optional[str] = None


# ============================================================
# Stage 2 Reranker
# ============================================================

class RerankCandidate(BaseModel):
    """
    recommend_backend에서 /rerank로 전달하는 후보 논문.
    """

    arxiv_id: str

    title: str = ""

    abstract_clean: Optional[str] = None


class RerankRequest(BaseModel):
    """
    CrossEncoder 재랭킹 요청.
    """

    profile_text: str

    candidates: List[RerankCandidate]

    # 다양성 재랭킹(MMR) 강도. 0.0=관련성만(기존 동작), 클수록 다양성↑ (권장 0~0.7)
    diversity: float = 0.0


class RerankResult(BaseModel):
    """
    CrossEncoder 재랭킹 결과.
    """

    arxiv_id: str

    title: str = ""

    abstract_clean: Optional[str] = None

    rerank_score: float


class RerankResponse(BaseModel):
    results: List[RerankResult]


# ============================================================
# Labeling Pool
# ============================================================

class LabelingPoolRequest(BaseModel):
    profile_text: str

    keywords: List[str] = []

    category: Optional[str] = None

    total: int = 60

    n_random: int = 10

    n_keyword: int = 30

    m_embedding: int = 70


# ============================================================
# Labels
# ============================================================

class LabelIn(BaseModel):
    profile_id: str

    arxiv_id: str

    labeler: str

    label: int

    source: Optional[str] = None

    tag: Optional[str] = None

    profile_version: str = "v1"


class LabelsUploadRequest(BaseModel):
    labels: List[LabelIn]


class LabelsUploadResponse(BaseModel):
    saved: int

    total: int


# ============================================================
# Helper
# ============================================================

def _attach_meta(
    candidates: List[dict],
) -> List[CandidateItem]:
    """
    후보 ID 리스트에 SQLite 메타데이터를 붙인다.
    """

    ids = [
        c["arxiv_id"]
        for c in candidates
    ]

    with get_conn() as conn:

        meta = get_papers_by_ids(
            conn,
            ids,
        )

    items = []

    for c in candidates:

        m = meta.get(
            c["arxiv_id"],
            {},
        )

        items.append(
            CandidateItem(
                arxiv_id=c["arxiv_id"],

                source=c["source"],

                title=m.get(
                    "title",
                    "",
                ),

                primary_category=m.get(
                    "primary_category",
                    "",
                ),

                submitted_date=m.get(
                    "submitted_date",
                    "",
                ),

                abs_url=m.get(
                    "abs_url",
                ),

                abstract_clean=m.get(
                    "abstract_clean",
                ),
            )
        )

    return items


# ============================================================
# Health
# ============================================================

@app.get("/health")
def health():

    with get_conn() as conn:

        total = count_papers(conn)

    return {
        "status": "ok",
        "paper_count": total,
    }


# ============================================================
# Stage 1 - Embedding Search
# ============================================================

@app.post(
    "/search",
    response_model=List[SearchResultItem],
    dependencies=[Depends(verify_api_key)],
)
def search(req: SearchRequest):
    """
    관심사 프로필 또는 임의의 쿼리 텍스트로
    Embedding top-k 검색.
    """

    raw_results = search_similar(
        req.query,
        top_k=req.top_k,
        category=req.category,
    )

    ids = [
        r[0]
        for r in raw_results
    ]

    with get_conn() as conn:

        meta_map = get_papers_by_ids(
            conn,
            ids,
        )

    items = []

    for (
        arxiv_id,
        distance,
        chroma_meta,
    ) in raw_results:

        full = meta_map.get(
            arxiv_id,
            {},
        )

        items.append(
            SearchResultItem(
                arxiv_id=arxiv_id,

                score=distance,

                title=(
                    full.get("title")
                    or chroma_meta.get(
                        "title",
                        "",
                    )
                ),

                primary_category=(
                    full.get(
                        "primary_category"
                    )
                    or chroma_meta.get(
                        "primary_category",
                        "",
                    )
                ),

                submitted_date=(
                    full.get(
                        "submitted_date"
                    )
                    or chroma_meta.get(
                        "submitted_date",
                        "",
                    )
                ),

                abs_url=full.get(
                    "abs_url"
                ),

                pdf_url=full.get(
                    "pdf_url"
                ),

                abstract_clean=full.get(
                    "abstract_clean"
                ),
            )
        )

    return items


# ============================================================
# Stage 1 - Hybrid Retrieval
# ============================================================

@app.post(
    "/retrieve",
    response_model=List[CandidateItem],
    dependencies=[Depends(verify_api_key)],
)
def retrieve(req: RetrieveRequest):
    """
    Stage 1 하이브리드 검색.

    keyword top N
    +
    embedding top M

    이후 중복 제거하여 후보를 반환한다.

    최종 순위는 Stage 2 /rerank에서 결정한다.
    """

    candidates = hybrid_retrieve(
        req.profile_text,
        req.keywords,
        req.category,
        req.n_keyword,
        req.m_embedding,
    )

    return _attach_meta(
        candidates
    )


# ============================================================
# Stage 2 - CrossEncoder Reranking
# ============================================================

@app.post(
    "/rerank",
    response_model=RerankResponse,
    dependencies=[Depends(verify_api_key)],
)
def rerank_candidates(
    req: RerankRequest,
):
    """
    Stage 2 CrossEncoder 재랭킹.

    입력:
        profile_text
        candidates

    처리:
        profile_text + title + abstract
        ↓
        CrossEncoder
        ↓
        rerank_score
        ↓
        내림차순 정렬

    모델은 최초 요청에서 한 번만 로드한다.
    """

    # 후보가 없는 경우
    if not req.candidates:

        return RerankResponse(
            results=[]
        )

    # --------------------------------------------------------
    # 모델 로드
    # --------------------------------------------------------

    model = get_reranker_model()

    # --------------------------------------------------------
    # CrossEncoder 입력 생성
    # --------------------------------------------------------

    rerank_candidates_data = []

    for candidate in req.candidates:

        title = (
            candidate.title
            or ""
        ).strip()

        abstract = (
            candidate.abstract_clean
            or ""
        ).strip()

        # reranker.py의 ABSTRACT_CHARS와 동일
        abstract = abstract[:1200]

        doc = (
            f"{title}. {abstract}"
        ).strip(". ").strip()

        rerank_candidates_data.append(
            {
                "arxiv_id": candidate.arxiv_id,

                "title": candidate.title,

                "abstract_clean": (
                    candidate.abstract_clean
                ),

                "doc": doc,
            }
        )

    # --------------------------------------------------------
    # CrossEncoder 실행
    # --------------------------------------------------------

    try:

        ranked = run_rerank(
            model,
            req.profile_text,
            rerank_candidates_data,
        )

    except Exception as e:

        print(
            f"[Reranker] 실행 실패: {e}"
        )

        raise HTTPException(
            status_code=500,
            detail=f"Reranker execution failed: {e}",
        )

    # --------------------------------------------------------
    # 다양성 재랭킹 (MMR) — diversity > 0 일 때만
    # 관련도(rerank_score) + 후보 임베딩(코사인)으로 재정렬.
    # --------------------------------------------------------
    if req.diversity and req.diversity > 0 and len(ranked) > 2:
        try:
            from embedder import embed_texts
            from reranker import mmr_order

            embs = embed_texts([it["doc"] for it in ranked])
            rels = [it["rerank_score"] for it in ranked]
            lam = max(0.0, min(1.0, 1.0 - float(req.diversity)))  # diversity→lambda
            order = mmr_order(rels, embs, lambda_mult=lam)
            ranked = [ranked[i] for i in order]
        except Exception as e:  # noqa: BLE001
            print(f"[Reranker] MMR 실패(관련성 순서 유지): {e}")

    # --------------------------------------------------------
    # Response 생성
    # --------------------------------------------------------

    results = []

    for item in ranked:

        results.append(
            RerankResult(
                arxiv_id=item[
                    "arxiv_id"
                ],

                title=item.get(
                    "title",
                    "",
                ),

                abstract_clean=item.get(
                    "abstract_clean"
                ),

                rerank_score=float(
                    item[
                        "rerank_score"
                    ]
                ),
            )
        )

    return RerankResponse(
        results=results
    )


# ============================================================
# Keyword DF
# ============================================================

@app.get(
    "/keyword_df",
    dependencies=[Depends(verify_api_key)],
)
def get_keyword_df(
    term: str = Query(...),
    category: Optional[str] = None,
):
    """
    키워드 Document Frequency 조회.

    recommend_backend의 KeywordExtractionService가
    키워드 검증에 사용한다.
    """

    with get_conn() as conn:

        df_all = keyword_df(
            conn,
            term,
            category=None,
        )

        df_cat = (
            keyword_df(
                conn,
                term,
                category=category,
            )
            if category
            else None
        )

    return {
        "term": term,
        "df": df_all,
        "df_in_category": df_cat,
    }


# ============================================================
# Labeling Pool
# ============================================================

@app.post(
    "/labeling_pool",
    response_model=List[CandidateItem],
    dependencies=[Depends(verify_api_key)],
)
def labeling_pool(
    req: LabelingPoolRequest,
):
    """
    골드셋 라벨링 후보 풀.

    Hybrid Retrieval
    +
    Random Sample
    """

    candidates = build_labeling_pool(
        req.profile_text,
        req.keywords,
        category=req.category,
        total=req.total,
        n_random=req.n_random,
        n_keyword=req.n_keyword,
        m_embedding=req.m_embedding,
    )

    return _attach_meta(
        candidates
    )


# ============================================================
# Labels Upload
# ============================================================

@app.post(
    "/labels",
    response_model=LabelsUploadResponse,
    dependencies=[Depends(verify_api_key)],
)
def upload_labels(
    req: LabelsUploadRequest,
):
    """
    골드셋 라벨 업로드.
    """

    with get_conn() as conn:

        for lab in req.labels:

            upsert_label(
                conn,

                lab.profile_id,

                lab.arxiv_id,

                lab.labeler,

                lab.label,

                source=lab.source,

                tag=lab.tag,

                profile_version=(
                    lab.profile_version
                ),
            )

        total = count_labels(
            conn
        )

    return LabelsUploadResponse(
        saved=len(req.labels),
        total=total,
    )


# ============================================================
# Labels List
# ============================================================

@app.get(
    "/labels",
    dependencies=[Depends(verify_api_key)],
)
def list_labels(
    profile_id: Optional[str] = None,
):
    """
    저장된 라벨 조회.
    """

    with get_conn() as conn:

        return get_labels(
            conn,
            profile_id,
        )


# ============================================================
# Paper Ingest
# ============================================================

@app.post(
    "/papers/ingest",
    response_model=IngestResponse,
    dependencies=[Depends(verify_api_key)],
)
def ingest_papers(
    req: IngestRequest,
):
    """
    논문 수집 결과를 서버 DB에 반영.

    abstract_clean은 서버에서 생성.

    auto_embed=True이면 저장 후 임베딩까지 수행한다.
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

                abstract_clean=clean_abstract(
                    p.abstract_raw
                ),

                categories=p.categories,

                primary_category=(
                    p.primary_category
                ),

                comments=p.comments,

                submitted_date=(
                    p.submitted_date
                ),

                updated_date=(
                    p.updated_date
                ),

                abs_url=p.abs_url,

                pdf_url=p.pdf_url,
            )

            if upsert_paper(
                conn,
                record,
            ):
                changed += 1

    newly_embedded = (
        embed_pending_papers()
        if req.auto_embed
        else 0
    )

    return IngestResponse(
        ingested_or_updated=changed,
        newly_embedded=newly_embedded,
    )


# ============================================================
# Paper Delete
# ============================================================

@app.post(
    "/papers/delete",
    response_model=DeleteResponse,
    dependencies=[Depends(verify_api_key)],
)
def delete_papers(
    req: DeleteRequest,
):
    """
    논문 삭제.

    SQLite + ChromaDB에서 함께 삭제한다.
    """

    if not req.confirm:

        raise HTTPException(
            status_code=400,
            detail=(
                "confirm=true가 필요합니다 "
                "(실수 방지)"
            ),
        )

    if bool(req.arxiv_ids) == bool(
        req.category
    ):

        raise HTTPException(
            status_code=400,
            detail=(
                "arxiv_ids 또는 category 중 "
                "정확히 하나만 지정하세요"
            ),
        )

    with get_conn() as conn:

        if req.category:

            ids = (
                get_ids_by_primary_category(
                    conn,
                    req.category,
                )
            )

        else:

            ids = req.arxiv_ids

        deleted = delete_papers_by_ids(
            conn,
            ids,
        )

        remaining = count_papers(
            conn
        )

    delete_vectors(ids)

    return DeleteResponse(
        deleted=deleted,
        remaining=remaining,
    )


# ============================================================
# Single Paper
# ============================================================

@app.get(
    "/papers/{arxiv_id}",
    response_model=PaperDetail,
    dependencies=[Depends(verify_api_key)],
)
def get_paper(
    arxiv_id: str,
):
    with get_conn() as conn:

        record = get_paper_by_id(
            conn,
            arxiv_id,
        )

    if record is None:

        raise HTTPException(
            status_code=404,
            detail="Paper not found",
        )

    return record


# ============================================================
# Multiple Papers
# ============================================================

@app.get(
    "/papers",
    response_model=List[PaperDetail],
    dependencies=[Depends(verify_api_key)],
)
def list_papers_by_ids(
    ids: str = Query(
        ...,
        description=(
            "쉼표로 구분된 arxiv_id 목록"
        ),
    ),
):
    """
    여러 논문을 한 번에 조회한다.

    예:
        /papers?ids=2401.12345,2402.54321
    """

    id_list = [
        i.strip()
        for i in ids.split(",")
        if i.strip()
    ]

    with get_conn() as conn:

        meta_map = get_papers_by_ids(
            conn,
            id_list,
        )

    return list(
        meta_map.values()
    )