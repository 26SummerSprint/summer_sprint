"""
임베딩 계산 + 벡터 저장 모듈.

- 모델: sentence-transformers (로컬 실행, 무료, 논문당 1회만 계산해서 캐싱)
- 저장소: Chroma (팀 공유 서버 또는 로컬 파일, 메타데이터 필터링이 내장되어 있어
  "카테고리 필터 + 유사도 검색" 같은 하이브리드 쿼리를 바로 지원 -> 벤치마크의
  '키워드+임베딩 검색 상위 결과 합치기' 단계에서 바로 활용 가능)

pip install sentence-transformers chromadb
"""

from typing import List, Optional

import chromadb
from sentence_transformers import SentenceTransformer

from config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_MODEL_NAME,
)
from db import get_conn, get_papers_needing_embedding, mark_embedded

_model: Optional[SentenceTransformer] = None
_client = None
_collection = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model


def get_collection():
    """
    Chroma는 이 프로세스(수집 파이프라인 또는 FastAPI 서버) 안에서 로컬 파일로 실행된다.
    팀원과의 공유는 이 프로세스 위에 얹힌 FastAPI가 담당하므로, Chroma 자체는
    항상 PersistentClient로 충분하다.
    """
    global _client, _collection
    if _collection is not None:
        return _collection

    _client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    _collection = _client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    return _collection


def embed_texts(texts: List[str]) -> List[List[float]]:
    model = get_model()
    embeddings = model.encode(
        texts,
        batch_size=EMBEDDING_BATCH_SIZE,
        normalize_embeddings=True,  # 코사인 유사도 사용 전제
        show_progress_bar=False,
    )
    return embeddings.tolist()


def embed_pending_papers(batch_size: int = EMBEDDING_BATCH_SIZE) -> int:
    """
    아직 벡터화되지 않은(embedded=0) 논문만 골라 임베딩 계산 후 Chroma에 저장.
    이미 벡터가 있는 논문은 다시 계산하지 않는다 (논문당 1회 계산 원칙).
    반환값: 새로 임베딩된 논문 수.
    """
    collection = get_collection()
    total = 0

    with get_conn() as conn:
        rows = get_papers_needing_embedding(conn)
        if not rows:
            return 0

        for i in range(0, len(rows), batch_size):
            chunk = rows[i : i + batch_size]
            ids = [r[0] for r in chunk]
            texts = [r[1] for r in chunk]  # abstract_clean
            titles = [r[2] for r in chunk]
            categories = [r[3] for r in chunk]
            submitted = [r[4] for r in chunk]

            vectors = embed_texts(texts)

            collection.upsert(
                ids=ids,
                embeddings=vectors,
                documents=texts,
                metadatas=[
                    {"title": t, "primary_category": c, "submitted_date": s}
                    for t, c, s in zip(titles, categories, submitted)
                ],
            )

            mark_embedded(conn, ids)
            total += len(ids)

    return total


def delete_vectors(arxiv_ids: List[str]) -> None:
    """모든 Chroma 컬렉션에서 해당 id의 벡터 삭제 (SQLite 삭제와 함께 사용).
    모델별로 컬렉션이 분리되어 있을 수 있으므로 전체 컬렉션을 순회한다."""
    ids = list(arxiv_ids)
    if not ids:
        return
    get_collection()  # _client 초기화 보장
    for c in _client.list_collections():
        col = _client.get_collection(c.name)
        for i in range(0, len(ids), 500):
            col.delete(ids=ids[i : i + 500])


def search_similar(
    query_text: str,
    top_k: int = 20,
    category: Optional[str] = None,
):
    """
    벤치마크 후보 풀 구성용: 관심사 프로필 텍스트로 유사 논문 top_k 검색.
    category를 지정하면 해당 카테고리로 필터링.
    """
    collection = get_collection()
    query_vec = embed_texts([query_text])[0]

    where = {"primary_category": category} if category else None
    results = collection.query(
        query_embeddings=[query_vec],
        n_results=top_k,
        where=where,
    )
    return list(
        zip(
            results["ids"][0],
            results["distances"][0],
            results["metadatas"][0],
        )
    )


if __name__ == "__main__":
    n = embed_pending_papers()
    print(f"새로 임베딩된 논문 수: {n}")
