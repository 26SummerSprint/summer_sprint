"""
Stage 1 하이브리드 검색 (쿼터 기반 union) + 골드셋 후보 풀 구성.

★확정 설계: 임베딩 점수와 BM25 점수를 합산/정규화하지 않는다.
키워드 상위 N편 + 임베딩 상위 M편을 정해진 편수만큼 합쳐(중복 제거) 후보를 만든다.
최종 순위는 Stage2 재랭커가 다시 매기므로, 여기서는 recall 확보만 담당한다.

- hybrid_retrieve(): 데일리 추천 Stage1 후보 (키워드 N + 임베딩 M)
- build_labeling_pool(): 골드셋 라벨링 후보 (하이브리드 + 랜덤 샘플, pool bias 완화)

이 모듈은 db(키워드/랜덤)와 embedder(임베딩)를 함께 쓰므로 서버(EC2)에서만 import 가능.
"""

from typing import List, Optional

from db import get_conn, sample_random, search_keyword
from embedder import search_similar


def hybrid_retrieve(
    profile_text: str,
    keywords: List[str],
    category: Optional[str] = None,
    n_keyword: int = 30,
    m_embedding: int = 70,
) -> List[dict]:
    """
    쿼터 union: 키워드 검색 상위 n_keyword편(보장) + 임베딩 검색 상위 m_embedding편, 중복 제거.
    반환: [{"arxiv_id": ..., "source": "keyword"|"embedding"|"both"}] (키워드 → 임베딩 순서)
    """
    # 키워드 축 (BM25)
    if keywords:
        with get_conn() as conn:
            kw_hits = search_keyword(conn, keywords, top_k=n_keyword, category=category)
    else:
        kw_hits = []
    kw_ids = [aid for aid, _ in kw_hits]
    kw_set = set(kw_ids)

    # 임베딩 축 (cosine)
    emb_hits = search_similar(profile_text, top_k=m_embedding, category=category)
    emb_ids = [aid for aid, _, _ in emb_hits]

    # 출처 태그
    source = {aid: "keyword" for aid in kw_ids}
    for aid in emb_ids:
        source[aid] = "both" if aid in kw_set else "embedding"

    # 키워드 보장분 먼저, 그다음 임베딩에서 중복 아닌 것
    ordered = kw_ids + [aid for aid in emb_ids if aid not in kw_set]
    return [{"arxiv_id": aid, "source": source[aid]} for aid in ordered]


def build_labeling_pool(
    profile_text: str,
    keywords: List[str],
    category: Optional[str] = None,
    total: int = 60,
    n_random: int = 10,
    n_keyword: int = 30,
    m_embedding: int = 70,
) -> List[dict]:
    """
    골드셋 라벨링 후보 = 하이브리드(키워드+임베딩) 상위 + 랜덤 샘플, 합쳐서 total편.
    랜덤 샘플은 pool bias를 완화하고 0(무관) 라벨을 확보하기 위한 것.

    구성: 하이브리드 상위 (total - n_random)편 + 랜덤 n_random편.
    (프로필당 라벨링 워크로드를 total로 고정 — 회의 확정 60편)
    반환: [{"arxiv_id": ..., "source": ...}]
    """
    hybrid = hybrid_retrieve(profile_text, keywords, category, n_keyword, m_embedding)
    n_hybrid = max(0, total - n_random)
    hybrid = hybrid[:n_hybrid]
    hybrid_ids = [c["arxiv_id"] for c in hybrid]

    need = total - len(hybrid)  # 하이브리드가 부족하면 랜덤으로 total까지 채움
    with get_conn() as conn:
        rnd_ids = sample_random(conn, category, need, exclude_ids=hybrid_ids) if need > 0 else []

    return hybrid + [{"arxiv_id": aid, "source": "random"} for aid in rnd_ids]


if __name__ == "__main__":
    # 간단 동작 확인
    demo = (
        "I am interested in language-conditioned robot manipulation, especially "
        "vision-language-action models that interpret ambiguous instructions."
    )
    kws = ["language-conditioned manipulation", "vision-language-action model", "instruction following"]
    pool = build_labeling_pool(demo, kws, category="cs.RO", total=60, n_random=10)
    from collections import Counter

    print(f"후보 {len(pool)}편, 출처 분포:", Counter(c["source"] for c in pool))
    for c in pool[:5]:
        print(" ", c)
