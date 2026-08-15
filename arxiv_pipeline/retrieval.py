"""
Stage 1 하이브리드 검색.

구성:
    keyword top N
        +
    embedding top M
        ↓
    중복 제거
        ↓
    후보 union

점수 합산/정규화는 하지 않는다.
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

    # 키워드 검색
    if keywords:
        with get_conn() as conn:
            kw_hits = search_keyword(
                conn,
                keywords,
                top_k=n_keyword,
                category=category,
            )
    else:
        kw_hits = []

    kw_ids = [aid for aid, _ in kw_hits]
    kw_set = set(kw_ids)

    print(
        f"[RETRIEVE] keyword: requested={n_keyword}, "
        f"returned={len(kw_ids)}, category={category}"
    )

    # 임베딩 검색
    emb_hits = search_similar(
        profile_text,
        top_k=m_embedding,
        category=category,
    )

    emb_ids = [aid for aid, _, _ in emb_hits]

    print(
        f"[RETRIEVE] embedding: requested={m_embedding}, "
        f"returned={len(emb_ids)}, category={category}"
    )

    # source
    source = {}

    for aid in kw_ids:
        source[aid] = "keyword"

    for aid in emb_ids:
        if aid in kw_set:
            source[aid] = "both"
        else:
            source[aid] = "embedding"

    # keyword 먼저 + embedding 중복 제거
    ordered = kw_ids + [
        aid for aid in emb_ids
        if aid not in kw_set
    ]

    result = [
        {
            "arxiv_id": aid,
            "source": source[aid],
        }
        for aid in ordered
    ]

    print(
        f"[RETRIEVE] final={len(result)}, "
        f"keyword={len(kw_ids)}, "
        f"embedding={len(emb_ids)}, "
        f"both={sum(1 for aid in kw_ids if aid in set(emb_ids))}"
    )

    return result

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
    골드셋 라벨링 후보 풀.

    구성:
        hybrid candidates
        +
        random candidates

    총 개수는 최대 total.
    """

    hybrid = hybrid_retrieve(
        profile_text=profile_text,
        keywords=keywords,
        category=category,
        n_keyword=n_keyword,
        m_embedding=m_embedding,
    )

    n_hybrid = max(0, total - n_random)

    hybrid = hybrid[:n_hybrid]

    hybrid_ids = [
        candidate["arxiv_id"]
        for candidate in hybrid
    ]

    need = total - len(hybrid)

    if need > 0:
        with get_conn() as conn:
            rnd_ids = sample_random(
                conn,
                category,
                need,
                exclude_ids=hybrid_ids,
            )
    else:
        rnd_ids = []

    result = hybrid + [
        {
            "arxiv_id": aid,
            "source": "random",
        }
        for aid in rnd_ids
    ]

    print(
        f"[LABELING_POOL] "
        f"hybrid={len(hybrid)} "
        f"random={len(rnd_ids)} "
        f"total={len(result)}"
    )

    return result


if __name__ == "__main__":
    from collections import Counter

    demo = (
        "I am interested in language-conditioned robot manipulation, "
        "especially vision-language-action models that interpret "
        "ambiguous instructions."
    )

    kws = [
        "language-conditioned manipulation",
        "vision-language-action model",
        "instruction following",
    ]

    pool = build_labeling_pool(
        demo,
        kws,
        category="cs.RO",
        total=60,
        n_random=10,
    )

    print(
        f"후보 {len(pool)}편, "
        f"출처 분포:",
        Counter(c["source"] for c in pool),
    )

    for c in pool[:5]:
        print(" ", c)