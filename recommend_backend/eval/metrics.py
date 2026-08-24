"""
평가 지표 (Recall@K, Precision@K, NDCG@K).

이진 relevance(정답 목록에 있으면 1, 없으면 0)를 가정한 표준 정의를 그대로
구현한다. 외부 라이브러리(sklearn 등) 의존성 없이 순수 함수로 작성했다.
"""

import math
from typing import Sequence


def recall_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    """상위 k개 검색 결과 중 정답을 얼마나 회수했는지 (정답 집합 기준 비율)."""
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    hits = sum(1 for aid in retrieved[:k] if aid in relevant_set)
    return hits / len(relevant_set)


def precision_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    """상위 k개 중 정답 비율. k보다 결과가 적으면 분모는 그대로 k를 쓴다(표준 정의)."""
    if k <= 0:
        return 0.0
    relevant_set = set(relevant)
    hits = sum(1 for aid in retrieved[:k] if aid in relevant_set)
    return hits / k


def ndcg_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    """
    이진 relevance 기준 NDCG@k.
    DCG@k = Σ rel_i / log2(rank_i + 1)  (rank는 1부터 시작)
    IDCG@k = 정답이 상위 k칸을 이상적으로 모두 채웠을 때의 DCG@k
    """
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0

    dcg = 0.0
    for rank, aid in enumerate(retrieved[:k], start=1):
        if aid in relevant_set:
            dcg += 1.0 / math.log2(rank + 1)

    ideal_hits = min(len(relevant_set), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    if idcg == 0:
        return 0.0
    return dcg / idcg
