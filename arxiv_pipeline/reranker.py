"""
Stage 2 재랭커 (cross-encoder) — Stage1 후보의 순위를 다시 매긴다.

설계
- 입력: (프로필 텍스트, 논문 제목+초록) 쌍 → 관련도 점수 1개
- 모델: sentence-transformers CrossEncoder 파인튜닝
- 학습 라벨: 골드셋 0/1 (label=1 positive, label=0 및 hard negative)
- 평가: 프로필별 time-split(과거 학습 / 최근 평가), Recall@k / MRR
- 베이스라인: 임베딩(cosine) 순위와 비교해 "재랭커가 이기는지" 확인

torch / sentence-transformers가 필요하므로 GPU(Colab)에서 실행 권장.
이 모듈은 db/chroma에 의존하지 않는다 — 라벨·메타·프로필 텍스트를 인자로 받는다.
서버에서 서빙하려면 load_reranker()로 저장 모델을 불러 rerank()만 쓰면 된다.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import List, Optional

# 사전학습 랭킹 cross-encoder에서 출발해 소량 골드셋으로 파인튜닝(데이터가 적어 from-scratch 지양)
DEFAULT_BASE_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
ABSTRACT_CHARS = 1200  # 초록을 잘라 입력 길이 관리


def doc_text(meta: dict) -> str:
    """논문 메타(title, abstract_clean) → 재랭커 입력 문서 텍스트."""
    title = (meta.get("title") or "").strip()
    abstract = (meta.get("abstract_clean") or meta.get("abstract") or "").strip()[:ABSTRACT_CHARS]
    return f"{title}. {abstract}".strip(". ").strip()


def build_examples(labels, meta_by_id, profile_text_by_id, prefer_human=True) -> List[dict]:
    """골드셋 라벨 → 학습/평가 예시 리스트.

    labels: [{profile_id, arxiv_id, label, source, tag, labeler, ...}]
    meta_by_id: {arxiv_id: {title, abstract_clean, submitted_date, ...}}
    profile_text_by_id: {profile_id: 영어 프로필 텍스트}
    같은 (profile, arxiv)에 여러 labeler가 있으면 사람 라벨 우선(prefer_human).
    반환 항목: {profile_id, arxiv_id, query, doc, label, submitted_date, source, tag}
    """
    chosen = {}
    for lab in labels:
        key = (lab["profile_id"], lab["arxiv_id"])
        prev = chosen.get(key)
        if prev is None:
            chosen[key] = lab
        elif prefer_human:
            is_human = (lab.get("labeler") or "").lower() != "judge"
            prev_judge = (prev.get("labeler") or "").lower() == "judge"
            if is_human and prev_judge:
                chosen[key] = lab

    out = []
    for (pid, aid), lab in chosen.items():
        query = profile_text_by_id.get(pid)
        meta = meta_by_id.get(aid)
        if not query or not meta:
            continue
        d = doc_text(meta)
        if not d:
            continue
        try:
            label = int(lab["label"])
        except (TypeError, ValueError):
            continue  # 빈 칸 / uncertain 등은 제외
        out.append({
            "profile_id": pid, "arxiv_id": aid, "query": query, "doc": d,
            "label": label, "submitted_date": meta.get("submitted_date", ""),
            "source": lab.get("source"), "tag": lab.get("tag"),
        })
    return out


def time_split(examples: List[dict], test_frac: float = 0.3):
    """프로필별로 submitted_date 오름차순 정렬 후 뒤쪽(최근) test_frac을 test로 뗀다.
    과거로 학습하고 최근으로 평가해 시간 누수를 막는다. 반환: (train, test)."""
    by_p = defaultdict(list)
    for e in examples:
        by_p[e["profile_id"]].append(e)
    train, test = [], []
    for _pid, lst in by_p.items():
        lst = sorted(lst, key=lambda e: e["submitted_date"] or "")
        n_test = max(1, round(len(lst) * test_frac)) if len(lst) >= 4 else 0
        cut = len(lst) - n_test
        train += lst[:cut]
        test += lst[cut:]
    return train, test


def train_reranker(train_examples: List[dict], base_model: str = DEFAULT_BASE_MODEL,
                   epochs: int = 3, batch_size: int = 16, lr: float = 2e-5, seed: int = 42):
    """CrossEncoder 파인튜닝. 반환: 학습된 CrossEncoder."""
    import torch
    from torch.utils.data import DataLoader
    from sentence_transformers import CrossEncoder, InputExample

    random.seed(seed)
    torch.manual_seed(seed)

    samples = [InputExample(texts=[e["query"], e["doc"]], label=float(e["label"]))
               for e in train_examples]
    loader = DataLoader(samples, shuffle=True, batch_size=batch_size)
    model = CrossEncoder(base_model, num_labels=1, max_length=512)
    warmup = max(1, int(0.1 * len(loader) * epochs))
    model.fit(train_dataloader=loader, epochs=epochs, warmup_steps=warmup,
              optimizer_params={"lr": lr}, show_progress_bar=True)
    return model


def _predict(model, query: str, docs: List[str], batch_size: int = 32):
    if not docs:
        return []
    return list(model.predict([[query, d] for d in docs],
                              batch_size=batch_size, show_progress_bar=False))


def rerank(model, query: str, candidates: List[dict], batch_size: int = 32) -> List[dict]:
    """candidates: [{arxiv_id, doc, ...}] → 각 항목에 rerank_score를 넣고 내림차순 정렬해 반환."""
    scores = _predict(model, query, [c["doc"] for c in candidates], batch_size)
    for c, s in zip(candidates, scores):
        c["rerank_score"] = float(s)
    return sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)


# ── 지표 ──────────────────────────────────────────────
def recall_at_k(labels_ranked: List[int], k: int) -> Optional[float]:
    rel = sum(labels_ranked)
    if rel == 0:
        return None  # positive가 없는 프로필은 평균에서 제외
    return sum(labels_ranked[:k]) / rel


def mrr(labels_ranked: List[int]) -> float:
    for i, label in enumerate(labels_ranked, 1):
        if label == 1:
            return 1.0 / i
    return 0.0


def _dcg(labels_ranked: List[int], k: int) -> float:
    import math
    return sum(l / math.log2(i + 2) for i, l in enumerate(labels_ranked[:k]))


def ndcg_at_k(labels_ranked: List[int], k: int) -> Optional[float]:
    """이진 관련도 기준 nDCG@k. positive가 없으면 None(평균에서 제외).
    positive가 아주 많은 프로필에서도 '얼마나 위로 몰았나'를 구분해준다."""
    idcg = _dcg(sorted(labels_ranked, reverse=True), k)
    if idcg == 0:
        return None
    return _dcg(labels_ranked, k) / idcg


def _macro(values) -> float:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else 0.0


def _embed_order(embed_model, query: str, docs: List[str]) -> List[int]:
    """임베딩(cosine) 기준 내림차순 인덱스 — Stage1 임베딩 순위와 동일한 베이스라인."""
    import numpy as np
    q = embed_model.encode([query], normalize_embeddings=True)[0]
    d = embed_model.encode(docs, normalize_embeddings=True)
    sims = d @ q
    return list(np.argsort(-sims))


def evaluate(model, test_examples: List[dict], embed_model=None, k: int = 10) -> dict:
    """프로필별로 재랭커(그리고 embed_model을 주면 임베딩 베이스라인) Recall@k / MRR을 재고
    macro 평균한다. 반환: {reranker, baseline|None, per_profile, k}."""
    by_p = defaultdict(list)
    for e in test_examples:
        by_p[e["profile_id"]].append(e)

    rr_recall, rr_mrr, bl_recall, bl_mrr, per = [], [], [], [], {}
    for pid, lst in by_p.items():
        labels = [e["label"] for e in lst]
        docs = [e["doc"] for e in lst]
        query = lst[0]["query"]

        scores = _predict(model, query, docs)
        order = sorted(range(len(lst)), key=lambda i: scores[i], reverse=True)
        rr_sorted = [labels[i] for i in order]
        r_recall, r_mrr = recall_at_k(rr_sorted, k), mrr(rr_sorted)
        rr_recall.append(r_recall)
        rr_mrr.append(r_mrr)
        entry = {"n": len(lst), "pos": sum(labels),
                 f"reranker_recall@{k}": r_recall, "reranker_mrr": r_mrr}

        if embed_model is not None:
            b_order = _embed_order(embed_model, query, docs)
            bl_sorted = [labels[i] for i in b_order]
            b_recall, b_mrr = recall_at_k(bl_sorted, k), mrr(bl_sorted)
            bl_recall.append(b_recall)
            bl_mrr.append(b_mrr)
            entry[f"baseline_recall@{k}"] = b_recall
            entry["baseline_mrr"] = b_mrr
        per[pid] = entry

    result = {
        "reranker": {f"recall@{k}": _macro(rr_recall), "mrr": _macro(rr_mrr)},
        "baseline": None, "per_profile": per, "k": k,
    }
    if embed_model is not None:
        result["baseline"] = {f"recall@{k}": _macro(bl_recall), "mrr": _macro(bl_mrr)}
    return result


def evaluate_lopo(
    examples: List[dict],
    base_model: str = DEFAULT_BASE_MODEL,
    embed_model=None,
    k: int = 10,
    epochs: int = 3,
    batch_size: int = 16,
) -> dict:
    """Leave-One-Profile-Out 평가 (소규모 골드셋용 권장).

    각 프로필을 하나씩 빼고 나머지 프로필로 재랭커를 학습한 뒤, 뺀 프로필의
    '전체 후보 풀'을 재랭킹해 Recall@k / MRR / nDCG@k를 잰다.
    - 학습에 안 쓴 프로필로 평가 → 누수 없음
    - 풀 크기(≈30) > k 라서 Recall@k가 degenerate(항상 1.0) 해지지 않음
    - embed_model을 주면 임베딩(cosine) 베이스라인과 비교
    반환: {reranker, baseline|None, per_profile, k}. (프로필 수만큼 학습하므로 느림)
    """
    by_p = defaultdict(list)
    for e in examples:
        by_p[e["profile_id"]].append(e)

    agg = {"reranker": defaultdict(list), "baseline": defaultdict(list)}
    per = {}
    for held in by_p:
        train_ex = [e for pid, lst in by_p.items() if pid != held for e in lst]
        test_ex = by_p[held]
        if not train_ex or not test_ex:
            continue
        model = train_reranker(train_ex, base_model, epochs, batch_size)
        labels = [e["label"] for e in test_ex]
        docs = [e["doc"] for e in test_ex]
        query = test_ex[0]["query"]

        scores = _predict(model, query, docs)
        order = sorted(range(len(test_ex)), key=lambda i: scores[i], reverse=True)
        rr = [labels[i] for i in order]
        entry = {"n": len(test_ex), "pos": sum(labels),
                 f"rr_recall@{k}": recall_at_k(rr, k), "rr_mrr": mrr(rr), f"rr_ndcg@{k}": ndcg_at_k(rr, k)}
        agg["reranker"][f"recall@{k}"].append(entry[f"rr_recall@{k}"])
        agg["reranker"]["mrr"].append(entry["rr_mrr"])
        agg["reranker"][f"ndcg@{k}"].append(entry[f"rr_ndcg@{k}"])

        if embed_model is not None:
            b_order = _embed_order(embed_model, query, docs)
            bl = [labels[i] for i in b_order]
            entry[f"bl_recall@{k}"] = recall_at_k(bl, k)
            entry["bl_mrr"] = mrr(bl)
            entry[f"bl_ndcg@{k}"] = ndcg_at_k(bl, k)
            agg["baseline"][f"recall@{k}"].append(entry[f"bl_recall@{k}"])
            agg["baseline"]["mrr"].append(entry["bl_mrr"])
            agg["baseline"][f"ndcg@{k}"].append(entry[f"bl_ndcg@{k}"])
        per[held] = entry

    def _macro_dict(d):
        return {m: _macro(v) for m, v in d.items()}

    return {
        "reranker": _macro_dict(agg["reranker"]),
        "baseline": _macro_dict(agg["baseline"]) if embed_model is not None else None,
        "per_profile": per, "k": k,
    }


def save_reranker(model, path: str) -> None:
    model.save(path)


def load_reranker(path: str):
    from sentence_transformers import CrossEncoder
    return CrossEncoder(path)
