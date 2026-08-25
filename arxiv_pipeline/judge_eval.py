"""
A안 평가 스크립트 — LLM judge를 '독립 라벨러'로 사용해 시스템 랭킹 품질을 정량 평가.

무엇을 재는가
--------------------------------------------------------------------
프로필 → arxiv-api /retrieve(하이브리드 검색) → /rerank(CrossEncoder) = **시스템 랭킹**
       → 각 후보를 judge가 **독립적으로 0/1 라벨**(리스트가 아니라 논문 1편씩)
       → judge 라벨을 정답으로 Recall@k / Precision@k / nDCG@k / MRR (macro 평균)

왜 이 방식인가 (순환 회피)
--------------------------------------------------------------------
- Stage 3(최종 선정)도 Gemini다. judge가 "시스템이 고른 top10"을 채점하면
  Gemini가 Gemini를 채점 → 순환. 그래서 judge는 **선정 결과를 보지 않고**
  후보 각각을 pointwise로 0/1 라벨한다(다른 태스크·다른 프롬프트).
- 가능하면 JUDGE_MODEL을 Stage 3와 다른 모델로 두라(env로 지정 가능).
- --anchor: judge 라벨을 사람 골드셋(/labels)과 대조해 judge 신뢰도(정확도/F1)를
  함께 보고한다. 이게 judge를 쓰는 유일한 정당화 근거(사람 라벨의 스케일 확장).

준비
--------------------------------------------------------------------
    pip install google-genai requests
    export GEMINI_API_KEY="..."          # judge 호출용
    export ARXIV_API_KEY="..."           # arxiv-api 인증(없으면 config.API_KEY)
    export ARXIV_PIPELINE_URL="http://127.0.0.1:8000"   # (선택) arxiv-api 주소

실행
--------------------------------------------------------------------
    python judge_eval.py                         # 전체 12프로필, top30, k=10
    python judge_eval.py --profiles P1,P2 --top 20
    python judge_eval.py --k 10 --out judge_eval_result.json
    python judge_eval.py --no-anchor             # 사람 골드셋 대조 생략
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional

import requests

from config import API_KEY, GEMINI_API_KEY, GEMINI_MODEL
from reranker import mrr, ndcg_at_k, recall_at_k, _macro

ARXIV_URL = os.environ.get("ARXIV_PIPELINE_URL", "http://127.0.0.1:8000").rstrip("/")
JUDGE_MODEL_ENV = os.environ.get("JUDGE_MODEL", "")  # 지정 시 자동선택 대신 사용


# ============================================================
# arxiv-api 호출 (시스템 랭킹)
# ============================================================

def _headers() -> dict:
    return {"X-API-Key": os.environ.get("ARXIV_API_KEY", API_KEY),
            "Content-Type": "application/json"}


def retrieve(profile_text: str, keywords: List[str], category: Optional[str],
             n_keyword: int, m_embedding: int) -> List[dict]:
    r = requests.post(f"{ARXIV_URL}/retrieve", headers=_headers(), timeout=60,
                      json={"profile_text": profile_text, "keywords": keywords,
                            "category": category, "n_keyword": n_keyword,
                            "m_embedding": m_embedding})
    r.raise_for_status()
    return r.json()


def rerank(profile_text: str, candidates: List[dict]) -> List[dict]:
    payload_c = [{"arxiv_id": c.get("arxiv_id", ""), "title": c.get("title", ""),
                  "abstract_clean": c.get("abstract_clean")} for c in candidates]
    r = requests.post(f"{ARXIV_URL}/rerank", headers=_headers(), timeout=120,
                      json={"profile_text": profile_text, "candidates": payload_c})
    r.raise_for_status()
    return r.json().get("results", [])


def human_labels(profile_id: str) -> Dict[str, int]:
    """사람 골드셋 라벨(judge 제외)을 {arxiv_id: 0/1}로. judge 신뢰도 대조용."""
    try:
        r = requests.get(f"{ARXIV_URL}/labels", headers=_headers(),
                         params={"profile_id": profile_id}, timeout=30)
        r.raise_for_status()
        out = {}
        for rec in r.json():
            if (str(rec.get("labeler", "")).lower() == "judge"):
                continue  # 사람 라벨만
            try:
                out[rec["arxiv_id"]] = int(float(rec["label"]))
            except (KeyError, TypeError, ValueError):
                pass
        return out
    except requests.RequestException:
        return {}


# ============================================================
# Judge (독립 라벨러) — pointwise 0/1
# ============================================================

_PREFER = ["gemini-flash-latest", "gemini-2.0-flash", "gemini-2.0-flash-001",
           "gemini-2.5-flash-lite", "gemini-2.5-flash"]

JUDGE_PROMPT = """You are an independent relevance labeler for a research-paper recommender.
Decide whether ONE paper matches a researcher's interest profile.

Return ONLY a single character: "1" if relevant, "0" if not relevant. No other text.

Rules:
- Answer 1 only if the paper's CORE topic matches the profile's core interest.
- Answer 0 if the paper merely mentions the topic in passing, is only tangentially related,
  or violates the exclusion condition below.

Researcher profile:
{profile}

Exclusion (NOT interested in):
{exclusion}

Paper title: {title}
Paper abstract: {abstract}

Answer (0 or 1):"""


def _make_client():
    if not GEMINI_API_KEY:
        sys.exit("GEMINI_API_KEY가 없습니다. export GEMINI_API_KEY=... 후 실행하세요.")
    try:
        from google import genai
    except ImportError:
        sys.exit("google-genai 미설치. `pip install google-genai` 후 실행하세요.")
    return genai.Client(api_key=GEMINI_API_KEY)


def _pick_model(client) -> str:
    if JUDGE_MODEL_ENV:
        return JUDGE_MODEL_ENV
    try:
        avail = [m.name.split("/")[-1] for m in client.models.list()
                 if "generateContent" in (getattr(m, "supported_actions", None) or ["generateContent"])]
        for name in _PREFER:
            if name in avail:
                return name
        for name in avail:
            if "flash" in name and not any(x in name for x in ("vision", "thinking", "preview", "exp", "tts", "image")):
                return name
        if avail:
            return avail[0]
    except Exception as e:
        print(f"모델 목록 조회 실패, 기본값({GEMINI_MODEL}) 사용:", e)
    return GEMINI_MODEL


def judge_label(client, model: str, profile: str, exclusion: str,
                title: str, abstract: str, max_retries: int = 6) -> Optional[int]:
    """judge가 논문 1편을 0/1로 라벨. 파싱 실패 시 None(평가에서 제외)."""
    prompt = JUDGE_PROMPT.format(profile=profile, exclusion=exclusion or "(none)",
                                 title=title, abstract=(abstract or "")[:1500])
    for attempt in range(max_retries):
        try:
            resp = client.models.generate_content(model=model, contents=prompt)
            text = (resp.text or "").strip()
            for ch in text:
                if ch in "01":
                    return int(ch)
            return None
        except Exception as e:
            if ("429" in str(e)) or ("RESOURCE_EXHAUSTED" in str(e)):
                print(f"    429 → 62초 대기 후 재시도 ({attempt + 1}/{max_retries})")
                time.sleep(62)
                continue
            print(f"    judge 오류(건너뜀): {e}")
            return None
    return None


# ============================================================
# 지표
# ============================================================

def precision_at_k(labels_ranked: List[int], k: int) -> Optional[float]:
    kk = min(k, len(labels_ranked))
    return sum(labels_ranked[:kk]) / kk if kk else None


def _f1(tp: int, fp: int, fn: int) -> float:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    return 2 * p * r / (p + r) if (p + r) else 0.0


# ============================================================
# 메인
# ============================================================

def main():
    ap = argparse.ArgumentParser(description="LLM judge 독립 라벨 기반 시스템 랭킹 평가(A안)")
    ap.add_argument("--profiles-path", default="profiles.json")
    ap.add_argument("--profiles", default="", help="쉼표로 부분집합 지정 (예: P1,P2). 비우면 전체")
    ap.add_argument("--k", type=int, default=10, help="Recall/Precision/nDCG@k")
    ap.add_argument("--top", type=int, default=30, help="프로필당 judge 라벨할 상위 후보 수")
    ap.add_argument("--n-keyword", type=int, default=30)
    ap.add_argument("--m-embedding", type=int, default=70)
    ap.add_argument("--spacing", type=float, default=0.5, help="judge 호출 간 대기(초)")
    ap.add_argument("--cache", default="judge_eval_cache.json", help="judge 라벨 캐시 파일")
    ap.add_argument("--out", default="judge_eval_result.json")
    ap.add_argument("--no-anchor", action="store_true", help="사람 골드셋 대조(judge 신뢰도) 생략")
    args = ap.parse_args()

    with open(args.profiles_path, encoding="utf-8") as f:
        profiles = json.load(f)
    if args.profiles:
        want = {p.strip() for p in args.profiles.split(",")}
        profiles = [p for p in profiles if p["profile_id"] in want]

    # judge 라벨 캐시 (재실행 비용 절감)
    cache: Dict[str, int] = {}
    if os.path.exists(args.cache):
        try:
            cache = json.load(open(args.cache, encoding="utf-8"))
        except (OSError, ValueError):
            cache = {}

    client = _make_client()
    model = _pick_model(client)
    print(f"[judge] 모델 = {model}  (Stage 3 선정 모델과 다르게 두는 것을 권장)")
    print(f"[system] arxiv-api = {ARXIV_URL}\n")

    per_profile = {}
    agg = {"recall": [], "precision": [], "ndcg": [], "mrr": []}
    anchor_tp = anchor_fp = anchor_fn = anchor_tn = 0

    for p in profiles:
        pid = p["profile_id"]
        profile_text = p.get("profile_text") or p.get("profile_text_ko") or ""
        keywords = p.get("keywords") or []
        category = p.get("category")
        exclusion = p.get("exclusion") or ""

        # 1) 시스템 랭킹 = retrieve → rerank
        try:
            cands = retrieve(profile_text, keywords, category, args.n_keyword, args.m_embedding)
            ranked = rerank(profile_text, cands)[:args.top]
        except requests.RequestException as e:
            print(f"[{pid}] arxiv-api 호출 실패 → 건너뜀: {e}")
            continue
        if not ranked:
            print(f"[{pid}] 후보 0편 → 건너뜀")
            continue

        # 2) judge 독립 라벨 (순서대로)
        labels: List[int] = []
        h_labels = {} if args.no_anchor else human_labels(pid)
        print(f"[{pid}] 후보 {len(ranked)}편 라벨링…")
        for item in ranked:
            aid = item.get("arxiv_id", "")
            ckey = f"{pid}|{aid}"
            if ckey in cache:
                lab = cache[ckey]
            else:
                lab = judge_label(client, model, profile_text, exclusion,
                                  item.get("title", ""), item.get("abstract_clean", ""))
                if lab is not None:
                    cache[ckey] = lab
                    json.dump(cache, open(args.cache, "w", encoding="utf-8"), ensure_ascii=False)
                if args.spacing:
                    time.sleep(args.spacing)
            labels.append(lab if lab is not None else 0)

            # judge vs 사람 골드셋 대조(겹치는 것만)
            if aid in h_labels and lab is not None:
                if lab == 1 and h_labels[aid] == 1:
                    anchor_tp += 1
                elif lab == 1 and h_labels[aid] == 0:
                    anchor_fp += 1
                elif lab == 0 and h_labels[aid] == 1:
                    anchor_fn += 1
                else:
                    anchor_tn += 1

        # 3) 지표 (judge 라벨을 정답으로)
        k = args.k
        r_at = recall_at_k(labels, k)
        p_at = precision_at_k(labels, k)
        n_at = ndcg_at_k(labels, k)
        m_at = mrr(labels)
        per_profile[pid] = {
            "n": len(labels), "pos(judge)": sum(labels),
            f"recall@{k}": r_at, f"precision@{k}": p_at,
            f"ndcg@{k}": n_at, "mrr": m_at,
        }
        agg["recall"].append(r_at)
        agg["precision"].append(p_at)
        agg["ndcg"].append(n_at)
        agg["mrr"].append(m_at)
        print(f"    → recall@{k}={_fmt(r_at)} precision@{k}={_fmt(p_at)} "
              f"ndcg@{k}={_fmt(n_at)} mrr={_fmt(m_at)}  (judge pos={sum(labels)}/{len(labels)})")

    # ── 요약 ──────────────────────────────────────────
    k = args.k
    summary = {
        f"recall@{k}": _macro(agg["recall"]),
        f"precision@{k}": _macro(agg["precision"]),
        f"ndcg@{k}": _macro(agg["ndcg"]),
        "mrr": _macro(agg["mrr"]),
        "judge_model": model, "profiles": len(per_profile), "top": args.top,
    }
    print("\n==================== 시스템 랭킹 (judge 정답 기준, macro) ====================")
    print(f"  Recall@{k}    = {summary[f'recall@{k}']:.3f}")
    print(f"  Precision@{k} = {summary[f'precision@{k}']:.3f}")
    print(f"  nDCG@{k}      = {summary[f'ndcg@{k}']:.3f}")
    print(f"  MRR          = {summary['mrr']:.3f}")

    anchor = None
    if not args.no_anchor and (anchor_tp + anchor_fp + anchor_fn + anchor_tn) > 0:
        total = anchor_tp + anchor_fp + anchor_fn + anchor_tn
        acc = (anchor_tp + anchor_tn) / total
        f1 = _f1(anchor_tp, anchor_fp, anchor_fn)
        anchor = {"overlap": total, "accuracy": acc, "f1": f1,
                  "tp": anchor_tp, "fp": anchor_fp, "fn": anchor_fn, "tn": anchor_tn}
        print("\n---- judge 신뢰도 (사람 골드셋과 대조, 겹치는 라벨만) ----")
        print(f"  겹침 {total}건 · 정확도 {acc:.3f} · F1 {f1:.3f}  "
              f"(TP{anchor_tp}/FP{anchor_fp}/FN{anchor_fn}/TN{anchor_tn})")
        print("  ※ judge가 사람과 충분히 일치해야 위 지표가 신뢰 가능")

    result = {"summary": summary, "per_profile": per_profile, "judge_vs_human": anchor}
    json.dump(result, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n결과 저장: {args.out}  (judge 라벨 캐시: {args.cache})")


def _fmt(x) -> str:
    return "  na" if x is None else f"{x:.3f}"


if __name__ == "__main__":
    main()
