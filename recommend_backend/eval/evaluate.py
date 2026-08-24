"""
프로젝트 최종 평가 스크립트.

평가 구조
------------------------------------------------------------

profiles.json
    P1 ~ P12 프로필
        |
        v
Stage 1
    Keyword Extraction
        |
        v
    Hybrid Retrieval
        |
        +--> Recall@50
        |
        +--> Recall@100
        |
        v
Stage 2
    CrossEncoder Reranker
        |
        +--> Rerank Recall@25
        |
        v
Stage 3
    Gemini Final Selection
        |
        +--> Precision@10
        +--> NDCG@10
        +--> Recall@10


Ground Truth
------------------------------------------------------------

eval/gold_sets/P1.xlsx ~ P12.xlsx

각 XLSX의:

    label == 1

인 arxiv_id를 해당 프로필의 정답 논문으로 사용.


전체 평가
------------------------------------------------------------

python -m recommend_backend.eval.evaluate \
    --gold-dir recommend_backend/eval/gold_sets \
    --profiles recommend_backend/eval/profiles.json \
    --output recommend_backend/eval/results.json \
    --concurrency 1


특정 프로필만 재평가
------------------------------------------------------------

python -m recommend_backend.eval.evaluate \
    --gold-dir recommend_backend/eval/gold_sets \
    --profiles recommend_backend/eval/profiles.json \
    --output recommend_backend/eval/results.json \
    --concurrency 1 \
    --profiles-only P3 P11


기존 결과와 병합
------------------------------------------------------------

python -m recommend_backend.eval.evaluate \
    --gold-dir recommend_backend/eval/gold_sets \
    --profiles recommend_backend/eval/profiles.json \
    --output recommend_backend/eval/results.json \
    --concurrency 1 \
    --profiles-only P3 P11 \
    --merge-existing


중요
------------------------------------------------------------

--profiles-only를 사용하면 지정한 프로필만 다시 실행한다.

--merge-existing를 함께 사용하면:

    기존 results.json
        +
    이번에 다시 평가한 프로필

을 합쳐서 P1~P12 전체 결과를 다시 계산한다.

따라서 P3/P11만 다시 돌려도 최종 summary는
P1~P12 전체 기준으로 다시 계산된다.
"""

import argparse
import asyncio
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import pandas as pd

from ..clients.arxiv_pipeline_client import (
    ArxivPipelineClient,
    ArxivPipelineClientError,
)

from ..clients.rerank_pipeline_client import (
    RerankPipelineClient,
    RerankPipelineClientError,
)

from ..config import (
    M_EMBEDDING,
    N_KEYWORD,
    RERANK_COMPRESS_COUNT,
)

from ..services.keyword_extraction_service import (
    KeywordExtractionService,
)

from ..services.final_selection_service import (
    FinalSelectionService,
)

from .metrics import (
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


# ============================================================
# 평가 설정
# ============================================================

RETRIEVAL_K = 50
RETRIEVAL_K_2 = 100

RERANK_K = RERANK_COMPRESS_COUNT

FINAL_K = 10

PROFILE_COUNT = 12


# ============================================================
# 결과 스키마
# ============================================================

@dataclass
class CaseResult:

    profile_id: str

    category: Optional[str]

    # --------------------------------------------------------
    # Ground Truth
    # --------------------------------------------------------

    n_relevant: int

    # --------------------------------------------------------
    # Stage 1 - Retrieval
    # --------------------------------------------------------

    retrieved_count: int

    retrieval_recall_at_50: float

    retrieval_recall_at_100: float

    # --------------------------------------------------------
    # Stage 2 - Reranker
    # --------------------------------------------------------

    rerank_count: int

    rerank_recall_at_25: float

    # --------------------------------------------------------
    # Stage 3 - Gemini
    # --------------------------------------------------------

    final_count: int

    final_precision_at_10: float

    final_ndcg_at_10: float

    final_recall_at_10: float

    # --------------------------------------------------------
    # Error
    # --------------------------------------------------------

    error: Optional[str] = None


# ============================================================
# profiles.json 로드
# ============================================================

def load_profiles(
    profile_path: str,
) -> Dict[str, Dict[str, Any]]:

    path = Path(profile_path)

    if not path.exists():
        raise FileNotFoundError(
            f"profiles.json을 찾을 수 없습니다: {path}"
        )

    try:
        data = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

    except json.JSONDecodeError as e:

        raise ValueError(
            f"profiles.json JSON 파싱 실패: {e}"
        ) from e

    if not isinstance(data, list):

        raise ValueError(
            "profiles.json은 "
            "[{...}, {...}] 형태의 JSON 배열이어야 합니다."
        )

    profiles: Dict[
        str,
        Dict[str, Any],
    ] = {}

    for item in data:

        if not isinstance(item, dict):
            continue

        profile_id = item.get(
            "profile_id"
        )

        if not profile_id:
            continue

        profile_text = item.get(
            "profile_text",
            "",
        )

        category = item.get(
            "category"
        )

        profiles[profile_id] = {
            "profile": profile_text,
            "category": category,
        }

    # --------------------------------------------------------
    # P1 ~ P12 확인
    # --------------------------------------------------------

    for i in range(
        1,
        PROFILE_COUNT + 1,
    ):

        profile_id = f"P{i}"

        if profile_id not in profiles:

            raise ValueError(
                f"profiles.json에 "
                f"{profile_id}가 없습니다."
            )

        if not profiles[
            profile_id
        ]["profile"]:

            raise ValueError(
                f"{profile_id}의 "
                f"profile_text가 비어 있습니다."
            )

    return profiles


# ============================================================
# XLSX Ground Truth 로드
# ============================================================

def load_gold_xlsx(
    path: Path,
) -> List[str]:

    if not path.exists():

        raise FileNotFoundError(
            f"Gold 파일을 찾을 수 없습니다: {path}"
        )

    try:

        df = pd.read_excel(
            path
        )

    except Exception as e:

        raise ValueError(
            f"{path.name}을 읽을 수 없습니다: {e}"
        ) from e

    required_columns = {
        "arxiv_id",
        "label",
    }

    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:

        raise ValueError(
            f"{path.name}에 필요한 "
            f"컬럼이 없습니다: "
            f"{sorted(missing)}"
        )

    labels = pd.to_numeric(
        df["label"],
        errors="coerce",
    )

    relevant_ids = (
        df.loc[
            labels == 1,
            "arxiv_id",
        ]
        .dropna()
        .astype(str)
        .str.strip()
    )

    relevant_ids = relevant_ids[
        relevant_ids != ""
    ]

    relevant_ids = list(
        dict.fromkeys(
            relevant_ids.tolist()
        )
    )

    return relevant_ids


# ============================================================
# 평가 케이스 생성
# ============================================================

def load_cases(
    gold_dir: str,
    profile_path: str,
) -> List[Dict[str, Any]]:

    gold_path = Path(
        gold_dir
    )

    if not gold_path.exists():

        raise SystemExit(
            f"Gold 디렉터리를 "
            f"찾을 수 없습니다: "
            f"{gold_path}"
        )

    profiles = load_profiles(
        profile_path
    )

    cases: List[
        Dict[str, Any]
    ] = []

    print()
    print("=" * 78)
    print("Ground Truth 로드")
    print("=" * 78)

    for i in range(
        1,
        PROFILE_COUNT + 1,
    ):

        profile_id = f"P{i}"

        profile_info = profiles[
            profile_id
        ]

        profile = profile_info[
            "profile"
        ]

        category = profile_info.get(
            "category"
        )

        xlsx_path = (
            gold_path
            / f"{profile_id}.xlsx"
        )

        relevant_ids = load_gold_xlsx(
            xlsx_path
        )

        cases.append(
            {
                "profile_id": profile_id,
                "profile": profile,
                "category": category,
                "relevant_arxiv_ids": relevant_ids,
            }
        )

        print(
            f"[Gold] {profile_id:<4} "
            f"category={str(category):<10} "
            f"정답 논문="
            f"{len(relevant_ids)}개"
        )

    print("=" * 78)
    print()

    return cases


# ============================================================
# ID 중복 제거
# ============================================================

def unique_ids(
    ids: List[Any],
) -> List[str]:

    result = []

    seen = set()

    for value in ids:

        if value is None:
            continue

        value = str(
            value
        ).strip()

        if not value:
            continue

        if value in seen:
            continue

        seen.add(value)

        result.append(
            value
        )

    return result


# ============================================================
# 후보 dict 변환
# ============================================================

def normalize_candidate(
    candidate: Dict[str, Any],
) -> Dict[str, Any]:

    return {
        "arxiv_id": candidate.get(
            "arxiv_id",
            "",
        ),

        "title": candidate.get(
            "title",
            "",
        ),

        "primary_category": candidate.get(
            "primary_category",
            "",
        ),

        "submitted_date": candidate.get(
            "submitted_date",
            "",
        ),

        "abs_url": candidate.get(
            "abs_url"
        ),

        "abstract_clean": candidate.get(
            "abstract_clean"
        ),

        "source": candidate.get(
            "source",
            "",
        ),
    }


# ============================================================
# Stage 2 Rerank
# ============================================================

async def run_reranker(
    rerank_client: RerankPipelineClient,
    profile_text_en: str,
    candidates: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:

    if not candidates:
        return []

    if not rerank_client.is_configured:

        print(
            "[Evaluate] "
            "RERANK_PIPELINE_URL 없음 → "
            "Stage 2 skip"
        )

        return candidates[
            :RERANK_K
        ]

    payload_candidates = []

    for candidate in candidates:

        payload_candidates.append(
            {
                "arxiv_id": candidate.get(
                    "arxiv_id"
                ),

                "title": candidate.get(
                    "title",
                    "",
                ),

                "abstract_clean": candidate.get(
                    "abstract_clean"
                ),
            }
        )

    try:

        ranked = (
            await asyncio.to_thread(
                rerank_client.rerank,
                profile_text=profile_text_en,
                candidates=payload_candidates,
                diversity=0.0,
                boost_ids=None,
            )
        )

    except RerankPipelineClientError as e:

        print(
            "[Evaluate] "
            f"Reranker 호출 실패: {e}"
        )

        # 평가에서는 reranker 실패를 명확히 표시해야 하지만
        # 전체 평가 자체가 죽지는 않도록 Stage 1 순서를 사용한다.

        return candidates[
            :RERANK_K
        ]

    if not ranked:

        print(
            "[Evaluate] "
            "Reranker 결과 없음 → "
            "Stage 1 순서 사용"
        )

        return candidates[
            :RERANK_K
        ]

    by_id = {
        candidate.get(
            "arxiv_id"
        ): candidate
        for candidate in candidates
    }

    reranked = []

    used = set()

    for item in ranked:

        arxiv_id = item.get(
            "arxiv_id"
        )

        if not arxiv_id:
            continue

        candidate = by_id.get(
            arxiv_id
        )

        if candidate is None:
            continue

        if arxiv_id in used:
            continue

        used.add(
            arxiv_id
        )

        reranked.append(
            candidate
        )

    # --------------------------------------------------------
    # reranker가 일부만 반환한 경우
    # 나머지는 Stage 1 순서로 뒤에 추가
    # --------------------------------------------------------

    for candidate in candidates:

        arxiv_id = candidate.get(
            "arxiv_id"
        )

        if arxiv_id in used:
            continue

        reranked.append(
            candidate
        )

    return reranked[
        :RERANK_K
    ]


# ============================================================
# 기존 결과 로드
# ============================================================

def load_existing_results(
    output_path: str,
) -> Dict[str, CaseResult]:

    path = Path(
        output_path
    )

    if not path.exists():

        print(
            "[Merge] 기존 결과 파일 없음:"
            f" {path}"
        )

        return {}

    try:

        payload = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

    except Exception as e:

        print(
            "[Merge] 기존 결과 읽기 실패:"
            f" {e}"
        )

        return {}

    cases = payload.get(
        "cases",
        []
    )

    results = {}

    for item in cases:

        if not isinstance(
            item,
            dict,
        ):
            continue

        profile_id = item.get(
            "profile_id"
        )

        if not profile_id:
            continue

        try:

            result = CaseResult(
                profile_id=profile_id,

                category=item.get(
                    "category"
                ),

                n_relevant=int(
                    item.get(
                        "n_relevant",
                        0,
                    )
                ),

                retrieved_count=int(
                    item.get(
                        "retrieved_count",
                        0,
                    )
                ),

                retrieval_recall_at_50=float(
                    item.get(
                        "retrieval_recall_at_50",
                        0.0,
                    )
                ),

                retrieval_recall_at_100=float(
                    item.get(
                        "retrieval_recall_at_100",
                        0.0,
                    )
                ),

                rerank_count=int(
                    item.get(
                        "rerank_count",
                        0,
                    )
                ),

                rerank_recall_at_25=float(
                    item.get(
                        "rerank_recall_at_25",
                        0.0,
                    )
                ),

                final_count=int(
                    item.get(
                        "final_count",
                        0,
                    )
                ),

                final_precision_at_10=float(
                    item.get(
                        "final_precision_at_10",
                        0.0,
                    )
                ),

                final_ndcg_at_10=float(
                    item.get(
                        "final_ndcg_at_10",
                        0.0,
                    )
                ),

                final_recall_at_10=float(
                    item.get(
                        "final_recall_at_10",
                        0.0,
                    )
                ),

                error=item.get(
                    "error"
                ),
            )

            results[
                profile_id
            ] = result

        except Exception as e:

            print(
                "[Merge] "
                f"{profile_id} 결과 파싱 실패: "
                f"{e}"
            )

    print(
        "[Merge] 기존 결과 "
        f"{len(results)}개 로드"
    )

    return results


# ============================================================
# 프로필 하나 평가
# ============================================================

async def evaluate_case(
    case: Dict[str, Any],
    arxiv_client: ArxivPipelineClient,
    keyword_extraction_service: KeywordExtractionService,
    rerank_client: RerankPipelineClient,
    final_selection_service: FinalSelectionService,
) -> CaseResult:

    profile_id = case[
        "profile_id"
    ]

    profile = case[
        "profile"
    ]

    category = case.get(
        "category"
    )

    relevant = case.get(
        "relevant_arxiv_ids"
    ) or []

    try:

        # ====================================================
        # Stage 1a
        # Keyword Extraction
        # ====================================================

        extracted = (
            await keyword_extraction_service.extract(
                profile,
                category,
            )
        )

        print(
            "[Evaluate] "
            f"{profile_id} keywords="
            f"{extracted.keywords}"
        )

        print(
            "[Evaluate] "
            f"{profile_id} exclude="
            f"{extracted.exclude}"
        )

        # ====================================================
        # Stage 1b
        # Retrieval
        # ====================================================

        raw_candidates = (
            await arxiv_client.retrieve(
                profile_text=(
                    extracted.profile_text_en
                ),

                keywords=(
                    extracted.keywords
                ),

                category=category,

                n_keyword=N_KEYWORD,

                m_embedding=M_EMBEDDING,
            )
        )

        candidates = [
            normalize_candidate(
                candidate
            )
            for candidate in raw_candidates
        ]

        retrieved_ids = unique_ids(
            [
                candidate.get(
                    "arxiv_id"
                )
                for candidate in candidates
            ]
        )

        # ----------------------------------------------------
        # Retrieval Recall@50
        # ----------------------------------------------------

        retrieval_recall_50 = recall_at_k(
            retrieved_ids,
            relevant,
            RETRIEVAL_K,
        )

        # ----------------------------------------------------
        # Retrieval Recall@100
        # ----------------------------------------------------

        retrieval_recall_100 = recall_at_k(
            retrieved_ids,
            relevant,
            RETRIEVAL_K_2,
        )

        print(
            "[Evaluate] "
            f"{profile_id} "
            f"Retrieval={len(retrieved_ids)} | "
            f"R@50={retrieval_recall_50:.4f} | "
            f"R@100={retrieval_recall_100:.4f}"
        )

        # ====================================================
        # Stage 2
        # Reranker
        # ====================================================

        reranked = await run_reranker(
            rerank_client=rerank_client,

            profile_text_en=(
                extracted.profile_text_en
            ),

            candidates=candidates,
        )

        reranked_ids = unique_ids(
            [
                candidate.get(
                    "arxiv_id"
                )
                for candidate in reranked
            ]
        )

        rerank_recall = recall_at_k(
            reranked_ids,
            relevant,
            RERANK_K,
        )

        print(
            "[Evaluate] "
            f"{profile_id} "
            f"Rerank={len(reranked_ids)} | "
            f"R@{RERANK_K}="
            f"{rerank_recall:.4f}"
        )

        # ====================================================
        # Stage 3
        # Gemini Final Selection
        # ====================================================

        try:

            final_recs = (
                await final_selection_service.select(
                    profile,
                    reranked,
                )
            )

        except Exception as e:

            print(
                "[Evaluate] "
                f"{profile_id}: "
                f"Gemini 최종 선정 호출 실패: "
                f"{e}"
            )

            return CaseResult(
                profile_id=profile_id,

                category=category,

                n_relevant=len(
                    relevant
                ),

                retrieved_count=len(
                    retrieved_ids
                ),

                retrieval_recall_at_50=(
                    retrieval_recall_50
                ),

                retrieval_recall_at_100=(
                    retrieval_recall_100
                ),

                rerank_count=len(
                    reranked_ids
                ),

                rerank_recall_at_25=(
                    rerank_recall
                ),

                final_count=0,

                final_precision_at_10=0.0,

                final_ndcg_at_10=0.0,

                final_recall_at_10=0.0,

                error=(
                    "Gemini 최종 선정 호출 실패: "
                    f"{e}"
                ),
            )

        # ----------------------------------------------------
        # Gemini 결과 ID
        # ----------------------------------------------------

        final_ids = unique_ids(
            [
                getattr(
                    rec,
                    "arxiv_id",
                    None,
                )
                for rec in final_recs
            ]
        )

        final_ids = final_ids[
            :FINAL_K
        ]

        # ----------------------------------------------------
        # Final Precision@10
        # ----------------------------------------------------

        final_precision = precision_at_k(
            final_ids,
            relevant,
            FINAL_K,
        )

        # ----------------------------------------------------
        # Final NDCG@10
        # ----------------------------------------------------

        final_ndcg = ndcg_at_k(
            final_ids,
            relevant,
            FINAL_K,
        )

        # ----------------------------------------------------
        # Final Recall@10
        # ----------------------------------------------------

        final_recall = recall_at_k(
            final_ids,
            relevant,
            FINAL_K,
        )

        print(
            "[Evaluate] "
            f"{profile_id} "
            f"Final={len(final_ids)} | "
            f"P@10={final_precision:.4f} | "
            f"NDCG@10={final_ndcg:.4f} | "
            f"R@10={final_recall:.4f}"
        )

        return CaseResult(

            profile_id=profile_id,

            category=category,

            n_relevant=len(
                relevant
            ),

            retrieved_count=len(
                retrieved_ids
            ),

            retrieval_recall_at_50=(
                retrieval_recall_50
            ),

            retrieval_recall_at_100=(
                retrieval_recall_100
            ),

            rerank_count=len(
                reranked_ids
            ),

            rerank_recall_at_25=(
                rerank_recall
            ),

            final_count=len(
                final_ids
            ),

            final_precision_at_10=(
                final_precision
            ),

            final_ndcg_at_10=(
                final_ndcg
            ),

            final_recall_at_10=(
                final_recall
            ),

            error=None,
        )

    except Exception as e:

        return CaseResult(

            profile_id=profile_id,

            category=category,

            n_relevant=len(
                relevant
            ),

            retrieved_count=0,

            retrieval_recall_at_50=0.0,

            retrieval_recall_at_100=0.0,

            rerank_count=0,

            rerank_recall_at_25=0.0,

            final_count=0,

            final_precision_at_10=0.0,

            final_ndcg_at_10=0.0,

            final_recall_at_10=0.0,

            error=str(e),
        )


# ============================================================
# 결과 집계
# ============================================================

def aggregate_results(
    results: List[CaseResult],
) -> Dict[str, Any]:

    valid = [
        result
        for result in results
        if (
            result.error is None
            and result.n_relevant > 0
        )
    ]

    n = len(
        valid
    )

    if n == 0:

        return {
            "n_cases": len(
                results
            ),

            "n_valid_cases": 0,

            "retrieval_recall_at_50": 0.0,

            "retrieval_recall_at_100": 0.0,

            "rerank_recall_at_25": 0.0,

            "final_precision_at_10": 0.0,

            "final_ndcg_at_10": 0.0,

            "final_recall_at_10": 0.0,
        }

    return {

        "n_cases": len(
            results
        ),

        "n_valid_cases": n,

        "retrieval_recall_at_50": (
            sum(
                r.retrieval_recall_at_50
                for r in valid
            )
            / n
        ),

        "retrieval_recall_at_100": (
            sum(
                r.retrieval_recall_at_100
                for r in valid
            )
            / n
        ),

        "rerank_recall_at_25": (
            sum(
                r.rerank_recall_at_25
                for r in valid
            )
            / n
        ),

        "final_precision_at_10": (
            sum(
                r.final_precision_at_10
                for r in valid
            )
            / n
        ),

        "final_ndcg_at_10": (
            sum(
                r.final_ndcg_at_10
                for r in valid
            )
            / n
        ),

        "final_recall_at_10": (
            sum(
                r.final_recall_at_10
                for r in valid
            )
            / n
        ),
    }


# ============================================================
# 결과 출력
# ============================================================

def print_report(
    summary: Dict[str, Any],
    results: List[CaseResult],
) -> None:

    bar = "=" * 100

    print()
    print(bar)
    print("프로젝트 최종 평가 결과")
    print(bar)

    print(
        f"평가 프로필: "
        f"{summary['n_cases']}개"
    )

    print(
        f"유효 프로필: "
        f"{summary['n_valid_cases']}개"
    )

    print()

    # --------------------------------------------------------
    # Profile별 결과
    # --------------------------------------------------------

    print(
        f"{'Profile':<9}"
        f"{'Gold':>7}"
        f"{'R@50':>10}"
        f"{'R@100':>10}"
        f"{'RR@25':>10}"
        f"{'P@10':>10}"
        f"{'NDCG@10':>12}"
        f"{'R@10':>10}"
    )

    print("-" * 100)

    for result in results:

        if result.error:

            print(
                f"{result.profile_id:<9}"
                f"{result.n_relevant:>7}"
                f"{result.retrieval_recall_at_50:>10.4f}"
                f"{result.retrieval_recall_at_100:>10.4f}"
                f"{result.rerank_recall_at_25:>10.4f}"
                f"{'ERROR':>10}"
            )

            print(
                f"          ERROR: "
                f"{result.error}"
            )

            continue

        print(
            f"{result.profile_id:<9}"
            f"{result.n_relevant:>7}"
            f"{result.retrieval_recall_at_50:>10.4f}"
            f"{result.retrieval_recall_at_100:>10.4f}"
            f"{result.rerank_recall_at_25:>10.4f}"
            f"{result.final_precision_at_10:>10.4f}"
            f"{result.final_ndcg_at_10:>12.4f}"
            f"{result.final_recall_at_10:>10.4f}"
        )

    print("-" * 100)

    print(
        f"{'AVERAGE':<9}"
        f"{'':>7}"
        f"{summary['retrieval_recall_at_50']:>10.4f}"
        f"{summary['retrieval_recall_at_100']:>10.4f}"
        f"{summary['rerank_recall_at_25']:>10.4f}"
        f"{summary['final_precision_at_10']:>10.4f}"
        f"{summary['final_ndcg_at_10']:>12.4f}"
        f"{summary['final_recall_at_10']:>10.4f}"
    )

    print(bar)

    print()
    print("[핵심 평가 지표]")

    print(
        f"  Retrieval Recall@50  : "
        f"{summary['retrieval_recall_at_50']:.4f}"
    )

    print(
        f"  Retrieval Recall@100 : "
        f"{summary['retrieval_recall_at_100']:.4f}"
    )

    print(
        f"  Rerank Recall@25     : "
        f"{summary['rerank_recall_at_25']:.4f}"
    )

    print(
        f"  Final Precision@10   : "
        f"{summary['final_precision_at_10']:.4f}"
    )

    print(
        f"  Final NDCG@10        : "
        f"{summary['final_ndcg_at_10']:.4f}"
    )

    print(
        f"  Final Recall@10      : "
        f"{summary['final_recall_at_10']:.4f}"
    )

    # --------------------------------------------------------
    # 단계별 진단
    # --------------------------------------------------------

    print()
    print("[단계별 진단]")

    r100 = summary[
        "retrieval_recall_at_100"
    ]

    rr25 = summary[
        "rerank_recall_at_25"
    ]

    fr10 = summary[
        "final_recall_at_10"
    ]

    print(
        f"  Retrieval → Reranker : "
        f"{r100:.4f} → {rr25:.4f}"
    )

    print(
        f"  Reranker → Gemini    : "
        f"{rr25:.4f} → {fr10:.4f}"
    )

    if r100 > 0:

        rerank_drop = (
            r100 - rr25
        )

        print(
            f"  Reranker 손실        : "
            f"{rerank_drop:.4f}"
        )

    if rr25 > 0:

        gemini_drop = (
            rr25 - fr10
        )

        print(
            f"  Gemini 손실          : "
            f"{gemini_drop:.4f}"
        )

    # --------------------------------------------------------
    # 오류
    # --------------------------------------------------------

    errors = [
        result
        for result in results
        if result.error
    ]

    if errors:

        print()
        print(
            f"[오류] "
            f"{len(errors)}개 프로필"
        )

        for result in errors:

            print(
                f"  - {result.profile_id}: "
                f"{result.error}"
            )

    print()


# ============================================================
# JSON 저장
# ============================================================

def save_results(
    output_path: str,
    results: List[CaseResult],
) -> None:

    summary = aggregate_results(
        results
    )

    output = Path(
        output_path
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {

        "summary": summary,

        "cases": [
            asdict(result)
            for result in results
        ],
    }

    output.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"상세 결과 저장: "
        f"{output}"
    )


# ============================================================
# 전체 실행
# ============================================================

async def run(
    gold_dir: str,
    profile_path: str,
    output_path: Optional[str],
    concurrency: int,
    profiles_only: Optional[List[str]] = None,
    merge_existing: bool = False,
) -> None:

    # --------------------------------------------------------
    # 1. Gold + Profile 로드
    # --------------------------------------------------------

    cases = load_cases(
        gold_dir=gold_dir,
        profile_path=profile_path,
    )

    # --------------------------------------------------------
    # 2. 특정 프로필만 평가
    # --------------------------------------------------------

    selected_profiles: Optional[
        Set[str]
    ] = None

    if profiles_only:

        selected_profiles = set(
            profiles_only
        )

        valid_profile_ids = {
            f"P{i}"
            for i in range(
                1,
                PROFILE_COUNT + 1,
            )
        }

        invalid = (
            selected_profiles
            - valid_profile_ids
        )

        if invalid:

            raise ValueError(
                "존재하지 않는 프로필: "
                f"{sorted(invalid)}"
            )

        cases = [
            case
            for case in cases
            if case[
                "profile_id"
            ] in selected_profiles
        ]

        print()
        print(
            "[Partial Evaluation]"
        )

        print(
            "평가 대상: "
            f"{', '.join(sorted(selected_profiles))}"
        )

        print()

    # --------------------------------------------------------
    # 3. 기존 결과 로드
    # --------------------------------------------------------

    existing_results: Dict[
        str,
        CaseResult,
    ] = {}

    if merge_existing:

        if not output_path:

            raise ValueError(
                "--merge-existing를 "
                "사용하려면 "
                "--output이 필요합니다."
            )

        existing_results = (
            load_existing_results(
                output_path
            )
        )

    # --------------------------------------------------------
    # 4. 실제 서비스 객체 생성
    # --------------------------------------------------------

    arxiv_client = (
        ArxivPipelineClient()
    )

    keyword_extraction_service = (
        KeywordExtractionService(
            arxiv_client
        )
    )

    rerank_client = (
        RerankPipelineClient()
    )

    final_selection_service = (
        FinalSelectionService()
    )

    # --------------------------------------------------------
    # 5. 동시 실행 제한
    # --------------------------------------------------------

    semaphore = asyncio.Semaphore(
        max(
            1,
            concurrency,
        )
    )

    async def run_one(
        case: Dict[str, Any],
    ) -> CaseResult:

        async with semaphore:

            profile_id = case[
                "profile_id"
            ]

            print(
                f"[START] "
                f"{profile_id}"
            )

            result = await evaluate_case(

                case=case,

                arxiv_client=arxiv_client,

                keyword_extraction_service=(
                    keyword_extraction_service
                ),

                rerank_client=(
                    rerank_client
                ),

                final_selection_service=(
                    final_selection_service
                ),
            )

            if result.error:

                print(
                    f"[ERROR] "
                    f"{result.profile_id}: "
                    f"{result.error}"
                )

            else:

                print(
                    f"[DONE] "
                    f"{result.profile_id} | "

                    f"R@50="
                    f"{result.retrieval_recall_at_50:.4f} | "

                    f"R@100="
                    f"{result.retrieval_recall_at_100:.4f} | "

                    f"RR@25="
                    f"{result.rerank_recall_at_25:.4f} | "

                    f"P@10="
                    f"{result.final_precision_at_10:.4f} | "

                    f"NDCG@10="
                    f"{result.final_ndcg_at_10:.4f} | "

                    f"R@10="
                    f"{result.final_recall_at_10:.4f}"
                )

            return result

    # --------------------------------------------------------
    # 6. 평가 실행
    # --------------------------------------------------------

    new_results = await asyncio.gather(
        *(
            run_one(case)
            for case in cases
        )
    )

    # --------------------------------------------------------
    # 7. 결과 병합
    # --------------------------------------------------------

    result_map: Dict[
        str,
        CaseResult,
    ] = {}

    if merge_existing:

        result_map.update(
            existing_results
        )

    # 새 결과가 기존 결과를 덮어씀
    for result in new_results:

        result_map[
            result.profile_id
        ] = result

    # --------------------------------------------------------
    # 8. 전체 프로필 순서 정렬
    # --------------------------------------------------------

    results = []

    for i in range(
        1,
        PROFILE_COUNT + 1,
    ):

        profile_id = f"P{i}"

        result = result_map.get(
            profile_id
        )

        if result is not None:

            results.append(
                result
            )

    # --------------------------------------------------------
    # 9. 결과 출력
    # --------------------------------------------------------

    summary = aggregate_results(
        results
    )

    print_report(
        summary,
        results,
    )

    # --------------------------------------------------------
    # 10. JSON 저장
    # --------------------------------------------------------

    if output_path:

        save_results(
            output_path,
            results,
        )


# ============================================================
# CLI
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "P1~P12 XLSX Ground Truth 기반 "
            "Retrieval / Reranker / Gemini "
            "단계별 평가"
        )
    )

    parser.add_argument(
        "--gold-dir",
        required=True,
        help=(
            "P1.xlsx ~ P12.xlsx가 "
            "있는 디렉터리"
        ),
    )

    parser.add_argument(
        "--profiles",
        default=(
            "recommend_backend/eval/"
            "profiles.json"
        ),
        help=(
            "profiles.json 경로"
        ),
    )

    parser.add_argument(
        "--output",
        default=(
            "recommend_backend/eval/"
            "results.json"
        ),
        help=(
            "평가 결과 JSON 경로"
        ),
    )

    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help=(
            "동시 실행 프로필 수 "
            "(기본값: 1)"
        ),
    )

    parser.add_argument(
        "--profiles-only",
        nargs="+",
        default=None,
        help=(
            "특정 프로필만 재평가. "
            "예: --profiles-only P3 P11"
        ),
    )

    parser.add_argument(
        "--merge-existing",
        action="store_true",
        help=(
            "기존 results.json과 "
            "이번 평가 결과를 병합"
        ),
    )

    args = parser.parse_args()

    asyncio.run(
        run(

            gold_dir=args.gold_dir,

            profile_path=args.profiles,

            output_path=args.output,

            concurrency=args.concurrency,

            profiles_only=args.profiles_only,

            merge_existing=args.merge_existing,
        )
    )


if __name__ == "__main__":
    main()