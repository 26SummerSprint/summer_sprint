"""
RecommendService: 전체 논문 추천 오케스트레이션.

구조
------------------------------------------------------------
Streamlit
    ↓
recommend_backend
    │
    ├─ Stage 1a
    │    KeywordExtractionService
    │    - Gemini로 프로필 분석
    │    - 영어 검색 쿼리 생성
    │    - 키워드 추출
    │    - 제외조건 추출
    │    - arxiv_pipeline /keyword_df로 DF 검증
    │
    ├─ Stage 1b
    │    arxiv_pipeline /retrieve
    │    - BM25
    │    - Embedding
    │    - 후보 병합 / 중복 제거
    │
    ├─ Stage 2
    │    arxiv_pipeline /rerank
    │    - CrossEncoder
    │    - reranker.py는 EC2의 arxiv_pipeline에만 존재
    │    - recommend_backend는 HTTP로만 호출
    │
    ├─ Stage 3
    │    FinalSelectionService
    │    - Gemini 최종 선정
    │    - 추천 이유 생성
    │
    └─ Stage 4
         arxiv_pipeline /papers
         - pdf_url 등 상세정보 보강

중요:
- recommend_backend에는 reranker.py가 필요하지 않다.
- recommend_backend에는 sentence-transformers / torch가 필요하지 않다.
- reranker는 arxiv_pipeline EC2에서 실행된다.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..clients.arxiv_pipeline_client import (
    ArxivPipelineClient,
    ArxivPipelineClientError,
)

from ..clients.rerank_pipeline_client import (
    RerankPipelineClient,
    RerankPipelineClientError,
)

from ..config import (
    FINAL_RECOMMEND_COUNT,
    M_EMBEDDING,
    N_KEYWORD,
    RERANK_COMPRESS_COUNT,
)

from ..schemas import (
    RecommendedPaperOut,
    RecommendResponse,
)

from .final_selection_service import FinalSelectionService
from .keyword_extraction_service import KeywordExtractionService
from .feedback_store import feedback_sets


# ============================================================
# Candidate
# ============================================================

@dataclass
class _Candidate:
    """
    arxiv_pipeline /retrieve에서 반환된 후보 논문 1건.
    """

    arxiv_id: str
    title: str
    primary_category: str
    submitted_date: str
    abs_url: Optional[str]
    abstract_clean: Optional[str]
    source: str = ""

    def as_dict(self) -> Dict[str, Any]:
        """
        Gemini 최종 선정 단계에 전달할 수 있는 dict 형태.
        """

        return {
            "arxiv_id": self.arxiv_id,
            "title": self.title,
            "primary_category": self.primary_category,
            "submitted_date": self.submitted_date,
            "abs_url": self.abs_url,
            "abstract_clean": self.abstract_clean,
            "source": self.source,
        }

    @classmethod
    def from_raw(
        cls,
        raw: Dict[str, Any],
    ) -> "_Candidate":
        """
        arxiv_pipeline /retrieve 응답을 Candidate로 변환.
        """

        return cls(
            arxiv_id=raw.get(
                "arxiv_id",
                "",
            ),
            title=raw.get(
                "title",
                "",
            ),
            primary_category=raw.get(
                "primary_category",
                "",
            ),
            submitted_date=raw.get(
                "submitted_date",
                "",
            ),
            abs_url=raw.get(
                "abs_url",
            ),
            abstract_clean=raw.get(
                "abstract_clean",
            ),
            source=raw.get(
                "source",
                "",
            ),
        )


# ============================================================
# RecommendService
# ============================================================

class RecommendService:
    """
    전체 추천 파이프라인을 오케스트레이션한다.

    Stage 1
        Gemini 프로필 분석
        +
        arxiv_pipeline Hybrid Retrieval

    Stage 2
        arxiv_pipeline /rerank HTTP 호출

    Stage 3
        Gemini 최종 논문 선정

    Stage 4
        arxiv_pipeline /papers 상세정보 보강
    """

    def __init__(
        self,
        arxiv_client: Optional[
            ArxivPipelineClient
        ] = None,

        rerank_client: Optional[
            RerankPipelineClient
        ] = None,

        keyword_extraction_service: Optional[
            KeywordExtractionService
        ] = None,

        final_selection_service: Optional[
            FinalSelectionService
        ] = None,

        n_keyword: int = N_KEYWORD,

        m_embedding: int = M_EMBEDDING,

        rerank_compress_count: int = (
            RERANK_COMPRESS_COUNT
        ),

        final_recommend_count: int = (
            FINAL_RECOMMEND_COUNT
        ),
    ):
        # ----------------------------------------------------
        # Stage 1 / Stage 4
        # ----------------------------------------------------

        self._arxiv_client = (
            arxiv_client
            or ArxivPipelineClient()
        )

        # ----------------------------------------------------
        # Stage 2
        # ----------------------------------------------------

        self._rerank_client = (
            rerank_client
            or RerankPipelineClient()
        )

        # ----------------------------------------------------
        # Stage 1a
        # ----------------------------------------------------

        self._keyword_extraction_service = (
            keyword_extraction_service
            or KeywordExtractionService(
                self._arxiv_client
            )
        )

        # ----------------------------------------------------
        # Stage 3
        # ----------------------------------------------------

        self._final_selection_service = (
            final_selection_service
            or FinalSelectionService()
        )

        # ----------------------------------------------------
        # 설정값
        # ----------------------------------------------------

        self._n_keyword = n_keyword

        self._m_embedding = m_embedding

        self._rerank_compress_count = (
            rerank_compress_count
        )

        self._final_recommend_count = (
            final_recommend_count
        )

    # ========================================================
    # 전체 추천
    # ========================================================

    async def recommend(
        self,
        profile: str,
        category: Optional[str] = None,
        diversity: float = 0.0,
    ) -> RecommendResponse:
        """
        전체 추천 파이프라인.

        profile
            사용자가 입력한 연구 관심사

        category
            선택적 arXiv 카테고리
            예: cs.RO
        """

        # ====================================================
        # Stage 1a
        #
        # 사용자 프로필
        #       ↓
        # Gemini
        #       ↓
        # 영어 검색 쿼리
        # 키워드
        # 제외조건
        #       ↓
        # /keyword_df 검증
        # ====================================================

        extracted_profile = (
            await self._keyword_extraction_service.extract(
                profile,
                category,
            )
        )

        print(
            "[RecommendService] "
            "Stage 1a 프로필 추출 완료"
        )

        print(
            "[RecommendService] "
            f"keywords="
            f"{extracted_profile.keywords}"
        )

        print(
            "[RecommendService] "
            f"exclude="
            f"{extracted_profile.exclude}"
        )

        # ====================================================
        # Stage 1b
        #
        # arxiv_pipeline /retrieve
        #
        # BM25 top N
        # +
        # Embedding top M
        # →
        # union / dedupe
        # ====================================================

        try:
            raw_candidates = (
                await self._arxiv_client.retrieve(
                    profile_text=(
                        extracted_profile.profile_text_en
                    ),
                    keywords=(
                        extracted_profile.keywords
                    ),
                    category=category,
                    n_keyword=self._n_keyword,
                    m_embedding=self._m_embedding,
                )
            )

        except ArxivPipelineClientError as e:
            print(
                "[RecommendService] "
                f"Stage 1 retrieve 실패: {e}"
            )

            raise

        candidates = [
            _Candidate.from_raw(raw)
            for raw in raw_candidates
        ]

        print(
            "[RecommendService] "
            f"Stage 1 후보: "
            f"{len(candidates)}편"
        )

        # ====================================================
        # 피드백 반영 (키워드 단위 — 프로필 무관)
        #   다운보트 누적 논문 제외(후보 단계) /
        #   업보트 누적 논문은 Stage 2에서 '유사 논문 부스트'로 반영
        # ====================================================
        excluded_ids, upvoted_ids = feedback_sets(
            extracted_profile.keywords
        )
        if excluded_ids:
            kept = [
                c for c in candidates
                if c.arxiv_id not in excluded_ids
            ]
            if len(kept) != len(candidates):
                print(
                    "[RecommendService] "
                    f"피드백: 다운보트 {len(candidates) - len(kept)}편 제외"
                )
            candidates = kept

        # 후보가 하나도 없으면 바로 종료
        if not candidates:

            return RecommendResponse(
                profile=profile,

                extracted_profile=(
                    extracted_profile
                ),

                count=0,

                recommendations=[],
            )

        # ====================================================
        # Stage 2
        #
        # arxiv_pipeline /rerank
        #
        # CrossEncoder
        #
        # recommend_backend에서는
        # reranker.py를 직접 import하지 않는다.
        # ====================================================

        compressed = (
            self._rerank_and_compress(
                profile_text_en=(
                    extracted_profile.profile_text_en
                ),
                candidates=candidates,
                diversity=diversity,
                boost_ids=list(upvoted_ids),
            )
        )

        print(
            "[RecommendService] "
            f"Stage 2 압축: "
            f"{len(candidates)}편 → "
            f"{len(compressed)}편"
        )

        # 혹시 rerank 결과가 비어 있으면
        # Stage 1 후보를 사용
        if not compressed:

            compressed = candidates[
                :self._rerank_compress_count
            ]

        # ====================================================
        # Stage 3
        #
        # Gemini 최종 선정
        #
        # 후보 20~30편
        #       ↓
        # Gemini
        #       ↓
        # 최종 최대 10편
        # ====================================================

        compressed_by_id = {
            c.arxiv_id: c
            for c in compressed
        }

        try:
            final_recs = (
                await self._final_selection_service.select(
                    profile,
                    [
                        c.as_dict()
                        for c in compressed
                    ],
                )
            )

        except Exception as e:
            print(
                "[RecommendService] "
                f"Stage 3 Gemini 최종 선정 실패: {e}"
            )

            raise

        # ====================================================
        # 최종 추천 개수 제한
        # ====================================================

        final_recs = (
            final_recs[
                :self._final_recommend_count
            ]
        )

        # ====================================================
        # Gemini 결과
        # →
        # API Response
        # ====================================================

        recommendations = (
            self._build_recommendations(
                final_recs,
                compressed_by_id,
            )
        )

        # 업보트 반영은 Stage 2 rerank 단계의 유사도 부스트(boost_ids)로 처리한다.
        # (업보트한 논문 '자체'가 아니라 그와 유사한 후보를 상위로 → Gemini에 더 노출)

        print(
            "[RecommendService] "
            f"Stage 3 최종 추천: "
            f"{len(recommendations)}편"
        )

        # ====================================================
        # Stage 4
        #
        # 최종 선정된 논문에 대해서만
        # arxiv_pipeline /papers 호출
        #
        # pdf_url 등 상세정보 보강
        # ====================================================

        await self._enrich_with_paper_details(
            recommendations
        )

        # ====================================================
        # 최종 Response
        # ====================================================

        return RecommendResponse(
            profile=profile,

            extracted_profile=(
                extracted_profile
            ),

            count=len(
                recommendations
            ),

            recommendations=(
                recommendations
            ),
        )

    # ========================================================
    # Stage 2
    # HTTP Reranker
    # ========================================================

    def _rerank_and_compress(
        self,
        profile_text_en: str,
        candidates: List[_Candidate],
        diversity: float = 0.0,
        boost_ids: Optional[List[str]] = None,
    ) -> List[_Candidate]:
        """
        arxiv_pipeline의 /rerank를 호출하여
        후보를 재랭킹하고 상위 N개만 반환한다.

        RERANK_PIPELINE_URL이 비어 있으면
        reranker를 건너뛰고 Stage 1 순서를 유지한다.

        중요:
        CrossEncoder는 이 프로세스에서 실행되지 않는다.

        실제 CrossEncoder는 EC2의 arxiv_pipeline에서
        reranker.py를 통해 실행된다.
        """

        if not candidates:
            return []

        # ----------------------------------------------------
        # Reranker가 설정되지 않은 경우
        # ----------------------------------------------------

        if not self._rerank_client.is_configured:

            print(
                "[RecommendService] "
                "RERANK_PIPELINE_URL 없음 → "
                "Stage 2 skip"
            )

            return candidates[
                :self._rerank_compress_count
            ]

        # ----------------------------------------------------
        # arxiv_pipeline /rerank에 전달할 후보 생성
        # ----------------------------------------------------

        payload_candidates = []

        for candidate in candidates:

            payload_candidates.append(
                {
                    "arxiv_id": (
                        candidate.arxiv_id
                    ),

                    "title": (
                        candidate.title
                    ),

                    "abstract_clean": (
                        candidate.abstract_clean
                    ),
                }
            )

        # ----------------------------------------------------
        # /rerank 호출
        # ----------------------------------------------------

        try:

            ranked = (
                self._rerank_client.rerank(
                    profile_text=profile_text_en,

                    candidates=(
                        payload_candidates
                    ),

                    diversity=diversity,

                    boost_ids=boost_ids,
                )
            )

        except RerankPipelineClientError as e:

            print(
                "[RecommendService] "
                f"Stage 2 rerank 호출 실패: {e}"
            )

            # reranker 실패가
            # 전체 추천 실패로 이어지지 않도록
            # Stage 1 순서 유지

            return candidates[
                :self._rerank_compress_count
            ]

        # ----------------------------------------------------
        # API 응답 검증
        # ----------------------------------------------------

        if not ranked:

            print(
                "[RecommendService] "
                "rerank 결과가 비어 있음 → "
                "Stage 1 순서 유지"
            )

            return candidates[
                :self._rerank_compress_count
            ]

        # ----------------------------------------------------
        # arxiv_id → 원본 Candidate
        # ----------------------------------------------------

        by_id = {
            candidate.arxiv_id: candidate
            for candidate in candidates
        }

        reranked_candidates: List[
            _Candidate
        ] = []

        # ----------------------------------------------------
        # reranker 순서대로 복원
        # ----------------------------------------------------

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

            reranked_candidates.append(
                candidate
            )

        # ----------------------------------------------------
        # reranker가 일부 후보만 반환한 경우
        #
        # 나머지는 Stage 1 순서 그대로 뒤에 붙임
        # ----------------------------------------------------

        ranked_ids = {
            candidate.arxiv_id
            for candidate in reranked_candidates
        }

        for candidate in candidates:

            if (
                candidate.arxiv_id
                not in ranked_ids
            ):

                reranked_candidates.append(
                    candidate
                )

        # ----------------------------------------------------
        # 상위 N개 압축
        # ----------------------------------------------------

        return reranked_candidates[
            :self._rerank_compress_count
        ]

    # ========================================================
    # Stage 3
    # Gemini 결과 → API Response
    # ========================================================

    @staticmethod
    def _build_recommendations(
        final_recs: List,
        compressed_by_id: Dict[
            str,
            _Candidate,
        ],
    ) -> List[
        RecommendedPaperOut
    ]:
        """
        Gemini가 반환한 최종 추천 결과를
        RecommendedPaperOut으로 변환한다.

        Gemini가 후보에 없는 arXiv ID를
        반환한 경우 해당 항목은 제거한다.
        """

        recommendations: List[
            RecommendedPaperOut
        ] = []

        for rec in final_recs:

            candidate = (
                compressed_by_id.get(
                    rec.arxiv_id
                )
            )

            # ------------------------------------------------
            # Gemini hallucination 방어
            # ------------------------------------------------

            if candidate is None:

                print(
                    "[RecommendService] "
                    f"Gemini가 존재하지 않는 "
                    f"arxiv_id 반환: "
                    f"{rec.arxiv_id}"
                )

                continue

            recommendations.append(
                RecommendedPaperOut(
                    rank=rec.rank,

                    paper=(
                        candidate.as_dict()
                    ),

                    reason=(
                        rec.reason
                    ),
                )
            )

        # ----------------------------------------------------
        # Gemini rank 순으로 정렬
        # ----------------------------------------------------

        recommendations.sort(
            key=lambda x: x.rank
        )

        # ----------------------------------------------------
        # rank를 실제 반환 순서 기준으로 정규화
        #
        # Gemini가 1, 3, 5처럼 이상하게 반환하는 경우
        # API 결과는 항상 1,2,3...이 되도록 처리
        # ----------------------------------------------------

        for index, recommendation in enumerate(
            recommendations,
            start=1,
        ):
            recommendation.rank = index

        return recommendations

    # ========================================================
    # Stage 4
    # 논문 상세정보 보강
    # ========================================================

    async def _enrich_with_paper_details(
        self,
        recommendations: List[
            RecommendedPaperOut
        ],
    ) -> None:
        """
        최종 추천된 논문에 대해서만
        arxiv_pipeline /papers를 호출한다.

        /retrieve에는 pdf_url이 없으므로
        최종 선정된 논문에 대해 상세정보를 가져온다.

        상세정보 API 실패는 추천 전체 실패로 처리하지 않는다.
        """

        # ----------------------------------------------------
        # arxiv_id 추출
        # ----------------------------------------------------

        ids = []

        for rec in recommendations:

            arxiv_id = rec.paper.get(
                "arxiv_id"
            )

            if arxiv_id:
                ids.append(
                    arxiv_id
                )

        if not ids:
            return

        # ----------------------------------------------------
        # /papers 호출
        # ----------------------------------------------------

        try:

            detail_map = (
                await self._arxiv_client.get_papers(
                    ids
                )
            )

        except ArxivPipelineClientError as e:

            print(
                "[RecommendService] "
                f"논문 상세정보 보강 실패: {e}"
            )

            # pdf_url이 없어도
            # 추천 자체는 정상 반환

            return

        # ----------------------------------------------------
        # 상세정보 반영
        # ----------------------------------------------------

        for rec in recommendations:

            arxiv_id = (
                rec.paper.get(
                    "arxiv_id"
                )
            )

            if not arxiv_id:
                continue

            detail = (
                detail_map.get(
                    arxiv_id
                )
            )

            if not detail:
                continue

            # ------------------------------------------------
            # PDF URL
            # ------------------------------------------------

            pdf_url = detail.get(
                "pdf_url"
            )

            if pdf_url:

                rec.paper[
                    "pdf_url"
                ] = pdf_url

            # ------------------------------------------------
            # abs URL
            # ------------------------------------------------

            abs_url = detail.get(
                "abs_url"
            )

            if abs_url:

                rec.paper[
                    "abs_url"
                ] = abs_url

            # ------------------------------------------------
            # abstract
            # ------------------------------------------------

            if not rec.paper.get(
                "abstract_clean"
            ):

                rec.paper[
                    "abstract_clean"
                ] = detail.get(
                    "abstract_clean"
                )

            # ------------------------------------------------
            # title
            # ------------------------------------------------

            if not rec.paper.get(
                "title"
            ):

                rec.paper[
                    "title"
                ] = detail.get(
                    "title",
                    "",
                )

            # ------------------------------------------------
            # category
            # ------------------------------------------------

            if not rec.paper.get(
                "primary_category"
            ):

                rec.paper[
                    "primary_category"
                ] = detail.get(
                    "primary_category",
                    "",
                )

            # ------------------------------------------------
            # submitted date
            # ------------------------------------------------

            if not rec.paper.get(
                "submitted_date"
            ):

                rec.paper[
                    "submitted_date"
                ] = detail.get(
                    "submitted_date",
                    "",
                )