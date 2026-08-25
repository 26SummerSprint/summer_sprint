
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
    │    - 입력 유효성 검사
    │    - 영어 검색 쿼리 생성
    │    - 키워드 추출
    │    - 제외조건 추출
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
    │
    ├─ Stage 3
    │    FinalSelectionService
    │    - Gemini 최종 선정
    │    - 추천 이유 생성
    │
    └─ Stage 4
         arxiv_pipeline /papers
         - 상세정보 보강

중요
------------------------------------------------------------
- invalid 입력만 RecommendService에서 검색을 중단한다.
- valid 입력은 기존 추천 파이프라인을 그대로 수행한다.
- 기존 /retrieve 호출 방식은 변경하지 않는다.
- reranker / final selection / paper enrichment 로직은 기존과 동일하다.
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
    ExtractedProfile,
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

    Stage 1a
        Gemini 입력 유효성 검사 + 프로필 분석

    Stage 1b
        arxiv_pipeline Hybrid Retrieval

    Stage 2
        arxiv_pipeline /rerank

    Stage 3
        Gemini 최종 논문 선정

    Stage 4
        arxiv_pipeline /papers

    invalid 입력:
        Gemini가 valid=false를 반환하면
        /retrieve를 호출하지 않고 즉시 빈 결과를 반환한다.

    valid 입력:
        기존 추천 파이프라인을 그대로 수행한다.
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

        # ----------------------------------------------------
        # Evaluation / Debug Trace
        # ----------------------------------------------------

        self.last_trace: Dict[str, Any] = {
            "retrieved_ids": [],
            "reranked_ids": [],
            "compressed_ids": [],
            "final_ids": [],
        }

    # ========================================================
    # 전체 추천
    # ========================================================

    async def recommend(
        self,
        profile: str,
        category: Optional[str] = None,
        diversity: float = 0.0,
        override_keywords: Optional[List[str]] = None,
        exclude_arxiv_ids: Optional[List[str]] = None,
    ) -> RecommendResponse:
        """
        전체 추천 파이프라인.

        invalid 입력은 Stage 1a에서 차단한다.

        valid 입력은 기존 검색 → rerank →
        Gemini 최종선정 → 상세정보 보강 흐름을 그대로 따른다.
        """

        # ====================================================
        # Evaluation Trace 초기화
        # ====================================================

        self.last_trace = {
            "retrieved_ids": [],
            "reranked_ids": [],
            "compressed_ids": [],
            "final_ids": [],
        }

        # ====================================================
        # Stage 1a
        #
        # 사용자 프로필
        #       ↓
        # Gemini
        #       ↓
        # valid
        # profile_text_en
        # keywords
        # exclusion
        # ====================================================

        if override_keywords:

            extracted_profile = ExtractedProfile(
                valid=True,

                profile_text_en="; ".join(
                    override_keywords
                ),

                keywords=override_keywords,

                exclude=[],
            )

            print(
                "[RecommendService] "
                "Stage 1a 건너뜀 "
                "(override_keywords 사용): "
                f"{override_keywords}"
            )

        else:

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

        # ----------------------------------------------------
        # 입력 유효성 검사
        #
        # 중요:
        # invalid인 경우 /retrieve를 호출하지 않는다.
        # ----------------------------------------------------

        if not extracted_profile.valid:

            print(
                "[RecommendService] "
                "INVALID 입력 → "
                "Stage 1 retrieve 중단"
            )

            return RecommendResponse(
                profile=profile,

                extracted_profile=(
                    extracted_profile
                ),

                count=0,

                recommendations=[],
            )

        # ----------------------------------------------------
        # 추출 결과 출력
        # ----------------------------------------------------

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
        # 기존 로직 그대로 유지
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

        # ----------------------------------------------------
        # Evaluation Trace
        # ----------------------------------------------------

        self.last_trace["retrieved_ids"] = [
            candidate.arxiv_id
            for candidate in candidates
            if candidate.arxiv_id
        ]

        print(
            "[RecommendService] "
            f"Stage 1 Retrieval 후보: "
            f"{len(candidates)}편"
        )

        # ----------------------------------------------------
        # 기준 논문 제거
        # ----------------------------------------------------

        if exclude_arxiv_ids:

            exclude_set = set(
                exclude_arxiv_ids
            )

            candidates = [
                candidate
                for candidate in candidates
                if candidate.arxiv_id
                not in exclude_set
            ]

        # ----------------------------------------------------
        # 피드백 반영
        # ----------------------------------------------------

        excluded_ids, upvoted_ids = feedback_sets(
            extracted_profile.keywords
        )

        hide_ids = (
            set(excluded_ids)
            | set(upvoted_ids)
        )

        if hide_ids:

            n_down = sum(
                1
                for candidate in candidates
                if candidate.arxiv_id
                in excluded_ids
            )

            n_up = sum(
                1
                for candidate in candidates
                if candidate.arxiv_id
                in upvoted_ids
            )

            candidates = [
                candidate
                for candidate in candidates
                if candidate.arxiv_id
                not in hide_ids
            ]

            if n_down or n_up:

                print(
                    "[RecommendService] "
                    f"피드백: 다운보트 {n_down}편 제외, "
                    f"업보트 {n_up}편 숨김 "
                    "(유사 논문은 부스트)"
                )

        print(
            "[RecommendService] "
            f"Stage 1 후보: "
            f"{len(candidates)}편"
        )

        # ----------------------------------------------------
        # 후보가 하나도 없으면 종료
        # ----------------------------------------------------

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
        # ====================================================

        compressed = (
            self._rerank_and_compress(
                profile_text_en=(
                    extracted_profile.profile_text_en
                ),

                candidates=candidates,

                diversity=diversity,

                boost_ids=list(
                    upvoted_ids
                ),
            )
        )

        print(
            "[RecommendService] "
            f"Stage 2 압축: "
            f"{len(candidates)}편 → "
            f"{len(compressed)}편"
        )

        # ----------------------------------------------------
        # Reranker fallback
        # ----------------------------------------------------

        if not compressed:

            compressed = candidates[
                :self._rerank_compress_count
            ]

            print(
                "[RecommendService] "
                "Stage 2 fallback → "
                "Stage 1 순서 사용"
            )

        # ----------------------------------------------------
        # Evaluation Trace
        # ----------------------------------------------------

        self.last_trace["reranked_ids"] = [
            candidate.arxiv_id
            for candidate in compressed
            if candidate.arxiv_id
        ]

        self.last_trace["compressed_ids"] = [
            candidate.arxiv_id
            for candidate in compressed
            if candidate.arxiv_id
        ]

        print(
            "[RecommendService] "
            f"Stage 2 최종 후보: "
            f"{len(self.last_trace['reranked_ids'])}편"
        )

        # ====================================================
        # Stage 3
        #
        # Gemini 최종 선정
        # ====================================================

        compressed_by_id = {
            candidate.arxiv_id: candidate
            for candidate in compressed
        }

        try:

            final_recs = (
                await self._final_selection_service.select(
                    profile,

                    [
                        candidate.as_dict()
                        for candidate in compressed
                    ],
                )
            )

        except Exception as e:

            print(
                "[RecommendService] "
                f"Stage 3 Gemini 최종 선정 실패: {e}"
            )

            raise

        # ----------------------------------------------------
        # 최종 추천 개수 제한
        # ----------------------------------------------------

        final_recs = (
            final_recs[
                :self._final_recommend_count
            ]
        )

        # ----------------------------------------------------
        # Gemini 결과 → API Response
        # ----------------------------------------------------

        recommendations = (
            self._build_recommendations(
                final_recs,
                compressed_by_id,
            )
        )

        # ----------------------------------------------------
        # Evaluation Trace
        # ----------------------------------------------------

        self.last_trace["final_ids"] = [
            recommendation.paper.get(
                "arxiv_id"
            )
            for recommendation in recommendations
            if recommendation.paper.get(
                "arxiv_id"
            )
        ]

        print(
            "[RecommendService] "
            f"Stage 3 최종 추천: "
            f"{len(recommendations)}편"
        )

        # ====================================================
        # Stage 4
        #
        # 최종 논문 상세정보
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

        if not candidates:
            return []

        # ----------------------------------------------------
        # Reranker 미설정
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
        # Payload
        # ----------------------------------------------------

        payload_candidates = []

        for candidate in candidates:

            payload_candidates.append(
                {
                    "arxiv_id": candidate.arxiv_id,

                    "title": candidate.title,

                    "abstract_clean": (
                        candidate.abstract_clean
                    ),
                }
            )

        # ----------------------------------------------------
        # Rerank
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

            return candidates[
                :self._rerank_compress_count
            ]

        # ----------------------------------------------------
        # 빈 결과
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
        # 원본 Candidate 매핑
        # ----------------------------------------------------

        by_id = {
            candidate.arxiv_id: candidate
            for candidate in candidates
        }

        reranked_candidates: List[
            _Candidate
        ] = []

        # ----------------------------------------------------
        # Reranker 순서대로 복원
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
        # 누락 후보 뒤에 추가
        # ----------------------------------------------------

        ranked_ids = {
            candidate.arxiv_id
            for candidate
            in reranked_candidates
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
        # 상위 N개
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
                    "Gemini가 존재하지 않는 "
                    "arxiv_id 반환: "
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
        # rank 정렬
        # ----------------------------------------------------

        recommendations.sort(
            key=lambda x: x.rank
        )

        # ----------------------------------------------------
        # rank 정규화
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

        # ----------------------------------------------------
        # arxiv_id 추출
        # ----------------------------------------------------

        ids = []

        for recommendation in recommendations:

            arxiv_id = (
                recommendation.paper.get(
                    "arxiv_id"
                )
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

            return

        # ----------------------------------------------------
        # 상세정보 반영
        # ----------------------------------------------------

        for recommendation in recommendations:

            arxiv_id = (
                recommendation.paper.get(
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

                recommendation.paper[
                    "pdf_url"
                ] = pdf_url

            # ------------------------------------------------
            # abs URL
            # ------------------------------------------------

            abs_url = detail.get(
                "abs_url"
            )

            if abs_url:

                recommendation.paper[
                    "abs_url"
                ] = abs_url

            # ------------------------------------------------
            # abstract
            # ------------------------------------------------

            if not recommendation.paper.get(
                "abstract_clean"
            ):

                recommendation.paper[
                    "abstract_clean"
                ] = detail.get(
                    "abstract_clean"
                )

            # ------------------------------------------------
            # title
            # ------------------------------------------------

            if not recommendation.paper.get(
                "title"
            ):

                recommendation.paper[
                    "title"
                ] = detail.get(
                    "title",
                    "",
                )

            # ------------------------------------------------
            # category
            # ------------------------------------------------

            if not recommendation.paper.get(
                "primary_category"
            ):

                recommendation.paper[
                    "primary_category"
                ] = detail.get(
                    "primary_category",
                    "",
                )

            # ------------------------------------------------
            # submitted date
            # ------------------------------------------------

            if not recommendation.paper.get(
                "submitted_date"
            ):

                recommendation.paper[
                    "submitted_date"
                ] = detail.get(
                    "submitted_date",
                    "",
                )

