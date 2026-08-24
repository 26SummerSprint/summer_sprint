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

평가 Trace
------------------------------------------------------------
Evaluation 시 다음 단계별 결과를 기록한다.

Stage 1
    retrieved_ids

Stage 2
    reranked_ids
    compressed_ids

Stage 3
    final_ids

이를 이용하여:

    Retrieval Recall@50
    Retrieval Recall@100
    Rerank Recall@25
    Final Precision@10
    Final NDCG@10
    Final Recall@10

을 단계별로 분석할 수 있다.

중요:
- recommend_backend에는 reranker.py가 필요하지 않다.
- recommend_backend에는 sentence-transformers / torch가 필요하지 않다.
- reranker는 arxiv_pipeline EC2에서 실행된다.
- Evaluation Trace는 추천 동작 자체에는 영향을 주지 않는다.
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


    Evaluation Trace
    --------------------------------------------------------
    self.last_trace에 다음 정보를 기록한다.

        retrieved_ids
            Stage 1 Retrieval 결과

        reranked_ids
            Stage 2 Reranker 결과

        compressed_ids
            Gemini에 실제 전달된 후보

        final_ids
            Gemini 최종 선정 결과

    평가 코드에서는 다음과 같이 사용할 수 있다.

        trace = service.last_trace

        retrieved_ids = trace["retrieved_ids"]
        reranked_ids = trace["reranked_ids"]
        compressed_ids = trace["compressed_ids"]
        final_ids = trace["final_ids"]
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
        #
        # 평가 시 Retrieval → Reranker → Gemini
        # 각 단계의 후보 ID를 추적한다.
        #
        # 일반 추천 동작에는 영향을 주지 않는다.
        #

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

        profile
            사용자가 입력한 연구 관심사.

        category
            선택적 arXiv 카테고리.
            예: cs.RO

        override_keywords
            주어지면 Stage 1a(Gemini 프로필 분석)를 건너뛰고
            이 키워드를 그대로 검색 조건으로 사용한다.

        exclude_arxiv_ids
            Stage 1b 직후 해당 arxiv_id를 후보에서 제거한다.
        """

        # ====================================================
        # Evaluation Trace 초기화
        # ====================================================
        #
        # recommend() 호출마다 반드시 초기화한다.
        #
        # 이렇게 하지 않으면 이전 프로필의 trace가
        # 다음 프로필 평가에 섞일 수 있다.
        #

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
        # 영어 검색 쿼리
        # 키워드
        # 제외조건
        #       ↓
        # /keyword_df 검증
        # ====================================================

        if override_keywords:
            extracted_profile = ExtractedProfile(
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

        # ----------------------------------------------------
        # Evaluation Trace
        # Stage 1 Retrieval 결과
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
        # '이 논문으로 다시 추천받기'
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
                    f"업보트 {n_up}편 숨김"
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
        #
        # CrossEncoder
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
        # Reranker 결과가 비어 있으면
        # Stage 1 순서를 유지
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
        #
        # 최종적으로 Gemini에 전달되는 후보
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
        #
        # 후보 20~30편
        #       ↓
        # Gemini
        #       ↓
        # 최종 최대 10편
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
        # Gemini 결과
        # →
        # API Response
        # ----------------------------------------------------

        recommendations = (
            self._build_recommendations(
                final_recs,
                compressed_by_id,
            )
        )

        # ----------------------------------------------------
        # Evaluation Trace
        #
        # 실제 API로 반환되는 최종 논문
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

        # ----------------------------------------------------
        # 업보트 반영
        # ----------------------------------------------------

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

        Reranker 실패 시에도
        Stage 1 후보를 그대로 fallback으로 사용한다.
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

            # ------------------------------------------------
            # Reranker 실패
            #
            # Stage 1 순서를 유지한다.
            # ------------------------------------------------

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
        # Reranker가 일부 후보만 반환한 경우
        #
        # 나머지는 Stage 1 순서 그대로 뒤에 붙인다.
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
        # Gemini rank 순으로 정렬
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

        for recommendation in recommendations:

            arxiv_id = recommendation.paper.get(
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