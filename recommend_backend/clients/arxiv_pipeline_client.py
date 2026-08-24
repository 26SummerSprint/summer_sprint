"""
arxiv_pipeline HTTP 클라이언트.

arxiv_pipeline은 별도 프로젝트로 이미 존재하며, 이 클라이언트는 그 REST API를
호출만 한다 (구현/수정하지 않음). keyword_pipeline이 arxiv_pipeline에
합쳐지면서, 하이브리드 검색(BM25+임베딩 병합)과 키워드 DF 조회도 이제
arxiv_pipeline이 직접 제공한다 (더 이상 별도 keyword_pipeline 서비스가 없음).

계약 (arxiv_pipeline/app.py 기준, 실제 코드로 확인함):

    POST {ARXIV_PIPELINE_URL}/search
    Body: {"query": str, "top_k": int, "category": str | None}
    Response: [{"arxiv_id","score","title","primary_category","submitted_date",
                "abs_url","pdf_url","abstract_clean"}, ...]
    (embedding 단독 검색. score = cosine distance, 작을수록 유사)

    POST {ARXIV_PIPELINE_URL}/retrieve
    Body: {"profile_text": str, "keywords": [str, ...], "category": str | None,
           "n_keyword": int, "m_embedding": int}
    Response: [{"arxiv_id","source","title","primary_category","submitted_date",
                "abs_url","abstract_clean"}, ...]
    (Stage 1 하이브리드 검색: 키워드 top n_keyword + 임베딩 top m_embedding,
     중복 제거는 arxiv_pipeline이 직접 수행. source는 "keyword"/"embedding"/
     "both". 점수는 반환하지 않음 - 팀이 점수 합산/정규화를 하지 않기로
     확정했기 때문. pdf_url도 포함되지 않음.)

    GET {ARXIV_PIPELINE_URL}/keyword_df?term=...&category=...
    Response: {"term": str, "df": int, "df_in_category": int | null}
    (키워드 Document Frequency 조회. 권장 검증 구간 50~500.)

    GET {ARXIV_PIPELINE_URL}/papers?ids=id1,id2,...
    Response: [{"arxiv_id","version","title","authors","abstract_raw",
                "abstract_clean","categories","primary_category","comments",
                "submitted_date","updated_date","abs_url","pdf_url"}, ...]
    (여러 논문 상세 조회. /retrieve 결과에는 없는 pdf_url 등을 보강할 때 사용.)

    모든 엔드포인트는 X-API-Key 헤더 인증이 필요하다 (health 제외).
"""

from typing import Any, Dict, List, Optional

import httpx

from ..config import ARXIV_PIPELINE_API_KEY, ARXIV_PIPELINE_URL, HTTP_REQUEST_TIMEOUT


class ArxivPipelineClientError(Exception):
    """arxiv_pipeline 호출 실패 시 사용하는 예외."""


class ArxivPipelineClient:
    def __init__(
        self,
        base_url: str = ARXIV_PIPELINE_URL,
        api_key: str = ARXIV_PIPELINE_API_KEY,
        timeout: int = HTTP_REQUEST_TIMEOUT,
    ):
        self._base_url = base_url.rstrip("/")
        self._headers = {"X-API-Key": api_key}
        self._timeout = timeout

    async def search(
        self,
        query: str,
        top_k: int = 20,
        category: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        embedding 단독 검색 (POST /search). 현재 오케스트레이션의 Stage 1은
        /retrieve를 쓰므로 직접 쓰이진 않지만, 다른 용도(유사 논문 찾기 등)를
        위해 남겨둔다.
        """
        payload: Dict[str, Any] = {"query": query, "top_k": top_k}
        if category:
            payload["category"] = category
        return await self._post("/search", payload)

    async def retrieve(
        self,
        profile_text: str,
        keywords: List[str],
        category: Optional[str] = None,
        n_keyword: int = 30,
        m_embedding: int = 70,
    ) -> List[Dict[str, Any]]:
        """
        Stage 1 하이브리드 검색 (POST /retrieve).
        키워드(BM25) top n_keyword + 임베딩 top m_embedding을 arxiv_pipeline이
        직접 병합/중복제거해서 반환한다 - recommend_backend는 더 이상 이
        merge 로직을 직접 구현하지 않는다.
        """
        payload: Dict[str, Any] = {
            "profile_text": profile_text,
            "keywords": keywords,
            "n_keyword": n_keyword,
            "m_embedding": m_embedding,
        }
        if category:
            payload["category"] = category
        return await self._post("/retrieve", payload)

    async def keyword_df(self, term: str, category: Optional[str] = None) -> Dict[str, Any]:
        """
        키워드 Document Frequency 조회 (GET /keyword_df).
        반환: {"term": str, "df": int, "df_in_category": int | None}
        """
        self._require_base_url()
        params: Dict[str, Any] = {"term": term}
        if category:
            params["category"] = category

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            try:
                response = await client.get(
                    f"{self._base_url}/keyword_df", params=params, headers=self._headers
                )
                response.raise_for_status()
            except httpx.HTTPError as e:
                raise ArxivPipelineClientError(
                    f"arxiv_pipeline /keyword_df 호출 실패: {e}"
                ) from e
        return response.json()

    async def get_papers(self, ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        여러 논문 상세 조회 (GET /papers?ids=...).
        /retrieve 결과에는 없는 pdf_url 등을 최종 추천작에만 보강할 때 쓴다.
        반환: {arxiv_id: PaperDetail dict}
        """
        if not ids:
            return {}
        self._require_base_url()

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            try:
                response = await client.get(
                    f"{self._base_url}/papers",
                    params={"ids": ",".join(ids)},
                    headers=self._headers,
                )
                response.raise_for_status()
            except httpx.HTTPError as e:
                raise ArxivPipelineClientError(f"arxiv_pipeline /papers 호출 실패: {e}") from e

        papers = response.json()
        return {p["arxiv_id"]: p for p in papers}

    async def _post(self, path: str, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        self._require_base_url()
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            try:
                response = await client.post(
                    f"{self._base_url}{path}", json=payload, headers=self._headers
                )
                response.raise_for_status()
            except httpx.HTTPError as e:
                raise ArxivPipelineClientError(f"arxiv_pipeline {path} 호출 실패: {e}") from e
        return response.json()

    def _require_base_url(self) -> None:
        if not self._base_url:
            raise ArxivPipelineClientError(
                "ARXIV_PIPELINE_URL이 설정되지 않았습니다. "
                'export ARXIV_PIPELINE_URL="http://..."'
            )
