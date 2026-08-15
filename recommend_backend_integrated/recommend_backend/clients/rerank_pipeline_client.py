"""
arxiv_pipeline의 /rerank endpoint를 호출하는 HTTP 클라이언트.

현재 구조에서는 별도의 rerank_pipeline 서버가 존재하지 않는다.

arxiv_pipeline EC2 서버가 다음을 제공한다.

    POST /retrieve
    POST /rerank
    GET  /keyword_df
    GET  /papers

따라서 recommend_backend는 CrossEncoder를 직접 로드하지 않고
arxiv_pipeline의 /rerank API를 호출한다.
"""

from typing import Any, Dict, List

import requests

from ..config import (
    ARXIV_PIPELINE_API_KEY,
    ARXIV_PIPELINE_URL,
    HTTP_REQUEST_TIMEOUT,
)


class RerankPipelineClientError(Exception):
    """rerank API 호출 실패."""


class RerankPipelineClient:

    def __init__(
        self,
        base_url: str = ARXIV_PIPELINE_URL,
        api_key: str = ARXIV_PIPELINE_API_KEY,
        timeout: int = HTTP_REQUEST_TIMEOUT,
    ):
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout

    @property
    def is_configured(self) -> bool:
        return bool(self._base_url)

    def rerank(
        self,
        profile_text: str,
        candidates: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        arxiv_pipeline /rerank 호출.

        입력:

        {
            "profile_text": "...",
            "candidates": [
                {
                    "arxiv_id": "...",
                    "title": "...",
                    "abstract_clean": "..."
                }
            ]
        }

        반환:

        [
            {
                "arxiv_id": "...",
                "title": "...",
                "abstract_clean": "...",
                "rerank_score": 0.87
            }
        ]
        """

        if not candidates:
            return []

        if not self.is_configured:
            raise RerankPipelineClientError(
                "RERANK_PIPELINE_URL이 설정되지 않았습니다."
            )

        url = f"{self._base_url}/rerank"

        headers = {
            "X-API-Key": self._api_key,
            "Content-Type": "application/json",
        }

        payload = {
            "profile_text": profile_text,
            "candidates": [
                {
                    "arxiv_id": c.get(
                        "arxiv_id",
                        "",
                    ),
                    "title": c.get(
                        "title",
                        "",
                    ),
                    "abstract_clean": c.get(
                        "abstract_clean",
                    ),
                }
                for c in candidates
            ],
        }

        try:
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=self._timeout,
            )

        except requests.RequestException as e:
            raise RerankPipelineClientError(
                f"rerank API 요청 실패: {e}"
            ) from e

        if response.status_code != 200:
            raise RerankPipelineClientError(
                f"rerank API 오류 "
                f"(status={response.status_code}): "
                f"{response.text}"
            )

        try:
            data = response.json()

        except ValueError as e:
            raise RerankPipelineClientError(
                f"rerank API 응답 JSON 파싱 실패: "
                f"{response.text}"
            ) from e

        results = data.get(
            "results",
            [],
        )

        if not isinstance(results, list):
            raise RerankPipelineClientError(
                "rerank API 응답의 results가 list가 아닙니다."
            )

        return results