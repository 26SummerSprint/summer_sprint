"""
Ground Truth Labeler API Client.

다른 프로젝트에서 Ground Truth Labeling API를 호출하기 위한 클라이언트.

사용 예:
    client = GroundTruthClient("http://localhost:8300")

    result = client.label(
        profile="RAG 연구, Citation/Grounding 관심",
        title="Paper Title",
        abstract="Paper abstract..."
    )

    print(result["label"])
"""

from typing import Any, Dict, List, Optional

import requests


class GroundTruthClientError(Exception):
    """Ground Truth Labeler API 호출 실패 예외."""


class GroundTruthClient:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8300",
        api_key: Optional[str] = None,
        timeout: int = 120,
    ):
        """
        Args:
            base_url:
                Ground Truth Labeler API 주소

            api_key:
                서버에서 GROUND_TRUTH_LABELER_API_KEY를 설정한 경우 필요

            timeout:
                HTTP 요청 timeout (초)
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def _headers(self) -> Dict[str, str]:
        headers = {
            "Content-Type": "application/json",
        }

        if self.api_key:
            headers["X-API-Key"] = self.api_key

        return headers

    def health(self) -> Dict[str, Any]:
        """
        API 서버 상태 확인.
        """
        try:
            response = requests.get(
                f"{self.base_url}/health",
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response.json()

        except requests.RequestException as e:
            raise GroundTruthClientError(
                f"Health check 실패: {e}"
            ) from e

    def label(
        self,
        profile: str,
        title: str,
        abstract: str = "",
    ) -> Dict[str, Any]:
        """
        논문 1개 Ground Truth Labeling.

        Returns:
            {
                "score": 90,
                "decision": "정답",
                "label": 1,
                "tag": "none",
                "reason": "...",
                "parse_warnings": []
            }
        """

        payload = {
            "profile": profile,
            "title": title,
            "abstract": abstract,
        }

        try:
            response = requests.post(
                f"{self.base_url}/label",
                headers=self._headers(),
                json=payload,
                timeout=self.timeout,
            )

            response.raise_for_status()
            return response.json()

        except requests.RequestException as e:
            raise GroundTruthClientError(
                f"Label API 호출 실패: {e}"
            ) from e

    def label_batch(
        self,
        profile: str,
        papers: List[Dict[str, Any]],
        keep_raw: bool = False,
        title_key: str = "title",
        abstract_key: str = "abstract_clean",
        id_key: str = "arxiv_id",
    ) -> Dict[str, Any]:
        """
        여러 논문 Batch Ground Truth Labeling.

        papers 예:
        [
            {
                "arxiv_id": "2401.00001",
                "title": "...",
                "abstract_clean": "..."
            }
        ]

        Returns:
            {
                "count": 100,
                "success_count": 100,
                "positive_count": 30,
                "results": [...]
            }
        """

        payload = {
            "profile": profile,
            "papers": papers,
            "keep_raw": keep_raw,
            "title_key": title_key,
            "abstract_key": abstract_key,
            "id_key": id_key,
        }

        try:
            response = requests.post(
                f"{self.base_url}/label/batch",
                headers=self._headers(),
                json=payload,
                timeout=self.timeout,
            )

            response.raise_for_status()
            return response.json()

        except requests.RequestException as e:
            raise GroundTruthClientError(
                f"Batch Label API 호출 실패: {e}"
            ) from e