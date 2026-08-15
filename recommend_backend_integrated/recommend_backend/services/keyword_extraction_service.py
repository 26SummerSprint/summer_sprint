"""
Stage 1a: 프로필 -> 영어 검색쿼리 + 키워드 3~5개 + 제외조건 추출.

흐름:
1. Gemini로 사용자 프로필을 분석
2. 영어 embedding 검색 쿼리 생성
3. BM25 검색용 키워드 3~5개 생성
4. 제외조건 생성

※ DF 검증은 현재 사용하지 않는다.
   추출된 키워드는 그대로 Stage 1 /retrieve에 전달한다.
"""

import json
import re
from typing import List, Optional, Tuple

import httpx

from ..schemas import ExtractedProfile
from ..config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    GEMINI_REQUEST_TIMEOUT,
)


# Gemini REST API endpoint
_GEMINI_ENDPOINT = (
    "https://generativelanguage.googleapis.com/"
    "v1beta/models/{model}:generateContent"
)


# -------------------------------------------------------------------
# Gemini Prompt
# -------------------------------------------------------------------
#
# 주의:
# .format(body=profile)을 사용하므로 JSON 예시의 { }는
# {{ }}로 작성해야 한다.
#
_PROMPT = """You are helping build a research-paper recommender over English arXiv abstracts.

Given a researcher's interest profile, produce:

1. "profile_text":
   A faithful, concise English translation/restatement of the researcher's
   interest profile. This will be used as an embedding search query over
   English arXiv abstracts.
   Preserve the specific research focus and important nuances.

2. "keywords":
   3-5 English keywords or technical phrases that best characterize the
   researcher's interests for BM25 keyword search.
   Prefer specific technical phrases over broad field names.
   For example:
   - "robot manipulation"
   - "vision-language-action"
   - "language instruction following"
   - "language-conditioned manipulation"

   Avoid overly generic terms such as:
   - "AI"
   - "machine learning"
   - "robotics"
   - "deep learning"

3. "exclusion":
   A short English phrase describing what should be excluded from the
   recommendation.
   If there is no exclusion condition, return an empty string.

Return ONLY valid JSON with exactly these keys:

{{
  "profile_text": "English search query",
  "keywords": [
    "keyword 1",
    "keyword 2",
    "keyword 3"
  ],
  "exclusion": "exclusion condition"
}}

Researcher profile:
{body}
"""


class KeywordExtractionServiceError(Exception):
    """Gemini 호출 또는 응답 파싱 실패."""


class KeywordExtractionService:
    def __init__(
        self,
        arxiv_client=None,
        api_key: str = GEMINI_API_KEY,
        model: str = GEMINI_MODEL,
        timeout: int = GEMINI_REQUEST_TIMEOUT,
    ):
        """
        arxiv_client는 기존 구조와의 호환성을 위해 유지한다.

        현재 DF 검증을 사용하지 않으므로 실제로는 사용하지 않는다.
        """
        self._arxiv_client = arxiv_client
        self._api_key = api_key
        self._model = model
        self._timeout = timeout

    # ----------------------------------------------------------------
    # Public API
    # ----------------------------------------------------------------

    async def extract(
        self,
        profile: str,
        category: Optional[str] = None,
    ) -> ExtractedProfile:
        """
        사용자 프로필을 Gemini로 분석하여
        영어 검색쿼리 / 키워드 / 제외조건을 반환한다.

        category는 기존 호출부와의 호환성을 위해 유지하지만
        현재는 사용하지 않는다.
        """

        if not profile or not profile.strip():
            raise KeywordExtractionServiceError(
                "프로필이 비어 있습니다."
            )

        if not self._api_key:
            raise KeywordExtractionServiceError(
                "GEMINI_API_KEY가 설정되지 않았습니다."
            )

        # Gemini 호출
        raw_text = await self._generate(profile)

        # JSON 파싱
        profile_text_en, candidate_keywords, exclusion = self._parse(
            raw_text
        )

        # ------------------------------------------------------------
        # 중요:
        # DF 검증을 하지 않는다.
        #
        # 기존:
        # validated_keywords = await self._filter_by_df(...)
        #
        # 현재:
        # Gemini가 생성한 키워드를 그대로 사용
        # ------------------------------------------------------------

        return ExtractedProfile(
            profile_text_en=profile_text_en or profile.strip(),
            keywords=candidate_keywords,
            exclude=[exclusion] if exclusion else [],
        )

    # ----------------------------------------------------------------
    # Gemini 호출
    # ----------------------------------------------------------------

    async def _generate(self, profile: str) -> str:
        """
        Gemini generateContent REST API 호출.
        """

        url = _GEMINI_ENDPOINT.format(
            model=self._model
        )

        prompt = _PROMPT.format(
            body=profile.strip()
        )

        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": prompt
                        }
                    ]
                }
            ]
        }

        async with httpx.AsyncClient(
            timeout=self._timeout
        ) as client:

            try:
                response = await client.post(
                    url,
                    params={
                        "key": self._api_key
                    },
                    json=payload,
                )

            except httpx.HTTPError as e:
                raise KeywordExtractionServiceError(
                    f"Gemini 키워드 추출 호출 실패: {e}"
                ) from e

        # ------------------------------------------------------------
        # HTTP 오류
        # ------------------------------------------------------------

        if response.status_code != 200:
            raise KeywordExtractionServiceError(
                "Gemini 키워드 추출 호출 실패: "
                f"HTTP {response.status_code} - {response.text}"
            )

        # ------------------------------------------------------------
        # JSON 응답 파싱
        # ------------------------------------------------------------

        try:
            data = response.json()

        except ValueError as e:
            raise KeywordExtractionServiceError(
                "Gemini 응답이 JSON 형식이 아닙니다."
            ) from e

        try:
            candidates = data["candidates"]

            if not candidates:
                raise KeywordExtractionServiceError(
                    f"Gemini 응답에 candidates가 없습니다: {data}"
                )

            text = candidates[0]["content"]["parts"][0]["text"]

        except (KeyError, IndexError, TypeError) as e:
            raise KeywordExtractionServiceError(
                f"Gemini 응답 형식이 예상과 다릅니다: {data}"
            ) from e

        return text

    # ----------------------------------------------------------------
    # Gemini JSON 파싱
    # ----------------------------------------------------------------

    @staticmethod
    def _parse(
        raw_text: str,
    ) -> Tuple[str, List[str], str]:
        """
        Gemini가 반환한 JSON을 안전하게 파싱한다.

        ```json
        {
          "profile_text": "...",
          "keywords": [...],
          "exclusion": "..."
        }
        ```

        형태도 처리한다.
        """

        if not raw_text:
            return "", [], ""

        cleaned = raw_text.strip()

        # ------------------------------------------------------------
        # Markdown code fence 제거
        # ------------------------------------------------------------

        if cleaned.startswith("```json"):
            cleaned = cleaned[len("```json"):].strip()

        elif cleaned.startswith("```"):
            cleaned = cleaned[len("```"):].strip()

        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].strip()

        # ------------------------------------------------------------
        # JSON 파싱
        # ------------------------------------------------------------

        try:
            parsed = json.loads(cleaned)

        except json.JSONDecodeError:

            # Gemini가 앞뒤에 설명을 붙였을 경우
            match = re.search(
                r"\{.*\}",
                cleaned,
                flags=re.DOTALL,
            )

            if not match:
                raise KeywordExtractionServiceError(
                    "Gemini 응답에서 JSON을 찾을 수 없습니다: "
                    f"{raw_text}"
                )

            try:
                parsed = json.loads(match.group(0))

            except json.JSONDecodeError as e:
                raise KeywordExtractionServiceError(
                    "Gemini JSON 파싱에 실패했습니다: "
                    f"{raw_text}"
                ) from e

        # ------------------------------------------------------------
        # profile_text
        # ------------------------------------------------------------

        profile_text = str(
            parsed.get("profile_text", "")
        ).strip()

        # ------------------------------------------------------------
        # keywords
        # ------------------------------------------------------------

        raw_keywords = parsed.get(
            "keywords",
            []
        )

        if not isinstance(raw_keywords, list):
            raw_keywords = []

        keywords: List[str] = []

        for keyword in raw_keywords:
            keyword = str(keyword).strip()

            if keyword and keyword not in keywords:
                keywords.append(keyword)

        # 3~5개를 권장하지만 Gemini 응답을 강제로 자르지는 않는다.
        # 단, 지나치게 많은 키워드가 오는 경우 최대 5개만 사용한다.
        keywords = keywords[:5]

        # ------------------------------------------------------------
        # exclusion
        # ------------------------------------------------------------

        exclusion = str(
            parsed.get("exclusion", "")
        ).strip()

        return (
            profile_text,
            keywords,
            exclusion,
        )