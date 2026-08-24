"""
'이 논문으로 다시 추천받기' 기능 전용 키워드 추출 서비스.

기존 KeywordExtractionService(Stage 1a)는 "사용자 프로필"을 분석해서
(영어 검색쿼리 + 키워드 + 제외조건) 세 가지를 만들어내지만, 이 서비스는
그와 별개로 "선택된 논문 한 편의 Abstract"만 분석해서 핵심 연구 키워드만
뽑아낸다. 제외조건도, 프로필 번역도 필요 없고 목적이 다르므로 기존
프롬프트를 재사용하지 않고 이 서비스 전용 프롬프트를 둔다.

출력 예:
    ["citation verification", "citation correctness", "evidence retrieval"]

주의:
- 논문의 핵심 연구 주제/방법/Task 중심으로만 추출한다.
- Abstract의 일반적인 단어(예: method, model, approach, performance)는
  추출하지 않는다.
"""

import json
import re
from typing import List

import httpx

from ..config import GEMINI_API_KEY, GEMINI_MODEL, GEMINI_REQUEST_TIMEOUT

_GEMINI_ENDPOINT = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
)

# 주의: .format(abstract=...)를 사용하므로 JSON 예시의 { }는 {{ }}로 작성해야 한다.
_PROMPT = """You are analyzing a single research paper abstract to extract its core research keywords.

Extract 3-5 English keywords or short technical phrases that capture the paper's core research
topic, method, or task - the specific things this paper is actually about.

Rules:
- Focus on the paper's central contribution, method, or task (not background, motivation, or
  general context).
- Prefer specific 2-3 word technical phrases (e.g. "citation verification", "speculative decoding")
  over broad field names (e.g. "natural language processing", "machine learning", "AI", "deep
  learning").
- Do not extract generic words that could describe almost any paper (e.g. "method", "model",
  "approach", "results", "performance", "framework").
- Do not repeat near-duplicate keywords.

Return ONLY valid JSON with exactly this shape:

{{
  "keywords": [
    "keyword 1",
    "keyword 2",
    "keyword 3"
  ]
}}

Abstract:
{abstract}
"""


class PaperKeywordExtractionServiceError(Exception):
    """Gemini 호출 또는 응답 파싱 실패."""


class PaperKeywordExtractionService:
    def __init__(
        self,
        api_key: str = GEMINI_API_KEY,
        model: str = GEMINI_MODEL,
        timeout: int = GEMINI_REQUEST_TIMEOUT,
    ):
        self._api_key = api_key
        self._model = model
        self._timeout = timeout

    # ----------------------------------------------------------------
    # Public API
    # ----------------------------------------------------------------

    async def extract(self, abstract: str) -> List[str]:
        """논문 Abstract에서 핵심 연구 키워드 3~5개를 추출한다."""

        if not abstract or not abstract.strip():
            raise PaperKeywordExtractionServiceError(
                "Abstract가 비어 있어 키워드를 추출할 수 없습니다."
            )

        if not self._api_key:
            raise PaperKeywordExtractionServiceError(
                "GEMINI_API_KEY가 설정되지 않았습니다."
            )

        raw_text = await self._generate(abstract.strip())
        return self._parse(raw_text)

    # ----------------------------------------------------------------
    # Gemini 호출
    # ----------------------------------------------------------------

    async def _generate(self, abstract: str) -> str:
        url = _GEMINI_ENDPOINT.format(model=self._model)
        prompt = _PROMPT.format(abstract=abstract)
        payload = {"contents": [{"parts": [{"text": prompt}]}]}

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            try:
                response = await client.post(
                    url, params={"key": self._api_key}, json=payload
                )
            except httpx.HTTPError as e:
                raise PaperKeywordExtractionServiceError(
                    f"Gemini 키워드 추출 호출 실패: {e}"
                ) from e

        if response.status_code != 200:
            raise PaperKeywordExtractionServiceError(
                f"Gemini 키워드 추출 호출 실패: HTTP {response.status_code} - {response.text}"
            )

        try:
            data = response.json()
            candidates = data["candidates"]
            if not candidates:
                raise PaperKeywordExtractionServiceError(
                    f"Gemini 응답에 candidates가 없습니다: {data}"
                )
            text = candidates[0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError, TypeError, ValueError) as e:
            raise PaperKeywordExtractionServiceError(
                f"Gemini 응답 형식이 예상과 다릅니다: {e}"
            ) from e

        return text

    # ----------------------------------------------------------------
    # 응답 파싱
    # ----------------------------------------------------------------

    @staticmethod
    def _parse(raw_text: str) -> List[str]:
        if not raw_text:
            return []

        cleaned = raw_text.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[len("```json"):].strip()
        elif cleaned.startswith("```"):
            cleaned = cleaned[len("```"):].strip()
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].strip()

        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
            if not match:
                raise PaperKeywordExtractionServiceError(
                    f"Gemini 응답에서 JSON을 찾을 수 없습니다: {raw_text}"
                )
            try:
                parsed = json.loads(match.group(0))
            except json.JSONDecodeError as e:
                raise PaperKeywordExtractionServiceError(
                    f"Gemini JSON 파싱에 실패했습니다: {raw_text}"
                ) from e

        raw_keywords = parsed.get("keywords", [])
        if not isinstance(raw_keywords, list):
            raw_keywords = []

        keywords: List[str] = []
        for kw in raw_keywords:
            kw = str(kw).strip()
            if kw and kw not in keywords:
                keywords.append(kw)

        return keywords[:5]
