"""
Stage 1a: 프로필 -> 입력 유효성 검사 + 영어 검색쿼리
          + 키워드 3~5개 + 제외조건 추출.

흐름:

1. Gemini로 사용자 프로필 분석
2. 입력이 실제 연구 관심사인지 판단
3. 지나치게 짧거나 명백한 테스트/무의미 문자열이면 invalid 처리
4. 유효한 경우:
   - 영어 embedding 검색 쿼리 생성
   - BM25 검색용 키워드 3~5개 생성
   - 제외조건 생성

중요:

- DF 검증은 사용하지 않는다.
- Gemini가 생성한 키워드는 그대로 Stage 1 /retrieve에 전달한다.
- invalid 입력은 빈 검색 조건으로 반환한다.
- 실제 retrieve 중단은 RecommendService에서 처리한다.
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


# -------------------------------------------------------------------
# Gemini REST API endpoint
# -------------------------------------------------------------------

_GEMINI_ENDPOINT = (
    "https://generativelanguage.googleapis.com/"
    "v1beta/models/{model}:generateContent"
)


# -------------------------------------------------------------------
# Gemini Prompt
# -------------------------------------------------------------------

_PROMPT = """
You are the input validation and query extraction component
of an English arXiv research-paper recommender.

Given a user's research-interest input, perform TWO tasks.

==================================================
STAGE 1: INPUT VALIDATION
==================================================

First determine whether the user's input contains a meaningful
research topic or research interest.

Set "valid" to false when the input is clearly unusable, such as:

- random characters
- meaningless Korean or English fragments
- keyboard mashing
- test strings
- nonsense strings
- repeated meaningless characters
- emoji-only input
- punctuation-only input
- extremely short input with no identifiable research meaning
- obvious placeholder text
- meaningless combinations of letters, numbers, or symbols
- strings such as "asdf", "qwer", "test", "hello" when they are not
  part of a meaningful research context

Examples that should be INVALID:

"ㅍㅅㅅ"

"ㅁ나ㅠ"

"asdf"

"qwer"

"test"

"ㅋㅋㅋㅋ"

"abc"

"123"

"!!!!"

"...."

"hello"

However, DO NOT reject a short input when it clearly identifies
a research topic or technical concept.

Examples that should be VALID:

"RAG"

"LLM"

"robotics"

"robot manipulation"

"LLM evaluation"

"3D object detection"

"BEV"

"BEV perception"

"vision-language-action"

"diffusion models"

"reinforcement learning"

The important rule is:

SHORT but MEANINGFUL -> valid = true

SHORT and MEANINGLESS -> valid = false

Do not infer a research topic from meaningless input.

Do not transform random text into a plausible research topic.

Do not guess what the user probably meant.

==================================================
STAGE 2: QUERY EXTRACTION
==================================================

Only when "valid" is true, generate the following.

1. "profile_text"

Create a faithful and concise English representation of the
user's research interest.

This text will be used as an embedding search query over
English arXiv abstracts.

Preserve the specific research focus and important nuances.

DO NOT introduce research topics that are not supported by
the user's input.

2. "keywords"

Generate 3-5 English keywords or technical phrases for BM25 search.

Prefer specific technical phrases over broad field names.

Good examples:

- "robot manipulation"
- "vision-language-action"
- "language instruction following"
- "language-conditioned manipulation"
- "3D object detection"
- "BEV perception"
- "citation grounding"
- "hallucination detection"

Avoid overly generic terms such as:

- "AI"
- "machine learning"
- "deep learning"
- "computer science"
- "technology"

Do NOT invent technical concepts that are not supported by
the user's input.

3. "exclusion"

Generate a short English phrase describing what should be excluded
from recommendation.

If there is no exclusion condition, return an empty string.

==================================================
IMPORTANT RULES
==================================================

- Do not hallucinate research interests.
- Do not guess what the user probably meant.
- Do not repair meaningless text into a meaningful topic.
- Short technical terms are allowed.
- Acronyms commonly used in research are allowed.
- A single valid research keyword can be enough to mark the input valid.
- Meaningless strings must be invalid.
- Invalid input must never produce keywords.
- Invalid input must never produce profile_text.
- Invalid input must return an empty exclusion.
- Do not recommend papers for invalid input.

==================================================
OUTPUT FORMAT
==================================================

Return ONLY valid JSON.

The JSON must contain exactly these keys:

{{
  "valid": true,
  "profile_text": "English search query",
  "keywords": [
    "keyword 1",
    "keyword 2",
    "keyword 3"
  ],
  "exclusion": "exclusion condition"
}}

For invalid input, return:

{{
  "valid": false,
  "profile_text": "",
  "keywords": [],
  "exclusion": ""
}}

For invalid input, DO NOT generate keywords.

Researcher input:

{body}
"""


# -------------------------------------------------------------------
# Exception
# -------------------------------------------------------------------

class KeywordExtractionServiceError(Exception):
    """Gemini 호출 또는 응답 파싱 실패."""


# -------------------------------------------------------------------
# Service
# -------------------------------------------------------------------

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

        현재 DF 검증은 사용하지 않는다.
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
        사용자 프로필을 Gemini로 분석한다.

        반환:

        valid
        profile_text_en
        keywords
        exclude
        """

        # ------------------------------------------------------------
        # 기본 입력 검사
        # ------------------------------------------------------------

        if not profile or not profile.strip():
            raise KeywordExtractionServiceError(
                "프로필이 비어 있습니다."
            )

        if not self._api_key:
            raise KeywordExtractionServiceError(
                "GEMINI_API_KEY가 설정되지 않았습니다."
            )

        # ------------------------------------------------------------
        # Gemini 호출
        # ------------------------------------------------------------

        raw_text = await self._generate(
            profile
        )

        # ------------------------------------------------------------
        # 응답 파싱
        # ------------------------------------------------------------

        (
            valid,
            profile_text_en,
            candidate_keywords,
            exclusion,
        ) = self._parse(
            raw_text
        )

        # ------------------------------------------------------------
        # INVALID INPUT
        # ------------------------------------------------------------

        if not valid:

            print(
                "[KeywordExtractionService] "
                f"INVALID 입력: {profile!r}"
            )

            return ExtractedProfile(
                valid=False,
                profile_text_en="",
                keywords=[],
                exclude=[],
            )

        # ------------------------------------------------------------
        # VALID INPUT
        # ------------------------------------------------------------

        print(
            "[KeywordExtractionService] "
            f"VALID 입력: {profile!r}"
        )

        print(
            "[KeywordExtractionService] "
            f"profile_text_en={profile_text_en}"
        )

        print(
            "[KeywordExtractionService] "
            f"keywords={candidate_keywords}"
        )

        print(
            "[KeywordExtractionService] "
            f"exclusion={exclusion}"
        )

        return ExtractedProfile(
            valid=True,
            profile_text_en=(
                profile_text_en
                or profile.strip()
            ),
            keywords=candidate_keywords,
            exclude=(
                [exclusion]
                if exclusion
                else []
            ),
        )

    # ----------------------------------------------------------------
    # Gemini 호출
    # ----------------------------------------------------------------

    async def _generate(
        self,
        profile: str,
    ) -> str:
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
                f"HTTP {response.status_code} - "
                f"{response.text}"
            )

        # ------------------------------------------------------------
        # Gemini 응답 JSON 파싱
        # ------------------------------------------------------------

        try:

            data = response.json()

        except ValueError as e:

            raise KeywordExtractionServiceError(
                "Gemini 응답이 JSON 형식이 아닙니다."
            ) from e

        # ------------------------------------------------------------
        # candidates 추출
        # ------------------------------------------------------------

        try:

            candidates = data["candidates"]

            if not candidates:

                raise KeywordExtractionServiceError(
                    f"Gemini 응답에 candidates가 없습니다: {data}"
                )

            text = (
                candidates[0]
                ["content"]
                ["parts"][0]
                ["text"]
            )

        except (
            KeyError,
            IndexError,
            TypeError,
        ) as e:

            raise KeywordExtractionServiceError(
                "Gemini 응답 형식이 예상과 다릅니다: "
                f"{data}"
            ) from e

        return text

    # ----------------------------------------------------------------
    # Gemini JSON 파싱
    # ----------------------------------------------------------------

    @staticmethod
    def _parse(
        raw_text: str,
    ) -> Tuple[bool, str, List[str], str]:
        """
        Gemini 응답을 안전하게 파싱한다.

        반환:

            (
                valid,
                profile_text,
                keywords,
                exclusion,
            )
        """

        # ------------------------------------------------------------
        # 빈 응답
        # ------------------------------------------------------------

        if not raw_text:

            return (
                False,
                "",
                [],
                "",
            )

        cleaned = raw_text.strip()

        # ------------------------------------------------------------
        # Markdown code fence 제거
        # ------------------------------------------------------------

        if cleaned.startswith("```json"):

            cleaned = cleaned[
                len("```json"):
            ].strip()

        elif cleaned.startswith("```"):

            cleaned = cleaned[
                len("```"):
            ].strip()

        if cleaned.endswith("```"):

            cleaned = cleaned[:-3].strip()

        # ------------------------------------------------------------
        # JSON 직접 파싱
        # ------------------------------------------------------------

        try:

            parsed = json.loads(
                cleaned
            )

        except json.JSONDecodeError:

            # Gemini가 JSON 앞뒤에 설명을 붙였을 경우
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

                parsed = json.loads(
                    match.group(0)
                )

            except json.JSONDecodeError as e:

                raise KeywordExtractionServiceError(
                    "Gemini JSON 파싱에 실패했습니다: "
                    f"{raw_text}"
                ) from e

        # ------------------------------------------------------------
        # JSON object 확인
        # ------------------------------------------------------------

        if not isinstance(
            parsed,
            dict,
        ):

            raise KeywordExtractionServiceError(
                "Gemini 응답 JSON이 객체 형식이 아닙니다."
            )

        # ------------------------------------------------------------
        # valid
        # ------------------------------------------------------------

        valid_value = parsed.get(
            "valid",
            False,
        )

        if isinstance(
            valid_value,
            bool,
        ):

            valid = valid_value

        elif isinstance(
            valid_value,
            str,
        ):

            valid = (
                valid_value
                .strip()
                .lower()
                in {
                    "true",
                    "yes",
                    "valid",
                    "1",
                }
            )

        elif isinstance(
            valid_value,
            int,
        ):

            valid = bool(
                valid_value
            )

        else:

            valid = False

        # ------------------------------------------------------------
        # INVALID이면 즉시 반환
        # ------------------------------------------------------------

        if not valid:

            return (
                False,
                "",
                [],
                "",
            )

        # ------------------------------------------------------------
        # profile_text
        # ------------------------------------------------------------

        profile_text = str(
            parsed.get(
                "profile_text",
                "",
            )
        ).strip()

        # ------------------------------------------------------------
        # keywords
        # ------------------------------------------------------------

        raw_keywords = parsed.get(
            "keywords",
            [],
        )

        if not isinstance(
            raw_keywords,
            list,
        ):

            raw_keywords = []

        keywords: List[str] = []

        for keyword in raw_keywords:

            if keyword is None:
                continue

            keyword = str(
                keyword
            ).strip()

            if not keyword:
                continue

            if keyword not in keywords:

                keywords.append(
                    keyword
                )

        # 최대 5개
        keywords = keywords[:5]

        # ------------------------------------------------------------
        # exclusion
        # ------------------------------------------------------------

        exclusion = str(
            parsed.get(
                "exclusion",
                "",
            )
        ).strip()

        # ------------------------------------------------------------
        # VALID인데 검색 정보가 전혀 없는 경우
        # ------------------------------------------------------------

        if not profile_text and not keywords:

            return (
                False,
                "",
                [],
                "",
            )

        return (
            True,
            profile_text,
            keywords,
            exclusion,
        )