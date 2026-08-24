"""
Stage 3: 최종 논문 선정 서비스.

Reranker에서 압축된 후보 논문들을 Gemini에게 전달하고,
사용자 프로필과 후보 논문의 실제 연구 내용을 종합적으로 판단하여
최종 추천 논문 최대 10편을 선정한다.

Gemini 출력 형식:

## Final Recommendations

### 1. [논문 제목]
- **arXiv ID:** 2401.12345
- **Link:** [https://arxiv.org/abs/2401.12345](https://arxiv.org/abs/2401.12345)
- **추천 이유:** ...

내부 판단 과정과 점수는 사용자에게 출력하지 않는다.
"""

import re
from dataclasses import dataclass
from typing import Any, Dict, List

import httpx

from ..config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    GEMINI_REQUEST_TIMEOUT,
)


# ============================================================
# Gemini API Endpoint
# ============================================================

_GEMINI_ENDPOINT = (
    "https://generativelanguage.googleapis.com/"
    "v1beta/models/{model}:generateContent"
)


# ============================================================
# Gemini Prompt
# ============================================================

_PROMPT_TEMPLATE = """
당신은 사용자의 연구 관심 프로필을 기반으로 여러 후보 논문 중
가장 적합한 논문을 선별하는 전문 연구 논문 추천 시스템이다.

사용자는 자신의 연구 관심 분야를 프로필 형태로 정의해 두었다.

프로필에는 다음 정보가 포함될 수 있다.

- Core Topic: 반드시 핵심적으로 다루어야 하는 주제
- AND Groups: 모두 만족해야 하는 조건
- OR Groups: 여러 조건 중 하나 이상 만족해야 하는 조건
- Exclusion Conditions: 논문에서 핵심적으로 다루면 제외해야 하는 주제
- 연구의 세부 관심 분야 및 선호 조건

입력으로 하나의 프로필과 여러 개의 후보 논문이 주어진다.

각 논문에는 다음 정보가 포함될 수 있다.

- arXiv ID
- 제목
- 초록
- 링크


============================================================
1. 핵심 판단 원칙
============================================================

단순한 키워드 매칭으로 판단하지 않는다.

Ground Truth Labeler와 유사하게 각 후보 논문의 실제 연구 내용,
연구 목적, 방법론, 실험 대상 및 사용자의 연구 관심 분야를
종합적으로 비교한다.

특히 다음을 구분한다.

- 키워드가 제목이나 초록에 단순히 등장하는 경우
- 해당 개념이 논문의 일부로 사용되는 경우
- 해당 개념이 논문의 핵심 연구 주제인 경우

키워드가 많이 등장한다고 해서 높은 적합도를 부여하지 않는다.

논문의 제목과 초록을 바탕으로 실제 연구 목적과 방법론을 파악하고
사용자의 연구 관심과 실질적으로 얼마나 가까운지를 판단한다.


============================================================
2. Core Topic
============================================================

논문의 핵심 연구 내용이 사용자의 Core Topic과 직접적으로 관련되는지 판단한다.

다음과 같이 판단한다.

- 핵심:
  논문의 주요 연구 목적, 방법론 또는 실험이 Core Topic과 직접 일치한다.

- 일부:
  관련 내용은 있지만 논문의 핵심 연구 주제는 아니다.

- 없음:
  실질적인 관련성이 없다.

단순 키워드 등장만으로 핵심이라고 판단하지 않는다.


============================================================
3. AND Groups
============================================================

프로필에 AND 조건이 존재한다면 중요한 AND 조건을 모두 확인한다.

각 조건이 단순히 언급되는 것이 아니라
논문의 핵심 연구 내용과 연결되어 있는지 판단한다.

중요한 AND 조건을 만족하지 못하는 논문은
추천 우선순위를 크게 낮춘다.


============================================================
4. OR Groups
============================================================

OR 조건은 여러 조건 중 하나 이상이 핵심적으로 충족되는지 확인한다.

하나 이상의 OR 조건을 강하게 만족하면 긍정적으로 평가한다.


============================================================
5. Exclusion Conditions
============================================================

제외 조건을 반드시 확인한다.

다음 두 경우를 구분한다.

1. 단순 언급:
   관련 개념이 배경, 비교 대상 또는 일부 실험으로 등장하는 경우.

2. 핵심 위반:
   제외 조건에 해당하는 주제가 논문의 핵심 연구 대상,
   핵심 방법론 또는 주요 실험인 경우.

Exclusion Condition을 핵심적으로 위반하는 논문은
최종 추천에서 제외한다.

단순 언급만으로 논문 전체를 제외하지 않는다.


============================================================
6. 종합 적합도
============================================================

각 후보 논문을 다음 기준으로 종합적으로 판단한다.

1. Core Topic과의 직접적인 관련성
2. AND 조건 충족 여부
3. OR 조건 충족 여부
4. Exclusion Condition 위반 여부
5. 논문의 실제 연구 목적과 사용자 관심의 일치 정도
6. 핵심 방법론과 사용자 관심의 일치 정도
7. 사용자의 실제 연구에 참고할 가치

특히 다음 질문을 기준으로 판단한다.

"이 논문이 실제로 사용자가 관심 있는 연구를 수행하고 있는가?"

키워드 개수 자체는 중요한 판단 기준이 아니다.


============================================================
7. 내부 적합도
============================================================

각 논문에 대해 내부적으로 0~100점의 적합도를 판단할 수 있다.

대략적인 기준:

90~100:
사용자의 연구 관심과 매우 직접적으로 일치

80~89:
높은 관련성이 있으며 추천 가치가 높음

70~79:
상당한 관련성이 있지만 일부 조건이 약함

50~69:
부분적으로 관련되지만 핵심 관심과는 거리가 있음

0~49:
관련성이 낮거나 핵심적인 제외 조건에 해당

단, 실제 연구 내용의 적합성을 점수보다 우선한다.

핵심적인 Exclusion Condition 위반 논문은
높은 적합도로 판단하지 않는다.


============================================================
8. 최종 논문 선정
============================================================

모든 후보 논문을 비교한 뒤 가장 적합한 논문을 최대 10개 선정한다.

후보 논문이 10개보다 적다면 적합한 논문만 반환한다.

반드시 10개를 채우기 위해 관련성이 낮은 논문을 억지로 포함하지 않는다.

다음 우선순위를 사용한다.

1. Core Topic 직접 일치
2. AND 조건 충족
3. OR 조건 중 중요한 관심 분야 충족
4. Exclusion Condition 위반 없음
5. 실제 연구 참고 가치

비슷한 논문이라면 사용자 연구 관심과
더 직접적으로 연결되는 논문을 우선한다.


============================================================
9. 중복 논문
============================================================

동일하거나 사실상 동일한 논문이 여러 번 등장하면 하나만 선택한다.

가능하면 arXiv ID를 기준으로 중복을 제거한다.


============================================================
10. 후보 외 논문 금지
============================================================

매우 중요한 규칙이다.

최종 추천 논문은 반드시 입력으로 제공된
Candidate Papers 안에 존재해야 한다.

입력 후보에 없는 논문을 새로 만들어내거나 추천하지 않는다.

arXiv ID가 후보 목록에 존재하지 않는 경우 해당 논문은 무효로 처리한다.

가능하면 입력 후보에 제공된 제목과 링크를 그대로 사용한다.


============================================================
11. 최종 출력
============================================================

사용자에게 내부 판단 과정을 절대로 출력하지 않는다.

출력하지 않는 정보:

- 내부 분석 과정
- Core Topic 판단
- AND Groups 판단
- OR Groups 판단
- Exclusion Conditions 판단
- 적합도 점수
- 내부 순위 계산 과정
- 탈락한 논문 목록
- 후보 논문별 상세 평가
- JSON
- Markdown 표

사용자에게는 오직 다음 정보만 제공한다.

1. 순위
2. 논문 제목
3. arXiv ID
4. 논문 링크
5. 추천 이유


============================================================
12. 출력 형식
============================================================

반드시 다음 형식을 사용한다.

## Final Recommendations

### 1. [논문 제목]

- **arXiv ID:** 2401.12345
- **Link:** [https://arxiv.org/abs/2401.12345](https://arxiv.org/abs/2401.12345)
- **추천 이유:** 사용자의 핵심 관심 분야와 직접적으로 연결되며, 주요 방법론과 실험이 해당 연구 관심과 밀접하게 일치한다.

### 2. [논문 제목]

- **arXiv ID:** 2402.12345
- **Link:** [https://arxiv.org/abs/2402.12345](https://arxiv.org/abs/2402.12345)
- **추천 이유:** ...

최대 10개까지만 출력한다.

추천 이유는 각 논문당 1~3문장으로 작성한다.

추천 이유에는 내부 점수나 내부 판단 과정을 포함하지 않는다.


============================================================
13. 최종 확인
============================================================

최종 출력 전에 반드시 다음을 확인한다.

- 추천 논문이 Candidate Papers에 실제로 존재하는가?
- arXiv ID가 정확한가?
- 동일 논문이 중복되지 않았는가?
- 최대 10개인가?
- 각 논문에 제목이 있는가?
- 각 논문에 arXiv ID가 있는가?
- 각 논문에 링크가 있는가?
- 각 논문에 추천 이유가 있는가?
- 내부 분석이나 점수를 출력하지 않았는가?


============================================================
입력
============================================================

## User Profile

{profile}

## Candidate Papers

{papers}

위 프로필과 후보 논문을 종합적으로 평가하고,
가장 적합한 논문을 최대 10개 선정하여
지정된 형식으로 출력하라.
"""


# ============================================================
# Markdown Parser
# ============================================================

# 중요:
# re.VERBOSE를 사용하지 않는다.
# '#'가 포함된 Markdown heading을 그대로 처리하기 위함이다.

_BLOCK_RE = re.compile(
    r"^###\s*(\d+)\.\s*(.+?)\s*$"
    r"(.*?)"
    r"(?=^###\s*\d+\.\s*|\Z)",
    re.MULTILINE | re.DOTALL,
)


_ARXIV_ID_RE = re.compile(
    r"\*\*arXiv\s*ID:\*\*\s*(\S+)",
    re.IGNORECASE,
)


_LINK_RE = re.compile(
    r"\*\*Link:\*\*\s*"
    r"(?:\[([^\]]+)\]\(([^)]+)\)|(\S+))",
    re.IGNORECASE,
)


_REASON_RE = re.compile(
    r"\*\*추천\s*이유:\*\*\s*"
    r"(.*?)(?=\n\s*-\s*\*\*|\Z)",
    re.DOTALL,
)


# ============================================================
# Exception
# ============================================================

class FinalSelectionServiceError(Exception):
    """Gemini 최종 선정 과정에서 발생하는 예외."""


# ============================================================
# Result Dataclass
# ============================================================

@dataclass
class FinalRecommendation:

    rank: int

    title: str

    arxiv_id: str

    link: str

    reason: str


# ============================================================
# Final Selection Service
# ============================================================

class FinalSelectionService:
    """
    Stage 3.

    Reranker에서 압축된 후보 논문을 Gemini에게 전달하고
    최종 추천 논문을 최대 10편 선정한다.
    """

    def __init__(
        self,
        api_key: str = GEMINI_API_KEY,
        model: str = GEMINI_MODEL,
        timeout: int = GEMINI_REQUEST_TIMEOUT,
    ):

        self._api_key = api_key

        self._model = model

        self._timeout = timeout


    # ========================================================
    # Public
    # ========================================================

    async def select(
        self,
        profile: str,
        candidates: List[Dict[str, Any]],
    ) -> List[FinalRecommendation]:

        if not candidates:
            return []

        if not self._api_key:

            raise FinalSelectionServiceError(
                "GEMINI_API_KEY가 설정되지 않았습니다. "
                '환경변수로 주입해주세요: '
                'GEMINI_API_KEY="..."'
            )

        prompt = _PROMPT_TEMPLATE.format(
            profile=profile,
            papers=self._build_papers_block(
                candidates
            ),
        )

        raw_text = await self._generate(
            prompt
        )

        return self._parse_recommendations(
            raw_text,
            candidates,
        )


    # ========================================================
    # Gemini API
    # ========================================================

    async def _generate(
        self,
        prompt: str,
    ) -> str:

        url = _GEMINI_ENDPOINT.format(
            model=self._model
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
            ],
            "generationConfig": {
                "temperature": 0.2,
            },
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

                response.raise_for_status()

                data = response.json()

                candidates_data = data.get(
                    "candidates"
                )

                if not candidates_data:

                    raise FinalSelectionServiceError(
                        "Gemini 응답에 candidates가 없습니다."
                    )

                first_candidate = (
                    candidates_data[0]
                )

                content = first_candidate.get(
                    "content"
                )

                if not content:

                    raise FinalSelectionServiceError(
                        "Gemini 응답에 content가 없습니다."
                    )

                parts = content.get(
                    "parts"
                )

                if not parts:

                    raise FinalSelectionServiceError(
                        "Gemini 응답에 parts가 없습니다."
                    )

                text = parts[0].get(
                    "text"
                )

                if not text:

                    raise FinalSelectionServiceError(
                        "Gemini 응답에 text가 없습니다."
                    )

                return text

            except FinalSelectionServiceError:

                raise

            except httpx.HTTPStatusError as e:

                detail = ""

                try:
                    detail = e.response.text[:1000]
                except Exception:
                    pass

                raise FinalSelectionServiceError(
                    "Gemini API HTTP 오류: "
                    f"{e.response.status_code} "
                    f"{detail}"
                ) from e

            except httpx.HTTPError as e:

                raise FinalSelectionServiceError(
                    f"Gemini API 호출 실패: {e}"
                ) from e

            except (
                KeyError,
                IndexError,
                TypeError,
            ) as e:

                raise FinalSelectionServiceError(
                    f"Gemini 응답 파싱 실패: {e}"
                ) from e


    # ========================================================
    # Candidate Papers Block
    # ========================================================

    @staticmethod
    def _build_papers_block(
        candidates: List[Dict[str, Any]],
    ) -> str:

        parts: List[str] = []

        for candidate in candidates:

            arxiv_id = str(
                candidate.get(
                    "arxiv_id",
                    ""
                ) or ""
            ).strip()

            title = str(
                candidate.get(
                    "title",
                    ""
                ) or ""
            ).strip()

            abstract = str(
                candidate.get(
                    "abstract_clean",
                    ""
                ) or ""
            ).strip()

            # Gemini에 너무 긴 초록을 보내지 않도록 제한
            abstract = abstract[:1000]

            link = str(
                candidate.get(
                    "abs_url",
                    ""
                ) or ""
            ).strip()

            if not link and arxiv_id:

                link = (
                    "https://arxiv.org/abs/"
                    f"{arxiv_id}"
                )

            parts.append(
                f"- arXiv ID: {arxiv_id}\n"
                f"  제목: {title}\n"
                f"  초록: {abstract}\n"
                f"  링크: {link}"
            )

        return "\n\n".join(
            parts
        )


    # ========================================================
    # Parse Gemini Markdown
    # ========================================================

    @staticmethod
    def _parse_recommendations(
        raw_text: str,
        candidates: List[Dict[str, Any]],
    ) -> List[FinalRecommendation]:

        if not raw_text:

            return []

        body = raw_text.strip()

        # ----------------------------------------------------
        # Markdown code fence 제거
        # ----------------------------------------------------

        body = re.sub(
            r"^```(?:markdown)?\s*",
            "",
            body,
            flags=re.IGNORECASE,
        )

        body = re.sub(
            r"\s*```$",
            "",
            body,
        )

        # ----------------------------------------------------
        # Final Recommendations 찾기
        # ----------------------------------------------------

        marker = re.search(
            r"##\s*Final\s+Recommendations",
            body,
            re.IGNORECASE,
        )

        if marker:

            body = body[
                marker.end():
            ]


        # ----------------------------------------------------
        # 후보 arXiv ID → Candidate
        # ----------------------------------------------------

        by_arxiv_id: Dict[
            str,
            Dict[str, Any],
        ] = {}

        for candidate in candidates:

            arxiv_id = str(
                candidate.get(
                    "arxiv_id",
                    ""
                ) or ""
            ).strip()

            if arxiv_id:

                by_arxiv_id[
                    arxiv_id
                ] = candidate


        # ----------------------------------------------------
        # Block Parsing
        # ----------------------------------------------------

        results: List[
            FinalRecommendation
        ] = []

        used_ids = set()

        for match in _BLOCK_RE.finditer(
            body
        ):

            try:

                rank = int(
                    match.group(1)
                )

            except ValueError:

                continue


            title = (
                match.group(2)
                .strip()
            )

            block_text = (
                match.group(3)
                .strip()
            )


            # ------------------------------------------------
            # 제목 [ ] 제거
            # ------------------------------------------------

            title_match = re.match(
                r"^\[(.*)\]$",
                title,
                re.DOTALL,
            )

            if title_match:

                title = (
                    title_match.group(1)
                    .strip()
                )


            # ------------------------------------------------
            # arXiv ID
            # ------------------------------------------------

            arxiv_match = (
                _ARXIV_ID_RE.search(
                    block_text
                )
            )

            if not arxiv_match:

                continue

            arxiv_id = (
                arxiv_match.group(1)
                .strip()
            )

            # Markdown / punctuation 제거
            arxiv_id = (
                arxiv_id
                .strip("`")
                .strip()
                .rstrip(".,)")
            )


            # ------------------------------------------------
            # 후보에 실제 존재하는지 확인
            # ------------------------------------------------

            candidate = by_arxiv_id.get(
                arxiv_id
            )

            if candidate is None:

                # Gemini가 후보 외 논문을 생성하면 무시
                continue


            # ------------------------------------------------
            # 중복 제거
            # ------------------------------------------------

            if arxiv_id in used_ids:

                continue

            used_ids.add(
                arxiv_id
            )


            # ------------------------------------------------
            # Link
            # ------------------------------------------------

            link = ""

            link_match = _LINK_RE.search(
                block_text
            )

            if link_match:

                # Markdown link:
                # [text](url)
                link = (
                    link_match.group(2)
                    or link_match.group(3)
                    or ""
                ).strip()


            # Link가 없으면 candidate 정보 사용
            if not link:

                link = str(
                    candidate.get(
                        "abs_url",
                        ""
                    ) or ""
                ).strip()


            # 그래도 없으면 arXiv URL 생성
            if not link:

                link = (
                    "https://arxiv.org/abs/"
                    f"{arxiv_id}"
                )


            # ------------------------------------------------
            # 추천 이유
            # ------------------------------------------------

            reason = ""

            reason_match = (
                _REASON_RE.search(
                    block_text
                )
            )

            if reason_match:

                reason = (
                    reason_match.group(1)
                    .strip()
                )

                # 줄바꿈 정리
                reason = re.sub(
                    r"\s+",
                    " ",
                    reason,
                ).strip()


            # ------------------------------------------------
            # 제목 fallback
            # ------------------------------------------------

            if not title:

                title = str(
                    candidate.get(
                        "title",
                        ""
                    ) or ""
                ).strip()


            # ------------------------------------------------
            # 최종 객체
            # ------------------------------------------------

            results.append(
                FinalRecommendation(
                    rank=rank,
                    title=title,
                    arxiv_id=arxiv_id,
                    link=link,
                    reason=reason,
                )
            )


            # 최대 10개
            if len(results) >= 10:

                break


        # ----------------------------------------------------
        # Rank 기준 정렬
        # ----------------------------------------------------

        results.sort(
            key=lambda x: x.rank
        )


        # ----------------------------------------------------
        # 최종 rank 1~10 정규화
        # ----------------------------------------------------

        normalized: List[
            FinalRecommendation
        ] = []

        for index, result in enumerate(
            results,
            start=1,
        ):

            normalized.append(
                FinalRecommendation(
                    rank=index,
                    title=result.title,
                    arxiv_id=result.arxiv_id,
                    link=result.link,
                    reason=result.reason,
                )
            )


        return normalized