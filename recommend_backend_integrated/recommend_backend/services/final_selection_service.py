"""
Stage 3: 최종 논문 선정 서비스.

압축된 후보 논문 전체를 한 번에 Gemini에게 보여주고, 최종 top ~10편 + 추천
이유를 정해진 마크다운 형식으로 받는다. ground_truth_labeler(논문 1편씩
개별 채점)는 이 흐름에서 더 이상 쓰이지 않는다 - 이 프롬프트 자체가
"Ground Truth Labeler와 유사하게" 종합 판단하되, 점수나 내부 분석 과정은
전혀 노출하지 않고 최종 순위/제목/arXiv ID/링크/이유만 마크다운으로
출력하도록 강제한다.

Gemini 출력 형식 (프롬프트가 이 형식을 강제함):

    ## Final Recommendations

    ### 1. [논문 제목]

    - **arXiv ID:** 2401.12345
    - **Link:** https://arxiv.org/abs/2401.12345
    - **추천 이유:** ...

    ### 2. [논문 제목]
    ...

이 모듈은 그 마크다운을 파싱해서 구조화된 FinalRecommendation 리스트로
변환한다.
"""

import re
from dataclasses import dataclass
from typing import Any, Dict, List

import httpx

from ..config import GEMINI_API_KEY, GEMINI_MODEL, GEMINI_REQUEST_TIMEOUT

_GEMINI_ENDPOINT = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
)

_PROMPT_TEMPLATE = """당신은 사용자의 연구 관심 프로필을 기반으로 여러 후보 논문 중 가장 적합한 논문을 선별하는 전문 연구 논문 추천 시스템이다.

사용자는 이미 자신의 연구 관심 분야를 프로필 형태로 정의해 두었다.

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

당신의 목표는 단순히 키워드가 포함되어 있는 논문을 찾는 것이 아니라,
Ground Truth Labeler와 유사하게 각 논문이 사용자의 연구 관심에 실제로 얼마나 적합한지를 종합적으로 판단한 뒤,
가장 적합한 논문을 최종적으로 약 10개 선정하는 것이다.

---

# 1. 판단 기준

각 논문을 다음 순서로 분석한다.

## Core Topic

논문의 핵심 연구 내용이 사용자의 Core Topic과 직접적으로 관련되어 있는지 판단한다.

단순히 제목이나 초록에 관련 키워드가 등장하는 것만으로는 충분하지 않다.

관련 주제가 논문의 핵심적인 연구 목적, 방법론 또는 주요 실험 대상이어야 한다.

다음과 같이 판단한다.

- 핵심: 논문의 주요 연구 내용과 직접적으로 일치
- 일부: 관련 내용은 있지만 논문의 핵심은 아님
- 없음: 실질적인 관련성이 없음

---

## AND Groups

프로필에 정의된 AND 조건을 모두 만족하는지 확인한다.

각 조건이 단순히 언급되는 것이 아니라 논문의 핵심적인 연구 내용과 연결되어 있는지를 판단한다.

AND 조건 중 중요한 조건이 하나라도 충족되지 않으면 추천 우선순위를 크게 낮춘다.

---

## OR Groups

OR 조건은 여러 조건 중 하나 이상이 핵심적으로 충족되는지 확인한다.

하나 이상의 OR 조건을 강하게 만족하면 긍정적으로 평가한다.

---

## Exclusion Conditions

제외 조건을 반드시 확인한다.

논문이 제외 조건에 해당하는 주제를 단순히 언급하는 정도인지,
아니면 논문의 핵심 연구 대상인지 구분한다.

특히 제외 조건이 논문의 핵심 연구 주제라면 높은 점수를 주어서는 안 된다.

제외 조건을 핵심적으로 위반하는 논문은 최종 추천에서 제외한다.

---

# 2. 논문 적합도 판단

각 후보 논문에 대해 다음을 종합적으로 판단한다.

1. Core Topic과의 직접적인 관련성
2. AND 조건 충족 여부
3. OR 조건 충족 여부
4. Exclusion Conditions 위반 여부
5. 논문의 실제 연구 목적과 사용자의 관심 분야의 일치 정도
6. 단순 키워드 매칭인지 실제 연구 주제가 일치하는지
7. 사용자의 연구에 실제로 참고할 가치가 있는지

중요한 점은 다음과 같다.

키워드가 많이 등장한다고 해서 높은 점수를 주지 않는다.

논문의 제목과 초록을 바탕으로 실제 연구 목적과 방법론을 파악하고,
사용자의 연구 관심 분야와 실질적으로 얼마나 가까운지를 판단한다.

---

# 3. 추천 점수

각 논문에 대해 내부적으로 0~100점의 적합도 점수를 계산한다.

대략적인 기준은 다음과 같다.

- 90~100: 사용자의 연구 관심과 매우 직접적으로 일치
- 80~89: 높은 관련성이 있으며 추천 가치가 높음
- 70~79: 상당한 관련성이 있지만 일부 조건이 약함
- 50~69: 부분적으로 관련되지만 핵심 관심과는 거리가 있음
- 0~49: 관련성이 낮거나 제외 조건에 해당

단, 점수 자체보다 실제 연구 내용의 적합성을 우선한다.

특히 Exclusion Condition을 핵심적으로 위반하는 논문은 높은 점수를 부여하지 않는다.

---

# 4. 최종 논문 선정

모든 후보 논문을 분석한 뒤 가장 적합한 논문을 약 10개 선정한다.

후보 논문이 10개보다 적다면 존재하는 논문 중 적합한 논문만 반환한다.

반드시 10개를 채우기 위해 관련성이 낮은 논문을 억지로 포함하지 않는다.

최종 추천 논문은 다음 조건을 우선한다.

1. Core Topic과 직접적으로 일치
2. AND 조건을 충족
3. OR 조건 중 중요한 관심 분야를 충족
4. Exclusion Condition을 위반하지 않음
5. 사용자의 실제 연구에 참고 가치가 높음

동점에 가까운 논문이 있다면 더 직접적인 연구 관련성을 가진 논문을 우선한다.

---

# 5. 중복 논문 처리

동일하거나 사실상 동일한 논문이 여러 후보로 들어온 경우 하나만 선택한다.

arXiv ID를 기준으로 중복을 우선적으로 판단한다.

---

# 6. 최종 결과 출력

최종 결과에는 내부적인 분석 과정이나 점수를 출력하지 않는다.

사용자에게는 오직 다음 정보만 제공한다.

- 순위
- 논문 제목
- arXiv ID
- 논문 링크
- 추천 이유

출력 형식은 반드시 다음과 같이 한다.

## Final Recommendations

### 1. [논문 제목]

- **arXiv ID:** 2401.12345
- **Link:** https://arxiv.org/abs/2401.12345
- **추천 이유:** 사용자의 핵심 관심 분야인 RAG와 citation grounding을 직접적으로 다루며, 논문의 주요 방법론과 실험이 해당 관심 분야와 밀접하게 일치한다.

### 2. [논문 제목]

- **arXiv ID:** 2402.12345
- **Link:** https://arxiv.org/abs/2402.12345
- **추천 이유:** ...

...

### 10. [논문 제목]

- **arXiv ID:** ...
- **Link:** ...
- **추천 이유:** ...

---

# 7. 매우 중요한 출력 제한

최종 응답에서는 다음 정보를 출력하지 않는다.

- 내부 분석 과정
- Core Topic 판단 결과
- AND Groups 판단 결과
- OR Groups 판단 결과
- Exclusion Conditions 판단 결과
- 적합도 점수
- 내부 순위 계산 과정
- 탈락한 논문 목록
- 후보 논문별 상세 평가
- JSON
- Markdown 표

최종 응답은 오직 선정된 논문의

1. 순위
2. 제목
3. arXiv ID
4. 링크
5. 추천 이유

만 포함해야 한다.

추천 이유는 각 논문이 왜 사용자의 연구 관심과 직접적으로 연결되는지를 1~3문장으로 간결하게 설명한다.

---

# 입력

## User Profile

{profile}

## Candidate Papers

{papers}

위 프로필과 후보 논문을 기반으로 평가하고,
가장 적합한 논문을 최대 10개 선정하여 지정된 형식으로 출력하라."""

_BLOCK_RE = re.compile(
    r"###\s*(\d+)\.\s*(.+?)\s*\n(.*?)(?=\n###\s*\d+\.|\Z)", re.DOTALL
)
_ARXIV_ID_RE = re.compile(r"\*\*arXiv ID:\*\*\s*(\S+)")
_LINK_RE = re.compile(r"\*\*Link:\*\*\s*(\S+)")
_REASON_RE = re.compile(r"\*\*추천\s*이유:\*\*\s*(.+?)(?=\n\s*-\s*\*\*|\Z)", re.DOTALL)


class FinalSelectionServiceError(Exception):
    """Gemini 호출 실패 시 사용하는 예외."""


@dataclass
class FinalRecommendation:
    rank: int
    title: str
    arxiv_id: str
    link: str
    reason: str


class FinalSelectionService:
    """Stage 3: 압축된 후보 중 최종 ~10편을 선정하고 추천 이유를 생성한다."""

    def __init__(
        self,
        api_key: str = GEMINI_API_KEY,
        model: str = GEMINI_MODEL,
        timeout: int = GEMINI_REQUEST_TIMEOUT,
    ):
        self._api_key = api_key
        self._model = model
        self._timeout = timeout

    async def select(
        self, profile: str, candidates: List[Dict[str, Any]]
    ) -> List[FinalRecommendation]:
        if not candidates:
            return []

        if not self._api_key:
            raise FinalSelectionServiceError(
                "GEMINI_API_KEY가 설정되지 않았습니다. "
                '환경변수로 주입해주세요: export GEMINI_API_KEY="..."'
            )

        prompt = _PROMPT_TEMPLATE.format(
            profile=profile, papers=self._build_papers_block(candidates)
        )
        raw_text = await self._generate(prompt)
        return self._parse_recommendations(raw_text, candidates)

    async def _generate(self, prompt: str) -> str:
        url = _GEMINI_ENDPOINT.format(model=self._model)
        payload = {"contents": [{"parts": [{"text": prompt}]}]}

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            try:
                response = await client.post(
                    url, params={"key": self._api_key}, json=payload
                )
                response.raise_for_status()
                data = response.json()
                return data["candidates"][0]["content"]["parts"][0]["text"]
            except (httpx.HTTPError, KeyError, IndexError) as e:
                raise FinalSelectionServiceError(f"Gemini 최종 선정 호출 실패: {e}") from e

    @staticmethod
    def _build_papers_block(candidates: List[Dict[str, Any]]) -> str:
        parts = []
        for c in candidates:
            arxiv_id = c.get("arxiv_id", "")
            title = c.get("title", "")
            abstract = (c.get("abstract_clean") or "")[:800]
            link = c.get("abs_url") or (
                f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else ""
            )
            parts.append(
                f"- arXiv ID: {arxiv_id}\n"
                f"  제목: {title}\n"
                f"  초록: {abstract}\n"
                f"  링크: {link}"
            )
        return "\n\n".join(parts)

    @staticmethod
    def _parse_recommendations(
        raw_text: str, candidates: List[Dict[str, Any]]
    ) -> List[FinalRecommendation]:
        marker = re.search(r"##\s*Final Recommendations", raw_text, re.IGNORECASE)
        body = raw_text[marker.end():] if marker else raw_text

        by_arxiv_id = {c.get("arxiv_id"): c for c in candidates if c.get("arxiv_id")}

        results: List[FinalRecommendation] = []
        for match in _BLOCK_RE.finditer(body):
            rank = int(match.group(1))
            title = re.sub(r"^\[|\]$", "", match.group(2).strip()).strip()
            block_text = match.group(3)

            arxiv_match = _ARXIV_ID_RE.search(block_text)
            link_match = _LINK_RE.search(block_text)
            reason_match = _REASON_RE.search(block_text)

            arxiv_id = arxiv_match.group(1).strip() if arxiv_match else ""
            link = link_match.group(1).strip() if link_match else ""
            reason = reason_match.group(1).strip() if reason_match else ""

            # 후보 목록에 있는 논문이면 제목/링크가 비어있을 때 원본 정보로 보정.
            candidate = by_arxiv_id.get(arxiv_id)
            if candidate:
                title = title or candidate.get("title", "")
                link = link or (candidate.get("abs_url") or "")

            results.append(
                FinalRecommendation(
                    rank=rank, title=title, arxiv_id=arxiv_id, link=link, reason=reason
                )
            )

        results.sort(key=lambda r: r.rank)
        return results
