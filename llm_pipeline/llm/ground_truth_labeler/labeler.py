"""
GroundTruthLabeler.

단일 논문:
label()
→ 단일 논문용 SYSTEM_PROMPT 사용
→ Gemini 1회 호출

다수 논문:
label_batch()
→ BATCH_SYSTEM_PROMPT 사용
→ BATCH_SIZE 단위로 자동 분할
→ 각 batch마다 Gemini 1회 호출

예:
30개 논문 + BATCH_SIZE=10

    Batch 1 → 10개 → Gemini 1회
    Batch 2 → 10개 → Gemini 1회
    Batch 3 → 10개 → Gemini 1회

총 Gemini 요청 = 3회

추가 안정화:
- 429 / 500 / 502 / 503 / 504 재시도
- exponential backoff + jitter
- Retry-After 지원
- batch 입력 길이 제한
- batch 실패 시 다른 batch는 계속 실행
"""

import asyncio
import json
import random
from typing import Any, Callable, Dict, List, Optional

import httpx

from .config import (
    DEFAULT_RETRY_SECONDS,
    GEMINI_API_KEY,
    GEMINI_MODEL,
    GEMINI_REQUEST_TIMEOUT,
    MAX_CONCURRENT_REQUESTS,
    MAX_RETRIES,
    BATCH_SIZE,
)

from .parser import parse_label_result

from .prompt_template import (
    SYSTEM_PROMPT,
    build_user_message,
)

from .prompt_batch import (
    BATCH_SYSTEM_PROMPT,
    build_batch_user_message,
)

from .schemas import LabelResult


# ============================================================
# Gemini Endpoint
# ============================================================

_GEMINI_ENDPOINT = (
    "https://generativelanguage.googleapis.com/"
    "v1beta/models/{model}:generateContent"
)


# ============================================================
# Retry 대상 HTTP Status
# ============================================================

_RETRYABLE_STATUS_CODES = {
    429,
    500,
    502,
    503,
    504,
}


# ============================================================
# 예외
# ============================================================

class GroundTruthLabelerError(Exception):
    """Gemini 호출 실패 시 사용하는 예외."""


# ============================================================
# Labeler
# ============================================================

class GroundTruthLabeler:

    def __init__(
        self,
        api_key: str = GEMINI_API_KEY,
        model: str = GEMINI_MODEL,
        timeout: int = GEMINI_REQUEST_TIMEOUT,
        max_concurrent_requests: int = MAX_CONCURRENT_REQUESTS,
        batch_size: int = BATCH_SIZE,
    ):
        self._api_key = api_key
        self._model = model
        self._timeout = timeout

        self._semaphore = asyncio.Semaphore(
            max_concurrent_requests
        )

        # 잘못된 값 방지
        self._batch_size = max(
            1,
            int(batch_size),
        )

    # ========================================================
    # 단일 논문 평가
    # ========================================================

    async def label(
        self,
        profile: str,
        paper_title: str,
        paper_abstract: str,
    ) -> LabelResult:
        """
        단일 논문을 평가한다.

        단일 논문용 SYSTEM_PROMPT를 사용한다.
        """

        if not self._api_key:
            raise GroundTruthLabelerError(
                "GEMINI_API_KEY가 설정되지 않았습니다."
            )

        user_message = build_user_message(
            profile,
            paper_title,
            paper_abstract,
        )

        raw_response = await self._generate(
            user_message=user_message,
            system_prompt=SYSTEM_PROMPT,
        )

        return parse_label_result(
            raw_response
        )

    # ========================================================
    # 다수 논문 평가
    # ========================================================

    async def label_batch(
        self,
        profile: str,
        papers: List[Dict[str, Any]],
        title_key: str = "title",
        abstract_key: str = "abstract_clean",
        id_key: Optional[str] = "arxiv_id",
        on_progress: Optional[
            Callable[
                [int, int, str, bool],
                None,
            ]
        ] = None,
    ) -> List[Dict[str, Any]]:
        """
        여러 논문을 BATCH_SIZE 단위로 자동 분할하여 평가한다.

        예:

            30개 논문
            BATCH_SIZE=10

            → Batch 1 = 10개
            → Batch 2 = 10개
            → Batch 3 = 10개

        각 batch는 Gemini에 1회씩 요청한다.

        중요:
            human label은 Gemini에 전달하지 않는다.
        """

        if not self._api_key:
            raise GroundTruthLabelerError(
                "GEMINI_API_KEY가 설정되지 않았습니다."
            )

        if not papers:
            return []

        total = len(papers)

        # ----------------------------------------------------
        # Gemini 입력용 데이터 구성
        # ----------------------------------------------------

        batch_papers = []

        for paper in papers:

            paper_id = (
                paper.get(id_key)
                if id_key
                else None
            )

            title = str(
                paper.get(
                    title_key,
                    "",
                )
                or ""
            ).strip()

            abstract = str(
                paper.get(
                    abstract_key,
                    "",
                )
                or ""
            ).strip()

            batch_papers.append(
                {
                    "arxiv_id": paper_id,
                    "title": title,
                    "abstract_clean": abstract,
                }
            )

        # ----------------------------------------------------
        # Batch 개수
        # ----------------------------------------------------

        chunks = [
            batch_papers[
                i:i + self._batch_size
            ]
            for i in range(
                0,
                len(batch_papers),
                self._batch_size,
            )
        ]

        print(
            f"[Gemini] 총 {total}개 논문 → "
            f"{len(chunks)}개 batch "
            f"(batch size={self._batch_size})"
        )

        # ----------------------------------------------------
        # 전체 결과
        # ----------------------------------------------------

        all_results = []

        processed = 0

        # ----------------------------------------------------
        # Batch 순차 실행
        #
        # 중요:
        # 동시에 여러 batch를 보내지 않는다.
        #
        # 무료/제한 환경에서는 동시 요청보다
        # 순차 실행이 안정적이다.
        # ----------------------------------------------------

        for batch_index, chunk in enumerate(
            chunks,
            start=1,
        ):

            print(
                f"[Gemini Batch "
                f"{batch_index}/{len(chunks)}] "
                f"{len(chunk)}개 논문 평가 시작"
            )

            batch_results = (
                await self._label_single_batch(
                    profile=profile,
                    papers=chunk,
                    batch_index=batch_index,
                    total_batches=len(chunks),
                )
            )

            # ------------------------------------------------
            # 결과 추가
            # ------------------------------------------------

            all_results.extend(
                batch_results
            )

            # ------------------------------------------------
            # Progress
            # ------------------------------------------------

            for result in batch_results:

                processed += 1

                title = result.get(
                    "title",
                    "",
                )

                success = (
                    result.get("result")
                    is not None
                    and result.get("error")
                    is None
                )

                if on_progress:

                    on_progress(
                        processed,
                        total,
                        title,
                        success,
                    )

            # ------------------------------------------------
            # Batch 사이에 아주 짧은 간격
            #
            # 서버에 연속 요청을 몰아넣는 것을 방지
            # ------------------------------------------------

            if batch_index < len(chunks):

                await asyncio.sleep(
                    1.5
                )

        return all_results

    # ========================================================
    # 단일 Batch 실행
    # ========================================================

    async def _label_single_batch(
        self,
        profile: str,
        papers: List[Dict[str, Any]],
        batch_index: int,
        total_batches: int,
    ) -> List[Dict[str, Any]]:
        """
        하나의 batch를 Gemini에 전달한다.

        실패하면 _generate() 내부에서
        429 / 5xx 재시도를 수행한다.
        """

        # ----------------------------------------------------
        # Batch Prompt
        # ----------------------------------------------------

        batch_papers = []

        for paper in papers:

            batch_papers.append(
                {
                    "arxiv_id": paper.get(
                        "arxiv_id"
                    ),
                    "title": paper.get(
                        "title",
                        "",
                    ),
                    "abstract_clean": self._truncate_text(
                        paper.get(
                            "abstract_clean",
                            "",
                        )
                    ),
                }
            )

        user_message = build_batch_user_message(
            profile,
            batch_papers,
        )

        # ----------------------------------------------------
        # Gemini 호출
        # ----------------------------------------------------

        try:

            async with self._semaphore:

                raw_response = await self._generate(
                    user_message=user_message,
                    system_prompt=BATCH_SYSTEM_PROMPT,
                )

        except asyncio.CancelledError:

            # Ctrl+C / task cancellation을
            # 일반 Exception으로 먹지 않는다.
            raise

        except Exception as e:

            print(
                f"[Gemini Batch "
                f"{batch_index}/{total_batches}] "
                f"실패: {e}"
            )

            return [
                {
                    "arxiv_id": paper.get(
                        "arxiv_id"
                    ),
                    "title": paper.get(
                        "title",
                        "",
                    ),
                    "result": None,
                    "error": str(e),
                }
                for paper in papers
            ]

        # ----------------------------------------------------
        # JSON Parsing
        # ----------------------------------------------------

        try:

            parsed_results = (
                self._parse_batch_response(
                    raw_response
                )
            )

        except Exception as e:

            print(
                f"[Gemini Batch "
                f"{batch_index}/{total_batches}] "
                f"JSON 파싱 실패: {e}"
            )

            return [
                {
                    "arxiv_id": paper.get(
                        "arxiv_id"
                    ),
                    "title": paper.get(
                        "title",
                        "",
                    ),
                    "result": None,
                    "error": (
                        f"Batch JSON 파싱 실패: {e}"
                    ),
                }
                for paper in papers
            ]

        # ----------------------------------------------------
        # arxiv_id 기준 매핑
        # ----------------------------------------------------

        result_by_id = {}

        for item in parsed_results:

            if not isinstance(
                item,
                dict,
            ):
                continue

            arxiv_id = str(
                item.get(
                    "arxiv_id",
                    "",
                )
            ).strip()

            if arxiv_id:

                result_by_id[
                    arxiv_id
                ] = item

        # ----------------------------------------------------
        # 입력 순서 유지
        # ----------------------------------------------------

        final_results = []

        for paper in papers:

            arxiv_id = paper.get(
                "arxiv_id"
            )

            title = paper.get(
                "title",
                "",
            )

            item = result_by_id.get(
                arxiv_id
            )

            # ------------------------------------------------
            # 결과 없음
            # ------------------------------------------------

            if item is None:

                final_results.append(
                    {
                        "arxiv_id": arxiv_id,
                        "title": title,
                        "result": None,
                        "error": (
                            "Gemini 응답에 "
                            "해당 arxiv_id가 없습니다."
                        ),
                    }
                )

                continue

            # ------------------------------------------------
            # LabelResult 변환
            # ------------------------------------------------

            try:

                label_result = (
                    self._dict_to_label_result(
                        item
                    )
                )

                final_results.append(
                    {
                        "arxiv_id": arxiv_id,
                        "title": title,
                        "result": label_result,
                        "error": None,
                    }
                )

            except Exception as e:

                final_results.append(
                    {
                        "arxiv_id": arxiv_id,
                        "title": title,
                        "result": None,
                        "error": (
                            f"결과 변환 실패: {e}"
                        ),
                    }
                )

        print(
            f"[Gemini Batch "
            f"{batch_index}/{total_batches}] "
            f"완료"
        )

        return final_results

    # ========================================================
    # Text 길이 제한
    # ========================================================

    @staticmethod
    def _truncate_text(
        text: Any,
        max_chars: int = 6000,
    ) -> str:
        """
        Gemini batch 입력 크기를 줄이기 위해
        abstract를 제한한다.

        6000 characters는 약 수천 token 수준이다.

        논문의 핵심 내용은 일반적으로 abstract 초반부에
        많이 포함되어 있으므로 너무 긴 abstract만 제한한다.
        """

        if text is None:
            return ""

        text = str(text).strip()

        if len(text) <= max_chars:
            return text

        return (
            text[:max_chars]
            + "\n[ABSTRACT TRUNCATED]"
        )

    # ========================================================
    # Batch JSON Parsing
    # ========================================================

    @staticmethod
    def _parse_batch_response(
        raw_response: str,
    ) -> List[Dict[str, Any]]:
        """
        Gemini의 batch JSON 응답을 파싱한다.

        Markdown code block이 포함되어도 처리한다.
        """

        text = raw_response.strip()

        # ----------------------------------------------------
        # ```json 제거
        # ----------------------------------------------------

        if text.startswith("```"):

            lines = text.splitlines()

            if lines:
                lines = lines[1:]

            if (
                lines
                and lines[-1].strip()
                == "```"
            ):
                lines = lines[:-1]

            text = "\n".join(
                lines
            ).strip()

        # ----------------------------------------------------
        # JSON 배열 찾기
        # ----------------------------------------------------

        start = text.find("[")

        end = text.rfind("]")

        if (
            start == -1
            or end == -1
            or end <= start
        ):

            raise ValueError(
                "JSON 배열을 찾을 수 없습니다."
            )

        text = text[
            start:end + 1
        ]

        data = json.loads(
            text
        )

        if not isinstance(
            data,
            list,
        ):

            raise ValueError(
                "Batch 결과가 "
                "JSON 배열이 아닙니다."
            )

        return data

    # ========================================================
    # Dict → LabelResult
    # ========================================================

    @staticmethod
    def _dict_to_label_result(
        item: Dict[str, Any],
    ) -> LabelResult:
        """
        Batch JSON 결과 하나를
        LabelResult로 변환한다.
        """

        score = item.get(
            "score"
        )

        decision = item.get(
            "decision"
        )

        label = item.get(
            "label"
        )

        tag = item.get(
            "tag"
        )

        reason = item.get(
            "reason"
        )

        # ----------------------------------------------------
        # score
        # ----------------------------------------------------

        if score is None:

            raise ValueError(
                "score가 없습니다."
            )

        try:

            score = int(score)

        except (
            TypeError,
            ValueError,
        ) as e:

            raise ValueError(
                f"score가 숫자가 아닙니다: "
                f"{score}"
            ) from e

        if not 0 <= score <= 100:

            raise ValueError(
                f"score는 0~100이어야 합니다: "
                f"{score}"
            )

        # ----------------------------------------------------
        # decision
        # ----------------------------------------------------

        if decision not in (
            "정답",
            "경계",
            "오답",
        ):

            raise ValueError(
                f"잘못된 decision: "
                f"{decision}"
            )

        # ----------------------------------------------------
        # label
        # ----------------------------------------------------

        try:

            label = int(label)

        except (
            TypeError,
            ValueError,
        ) as e:

            raise ValueError(
                f"label이 숫자가 아닙니다: "
                f"{label}"
            ) from e

        if label not in (
            0,
            1,
        ):

            raise ValueError(
                f"label은 0 또는 1이어야 합니다: "
                f"{label}"
            )

        # ----------------------------------------------------
        # tag
        # ----------------------------------------------------

        if tag not in (
            "none",
            "excl_violation",
            "kw_only",
            "uncertain",
        ):

            raise ValueError(
                f"잘못된 tag: {tag}"
            )

        # ----------------------------------------------------
        # reason
        # ----------------------------------------------------

        if reason is None:
            reason = ""

        reason = str(
            reason
        ).strip()

        return LabelResult(
            score=score,
            decision=decision,
            label=label,
            tag=tag,
            reason=reason,
            analysis="",
            raw_response="",
            parse_warnings=[],
        )

    # ========================================================
    # Gemini API
    # ========================================================

    async def _generate(
        self,
        user_message: str,
        system_prompt: str,
    ) -> str:
        """
        Gemini API를 호출한다.

        Retry:
            429
            500
            502
            503
            504

        exponential backoff + jitter 적용.

        재시도하지 않는 오류:
            400
            401
            403
            404
        """

        url = _GEMINI_ENDPOINT.format(
            model=self._model
        )

        payload = {
            "system_instruction": {
                "parts": [
                    {
                        "text": system_prompt
                    }
                ]
            },

            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": user_message
                        }
                    ],
                }
            ],

            # ------------------------------------------------
            # 출력 크기를 과도하게 늘리지 않음
            # ------------------------------------------------
            "generationConfig": {
                "temperature": 0.0,
                "maxOutputTokens": 4096,
                "responseMimeType": "application/json",
            },
        }

        async with httpx.AsyncClient(
            timeout=self._timeout
        ) as client:

            attempt = 0

            while True:

                attempt += 1

                try:

                    response = await client.post(
                        url,
                        params={
                            "key": self._api_key
                        },
                        json=payload,
                    )

                    status = (
                        response.status_code
                    )

                    # ----------------------------------------
                    # 성공
                    # ----------------------------------------

                    if 200 <= status < 300:

                        data = response.json()

                        return (
                            data[
                                "candidates"
                            ][0][
                                "content"
                            ][
                                "parts"
                            ][0][
                                "text"
                            ]
                        )

                    # ----------------------------------------
                    # Retry 대상
                    # ----------------------------------------

                    if (
                        status
                        in _RETRYABLE_STATUS_CODES
                    ):

                        print(
                            "=" * 60
                        )

                        print(
                            "GEMINI RETRYABLE ERROR"
                        )

                        print(
                            "STATUS:",
                            status,
                        )

                        print(
                            "MODEL:",
                            self._model,
                        )

                        print(
                            "ATTEMPT:",
                            f"{attempt}/{MAX_RETRIES + 1}",
                        )

                        print(
                            "BODY:",
                            response.text[:1000],
                        )

                        print(
                            "=" * 60
                        )

                        # ------------------------------------
                        # Retry 횟수 초과
                        # ------------------------------------

                        if attempt > MAX_RETRIES:

                            raise GroundTruthLabelerError(
                                f"Gemini 호출 실패 "
                                f"(HTTP {status})"
                            )

                        # ------------------------------------
                        # 대기 시간
                        # ------------------------------------

                        wait_seconds = (
                            self._calculate_retry_delay(
                                response=response,
                                attempt=attempt,
                            )
                        )

                        print(
                            f"[Gemini] HTTP {status}, "
                            f"{wait_seconds:.1f}초 후 재시도 "
                            f"({attempt}/{MAX_RETRIES})"
                        )

                        await asyncio.sleep(
                            wait_seconds
                        )

                        continue

                    # ----------------------------------------
                    # 재시도하면 안 되는 오류
                    # ----------------------------------------

                    print(
                        "=" * 60
                    )

                    print(
                        "GEMINI HTTP ERROR"
                    )

                    print(
                        "STATUS:",
                        status,
                    )

                    print(
                        "MODEL:",
                        self._model,
                    )

                    print(
                        "BODY:",
                        response.text,
                    )

                    print(
                        "=" * 60
                    )

                    response.raise_for_status()

                except asyncio.CancelledError:

                    raise

                except (
                    httpx.TimeoutException,
                ) as e:

                    if attempt > MAX_RETRIES:

                        raise GroundTruthLabelerError(
                            f"Gemini timeout: {e}"
                        ) from e

                    wait_seconds = (
                        self._calculate_retry_delay(
                            response=None,
                            attempt=attempt,
                        )
                    )

                    print(
                        f"[Gemini] timeout, "
                        f"{wait_seconds:.1f}초 후 재시도"
                    )

                    await asyncio.sleep(
                        wait_seconds
                    )

                except (
                    httpx.NetworkError,
                    httpx.RemoteProtocolError,
                ) as e:

                    if attempt > MAX_RETRIES:

                        raise GroundTruthLabelerError(
                            f"Gemini network error: {e}"
                        ) from e

                    wait_seconds = (
                        self._calculate_retry_delay(
                            response=None,
                            attempt=attempt,
                        )
                    )

                    print(
                        f"[Gemini] network error, "
                        f"{wait_seconds:.1f}초 후 재시도"
                    )

                    await asyncio.sleep(
                        wait_seconds
                    )

                except GroundTruthLabelerError:

                    raise

                except (
                    httpx.HTTPError,
                    KeyError,
                    IndexError,
                    TypeError,
                    ValueError,
                ) as e:

                    raise GroundTruthLabelerError(
                        f"Gemini 라벨링 호출 실패: "
                        f"{e}"
                    ) from e

    # ========================================================
    # Retry Delay
    # ========================================================

    @staticmethod
    def _calculate_retry_delay(
        response: Optional[httpx.Response],
        attempt: int,
    ) -> float:
        """
        Retry-After가 있으면 우선 사용한다.

        없으면 exponential backoff + jitter.

        예:
            attempt 1 → 약 2초
            attempt 2 → 약 4초
            attempt 3 → 약 8초
            attempt 4 → 약 16초
            attempt 5 → 약 32초

        최대 60초.
        """

        # ----------------------------------------------------
        # Retry-After
        # ----------------------------------------------------

        if response is not None:

            retry_after = (
                response.headers.get(
                    "Retry-After"
                )
            )

            if retry_after:

                try:

                    return min(
                        float(
                            retry_after
                        ),
                        120.0,
                    )

                except (
                    ValueError,
                    TypeError,
                ):

                    pass

        # ----------------------------------------------------
        # Exponential Backoff
        # ----------------------------------------------------

        base = max(
            1.0,
            float(
                DEFAULT_RETRY_SECONDS
            ),
        )

        delay = (
            base
            * (2 ** (attempt - 1))
        )

        # ----------------------------------------------------
        # Jitter
        # ----------------------------------------------------

        jitter = random.uniform(
            0,
            min(
                3.0,
                delay * 0.25,
            ),
        )

        delay += jitter

        return min(
            delay,
            60.0,
        )


# ============================================================
# Legacy helper
# ============================================================

def _retry_after_seconds(
    response: "httpx.Response",
) -> float:
    """
    기존 코드와의 호환성을 위한 함수.

    Retry-After 헤더가 없으면
    DEFAULT_RETRY_SECONDS 사용.
    """

    retry_after = response.headers.get(
        "Retry-After"
    )

    if retry_after:

        try:

            return float(
                retry_after
            )

        except ValueError:

            pass

    return DEFAULT_RETRY_SECONDS


# ============================================================
# Demo
# ============================================================

async def _demo():

    labeler = GroundTruthLabeler()

    profile = (
        "Retrieval-Augmented Generation 연구, "
        "특히 Citation/Grounding/Attribution에 관심. "
        "Hallucination Detection이나 Evaluation도 관심 있음. "
        "Medical Image 응용은 제외."
    )

    papers = [
        {
            "arxiv_id": "2401.00001",
            "title": (
                "Attributed Question Answering "
                "with Retrieval-Augmented LLMs"
            ),
            "abstract_clean": (
                "We study how to generate answers "
                "with fine-grained citations to source "
                "documents in retrieval-augmented "
                "generation systems."
            ),
        },
        {
            "arxiv_id": "2401.00002",
            "title": (
                "Medical Image Segmentation "
                "with Vision Transformers"
            ),
            "abstract_clean": (
                "We propose a transformer-based method "
                "for medical image segmentation."
            ),
        },
    ]

    results = await labeler.label_batch(
        profile=profile,
        papers=papers,
    )

    for result in results:

        print()
        print(
            result["arxiv_id"]
        )

        if result["result"]:

            print(
                "score:",
                result["result"].score,
            )

            print(
                "decision:",
                result["result"].decision,
            )

            print(
                "label:",
                result["result"].label,
            )

            print(
                "tag:",
                result["result"].tag,
            )

            print(
                "reason:",
                result["result"].reason,
            )

        else:

            print(
                "ERROR:",
                result["error"],
            )


if __name__ == "__main__":
    asyncio.run(_demo())