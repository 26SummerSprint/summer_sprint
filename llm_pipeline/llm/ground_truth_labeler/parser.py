"""
Gemini가 출력한 마크다운 형식:

## Core Topic
## AND Groups
## OR Groups
## Exclusion Conditions

## Final Result
### Score
### Decision
### Label
### Tag
### Reason

을 파싱해서 LabelResult로 변환한다.

LLM 출력은 완전히 결정적이지 않을 수 있으므로
각 필드는 어느 정도 유연하게(fuzzy) 파싱한다.

파싱에 실패한 경우:
- 안전한 기본값을 사용
- parse_warnings에 경고를 기록

단, Judge 평가에서는 파싱 오류가 조용히 정상 결과로
처리되지 않도록 경고를 남긴다.
"""

import re
from typing import List, Optional, Tuple

from .schemas import LabelResult


# ============================================================
# 허용 값
# ============================================================

_VALID_DECISIONS = (
    "정답",
    "경계",
    "오답",
)

_VALID_TAGS = (
    "excl_violation",
    "kw_only",
    "uncertain",
    "none",
)


# ============================================================
# Final Result Section
# ============================================================

_FINAL_RESULT_RE = re.compile(
    r"^##\s*\*?\s*Final\s+Result\s*\*?\s*$",
    re.IGNORECASE | re.MULTILINE,
)


# ============================================================
# Final Result 내부 필드
# ============================================================

_FINAL_FIELD_RE = re.compile(
    r"^###\s*\*?\s*"
    r"(Score|Decision|Label|Tag|Reason)"
    r"\s*\*?\s*$"
    r"(.*?)"
    r"(?=^###\s*\*?\s*"
    r"(?:Score|Decision|Label|Tag|Reason)"
    r"\s*\*?\s*$|\Z)",
    re.DOTALL
    | re.IGNORECASE
    | re.MULTILINE,
)


# ============================================================
# Public API
# ============================================================

def parse_label_result(
    raw_response: str,
) -> LabelResult:
    """
    Gemini raw response를 LabelResult로 변환한다.

    Parameters
    ----------
    raw_response:
        Gemini가 반환한 원본 텍스트

    Returns
    -------
    LabelResult
        파싱된 결과

    Notes
    -----
    파싱 실패 시 예외를 발생시키지 않고
    안전한 기본값과 parse_warnings를 반환한다.
    """

    warnings: List[str] = []

    if raw_response is None:
        raw_response = ""

    raw_response = str(raw_response)

    # --------------------------------------------------------
    # Analysis / Final Result 분리
    # --------------------------------------------------------

    analysis, final_result_text = (
        _split_analysis_and_final_result(
            raw_response
        )
    )

    if final_result_text is None:

        warnings.append(
            "'## Final Result' 섹션을 찾지 못했습니다. "
            "전체 텍스트에서 필드를 추출합니다."
        )

        final_result_text = raw_response

    # --------------------------------------------------------
    # Final Result 내부 필드 추출
    # --------------------------------------------------------

    sections = _extract_sections(
        final_result_text
    )

    # --------------------------------------------------------
    # Score
    # --------------------------------------------------------

    score, score_warning = _extract_score(
        sections.get("score", "")
    )

    if score_warning:
        warnings.append(
            score_warning
        )

    # --------------------------------------------------------
    # Decision
    # --------------------------------------------------------

    decision, decision_warning = (
        _extract_decision(
            sections.get("decision", "")
        )
    )

    if decision_warning:
        warnings.append(
            decision_warning
        )

    # --------------------------------------------------------
    # Label
    # --------------------------------------------------------

    label, label_warning = _extract_label(
        sections.get("label", ""),
        decision,
    )

    if label_warning:
        warnings.append(
            label_warning
        )

    # --------------------------------------------------------
    # Tag
    # --------------------------------------------------------

    tag, tag_warning = _extract_tag(
        sections.get("tag", ""),
        label,
    )

    if tag_warning:
        warnings.append(
            tag_warning
        )

    # --------------------------------------------------------
    # Reason
    # --------------------------------------------------------

    reason = sections.get(
        "reason",
        "",
    ).strip()

    if not reason:

        warnings.append(
            "'### Reason' 섹션을 찾지 못했습니다."
        )

    # --------------------------------------------------------
    # LabelResult
    # --------------------------------------------------------

    return LabelResult(
        score=score,
        decision=decision,
        label=label,
        tag=tag,
        reason=reason,
        analysis=analysis.strip(),
        raw_response=raw_response,
        parse_warnings=warnings,
    )


# ============================================================
# Analysis / Final Result 분리
# ============================================================

def _split_analysis_and_final_result(
    text: str,
) -> Tuple[str, Optional[str]]:
    """
    응답을 Analysis 부분과 Final Result 부분으로 나눈다.

    다음 형식을 모두 지원한다.

        ## Final Result

        ## *Final Result*

        ## Final Result *

    """

    if not text:
        return "", None

    marker_match = _FINAL_RESULT_RE.search(
        text
    )

    if not marker_match:

        return text, None

    analysis = text[
        :marker_match.start()
    ]

    final_result = text[
        marker_match.end():
    ]

    return (
        analysis,
        final_result,
    )


# ============================================================
# Final Result 필드 추출
# ============================================================

def _extract_sections(
    text: str,
) -> dict:
    """
    Final Result 내부에서 다음 필드를 추출한다.

        Score
        Decision
        Label
        Tag
        Reason

    예:

        ### Score
        85

        ### Decision
        정답

        ### Label
        1

        ### Tag
        none

        ### Reason
        RAG가 핵심 연구 주제이다.
    """

    sections = {}

    if not text:
        return sections

    for match in _FINAL_FIELD_RE.finditer(
        text
    ):

        name = (
            match.group(1)
            .strip()
            .lower()
        )

        content = (
            match.group(2)
            .strip()
        )

        sections[name] = content

    return sections


# ============================================================
# Score
# ============================================================

def _extract_score(
    raw: str,
) -> Tuple[int, str]:
    """
    Score를 추출한다.

    허용 예:

        85
        85/100
        Score: 85
        85점

    범위를 벗어나면 0~100으로 clamp한다.
    """

    if not raw:

        return (
            0,
            "'### Score' 섹션이 비어 있어 "
            "0으로 처리했습니다.",
        )

    # --------------------------------------------------------
    # 우선 0~100 범위의 명시적 숫자를 찾는다.
    # --------------------------------------------------------

    match = re.search(
        r"(?<!\d)"
        r"(100|[1-9]?\d)"
        r"(?!\d)",
        raw,
    )

    if not match:

        return (
            0,
            "'### Score'에서 숫자를 찾지 못해 "
            "0으로 처리했습니다.",
        )

    value = int(
        match.group(1)
    )

    if value > 100:

        return (
            100,
            f"Score {value}가 100을 초과해 "
            "100으로 clamp했습니다.",
        )

    return value, ""


# ============================================================
# Decision
# ============================================================

def _extract_decision(
    raw: str,
) -> Tuple[str, str]:
    """
    Decision을 추출한다.

    우선 정확한 한 줄 값을 찾고,
    실패하면 섹션 내부 substring을 확인한다.
    """

    if not raw:

        return (
            "오답",
            "'### Decision' 섹션이 비어 있어 "
            "'오답'으로 기본 처리했습니다.",
        )

    # --------------------------------------------------------
    # 1차: 정확한 값
    # --------------------------------------------------------

    normalized = raw.strip()

    if normalized in _VALID_DECISIONS:

        return normalized, ""

    # --------------------------------------------------------
    # 2차: 한 줄 단위 탐색
    # --------------------------------------------------------

    lines = [
        line.strip()
        for line in raw.splitlines()
        if line.strip()
    ]

    for line in lines:

        cleaned = re.sub(
            r"^[`*_:\-\s]+|[`*_:\-\s]+$",
            "",
            line,
        ).strip()

        if cleaned in _VALID_DECISIONS:

            return cleaned, ""

    # --------------------------------------------------------
    # 3차: substring
    # --------------------------------------------------------

    for candidate in _VALID_DECISIONS:

        if candidate in raw:

            return (
                candidate,
                f"'### Decision'에서 "
                f"'{candidate}'를 부분 문자열로 "
                "추출했습니다.",
            )

    # --------------------------------------------------------
    # 실패
    # --------------------------------------------------------

    return (
        "오답",
        "'### Decision'에서 "
        "정답/경계/오답을 찾지 못해 "
        "'오답'으로 기본 처리했습니다.",
    )


# ============================================================
# Label
# ============================================================

def _extract_label(
    raw: str,
    decision: str,
) -> Tuple[int, str]:
    """
    Label을 추출한다.

    우선 다음과 같은 명시적 형식을 찾는다.

        1
        0
        Label: 1
        label = 0

    설명문 안의 숫자를 무작정 사용하는 것을 피한다.

    파싱 실패 시:
        정답 -> 1
        경계/오답 -> 0

    으로 보정한다.
    """

    if not raw:

        inferred = (
            1
            if decision == "정답"
            else 0
        )

        return (
            inferred,
            "'### Label' 섹션이 비어 있어 "
            "Decision으로부터 유도했습니다.",
        )

    # --------------------------------------------------------
    # 1차: 전체 값이 0 또는 1
    # --------------------------------------------------------

    normalized = raw.strip()

    if normalized in (
        "0",
        "1",
    ):

        return (
            int(normalized),
            "",
        )

    # --------------------------------------------------------
    # 2차: Label: 0 / Label = 1 등
    # --------------------------------------------------------

    explicit_match = re.search(
        r"(?:label\s*[:=]\s*)"
        r"([01])"
        r"\b",
        raw,
        re.IGNORECASE,
    )

    if explicit_match:

        return (
            int(
                explicit_match.group(1)
            ),
            "",
        )

    # --------------------------------------------------------
    # 3차: 한 줄에 단독으로 0/1
    # --------------------------------------------------------

    for line in raw.splitlines():

        cleaned = line.strip()

        if cleaned in (
            "0",
            "1",
        ):

            return (
                int(cleaned),
                "",
            )

    # --------------------------------------------------------
    # 4차: 섹션 내부에서 0/1 검색
    # --------------------------------------------------------

    match = re.search(
        r"(?<!\d)([01])(?!\d)",
        raw,
    )

    if match:

        return (
            int(
                match.group(1)
            ),
            "'### Label'에서 "
            "명시적인 Label 형식을 찾지 못해 "
            "섹션 내부의 0/1을 사용했습니다.",
        )

    # --------------------------------------------------------
    # 실패 → Decision 기반 보정
    # --------------------------------------------------------

    inferred = (
        1
        if decision == "정답"
        else 0
    )

    return (
        inferred,
        "'### Label'에서 0/1을 찾지 못해 "
        "Decision으로부터 유도했습니다.",
    )


# ============================================================
# Tag
# ============================================================

def _extract_tag(
    raw: str,
    label: int,
) -> Tuple[str, str]:
    """
    Tag를 추출한다.

    허용 값:

        none
        excl_violation
        kw_only
        uncertain
    """

    if not raw:

        default = (
            "none"
            if label == 1
            else "uncertain"
        )

        return (
            default,
            "'### Tag' 섹션이 비어 있어 "
            "기본값으로 처리했습니다.",
        )

    # --------------------------------------------------------
    # 1차: 정확한 값
    # --------------------------------------------------------

    normalized = raw.strip()

    if normalized in _VALID_TAGS:

        tag = normalized

        # Positive label인데 exclusion violation 등의
        # 모순된 tag가 나온 경우 보정
        if (
            label == 1
            and tag != "none"
        ):

            return (
                "none",
                f"label=1인데 Tag가 "
                f"'{tag}'로 나와 "
                "'none'으로 보정했습니다.",
            )

        return tag, ""

    # --------------------------------------------------------
    # 2차: 한 줄 단위
    # --------------------------------------------------------

    lines = [
        line.strip()
        for line in raw.splitlines()
        if line.strip()
    ]

    for line in lines:

        cleaned = re.sub(
            r"^[`*_:\-\s]+|[`*_:\-\s]+$",
            "",
            line,
        ).strip()

        if cleaned in _VALID_TAGS:

            tag = cleaned

            if (
                label == 1
                and tag != "none"
            ):

                return (
                    "none",
                    f"label=1인데 Tag가 "
                    f"'{tag}'로 나와 "
                    "'none'으로 보정했습니다.",
                )

            return tag, ""

    # --------------------------------------------------------
    # 3차: substring
    # --------------------------------------------------------

    for candidate in _VALID_TAGS:

        if candidate in raw:

            tag = candidate

            if (
                label == 1
                and tag != "none"
            ):

                return (
                    "none",
                    f"label=1인데 Tag가 "
                    f"'{tag}'로 나와 "
                    "'none'으로 보정했습니다.",
                )

            return (
                tag,
                f"'### Tag'에서 "
                f"'{candidate}'를 "
                "부분 문자열로 추출했습니다.",
            )

    # --------------------------------------------------------
    # 실패
    # --------------------------------------------------------

    default = (
        "none"
        if label == 1
        else "uncertain"
    )

    return (
        default,
        "'### Tag'에서 유효한 태그를 찾지 못해 "
        "기본값으로 처리했습니다.",
    )