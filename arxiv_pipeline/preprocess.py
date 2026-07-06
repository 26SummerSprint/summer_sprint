"""
전처리 모듈.

- clean_abstract(): LaTeX 수식/명령어, 줄바꿈, 중복 공백 제거.
- dedupe_raw_papers(): 같은 arxiv_id가 여러 카테고리로 중복 조회된 경우 +
  버전이 여러 개 섞여 들어온 경우, 최신 버전 하나만 남긴다.
  (여러 카테고리 동시 등록 논문은 collector 단계에서 이미 categories 리스트에
   다 들어있지만, 페치 자체가 카테고리별로 겹쳐 나올 수 있으므로 여기서 최종 정리)
"""

import re
from typing import Dict, Iterable, List

from collector import RawPaper

# LaTeX 명령어 패턴: \command, \command{...}, $...$, $$...$$
_LATEX_CMD_RE = re.compile(r"\\[a-zA-Z]+(\{[^{}]*\})?")
_INLINE_MATH_RE = re.compile(r"\$\$?[^$]*\$\$?")
_MULTI_SPACE_RE = re.compile(r"\s+")
_MULTI_NEWLINE_RE = re.compile(r"[\r\n]+")


def clean_abstract(text: str) -> str:
    """초록 텍스트에서 LaTeX 수식/명령어와 줄바꿈을 정리해 순수 텍스트로 만든다."""
    if not text:
        return ""

    cleaned = text

    # LaTeX 수식 블록 제거 ($...$, $$...$$)
    cleaned = _INLINE_MATH_RE.sub(" ", cleaned)
    # LaTeX 명령어 제거 (\alpha, \textbf{...} 등)
    cleaned = _LATEX_CMD_RE.sub(" ", cleaned)
    # 남은 중괄호/백슬래시 정리
    cleaned = cleaned.replace("\\", " ").replace("{", "").replace("}", "")
    # 줄바꿈 -> 공백
    cleaned = _MULTI_NEWLINE_RE.sub(" ", cleaned)
    # 연속 공백 정리
    cleaned = _MULTI_SPACE_RE.sub(" ", cleaned).strip()

    return cleaned


def dedupe_raw_papers(papers: Iterable[RawPaper]) -> List[RawPaper]:
    """
    arxiv_id 기준 중복 제거. 같은 id가 여러 번 나오면 버전이 가장 높은 것만 유지.
    카테고리가 조회마다 다르게 잡혔을 수 있으므로, 유지되는 레코드의 categories는
    합집합으로 병합한다.
    """
    best: Dict[str, RawPaper] = {}
    merged_categories: Dict[str, set] = {}

    for p in papers:
        key = p.arxiv_id
        if key not in best:
            best[key] = p
            merged_categories[key] = set(p.categories)
            continue

        merged_categories[key].update(p.categories)

        if p.version > best[key].version:
            best[key] = p

    result = []
    for key, p in best.items():
        cats = sorted(merged_categories[key])
        result.append(
            RawPaper(
                arxiv_id=p.arxiv_id,
                version=p.version,
                title=p.title,
                authors=p.authors,
                abstract_raw=p.abstract_raw,
                categories=cats,
                primary_category=p.primary_category,
                comments=p.comments,
                submitted_date=p.submitted_date,
                updated_date=p.updated_date,
                abs_url=p.abs_url,
                pdf_url=p.pdf_url,
            )
        )
    return result
