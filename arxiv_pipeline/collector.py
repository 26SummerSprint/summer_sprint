"""
arXiv API 수집 모듈.

- pip install arxiv (공식 라이브러리 아니지만 가장 널리 쓰이는 무료 래퍼)
- 카테고리별 신규 논문을 날짜 범위로 조회.
- 초기 구축(backfill)과 매일 증분 수집(daily) 두 가지 모드를 지원.

주의: arXiv API 예의(etiquette) 상 delay_seconds >= 3초 권장, 너무 잦은 호출 금지.
"""

import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterator, List, Optional

import arxiv

from config import (
    ARXIV_DELAY_SECONDS,
    ARXIV_NUM_RETRIES,
    ARXIV_PAGE_SIZE,
    CATEGORIES,
)

_VERSION_RE = re.compile(r"^(?P<base>.+)v(?P<ver>\d+)$")


@dataclass
class RawPaper:
    arxiv_id: str          # 버전 제거된 base id, 예: 2401.12345
    version: int
    title: str
    authors: List[str]
    abstract_raw: str
    categories: List[str]
    primary_category: str
    comments: Optional[str]
    submitted_date: datetime
    updated_date: datetime
    abs_url: str
    pdf_url: str


def _split_id_version(short_id: str) -> (str, int):
    """'2401.12345v2' -> ('2401.12345', 2). 버전 표기가 없으면 1로 취급."""
    m = _VERSION_RE.match(short_id)
    if m:
        return m.group("base"), int(m.group("ver"))
    return short_id, 1


def _result_to_raw(result: arxiv.Result) -> RawPaper:
    short_id = result.get_short_id()
    base_id, version = _split_id_version(short_id)
    return RawPaper(
        arxiv_id=base_id,
        version=version,
        title=result.title.strip(),
        authors=[a.name for a in result.authors],
        abstract_raw=result.summary.strip(),
        categories=list(result.categories),
        primary_category=result.primary_category,
        comments=result.comment,
        submitted_date=result.published,
        updated_date=result.updated,
        abs_url=result.entry_id,
        pdf_url=result.pdf_url or "",
    )


def _make_client() -> arxiv.Client:
    return arxiv.Client(
        page_size=ARXIV_PAGE_SIZE,
        delay_seconds=ARXIV_DELAY_SECONDS,
        num_retries=ARXIV_NUM_RETRIES,
    )


def _category_clause(categories: List[str]) -> str:
    return "(" + " OR ".join(f"cat:{c}" for c in categories) + ")"


def _date_clause(start: datetime, end: datetime) -> str:
    fmt = "%Y%m%d%H%M%S"
    return f"submittedDate:[{start.strftime(fmt)} TO {end.strftime(fmt)}]"


def fetch_by_date_range(
    start: datetime,
    end: datetime,
    categories: List[str] = None,
) -> Iterator[RawPaper]:
    """
    초기 구축용: 지정한 기간 내 제출된 논문을 모두 가져온다.
    예) 최근 3개월치 cs.RO/cs.CV/cs.CL 일괄 수집.
    """
    categories = categories or CATEGORIES
    query = f"{_category_clause(categories)} AND {_date_clause(start, end)}"

    client = _make_client()
    search = arxiv.Search(
        query=query,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
        max_results=None,  # 제한 없음 (arxiv 라이브러리가 페이지네이션 처리)
    )

    for result in client.results(search):
        yield _result_to_raw(result)


def fetch_since(last_run: datetime, categories: List[str] = None) -> Iterator[RawPaper]:
    """
    매일 증분 수집용: 마지막 실행 시각 이후 제출/갱신된 논문만 가져온다.
    """
    now = datetime.now(timezone.utc)
    yield from fetch_by_date_range(last_run, now, categories)


def fetch_recent_months(months: int = 3, categories: List[str] = None) -> Iterator[RawPaper]:
    """초기 구축 편의 함수: 최근 N개월치 논문 수집."""
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=30 * months)
    yield from fetch_by_date_range(start, end, categories)


if __name__ == "__main__":
    # 간단 동작 확인: 최근 1일치만 조회해서 몇 건 나오는지 확인
    since = datetime.now(timezone.utc) - timedelta(days=1)
    count = 0
    for paper in fetch_since(since):
        count += 1
        if count <= 3:
            print(paper.arxiv_id, "-", paper.title[:60])
    print(f"총 {count}건 조회됨 (최근 1일, {CATEGORIES})")
