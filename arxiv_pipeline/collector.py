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


def fetch_by_date_range_chunked(
    start: datetime,
    end: datetime,
    categories: List[str] = None,
    chunk_days: int = 7,
    max_retries: int = 5,
    retry_base_seconds: int = 30,
):
    """
    arXiv API는 한 쿼리의 페이지네이션 offset이 대략 10,000을 넘으면 HTTP 500을 낸다
    (arXiv 서버 쪽 제약, 라이브러리 버그 아님). 그래서 긴 기간을 한 번에 조회하면
    카테고리 3개 x 몇 달치처럼 결과가 많을 때 쉽게 이 한도에 걸린다.

    이 함수는 전체 기간을 chunk_days 단위 창으로 쪼개서 **start부터 시간순으로**
    조회하고, 각 창의 결과를 (window_start, window_end, papers) 튜플로 하나씩
    yield한다. 창 하나의 결과가 10,000건을 넘는 일은 거의 없다.

    또한 몇 시간짜리 장시간 수집 중 arXiv가 rate limit(429)이나 일시 장애(503)를
    걸 수 있는데, arxiv 라이브러리 자체 재시도(num_retries)가 다 소진되면 예외가
    그대로 올라와서 전체 작업이 죽어버린다. 그래서 창 단위로 한 번 더 재시도를
    감싸서(대기 시간을 점점 늘려가며), 일시적인 장애로 몇 시간의 진행 상황이
    통째로 날아가지 않게 한다. max_retries를 다 소진한 창은 빈 결과로 건너뛰고
    경고를 출력한다 (나중에 그 구간만 --start-date/--end-date로 재수집 가능).
    """
    window_start = start
    while window_start < end:
        window_end = min(end, window_start + timedelta(days=chunk_days))
        label = f"{window_start.date()}~{window_end.date()}"

        papers = []
        for attempt in range(1, max_retries + 1):
            try:
                papers = list(fetch_by_date_range(window_start, window_end, categories))
                break
            except Exception as e:
                if attempt >= max_retries:
                    print(
                        f"[경고] {label} 구간 조회가 {max_retries}회 모두 실패했습니다 "
                        f"({e}). 이 구간은 건너뜁니다 — 나중에 이 기간만 "
                        f"--start-date {window_start.date()} --end-date {window_end.date()} "
                        f"로 다시 수집하세요."
                    )
                    papers = []
                    break
                wait = retry_base_seconds * attempt
                print(
                    f"[경고] {label} 구간 조회 실패({e}), {wait}초 대기 후 재시도 "
                    f"({attempt}/{max_retries})..."
                )
                time.sleep(wait)

        yield window_start, window_end, papers
        window_start = window_end


if __name__ == "__main__":
    # 간단 동작 확인: 최근 1일치만 조회해서 몇 건 나오는지 확인
    since = datetime.now(timezone.utc) - timedelta(days=1)
    count = 0
    for paper in fetch_since(since):
        count += 1
        if count <= 3:
            print(paper.arxiv_id, "-", paper.title[:60])
    print(f"총 {count}건 조회됨 (최근 1일, {CATEGORIES})")
