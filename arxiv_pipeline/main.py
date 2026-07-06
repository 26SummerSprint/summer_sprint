"""
파이프라인 실행 스크립트.

사용 예:
  # 초기 구축: 최근 3개월치 수집 + 전처리 + 임베딩까지 한 번에
  python main.py --backfill 3

  # 매일 실행 (cron/스케줄러에 등록): 마지막 실행 이후 신규 논문만 수집
  python main.py --daily

  # 수집만 하고 임베딩은 나중에 따로 돌리고 싶을 때
  python main.py --backfill 3 --skip-embedding
"""

import argparse
import json
from datetime import datetime, timezone

from collector import fetch_recent_months, fetch_since
from config import LAST_RUN_PATH
from db import init_db, get_conn, upsert_paper, count_papers, PaperRecord
from preprocess import clean_abstract, dedupe_raw_papers
from embedder import embed_pending_papers


def _load_last_run() -> datetime:
    if LAST_RUN_PATH.exists():
        data = json.loads(LAST_RUN_PATH.read_text())
        return datetime.fromisoformat(data["last_run"])
    # 파일이 없으면 기본값: 하루 전
    from datetime import timedelta

    return datetime.now(timezone.utc) - timedelta(days=1)


def _save_last_run(ts: datetime):
    LAST_RUN_PATH.write_text(json.dumps({"last_run": ts.isoformat()}))


def _ingest(raw_papers) -> int:
    """RawPaper 목록을 전처리 후 DB에 upsert. 새로/갱신된 건수를 반환."""
    deduped = dedupe_raw_papers(raw_papers)
    changed = 0

    with get_conn() as conn:
        for p in deduped:
            record = PaperRecord(
                arxiv_id=p.arxiv_id,
                version=p.version,
                title=p.title,
                authors=p.authors,
                abstract_raw=p.abstract_raw,
                abstract_clean=clean_abstract(p.abstract_raw),
                categories=p.categories,
                primary_category=p.primary_category,
                comments=p.comments,
                submitted_date=p.submitted_date.isoformat(),
                updated_date=p.updated_date.isoformat(),
                abs_url=p.abs_url,
                pdf_url=p.pdf_url,
            )
            if upsert_paper(conn, record):
                changed += 1

    return changed


def run_backfill(months: int, skip_embedding: bool):
    print(f"[backfill] 최근 {months}개월치 수집 시작...")
    raw_papers = list(fetch_recent_months(months=months))
    print(f"[backfill] arXiv API로 {len(raw_papers)}건 조회됨 (중복 제거 전)")

    changed = _ingest(raw_papers)
    print(f"[backfill] DB에 신규/갱신 {changed}건 저장. 총 보유 논문 수: ")

    with get_conn() as conn:
        print(f"[backfill] 총 {count_papers(conn)}건")

    if not skip_embedding:
        n = embed_pending_papers()
        print(f"[backfill] 임베딩 신규 계산: {n}건")

    _save_last_run(datetime.now(timezone.utc))


def run_daily(skip_embedding: bool):
    last_run = _load_last_run()
    print(f"[daily] 마지막 실행: {last_run.isoformat()} 이후 신규 논문 수집")

    raw_papers = list(fetch_since(last_run))
    print(f"[daily] arXiv API로 {len(raw_papers)}건 조회됨")

    changed = _ingest(raw_papers)
    print(f"[daily] DB에 신규/갱신 {changed}건 저장")

    if not skip_embedding:
        n = embed_pending_papers()
        print(f"[daily] 임베딩 신규 계산: {n}건")

    _save_last_run(datetime.now(timezone.utc))


def main():
    parser = argparse.ArgumentParser(description="arXiv 논문 수집/임베딩 파이프라인")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--backfill", type=int, metavar="MONTHS",
        help="초기 구축: 최근 N개월치 논문 일괄 수집",
    )
    group.add_argument(
        "--daily", action="store_true",
        help="매일 실행용: 마지막 실행 이후 신규 논문만 수집",
    )
    parser.add_argument(
        "--skip-embedding", action="store_true",
        help="수집/전처리만 하고 임베딩 계산은 건너뛰기",
    )
    args = parser.parse_args()

    init_db()

    if args.backfill is not None:
        run_backfill(args.backfill, args.skip_embedding)
    else:
        run_daily(args.skip_embedding)


if __name__ == "__main__":
    main()
