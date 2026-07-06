"""
팀원용: 본인 노트북에서 arXiv 논문을 수집해서 공유 서버(EC2)의 DB에 바로 반영.

collector.py / preprocess.py를 그대로 재사용하므로 무거운 임베딩 라이브러리
(chromadb, sentence-transformers, torch)는 설치할 필요 없음.
임베딩 계산은 서버가 담당함.

준비 (최초 1회):
    pip install -r requirements-client.txt   # requests + arxiv 만 설치됨
    cp local_config.example.py local_config.py
    # local_config.py에 API_BASE, API_KEY 입력

사용 예:
    python collect_and_push.py --days 7
    python collect_and_push.py --days 14 --categories cs.RO cs.CV
"""

import argparse
from datetime import datetime, timedelta, timezone

import requests

from collector import fetch_by_date_range
from preprocess import dedupe_raw_papers

try:
    from local_config import API_BASE, API_KEY
except ImportError:
    raise SystemExit(
        "local_config.py가 없습니다. local_config.example.py를 복사해서 "
        "local_config.py로 저장한 뒤 실제 값을 채워주세요."
    )

HEADERS = {"X-API-Key": API_KEY}
BATCH_SIZE = 200  # 한 번에 서버로 보내는 논문 수 (너무 크면 요청이 오래 걸림)


def _to_payload(p):
    return {
        "arxiv_id": p.arxiv_id,
        "version": p.version,
        "title": p.title,
        "authors": p.authors,
        "abstract_raw": p.abstract_raw,
        "categories": p.categories,
        "primary_category": p.primary_category,
        "comments": p.comments,
        "submitted_date": p.submitted_date.isoformat(),
        "updated_date": p.updated_date.isoformat(),
        "abs_url": p.abs_url,
        "pdf_url": p.pdf_url,
    }


def collect_and_push(days: int, categories: list):
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)

    print(f"arXiv 조회 중... (최근 {days}일, {categories})")
    raw_papers = list(fetch_by_date_range(start, end, categories))
    print(f"조회됨: {len(raw_papers)}건 (중복 제거 전)")

    deduped = dedupe_raw_papers(raw_papers)
    print(f"중복 제거 후: {len(deduped)}건. 서버로 전송 시작 ({API_BASE})")

    total_changed = 0
    total_embedded = 0
    for i in range(0, len(deduped), BATCH_SIZE):
        chunk = deduped[i : i + BATCH_SIZE]
        payload = {"papers": [_to_payload(p) for p in chunk], "auto_embed": True}

        resp = requests.post(
            f"{API_BASE}/papers/ingest", json=payload, headers=HEADERS, timeout=180
        )
        resp.raise_for_status()
        result = resp.json()
        total_changed += result["ingested_or_updated"]
        total_embedded += result["newly_embedded"]
        print(
            f"  배치 {i // BATCH_SIZE + 1}: 신규/갱신 {result['ingested_or_updated']}건, "
            f"임베딩 {result['newly_embedded']}건"
        )

    print(f"완료: 총 신규/갱신 {total_changed}건, 임베딩 {total_embedded}건")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="본인 노트북에서 arXiv 수집 후 공유 서버 DB에 반영"
    )
    parser.add_argument("--days", type=int, default=7, help="최근 N일치 수집 (기본 7일)")
    parser.add_argument(
        "--categories", nargs="+", default=["cs.RO", "cs.CV", "cs.CL"],
        help="수집할 카테고리 목록 (기본: cs.RO cs.CV cs.CL)",
    )
    args = parser.parse_args()
    collect_and_push(args.days, args.categories)
