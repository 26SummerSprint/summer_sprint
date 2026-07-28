"""
DB 데이터 삭제 스크립트 (서버 전용).

카테고리 개편(예: cs.CL 제거, cs.LG 추가)이나 재수집이 필요할 때
SQLite(papers.db)와 Chroma(벡터 인덱스)의 데이터를 정리한다.

⚠️ 삭제는 되돌릴 수 없다. 실행 전 필요하면 data/ 폴더를 통째로 백업할 것:
    cp -r data data_backup_$(date +%Y%m%d)

사용 예:
    # 특정 primary 카테고리 논문만 삭제 (예: 개편으로 빠진 cs.CL)
    python reset_data.py --category cs.CL

    # 3년보다 오래된 논문 삭제 (보존 정책: 최근 3년치만 유지)
    python reset_data.py --older-than-years 3

    # 특정 날짜 이전 제출 논문 삭제
    python reset_data.py --before-date 2023-07-28

    # 전체 삭제 (SQLite 전체 + 모든 Chroma 컬렉션 + last_run.json)
    python reset_data.py --all

    # 확인 프롬프트 없이 실행 (스크립트/자동화용)
    python reset_data.py --older-than-years 3 --yes

의존성: chromadb만 필요 (sentence-transformers/torch 불필요 — 모델은 로드하지 않음).
"""

import argparse
import sqlite3
import sys
from datetime import datetime, timezone

import chromadb

from config import CHROMA_DIR, LAST_RUN_PATH, SQLITE_PATH

# Chroma delete를 한 번에 너무 많은 id로 호출하면 느려지므로 나눠서 처리
DELETE_BATCH = 500


def _confirm(message: str, assume_yes: bool) -> bool:
    if assume_yes:
        return True
    answer = input(f"{message} 계속하려면 'delete'를 입력하세요: ").strip()
    return answer == "delete"


def _get_chroma_client():
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def _fts_delete(conn: sqlite3.Connection, ids=None):
    """papers_fts 동기화 삭제. ids=None이면 전체. (FTS 인덱스가 없으면 무시)"""
    if not conn.execute(
        "SELECT 1 FROM sqlite_master WHERE name = 'papers_fts'"
    ).fetchone():
        return
    if ids is None:
        conn.execute("DELETE FROM papers_fts")
        return
    for i in range(0, len(ids), DELETE_BATCH):
        chunk = ids[i : i + DELETE_BATCH]
        placeholders = ", ".join("?" for _ in chunk)
        conn.execute(f"DELETE FROM papers_fts WHERE arxiv_id IN ({placeholders})", chunk)


def _delete_vectors(ids):
    """모든 Chroma 컬렉션에서 해당 id 벡터 삭제."""
    client = _get_chroma_client()
    for c in client.list_collections():
        collection = client.get_collection(c.name)
        for i in range(0, len(ids), DELETE_BATCH):
            collection.delete(ids=ids[i : i + DELETE_BATCH])
        print(f"Chroma 컬렉션 {c.name}: 해당 id 삭제 요청 완료")


def _years_ago(n: int) -> str:
    """오늘 기준 n년 전 날짜(YYYY-MM-DD). 2/29은 2/28로 보정."""
    today = datetime.now(timezone.utc).date()
    try:
        cutoff = today.replace(year=today.year - n)
    except ValueError:
        cutoff = today.replace(year=today.year - n, day=28)
    return cutoff.isoformat()


def delete_all(assume_yes: bool = False) -> None:
    """SQLite papers 전체 + 모든 Chroma 컬렉션 + last_run.json 삭제."""
    conn = sqlite3.connect(SQLITE_PATH, timeout=30)
    total = conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]

    client = _get_chroma_client()
    collections = client.list_collections()
    col_desc = ", ".join(c.name for c in collections) or "(없음)"

    print(f"삭제 대상: SQLite {total}건, Chroma 컬렉션 [{col_desc}], last_run.json")
    if not _confirm("⚠️ 전체 데이터를 삭제합니다.", assume_yes):
        print("취소했습니다.")
        conn.close()
        return

    conn.execute("DELETE FROM papers")
    _fts_delete(conn)  # FTS 인덱스도 함께 비움
    conn.commit()
    conn.close()

    for c in collections:
        client.delete_collection(c.name)
        print(f"Chroma 컬렉션 삭제: {c.name}")

    if LAST_RUN_PATH.exists():
        LAST_RUN_PATH.unlink()
        print("last_run.json 삭제 (다음 --daily는 기본값(1일 전)부터 수집)")

    print(f"완료: SQLite {total}건 + Chroma 컬렉션 {len(collections)}개 삭제됨")


def delete_by_category(category: str, assume_yes: bool = False) -> None:
    """primary_category가 일치하는 논문만 SQLite + 모든 Chroma 컬렉션에서 삭제.

    주의: categories 리스트에 해당 카테고리를 '포함'하는 논문이 아니라,
    primary_category가 정확히 일치하는 논문만 삭제한다. (크로스리스트 논문은
    주 카테고리가 남아있는 한 유지 — 다른 카테고리의 정당한 수집 대상이므로)
    """
    conn = sqlite3.connect(SQLITE_PATH, timeout=30)
    rows = conn.execute(
        "SELECT arxiv_id FROM papers WHERE primary_category = ?", (category,)
    ).fetchall()
    ids = [r[0] for r in rows]

    if not ids:
        print(f"primary_category = {category} 인 논문이 없습니다.")
        conn.close()
        return

    print(f"삭제 대상: primary_category = {category} 논문 {len(ids)}건 (SQLite + Chroma)")
    if not _confirm(f"⚠️ {category} 데이터를 삭제합니다.", assume_yes):
        print("취소했습니다.")
        conn.close()
        return

    conn.execute("DELETE FROM papers WHERE primary_category = ?", (category,))
    _fts_delete(conn, ids)
    conn.commit()
    conn.close()

    # 모델별로 컬렉션이 나뉘어 있을 수 있으므로 전체 컬렉션에서 해당 id 제거
    _delete_vectors(ids)

    print(f"완료: {category} 논문 {len(ids)}건 삭제됨")


def delete_before(cutoff_date: str, assume_yes: bool = False) -> None:
    """submitted_date가 cutoff_date(YYYY-MM-DD) 이전인 논문을 SQLite+FTS+Chroma에서 삭제.

    제출일 문자열 앞 10자(YYYY-MM-DD)만 비교하므로 타임존/시각 표기와 무관하게 안전하다.
    보존 정책(최근 N년치만 유지)에 사용.
    """
    conn = sqlite3.connect(SQLITE_PATH, timeout=30)
    rows = conn.execute(
        "SELECT arxiv_id FROM papers WHERE substr(submitted_date, 1, 10) < ?",
        (cutoff_date,),
    ).fetchall()
    ids = [r[0] for r in rows]

    if not ids:
        print(f"{cutoff_date} 이전 제출 논문이 없습니다. (삭제할 것 없음)")
        conn.close()
        return

    print(f"삭제 대상: submitted_date < {cutoff_date} 논문 {len(ids)}건 (SQLite + FTS + Chroma)")
    if not _confirm(f"⚠️ {cutoff_date} 이전 데이터를 삭제합니다.", assume_yes):
        print("취소했습니다.")
        conn.close()
        return

    for i in range(0, len(ids), DELETE_BATCH):
        chunk = ids[i : i + DELETE_BATCH]
        placeholders = ", ".join("?" for _ in chunk)
        conn.execute(f"DELETE FROM papers WHERE arxiv_id IN ({placeholders})", chunk)
    _fts_delete(conn, ids)
    conn.commit()
    conn.close()

    _delete_vectors(ids)
    print(f"완료: {cutoff_date} 이전 {len(ids)}건 삭제됨")


def main():
    parser = argparse.ArgumentParser(description="SQLite/Chroma 데이터 삭제")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true", help="전체 데이터 삭제")
    group.add_argument(
        "--category", type=str, metavar="CAT",
        help="primary_category가 일치하는 논문만 삭제 (예: cs.CL)",
    )
    group.add_argument(
        "--before-date", type=str, metavar="YYYY-MM-DD",
        help="지정 날짜 이전에 제출된 논문 삭제",
    )
    group.add_argument(
        "--older-than-years", type=int, metavar="N",
        help="N년보다 오래된 논문 삭제 (보존 정책용, 예: 3)",
    )
    parser.add_argument(
        "--backup", action="store_true",
        help="삭제 전 data/를 backups/에 자동 백업 (강력 권장)",
    )
    parser.add_argument(
        "--yes", action="store_true",
        help="확인 프롬프트 생략 (자동화용, 신중히 사용)",
    )
    args = parser.parse_args()

    if not SQLITE_PATH.exists():
        sys.exit(f"DB 파일이 없습니다: {SQLITE_PATH}")

    if args.backup:
        from backup_data import backup
        print("삭제 전 백업 중...")
        backup(label="before_reset")

    if args.all:
        delete_all(assume_yes=args.yes)
    elif args.category:
        delete_by_category(args.category, assume_yes=args.yes)
    elif args.before_date:
        delete_before(args.before_date, assume_yes=args.yes)
    else:  # older-than-years
        cutoff = _years_ago(args.older_than_years)
        print(f"기준일: {cutoff} (오늘 - {args.older_than_years}년)")
        delete_before(cutoff, assume_yes=args.yes)


if __name__ == "__main__":
    main()
