"""
DB 데이터 삭제 스크립트 (서버 전용).

카테고리 개편(예: cs.CL 제거, cs.LG 추가)이나 재수집이 필요할 때
SQLite(papers.db)와 Chroma(벡터 인덱스)의 데이터를 정리한다.

⚠️ 삭제는 되돌릴 수 없다. 실행 전 필요하면 data/ 폴더를 통째로 백업할 것:
    cp -r data data_backup_$(date +%Y%m%d)

사용 예:
    # 특정 primary 카테고리 논문만 삭제 (예: 개편으로 빠진 cs.CL)
    python reset_data.py --category cs.CL

    # 전체 삭제 (SQLite 전체 + 모든 Chroma 컬렉션 + last_run.json)
    python reset_data.py --all

    # 확인 프롬프트 없이 실행 (스크립트/자동화용)
    python reset_data.py --all --yes

의존성: chromadb만 필요 (sentence-transformers/torch 불필요 — 모델은 로드하지 않음).
"""

import argparse
import sqlite3
import sys

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
    conn.commit()
    conn.close()

    # 모델별로 컬렉션이 나뉘어 있을 수 있으므로 전체 컬렉션에서 해당 id 제거
    client = _get_chroma_client()
    for c in client.list_collections():
        collection = client.get_collection(c.name)
        for i in range(0, len(ids), DELETE_BATCH):
            collection.delete(ids=ids[i : i + DELETE_BATCH])
        print(f"Chroma 컬렉션 {c.name}: 해당 id 삭제 요청 완료")

    print(f"완료: {category} 논문 {len(ids)}건 삭제됨")


def main():
    parser = argparse.ArgumentParser(description="SQLite/Chroma 데이터 삭제")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true", help="전체 데이터 삭제")
    group.add_argument(
        "--category", type=str, metavar="CAT",
        help="primary_category가 일치하는 논문만 삭제 (예: cs.CL)",
    )
    parser.add_argument(
        "--yes", action="store_true",
        help="확인 프롬프트 생략 (자동화용, 신중히 사용)",
    )
    args = parser.parse_args()

    if not SQLITE_PATH.exists():
        sys.exit(f"DB 파일이 없습니다: {SQLITE_PATH}")

    if args.all:
        delete_all(assume_yes=args.yes)
    else:
        delete_by_category(args.category, assume_yes=args.yes)


if __name__ == "__main__":
    main()
