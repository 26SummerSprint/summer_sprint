"""
FTS5 키워드 인덱스 최초 구축 스크립트 (서버 1회 실행).

기존 papers 테이블(약 6만 건)을 papers_fts로 일괄 재적재한다.
이후 신규 논문은 upsert_paper가 자동으로 FTS를 동기화하므로 재실행 불필요.
(카테고리 개편 등으로 대량 삭제/재수집한 뒤 인덱스를 새로 맞추고 싶을 때만 다시 실행)

실행:
    python migrate_fts.py
"""

from db import get_conn, init_db, rebuild_fts

if __name__ == "__main__":
    init_db()  # papers_fts / labels 테이블이 없으면 생성
    with get_conn() as conn:
        n = rebuild_fts(conn)
    print(f"papers_fts 재적재 완료: {n}건")
