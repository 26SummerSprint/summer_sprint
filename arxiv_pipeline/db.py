"""
SQLite 메타데이터 저장소.

- papers 테이블: arxiv_id(버전 제거된 base id)를 PK로 사용 -> 자동으로 카테고리 중복/버전 중복 방지.
- upsert_paper(): 이미 있는 논문이면 더 높은 버전으로만 갱신, 없으면 새로 삽입.
- get_papers_needing_embedding(): 아직 벡터화 안 된 논문만 조회 (임베딩은 논문당 1회만 계산).
"""

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterable, Optional

from config import SQLITE_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
    arxiv_id        TEXT PRIMARY KEY,   -- 버전 제거된 base id, 예: 2401.12345
    version         INTEGER NOT NULL,   -- 현재 저장된 버전 번호, 예: 2
    title           TEXT NOT NULL,
    authors         TEXT NOT NULL,      -- '; ' 로 join
    abstract_raw    TEXT NOT NULL,      -- 원문 초록
    abstract_clean  TEXT NOT NULL,      -- 전처리된 초록 (임베딩 입력용)
    categories      TEXT NOT NULL,      -- ', ' 로 join, 예: "cs.RO, cs.CV"
    primary_category TEXT NOT NULL,
    comments        TEXT,               -- 학회/저널 정보 파싱용 원문 comment
    submitted_date  TEXT NOT NULL,      -- 최초 제출일 (ISO 8601)
    updated_date    TEXT NOT NULL,      -- 현재 버전 갱신일 (ISO 8601)
    abs_url         TEXT NOT NULL,
    pdf_url         TEXT NOT NULL,
    embedded        INTEGER NOT NULL DEFAULT 0,  -- 0/1, 벡터 저장 여부
    inserted_at     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_papers_submitted ON papers(submitted_date);
CREATE INDEX IF NOT EXISTS idx_papers_category ON papers(primary_category);
CREATE INDEX IF NOT EXISTS idx_papers_embedded ON papers(embedded);
"""


@dataclass
class PaperRecord:
    arxiv_id: str
    version: int
    title: str
    authors: list
    abstract_raw: str
    abstract_clean: str
    categories: list
    primary_category: str
    comments: Optional[str]
    submitted_date: str
    updated_date: str
    abs_url: str
    pdf_url: str


@contextmanager
def get_conn():
    # timeout을 넉넉히 잡음: 팀원 여러 명이 동시에 /papers/ingest로 쓰기 요청을 보낼 수 있어서
    # (SQLite는 동시 쓰기가 1개씩 순차 처리되므로, 락 대기 중 바로 에러내지 않도록)
    conn = sqlite3.connect(SQLITE_PATH, timeout=30)
    conn.execute("PRAGMA foreign_keys = ON;")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with get_conn() as conn:
        conn.executescript(SCHEMA)


def upsert_paper(conn: sqlite3.Connection, rec: PaperRecord) -> bool:
    """
    같은 arxiv_id가 없으면 삽입, 있으면 더 높은 버전일 때만 덮어씀.
    반환값: 실제로 삽입/갱신되었으면 True (임베딩 재계산이 필요한 케이스만 True로 취급하려면
            호출부에서 별도 로직 추가 가능. 여기서는 저장 여부만 반환).
    """
    cur = conn.execute(
        "SELECT version FROM papers WHERE arxiv_id = ?", (rec.arxiv_id,)
    )
    row = cur.fetchone()

    if row is None:
        conn.execute(
            """
            INSERT INTO papers (
                arxiv_id, version, title, authors, abstract_raw, abstract_clean,
                categories, primary_category, comments, submitted_date, updated_date,
                abs_url, pdf_url, embedded, inserted_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?)
            """,
            (
                rec.arxiv_id,
                rec.version,
                rec.title,
                "; ".join(rec.authors),
                rec.abstract_raw,
                rec.abstract_clean,
                ", ".join(rec.categories),
                rec.primary_category,
                rec.comments,
                rec.submitted_date,
                rec.updated_date,
                rec.abs_url,
                rec.pdf_url,
                datetime.utcnow().isoformat(),
            ),
        )
        return True

    existing_version = row[0]
    if rec.version > existing_version:
        # 버전이 올라간 경우: 내용 갱신 + 재임베딩 필요하므로 embedded=0 으로 리셋
        conn.execute(
            """
            UPDATE papers
            SET version = ?, title = ?, authors = ?, abstract_raw = ?, abstract_clean = ?,
                categories = ?, primary_category = ?, comments = ?, updated_date = ?,
                abs_url = ?, pdf_url = ?, embedded = 0
            WHERE arxiv_id = ?
            """,
            (
                rec.version,
                rec.title,
                "; ".join(rec.authors),
                rec.abstract_raw,
                rec.abstract_clean,
                ", ".join(rec.categories),
                rec.primary_category,
                rec.comments,
                rec.updated_date,
                rec.abs_url,
                rec.pdf_url,
                rec.arxiv_id,
            ),
        )
        return True

    return False  # 이미 최신 버전 보유 중, 아무 것도 안 함


def get_papers_needing_embedding(conn: sqlite3.Connection):
    cur = conn.execute(
        """
        SELECT arxiv_id, abstract_clean, title, primary_category, submitted_date
        FROM papers WHERE embedded = 0
        """
    )
    return cur.fetchall()


def mark_embedded(conn: sqlite3.Connection, arxiv_ids: Iterable[str]):
    conn.executemany(
        "UPDATE papers SET embedded = 1 WHERE arxiv_id = ?",
        [(aid,) for aid in arxiv_ids],
    )


def count_papers(conn: sqlite3.Connection) -> int:
    return conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]


_PAPER_COLUMNS = [
    "arxiv_id", "version", "title", "authors", "abstract_raw", "abstract_clean",
    "categories", "primary_category", "comments", "submitted_date", "updated_date",
    "abs_url", "pdf_url",
]


def get_paper_by_id(conn: sqlite3.Connection, arxiv_id: str) -> Optional[dict]:
    """단일 논문의 전체 메타데이터를 dict로 반환 (없으면 None). API의 /papers/{id}용."""
    cur = conn.execute(
        f"SELECT {', '.join(_PAPER_COLUMNS)} FROM papers WHERE arxiv_id = ?",
        (arxiv_id,),
    )
    row = cur.fetchone()
    if row is None:
        return None
    return dict(zip(_PAPER_COLUMNS, row))


def get_papers_by_ids(conn: sqlite3.Connection, arxiv_ids: Iterable[str]) -> dict:
    """여러 arxiv_id에 대한 메타데이터를 한 번에 조회 (Chroma 검색 결과와 join할 때 사용)."""
    ids = list(arxiv_ids)
    if not ids:
        return {}
    placeholders = ", ".join("?" for _ in ids)
    cur = conn.execute(
        f"SELECT {', '.join(_PAPER_COLUMNS)} FROM papers WHERE arxiv_id IN ({placeholders})",
        ids,
    )
    return {row[0]: dict(zip(_PAPER_COLUMNS, row)) for row in cur.fetchall()}
