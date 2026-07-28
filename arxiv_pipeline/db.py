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

-- 키워드 검색용 FTS5 전문 인덱스 (Stage1 하이브리드의 키워드 축).
-- papers와 별도 테이블로 두고 upsert/delete 시 동기화한다. bm25()로 랭킹.
CREATE VIRTUAL TABLE IF NOT EXISTS papers_fts USING fts5(
    arxiv_id UNINDEXED,
    title,
    abstract_clean,
    tokenize = 'porter unicode61'
);

-- 골드셋(정답지) 라벨. (profile_id, arxiv_id, labeler)가 PK라
-- judge 라벨(labeler='judge')과 사람 검증 라벨(labeler=이름)을 한 논문에 함께 저장 가능.
CREATE TABLE IF NOT EXISTS labels (
    profile_id       TEXT NOT NULL,
    profile_version  TEXT NOT NULL DEFAULT 'v1',
    arxiv_id         TEXT NOT NULL,
    source           TEXT,               -- keyword | embedding | both | random | seed
    labeler          TEXT NOT NULL,      -- 'judge' 또는 사람 이름
    label            INTEGER NOT NULL,   -- 0/1 (또는 0~3)
    tag              TEXT,               -- kw_only | excl_violation | excl_borderline
    labeled_at       TEXT NOT NULL,
    PRIMARY KEY (profile_id, arxiv_id, labeler)
);
CREATE INDEX IF NOT EXISTS idx_labels_profile ON labels(profile_id);
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
        _fts_upsert(conn, rec.arxiv_id, rec.title, rec.abstract_clean)
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
        _fts_upsert(conn, rec.arxiv_id, rec.title, rec.abstract_clean)
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


def delete_papers_by_ids(conn: sqlite3.Connection, arxiv_ids: Iterable[str]) -> int:
    """arxiv_id 목록에 해당하는 논문을 삭제. 반환값: 실제 삭제된 행 수.
    (Chroma 벡터 삭제는 별도 — embedder.delete_vectors()를 함께 호출할 것)"""
    ids = list(arxiv_ids)
    total = 0
    for i in range(0, len(ids), 500):
        chunk = ids[i : i + 500]
        placeholders = ", ".join("?" for _ in chunk)
        cur = conn.execute(
            f"DELETE FROM papers WHERE arxiv_id IN ({placeholders})", chunk
        )
        conn.execute(
            f"DELETE FROM papers_fts WHERE arxiv_id IN ({placeholders})", chunk
        )
        total += cur.rowcount
    return total


# ── 키워드 검색 (FTS5) ────────────────────────────────────
def _fts_upsert(conn: sqlite3.Connection, arxiv_id: str, title: str, abstract_clean: str):
    """papers_fts를 papers와 동기화 (upsert_paper 내부에서 호출)."""
    conn.execute("DELETE FROM papers_fts WHERE arxiv_id = ?", (arxiv_id,))
    conn.execute(
        "INSERT INTO papers_fts (arxiv_id, title, abstract_clean) VALUES (?, ?, ?)",
        (arxiv_id, title, abstract_clean),
    )


def rebuild_fts(conn: sqlite3.Connection) -> int:
    """기존 papers 전체를 papers_fts로 일괄 재적재 (최초 1회 마이그레이션용). 반환: 적재 건수."""
    conn.execute("DELETE FROM papers_fts")
    conn.execute(
        "INSERT INTO papers_fts (arxiv_id, title, abstract_clean) "
        "SELECT arxiv_id, title, abstract_clean FROM papers"
    )
    return conn.execute("SELECT COUNT(*) FROM papers_fts").fetchone()[0]


def _build_match_query(terms: Iterable[str]) -> str:
    """키워드 리스트를 FTS5 MATCH 쿼리로 변환. 각 구(phrase)를 큰따옴표로 감싸 OR로 연결.
    (하이픈/특수문자가 FTS5 연산자로 오인되지 않게 phrase로 처리)"""
    parts = []
    for t in terms:
        t = (t or "").replace('"', " ").strip()
        if t:
            parts.append(f'"{t}"')
    return " OR ".join(parts)


def search_keyword(
    conn: sqlite3.Connection,
    terms: Iterable[str],
    top_k: int = 30,
    category: Optional[str] = None,
):
    """키워드(구) 리스트로 BM25 검색. 반환: [(arxiv_id, bm25_score)] — score는 작을수록 매칭 강함."""
    match = _build_match_query(terms)
    if not match:
        return []
    if category:
        rows = conn.execute(
            """
            SELECT f.arxiv_id, bm25(papers_fts) AS score
            FROM papers_fts f
            JOIN papers p ON p.arxiv_id = f.arxiv_id
            WHERE papers_fts MATCH ? AND p.primary_category = ?
            ORDER BY score
            LIMIT ?
            """,
            (match, category, top_k),
        ).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT arxiv_id, bm25(papers_fts) AS score
            FROM papers_fts
            WHERE papers_fts MATCH ?
            ORDER BY score
            LIMIT ?
            """,
            (match, top_k),
        ).fetchall()
    return [(r[0], r[1]) for r in rows]


def keyword_df(conn: sqlite3.Connection, term: str, category: Optional[str] = None) -> int:
    """키워드(구)가 등장하는 문서 수(DF). 프로필 키워드의 50~500 검증에 사용.
    (키워드 검색과 동일한 FTS 매칭 기준으로 세므로 실제 후보 편수를 정확히 예측)"""
    match = _build_match_query([term])
    if not match:
        return 0
    if category:
        row = conn.execute(
            """
            SELECT COUNT(*) FROM papers_fts f
            JOIN papers p ON p.arxiv_id = f.arxiv_id
            WHERE papers_fts MATCH ? AND p.primary_category = ?
            """,
            (match, category),
        ).fetchone()
    else:
        row = conn.execute(
            "SELECT COUNT(*) FROM papers_fts WHERE papers_fts MATCH ?", (match,)
        ).fetchone()
    return row[0]


def sample_random(
    conn: sqlite3.Connection,
    category: Optional[str],
    n: int,
    exclude_ids: Optional[Iterable[str]] = None,
) -> list:
    """카테고리 내 무작위 논문 n편 (골드셋 후보의 negative/pool-bias 완화용)."""
    exclude = set(exclude_ids or [])
    limit = n + len(exclude)
    if category:
        rows = conn.execute(
            "SELECT arxiv_id FROM papers WHERE primary_category = ? ORDER BY RANDOM() LIMIT ?",
            (category, limit),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT arxiv_id FROM papers ORDER BY RANDOM() LIMIT ?", (limit,)
        ).fetchall()
    out = [r[0] for r in rows if r[0] not in exclude]
    return out[:n]


# ── 골드셋 라벨 ───────────────────────────────────────────
def upsert_label(
    conn: sqlite3.Connection,
    profile_id: str,
    arxiv_id: str,
    labeler: str,
    label: int,
    source: Optional[str] = None,
    tag: Optional[str] = None,
    profile_version: str = "v1",
):
    """라벨 1건 저장/갱신 (같은 profile_id+arxiv_id+labeler면 덮어씀)."""
    conn.execute(
        """
        INSERT INTO labels (profile_id, profile_version, arxiv_id, source, labeler, label, tag, labeled_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(profile_id, arxiv_id, labeler) DO UPDATE SET
            label = excluded.label,
            tag = excluded.tag,
            source = COALESCE(excluded.source, labels.source),
            labeled_at = excluded.labeled_at
        """,
        (
            profile_id,
            profile_version,
            arxiv_id,
            source,
            labeler,
            int(label),
            tag,
            datetime.utcnow().isoformat(),
        ),
    )


_LABEL_COLUMNS = ["profile_id", "profile_version", "arxiv_id", "source", "labeler", "label", "tag", "labeled_at"]


def get_labels(conn: sqlite3.Connection, profile_id: Optional[str] = None) -> list:
    """라벨 조회 (profile_id 지정 시 해당 프로필만)."""
    if profile_id:
        rows = conn.execute(
            f"SELECT {', '.join(_LABEL_COLUMNS)} FROM labels WHERE profile_id = ?",
            (profile_id,),
        ).fetchall()
    else:
        rows = conn.execute(
            f"SELECT {', '.join(_LABEL_COLUMNS)} FROM labels"
        ).fetchall()
    return [dict(zip(_LABEL_COLUMNS, r)) for r in rows]


def count_labels(conn: sqlite3.Connection) -> int:
    return conn.execute("SELECT COUNT(*) FROM labels").fetchone()[0]


def get_ids_by_primary_category(conn: sqlite3.Connection, category: str) -> list:
    """primary_category가 일치하는 arxiv_id 목록 (삭제 대상 조회용)."""
    cur = conn.execute(
        "SELECT arxiv_id FROM papers WHERE primary_category = ?", (category,)
    )
    return [r[0] for r in cur.fetchall()]


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
