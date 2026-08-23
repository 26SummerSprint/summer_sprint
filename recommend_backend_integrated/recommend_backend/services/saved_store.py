"""보관함(재열람용) 저장소 — JSONL 파일에 추천 논문을 맥락과 함께 저장/조회/삭제.

학습 신호(👍/👎, feedback_store)와 분리된 순수 '나중에 다시 보기' 용도.
같은 arxiv_id는 최신 1건만 유지(중복 저장 방지).
경로: config.SAVED_LOG_PATH (기본 recommend_backend/saved_papers.jsonl).
"""

import json
import os
from datetime import datetime, timezone
from typing import List

from ..config import SAVED_LOG_PATH


def _load() -> List[dict]:
    recs: List[dict] = []
    if not os.path.exists(SAVED_LOG_PATH):
        return recs
    try:
        with open(SAVED_LOG_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    recs.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    except OSError:
        pass
    return recs


def _save_all(recs: List[dict]) -> None:
    with open(SAVED_LOG_PATH, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def add_saved(record: dict) -> dict:
    """보관함에 1건 추가. 같은 arxiv_id가 있으면 최신으로 교체(중복 방지)."""
    record = dict(record)
    record["ts"] = datetime.now(timezone.utc).isoformat()
    recs = [r for r in _load() if r.get("arxiv_id") != record.get("arxiv_id")]
    recs.append(record)
    _save_all(recs)
    return record


def list_saved() -> List[dict]:
    """보관 목록(최신순)."""
    return sorted(_load(), key=lambda r: r.get("ts", ""), reverse=True)


def remove_saved(arxiv_id: str) -> int:
    """arxiv_id 1건 삭제. 삭제된 개수 반환."""
    recs = _load()
    kept = [r for r in recs if r.get("arxiv_id") != arxiv_id]
    if len(kept) != len(recs):
        _save_all(kept)
    return len(recs) - len(kept)
