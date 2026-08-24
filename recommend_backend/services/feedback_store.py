"""사용자 피드백(키워드 단위) 로그 읽기 + 세션 반영 계산.

프로필이 달라도 **겹치는 키워드**가 있으면 과거 vote를 반영한다:
- 다운보트가 누적된 논문 → 후보에서 제외(다시 추천 안 함)
- 업보트가 누적된 논문 → 최종 추천 상위로

피드백은 config.FEEDBACK_LOG_PATH(JSONL)에 {ts, keywords, arxiv_id, vote}로 쌓인다.
"""

import json
import os
import re
from collections import defaultdict
from typing import List, Set, Tuple

from ..config import FEEDBACK_LOG_PATH

# 토큰 매칭 시 무시할 일반어(이것만 겹치는 건 "같은 키워드"로 보지 않음)
_STOP = {
    "model", "models", "based", "using", "use", "method", "methods",
    "approach", "approaches", "llm", "llms", "ai", "system", "systems",
    "framework", "frameworks", "language", "learning", "the", "a", "an",
    "of", "for", "and", "with", "in", "on", "to", "via", "study",
}


def _tokens(keywords: List[str]) -> Set[str]:
    """키워드 리스트 → 정규화 토큰 집합.
    문자열이 달라도(예: 'autonomous research agent' vs 'LLM autonomous agents')
    의미가 겹치는 토큰(autonomous, agent)으로 매칭되게 한다."""
    toks: Set[str] = set()
    for k in keywords or []:
        for w in re.split(r"[^a-z0-9]+", str(k).lower()):
            if len(w) < 3:
                continue
            if len(w) > 4 and w.endswith("s") and not w.endswith("ss"):
                w = w[:-1]  # 간단 단수화 (agents→agent) — 양쪽 동일 규칙이라 일관 매칭
            if w in _STOP:
                continue
            toks.add(w)
    return toks


def _load() -> List[dict]:
    recs: List[dict] = []
    if not os.path.exists(FEEDBACK_LOG_PATH):
        return recs
    try:
        with open(FEEDBACK_LOG_PATH, encoding="utf-8") as f:
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


def _net_votes(session_keywords: List[str]) -> dict:
    """세션 키워드와 겹치는 피드백만 반영해 arxiv_id별 net(up-down)을 집계.
    매칭은 토큰 단위 — 문자열이 달라도 의미가 겹치면(agent/autonomous 등) 반영된다."""
    sk = _tokens(session_keywords)
    net: dict = defaultdict(int)
    if not sk:
        return net
    for r in _load():
        rk = _tokens(r.get("keywords"))
        if not (rk & sk):  # 토큰 교집합 없으면 무시
            continue
        aid = r.get("arxiv_id")
        if not aid:
            continue
        vote = r.get("vote")
        if vote == "up":
            net[aid] += 1
        elif vote == "down":
            net[aid] -= 1
    return net


def feedback_sets(session_keywords: List[str], down_threshold: int = -1) -> Tuple[Set[str], Set[str]]:
    """반환: (제외 id set = 다운보트 누적 ≤ threshold, 승격 id set = 업보트 누적 > 0)."""
    net = _net_votes(session_keywords)
    excluded = {aid for aid, n in net.items() if n <= down_threshold}
    upvoted = {aid for aid, n in net.items() if n > 0}
    return excluded, upvoted
