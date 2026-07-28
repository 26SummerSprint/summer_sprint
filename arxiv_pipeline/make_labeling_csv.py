"""
골드셋 라벨링 후보 CSV 생성 스크립트 (서버 실행).

profiles.json의 각 프로필에 대해 build_labeling_pool로 후보(하이브리드 ∪ 랜덤)를 뽑고,
논문 메타데이터를 붙여 프로필별 CSV로 저장한다. 이 CSV가 judge 채점 / 사람 라벨링의 입력.

CSV 컬럼:
    profile_id, arxiv_id, source, label, tag, title, primary_category, submitted_date, abstract
    - label/tag는 빈 칸 → 라벨러(사람/judge)가 채움
    - source: keyword | embedding | both | random  (나중에 하이브리드 ablation, kw_only 태깅에 사용)

실행 전 migrate_fts.py로 FTS 인덱스가 구축되어 있어야 한다.
embedder(torch)를 쓰므로 서버(EC2)에서 실행.

실행:
    python make_labeling_csv.py                    # profiles.json → labeling/*.csv
    python make_labeling_csv.py --n-random 40      # 프로필당 랜덤 샘플 수
"""

import argparse
import csv
import json
import os

from db import get_conn, get_papers_by_ids
from retrieval import build_labeling_pool

CSV_COLUMNS = [
    "profile_id", "arxiv_id", "source", "label", "tag",
    "title", "primary_category", "submitted_date", "abstract",
]


def main(profiles_path: str, outdir: str, total: int, n_random: int, n_keyword: int, m_embedding: int):
    os.makedirs(outdir, exist_ok=True)
    with open(profiles_path, encoding="utf-8") as f:
        profiles = json.load(f)

    for p in profiles:
        pool = build_labeling_pool(
            p["profile_text"], p["keywords"], category=p["category"],
            total=total, n_random=n_random, n_keyword=n_keyword, m_embedding=m_embedding,
        )
        ids = [c["arxiv_id"] for c in pool]
        with get_conn() as conn:
            meta = get_papers_by_ids(conn, ids)

        outpath = os.path.join(outdir, f"{p['profile_id']}.csv")
        with open(outpath, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(CSV_COLUMNS)
            for c in pool:
                m = meta.get(c["arxiv_id"], {})
                w.writerow([
                    p["profile_id"], c["arxiv_id"], c["source"], "", "",
                    m.get("title", ""), m.get("primary_category", ""),
                    m.get("submitted_date", ""), (m.get("abstract_clean") or "")[:1500],
                ])

        from collections import Counter
        dist = Counter(c["source"] for c in pool)
        print(f"{outpath}: {len(pool)}건  {dict(dist)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="프로필별 골드셋 라벨링 후보 CSV 생성")
    ap.add_argument("--profiles", default="profiles.json")
    ap.add_argument("--outdir", default="labeling")
    ap.add_argument("--total", type=int, default=60, help="프로필당 후보 편수 (회의 확정 60)")
    ap.add_argument("--n-random", type=int, default=0, help="그중 랜덤 샘플 편수 (기본 0=하이브리드만)")
    ap.add_argument("--n-keyword", type=int, default=30)
    ap.add_argument("--m-embedding", type=int, default=70)
    args = ap.parse_args()
    main(args.profiles, args.outdir, args.total, args.n_random, args.n_keyword, args.m_embedding)
