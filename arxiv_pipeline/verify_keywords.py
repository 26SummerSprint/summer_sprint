"""
프로필 키워드 DF(문서빈도) 검증 스크립트.

profiles.json의 각 키워드가 DB에서 몇 건이나 매칭되는지 세어, 권장 구간(기본 50~500)을
벗어나는 키워드에 플래그를 붙인다. 키워드 검색과 동일한 FTS 매칭 기준으로 세므로
"이 키워드가 실제로 후보를 몇 편 물어올지"를 정확히 예측한다.

- 너무 흔함(> 상한): 후보가 폭발 → 더 좁은 구(phrase)로 수정
- 너무 희귀(< 하한): 하루치 신규 논문에서 거의 안 걸림 → 다른 키워드로 보완

주의: 임계값 50~500은 약 6만 건(1년치) 기준. DB 규모가 다르면 전체의 0.1~1% 비율로 조정.
실행 전 migrate_fts.py로 FTS 인덱스가 구축되어 있어야 한다.

실행:
    python verify_keywords.py                 # profiles.json, 50~500
    python verify_keywords.py 100 1000        # 하한 100, 상한 1000
"""

import json
import sys

from db import get_conn, keyword_df


def main(path: str = "profiles.json", lo: int = 50, hi: int = 500):
    with open(path, encoding="utf-8") as f:
        profiles = json.load(f)

    n_flag = 0
    with get_conn() as conn:
        for p in profiles:
            print(f"\n=== {p['profile_id']} [{p['category']}] ===")
            for kw in p["keywords"]:
                df = keyword_df(conn, kw, category=None)           # 전체 DB 기준
                dfc = keyword_df(conn, kw, category=p["category"])  # 카테고리 내
                if df > hi:
                    flag = "TOO_COMMON"
                elif df < lo:
                    flag = "TOO_RARE"
                else:
                    flag = "OK"
                if flag != "OK":
                    n_flag += 1
                print(f"  [{flag:11s}] DF={df:6d} (해당카테고리={dfc:5d})  {kw}")

    print(f"\n조정 필요 키워드: {n_flag}개 (구간 {lo}~{hi})")


if __name__ == "__main__":
    lo = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    hi = int(sys.argv[2]) if len(sys.argv) > 2 else 500
    main(lo=lo, hi=hi)
