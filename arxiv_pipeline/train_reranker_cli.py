"""
서빙용 Stage2 재랭커 학습 스크립트 (서버 실행).

골드셋 라벨 CSV(라벨링 끝난 P1.csv~P12.csv)로 cross-encoder 재랭커를 학습해 저장한다.
app.py의 /recommend가 이 모델(config.RERANKER_PATH)을 로드해 Stage1 후보를 재랭킹한다.

확정 모델 = MiniLM 파인튜닝(BCE). (실험 결과: MiniLM-BCE가 nDCG 0.771로
zero-shot LLM·임베딩 baseline 압도, BGE(0.787)와 근접하면서 12배 가벼움)

준비:
    pip install "sentence-transformers" datasets    # datasets는 CrossEncoder.fit에 필요
    라벨 CSV들을 --labeling-dir 폴더에 둔다 (컬럼: arxiv_id,label,title,abstract_clean,...)

실행:
    python train_reranker_cli.py                              # labeling_csv/ → models/stage2_reranker
    python train_reranker_cli.py --labeling-dir labeling_csv --epochs 3
    python train_reranker_cli.py --base-model BAAI/bge-reranker-base   # 품질 우선 시
"""

import argparse
import csv
import glob
import json
import os

from config import RERANKER_PATH
from reranker import doc_text, save_reranker, train_reranker


def load_examples_from_csv(labeling_dir: str, profiles_path: str):
    """라벨 CSV들 → [{query, doc, label}]. profile_id는 파일명(P1.csv → P1)에서.
    label은 '1'/'1.0'/1 모두 허용, 빈칸·uncertain은 제외."""
    with open(profiles_path, encoding="utf-8") as f:
        prof = {p["profile_id"]: p["profile_text"] for p in json.load(f)}

    exs = []
    for path in sorted(glob.glob(os.path.join(labeling_dir, "*.csv"))):
        pid = os.path.basename(path).split(".")[0]
        query = prof.get(pid)
        if not query:
            print(f"[skip] {os.path.basename(path)} — 프로필 텍스트 없음(pid={pid})")
            continue
        with open(path, encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                try:
                    label = int(float(str(r.get("label", "")).strip()))
                except (ValueError, TypeError):
                    continue
                if label not in (0, 1):
                    continue
                d = doc_text({"title": r.get("title", ""), "abstract_clean": r.get("abstract_clean", "")})
                if d:
                    exs.append({"query": query, "doc": d, "label": label})
    return exs


def main(labeling_dir: str, profiles_path: str, out: str, base_model: str, epochs: int, batch_size: int):
    exs = load_examples_from_csv(labeling_dir, profiles_path)
    pos = sum(e["label"] for e in exs)
    print(f"학습 예시 {len(exs)}개 (pos {pos} / neg {len(exs) - pos})")
    if not exs:
        raise SystemExit("학습 예시가 0개입니다. --labeling-dir 경로와 CSV의 label 컬럼을 확인하세요.")

    model = train_reranker(exs, base_model=base_model, epochs=epochs, batch_size=batch_size)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    save_reranker(model, out)
    print(f"저장 완료: {out}\n→ app.py /recommend가 이 모델을 자동 로드합니다 (RERANKER_PATH).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="서빙용 Stage2 재랭커 학습")
    ap.add_argument("--labeling-dir", default="labeling_csv", help="라벨 CSV들이 있는 폴더")
    ap.add_argument("--profiles", default="profiles.json")
    ap.add_argument("--out", default=RERANKER_PATH, help="모델 저장 경로 (기본=config.RERANKER_PATH)")
    ap.add_argument("--base-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2",
                    help="확정 MiniLM. 품질 우선 시 BAAI/bge-reranker-base")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()
    main(args.labeling_dir, args.profiles, args.out, args.base_model, args.epochs, args.batch_size)
