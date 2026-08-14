"""
서빙용 Stage2 재랭커 학습 스크립트 (서버 실행).

골드셋 라벨 CSV(라벨링 끝난 P1.csv~P12.csv)로 cross-encoder 재랭커를 학습해 저장한다.
app.py의 /recommend가 이 모델(config.RERANKER_PATH)을 로드해 Stage1 후보를 재랭킹한다.

확정 모델 = MiniLM 파인튜닝(BCE). (실험 결과: MiniLM-BCE가 nDCG 0.771로
zero-shot LLM·임베딩 baseline 압도, BGE(0.787)와 근접하면서 12배 가벼움)

준비:
    pip install "sentence-transformers" datasets openpyxl   # datasets=fit, openpyxl=엑셀형 CSV 대응
    라벨 파일들을 --labeling-dir 폴더에 둔다 (컬럼: arxiv_id,label,title,abstract_clean,...)
    (진짜 CSV든, 이름만 .csv인 엑셀이든, 인코딩이 뭐든 자동 인식한다)

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


def _read_records(path: str):
    """CSV(인코딩 무관) 또는 '이름만 .csv인 엑셀'을 모두 읽어 dict 레코드 리스트로."""
    import io
    with open(path, "rb") as f:
        raw = f.read()
    if raw[:4] == b"PK\x03\x04":                      # 실제 내용이 xlsx(zip)
        import pandas as pd  # read_excel엔 openpyxl 필요: pip install openpyxl
        return pd.read_excel(io.BytesIO(raw), engine="openpyxl").to_dict("records")
    text = None
    for enc in ("utf-8-sig", "cp949", "euc-kr", "latin-1"):
        try:
            text = raw.decode(enc); break
        except UnicodeDecodeError:
            continue
    return list(csv.DictReader(io.StringIO(text if text is not None else raw.decode("latin-1"))))


def _parse_label(v):
    try:
        return int(float(str(v).strip()))      # "1"/"1.0"/1/1.0 → 1
    except (ValueError, TypeError):
        return None


def _s(x):
    """None/NaN → '' (pandas가 빈칸을 NaN(float)으로 주는 경우 대응)."""
    if x is None or (isinstance(x, float) and x != x):
        return ""
    return str(x)


def load_examples_from_csv(labeling_dir: str, profiles_path: str):
    """라벨 파일들 → [{query, doc, label}]. profile_id는 파일명(P1.csv → P1)에서.
    label은 '1'/'1.0'/1 모두 허용, 빈칸·uncertain은 제외. 파일별 진단을 출력한다."""
    with open(profiles_path, encoding="utf-8") as f:
        prof = {p["profile_id"]: p["profile_text"] for p in json.load(f)}

    paths = sorted(glob.glob(os.path.join(labeling_dir, "*.csv")) +
                   glob.glob(os.path.join(labeling_dir, "*.xlsx")))
    print(f"labeling_dir = {os.path.abspath(labeling_dir)} | 파일 {len(paths)}개")
    if not paths:
        contents = os.listdir(labeling_dir) if os.path.isdir(labeling_dir) else "(폴더 없음)"
        print("  폴더 내용:", contents)
        return []

    exs = []
    for path in paths:
        pid = os.path.basename(path).split(".")[0]
        query = prof.get(pid)
        try:
            recs = _read_records(path)
        except Exception as e:
            print(f"  {os.path.basename(path)}: 읽기 실패 — {e}")
            continue
        fmt = "xlsx" if open(path, "rb").read(4) == b"PK\x03\x04" else "csv"
        v = 0
        for r in recs:
            lb = _parse_label(r.get("label"))
            if lb not in (0, 1):
                continue
            d = doc_text({"title": _s(r.get("title")), "abstract_clean": _s(r.get("abstract_clean"))})
            if query and d:
                exs.append({"query": query, "doc": d, "label": lb}); v += 1
        print(f"  {os.path.basename(path):16s} fmt={fmt} query={'O' if query else 'X'} 유효 {v}")
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
