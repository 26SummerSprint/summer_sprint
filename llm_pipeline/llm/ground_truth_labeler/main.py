"""
Ground Truth Labeler 엔트리포인트.

두 가지 방식을 지원한다.

1) CLI 모드

python -m ground_truth_labeler.main ^
    --profile-file profiles.json ^
    --papers papers.json ^
    --output results.json

2) 인터랙티브 모드

python -m ground_truth_labeler.main
"""

import argparse
import asyncio
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import GEMINI_API_KEY
from .labeler import GroundTruthLabeler, GroundTruthLabelerError
from .schemas import LabelResult

# evaluate_judge.py의 profiles.json / gold_sets(P*.csv, P*.xlsx) 로딩 로직을
# 그대로 재사용한다 (다수 프로필을 한 번에 라벨링하는 --input-dir 모드에서 사용).
from .evaluate_judge import load_gold_file, load_profiles


# ============================================================
# CLI
# ============================================================

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ground_truth_labeler",
        description="Gemini 기반 논문 라벨링"
    )

    # 단건/배치(단일 프로필) 모드에서만 필수이므로 required=False로 두고,
    # main()에서 모드에 따라 직접 검증한다.
    profile_group = parser.add_mutually_exclusive_group()

    profile_group.add_argument(
        "--profile",
        help="프로필 텍스트 직접 입력"
    )

    profile_group.add_argument(
        "--profile-file",
        help="프로필 JSON/TXT 파일 (profile_text 하나)"
    )

    paper_group = parser.add_mutually_exclusive_group()

    paper_group.add_argument(
        "--title",
        help="단일 논문 제목"
    )

    paper_group.add_argument(
        "--papers",
        help="논문 목록 JSON/CSV/XLSX 파일"
    )

    parser.add_argument(
        "--abstract",
        help="단일 논문 초록"
    )

    parser.add_argument(
        "--output",
        help="결과 저장 경로 (.json 또는 .jsonl). 단건/배치(단일 프로필) 모드용"
    )

    parser.add_argument(
        "--keep-raw",
        action="store_true",
        help="Gemini raw response 저장"
    )

    # --------------------------------------------------------
    # 다수 프로필 파이프라인 모드
    #
    # evaluate_judge.py와 동일한 --input-dir/profiles.json 구조를 그대로
    # 재사용해서, gold_sets/P*.xlsx(또는 .csv) 전체를 각자의 프로필로
    # 한 번에 라벨링한다 (사람 label 컬럼은 사용하지 않고 무시한다).
    # 이 결과를 evaluate_judge.py --labels-dir로 넘기면 성능평가로
    # 이어진다.
    # --------------------------------------------------------

    parser.add_argument(
        "--profiles-file",
        help=(
            "다수 프로필 모드: P1~P12 프로필이 들어 있는 profiles.json. "
            "--input-dir와 함께 사용."
        ),
    )

    parser.add_argument(
        "--input-dir",
        help=(
            "다수 프로필 모드: P1.xlsx~P12.xlsx (또는 .csv)가 있는 "
            "디렉터리 (evaluate_judge.py의 gold_sets와 동일 형식, "
            "label 컬럼은 무시함). --profiles-file과 함께 사용."
        ),
    )

    parser.add_argument(
        "--profile-id",
        help=(
            "다수 프로필 모드에서 특정 프로필 하나만 라벨링. "
            "예: P1"
        ),
    )

    parser.add_argument(
        "--output-dir",
        help=(
            "다수 프로필 모드: 프로필별 결과를 "
            "{output-dir}/{profile_id}.json 으로 저장"
        ),
    )

    return parser


# ============================================================
# Profile
# ============================================================

def load_profile_file(path: str) -> str:
    file_path = Path(path)

    if not file_path.exists():
        raise SystemExit(
            f"프로필 파일을 찾을 수 없습니다: {file_path}"
        )

    # TXT
    if file_path.suffix.lower() == ".txt":
        return file_path.read_text(
            encoding="utf-8"
        ).strip()

    # JSON
    if file_path.suffix.lower() == ".json":
        try:
            data = json.loads(
                file_path.read_text(
                    encoding="utf-8-sig"
                )
            )
        except json.JSONDecodeError as e:
            raise SystemExit(
                f"JSON 형식 오류: {e}"
            )

        # 단일 profile object
        if isinstance(data, dict):
            if "profile_text" in data:
                return str(data["profile_text"]).strip()

        # profile list
        if isinstance(data, list):
            if not data:
                raise SystemExit("프로필 JSON이 비어 있습니다.")

            if "profile_text" in data[0]:
                return str(
                    data[0]["profile_text"]
                ).strip()

        raise SystemExit(
            "JSON에서 profile_text를 찾을 수 없습니다."
        )

    raise SystemExit(
        "지원하지 않는 프로필 파일 형식입니다. "
        "TXT 또는 JSON을 사용하세요."
    )


# ============================================================
# Papers
# ============================================================

def load_papers(path_str: str) -> List[Dict[str, Any]]:
    path = Path(path_str)

    if not path.exists():
        raise SystemExit(
            f"논문 파일을 찾을 수 없습니다: {path}"
        )

    suffix = path.suffix.lower()

    # --------------------------------------------------------
    # JSON
    # --------------------------------------------------------

    if suffix == ".json":

        try:
            data = json.loads(
                path.read_text(
                    encoding="utf-8-sig"
                )
            )
        except json.JSONDecodeError as e:
            raise SystemExit(
                f"JSON 형식 오류: {e}"
            )

        if not isinstance(data, list):
            raise SystemExit(
                "논문 JSON은 배열이어야 합니다."
            )

        return data

    # --------------------------------------------------------
    # CSV / XLSX
    # --------------------------------------------------------

    if suffix in {".csv", ".xlsx"}:

        try:
            import pandas as pd
        except ImportError:
            raise SystemExit(
                "CSV/XLSX를 사용하려면 pandas와 openpyxl이 필요합니다.\n"
                "설치:\n"
                "pip install pandas openpyxl"
            )

        if suffix == ".csv":
            df = pd.read_csv(
                path,
                encoding="utf-8-sig"
            )
        else:
            df = pd.read_excel(path)

        required = {
            "arxiv_id",
            "title",
            "abstract_clean",
        }

        missing = required - set(df.columns)

        if missing:
            raise SystemExit(
                f"{path.name}에 필요한 컬럼이 없습니다: "
                f"{', '.join(sorted(missing))}\n"
                f"필요 컬럼: arxiv_id, title, abstract_clean"
            )

        return df.fillna("").to_dict(
            orient="records"
        )

    raise SystemExit(
        "지원 파일 형식: JSON, CSV, XLSX"
    )


# ============================================================
# Result
# ============================================================

def result_to_dict(
    result: LabelResult,
    keep_raw: bool
) -> Dict[str, Any]:

    data = asdict(result)

    if not keep_raw:
        data.pop("raw_response", None)
        data.pop("analysis", None)

    return data


# ============================================================
# Single
# ============================================================

async def run_single(
    labeler: GroundTruthLabeler,
    profile: str,
    title: str,
    abstract: str,
    keep_raw: bool,
) -> Dict[str, Any]:

    try:

        result = await labeler.label(
            profile,
            title,
            abstract
        )

        return {
            "title": title,
            "result": result_to_dict(
                result,
                keep_raw
            ),
            "error": None,
        }

    except GroundTruthLabelerError as e:

        return {
            "title": title,
            "result": None,
            "error": str(e),
        }


# ============================================================
# Batch
# ============================================================

async def run_batch(
    labeler: GroundTruthLabeler,
    profile: str,
    papers: List[Dict[str, Any]],
    keep_raw: bool,
) -> List[Dict[str, Any]]:

    def on_progress(
        completed: int,
        total: int,
        title: str,
        success: bool
    ):

        status = "완료" if success else "실패"

        print(
            f"[{completed}/{total}] "
            f"{title} ... {status}",
            file=sys.stderr
        )

    results = await labeler.label_batch(
        profile,
        papers,
        title_key="title",
        abstract_key="abstract_clean",
        id_key="arxiv_id",
        on_progress=on_progress,
    )

    output = []

    for r in results:

        output.append(
            {
                "arxiv_id": r.get("arxiv_id"),
                "title": r.get("title"),
                "result": (
                    result_to_dict(
                        r["result"],
                        keep_raw
                    )
                    if r.get("result")
                    else None
                ),
                "error": r.get("error"),
            }
        )

    return output


# ============================================================
# 다수 프로필 파이프라인 (논문 평가 단계)
# ============================================================

async def run_multi_profile(
    labeler: GroundTruthLabeler,
    profiles_file: str,
    input_dir: str,
    output_dir: str,
    keep_raw: bool,
    profile_id: Optional[str] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    profiles.json에 있는 프로필 각각에 대해, 같은 이름의
    gold_sets/P{n}.xlsx(또는 .csv)에 들어 있는 논문들을 라벨링한다.

    사람이 매긴 label 컬럼은 읽지 않는다 (Judge에게 전달하지 않음).

    각 프로필의 결과를 {output_dir}/{profile_id}.json 에 저장한다.
    이 파일은 evaluate_judge.py --labels-dir 로 그대로 넘길 수 있는
    형식이다.
    """

    profiles = load_profiles(Path(profiles_file))

    profile_map = {
        profile["profile_id"]: profile
        for profile in profiles
    }

    input_path = Path(input_dir)

    if not input_path.exists():
        raise SystemExit(
            f"입력 디렉터리를 찾을 수 없습니다: {input_path}"
        )

    gold_paths = sorted(
        [*input_path.glob("P*.csv"), *input_path.glob("P*.xlsx")],
        key=lambda p: p.stem,
    )

    if profile_id:
        gold_paths = [p for p in gold_paths if p.stem == profile_id]

    if not gold_paths:
        raise SystemExit(
            "라벨링할 CSV 또는 XLSX 파일을 찾지 못했습니다.\n"
            f"검색 위치: {input_path}"
        )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_records: Dict[str, List[Dict[str, Any]]] = {}

    for path in gold_paths:

        name = path.stem

        if name not in profile_map:
            print(
                f"[경고] {name} 프로필이 profiles.json에 없습니다. "
                f"건너뜁니다.",
                file=sys.stderr,
            )
            continue

        profile_text = str(
            profile_map[name].get("profile_text", "")
        ).strip()

        if not profile_text:
            print(
                f"[경고] {name}의 profile_text가 비어 있습니다. "
                f"건너뜁니다.",
                file=sys.stderr,
            )
            continue

        # 사람 label 컬럼이 있어도 무시하고 arxiv_id/title/abstract_clean만 사용
        gold_rows = load_gold_file(path)

        papers = [
            {
                "arxiv_id": row["arxiv_id"],
                "title": row["title"],
                "abstract_clean": row["abstract_clean"],
            }
            for row in gold_rows
        ]

        print(
            f"[{name}] {len(papers)}개 논문 평가 시작",
            file=sys.stderr,
        )

        records = await run_batch(
            labeler,
            profile_text,
            papers,
            keep_raw,
        )

        out_path = out_dir / f"{name}.json"

        write_output(records, str(out_path))

        success = sum(1 for r in records if r.get("result"))

        print(
            f"[{name}] 완료: {success}/{len(records)} 성공 "
            f"-> {out_path}",
            file=sys.stderr,
        )

        all_records[name] = records

    return all_records


# ============================================================
# Output
# ============================================================

def write_output(
    records: List[Dict[str, Any]],
    output_path: Optional[str]
):

    if not output_path:

        for record in records:
            print(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    indent=2
                )
            )

        return

    path = Path(output_path)

    if path.suffix.lower() == ".jsonl":

        with path.open(
            "w",
            encoding="utf-8"
        ) as f:

            for record in records:

                f.write(
                    json.dumps(
                        record,
                        ensure_ascii=False
                    )
                    + "\n"
                )

    else:

        path.write_text(
            json.dumps(
                records,
                ensure_ascii=False,
                indent=2
            ),
            encoding="utf-8"
        )


# ============================================================
# Main
# ============================================================

def main():

    if not GEMINI_API_KEY:

        raise SystemExit(
            "GEMINI_API_KEY가 설정되어 있지 않습니다."
        )

    parser = _build_arg_parser()

    args = parser.parse_args()

    # --------------------------------------------------------
    # Labeler
    # --------------------------------------------------------

    labeler = GroundTruthLabeler()

    # --------------------------------------------------------
    # 다수 프로필 파이프라인 모드
    # --------------------------------------------------------

    if args.profiles_file or args.input_dir:

        if not (args.profiles_file and args.input_dir):
            raise SystemExit(
                "--profiles-file과 --input-dir은 함께 사용해야 합니다."
            )

        if not args.output_dir:
            raise SystemExit(
                "다수 프로필 모드에서는 --output-dir이 필요합니다."
            )

        asyncio.run(
            run_multi_profile(
                labeler,
                args.profiles_file,
                args.input_dir,
                args.output_dir,
                args.keep_raw,
                profile_id=args.profile_id,
            )
        )

        return

    # --------------------------------------------------------
    # Profile (단건 / 배치-단일-프로필 모드)
    # --------------------------------------------------------

    if not (args.profile or args.profile_file):
        raise SystemExit(
            "--profile 또는 --profile-file이 필요합니다 "
            "(다수 프로필 모드는 --profiles-file + --input-dir 사용)."
        )

    if not (args.title or args.papers):
        raise SystemExit(
            "--title 또는 --papers가 필요합니다."
        )

    if args.profile:

        profile = args.profile

    else:

        profile = load_profile_file(
            args.profile_file
        )

    # --------------------------------------------------------
    # Single
    # --------------------------------------------------------

    if args.title:

        if not args.abstract:

            raise SystemExit(
                "--title을 사용할 경우 "
                "--abstract도 필요합니다."
            )

        records = [
            asyncio.run(
                run_single(
                    labeler,
                    profile,
                    args.title,
                    args.abstract,
                    args.keep_raw,
                )
            )
        ]

    # --------------------------------------------------------
    # Batch
    # --------------------------------------------------------

    else:

        papers = load_papers(
            args.papers
        )

        print(
            f"총 {len(papers)}개 논문 평가 시작",
            file=sys.stderr
        )

        records = asyncio.run(
            run_batch(
                labeler,
                profile,
                papers,
                args.keep_raw,
            )
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    write_output(
        records,
        args.output
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    success = sum(
        1
        for r in records
        if r.get("result")
    )

    positive = sum(
        1
        for r in records
        if r.get("result")
        and r["result"].get("label") == 1
    )

    print(
        "\n==============================",
        file=sys.stderr
    )

    print(
        "라벨링 완료",
        file=sys.stderr
    )

    print(
        f"전체: {len(records)}",
        file=sys.stderr
    )

    print(
        f"성공: {success}",
        file=sys.stderr
    )

    print(
        f"Positive: {positive}",
        file=sys.stderr
    )

    if args.output:

        print(
            f"결과 저장: {args.output}",
            file=sys.stderr
        )

    print(
        "==============================",
        file=sys.stderr
    )


if __name__ == "__main__":
    main()