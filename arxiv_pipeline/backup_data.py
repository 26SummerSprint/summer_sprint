"""
데이터 백업/복원 스크립트 (서버 전용).

data/ 안의 세 가지를 타임스탬프 폴더로 백업한다:
  - papers.db      (SQLite; sqlite3 백업 API로 안전한 핫 백업)
  - chroma/        (벡터 인덱스; 파일 복사)
  - last_run.json  (마지막 수집 시각)

백업은 arxiv_pipeline/backups/backup_YYYYmmdd_HHMMSS/ 에 저장되며,
backups/ 는 .gitignore에 등록되어 커밋되지 않는다.

⚠️ 완전히 일관된 Chroma 백업을 원하면 백업 중 API 서버를 잠깐 멈추는 것을 권장.
   (papers.db는 sqlite 백업 API를 쓰므로 서버가 떠 있어도 안전)

사용 예:
    python backup_data.py                      # 지금 백업
    python backup_data.py --list               # 백업 목록
    python backup_data.py --restore backup_20260728_153000        # 복원(확인 프롬프트)
    python backup_data.py --restore backup_20260728_153000 --yes  # 확인 생략

의존성: 표준 라이브러리만 (chromadb/torch 불필요).
"""

import argparse
import shutil
import sqlite3
import sys
from datetime import datetime

from config import BASE_DIR, CHROMA_DIR, LAST_RUN_PATH, SQLITE_PATH

BACKUPS_DIR = BASE_DIR / "backups"


def _confirm(message: str, assume_yes: bool) -> bool:
    if assume_yes:
        return True
    return input(f"{message} 계속하려면 'yes'를 입력하세요: ").strip() == "yes"


def _dir_size_mb(path) -> float:
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total / (1024 * 1024)


def backup(label: str = None) -> "Path":
    """현재 data/를 타임스탬프 폴더로 백업. 반환: 백업 폴더 경로."""
    BACKUPS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"backup_{ts}" + (f"_{label}" if label else "")
    dest = BACKUPS_DIR / name
    dest.mkdir()

    # 1) papers.db — sqlite 백업 API (서버가 떠 있어도 일관성 보장)
    if SQLITE_PATH.exists():
        src = sqlite3.connect(str(SQLITE_PATH))
        dst = sqlite3.connect(str(dest / "papers.db"))
        with dst:
            src.backup(dst)
        src.close()
        dst.close()
        print(f"  papers.db 백업 완료")
    else:
        print("  papers.db 없음 — 건너뜀")

    # 2) chroma/ — 파일 복사
    if CHROMA_DIR.exists():
        shutil.copytree(CHROMA_DIR, dest / "chroma")
        print(f"  chroma/ 백업 완료")
    else:
        print("  chroma/ 없음 — 건너뜀")

    # 3) last_run.json
    if LAST_RUN_PATH.exists():
        shutil.copy2(LAST_RUN_PATH, dest / "last_run.json")
        print(f"  last_run.json 백업 완료")

    print(f"백업 완료: {dest}  ({_dir_size_mb(dest):.1f} MB)")
    return dest


def list_backups():
    if not BACKUPS_DIR.exists() or not any(BACKUPS_DIR.iterdir()):
        print("백업이 없습니다.")
        return
    print(f"백업 목록 ({BACKUPS_DIR}):")
    for d in sorted(BACKUPS_DIR.iterdir()):
        if d.is_dir():
            print(f"  {d.name:35s} {_dir_size_mb(d):8.1f} MB")


def restore(name: str, assume_yes: bool = False):
    """백업을 data/로 복원. ⚠️ 복원 전 API 서버를 반드시 멈출 것 (DB 파일 잠금 충돌 방지)."""
    src = BACKUPS_DIR / name
    if not src.is_dir():
        sys.exit(f"백업을 찾을 수 없습니다: {src}")

    print(f"복원 대상: {src}  →  {SQLITE_PATH.parent}")
    print("⚠️ 복원 전 API 서버(uvicorn)를 반드시 멈추세요. 현재 data/를 덮어씁니다.")
    if not _confirm("현재 데이터를 백업본으로 덮어씁니다.", assume_yes):
        print("취소했습니다.")
        return

    if (src / "papers.db").exists():
        shutil.copy2(src / "papers.db", SQLITE_PATH)
        print("  papers.db 복원 완료")
    if (src / "chroma").exists():
        if CHROMA_DIR.exists():
            shutil.rmtree(CHROMA_DIR)
        shutil.copytree(src / "chroma", CHROMA_DIR)
        print("  chroma/ 복원 완료")
    if (src / "last_run.json").exists():
        shutil.copy2(src / "last_run.json", LAST_RUN_PATH)
        print("  last_run.json 복원 완료")

    print(f"복원 완료: {name}")


def main():
    ap = argparse.ArgumentParser(description="데이터 백업/복원")
    ap.add_argument("--list", action="store_true", help="백업 목록 표시")
    ap.add_argument("--restore", type=str, metavar="NAME", help="지정 백업을 data/로 복원")
    ap.add_argument("--label", type=str, default=None, help="백업 폴더명에 붙일 라벨")
    ap.add_argument("--yes", action="store_true", help="확인 프롬프트 생략")
    args = ap.parse_args()

    if args.list:
        list_backups()
    elif args.restore:
        restore(args.restore, assume_yes=args.yes)
    else:
        backup(label=args.label)


if __name__ == "__main__":
    main()
