"""
전역 설정값 모음.
- 카테고리, 저장 경로, 임베딩 모델 등을 한 곳에서 관리합니다.
"""

from pathlib import Path

# ── 수집 대상 카테고리 ──────────────────────────────────────
CATEGORIES = ["cs.RO", "cs.CV", "cs.CL"]

# ── 저장 경로 ────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

SQLITE_PATH = DATA_DIR / "papers.db"
CHROMA_DIR = DATA_DIR / "chroma"
COLLECTION_NAME = "arxiv_papers"

# Chroma는 항상 이 서버(AWS 인스턴스) 안에서 PersistentClient로 로컬 실행됨.
# 팀원과의 공유는 아래 FastAPI 계층이 담당하므로, 팀원 쪽 머신은 chromadb/임베딩
# 모델을 설치할 필요가 없음 (요청 보내는 쪽은 requests만 있으면 됨).

# ── FastAPI 서버 설정 (AWS 인스턴스에서 이 값으로 uvicorn 실행) ──────────
import os

API_HOST = "0.0.0.0"
API_PORT = 8000
# 인증키. 실제 운영 시 환경변수로 주입 권장: export ARXIV_API_KEY="..."
API_KEY = os.environ.get("ARXIV_API_KEY", "change-me-team-secret")

# ── 임베딩 모델 ──────────────────────────────────────────
# 로컬에서 무료로 돌아가는 sentence-transformers 모델.
# 영어 초록 검색 품질 대비 속도가 좋아 기본값으로 추천.
# 더 높은 품질이 필요하면 "BAAI/bge-base-en-v1.5" 로 교체 가능(속도는 느려짐).
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBEDDING_BATCH_SIZE = 64

# ── arXiv API 호출 관련 ───────────────────────────────────
# 라이브러리 내부적으로 페이지당 결과 수. 너무 크게 잡으면 API가 종종 끊기므로 100 권장.
ARXIV_PAGE_SIZE = 100
# 연속 호출 사이 대기시간(초). arXiv 정책상 3초 이상 권장.
ARXIV_DELAY_SECONDS = 3.0
ARXIV_NUM_RETRIES = 3

# 마지막 수집 시각을 기록해두는 파일(매일 증분 수집 시 사용)
LAST_RUN_PATH = DATA_DIR / "last_run.json"
