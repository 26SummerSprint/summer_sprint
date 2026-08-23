"""
전역 설정값 모음.
- 카테고리, 저장 경로, 임베딩 모델 등을 한 곳에서 관리합니다.
"""

from pathlib import Path

# ── 수집 대상 카테고리 ──────────────────────────────────────
# 로봇(cs.RO)·비전(cs.CV)·자연어처리(cs.CL). 팀 논의 결과 초기 구성 유지로 확정.
# (카테고리를 바꾸게 되면 빠진 카테고리의 데이터 정리는 reset_data.py 참고)
CATEGORIES = ["cs.RO", "cs.CV", "cs.CL"]

# ── 저장 경로 ────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

SQLITE_PATH = DATA_DIR / "papers.db"
CHROMA_DIR = DATA_DIR / "chroma"

# Chroma는 항상 이 서버(AWS 인스턴스) 안에서 PersistentClient로 로컬 실행됨.
# 팀원과의 공유는 아래 FastAPI 계층이 담당하므로, 팀원 쪽 머신은 chromadb/임베딩
# 모델을 설치할 필요가 없음 (요청 보내는 쪽은 requests만 있으면 됨).

# ── FastAPI 서버 설정 (AWS 인스턴스에서 이 값으로 uvicorn 실행) ──────────
import os

API_HOST = "0.0.0.0"
API_PORT = 8000
# 인증키. 실제 운영 시 환경변수로 주입 권장: export ARXIV_API_KEY="..."
API_KEY = os.environ.get("ARXIV_API_KEY", "change-me-team-secret")

# ── Gemini (프로필 번역·키워드 추출, 논문 선정·이유 생성) ──────────
# 회의 확정: 논문 선정/키워드 추출/이유 생성 = Gemini(flash 계열), judge = Qwen(별도)
# export GEMINI_API_KEY="..." 로 주입
# GEMINI_MODEL은 fallback 기본값 — 코드가 models.list()로 사용 가능한 flash 모델을
# 자동 선택하고, 실패 시에만 이 값을 쓴다. (gemini-2.5-flash는 신규 계정에서 404가 남)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")

# ── 임베딩 모델 ──────────────────────────────────────────
# 로컬에서 무료로 돌아가는 sentence-transformers 모델.
# 영어 초록 검색 품질 대비 속도가 좋아 기본값으로 추천.
# 더 높은 품질이 필요하면 "BAAI/bge-base-en-v1.5" 로 교체 가능(속도는 느려짐).
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBEDDING_BATCH_SIZE = 64

# ── Stage 2 재랭커 ────────────────────────────────────────
# 학습된 cross-encoder 재랭커 경로. train_reranker_cli.py로 생성, app.py의 /recommend가 로드.
# 확정 모델 = MiniLM 파인튜닝(BCE). 파일 없으면 /recommend는 Stage1 순서로 폴백.
RERANKER_PATH = os.environ.get("RERANKER_PATH", str(BASE_DIR / "models" / "stage2_reranker"))
RERANK_TOP_K = int(os.environ.get("RERANK_TOP_K", "10"))

# 모델별로 컬렉션을 분리해서 저장. 임베딩 모델을 바꿔가며 실험해도
# 기존 컬렉션(벡터)이 삭제/충돌되지 않고 그대로 남아있음 (모델명이 바뀌면
# 자동으로 새 컬렉션이 생성되고, 이전 모델의 컬렉션은 그대로 보존됨).
# 단, 기존 기본 모델("BAAI/bge-small-en-v1.5")은 이미 서버에 "arxiv_papers"라는
# 이름으로 데이터가 쌓여있으므로 하위 호환을 위해 그대로 유지하고,
# 그 외 모델로 바꿨을 때만 새 규칙(모델명을 붙인 이름)을 적용함.
# 예: "BAAI/bge-base-en-v1.5" -> "arxiv_papers_BAAI_bge-base-en-v1.5"
_DEFAULT_MODEL_NAME = "BAAI/bge-small-en-v1.5"
if EMBEDDING_MODEL_NAME == _DEFAULT_MODEL_NAME:
    COLLECTION_NAME = "arxiv_papers"
else:
    COLLECTION_NAME = "arxiv_papers_" + EMBEDDING_MODEL_NAME.replace("/", "_")

# ── arXiv API 호출 관련 ───────────────────────────────────
# 라이브러리 내부적으로 페이지당 결과 수. 너무 크게 잡으면 API가 종종 끊기므로 100 권장.
ARXIV_PAGE_SIZE = 100
# 연속 호출 사이 대기시간(초). arXiv 정책상 3초 이상 권장.
ARXIV_DELAY_SECONDS = 3.0
ARXIV_NUM_RETRIES = 3

# 마지막 수집 시각을 기록해두는 파일(매일 증분 수집 시 사용)
LAST_RUN_PATH = DATA_DIR / "last_run.json"
