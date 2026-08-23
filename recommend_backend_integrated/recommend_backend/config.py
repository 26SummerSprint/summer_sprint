"""
recommend_backend 설정.

환경변수:
- ARXIV_PIPELINE_URL
- ARXIV_PIPELINE_API_KEY
- RERANK_PIPELINE_URL
- GEMINI_API_KEY
"""

import os


# ============================================================
# Service URLs
# ============================================================

# arxiv_pipeline
ARXIV_PIPELINE_URL = os.getenv(
    "ARXIV_PIPELINE_URL",
    "http://127.0.0.1:8000",
).rstrip("/")

ARXIV_PIPELINE_API_KEY = os.getenv(
    "ARXIV_API_KEY",
    "",
)


# rerank_pipeline
#
# 현재 rerank_pipeline을 별도 HTTP 서버로 사용하지 않는다면
# 빈 문자열로 둔다.
#
# 나중에 서버가 생기면:
# RERANK_PIPELINE_URL=http://127.0.0.1:8002
#
RERANK_PIPELINE_URL = os.getenv(
    "RERANK_PIPELINE_URL",
    "",
).rstrip("/")


# ============================================================
# HTTP
# ============================================================

HTTP_REQUEST_TIMEOUT = float(
    os.getenv(
        "HTTP_REQUEST_TIMEOUT",
        "60",
    )
)

GEMINI_REQUEST_TIMEOUT = float(
    os.getenv(
        "GEMINI_REQUEST_TIMEOUT",
        "120",
    )
)


# ============================================================
# Gemini
# ============================================================

GEMINI_API_KEY = os.getenv(
    "GEMINI_API_KEY",
    "",
)

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-flash-latest",
)


# ============================================================
# Stage 1
# Hybrid Retrieval
# ============================================================

# 키워드 검색 후보 수
N_KEYWORD = int(
    os.getenv(
        "N_KEYWORD",
        "30",
    )
)

# 임베딩 검색 후보 수
M_EMBEDDING = int(
    os.getenv(
        "M_EMBEDDING",
        "70",
    )
)


# ============================================================
# Keyword DF validation
# ============================================================

# 키워드가 너무 희귀하면 검색에 적합하지 않다고 판단
KEYWORD_DF_MIN = int(
    os.getenv(
        "KEYWORD_DF_MIN",
        "50",
    )
)

# 키워드가 너무 흔하면 검색력이 떨어진다고 판단
KEYWORD_DF_MAX = int(
    os.getenv(
        "KEYWORD_DF_MAX",
        "500",
    )
)


# ============================================================
# Stage 2
# Reranker
# ============================================================

# rerank_pipeline이 없는 현재 환경에서는 사용하지 않음.
#
# 나중에 로컬 reranker를 직접 사용하는 구조로 바꿀 경우
# 모델 경로를 환경변수로 지정할 수 있도록 남겨둔다.
RERANKER_MODEL_PATH = os.getenv(
    "RERANKER_MODEL_PATH",
    "",
)

# Stage 2 결과를 Stage 3 Gemini에 넘길 후보 수
RERANK_COMPRESS_COUNT = int(
    os.getenv(
        "RERANK_COMPRESS_COUNT",
        "25",
    )
)


# ============================================================
# Stage 3
# Final Recommendation
# ============================================================

# 최종 추천 최대 개수
FINAL_RECOMMEND_COUNT = int(
    os.getenv(
        "FINAL_RECOMMEND_COUNT",
        "10",
    )
)


# ============================================================
# Validation
# ============================================================

if KEYWORD_DF_MIN < 0:
    raise ValueError(
        "KEYWORD_DF_MIN은 0 이상이어야 합니다."
    )

if KEYWORD_DF_MAX < KEYWORD_DF_MIN:
    raise ValueError(
        "KEYWORD_DF_MAX는 KEYWORD_DF_MIN보다 크거나 같아야 합니다."
    )

if N_KEYWORD <= 0:
    raise ValueError(
        "N_KEYWORD는 1 이상이어야 합니다."
    )

if M_EMBEDDING <= 0:
    raise ValueError(
        "M_EMBEDDING은 1 이상이어야 합니다."
    )

if RERANK_COMPRESS_COUNT <= 0:
    raise ValueError(
        "RERANK_COMPRESS_COUNT는 1 이상이어야 합니다."
    )

if FINAL_RECOMMEND_COUNT <= 0:
    raise ValueError(
        "FINAL_RECOMMEND_COUNT는 1 이상이어야 합니다."
    )

# ============================================================
# 피드백 로그 (사용자 👍/👎/저장 → 골드셋 확장·재랭커 재학습 재료)
# ============================================================
FEEDBACK_LOG_PATH = os.getenv(
    "FEEDBACK_LOG_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "feedback_log.jsonl"),
)
