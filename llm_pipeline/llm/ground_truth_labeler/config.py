"""
GroundTruthLabeler 설정.
"""

import os


# ============================================================
# 이 서비스 자체의 API 인증 (X-API-Key 헤더, api/deps.py에서 사용)
# ============================================================

# 비워두면(로컬 개발 등) 인증을 강제하지 않는다.
API_KEY = os.environ.get(
    "GROUND_TRUTH_LABELER_API_KEY",
    "",
)


# ============================================================
# Gemini
# ============================================================

GEMINI_API_KEY = os.environ.get(
    "GEMINI_API_KEY",
    "",
)

GEMINI_MODEL = os.environ.get(
    "GEMINI_MODEL",
    "gemini-flash-latest",
)

GEMINI_REQUEST_TIMEOUT = int(
    os.environ.get(
        "GEMINI_REQUEST_TIMEOUT",
        "120",
    )
)


# ============================================================
# Batch
# ============================================================

# Gemini 한 번의 요청에 넣을 논문 수
BATCH_SIZE = int(
    os.environ.get(
        "BATCH_SIZE",
        "5",
    )
)


# 동시에 보내는 Gemini 요청 수
#
# 무료 API 사용 시 1 권장
MAX_CONCURRENT_REQUESTS = int(
    os.environ.get(
        "MAX_CONCURRENT_REQUESTS",
        "1",
    )
)


# Gemini 요청 사이 대기 시간
#
# 초 단위
REQUEST_INTERVAL = float(
    os.environ.get(
        "REQUEST_INTERVAL",
        "5",
    )
)


# ============================================================
# Free Tier 보호
# ============================================================

# 한 번의 프로그램 실행에서 사용할 최대 Gemini 요청 수
#
# 무료 Tier를 안전하게 사용하려면 20 권장.
#
# 0이면 제한 없음.
MAX_REQUESTS_PER_RUN = int(
    os.environ.get(
        "MAX_REQUESTS_PER_RUN",
        "20",
    )
)


# 429 발생 시 최대 재시도 횟수
MAX_RETRIES = int(
    os.environ.get(
        "MAX_RETRIES",
        "5",
    )
)


# Retry-After 정보가 없을 경우 사용할 기본 대기 시간
DEFAULT_RETRY_SECONDS = float(
    os.environ.get(
        "DEFAULT_RETRY_SECONDS",
        "40",
    )
)