"""API 인증 - X-API-Key 헤더 검증."""

from fastapi import Header, HTTPException

from ..config import API_KEY


async def verify_api_key(x_api_key: str = Header(default="")) -> None:
    if not API_KEY:
        # GROUND_TRUTH_LABELER_API_KEY가 설정되어 있지 않으면 인증을 강제하지
        # 않는다 (로컬 개발/테스트 편의). 운영 환경에서는 반드시 설정할 것.
        return
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key header.")
