"""
프로필 처리 스크립트 — Gemini로 한국어 프로필을 영어로 번역 + 키워드/제외조건 추출.

회의 확정: 키워드 추출은 Gemini 2.5 Flash 사용.
profiles.json의 각 프로필(profile_text_ko, 한국어 원문)을 읽어 Gemini에 보내:
  1) 임베딩 검색 쿼리용 영어 본문(profile_text)
  2) 키워드(BM25 검색용) 3~5개 (영어, 2~3단어 구 위주)
  3) 제외조건 요약
을 받아 profiles.json을 갱신한다.

- keywords_confirmed=true 인 프로필(사람이 직접 준 키워드: P1/P5/P9)은 키워드를 덮어쓰지 않고
  번역(profile_text)만 갱신한다. (--overwrite-confirmed 로 강제 덮어쓰기 가능)
- 갱신 후 verify_keywords.py로 DF(50~500)를 재검증하는 것을 권장.

준비:
    pip install google-genai
    export GEMINI_API_KEY="..."      # 또는 config.py

실행:
    python extract_profiles.py                       # profiles.json 제자리 갱신
    python extract_profiles.py --dry-run             # 출력만, 파일 미수정
    python extract_profiles.py --only-unconfirmed    # keywords_confirmed=false 만 처리
"""

import argparse
import json
import sys
import time

from config import GEMINI_API_KEY, GEMINI_MODEL

PROMPT = """You are helping build a research-paper recommender over English arXiv abstracts.
Given a researcher's interest profile written in Korean, produce:
1. "profile_text": a faithful, concise English translation of the profile, to be used as an
   embedding search query over English abstracts. Keep the researcher's specific focus and nuance.
2. "keywords": 3-5 English keywords/phrases that best characterize the interest for keyword (BM25)
   search. Prefer specific 2-3 word technical phrases (e.g. "speculative decoding") over broad field
   names (e.g. "machine learning"). Avoid duplicates and overly generic terms.
3. "exclusion": one short English phrase summarizing what to filter out (from the profile's
   exclusion condition). Empty string if none.

Return ONLY valid JSON with exactly these keys: profile_text, keywords (array of strings), exclusion.

Korean profile:
{body}
"""


def _make_client():
    if not GEMINI_API_KEY:
        sys.exit("GEMINI_API_KEY가 없습니다. export GEMINI_API_KEY=... 후 다시 실행하세요.")
    try:
        from google import genai  # pip install google-genai
    except ImportError:
        sys.exit("google-genai 미설치. `pip install google-genai` 후 실행하세요.")
    return genai.Client(api_key=GEMINI_API_KEY)


# gemini-2.5-flash가 신규 계정에서 막히는(404) 문제 → 사용 가능한 flash 모델을 자동 선택
_PREFER = ["gemini-flash-latest", "gemini-2.0-flash", "gemini-2.0-flash-001",
           "gemini-2.5-flash-lite", "gemini-2.5-flash"]


def _pick_model(client) -> str:
    try:
        avail = [
            m.name.split("/")[-1] for m in client.models.list()
            if "generateContent" in (getattr(m, "supported_actions", None) or ["generateContent"])
        ]
        for name in _PREFER:
            if name in avail:
                return name
        for name in avail:
            if "flash" in name and not any(x in name for x in ("vision", "thinking", "preview", "exp", "tts", "image")):
                return name
        if avail:
            return avail[0]
    except Exception as e:
        print(f"모델 목록 조회 실패, 기본값({GEMINI_MODEL}) 사용:", e)
    return GEMINI_MODEL


def _extract_one(client, model: str, body: str, max_retries: int = 6) -> dict:
    """Gemini에 프로필 본문을 보내 {profile_text, keywords, exclusion} JSON을 받는다.
    429(분당 요청 제한) 발생 시 대기 후 재시도."""
    from google.genai import types

    for attempt in range(max_retries):
        try:
            resp = client.models.generate_content(
                model=model,
                contents=PROMPT.format(body=body),
                config=types.GenerateContentConfig(response_mime_type="application/json"),
            )
            text = (resp.text or "").strip()
            # 혹시 코드펜스가 붙어오면 제거
            if text.startswith("```"):
                text = text.strip("`")
                text = text[text.find("{"): text.rfind("}") + 1]
            return json.loads(text)
        except Exception as e:
            if ("429" in str(e)) or ("RESOURCE_EXHAUSTED" in str(e)):
                print(f"  429 rate limit → 62초 대기 후 재시도 ({attempt + 1}/{max_retries})")
                time.sleep(62)
                continue
            raise
    raise RuntimeError("재시도 초과 (429)")


def main(path: str, dry_run: bool, only_unconfirmed: bool, overwrite_confirmed: bool):
    with open(path, encoding="utf-8") as f:
        profiles = json.load(f)

    client = _make_client()
    model = _pick_model(client)
    print(f"사용 모델: {model}")

    sent = 0
    for p in profiles:
        confirmed = p.get("keywords_confirmed", False)
        if only_unconfirmed and confirmed:
            print(f"[skip] {p['profile_id']} (keywords_confirmed)")
            continue

        if sent:
            time.sleep(13)  # 무료 티어 분당 요청 제한(≈5/min) 회피
        sent += 1
        body = p.get("profile_text_ko") or p.get("profile_text", "")
        try:
            out = _extract_one(client, model, body)
        except Exception as e:
            print(f"[오류] {p['profile_id']}: {e}")
            continue

        new_text = out.get("profile_text", "").strip()
        new_keywords = [k.strip() for k in out.get("keywords", []) if k.strip()]
        new_exclusion = out.get("exclusion", "").strip()

        print(f"\n=== {p['profile_id']} [{p['category']}] ===")
        print("  EN:", new_text[:90], "...")
        print("  keywords:", new_keywords)

        if not dry_run:
            if new_text:
                p["profile_text"] = new_text
            # 사람이 확정한 키워드는 보존 (강제 덮어쓰기 옵션 시 예외)
            if new_keywords and (not confirmed or overwrite_confirmed):
                p["keywords"] = new_keywords
                p["keywords_confirmed"] = False  # Gemini 추출본 → DF 검증 대상
            if new_exclusion:
                p["exclusion_en"] = new_exclusion

    if dry_run:
        print("\n(dry-run: 파일 미수정)")
        return

    with open(path, "w", encoding="utf-8") as f:
        json.dump(profiles, f, ensure_ascii=False, indent=2)
    print(f"\n{path} 갱신 완료. 다음: python verify_keywords.py 로 DF 재검증")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Gemini로 프로필 번역+키워드 추출")
    ap.add_argument("--profiles", default="profiles.json")
    ap.add_argument("--dry-run", action="store_true", help="출력만, 파일 미수정")
    ap.add_argument("--only-unconfirmed", action="store_true",
                    help="keywords_confirmed=false 프로필만 처리")
    ap.add_argument("--overwrite-confirmed", action="store_true",
                    help="사람이 확정한 키워드도 Gemini 결과로 덮어쓰기")
    args = ap.parse_args()
    main(args.profiles, args.dry_run, args.only_unconfirmed, args.overwrite_confirmed)
