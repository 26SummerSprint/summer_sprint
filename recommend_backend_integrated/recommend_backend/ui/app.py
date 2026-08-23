"""
Streamlit UI — AI 논문 추천 데모 (HTML 카드 + 피드백 버튼).

전체 파이프라인: 키워드(BM25)+임베딩 하이브리드 검색 → CrossEncoder 재랭킹
→ Gemini 최종 선정(제외조건 반영). 각 추천에 👍/👎/저장 피드백을 남기면
백엔드가 로깅(골드셋 확장·재랭커 재학습 재료).

실행:
    uvicorn recommend_backend.main:app --reload --port 8100   # 백엔드
    streamlit run recommend_backend/ui/app.py                  # UI
"""

import html
import os

import requests
import streamlit as st

DEFAULT_BACKEND = os.environ.get("RECOMMEND_BACKEND_URL", "http://localhost:8100")

st.set_page_config(page_title="AI Paper Recommender", page_icon="📚", layout="wide")

st.markdown(
    """
    <style>
      .block-container { max-width: 960px; }
      .chip { display:inline-block; padding:3px 10px; margin:3px 4px 3px 0;
        border-radius:999px; font-size:0.82rem; background:#e8f0fe; color:#1a56db;
        border:1px solid #c7d7fb; }
      .chip-ex { background:#fdecec; color:#c0392b; border-color:#f5c6cb; }
      .card { border:1px solid #e5e7eb; border-radius:14px; padding:16px 18px;
        margin-bottom:6px; background:#ffffff; box-shadow:0 1px 3px rgba(0,0,0,0.04); }
      .card-head { display:flex; align-items:flex-start; gap:12px; }
      .rank { flex:0 0 auto; width:30px; height:30px; border-radius:50%;
        background:#1a56db; color:#fff; font-weight:700; font-size:0.9rem;
        display:flex; align-items:center; justify-content:center; }
      .p-title { font-size:1.06rem; font-weight:700; line-height:1.35; margin:0; }
      .p-title a { color:#111827; text-decoration:none; }
      .p-title a:hover { text-decoration:underline; }
      .p-meta { color:#6b7280; font-size:0.8rem; margin:4px 0 8px 0; }
      .p-reason { background:#f8fafc; border-left:3px solid #1a56db; padding:8px 12px;
        border-radius:6px; font-size:0.92rem; color:#374151; }
      .p-links a { font-size:0.82rem; margin-right:12px; color:#1a56db; text-decoration:none; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📚 AI Paper Recommender")
st.caption("키워드(BM25) + 임베딩 하이브리드 검색 → 재랭킹 → Gemini 최종 선정(제외조건 반영)")

with st.sidebar:
    st.header("설정")
    backend_url = st.text_input("백엔드 주소", value=DEFAULT_BACKEND)
    st.caption("`POST {주소}/api/v1/recommend` 를 호출합니다.")


def _esc(x) -> str:
    return html.escape(str(x or ""))


def _arxiv_id(paper: dict) -> str:
    aid = paper.get("arxiv_id")
    if aid:
        return str(aid)
    url = paper.get("abs_url") or ""
    return url.rstrip("/").split("/")[-1] if url else ""


def post_feedback(arxiv_id: str, title: str, vote: str):
    """피드백 1건을 백엔드에 기록. Streamlit 버튼 on_click 콜백."""
    try:
        requests.post(
            f"{backend_url.rstrip('/')}/api/v1/feedback",
            json={
                "profile": st.session_state.get("profile", ""),
                "category": st.session_state.get("category"),
                "arxiv_id": arxiv_id,
                "title": title,
                "vote": vote,
            },
            timeout=10,
        )
        st.toast({"up": "👍 좋아요 기록", "down": "👎 관심없음 기록", "save": "🔖 저장됨"}[vote])
    except requests.RequestException as e:
        st.toast(f"피드백 실패: {e}")


# ── 입력 ──────────────────────────────────────────────────
profile = st.text_area(
    "연구 관심사 프로필",
    placeholder=(
        "예) Retrieval-Augmented Generation과 citation grounding에 관심. "
        "특히 생성 답변의 사실성(hallucination) 검증. 순수 IR 랭킹 연구는 제외."
    ),
    height=130,
)
col_cat, col_btn = st.columns([3, 1])
with col_cat:
    category = st.text_input("arXiv 카테고리 (선택)", placeholder="cs.CL")
with col_btn:
    st.write("")
    st.write("")
    go = st.button("추천 받기", type="primary", use_container_width=True)

# ── 요청 ──────────────────────────────────────────────────
if go:
    if not profile.strip():
        st.warning("프로필을 입력해주세요.")
        st.stop()
    with st.spinner("검색 → 재랭킹 → Gemini 최종 선정 중… (최대 2분)"):
        try:
            resp = requests.post(
                f"{backend_url.rstrip('/')}/api/v1/recommend",
                json={"profile": profile, "category": category or None},
                timeout=180,
            )
            resp.raise_for_status()
            st.session_state["result"] = resp.json()
            st.session_state["profile"] = profile
            st.session_state["category"] = category or None
        except requests.RequestException as e:
            st.error(f"요청 실패: {e}")
            st.session_state.pop("result", None)

# ── 렌더 (session_state 기반 → 피드백 클릭해도 결과 유지) ──────
data = st.session_state.get("result")
if data:
    ext = data.get("extracted_profile", {}) or {}
    kws, exs = ext.get("keywords") or [], ext.get("exclude") or []
    if kws or exs:
        chips = "".join(f'<span class="chip">{_esc(k)}</span>' for k in kws)
        chips += "".join(f'<span class="chip chip-ex">✕ {_esc(x)}</span>' for x in exs)
        st.markdown("**추출된 키워드 / 제외 주제**")
        st.markdown(f"<div>{chips}</div>", unsafe_allow_html=True)
        st.write("")

    recs = data.get("recommendations") or []
    st.success(f"최종 추천 {data.get('count', len(recs))}편")

    for rec in recs:
        paper = rec.get("paper", {}) or {}
        rank = rec.get("rank", "")
        title = paper.get("title", "(제목 없음)")
        url = paper.get("abs_url") or "#"
        pdf = paper.get("pdf_url")
        cat = _esc(paper.get("primary_category", ""))
        reason = _esc(rec.get("reason", ""))
        aid = _arxiv_id(paper)

        links = f'<a href="{_esc(url)}" target="_blank">abstract ↗</a>'
        if pdf:
            links += f'<a href="{_esc(pdf)}" target="_blank">PDF ↗</a>'

        st.markdown(
            f"""
            <div class="card">
              <div class="card-head">
                <div class="rank">{_esc(rank)}</div>
                <div style="flex:1 1 auto;">
                  <p class="p-title"><a href="{_esc(url)}" target="_blank">{_esc(title)}</a></p>
                  <div class="p-meta">{cat}{(' · ' + _esc(aid)) if aid else ''}</div>
                  {'<div class="p-reason">💡 ' + reason + '</div>' if reason else ''}
                  <div class="p-links" style="margin-top:8px;">{links}</div>
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        b1, b2, b3, sp = st.columns([1, 1, 1, 7])
        b1.button("👍", key=f"up_{rank}_{aid}", help="좋아요",
                  on_click=post_feedback, args=(aid, title, "up"))
        b2.button("👎", key=f"down_{rank}_{aid}", help="관심 없음",
                  on_click=post_feedback, args=(aid, title, "down"))
        b3.button("🔖", key=f"save_{rank}_{aid}", help="저장",
                  on_click=post_feedback, args=(aid, title, "save"))

        abstract = paper.get("abstract_clean") or ""
        if abstract:
            with st.expander("초록 보기"):
                st.write(abstract)

    if not recs:
        st.info("추천 결과가 없습니다. 프로필을 더 구체적으로 적어보세요.")
