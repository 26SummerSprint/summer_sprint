"""
Streamlit UI — AI 논문 추천 데모 (HTML 카드 스타일).

전체 파이프라인: 키워드(BM25)+임베딩 하이브리드 검색 → CrossEncoder 재랭킹
→ Gemini 최종 선정(제외조건 반영)까지 한 번의 요청으로 처리.

실행:
    uvicorn recommend_backend.main:app --reload --port 8100   # 백엔드 먼저
    streamlit run recommend_backend/ui/app.py                  # UI

백엔드 주소는 환경변수 RECOMMEND_BACKEND_URL 또는 사이드바에서 지정.
"""

import html
import os

import requests
import streamlit as st

DEFAULT_BACKEND = os.environ.get("RECOMMEND_BACKEND_URL", "http://localhost:8100")

st.set_page_config(page_title="AI Paper Recommender", page_icon="📚", layout="wide")

# ── 스타일 ────────────────────────────────────────────────
st.markdown(
    """
    <style>
      .block-container { max-width: 960px; }
      .chip {
        display:inline-block; padding:3px 10px; margin:3px 4px 3px 0;
        border-radius:999px; font-size:0.82rem; background:#e8f0fe; color:#1a56db;
        border:1px solid #c7d7fb;
      }
      .chip-ex { background:#fdecec; color:#c0392b; border-color:#f5c6cb; }
      .card {
        border:1px solid #e5e7eb; border-radius:14px; padding:16px 18px;
        margin-bottom:14px; background:#ffffff;
        box-shadow:0 1px 3px rgba(0,0,0,0.04);
      }
      .card-head { display:flex; align-items:flex-start; gap:12px; }
      .rank {
        flex:0 0 auto; width:30px; height:30px; border-radius:50%;
        background:#1a56db; color:#fff; font-weight:700; font-size:0.9rem;
        display:flex; align-items:center; justify-content:center;
      }
      .p-title { font-size:1.06rem; font-weight:700; line-height:1.35; margin:0; }
      .p-title a { color:#111827; text-decoration:none; }
      .p-title a:hover { text-decoration:underline; }
      .p-meta { color:#6b7280; font-size:0.8rem; margin:4px 0 8px 0; }
      .p-reason { background:#f8fafc; border-left:3px solid #1a56db;
        padding:8px 12px; border-radius:6px; font-size:0.92rem; color:#374151; }
      .p-links a { font-size:0.82rem; margin-right:12px; color:#1a56db; text-decoration:none; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📚 AI Paper Recommender")
st.caption("키워드(BM25) + 임베딩 하이브리드 검색 → 재랭킹 → Gemini 최종 선정(제외조건 반영)")

# ── 입력 ──────────────────────────────────────────────────
with st.sidebar:
    st.header("설정")
    backend_url = st.text_input("백엔드 주소", value=DEFAULT_BACKEND)
    st.caption("`POST {주소}/api/v1/recommend` 를 호출합니다.")

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


def _esc(x) -> str:
    return html.escape(str(x or ""))


# ── 실행 ──────────────────────────────────────────────────
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
            data = resp.json()
        except requests.RequestException as e:
            st.error(f"요청 실패: {e}")
            st.stop()

    # 추출된 프로필
    ext = data.get("extracted_profile", {}) or {}
    kws = ext.get("keywords") or []
    exs = ext.get("exclude") or []
    if kws or exs:
        chips = "".join(f'<span class="chip">{_esc(k)}</span>' for k in kws)
        chips += "".join(f'<span class="chip chip-ex">✕ {_esc(x)}</span>' for x in exs)
        st.markdown("**추출된 키워드 / 제외 주제**", help="Gemini가 프로필에서 뽑은 검색 키워드와 제외조건")
        st.markdown(f"<div>{chips}</div>", unsafe_allow_html=True)
        st.write("")

    recs = data.get("recommendations") or []
    st.success(f"최종 추천 {data.get('count', len(recs))}편")

    for rec in recs:
        paper = rec.get("paper", {}) or {}
        rank = rec.get("rank", "")
        title = _esc(paper.get("title", "(제목 없음)"))
        url = paper.get("abs_url") or "#"
        pdf = paper.get("pdf_url")
        cat = _esc(paper.get("primary_category", ""))
        reason = _esc(rec.get("reason", ""))

        links = f'<a href="{_esc(url)}" target="_blank">abstract ↗</a>'
        if pdf:
            links += f'<a href="{_esc(pdf)}" target="_blank">PDF ↗</a>'

        card = f"""
        <div class="card">
          <div class="card-head">
            <div class="rank">{_esc(rank)}</div>
            <div style="flex:1 1 auto;">
              <p class="p-title"><a href="{_esc(url)}" target="_blank">{title}</a></p>
              <div class="p-meta">{cat}</div>
              {'<div class="p-reason">💡 ' + reason + '</div>' if reason else ''}
              <div class="p-links" style="margin-top:8px;">{links}</div>
            </div>
          </div>
        </div>
        """
        st.markdown(card, unsafe_allow_html=True)

        abstract = paper.get("abstract_clean") or ""
        if abstract:
            with st.expander("초록 보기"):
                st.write(abstract)

    if not recs:
        st.info("추천 결과가 없습니다. 프로필을 더 구체적으로 적어보세요.")
