"""
Streamlit UI — AI 논문 추천 데모 (HTML 카드 + 피드백 + 보관함 페이지).

전체 파이프라인: 키워드(BM25)+임베딩 하이브리드 검색 → CrossEncoder 재랭킹
→ Gemini 최종 선정(제외조건 반영). 각 추천에 👍/👎/🔖 를 남기면 백엔드가
로깅(👍/👎 = 학습 신호, 🔖 = 재열람용 보관함).

두 개 페이지(사이드바 메뉴로 전환):
- 📚 추천     : 프로필 입력 → 추천
- 🔖 보관함   : 저장한 논문 열람/순위변경/삭제

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

if "page" not in st.session_state:
    st.session_state["page"] = "recommend"


# ============================================================
# 헬퍼
# ============================================================

def _esc(x) -> str:
    return html.escape(str(x or ""))


def _arxiv_id(paper: dict) -> str:
    aid = paper.get("arxiv_id")
    if aid:
        return str(aid)
    url = paper.get("abs_url") or ""
    return url.rstrip("/").split("/")[-1] if url else ""


def _api(path: str) -> str:
    return f"{backend_url.rstrip('/')}/api/v1{path}"


def set_page(p: str):
    st.session_state["page"] = p


def post_feedback(arxiv_id: str, title: str, vote: str):
    """학습 신호(👍/👎)를 백엔드에 기록. Streamlit 버튼 on_click 콜백."""
    try:
        requests.post(
            _api("/feedback"),
            json={
                "profile": st.session_state.get("profile", ""),
                "keywords": st.session_state.get("keywords", []),
                "category": st.session_state.get("category"),
                "arxiv_id": arxiv_id,
                "title": title,
                "vote": vote,
            },
            timeout=10,
        )
        st.toast({"up": "👍 좋아요 기록", "down": "👎 관심없음 기록"}[vote])
    except requests.RequestException as e:
        st.toast(f"피드백 실패: {e}")


def save_paper(paper: dict, reason: str):
    """🔖 재열람용 보관 — 추천 맥락(프로필·키워드·이유·초록)까지 함께 저장."""
    try:
        requests.post(
            _api("/saved"),
            json={
                "arxiv_id": _arxiv_id(paper),
                "title": paper.get("title"),
                "link": paper.get("abs_url"),
                "pdf_url": paper.get("pdf_url"),
                "profile": st.session_state.get("profile", ""),
                "keywords": st.session_state.get("keywords", []),
                "reason": reason,
                "abstract": paper.get("abstract_clean"),
                "category": paper.get("primary_category"),
            },
            timeout=10,
        )
        st.toast("🔖 보관함에 저장됨")
    except requests.RequestException as e:
        st.toast(f"보관 실패: {e}")


def load_saved() -> list:
    """보관함 목록을 백엔드에서 가져온다(표시 순서)."""
    try:
        r = requests.get(_api("/saved"), timeout=10)
        r.raise_for_status()
        return r.json()
    except requests.RequestException:
        return []


def remove_saved(arxiv_id: str):
    """보관함에서 1건 삭제. on_click 콜백."""
    try:
        requests.delete(_api(f"/saved/{arxiv_id}"), timeout=10)
        st.toast("보관함에서 삭제")
    except requests.RequestException as e:
        st.toast(f"삭제 실패: {e}")


def move_saved(arxiv_id: str, direction: str):
    """보관 논문 순위를 위/아래로 이동. on_click 콜백."""
    try:
        requests.post(
            _api(f"/saved/{arxiv_id}/move"),
            params={"direction": direction},
            timeout=10,
        )
    except requests.RequestException as e:
        st.toast(f"이동 실패: {e}")


def rerecommend_from_paper(arxiv_id: str, title: str, abstract: str, category):
    """
    '이 논문으로 다시 추천받기'. 이 논문의 abstract만으로 백엔드가 새 키워드를
    뽑아 추천을 처음부터 다시 실행한다 (기존 프로필/키워드는 쓰지 않음).
    기존 프로필 텍스트 상자 값은 건드리지 않는다 - 결과 화면만 새로 채운다.
    """
    if not abstract:
        st.session_state["reref_error"] = "이 논문은 초록 정보가 없어 재추천할 수 없습니다."
        return
    try:
        resp = requests.post(
            _api("/recommend/from_paper"),
            json={
                "arxiv_id": arxiv_id,
                "title": title,
                "abstract": abstract,
                "category": category,
            },
            timeout=180,
        )
        resp.raise_for_status()
        data = resp.json()
        st.session_state["result"] = data
        # post_feedback/save_paper가 참조하는 session_state["profile"]/["keywords"]만
        # 이번 재추천에 실제로 쓰인 값으로 갱신한다. 화면의 "연구 관심사 프로필"
        # 입력창(render_recommend_page의 지역 변수 profile)은 별개라 영향받지 않는다.
        st.session_state["profile"] = data.get("profile", "")
        st.session_state["keywords"] = (data.get("extracted_profile") or {}).get("keywords", [])
        st.session_state["category"] = category
        st.session_state["reref_source"] = {"arxiv_id": arxiv_id, "title": title}
        st.session_state.pop("reref_error", None)
    except requests.RequestException as e:
        st.session_state["reref_error"] = f"재추천 요청 실패: {e}"


# ============================================================
# 사이드바 (공용) — 설정 + 페이지 메뉴
# ============================================================

with st.sidebar:
    st.header("설정")
    backend_url = st.text_input("백엔드 주소", value=DEFAULT_BACKEND)
    diversity = st.slider(
        "다양성 (MMR)", 0.0, 0.7, 0.0, 0.1,
        help="0 = 관련성만, 값이 클수록 서로 다른 주제를 섞어 추천(다양성↑)",
    )

    st.divider()
    st.markdown("### 메뉴")
    _page = st.session_state["page"]
    st.button(
        "📚 추천", use_container_width=True,
        type=("primary" if _page == "recommend" else "secondary"),
        on_click=set_page, args=("recommend",),
    )
    st.button(
        "🔖 보관함", use_container_width=True,
        type=("primary" if _page == "saved" else "secondary"),
        on_click=set_page, args=("saved",),
    )


# ============================================================
# 페이지: 추천
# ============================================================

def render_recommend_page():
    st.title("📚 AI Paper Recommender")
    st.caption("키워드(BM25) + 임베딩 하이브리드 검색 → 재랭킹 → Gemini 최종 선정(제외조건 반영)")

    profile = st.text_area(
        "연구 관심사 프로필",
        placeholder=(
            "예) Retrieval-Augmented Generation과 citation grounding에 관심. "
            "특히 생성 답변의 사실성(hallucination) 검증. 순수 IR 랭킹 연구는 제외."
        ),
        height=130,
    )
    go = st.button("추천 받기", type="primary")

    if go:
        if not profile.strip():
            st.warning("프로필을 입력해주세요.")
            st.stop()
        with st.spinner("검색 → 재랭킹 → Gemini 최종 선정 중… (최대 2분)"):
            try:
                resp = requests.post(
                    _api("/recommend"),
                    json={"profile": profile, "category": None, "diversity": diversity},
                    timeout=180,
                )
                resp.raise_for_status()
                data = resp.json()
                st.session_state["result"] = data
                st.session_state["profile"] = profile
                st.session_state["category"] = None
                st.session_state["keywords"] = (data.get("extracted_profile") or {}).get("keywords", [])
            except requests.RequestException as e:
                st.error(f"요청 실패: {e}")
                st.session_state.pop("result", None)

    data = st.session_state.get("result")

    if st.session_state.get("reref_error"):
        st.error(st.session_state["reref_error"])

    if not data:
        return

    reref_source = st.session_state.get("reref_source")
    if reref_source:
        st.info(f"🔁 **[{reref_source.get('title', '')}]** 논문 기반 재추천 결과입니다.")

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

        abstract = paper.get("abstract_clean") or ""

        b1, b2, b3, b4, sp = st.columns([1, 1, 1, 3, 4])
        b1.button("👍", key=f"up_{rank}_{aid}", help="좋아요",
                  on_click=post_feedback, args=(aid, title, "up"))
        b2.button("👎", key=f"down_{rank}_{aid}", help="관심 없음",
                  on_click=post_feedback, args=(aid, title, "down"))
        b3.button("🔖", key=f"save_{rank}_{aid}", help="보관함에 저장",
                  on_click=save_paper, args=(paper, rec.get("reason", "")))
        b4.button(
            "🔁 이 논문으로 다시 추천받기",
            key=f"reref_{rank}_{aid}",
            help="이 논문의 초록만으로 새 키워드를 뽑아 추천을 처음부터 다시 실행합니다"
            " (기존 프로필/키워드는 사용하지 않음).",
            disabled=not abstract,
            on_click=rerecommend_from_paper,
            args=(aid, title, abstract, st.session_state.get("category")),
        )

        if abstract:
            with st.expander("초록 보기"):
                st.write(abstract)

    if not recs:
        st.info("추천 결과가 없습니다. 프로필을 더 구체적으로 적어보세요.")


# ============================================================
# 페이지: 보관함
# ============================================================

def render_saved_page():
    saved = load_saved()
    st.title(f"🔖 보관함 ({len(saved)})")
    st.caption("추천 페이지의 🔖 버튼으로 저장한 논문. ↑/↓로 순위를 바꾸고 🗑로 삭제할 수 있습니다.")

    if not saved:
        st.info("보관한 논문이 없습니다. 📚 추천 페이지에서 🔖 버튼으로 저장하세요.")
        return

    last = len(saved) - 1
    for idx, item in enumerate(saved, 1):
        aid = item.get("arxiv_id", "")
        title = item.get("title") or "(제목 없음)"

        c_num, c_title, c_up, c_dn, c_del = st.columns([0.6, 7.5, 0.8, 0.8, 0.9])
        c_num.markdown(f"### {idx}")
        c_title.markdown(f"**{title}**")
        c_up.button("⬆", key=f"up_{aid}", help="순위 올리기", disabled=(idx == 1),
                    on_click=move_saved, args=(aid, "up"))
        c_dn.button("⬇", key=f"dn_{aid}", help="순위 내리기", disabled=(idx - 1 == last),
                    on_click=move_saved, args=(aid, "down"))
        c_del.button("🗑", key=f"del_{aid}", help="보관 삭제",
                     on_click=remove_saved, args=(aid,))

        with st.expander("상세 보기"):
            if aid:
                st.caption(f"`{aid}`  ·  {item.get('category', '')}")
            link, pdf = item.get("link"), item.get("pdf_url")
            link_md = []
            if link:
                link_md.append(f"[abstract ↗]({link})")
            if pdf:
                link_md.append(f"[PDF ↗]({pdf})")
            if link_md:
                st.markdown("  ".join(link_md))

            kws = item.get("keywords") or []
            if kws:
                st.markdown("**추출 키워드**: " + ", ".join(kws))

            prof = item.get("profile")
            if prof:
                st.markdown("**추천 시 프로필**")
                st.caption(prof)

            reason = item.get("reason")
            if reason:
                st.markdown("**추천 이유**")
                st.info(reason)

            abstract = item.get("abstract")
            if abstract and st.toggle("초록 보기", key=f"abs_{aid}"):
                st.caption(abstract)

        st.divider()


# ============================================================
# 라우팅
# ============================================================

if st.session_state["page"] == "saved":
    render_saved_page()
else:
    render_recommend_page()
