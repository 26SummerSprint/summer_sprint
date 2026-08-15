"""
Streamlit UI - Hybrid Retrieval 추천 데모.

실행:
    streamlit run recommend_backend/ui/app.py

기본적으로 http://localhost:8100 백엔드(recommend_backend.main:app)를 호출한다.
"""

import os

import requests
import streamlit as st

BACKEND_URL = os.environ.get("RECOMMEND_BACKEND_URL", "http://localhost:8100")

st.set_page_config(page_title="AI Paper Recommender", page_icon="📚", layout="wide")
st.title("📚 AI Paper Recommender")
st.caption("키워드(BM25) + 임베딩 검색 → 재정렬 → Gemini 최종 선정까지 한 번에.")

profile = st.text_area(
    "연구 관심사 프로필",
    placeholder=(
        "VLM 기반 로봇 매니퓰레이션과 언어 지시 이해에 관심. "
        "특히 모호한 지시 해석. 순수 RL 이론은 제외."
    ),
    height=120,
)
category = st.text_input("arXiv 카테고리 (선택)", placeholder="cs.RO")

if st.button("추천 받기", type="primary"):
    if not profile.strip():
        st.warning("프로필을 입력해주세요.")
    else:
        with st.spinner("키워드/임베딩 검색 → 재정렬 → Gemini 최종 선정 중..."):
            try:
                resp = requests.post(
                    f"{BACKEND_URL}/api/v1/recommend",
                    json={"profile": profile, "category": category or None},
                    timeout=120,
                )
                resp.raise_for_status()
                data = resp.json()
            except requests.RequestException as e:
                st.error(f"요청 실패: {e}")
                data = None

        if data:
            extracted = data["extracted_profile"]
            st.markdown(f"**추출된 키워드**: {', '.join(extracted['keywords']) or '(없음)'}")
            if extracted["exclude"]:
                st.markdown(f"**제외 주제**: {', '.join(extracted['exclude'])}")

            st.success(f"{data['count']}편 추천")
            for rec in data["recommendations"]:
                paper = rec["paper"]
                with st.container(border=True):
                    title = paper.get("title", "")
                    url = paper.get("abs_url") or "#"
                    st.markdown(f"**{rec['rank']}. [{title}]({url})**")
                    st.caption(paper.get("primary_category", ""))
                    if rec.get("reason"):
                        st.write(f"💡 {rec['reason']}")
                    abstract = paper.get("abstract_clean") or ""
                    if abstract:
                        st.write(abstract[:400] + ("..." if len(abstract) > 400 else ""))
