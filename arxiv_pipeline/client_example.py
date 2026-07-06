"""
팀원용 초경량 클라이언트.

chromadb/sentence-transformers/torch 등 무거운 라이브러리를 설치할 필요 없이,
`requests`만으로 AWS에 떠 있는 FastAPI 서버에 질의한다.

pip install requests

사용 전 준비 (최초 1회):
    1) local_config.example.py 를 복사해서 local_config.py 로 저장
    2) local_config.py 안에 팀에서 공유받은 실제 API_BASE / API_KEY 입력

주의: 실제 서버 주소·API 키는 이 파일에 직접 적지 마세요.
      local_config.py는 .gitignore에 등록되어 GitHub에 올라가지 않지만,
      이 파일(client_example.py)은 커밋되는 파일이라 비밀값을 넣으면 그대로 공개됩니다.
"""

import requests

try:
    from local_config import API_BASE, API_KEY
except ImportError:
    raise SystemExit(
        "local_config.py가 없습니다. local_config.example.py를 복사해서 "
        "local_config.py로 저장한 뒤 실제 값을 채워주세요."
    )

HEADERS = {"X-API-Key": API_KEY}


def search(query: str, top_k: int = 20, category: str = None):
    """관심사 프로필 텍스트로 유사 논문 검색."""
    payload = {"query": query, "top_k": top_k}
    if category:
        payload["category"] = category
    resp = requests.post(f"{API_BASE}/search", json=payload, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return resp.json()


def get_paper(arxiv_id: str):
    """단일 논문 상세 정보 조회."""
    resp = requests.get(f"{API_BASE}/papers/{arxiv_id}", headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return resp.json()


def get_papers(arxiv_ids: list):
    """여러 논문 상세 정보 한 번에 조회 (라벨링용 후보 리스트 뿌릴 때)."""
    resp = requests.get(
        f"{API_BASE}/papers",
        params={"ids": ",".join(arxiv_ids)},
        headers=HEADERS,
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


if __name__ == "__main__":
    # 사용 예: 관심사 프로필로 후보 논문 20편 검색
    profile = (
        "VLM 기반 로봇 매니퓰레이션과 언어 지시 이해에 관심. "
        "특히 모호한 지시 해석. 순수 RL 이론은 제외."
    )
    results = search(profile, top_k=20, category="cs.RO")
    for r in results:
        print(f"{r['arxiv_id']} ({r['score']:.3f}) - {r['title']}")
