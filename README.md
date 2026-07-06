# 논문 추천 시스템 — 데이터/벤치마크 구축 프로세스

관심사 기반 논문 추천 시스템의 데이터 파이프라인(A)과 평가 벤치마크(B) 구축 과정을 정리한 문서입니다. 이 폴더는 GitHub로 팀원에게 공유되는 것을 전제로 합니다.

---

## ⚠️ GitHub에 공유하면 안 되는 파일

아래 항목은 실제 서버 주소·API 키 같은 비밀값 또는 자동 생성되는 대용량 데이터라 커밋하면 안 됩니다. `.gitignore`에 이미 등록해뒀지만, 혹시 강제로 `git add -f` 하지 않도록 주의하세요.

| 파일/폴더 | 이유 | 대신 커밋되는 것 |
|---|---|---|
| `arxiv_pipeline/local_config.py` | 실제 AWS 서버 퍼블릭 IP와 팀 공유 API 키가 평문으로 들어있음 | `local_config.example.py` (값이 비어있는 템플릿) |
| `arxiv_pipeline/data/` (`papers.db`, `chroma/`, `last_run.json`) | 파이프라인 실행 시 자동 생성되는 데이터. 버전관리 대상 아님(용량도 커짐) | 없음 (각자 `main.py --backfill`로 직접 생성) |
| `__pycache__/`, `*.pyc` | 파이썬 캐시, 커밋 의미 없음 | 없음 |

**팀원이 처음 클론했을 때 해야 할 일:**
```bash
cd arxiv_pipeline
cp local_config.example.py local_config.py   # Windows: copy local_config.example.py local_config.py
# local_config.py를 열어서 팀에서 공유받은 실제 API_BASE / API_KEY 입력
```

---

## 파일별 용도 · 역할 · 실행 방법

| 파일 | 공유 | 용도 | 전체 프로세스에서의 역할 | 실행 방법 |
|---|---|---|---|---|
| `config.py` | ✅ | 카테고리, 경로, 임베딩 모델명, API 호스트/포트 등 전역 설정값 | A-전체 단계에서 공통으로 참조하는 설정 허브 | 직접 실행 안 함 (다른 모듈이 import) |
| `collector.py` | ✅ | arXiv API로 논문 메타데이터(제목/저자/초록/카테고리/comments 등) 조회 | **A-1단계**: 논문 원본 수집 | `python collector.py` (동작 확인용 단독 실행 가능, 실제로는 `main.py`가 호출) |
| `preprocess.py` | ✅ | 중복 논문 제거(버전/카테고리 병합), 초록 LaTeX·줄바꿈 클린업 | **A-2단계**: 수집 데이터 정제 | 단독 실행 없음 (`main.py`가 호출) |
| `db.py` | ✅ | SQLite 스키마 정의, 논문 upsert/조회 함수 | **A-3단계**의 메타데이터 저장소 (Chroma와 별개로 관계형 데이터 관리) | 단독 실행 없음 (다른 모듈이 import) |
| `embedder.py` | ✅ | 초록을 벡터로 변환(sentence-transformers), Chroma에 저장, 유사도 검색 함수 제공 | **A-3단계**: 임베딩 계산·저장 + **B-2단계**: 벤치마크 후보 추출용 검색 | `python embedder.py` (밀린 임베딩만 계산하고 싶을 때 단독 실행 가능) |
| `main.py` | ✅ | 수집→전처리→임베딩까지 전체 파이프라인을 순서대로 실행하는 CLI | A 전체를 한 번에 묶어 실행하는 진입점 | `python main.py --backfill 3` (초기 구축) / `python main.py --daily` (매일 증분, cron 등록) |
| `app.py` | ✅ | FastAPI 서버. `/search`, `/papers/{id}` 등 REST API 제공, API 키 인증 처리 | 팀 공유 단계: AWS 서버에 떠서 팀원의 조회 요청을 받는 창구 | AWS 인스턴스에서 `uvicorn app:app --host 0.0.0.0 --port 8000` |
| `client_example.py` | ✅ (비밀값 제외) | FastAPI 서버에 질의하는 경량 클라이언트 예시 (requests만 사용) | 팀원이 벤치마크용 후보 논문을 검색/조회할 때 쓰는 창구 | `local_config.py` 준비 후 `python client_example.py` (또는 함수 import해서 사용) |
| `collect_and_push.py` | ✅ (비밀값 제외) | 팀원이 본인 노트북에서 arXiv를 직접 수집해 공유 서버 DB에 반영 (collector.py/preprocess.py 재사용, 임베딩은 서버가 계산) | **A-1단계**를 팀원 각자의 컴퓨터에서도 수행 가능하게 함 | `local_config.py` 준비 후 `python collect_and_push.py --days 7` |
| `requirements.txt` | ✅ | 서버(AWS)에서 필요한 전체 의존성 (arxiv, sentence-transformers, chromadb, fastapi, uvicorn) | A 파이프라인 + app.py 실행 환경 구성 | `pip install -r requirements.txt` |
| `requirements-client.txt` | ✅ | 팀원 로컬에서 조회+수집할 때 필요한 최소 의존성 (requests, arxiv — 둘 다 가벼움) | 팀원용 경량 실행 환경 구성 | `pip install -r requirements-client.txt` |
| `local_config.example.py` | ✅ | `local_config.py`의 템플릿 (값 비어있음) | 팀원이 본인 로컬에 실제 접속 정보를 세팅하기 위한 안내 | 복사해서 `local_config.py`로 저장 후 값 채우기 |
| `local_config.py` | ❌ (gitignore) | 실제 AWS 서버 IP + API 키 | client_example.py가 참조하는 실제 접속 정보 | 직접 실행 안 함 |
| `data/` | ❌ (gitignore) | `papers.db`(SQLite), `chroma/`(벡터 인덱스), `last_run.json` | A-3단계 산출물이 저장되는 위치 | `main.py` 실행 시 자동 생성 |

---

## 전체 구조

```
arXiv API 수집 → 전처리(중복/버전 정리, 텍스트 클린업) → 임베딩 계산 → 저장(SQLite + Chroma)
                                                                    ↓
                                                  관심사 프로필 ↔ 하이브리드 검색 → 벤치마크 라벨링
```

- **A. 논문 데이터베이스**: 시스템이 검색할 논문 풀. `arxiv_pipeline/` 코드로 자동화.
- **B. 평가 벤치마크**: 관심사 프로필 + 라벨링된 정답 논문 세트. 사람이 직접 작업(A의 결과물을 후보 추출에 활용).

---

## A. 논문 데이터베이스 구축

### 1단계: arXiv API 수집 (`collector.py`)

- `arxiv` 파이썬 라이브러리(무료)로 `cs.RO` / `cs.CV` / `cs.CL` 카테고리 논문 조회
- 수집 필드: arXiv ID, 제목, 저자, 초록, 카테고리, 제출일/갱신일, comments(학회 정보), abs/pdf 링크
- 두 가지 모드
  - **초기 구축(backfill)**: 최근 N개월치 일괄 수집 → 벤치마크용 논문 풀 겸용
  - **일일 증분(daily)**: 마지막 실행 시각 이후 신규 논문만 수집 (`data/last_run.json`에 기록)

### 2단계: 전처리 (`preprocess.py`)

- `dedupe_raw_papers()`: arXiv ID 기준 중복 제거. 여러 카테고리에 동시 등록된 논문은 카테고리를 합집합으로 병합
- `clean_abstract()`: LaTeX 수식(`$...$`)·명령어(`\textbf{}` 등)·줄바꿈 제거
- 버전 업데이트(v1→v2) 논문은 `db.py`의 upsert 로직에서 최신 버전만 유지(구버전 자동 대체)

### 3단계: 임베딩 계산 및 저장 (`embedder.py`)

- 모델: `BAAI/bge-small-en-v1.5` (sentence-transformers, 로컬/무료 실행, `config.py`에서 교체 가능)
- 논문당 1회만 계산 — DB의 `embedded` 플래그로 이미 벡터화된 논문은 재계산하지 않음
- 저장
  - 메타데이터: **SQLite** (`data/papers.db`) — 파일 기반, 설정 불필요
  - 벡터: **Chroma** (`data/chroma/`) — 로컬 파일 기반, 카테고리 필터 + 유사도 검색을 함께 지원(하이브리드 쿼리에 유리)

### 실행 방법

```bash
cd arxiv_pipeline
pip install -r requirements.txt

# 초기 구축: 최근 3개월치 수집 + 전처리 + 임베딩
python main.py --backfill 3

# 매일 실행 (cron 등록 권장): 신규 논문만 증분 수집
python main.py --daily
```

cron 예시 (매일 오전 7시 실행):

```
0 7 * * * cd /path/to/arxiv_pipeline && python main.py --daily >> daily.log 2>&1
```

### CPU vs GPU

이 규모(수천 편, 일일 증분 수십~백여 편)는 **CPU로 충분**합니다. `bge-small` 기준 노트북 CPU에서 초기 구축도 수 분 내 완료됩니다. GPU는 논문 수가 수만 편 이상으로 늘거나, 더 큰 임베딩 모델/파인튜닝 단계에 들어갈 때 고려하면 됩니다.

### Chroma DB를 팀원과 공유하기 (AWS + FastAPI)

Chroma 자체를 네트워크에 노출하는 대신, **FastAPI가 Chroma/SQLite 앞단에서 REST API를 제공**하고 팀원은 그 API만 호출하는 구조입니다. 장점: 팀원 쪽 컴퓨터에는 `chromadb`/`sentence-transformers`/`torch` 같은 무거운 패키지가 필요 없고, `requests`만 있으면 됨. API 키로 최소한의 접근 제어도 가능.

```
팀원 컴퓨터 (requests만 설치)
      │  HTTPS/HTTP + X-API-Key
      ▼
AWS EC2 인스턴스
  ├── app.py (FastAPI)         ← /search, /papers/{id} 제공
  ├── main.py (cron, --daily)  ← 매일 신규 논문 수집/임베딩
  ├── data/papers.db (SQLite)
  └── data/chroma/ (Chroma, PersistentClient)
```

**1) AWS 인스턴스에서 서버 준비 (SSH 접속 후)**

```bash
git clone <레포주소>
cd arxiv_pipeline
python3 -m venv venv
source venv/bin/activate

# torch는 CPU 전용 빌드를 먼저 설치 (GPU용 기본 설치는 CUDA 부속 패키지까지
#딸려와서 500MB+ 커지고, 작은 디스크의 EC2에서 "Disk quota exceeded" 남)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# API 키 설정 (팀끼리 공유할 임의의 문자열)
export ARXIV_API_KEY="팀에서_정한_비밀키"

# 최초 1회 초기 구축 (논문 수집 + 임베딩)
python main.py --backfill 3

# FastAPI 서버 실행
uvicorn app:app --host 0.0.0.0 --port 8000
```

SSH 세션을 끊어도 서버가 계속 떠 있게 하려면 `tmux`/`screen`으로 실행하거나, 아래처럼 백그라운드로 돌려두세요.

```bash
nohup uvicorn app:app --host 0.0.0.0 --port 8000 > server.log 2>&1 &
```

매일 수집은 EC2 인스턴스에 cron으로 등록해두면 됩니다.

```
0 7 * * * cd /home/ubuntu/arxiv_pipeline && ARXIV_API_KEY="..." python3 main.py --daily >> daily.log 2>&1
```

**2) EC2 보안 그룹(방화벽) 설정**

AWS 콘솔 → 인스턴스의 보안 그룹 → 인바운드 규칙에 TCP 8000번 포트를 팀원 IP(또는 편의상 0.0.0.0/0, 단 API 키로 방어)로 열어줍니다. 접속 주소는 인스턴스의 **퍼블릭 IP**를 사용해야 하며(프라이빗 IP는 VPC 내부에서만 통함), 재시작 시 IP가 바뀌지 않도록 Elastic IP 할당을 권장합니다.

**3) 팀원은 API만 호출 (조회 + 수집 둘 다 가능)**

```bash
pip install -r requirements-client.txt   # requests + arxiv 설치 (가벼움, torch 불필요)
cp local_config.example.py local_config.py   # 실제 값 채우기
```

조회(검색):

```python
# client_example.py 사용 예
from client_example import search

results = search("VLM 기반 로봇 매니퓰레이션...", top_k=20, category="cs.RO")
for r in results:
    print(r["arxiv_id"], r["title"])
```

수집(본인 노트북에서 arXiv 긁어와 공유 DB에 반영):

```bash
python collect_and_push.py --days 7
python collect_and_push.py --days 14 --categories cs.RO cs.CV
```

수집 자체(arXiv 조회, 중복 제거)는 팀원 컴퓨터에서 일어나고, 초록 클린업과 임베딩 계산은 서버가 맡아서 처리한 뒤 공유 DB에 바로 반영됩니다. 그래서 팀원 컴퓨터엔 `chromadb`/`sentence-transformers`/`torch` 설치가 필요 없어요.

또는 curl로도 확인 가능:

```bash
curl -X POST http://<EC2_퍼블릭_IP>:8000/search \
  -H "X-API-Key: 팀에서_정한_비밀키" \
  -H "Content-Type: application/json" \
  -d '{"query": "VLM 기반 로봇 매니퓰레이션", "top_k": 20, "category": "cs.RO"}'
```

**엔드포인트 요약**

| 메서드/경로 | 설명 |
|---|---|
| `GET /health` | 서버 상태 + 보유 논문 수 확인 (인증 불필요) |
| `POST /search` | 쿼리 텍스트로 유사 논문 top_k 검색 (서버가 임베딩까지 계산) |
| `POST /papers/ingest` | 팀원이 로컬에서 수집한 논문 배치를 공유 DB에 반영 (`collect_and_push.py`가 호출) |
| `GET /papers/{arxiv_id}` | 논문 1건 상세 조회 |
| `GET /papers?ids=id1,id2,...` | 여러 건 한 번에 조회 (라벨링 후보 리스트 화면 표시용) |

**보안/운영 참고**

- 지금 구조는 API 키 하나를 팀이 공유하는 최소 수준의 보호입니다. 학기 프로젝트 규모에는 충분하지만, HTTPS는 아니므로(평문 HTTP) 민감한 데이터를 다루게 되면 nginx+Let's Encrypt로 HTTPS를 추가하는 걸 권장합니다.
- `main.py --backfill`/`--daily`(EC2 자체 실행)와 팀원의 `collect_and_push.py`(로컬 실행) 둘 다 결국 같은 공유 DB에 upsert하는 구조라 동시에 여러 명이 돌려도 안전합니다(arxiv_id 기준 중복 자동 처리). 다만 서로 다른 사람이 동시에 대량 수집을 돌리면 SQLite 쓰기 대기 시간이 조금 늘어날 수 있어요.

---

## B. 평가 벤치마크 구축

코드 자동화 대상이 아닌 **팀 작업** 단계입니다. A의 결과(임베딩 검색)를 후보 추출에 활용합니다.

1. **관심사 프로필 작성 (5~10개)**: 팀원 3명이 각자 실제 관심사를 2~4문장으로 작성(1인 2~3개). 포함 조건뿐 아니라 제외 조건도 명시.
2. **라벨링 대상 논문 풀 확정**: A에서 수집한 논문 중 특정 기간(예: 2주치)을 고정. 필요하면 팀원 각자 `collect_and_push.py`로 원하는 기간/카테고리를 직접 보충 수집 가능. 프로필별로 키워드 검색 + `client_example.search()`(내부적으로 `embedder.search_similar()` 호출) 결과를 합쳐 후보 50~80편 구성.
3. **라벨링 기준 문서 작성**: "관련(1)/무관(0)" 판정 기준을 예시와 함께 문서화. 애매한 사례 5~10개를 셋이 같이 판정하며 기준 합의.
4. **라벨링 + 교차 검증**: 작성자 본인이 1차 라벨링 → 다른 팀원 1명이 2차 검증 → 불일치 항목만 셋이 논의해 확정. 불일치율은 라벨링 신뢰도 지표로 발표에 활용.
5. **학습/평가 분리**: 프로필 단위로 분리(예: 7개는 평가 전용, 3개는 파인튜닝용). 같은 데이터로 학습+평가하지 않도록 사전 분리.

작업량 감: 프로필 10개 × 60편 = 약 600건, 건당 1~2분 → 1인당 3~4시간.

---

## 일정

| 시점 | 목표 |
|---|---|
| ~7/7 | 기획서 제출, 회의 1회 인증, arXiv API·임베딩 파이프라인 검증, Gemini 무료 한도 테스트 |
| ~7/18 (1차 보고) | 논문 수집 자동화 완료, 임베딩 필터링 동작, 벤치마크 라벨링 기준 합의 + 프로필 3개 제작 |
| ~8/1 (2차 보고) | LLM 재랭킹+이유 생성 연동, 다이제스트 UI 베타, 벤치마크 완성 |
| ~8/15 (3차 보고) | 방법론 비교 실험 완료, (여유 시) 파인튜닝 착수 |
| ~8/26 (최종) | 통합 데모, 비교 실험 리포트, 발표자료 |
