"""
local_config.py 템플릿.

사용법:
    1) 이 파일을 복사해서 local_config.py 로 저장
       cp local_config.example.py local_config.py   (Windows: copy)
    2) local_config.py 안의 값을 팀에서 공유받은 실제 값으로 교체

local_config.py는 .gitignore에 등록되어 있어 GitHub에 올라가지 않습니다.
이 example 파일(placeholder)만 커밋됩니다.
"""

API_BASE = "http://<AWS 인스턴스 퍼블릭 IP>:8000"
API_KEY = "<팀 공유 API 키>"
