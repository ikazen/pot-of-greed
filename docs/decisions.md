# 결정 기록

## A. 모드 라우팅

**결정**: A2(사용자 수동 토글) + 자동분류 확장 가능한 인터페이스

**왜**: 초기에 분류기 훈련 데이터가 없고, 세무사가 직접 깊이를 판단하는 게 더 신뢰성 있음. 추후 자동분류(A1/A3)로 교체 가능하도록 `classify()` 시그니처만 고정.

---

## B. 키워드 검색 엔진

**결정**: PG tsvector (B1)

**왜**: 조문번호·판례번호 정확매칭이 핵심이고 tsvector로 충분. 자연어 키워드 재현율 부족 시 형태소 분석기(은전한닢)/ES 추후 도입 검토.

---

## C. 임베딩 모델

**결정**: `qwen3-embedding:8b`, 1024차원

**왜**: 온프레미스에서 운용 중인 모델. 인덱싱과 쿼리에 반드시 동일 모델·버전 사용(벡터 공간 일치). config 단일 참조로 강제.

---

## D. 데이터 수집 / 그래프 구축

**결정**: 법제처 OPEN API(open.law.go.kr) 1차 수입. 국세법령정보 예규/심판례는 나중. 참조조문/참조판례 구조화 필드로 판례→조문 그래프를 LLM 추출 없이 수입.

**왜**: 법제처 DRF API가 조/항/호 + 연혁/시행일자/참조조문/참조판례를 구조화 XML로 제공 → LLM 추출 없이 필드 매핑만으로 그래프까지 구성 가능. 수입 우선 순위: 소득세법·법인세법·부가가치세법(핵심 3법) 먼저, 이후 확장.

---

## E. 충분성 루프 반복 횟수 N

**결정**: 기본 2, 상한 3. config `SUFFICIENCY_MAX_ITER`로 튜닝.

**왜**: N=2~3이 복잡 모드 20초 상한(H) 내에 맞음. 무한루프 방지 + 재검색 기회 보장.

---

## F. 근거 검증 게이트 강도

**결정**: F1(전체 답변 1회 교차확인). F2(주장 단위) 확장 인터페이스 확보.

**왜**: F2는 느리지만 정확. 환각 발생률 측정 후 강화 여부 결정. `check_claim()` seam을 미리 정의해 교체 가능하게.

---

## G. DB 구성

**결정**: 2-DB 유지 (pgvector + Neo4j)

**왜**: pgvector = 하이브리드 검색(벡터+키워드 RRF). Neo4j = 관계 탐색(인용/준용/판례변경/개정이력). 역할이 겹치지 않아 단일화 시 품질 손실.

---

## H. 복잡 모드 응답 지연 상한

**결정**: 20초. config `COMPLEX_MODE_TIMEOUT_S`.

**왜**: E(충분성 N=2~3), F1(1회 검증), HyDE를 모두 넣어도 20초 내 가능. 충분성 루프에서 시간 추적해 조기 탈출.

---

## J. 웹 UI 툴 선택

**결정**: Chainlit (Python/async). 배포 = 전부 ops-vm — API 컨테이너 + UI 컨테이너 둘 다 nexus network join.

**왜**: FastAPI와 동일 Python/async 런타임 → 런타임 추가 없음. sources/warnings 인용카드 네이티브 렌더 지원. Node 불필요. Open WebUI·LibreChat은 OpenAI 호환 어댑터 필요 + Node 무거움. Next.js는 제품화 단계용.

---

## K. Chainlit 대화 영속화

**결정**: 공식 `SQLAlchemyDataLayer` + 기존 `potofgreed` Postgres. Literal AI 클라우드 사용 안 함. storage provider 보류(텍스트 전용).

**왜**: 외부 SaaS 의존 없이 이미 운용 중인 DB에 붙임. `SQLAlchemyDataLayer`가 공식 지원 경로라 커스텀 `BaseDataLayer` 직접 구현(메서드 ~15개) 불필요 — 단순함 우선. 출처/경고 `cl.Text` 엘리먼트는 storage provider 없으면 Chainlit이 경고 후 스킵(런타임 에러 없음). MinIO S3 storage client 연결로 추후 확장 가능.

---

## I. Hierarchical 청킹 / small-to-big

**결정**: 계층 = 조문 한정. 검색 child = 항/호, 컨텍스트 parent = 조. 계층 표현은 pgvector `parent_chunk_id` fetch만 사용(그래프에 계층 엣지 없음). 판례는 계층 미적용.

**왜**: 항/호로 검색해야 임베딩 희석 없이 정밀 매칭되고, 조 전체를 parent로 끌어와야 리랭커·LLM이 세법 문맥을 오독 없이 본다(small-to-big). 계층을 그래프에 넣으면 Neo4j가 관계 탐색과 계층 탐색 두 역할을 지게 되므로, 단순 fetch로 충분한 계층은 PG 한 컬럼(`parent_chunk_id`)에 두고 그래프(G)는 인용/준용 관계에만 집중시킨다. 상세는 design.md §3.3.

---

## L. LLM Provider 추상화

**결정**: `app/llm/LLMProvider` Protocol로 채팅 LLM 호출 추상화. 기본 provider = Gemini(`gemini-2.5-flash`). 교체는 `LLM_PROVIDER` 환경변수 한 줄로 가능(`"gemini"` | `"ollama"`).

**왜**: 기존 코드는 동일한 Ollama Cloud httpx 패턴이 7개 호출 지점에 중복. provider 교체 시 7곳 수정이 필요했음. 추상화 후 `get_llm_provider()` 팩토리 하나로 수렴 — 신규 provider 추가도 단일 파일로 격리 가능. Gemini SDK(`google-genai`)를 선택한 이유: REST 스펙 변동·SSE 스트리밍 파싱의 견고함. 임베딩/리랭커는 범위 밖 — pgvector 벡터공간이 기수집 코퍼스와 묶여 있어 교체 시 전체 재인제스트 필요(결정 C와 연동).
