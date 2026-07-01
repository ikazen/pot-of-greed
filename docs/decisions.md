# 결정 기록

## A. 모드 라우팅

**결정**: A2(사용자 수동 토글) + 자동분류 확장 가능한 인터페이스

**왜**: 초기에 분류기 훈련 데이터가 없고, 세무사가 직접 깊이를 판단하는 게 더 신뢰성 있음. 추후 자동분류(A1/A3)로 교체 가능하도록 `classify()` 시그니처만 고정.

---

## B. 키워드 검색 엔진

**결정**: PG tsvector (B1)

**왜**: 조문번호·판례번호 정확매칭이 핵심이고 tsvector로 충분. 자연어 키워드 재현율 부족 시 형태소 분석기(은전한닢)/ES 추후 도입 검토.

**수정(2026-07-01, BON-215)**: "정확매칭이 충분하다"는 검색(retrieval)에는 유효하지만 인용 존재검증(RARR attribution)에는 부족했다 — tsvector는 토큰 AND 매칭이라 "법명은 오귀속, 번호만 실재"인 인용도 통과시키는 오탐이 있었다(adversarial review C1). `verify_citations`(`app/rarr/citation.py`)는 tsvector 대신 `law_name`+`article_no`/`case_no` 컬럼 동등성 질의로 전환. `keyword_search`(하이브리드 검색용)는 그대로 tsvector 유지 — 검색 재현율과 인용 존재검증은 다른 요구사항이라 결정을 분리한다.

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

**측정 예정**: `python -m scripts.rarr_eval --mode both --report` 실행 후 p50/p95 실측치로 재설정(BON-208).

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

---

## M. 답변 전략 — RAG → RARR 전면 전환

**결정**: retrieve-then-generate(RAG) 폐기. RARR(Researching and Revising what LMs say) 전면 채택, 전 모드(단순·복잡). Gemini가 검색 없이 자유 초안 생성 → 코퍼스를 사후 research·revise·attribution 백본으로만 사용.

**왜**: 검색이 생성을 선제약하던 RAG에서 Gemini의 자연스러운 법리 서술이 청크에 갇혔다. RARR은 코퍼스를 "생성 제약"이 아니라 "사실확인 백본"으로 전환 — 법률 도메인 최대 리스크인 할루시네이션 조문/판례번호를 권위 코퍼스로 잡아 교정하고, 인용 없는 주장에 출처를 사후 부착한다. string-match 출처 휴리스틱보다 실제 attribution이 강력.

- 결정 A의 `classify()` seam은 폐기 대신 **RARR 강도 노브**로 재해석: simple=RARR-lite(CQGen 생략·단일검색·인용주장만 검증), complex=full RARR(CQGen+HyDE+2hop+충분성+3층).
- 결정 F의 `check_claim()` seam이 **agreement model**로 실현 (`app/rarr/agreement.py`).
- 결정 H 20초 상한 유지. 단순모드 2~4초 목표는 순수 RARR 비용으로 **실측 후 재설정** — 하니스: `python -m scripts.rarr_eval --mode both --report` (BON-208).
- **안전망**: 파이프라인 타임아웃/에러 시 순수 Gemini 초안 + `[미검증]` 배너로 degrade(서비스 유지 우선, 결정 F 폴백 철학과 동일).
- 재인제스트/스키마/임베딩/UI 계약 변경 없음. 검색 스택(`_retrieve_*`/rerank/graph)은 research 근거공급으로 호출 지점만 이동. 출력 계약(`ChatResponse` sources/warnings)·Chainlit 인용카드(결정 J/K) 유지.
- 파이프라인 흐름·모듈맵 상세는 `docs/design.md` RARR 섹션 참조.

---

## N. RARR 단계별 역할 모델 라우팅

**결정**: `get_llm_provider(role)` 역할 기반 라우팅 도입.

| 역할 | 파이프라인 단계 | provider:model |
|---|---|---|
| `draft` | 1 초안 생성 | gemini : gemini-2.5-flash |
| `edit` | 5 최소 수정 | gemini : gemini-2.5-flash |
| `reason` | 7 3층 법리 검토 | gemini : gemini-2.5-flash |
| `aux` | 2 주장분해 · 3b CQGen · 4 agreement | ollama(cloud) : glm-5.2 |

**왜**: 사용자에게 보이는 산문(초안·edit·법리)은 만족도 검증된 Gemini, 내부 기계 판정(분해·질문생성·일치판정)은 빠르고 싼 glm-5.2로. aux는 주장·질문 단위 다수 병렬 호출이라 비용·지연이 가장 크게 누적 → 저가 모델로 분리해야 RARR 예산이 맞는다. 결정 L의 `make_llm_provider(provider=, model=)`가 이미 provider/model override를 지원하므로, role→(provider,model) 맵 + per-role 캐시(`get_llm_provider(role)`)만 추가하면 추상화를 깨지 않는다. glm-5.2 ollama 모델 태그는 Ollama Cloud 카탈로그에서 정확 명칭 확인 후 config에 고정. 역할별 모델은 config로 노출 — 실측 후 1·5·7을 상위 모델로 올릴 여지 유지.
