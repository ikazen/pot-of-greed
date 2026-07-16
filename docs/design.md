# 세법 질의응답 RAG 백엔드 설계 초안

세무사용(소수 사용자) 세법/판례 질의응답 챗봇의 백엔드 서버 설계. Claude Code에서 구현을 시작하기 위한 기준 문서.

규모 전제: 동시 사용자 5인 이하 → 부하/인프라 부담은 낮음. **품질 최대화가 1순위 목표**이며, 사용자 수를 이유로 품질을 타협하지 않는다.

---

## 1. 목표와 비목표

목표
- 세법 조문과 판례에 근거한 정확한 답변
- 답변의 모든 주장에 출처(조문번호/판례번호) 부착
- 판례의 시점 유효성(개정/판례변경)을 투명하게 반영
- 단순 질의는 빠르게(2~4초), 복잡 질의는 깊게(10~20초 허용)

비목표 (이번 단계 제외)
- 프론트엔드 UX
- 대량 사용자 동시성/스케일아웃
- 데이터 수집/크롤링 자동화 파이프라인 (별도 작업, 본 설계는 "데이터가 적재된다" 전제)

---

## 2. 아키텍처 개요

```
[세무사 클라이언트]
        | HTTPS
        v
[API 서버 (FastAPI, async)]
        |
        +--> PostgreSQL + pgvector   : 벡터검색 + 키워드(tsvector) + 메타데이터
        +--> Neo4j                   : 인용/준용 그래프 + 판례 유효성 + 개정이력
        |
        +--(쿼리 임베딩)------------------> [온프레미스 Ollama: qwen3-embedding:8b]
        +--(리랭킹)----------------------> [온프레미스 Ollama: bge-reranker-v2-m3]
        +--(RARR draft/edit/reason)------> [Gemini Cloud: gemini-2.5-flash]
        +--(RARR aux: 분해/CQGen/agreement)--> [Ollama Cloud: glm-5.2]
```

데이터 저장소·모델 역할 분담

| 저장소/모델 | 역할 |
|---|---|
| PostgreSQL + pgvector | 벡터 의미검색, 키워드 정확매칭(tsvector), 조문/판례 원문, 메타데이터 |
| Neo4j | 조문↔판례 인용, 조문↔조문 준용, 판례변경(OVERRULED_BY), 조문 개정이력 그래프 탐색 |
| 온프레미스 Ollama | 임베딩(qwen3-embedding:8b) + 리랭킹(bge-reranker-v2-m3) |
| Gemini (Cloud) | RARR 초안(draft)·편집(edit)·3층 법리(reason) — 사용자 노출 산문 |
| Ollama Cloud (glm-5.2) | RARR 주장 분해·CQGen·agreement 판정 — 내부 기계적 판단 (결정 N) |

> pgvector와 Neo4j를 **둘 다** 유지하는 이유: 세법은 (a) 조문번호/판례번호 정확매칭과 (b) 개념 의미검색이 모두 필요하고(→ pgvector 하이브리드), (c) 인용/준용/판례변경 관계 탐색이 별도로 필요(→ Neo4j)하기 때문. 역할이 겹치지 않는다.

---

## 3. 데이터 모델

### 3.1 PostgreSQL (pgvector)

```sql
-- 조문 청크
CREATE TABLE article_chunks (
    chunk_id        TEXT PRIMARY KEY,        -- Neo4j 노드와 공유하는 식별자
    law_name        TEXT NOT NULL,           -- 소득세법 / 법인세법 / 부가가치세법 ...
    article_no      TEXT NOT NULL,           -- "제13조의2"
    clause_path     TEXT,                    -- "제1항 제3호"
    parent_chunk_id TEXT,                    -- small-to-big 검색용 상위 컨텍스트
    text            TEXT NOT NULL,
    effective_from  DATE NOT NULL,           -- 시행일 (시점 정합성 핵심)
    effective_to    DATE,                    -- NULL이면 현행
    is_current      BOOLEAN NOT NULL,
    embedding       VECTOR(1024),            -- bge-m3 기준 1024차원
    tsv             TSVECTOR                 -- 키워드 검색
);

-- 판례 청크
CREATE TABLE case_chunks (
    chunk_id        TEXT PRIMARY KEY,
    case_no         TEXT NOT NULL,           -- "2018두12345"
    court           TEXT,                    -- 대법원 / 고등법원 ...
    decided_at      DATE NOT NULL,
    is_en_banc      BOOLEAN,                 -- 전원합의체 여부
    validity_flag   TEXT,                    -- valid / overruled / law_amended / uncertain (인덱싱 시 계산)
    text            TEXT NOT NULL,
    embedding       VECTOR(1024),
    tsv             TSVECTOR
);

CREATE INDEX ON article_chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX ON case_chunks    USING hnsw (embedding vector_cosine_ops);
CREATE INDEX ON article_chunks USING gin (tsv);
CREATE INDEX ON case_chunks    USING gin (tsv);
```

> `validity_flag`는 런타임 추론이 아니라 **인덱싱 시점에 Neo4j 그래프를 보고 미리 계산해 박아둔다.** 단순 모드에서도 빠르게 유효성 경고를 붙이기 위함.

### 3.2 Neo4j 그래프

```cypher
// 노드
(:Article {chunk_id, law_name, article_no, effective_from, effective_to, is_current})
(:Case    {chunk_id, case_no, court, decided_at, is_en_banc})
(:Amendment {amendment_id, law_name, article_no, amended_at, summary})

// 관계
(:Case)-[:CITES]->(:Article)            // 판례가 조문 인용
(:Case)-[:CITES]->(:Case)               // 판례가 판례 인용
(:Article)-[:REFERS_TO]->(:Article)     // 조문 준용/참조
(:Case)-[:BASED_ON]->(:Article)         // 판결의 근거 조문
(:Case)-[:OVERRULED_BY]->(:Case)        // 판례 변경 (전합 등)
(:Article)-[:AMENDED_BY]->(:Amendment)  // 조문 개정 이력
```

`chunk_id`가 pgvector와 Neo4j를 잇는 공유 키. 벡터 검색 결과의 `chunk_id`를 Neo4j 탐색의 시작 노드로 그대로 사용.

> Neo4j 그래프는 **인용/준용 관계 전용**이다. 조→항 같은 계층(hierarchy)은 그래프에 넣지 않는다(§3.3). 그래프와 계층의 역할을 분리해 양쪽 모델을 단순하게 유지한다.

### 3.3 Hierarchical 청킹 / small-to-big

계층은 **조문 한정**이며, pgvector의 `parent_chunk_id` 한 컬럼으로만 표현한다(그래프 미사용).

```
계층:          법령 > 조(Article) > 항/호(Clause)
검색 child:    항/호 청크        (작게 → 정밀 매칭, 임베딩 희석 방지)
컨텍스트 parent: 조 전체 청크    (크게 → 리랭커·LLM에 문맥 공급)
```

적재 규약 (`article_chunks` 한 테이블에 두 레벨이 공존):

| | 조 청크 (parent) | 항/호 청크 (child) |
|---|---|---|
| `clause_path` | NULL | "제1항 제3호" |
| `parent_chunk_id` | NULL | 소속 조 청크의 `chunk_id` |
| 임베딩 대상 | 예 (조 단독 조회 대비) | 예 (주 검색 단위) |

검색 흐름(small-to-big):

```
1. 하이브리드 검색 + 리랭킹 → 항/호 child 청크 선별 (정밀)
2. 선별된 child의 parent_chunk_id로 조 청크를 fetch (1쿼리, 중복 제거)
3. LLM 컨텍스트 = child 본문 + parent 조문 본문
   (child가 이미 조 레벨이면 parent 없음 → 그대로 사용)
```

시점/유효성 입자: `effective_from/to`·`validity_flag`는 **조 단위**가 기준. 단, 항이 독자적 시행일/개정이력을 가지면 **항 우선**(child 값이 parent를 오버라이드).

판례(`case_chunks`)는 이 계층을 적용하지 않는다 — 판례는 조/항 같은 포함 구조가 없고, 판례 간 관계는 그래프(CITES/OVERRULED_BY)가 담당한다.

> 청킹 코드(조/항/호 경계 분할) 자체는 데이터 수집 작업(결정 D)에 속해 이번 범위 밖이다. 본 절은 **적재될 데이터가 따라야 할 계약(contract)**과 런타임 parent fetch를 규정한다.

---

## 4. 판례 유효성 처리 (정확성 핵심)

원칙: **유효성의 사실(fact)은 데이터에서, 유효성의 해석(judgment)은 에이전트에서.**
LLM에 "이 판례 아직 유효해?"를 통째로 물으면 환각 발생 → 구조화된 사실을 먼저 뽑아 제공.

3층 처리

```
1층 (기계적 차단, 인덱싱 시점):
    - OVERRULED_BY 엣지 있으면 -> validity_flag = overruled
    - BASED_ON 조문이 판결일 이후 AMENDED_BY 가지면 -> validity_flag = law_amended
    - 나머지 -> valid (또는 판단 보류 시 uncertain)

2층 (시점 정합성, 런타임):
    - 질의가 특정 거래시점을 명시하면 해당 시점 유효 조문/판례로 필터
    - 판결일 vs 근거조문 유효기간 자동 비교

3층 (법리 판단, 런타임 - 복잡 모드 한정):
    - "조문은 개정됐으나 법리는 유지되는가" 같은 해석은 LLM에 위임
    - 단, 1·2층이 제공한 사실(개정일/변경판례)을 컨텍스트로 받은 뒤 판단
```

답변 표기 방식: 유효성 의심 판례를 **버리지 않고 경고와 함께 제시**.

```
관련 판례: 2018두12345
[주의] 근거 조문(법인세법 제52조)이 2021년 개정됨. 현행법 적용 시 결론이 달라질 수 있음.
```

---

## 5. 검색 파이프라인

검색 스택은 최종 응답을 직접 생성하지 않는다 — RARR 3b 단계(research, §5.3)가 주장별 근거를 조달할 때 호출된다(`app/rarr/research.py`). 단순/복잡 모드가 검색 강도를 결정한다.

### 5.1 검색 스택

**단순 (RARR-lite research, `_retrieve_simple`)**
```
1. 쿼리 임베딩 (온프레미스 Ollama)
2. 하이브리드 검색: 벡터(top-30) + 키워드(top-30) --RRF--> 통합
3. 리랭킹 (top-30 -> top-5)
4. Neo4j 1홉 확장 (직접 인용 판례만)
5. small-to-big parent fetch
   ※ HyDE/CQGen/충분성 루프 생략
```

**복잡 (full research, `_search_complex`/`_retrieve_complex`)**
```
1. CQGen: 주장별 검증 질문 생성 (glm-5.2, RARR_QUESTIONS_PER_CLAIM cap)
2. 질문별 병렬: 직접 임베딩 + HyDE 임베딩 + 키워드 검색 --RRF--> 통합
3. 충분성 평가 루프 (부족 -> 쿼리 재작성 후 재검색, 최대 N회, SUFFICIENCY_MAX_ITER)
4. 리랭킹
5. Neo4j 2홉 확장 (인용 판례, 준용 조문, OVERRULED_BY, 개정이력)
6. 2층 시점필터 (거래시점 명시 시)
7. small-to-big parent fetch
```

공통 핵심(ROI 높아 두 모드 모두 유지): **하이브리드 검색 + RRF + 리랭킹**.

### 5.2 모드 라우팅

```
질의 -> 세무사 수동 토글 (simple | complex) — classify()는 패스스루 (결정 A/A2)
     -> 단순 모드 검색 top 점수가 임계값(fallback_score_threshold) 미달이면
        -> 자동으로 복잡 모드로 승격 (should_promote, app/router/mode_classifier.py)
```

자동 분류기(A1/A3)는 `classify()` 시그니처를 유지한 채 추후 교체 가능하나 현재 미구현(결정 A).

### 5.3 RARR 파이프라인 (결정 M)

RAG 폐기 후 현재 답변 경로:

```
1 DRAFT        Gemini 자유 초안 (검색 없음)
2 DECOMPOSE    초안 → 원자 주장 목록 (glm-5.2, decontextualized)
3a CITE CHECK  조문/판례번호 구조적 동등매칭 존재검증 (law_name+article_no/case_no 컬럼, 결정론, LLM 없음)
3b RESEARCH    주장별 RAG 검색 (simple=단일검색, complex=CQGen+HyDE+2hop)
4 AGREEMENT    (주장, 근거) 일치 판정 (glm-5.2) + cited_refs ⊆ supporting 불변식 위반 시 강등
5 EDIT         불일치/할루시네이션 주장 최소 수정 (Gemini)
   재검증      edit 후 텍스트에서 ref 재추출 → 3a 재실행, 미검증 ref는 [인용 삭제]로 치환
6 REASSEMBLE   수정 주장 재결합 → 최종 답변 (한도 초과 주장은 [미검증] 표식 후 원문 유지)
7 ATTRIBUTION  주장→근거 매핑, Source/Warning 조립 (결정론)
  (complex) 3층 법리 검토: legal_reasoning_layer (Gemini)
  (fallback) 파이프라인 실패 시 순수 Gemini 초안 + [미검증] 배너
```

튜닝 노브 (`app/config.py`): `RARR_MAX_CLAIMS` (주장 수 cap), `RARR_QUESTIONS_PER_CLAIM` (CQGen 질문 수 cap). 기본 0=무제한. 상세 수정 이력은 `docs/decisions.md` B/F/M 참조.

### 5.4 eval 메트릭 & 하니스

`app/rarr/metrics.py`: `compute_metrics(reports) -> RarrMetrics`

| 메트릭 | 정의 |
|---|---|
| attribution_score | evidence 부착된 주장 / 전체 주장 |
| preservation_score | 주장별 draft↔revised difflib 유사도 평균 |
| n_hallucinated | 코퍼스에 없는 인용이 있는 주장 수 |
| hallucination_correction_rate | 그중 edit이 정정한 비율 |

하니스 실행(인프라 가동 시):
```bash
python -m scripts.rarr_eval --mode both --report
# 결과: eval/results/<ts>.md
```

---

## 6. 권장 기술 스택

```
API 서버:   FastAPI (async) — Gemini/Ollama 양쪽 호출이 I/O 바운드
DB:         PostgreSQL 16+ / pgvector (hnsw), Neo4j 5.x
임베딩:     qwen3-embedding:8b (1024차원) — 온프레미스 Ollama
리랭커:     bge-reranker-v2-m3 — 온프레미스 Ollama
추론:       Gemini(gemini-2.5-flash, draft/edit/reason) + Ollama Cloud(glm-5.2, aux) — 역할 기반 분리(결정 N)
청킹:       조/항/호 경계 분할 + 부모 컨텍스트 (small-to-big)
오케스트레이션: 직접 구성 (LangChain/LlamaIndex 미사용 — 커스텀 하이브리드/검증 로직에 추상화가 방해)
인증:       JWT + 고정 계정
```

---

## 7. 컴포넌트 구조

```
app/
  main.py                  # FastAPI 엔트리
  config.py                # 설정 (모델명, 엔드포인트, 임계값)
  api/
    chat.py                # POST /chat, /chat/stream — 질의 진입점 + RARR 근거공급 헬퍼
    health.py               # GET /health — PG/Neo4j 상태
  auth/
    jwt.py                 # JWT 발급/검증, 고정 계정 인증
    routes.py               # POST /auth/token
  router/
    mode_classifier.py     # 단순/복잡 라우팅(수동 토글 패스스루) + fallback 승격
  retrieval/
    embedder.py            # 온프레미스 Ollama 임베딩 클라이언트
    vector_search.py       # pgvector 검색
    keyword_search.py      # tsvector 검색
    fusion.py              # RRF 융합
    reranker.py            # bge-reranker
    graph_expand.py        # Neo4j 1홉/2홉 확장 + 시점필터
    context_expand.py      # small-to-big parent fetch
    hyde.py                # 복잡 모드 가상답변 생성
  agent/
    decompose.py           # 하위질의 분해
    tool_router.py         # 하위질의 -> 도구 선택
    sufficiency.py         # 충분성 평가 루프
    grounding_check.py     # 근거 검증 게이트 (F2 seam, 결정 F — RARR agreement.py로 대체돼 현재 미호출)
  rarr/
    draft.py               # 1 초안 생성 (Gemini)
    claims.py               # 2 주장 분해 + ref 파싱
    citation.py             # 3a 인용 구조적 존재검증
    query_gen.py             # 3b CQGen
    research.py              # 3b 주장별 근거 조달
    agreement.py             # 4 agreement 판정
    edit.py                  # 5 최소 수정
    pipeline.py               # 전체 오케스트레이션 (run_rarr)
    metrics.py                # eval 메트릭
    types.py                   # Claim/Evidence/AttributionReport
  reasoning/
    llm_client.py           # RAG 시대 직접 추론 클라이언트 — RARR 전환 후 pipeline.py가 대체, 현재 미호출
    answer_builder.py       # Source/Warning 조립, 3층 법리(legal_reasoning_layer)
  llm/
    base.py                 # LLMProvider Protocol
    gemini.py                # Gemini provider
    ollama.py                # Ollama Cloud provider
    (get_llm_provider(role) 역할 기반 팩토리, 결정 L/N)
  ingest/
    law_api.py               # 법제처 OPEN API 클라이언트
    law_mapper.py             # 조문 XML → PG/Neo4j 매핑
    case_mapper.py             # 판례 → PG/Neo4j 매핑
    models.py                   # 인제스트 중간 데이터 모델
  db/
    pg.py
    neo4j.py
```

---

## 8. 결정이 필요한 부분

구현 들어가기 전에 정해야 진행이 갈리는 항목들.

### A. 모드 라우팅 방식
- (A1) 자동 분류만 / (A2) 사용자 토글만 / (A3) 자동 + "더 깊이" 수동 승격
- 추천: A3. 자동 판정하되 세무사가 직접 복잡 모드로 올릴 수 있게.

### B. 키워드 검색 엔진
- (B1) pgvector의 PostgreSQL `tsvector` / (B2) 별도 Elasticsearch/OpenSearch
- 한국어 형태소 분석 품질이 관건. tsvector는 한국어 형태소가 약함 → 조문번호 같은 정확매칭엔 충분하나 자연어 키워드엔 약할 수 있음.
- 추천: B1로 시작(조문/판례번호 정확매칭 중심), 자연어 키워드 재현율 부족 시 형태소 분석기(은전한닢 등) 또는 ES 도입 검토.

### C. 임베딩 모델 확정
- bge-m3(1024) 기준으로 작성했으나 온프레미스 GPU/CPU 스펙에 따라 더 크거나 작은 모델 선택 가능.
- 결정 필요: 온프레미스 하드웨어 스펙? 한국어 특화 모델 비교 평가 여부?
- 주의: 인덱싱과 쿼리에 **반드시 동일 모델/버전** 사용 (벡터 공간 일치).

### D. 판례→근거조문/판례변경 그래프 구축 방법
- 이 프로젝트 최대 비용 작업.
- (D1) 법제처/대법원 공개 메타데이터에서 직접 수입 / (D2) LLM으로 판례 본문에서 추출 / (D3) 혼합
- 추천: D3. 원천 메타데이터 우선 수입, 빈 부분만 인덱싱 시점 LLM 추출.
- 결정 필요: 확보 가능한 원천 데이터의 구조화 수준?

### E. 충분성 평가 루프 최대 반복 횟수(N)
- 품질↔지연 트레이드오프. N=2~3 권장(무한루프 방지 + 복잡모드 20초 목표 내).

### F. 근거 검증 게이트 강도
- (F1) 전체 답변 1회 검증 / (F2) 주장 단위 검증(느리지만 정확)
- 추천: F1로 시작, 환각 발생률 보고 F2로 강화.

### G. Neo4j 벡터인덱스 사용 여부
- Neo4j 5.x는 자체 벡터인덱스 지원 → 이론상 pgvector 없이 단일 DB 가능.
- 현재 설계는 하이브리드 검색(벡터+키워드 RRF)을 위해 pgvector 유지.
- 결정 필요: 하이브리드 검색을 포기하고 단순화할지 vs 현 설계(2-DB) 유지.
- 추천: 품질 우선이므로 현 설계 유지(pgvector가 키워드+벡터 융합에 유리).

### H. 응답 지연 상한 합의
- 복잡 모드 최대 허용 시간을 확정해야 N, HyDE, 근거검증 강도가 정해짐.
- 결정 필요: 15초? 20초? 30초?

---

## 9. 구현 권장 순서

```
1. DB 스키마 + 인덱스 구축 (PG, Neo4j) + 소량 샘플 데이터 적재
2. 임베딩 클라이언트 + 벡터검색 + 키워드검색 + RRF (단순 검색 동작 확인)
3. 리랭커 연결
4. Neo4j 그래프 확장 + validity_flag 인덱싱 로직
5. 단순 모드 end-to-end (검색 -> 추론 -> 출처)
6. 모드 라우터 + fallback 승격
7. 복잡 모드: 분해 -> 라우팅 -> 충분성 루프 -> 근거검증
8. 답변 빌더 (유효성 경고 병기) 마감
```

각 단계가 독립 검증 가능하도록 구성. 단순 모드(5단계)까지가 1차 동작 가능한 MVP.
