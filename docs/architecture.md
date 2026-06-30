# 아키텍처

## 전체 구조

```
[세무사 클라이언트]
        | HTTPS / JWT
        v
[API 서버 (FastAPI async)]
        |
        +--> PostgreSQL + pgvector   벡터검색 + tsvector 키워드검색 + 메타데이터
        +--> Neo4j                   조문↔판례 인용 그래프 + 판례 유효성 + 개정이력
        |
        +--(쿼리 임베딩)--> [온프레미스 Ollama: qwen3-embedding:8b]
        +--(리랭킹)-------> [온프레미스 Ollama: bge-reranker-v2-m3]
        +--(LLM 추론)-----> [Ollama Cloud]
```

## 저장소 역할 분담

| 저장소 | 역할 |
|---|---|
| PostgreSQL + pgvector | 벡터(hnsw) + 키워드(tsvector/gin) 하이브리드 검색, 조문/판례 원문, validity_flag |
| Neo4j | 인용(CITES), 준용(REFERS_TO), 판례변경(OVERRULED_BY), 개정이력(AMENDED_BY) 그래프 탐색 |
| 온프레미스 Ollama | 임베딩(qwen3-embedding:8b) + 리랭킹(bge-reranker-v2-m3) |
| Gemini (Cloud) | RARR 초안(draft), 편집(edit), 3층 법리(reason) — 사용자 노출 산문 |
| Ollama Cloud (glm-5.2) | RARR 주장 분해, CQGen, agreement 판정 — 내부 기계적 판단 (다량 병렬) |

pgvector와 Neo4j를 둘 다 유지하는 이유: 조문번호/판례번호 정확매칭 + 의미 벡터검색(→ pgvector), 인용/준용/판례변경 관계 탐색(→ Neo4j). 역할이 겹치지 않는다.

## 판례 유효성 처리 3층

```
1층 (인덱싱 시점, 기계적):
    OVERRULED_BY 있으면            → validity_flag = overruled
    BASED_ON 조문이 판결 후 개정  → validity_flag = law_amended
    나머지                         → valid (또는 uncertain)

2층 (런타임, 시점 정합):
    질의에 거래시점 명시 시 해당 시점 유효 조문/판례로 필터

3층 (런타임, 복잡 모드 한정):
    "조문 개정됐으나 법리 유지 여부" → Ollama Cloud 판단
    1·2층 사실을 컨텍스트로 제공 후 위임
```

유효성 의심 판례는 버리지 않고 [주의] 경고와 함께 제시.

## RARR 파이프라인 (결정 M)

검색-후-생성(RAG)에서 **RARR**(Retrofit Attribution using Research and Revision)으로 전환.
Gemini가 코퍼스 제약 없이 자유 초안 생성 → 주장 분해 → 주장별 코퍼스 research → 근거 검증 → 최소 수정.

### 공통 흐름

```
1 초안 (Gemini/draft) — 검색 없이 자유 생성
2 주장 분해 (glm-5.2/aux) — 원자 주장 목록
   ↓ 주장별 병렬
3a 인용 존재검증 (tsvector exact) — 할루시네이션 조문/판례 번호 prune
3b CQGen + 근거검색 (glm-5.2 + 기존 검색 스택)
4  Agreement (glm-5.2/aux) — 주장↔근거 일치 여부 + 지지 근거 반환
5  Edit (Gemini/edit) — 불일치 주장만 최소 수정, 인용 교정
   ↓ 재조립
6  Attribution (결정론) — 주장→근거 매핑, Source/Warning 생성
7  3층 법리 (Gemini/reason, complex + 경고 있을 때만)
```

### 단순 모드 (RARR-lite)
- 3b: CQGen 생략, 주장 텍스트 직접 단일 검색 (`_retrieve_simple`)
- 7: 3층 법리 생략

### 복잡 모드 (full RARR, 상한 20초)
- 3b: CQGen + HyDE + 2홉 확장 + 충분성 루프 + 시점필터
- 7: 경고 있을 때 Gemini 3층 법리

### 안전망
파이프라인 타임아웃/에러 시 순수 초안 + `[미검증]` 배너로 degrade.
