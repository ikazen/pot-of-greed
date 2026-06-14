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
| Ollama Cloud | 복잡 추론, 쿼리 분해, 충분성 평가, 근거 검증, HyDE |

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

## 파이프라인

### 단순 모드 (목표 2~4초)

```
쿼리 임베딩 → 하이브리드검색(벡터top30 + 키워드top30) → RRF
→ 리랭킹(top5) → Neo4j 1홉 확장 + validity_flag → 단일 추론 → 출처 부착
```

### 복잡 모드 (상한 20초)

```
쿼리 분해 → 하위질의별 도구 라우팅
→ HyDE + 하이브리드검색(넓게) + 강한 리랭킹
→ Neo4j 2홉 확장 (REFERS_TO, OVERRULED_BY, AMENDED_BY)
→ 충분성 평가 루프 (최대 N=2~3)
→ 강한 추론 → 근거 검증 게이트(F1) → 출처 + 유효성 경고 부착
```
