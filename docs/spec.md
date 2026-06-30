# API 스펙

## 엔드포인트

### POST /chat

질의응답 진입점. 전체 결과를 한 번에 반환.

**인증**: `Authorization: Bearer <JWT>`

**요청**
```json
{
  "query": "법인세법 제52조 부당행위계산 부인 요건은?",
  "mode": "simple"
}
```

| 필드 | 타입 | 설명 |
|---|---|---|
| `query` | string | 질의 텍스트 |
| `mode` | `"simple"` \| `"complex"` | 단순(2~4초) / 복잡(~20초) 선택 |

**응답**
```json
{
  "answer": "...",
  "sources": [
    {"type": "article", "ref": "법인세법 제52조", "chunk_id": "...", "summary": "..."},
    {"type": "case",    "ref": "2018두12345",     "chunk_id": "...", "summary": "..."}
  ],
  "warnings": [
    {
      "chunk_id": "...",
      "ref": "2018두12345",
      "validity_flag": "law_amended",
      "message": "근거 조문(법인세법 제52조)이 2021년 개정됨. 현행법 적용 시 결론이 달라질 수 있음."
    }
  ],
  "elapsed_ms": 1823
}
```

### POST /chat/stream

스트리밍 버전. SSE(`text/event-stream`) 형식으로 순차 반환.

**인증**: `Authorization: Bearer <JWT>` (요청 형식 동일)

**이벤트 흐름**

```
data: {"status": "검토 중"}

data: {"token": "법인세법 제"}
data: {"token": "52조에 따르면..."}
...

data: {"sources": [...], "warnings": [...]}

data: [DONE]
```

| 이벤트 | 설명 |
|---|---|
| `{"status": "검토 중"}` | RARR 파이프라인 시작 알림 |
| `{"token": "..."}` | 최종 답변 20자 단위 토큰 |
| `{"sources": [...], "warnings": [...]}` | 파이프라인 완료 후 출처·경고 (tail 프레임) |
| `[DONE]` | 스트림 종료 |

## 데이터 모델

### article_chunks (PostgreSQL)

| 컬럼 | 타입 | 설명 |
|---|---|---|
| chunk_id | TEXT PK | Neo4j 공유 키 |
| law_name | TEXT | 소득세법 / 법인세법 / 부가가치세법 … |
| article_no | TEXT | 제13조의2 |
| clause_path | TEXT | 제1항 제3호 |
| parent_chunk_id | TEXT | small-to-big 상위 청크 |
| text | TEXT | 원문 |
| effective_from | DATE | 시행일 |
| effective_to | DATE | NULL = 현행 |
| is_current | BOOLEAN | |
| embedding | VECTOR(1024) | qwen3-embedding:8b |
| tsv | TSVECTOR | 키워드 검색 |

### case_chunks (PostgreSQL)

| 컬럼 | 타입 | 설명 |
|---|---|---|
| chunk_id | TEXT PK | |
| case_no | TEXT | 2018두12345 |
| court | TEXT | 대법원 / 고등법원 … |
| decided_at | DATE | |
| is_en_banc | BOOLEAN | 전원합의체 여부 |
| validity_flag | TEXT | valid / overruled / law_amended / uncertain |
| text | TEXT | |
| embedding | VECTOR(1024) | |
| tsv | TSVECTOR | |

### Neo4j 그래프

```
(:Article)-[:CITES]->(:Article)
(:Case)-[:CITES]->(:Article)
(:Case)-[:CITES]->(:Case)
(:Article)-[:REFERS_TO]->(:Article)
(:Case)-[:BASED_ON]->(:Article)
(:Case)-[:OVERRULED_BY]->(:Case)
(:Article)-[:AMENDED_BY]->(:Amendment)
```

`chunk_id`가 PostgreSQL ↔ Neo4j 공유 키.

## warnings validity_flag 값

| 값 | 의미 |
|---|---|
| `overruled` | 판례 폐기됨 |
| `law_amended` | 근거 조문 개정됨 |
| `uncertain` | 유효성 불확실 |
| `correction` | RARR edit이 인용 번호를 정정함 (`[정정: 원본 → 수정]`) |
