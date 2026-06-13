# API 스펙

## 엔드포인트

### POST /chat

질의응답 진입점.

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
    {"type": "article", "ref": "법인세법 제52조", "chunk_id": "..."},
    {"type": "case",    "ref": "2018두12345",     "chunk_id": "..."}
  ],
  "warnings": [
    {
      "chunk_id": "...",
      "ref": "2018두12345",
      "validity_flag": "law_amended",
      "message": "근거 조문(법인세법 제52조)이 2021년 개정됨. 현행법 적용 시 결론이 달라질 수 있음."
    }
  ]
}
```

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
