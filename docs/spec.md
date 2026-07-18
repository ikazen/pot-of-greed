# API 스펙

## 엔드포인트

### POST /auth/token

JWT 발급. 고정 계정(`AUTH_USERS`) 사용자명/비밀번호로 로그인.

**요청**: `application/x-www-form-urlencoded` (OAuth2 password grant) — `username`, `password`

**응답**
```json
{"access_token": "...", "token_type": "bearer"}
```

### GET /health

인프라 상태 점검.

**인증**: `Authorization: Bearer <JWT>`

**응답**
```json
{"pg": "ok", "neo4j": "ok"}
```

각 값은 `"ok"` 또는 `"error"`.

### GET /healthz

무인증 헬스체크. `docker compose`의 컨테이너 헬스체크가 이 엔드포인트를 호출한다
(`python3 -c "import urllib.request; ..."`, compose.yml).

**인증**: 없음

**응답**: `/health`와 동일한 바디 형식(`{"pg": "ok"|"error", "neo4j": "ok"|"error"}`).
pg/neo4j 둘 다 `"ok"`면 status 200, 하나라도 `"error"`면 503.

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

`run_rarr`의 `on_progress` 콜백이 단계 완료마다 status 프레임을 흘린다(가짜
스트리밍이 아니라 실제 진행상태) — 초기 프레임 외의 status 이벤트 수·문구는
claim 개수에 따라 달라진다.

```
data: {"status": "검토 중"}

data: {"status": "초안 작성 완료"}
data: {"status": "3개 주장 분해 완료"}
data: {"status": "검증 1/3"}
data: {"status": "검증 2/3"}
data: {"status": "검증 3/3"}

data: {"token": "법인세법 제"}
data: {"token": "52조에 따르면..."}
...

data: {"sources": [...], "warnings": [...]}

data: [DONE]
```

| 이벤트 | 설명 |
|---|---|
| `{"status": "검토 중"}` | 요청 접수 직후 즉시 전송되는 최초 상태 |
| `{"status": "초안 작성 완료"}` | draft 단계 완료 |
| `{"status": "N개 주장 분해 완료"}` | decompose 단계 완료, claim 개수 확정 |
| `{"status": "검증 n/총"}` | claim 하나 처리(research+agreement+edit) 완료마다, 완료 순서대로 |
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
| `hallucination` | 코퍼스에서 확인되지 않은 인용이 결정론적으로 제거됨 (`[인용 삭제]`) |
| `deferred` | `rarr_max_claims` 한도 초과로 검증되지 않은 주장 |
